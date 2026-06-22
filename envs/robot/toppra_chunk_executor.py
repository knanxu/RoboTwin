"""
整段 TOPPRA 执行器.

对整段 action chunk (而非逐帧) 做一次 TOPPRA 时间重参数化, 得到连续时间
轨迹后按固定频率 (默认 250Hz, 对齐 Sapien 物理步) 密集采样, 供 take_chunk_action
逐点下发.

相比当前 RoboTwin 的"逐帧独立 TOPP"做法, 整段 TOPPRA:
  - 速度 / 加速度曲线全程连续, 段间不归零, 真正释放 vel/acc_scale 的加速潜力
  - TOPP 求解次数从 chunk_size 降到 1
  - 路径用 natural spline 插值, 更平滑

双臂 12 dof 一起做 TOPP (时间对齐), gripper 不参与 TOPP, 按路径参数位置插值.
"""

import numpy as np
import toppra
import toppra.constraint as cst
import toppra.algorithm as algo


# 物理天花板 (ARX5 / aloha-agilex). vel_scale/acc_scale 放大 base 约束后, 不允许
# 超过这两个值: acc=3.0 来自 curobo_left.yml:86 (权威), vel=3.0 为工程估计.
# 主防线是动作格点上界 (config.PHYS_*_CEIL), 这里是执行层 backstop.
PHYS_VEL_CEIL: float = 3.0   # rad/s
PHYS_ACC_CEIL: float = 3.0   # rad/s^2

# 方法 B (per_action 逐 action 非零边界) 的默认巡航路径速度 (= 12dof 联合关节速度幅值, rad/s).
# 保守初值, 云端按 "时间↓ / success 不降 / jerk 可接受" 实测调.
DEFAULT_CRUISE_SD: float = 1.0


def retime_chunk(
    current_state_arm: np.ndarray,
    chunk_arm: np.ndarray,
    current_gripper: np.ndarray,
    chunk_gripper: np.ndarray,
    joint_vel_limits: np.ndarray,
    joint_acc_limits: np.ndarray,
    vel_scale: float = 1.0,
    acc_scale: float = 1.0,
    exec_hz: int = 250,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    phys_acc_ceil: float = PHYS_ACC_CEIL,
    sd_start: float = 0.0,
    sd_end: float = 0.0,
):
    """
    对整段 chunk 做一次 TOPPRA, 返回密集采样的目标位置 / 速度 / gripper / 时长.

    Args:
        current_state_arm:  (dof_arm,)   当前双臂关节位置, dof_arm 通常 = 12
        chunk_arm:          (N, dof_arm) 目标路径航点 (不含当前点)
        current_gripper:    (dof_gripper,)  当前夹爪值, dof_gripper 通常 = 2
        chunk_gripper:      (N, dof_gripper) 目标夹爪序列
        joint_vel_limits:   (dof_arm,)   基础速度上限 (将被 vel_scale 放大)
        joint_acc_limits:   (dof_arm,)   基础加速度上限 (将被 acc_scale 放大)
        vel_scale, acc_scale: RL 控制的加速倍率, 1.0 = 基础约束
        exec_hz: 采样频率 (对齐 scene.set_timestep, RoboTwin 为 250)

    Returns:
        result: dict
            status:              'success' | 'fallback'
            fallback_reason:     None | str
            duration:            float, TOPP 重参数化后总时长 (s)
            dense_arm_pos:       (T, dof_arm)  目标位置序列
            dense_arm_vel:       (T, dof_arm)  目标速度序列
            dense_gripper:       (T, dof_gripper) 目标夹爪序列
            return_code:         toppra 求解返回码名字
    """
    fail = lambda reason: {
        "status": "fallback",
        "fallback_reason": reason,
        "duration": 0.0,
        "dense_arm_pos": None,
        "dense_arm_vel": None,
        "dense_gripper": None,
        "return_code": None,
    }

    chunk_arm = np.asarray(chunk_arm, dtype=np.float64)
    current_state_arm = np.asarray(current_state_arm, dtype=np.float64)
    chunk_gripper = np.asarray(chunk_gripper, dtype=np.float64)
    current_gripper = np.asarray(current_gripper, dtype=np.float64)

    if chunk_arm.ndim != 2 or chunk_arm.shape[0] == 0:
        return fail(f"invalid chunk_arm shape: {chunk_arm.shape}")
    if chunk_arm.shape[1] != current_state_arm.shape[0]:
        return fail(
            f"dof mismatch: chunk_arm={chunk_arm.shape[1]} vs current={current_state_arm.shape[0]}"
        )

    # 1. 拼成整段航点 (起点 + chunk)
    arm_waypoints = np.vstack((current_state_arm[None, :], chunk_arm))      # (N+1, dof)
    gripper_waypoints = np.vstack((current_gripper[None, :], chunk_gripper))  # (N+1, dof_g)

    # 2. 去除重复航点 (TOPPRA 不接受 segment_length=0)
    kept_idx = [0]
    for i in range(1, len(arm_waypoints)):
        if not np.allclose(arm_waypoints[i], arm_waypoints[kept_idx[-1]], atol=1e-8):
            kept_idx.append(i)
    # 注意: 不再强制追加末尾点 --- 若末尾与已保留的最后一点相同, 追加会制造重复.
    # 若整条路径都与起点相同, kept_idx 只剩 [0], 下面的长度检查会触发 fallback.
    if len(kept_idx) < 2:
        return fail("arm path degenerate (all waypoints identical)")

    kept_idx_arr = np.asarray(kept_idx, dtype=float)
    kept_arm = arm_waypoints[kept_idx]                                       # (K, dof)

    # 3. 累积路径长度作为 spline 参数
    segment_lengths = np.linalg.norm(np.diff(kept_arm, axis=0), axis=1)
    segment_lengths = np.maximum(segment_lengths, 1e-6)
    path_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))             # (K,)

    # 4. 为原始 (N+1) 个航点算出它们在 path_s 中的 s 位置 (gripper 对齐用)
    gripper_s = np.interp(
        np.arange(len(arm_waypoints), dtype=float),
        kept_idx_arr,
        path_s,
    )

    # 5. TOPPRA 求解 (应用 scale, 再钳到物理天花板, 保证不超物理限制)
    vel_lim = np.asarray(joint_vel_limits, dtype=np.float64) * float(vel_scale)
    acc_lim = np.asarray(joint_acc_limits, dtype=np.float64) * float(acc_scale)
    vel_lim = np.minimum(vel_lim, float(phys_vel_ceil))
    acc_lim = np.minimum(acc_lim, float(phys_acc_ceil))
    if vel_lim.shape[0] != kept_arm.shape[1] or acc_lim.shape[0] != kept_arm.shape[1]:
        return fail(
            f"limits dof mismatch: vel={vel_lim.shape} acc={acc_lim.shape} path_dof={kept_arm.shape[1]}"
        )

    try:
        geom = toppra.SplineInterpolator(path_s, kept_arm, bc_type="natural")
        pc_vel = cst.JointVelocityConstraint(vel_lim)
        pc_acc = cst.JointAccelerationConstraint(
            acc_lim, discretization_scheme=cst.DiscretizationType.Interpolation
        )
        gridpoints = _build_gridpoints(path_s, subdivisions_per_segment=4)
        instance = algo.TOPPRA(
            [pc_vel, pc_acc],
            geom,
            gridpoints=gridpoints,
            parametrizer="ParametrizeConstAccel",
        )
        retimed = instance.compute_trajectory(sd_start, sd_end)
    except Exception as exc:
        return fail(f"toppra solve exception: {exc}")

    return_code = instance.problem_data.return_code
    if retimed is None:
        return fail(f"toppra returned None ({return_code})")

    # 6. 按 exec_hz 密集采样
    duration = float(retimed.path_interval[1] - retimed.path_interval[0])
    if duration <= 0:
        return fail(f"non-positive duration: {duration}")

    dt = 1.0 / float(exec_hz)
    sample_times = np.arange(dt, max(duration, dt), dt, dtype=float)
    sample_times = np.append(sample_times, duration)
    sample_times = np.unique(
        np.clip(sample_times, retimed.path_interval[0], retimed.path_interval[1])
    )
    if sample_times.size == 0:
        sample_times = np.array([duration], dtype=float)

    try:
        dense_arm_pos = np.asarray(retimed(sample_times, 0))                 # (T, dof)
        dense_arm_vel = np.asarray(retimed(sample_times, 1))                 # (T, dof)
    except Exception as exc:
        return fail(f"trajectory sampling failed: {exc}")

    if dense_arm_pos.ndim != 2 or dense_arm_pos.shape[1] != kept_arm.shape[1]:
        return fail(f"unexpected sampled arm shape: {dense_arm_pos.shape}")

    # 7. gripper: 按采样时刻对应的"路径参数位置 s", 在 gripper_s ↔ gripper_waypoints 间线性插值
    try:
        sampled_s = _sample_path_position(retimed, sample_times)             # (T,)
    except Exception as exc:
        return fail(f"path position eval failed: {exc}")

    dense_gripper = np.column_stack(
        [
            np.interp(
                sampled_s,
                gripper_s,
                gripper_waypoints[:, d],
                left=gripper_waypoints[0, d],
                right=gripper_waypoints[-1, d],
            )
            for d in range(gripper_waypoints.shape[1])
        ]
    )

    return {
        "status": "success",
        "fallback_reason": None,
        "duration": duration,
        "dense_arm_pos": dense_arm_pos,
        "dense_arm_vel": dense_arm_vel,
        "dense_gripper": dense_gripper,
        "return_code": return_code.name if return_code is not None else None,
    }


def _build_gridpoints(path_s: np.ndarray, subdivisions_per_segment: int = 4) -> np.ndarray:
    """在相邻航点之间插入 subdivisions_per_segment 个等距点, 给 TOPPRA 更密的求解网格."""
    pts = [float(path_s[0])]
    for i in range(len(path_s) - 1):
        seg = np.linspace(
            float(path_s[i]), float(path_s[i + 1]),
            subdivisions_per_segment + 1, dtype=float,
        )
        pts.extend(seg[1:].tolist())
    return np.array(sorted(set(pts)), dtype=float)


def _sample_path_position(retimed, sample_times: np.ndarray) -> np.ndarray:
    """
    给定 TOPPRA 重参数化后的 jnt_traj 和采样时间, 求每个时刻对应的路径参数 s.

    优先用 retimed._eval_params (toppra 私有 API). 不可用时退化为"在 path_interval 上按时间
    线性映射到 s 的端点", 这是近似, 但对密集采样场景误差有限.
    """
    if hasattr(retimed, "_eval_params"):
        try:
            return np.asarray(retimed._eval_params(sample_times)[0])
        except Exception:
            pass

    # fallback: 把 sample_times 映射到 [path_s[0], path_s[-1]].
    # 对 ConstAccel parametrizer 只是近似, 但够用 (gripper 插值对 s 线性).
    t0, t1 = retimed.path_interval[0], retimed.path_interval[1]
    s_min, s_max = retimed.waypoints[0][0], retimed.waypoints[0][-1]
    if t1 - t0 <= 0:
        return np.full_like(sample_times, s_min, dtype=float)
    frac = (sample_times - t0) / (t1 - t0)
    return s_min + frac * (s_max - s_min)


def compute_segment_sd_bounds(
    q_current: np.ndarray,
    qd_actual: np.ndarray,
    target: np.ndarray,
    joint_vel_limits: np.ndarray,
    vel_scale: float,
    is_last: bool,
    v_cruise: float,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    safety: float = 0.9,
):
    """方法 B: 算逐 action 两点段在弧长参数化下的边界路径速度 (sd_start, sd_end).

    弧长参数化下 |dq/ds|=1, 故 sd = 关节空间速度幅值 (rad/s).
      - sd_start = 实测关节速度在本段单位切向上的投影 (闭环), clip >= 0.
      - sd_end   = 末段 0; 否则 min(v_cruise, safety * sd_max),
                   sd_max = min_j(scaled_vel_j / |tangent_j|) 由 per-joint vel 约束推.
    Returns: (sd_start, sd_end, tangent, seg_len). seg_len<1e-6 表退化 (tangent=None).
    """
    q_current = np.asarray(q_current, dtype=np.float64)
    qd_actual = np.asarray(qd_actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    seg = target - q_current
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        return 0.0, 0.0, None, seg_len
    tangent = seg / seg_len
    sd_start = max(0.0, float(qd_actual @ tangent))
    if is_last:
        sd_end = 0.0
    else:
        scaled_vel = np.minimum(
            np.asarray(joint_vel_limits, dtype=np.float64) * float(vel_scale),
            float(phys_vel_ceil),
        )
        sd_max = float(np.min(scaled_vel / np.maximum(np.abs(tangent), 1e-9)))
        sd_end = min(float(v_cruise), float(safety) * sd_max)
    return sd_start, sd_end, tangent, seg_len
