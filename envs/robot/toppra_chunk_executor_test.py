import numpy as np
from envs.robot.toppra_chunk_executor import (
    retime_chunk, compute_segment_sd_bounds, PHYS_VEL_CEIL,
)

# --- 12 dof 假数据: 一条非退化两点路径 (绝对值制) ---
def _dummy_retime(vel_limit=2.0, acc_limit=2.0, sd_start=0.0, sd_end=0.0):
    cur = np.zeros(12)
    target = np.full(12, 0.3)                     # 各关节移动 0.3 rad
    return retime_chunk(
        current_state_arm=cur, chunk_arm=target[None, :],
        current_gripper=np.zeros(2), chunk_gripper=np.zeros((1, 2)),
        vel_limit=vel_limit, acc_limit=acc_limit, exec_hz=250,
        sd_start=sd_start, sd_end=sd_end,
    )

# ---- retime_chunk: 绝对值制 ----
def test_retime_chunk_absolute_limits_no_base():
    # 绝对值制: vel_limit 直接作关节速度上限 (不再 ×base)
    r = _dummy_retime(vel_limit=2.0, acc_limit=2.0)
    assert r["status"] == "success"
    assert np.max(np.abs(r["dense_arm_vel"])) <= 2.0 + 1e-3

def test_retime_chunk_clamped_by_phys_ceil():
    # vel_limit 超 PHYS_VEL_CEIL 时被钳到天花板
    r = _dummy_retime(vel_limit=999.0, acc_limit=999.0)
    assert r["status"] == "success"
    assert np.max(np.abs(r["dense_arm_vel"])) <= PHYS_VEL_CEIL + 1e-3

def test_default_boundary_zero_terminal_speed():
    # 默认 sd_end=0: 末端速度 ≈ 0 (whole_chunk 整段默认行为, 仍支持)
    r = _dummy_retime()
    assert r["status"] == "success"
    assert np.linalg.norm(r["dense_arm_vel"][-1]) < 1e-2

def test_nonzero_sd_end_gives_nonzero_terminal_speed():
    # sd_end=V: 弧长参数化下末端速度幅值 ≈ V (per_action 段间不归零靠它)
    V = 0.8
    r = _dummy_retime(sd_end=V)
    assert r["status"] == "success"
    assert abs(np.linalg.norm(r["dense_arm_vel"][-1]) - V) < 0.1

# ---- compute_segment_sd_bounds: 去 v_cruise / 去 is_last 归零 / safety 0.99 / sd_reachable 可控钳 ----
def test_sd_end_velocity_dominated():
    # 长段 + 大 acc → sd_reachable > sd_max，sd_end = safety*sd_max（vel_limit 主导、生效）
    q = np.zeros(12); qd = np.zeros(12)
    target = np.zeros(12); target[0] = 1.0    # seg_len=1, mt=1
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q, qd, target, vel_limit=2.0, acc_limit=8.0, safety=0.99,
    )
    assert tangent is not None and seg_len > 0
    # sd_max=2; sd_reachable=sqrt(2*8*1)=4 > sd_max → sd_end=0.99*2
    assert abs(sd_end - 0.99 * 2.0) < 1e-6
    assert sd_end > 0.0   # 不再有 is_last 归零、不再有 v_cruise 钳

def test_sd_end_reachable_dominated_short_seg():
    # 短段 → sd_reachable < sd_max，sd_end 被段内加速可达钳（保证两点 TOPPRA 可控）
    q = np.zeros(12); qd = np.zeros(12)
    target = np.zeros(12); target[0] = 0.02   # 极短段, mt=1
    _, sd_end, _, _ = compute_segment_sd_bounds(q, qd, target, vel_limit=5.0, acc_limit=8.0)
    # sd_max=5; sd_reachable=sqrt(2*8*0.02)=0.566 < sd_max → sd_end=0.99*0.566
    assert abs(sd_end - 0.99 * np.sqrt(2 * 8.0 * 0.02)) < 1e-6
    assert sd_end < 5.0

def test_sd_start_is_velocity_projection_clipped():
    # qd 沿 +tangent → sd_start = |qd 投影|; 反向 → 0
    target = np.full(12, 1.0)
    tang = target / np.linalg.norm(target)
    qd_forward = tang * 0.5
    s1, _, _, _ = compute_segment_sd_bounds(np.zeros(12), qd_forward, target, vel_limit=3.0, acc_limit=8.0)
    assert abs(s1 - 0.5) < 1e-6
    s2, _, _, _ = compute_segment_sd_bounds(np.zeros(12), -qd_forward, target, vel_limit=3.0, acc_limit=8.0)
    assert s2 == 0.0

def test_sd_end_clamped_by_phys_ceil():
    # vel_limit 超 PHYS_VEL_CEIL → sd_max 用钳后的 PHYS_VEL_CEIL；长段使 sd_reachable 不主导
    target = np.zeros(12); target[0] = 2.0   # seg_len=2, mt=1
    _, sd_end, _, _ = compute_segment_sd_bounds(np.zeros(12), np.zeros(12), target,
                                                vel_limit=999.0, acc_limit=999.0)
    # sd_max=min(999,5)/1=5; sd_reachable=sqrt(2*min(999,10)*2)=sqrt(40)=6.32 > 5 → sd_end=0.99*5
    assert abs(sd_end - 0.99 * PHYS_VEL_CEIL) < 1e-3

def test_degenerate_segment_flagged():
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q_current=np.full(12, 0.5), qd_actual=np.zeros(12), target=np.full(12, 0.5),
        vel_limit=2.0, acc_limit=8.0,
    )
    assert seg_len < 1e-6 and tangent is None
