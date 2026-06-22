import numpy as np
import pytest
from envs.robot.toppra_chunk_executor import (
    retime_chunk, compute_segment_sd_bounds, DEFAULT_CRUISE_SD, PHYS_VEL_CEIL,
)

# --- 12 dof 假数据: 一条非退化两点路径 ---
def _dummy_retime(sd_start=0.0, sd_end=0.0):
    cur = np.zeros(12)
    target = np.full(12, 0.3)                     # 各关节移动 0.3 rad
    return retime_chunk(
        current_state_arm=cur, chunk_arm=target[None, :],
        current_gripper=np.zeros(2), chunk_gripper=np.zeros((1, 2)),
        joint_vel_limits=np.full(12, 2.0), joint_acc_limits=np.full(12, 2.0),
        vel_scale=1.0, acc_scale=1.0, exec_hz=250, sd_start=sd_start, sd_end=sd_end,
    )

def test_default_boundary_backward_compatible():
    # 默认 (0,0): 末端速度 ≈ 0 (whole_chunk 行为不变)
    r = _dummy_retime()
    assert r["status"] == "success"
    assert np.linalg.norm(r["dense_arm_vel"][-1]) < 1e-2

def test_nonzero_sd_end_gives_nonzero_terminal_speed():
    # sd_end=V: 弧长参数化下末端速度幅值 ≈ V
    V = 0.8
    r = _dummy_retime(sd_end=V)
    assert r["status"] == "success"
    assert abs(np.linalg.norm(r["dense_arm_vel"][-1]) - V) < 0.1

def test_sd_bounds_last_segment_is_zero():
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q_current=np.zeros(12), qd_actual=np.zeros(12), target=np.full(12, 0.2),
        joint_vel_limits=np.full(12, 2.0), vel_scale=1.0, is_last=True,
        v_cruise=DEFAULT_CRUISE_SD,
    )
    assert sd_end == 0.0
    assert seg_len > 1e-6 and tangent is not None

def test_sd_start_is_velocity_projection_clipped():
    # qd 沿 +tangent → sd_start = |qd 投影|; 反向 → 0
    target = np.full(12, 1.0)
    tang = target / np.linalg.norm(target)
    qd_forward = tang * 0.5
    s1, _, _, _ = compute_segment_sd_bounds(np.zeros(12), qd_forward, target,
                                            np.full(12, 3.0), 1.0, False, DEFAULT_CRUISE_SD)
    assert abs(s1 - 0.5) < 1e-6
    s2, _, _, _ = compute_segment_sd_bounds(np.zeros(12), -qd_forward, target,
                                            np.full(12, 3.0), 1.0, False, DEFAULT_CRUISE_SD)
    assert s2 == 0.0

def test_sd_end_clamped_by_vel_limit():
    # v_cruise 极大 → sd_end 被 0.9*sd_max 钳住 (sd_max 由 per-joint vel 约束推)
    target = np.full(12, 1.0)
    _, sd_end, _, _ = compute_segment_sd_bounds(np.zeros(12), np.zeros(12), target,
                                                joint_vel_limits=np.full(12, 2.0), vel_scale=1.0,
                                                is_last=False, v_cruise=1e6)
    # tangent 各分量 = 1/sqrt(12); scaled_vel=min(2.0,3.0)=2.0; sd_max=2.0/(1/sqrt(12))=2*sqrt(12)
    assert abs(sd_end - 0.9 * 2.0 * np.sqrt(12)) < 1e-3

def test_degenerate_segment_flagged():
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q_current=np.full(12, 0.5), qd_actual=np.zeros(12), target=np.full(12, 0.5),
        joint_vel_limits=np.full(12, 2.0), vel_scale=1.0, is_last=False, v_cruise=1.0,
    )
    assert seg_len < 1e-6 and tangent is None
