import numpy as np

from envs._base_task import Base_Task
import envs.robot.toppra_chunk_executor as tce


class FakeRobot:
    def get_left_arm_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_right_arm_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_left_arm_real_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_right_arm_real_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_left_gripper_val(self):
        return 0.0

    def get_right_gripper_val(self):
        return 0.0


def _fake_retime_capture(captured):
    # 让 retime 在入口处"截停"(status!=success → take_chunk_action 走 fallback 立即返回),
    # 免去 stub 250Hz 下发循环; 仍能捕获传入的 chunk_arm (= _apply_k_skip 之后的帧).
    def fake_retime_chunk(**kwargs):
        captured.update(kwargs)
        return {
            "status": "fallback",
            "fallback_reason": "stop after input capture",
            "duration": 0.0,
            "dense_arm_pos": None,
            "dense_arm_vel": None,
            "dense_gripper": None,
            "return_code": None,
        }
    return fake_retime_chunk


def _make_env():
    env = Base_Task()
    env.robot = FakeRobot()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    return env


def test_whole_chunk_k_skip_truncates_chunk_before_toppra(monkeypatch):
    captured = {}
    monkeypatch.setattr(tce, "retime_chunk", _fake_retime_capture(captured))
    env = _make_env()

    # 5-frame chunk, execution_steps=2 → 只前 2 帧进入整段 TOPPRA
    info = env.take_chunk_action(
        np.zeros((5, 14)), vel_limit=1.0, execution_steps=2,
    )

    chunk_arm = np.asarray(captured["chunk_arm"], dtype=float)
    assert chunk_arm.shape[0] == 2, chunk_arm.shape
    assert info["take_action_cnt_delta"] == 2, info


def test_whole_chunk_none_execution_steps_keeps_full_chunk(monkeypatch):
    captured = {}
    monkeypatch.setattr(tce, "retime_chunk", _fake_retime_capture(captured))
    env = _make_env()

    # execution_steps=None → 整段保留（专家回放路径）
    info = env.take_chunk_action(
        np.zeros((5, 14)), vel_limit=1.0, execution_steps=None,
    )

    chunk_arm = np.asarray(captured["chunk_arm"], dtype=float)
    assert chunk_arm.shape[0] == 5, chunk_arm.shape
    assert info["take_action_cnt_delta"] == 5, info
