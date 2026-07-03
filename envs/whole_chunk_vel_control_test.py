import numpy as np

from envs._base_task import Base_Task
import envs.robot.toppra_chunk_executor as tce


class FakeScene:
    def step(self):
        return None


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

    def set_arm_joints(self, position, velocity, side):
        return None

    def set_gripper(self, position, side):
        return None


def _make_env():
    env = Base_Task()
    env.robot = FakeRobot()
    env.scene = FakeScene()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    env.render_freq = 0
    env.eval_video_path = None
    env._update_render = lambda: None
    env.check_success = lambda: False
    return env


def test_whole_chunk_truncates_before_toppra_and_derives_acceleration(monkeypatch):
    captured = {}

    def fake_retime_chunk(**kwargs):
        captured.update(kwargs)
        return {
            "status": "fallback",
            "fallback_reason": "capture only",
            "duration": 0.0,
            "dense_arm_pos": None,
            "dense_arm_vel": None,
            "dense_gripper": None,
            "return_code": None,
        }

    monkeypatch.setattr(tce, "retime_chunk", fake_retime_chunk)
    env = _make_env()
    info = env.take_chunk_action(
        np.zeros((5, 14), dtype=np.float64),
        vel_limit=2.5,
        execution_steps=3,
    )

    assert captured["chunk_arm"].shape == (3, 12)
    assert captured["vel_limit"] == 2.5
    assert captured["acc_limit"] == 25.0
    assert info["vel_limit"] == 2.5
    assert info["acc_limit"] == 25.0
    assert info["execution_steps"] == 3
    assert info["take_action_cnt_delta"] == 3


def test_whole_chunk_reports_planned_cruise_fraction(monkeypatch):
    dense_vel = np.zeros((4, 12), dtype=np.float64)
    dense_vel[:, 0] = [0.0, 1.9, 2.0, 0.0]

    def fake_retime_chunk(**kwargs):
        return {
            "status": "success",
            "fallback_reason": None,
            "duration": 4.0 / 250.0,
            "dense_arm_pos": np.zeros((4, 12), dtype=np.float64),
            "dense_arm_vel": dense_vel,
            "dense_gripper": np.zeros((4, 2), dtype=np.float64),
            "return_code": "Ok",
        }

    monkeypatch.setattr(tce, "retime_chunk", fake_retime_chunk)
    env = _make_env()
    info = env.take_chunk_action(
        np.zeros((4, 14), dtype=np.float64),
        vel_limit=2.0,
        execution_steps=None,
    )

    assert info["planned_cruise_fraction"] == 0.5
    assert info["dense_steps"] == 4
    assert info["execution_steps"] == 4
