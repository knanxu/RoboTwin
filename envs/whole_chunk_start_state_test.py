import numpy as np

from envs._base_task import Base_Task
import envs.robot.toppra_chunk_executor as tce


class FakeRobot:
    def get_left_arm_jointState(self):
        return [10.0] * 6 + [0.0]

    def get_right_arm_jointState(self):
        return [20.0] * 6 + [0.0]

    def get_left_arm_real_jointState(self):
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]

    def get_right_arm_real_jointState(self):
        return [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0]

    def get_left_gripper_val(self):
        return 0.0

    def get_right_gripper_val(self):
        return 0.0


def test_whole_chunk_retime_starts_from_measured_qpos(monkeypatch):
    captured = {}

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

    monkeypatch.setattr(tce, "retime_chunk", fake_retime_chunk)
    env = Base_Task()
    env.robot = FakeRobot()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    env.take_chunk_action(
        np.zeros((2, 14)), vel_limit=1.0, execution_steps=None
    )

    expected = np.array(
        [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], dtype=float
    )
    actual = np.asarray(captured["current_state_arm"], dtype=float)
    assert np.allclose(actual, expected), (actual, expected)
