import numpy as np

from envs._base_task import Base_Task


class FakeScene:
    def step(self):
        return None


class FakeRobot:
    def get_left_arm_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_right_arm_jointState(self):
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


def _two_frame_chunk(delta):
    chunk = np.zeros((2, 14), dtype=np.float64)
    chunk[1, 0] = float(delta)
    return chunk


def test_streaming_marks_planned_velocity_above_four():
    env = _make_env()
    info = env.take_chunk_action_streaming(
        _two_frame_chunk(0.3), v=1.0, hold_steps=15, max_actions=None
    )

    assert info["fixed_time_speed_violation"] is True
    assert np.isclose(info["max_planned_qvel"], 5.0)


def test_streaming_keeps_safe_planned_velocity_unflagged():
    env = _make_env()
    info = env.take_chunk_action_streaming(
        _two_frame_chunk(0.12), v=1.0, hold_steps=15, max_actions=None
    )

    assert info["fixed_time_speed_violation"] is False
    assert np.isclose(info["max_planned_qvel"], 2.0)
