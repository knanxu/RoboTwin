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

    def get_left_arm_real_jointVelocity(self):
        return [0.0] * 6

    def get_right_arm_real_jointVelocity(self):
        return [0.0] * 6

    def get_left_gripper_val(self):
        return 0.0

    def get_right_gripper_val(self):
        return 0.0

    def set_arm_joints(self, position, velocity, arm_tag):
        return None

    def set_gripper(self, value, arm_tag):
        return None


class FakeScene:
    def step(self):
        return None


def test_per_action_retime_uses_zero_velocity_boundaries(monkeypatch):
    boundaries = []

    def fake_retime_chunk(**kwargs):
        boundaries.append((float(kwargs["sd_start"]), float(kwargs["sd_end"])))
        target = np.asarray(kwargs["chunk_arm"], dtype=float)[-1]
        gripper = np.asarray(kwargs["chunk_gripper"], dtype=float)[-1]
        return {
            "status": "success",
            "fallback_reason": None,
            "duration": 1.0 / 250.0,
            "dense_arm_pos": target[None, :],
            "dense_arm_vel": np.zeros((1, 12), dtype=float),
            "dense_gripper": gripper[None, :],
            "return_code": "Success",
        }

    monkeypatch.setattr(tce, "retime_chunk", fake_retime_chunk)
    env = Base_Task()
    env.robot = FakeRobot()
    env.scene = FakeScene()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    env.render_freq = 0
    env.eval_video_path = None
    env._last_success_obs_time = 0.0
    env._update_render = lambda: None
    env._tick_eval_video = lambda frequency: None
    env.check_success = lambda: False

    actions = np.zeros((2, 14), dtype=float)
    actions[0, :6] = 0.1
    actions[0, 7:13] = -0.1
    actions[1, :6] = 0.2
    actions[1, 7:13] = -0.2
    env.take_chunk_action_per_action(
        actions,
        vel_limit=1.0,
        acc_limit=1.0,
        v=1.0,
        max_actions=None,
        video_save_freq=-1,
    )

    assert len(boundaries) == 2
    assert boundaries == [(0.0, 0.0), (0.0, 0.0)], boundaries
