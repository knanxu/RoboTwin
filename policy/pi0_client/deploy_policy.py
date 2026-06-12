"""Websocket client 版 pi0/pi0.5 部署: 仿真端不加载模型、不依赖 jax.

配合 openpi 端的 scripts/serve_policy.py 使用:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<train_config> --policy.dir=<ckpt_dir> --port=8000
"""
import os
import sys

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))


def _import_openpi_client():
    try:
        from openpi_client import websocket_client_policy  # noqa: F401
        return websocket_client_policy
    except ImportError:
        pass
    # openpi-client 是纯 python 包, 仓库内 vendored 路径直接可用
    for vendored in ("policy/pi05/packages/openpi-client/src",
                     "policy/pi0/packages/openpi-client/src"):
        path = os.path.join(_REPO_ROOT, vendored)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
    from openpi_client import websocket_client_policy
    return websocket_client_policy


_websocket_client_policy = _import_openpi_client()


class PI0Client:

    def __init__(self, host, port, pi0_step):
        print(f"connecting to policy server at {host}:{port} ...")
        self.client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        self.pi0_step = int(pi0_step)
        self.instruction = None

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction: {instruction}")

    def get_action(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        obs = {
            "state": np.asarray(state),
            "images": {
                "cam_high": np.transpose(img_front, (2, 0, 1)),
                "cam_left_wrist": np.transpose(img_left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(img_right, (2, 0, 1)),
            },
            "prompt": self.instruction,
        }
        return np.asarray(self.client.infer(obs)["actions"])

    def reset(self):
        self.instruction = None


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    return PI0Client(usr_args["server_host"], usr_args["server_port"], usr_args["pi0_step"])


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    actions = model.get_action(input_rgb_arr, input_state)[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)


def reset_model(model):
    model.reset()
