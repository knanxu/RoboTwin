"""Websocket client 版 pi0/pi0.5 部署: 仿真端不加载模型、不依赖 jax.

配合 openpi 端的 scripts/serve_policy.py 使用:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<train_config> --policy.dir=<ckpt_dir> --port=8000
"""
import os
import sys
import time

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

    def __init__(self, host, port, pi0_step, vel_scale=1.0, acc_scale=1.0, v=1.0,
                 video_save_freq=25, infer_latency_ms=None):
        print(f"connecting to policy server at {host}:{port} ...")
        self.client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        self.pi0_step = int(pi0_step)
        self.vel_scale = float(vel_scale)
        self.acc_scale = float(acc_scale)
        self.v = float(v)
        self.video_save_freq = int(video_save_freq)
        # 推理延迟口径优先级 (用于视频里的推理停顿时长): 固定覆盖 > 服务端纯前向 > 墙钟回退.
        # infer_latency_s: None=不固定; 设数值(ms)则固定该值, 对齐目标真机板载推理时延.
        self.infer_latency_s = (float(infer_latency_ms) / 1000.0) if infer_latency_ms is not None else None
        # last_infer_ms: 每次 get_action 后存服务端回传的纯模型前向耗时 (ms, 排除 websocket
        # 网络/序列化); server 不回传 server_timing 时为 None, eval 回退到实测墙钟.
        self.last_infer_ms = None
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
        result = self.client.infer(obs)
        # 取服务端纯模型前向耗时 (排除网络); 老 server 不回传时记 None (eval 回退墙钟).
        timing = result.get("server_timing", {}) if isinstance(result, dict) else {}
        self.last_infer_ms = timing.get("infer_ms", None)
        return np.asarray(result["actions"])

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
    return PI0Client(
        usr_args["server_host"],
        usr_args["server_port"],
        usr_args["pi0_step"],
        vel_scale=usr_args.get("vel_scale", 1.0),
        acc_scale=usr_args.get("acc_scale", 1.0),
        v=usr_args.get("v", 1.0),
        video_save_freq=usr_args.get("video_save_freq", 25),
        infer_latency_ms=usr_args.get("infer_latency_ms", None),
    )


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)

    # ---- 推理 (阻塞), 实测墙钟仅作回退 ----
    _t_infer = time.perf_counter()
    actions = model.get_action(input_rgb_arr, input_state)[:model.pi0_step]
    wall_s = time.perf_counter() - _t_infer

    # 推理延迟计入视频时间轴: 同步部署下真机此间静止等待. 用 hold_and_render 空跑物理步
    # (机器人保持当前位姿不动), 期间正常写帧, 使视频回放速度 = 实机速度.
    # 口径优先级: 固定覆盖(--infer_latency_ms) > 服务端纯前向 infer_ms(排除网络) > 墙钟回退.
    if model.infer_latency_s is not None:
        idle_s = model.infer_latency_s
    elif model.last_infer_ms is not None:
        idle_s = float(model.last_infer_ms) / 1000.0
    else:
        idle_s = wall_s
    TASK_ENV.hold_and_render(idle_s)

    # 按 TASK_ENV.exec_backend 选择三种执行后端 (streaming/per_action/whole_chunk),
    # 由 eval 脚本设置, 缺省 whole_chunk = 整段 TOPPRA (原行为). 内部按物理步检查
    # check_success 并消耗 take_action_cnt 预算. B/C 的 vel/acc 缺省为 1.0.
    TASK_ENV.take_chunk_action_backend(
        actions,
        vel_scale=model.vel_scale,
        acc_scale=model.acc_scale,
        v=model.v,
        video_save_freq=model.video_save_freq,
    )


def reset_model(model):
    model.reset()
