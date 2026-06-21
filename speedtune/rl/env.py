"""
Gym Env: 把 RoboTwin + 远端 pi0.5 包装成一个 SAC 可以训的环境.

  - 一个 step = 一个 chunk:
      1. 远端 (websocket) 调 pi0.5.infer_with_hidden 拿 chunk + cond_emb
      2. SAC actor 从 obs 出 (v, vel_scale, acc_scale)
      3. 调 take_chunk_action 执行
      4. reward = r_v + r_task (fallback 时 r_v 屏蔽, 给 -1)
      5. obs' = pool(下次推理 cond_emb) + last_action + cnt 进度 + last_fallback
  - 一个 episode = 一次 setup_demo, 跑到 success / truncated / 物理崩溃.
  - pi0.5 推理在云端 GPU 跑, 本机只跑 RoboTwin sim + SAC.

Episode-level 计时 (对齐 speedtune/tests/smoke_replay_expert.py 的口径):
  - ``_episode_wall_time``: 从 reset 后第一次 step 进入到当前的累计 wall time.
  - ``_episode_obs_time``:  RoboTwin 内部成功分支里 get_obs() (图像渲染) 的耗时,
                            需要从 wall_time 里剔除才是 "真实 episode 执行时间".
  - ``_episode_wall_time_no_obs = _episode_wall_time - _episode_obs_time``.
"""
from __future__ import annotations

import importlib
import os
import random
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from .config import (
    ActionSpaceConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
)


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_repo_paths():
    """对齐 script/eval_policy.py 的 sys.path 设置."""
    paths = [
        _REPO_ROOT,
        os.path.join(_REPO_ROOT, "policy"),
        os.path.join(_REPO_ROOT, "description/utils"),
        # openpi-client 是 pure-python, 直接加路径就能 import
        os.path.join(_REPO_ROOT, "policy/pi05/packages/openpi-client/src"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.append(p)


class ChunkSpeedupEnv:
    """SB3 风格 gym-like 接口, 但只暴露 reset / step / close, 不依赖 gym 注册."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        action_cfg: ActionSpaceConfig,
        reward_cfg: RewardConfig,
        policy_cfg: PolicyConfig,
        seed: int = 0,
        device: str = "cuda",
        verbose: bool = False,
    ):
        _ensure_repo_paths()
        self.env_cfg = env_cfg
        self.action_cfg = action_cfg
        self.reward_cfg = reward_cfg
        self.policy_cfg = policy_cfg
        self.device = device
        self.verbose = verbose
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        # reset 失败重试上限 (setup_demo 偶发失败时换 seed 重试; 无上限会无限递归)
        self._max_reset_retries = 20
        self._reset_failures = 0

        # 动作上下界 (对齐 actor 输出). action_mode 决定维度:
        #   "scalar_v"  -> [v]            (后端 A, 论文式)
        #   "v_vel_acc" -> [v, vel, acc]  (后端 B/C)
        self.action_mode = getattr(action_cfg, "action_mode", "v_vel_acc")
        if self.action_mode == "scalar_v":
            self.action_low = np.array([action_cfg.v_low], dtype=np.float32)
            self.action_high = np.array([action_cfg.v_high], dtype=np.float32)
        else:
            self.action_low = np.array(
                [action_cfg.v_low, action_cfg.vel_low, action_cfg.acc_low], dtype=np.float32
            )
            self.action_high = np.array(
                [action_cfg.v_high, action_cfg.vel_high, action_cfg.acc_high], dtype=np.float32
            )
        self.action_dim = int(self.action_low.shape[0])

        # 执行后端 & 论文式流式参数
        self.exec_backend = getattr(env_cfg, "exec_backend", "whole_chunk")
        self.stream_hold_steps = int(getattr(env_cfg, "stream_hold_steps", 15))
        # k_skip: A/B 每个 env.step 执行的动作数 (论文式 frame skip); C(整段 TOPPRA) 忽略.
        _ks = getattr(env_cfg, "k_skip", None)
        self.k_skip = int(_ks) if (_ks and int(_ks) > 0) else None

        # state = cond_emb + last_action(action_dim) + cnt(1) + last_fallback(1)
        self.state_dim = env_cfg.cond_emb_dim + self.action_dim + 2

        # 内部状态
        self._task_env = None
        self._task_args = None       # 缓存 build 出的 args, 给 setup_demo 用
        self._chunk_idx = 0
        self._last_action = np.ones(self.action_dim, dtype=np.float32)
        self._last_fallback = 0.0
        self._latest_cond_emb = None   # 上次 pi0.5 推理 pool 后 (cond_emb_dim,)
        self._latest_suffix = None     # 上次 pi0.5 的 action-expert suffix_out (action_horizon, expert_width); 备用, 暂不进 state
        self._latest_chunk = None      # 上次推理出的 chunk (T, action_dim)
        self._episode_total_time = 0.0   # 累计 chunk TOPPRA duration (秒, 仿真物理时间)
        self._episode_step_count = 0
        # wall-clock 计时 (对齐 smoke_replay_expert 口径)
        self._episode_wall_t0 = None         # reset 后第一次 step 进入的 perf_counter
        self._episode_obs_time = 0.0         # 累计 success 分支里 get_obs() 的耗时
        self._episode_wall_time = 0.0        # 当前累计 wall time
        self._episode_wall_time_no_obs = 0.0 # wall_time - obs_time
        # 拆分: pi0.5 推理 vs chunk 执行 (deploy time = 这两者之和)
        self._episode_pi_infer_time = 0.0    # 累计 _infer_pi 耗时 (含 reset 首发)
        self._episode_chunk_exec_time = 0.0  # 累计 take_chunk_action 耗时
        # 真实机器人执行时间 (按 dense_steps / 250Hz 累加, 不受仿真器快慢影响)
        self._episode_real_exec_time = 0.0
        self._sim_hz = 250.0

        # pi0.5 远端 client (lazy 创建, 因为云端 server 可能晚启动)
        from openpi_client import websocket_client_policy as _ws  # noqa: E402

        self._WebsocketClient = _ws.WebsocketClientPolicy
        self._policy_client = None
        self._instruction = None

        # 任务类与 args
        self._build_env_args()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _build_env_args(self):
        """对齐 speedtune/tests/smoke_replay_expert.py:_build_env_args."""
        from envs import CONFIGS_PATH  # noqa: E402

        with open(f"{_REPO_ROOT}/task_config/{self.env_cfg.task_config}.yml", "r") as f:
            args = yaml.safe_load(f)
        args["task_name"] = self.env_cfg.task_name
        args["task_config"] = self.env_cfg.task_config

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
            emb_types = yaml.safe_load(f)
        et = args.get("embodiment")
        if len(et) != 1:
            raise NotImplementedError("ChunkSpeedupEnv only supports single-embodiment tasks for now")
        args["left_robot_file"] = emb_types[et[0]]["file_path"]
        args["right_robot_file"] = emb_types[et[0]]["file_path"]
        args["dual_arm_embodied"] = True

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
            cam_cfg = yaml.safe_load(f)
        head = args["camera"]["head_camera_type"]
        args["head_camera_h"] = cam_cfg[head]["h"]
        args["head_camera_w"] = cam_cfg[head]["w"]

        def _get_emb_cfg(file_path: str) -> dict:
            with open(os.path.join(file_path, "config.yml"), "r") as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_emb_cfg(args["left_robot_file"])
        args["right_embodiment_config"] = _get_emb_cfg(args["right_robot_file"])
        args["play_once_path_file_list"] = []
        args["eval_mode"] = True
        args["collect_data"] = False
        args["eval_video_log"] = False
        self._task_args = args

    def _make_task_env(self):
        mod = importlib.import_module(f"envs.{self.env_cfg.task_name}")
        cls = getattr(mod, self.env_cfg.task_name)
        return cls()

    def _ensure_policy(self):
        if self._policy_client is None:
            print(
                f"[env] connecting to pi0.5 server at "
                f"{self.policy_cfg.server_host}:{self.policy_cfg.server_port} ...",
                flush=True,
            )
            self._policy_client = self._WebsocketClient(
                host=self.policy_cfg.server_host,
                port=self.policy_cfg.server_port,
                api_key=self.policy_cfg.api_key,
            )
            print("[env] pi0.5 server connected", flush=True)

    # ------------------------------------------------------------------
    # pi0.5 inference (远端 websocket)
    # ------------------------------------------------------------------
    def _build_pi_obs(self) -> Dict[str, Any]:
        """对齐 policy/pi05/deploy_policy.py:encode_obs.

        注意: msgpack-numpy 会自动处理 ndarray, 不需要手动序列化.
        """
        observation = self._task_env.get_obs()
        img_front = observation["observation"]["head_camera"]["rgb"]
        img_right = observation["observation"]["right_camera"]["rgb"]
        img_left = observation["observation"]["left_camera"]["rgb"]
        state = observation["joint_action"]["vector"]
        return {
            "state": state,
            "images": {
                "cam_high": np.transpose(img_front, (2, 0, 1)),
                "cam_left_wrist": np.transpose(img_left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(img_right, (2, 0, 1)),
            },
            "prompt": self._instruction,
        }

    def _infer_pi(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (chunk[T, action_dim], cond_emb[D], suffix_out[T, W]). 累计 pi0.5 推理墙钟到 episode 计时器.

        suffix_out = action expert 逐 token 的 hidden ([action_horizon, expert_width]), 与 cond_emb 一同由
        openpi server 的 infer_with_hidden 返回 (policy.py:154-155). 当前仅缓存到 self._latest_suffix 备用,
        暂不进 _assemble_state (喂网络是后续工作).
        """
        obs = self._build_pi_obs()
        _t0 = time.perf_counter()
        out = self._policy_client.infer_with_hidden(obs)
        self._episode_pi_infer_time += time.perf_counter() - _t0
        actions = np.asarray(out["actions"])[: self.policy_cfg.pi0_step]
        cond = np.asarray(out["cond_emb"], dtype=np.float32)
        if cond.shape[0] != self.env_cfg.cond_emb_dim:
            raise RuntimeError(
                f"cond_emb dim mismatch: got {cond.shape[0]} vs "
                f"EnvConfig.cond_emb_dim={self.env_cfg.cond_emb_dim}"
            )
        suffix = np.asarray(out["suffix_out"], dtype=np.float32)
        return actions, cond, suffix

    # ------------------------------------------------------------------
    # state assembly
    # ------------------------------------------------------------------
    def _assemble_state(self, cond_emb: np.ndarray) -> np.ndarray:
        cnt_norm = float(self._task_env.take_action_cnt) / max(1, int(self._task_env.step_lim))
        s = np.concatenate(
            [
                cond_emb.astype(np.float32),
                self._last_action.astype(np.float32),
                np.array([cnt_norm], dtype=np.float32),
                np.array([self._last_fallback], dtype=np.float32),
            ],
            axis=0,
        )
        if s.shape[0] != self.state_dim:
            raise RuntimeError(
                f"state_dim mismatch: assembled {s.shape[0]} vs config {self.state_dim}; "
                "check EnvConfig.cond_emb_dim"
            )
        return s

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def _reward_speed(self, action: np.ndarray) -> float:
        """knob 奖励的速度项. A (scalar_v) 只有 v; B/C 加 vel/acc 两项."""
        rc = self.reward_cfg
        v = float(action[0])
        r = rc.alpha_v * (v ** rc.beta_v)
        if action.shape[0] >= 3:
            vs, a_s = float(action[1]), float(action[2])
            r += rc.alpha_vs * (vs ** rc.beta_vs) + rc.alpha_as * (a_s ** rc.beta_as)
        return r

    def _compute_reward(self, action: np.ndarray, chunk_info: Dict[str, Any],
                        is_success: bool) -> Tuple[float, float]:
        """按 reward_mode 算 (reward, r_v). r_v 仅用于日志.

        time 模式: r = -alpha_time·(dense_steps/sim_hz) + 1{success}; fallback 按
          fallback_seconds 计时 (防 agent 故意触发 fallback 逃避时间惩罚). 防 hack.
        knob 模式: 论文式 alpha·v^beta(+vel/acc) + r_task; fallback/truncated 特判.
          ⚠️ 仅后端 A 安全; B/C 用此为消融 (会 reward-hack, 见 config.RewardConfig).
        """
        rc = self.reward_cfg
        status = chunk_info.get("status", "success")
        if rc.reward_mode == "time":
            if status == "topp_fallback":
                t = float(rc.fallback_seconds)
            else:
                t = float(chunk_info.get("dense_steps", 0)) / self._sim_hz
            r_time = -float(rc.alpha_time) * t
            return r_time + (rc.success_bonus if is_success else 0.0), r_time
        # knob mode
        if status == "topp_fallback":
            return float(rc.fallback_penalty), 0.0
        if status == "truncated":
            # 入口截断: chunk 没执行 (dense_steps=0), 给速度奖励是错误归因.
            return (rc.success_bonus if is_success else 0.0), 0.0
        r_v = self._reward_speed(action)
        return r_v + (rc.success_bonus if is_success else 0.0), r_v

    def _crash_reward(self) -> float:
        """环境崩溃的 terminal 奖励 (按 reward_mode)."""
        rc = self.reward_cfg
        if rc.reward_mode == "time":
            return -float(rc.alpha_time) * float(rc.crash_seconds)
        return float(rc.crash_penalty)

    # ------------------------------------------------------------------
    # gym-like API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._seed = int(seed)
            self._rng = random.Random(self._seed)
        self._ensure_policy()

        if self._task_env is not None:
            try:
                self._task_env.close_env()
            except Exception:
                pass
            self._task_env = None

        next_seed = self._rng.randint(0, 10**8)
        self._task_env = self._make_task_env()
        try:
            self._task_env.setup_demo(now_ep_num=0, seed=next_seed, is_test=True, **self._task_args)
        except Exception:
            try:
                self._task_env.close_env()
            except Exception:
                pass
            self._reset_failures += 1
            if self._reset_failures > self._max_reset_retries:
                raise RuntimeError(
                    f"setup_demo failed {self._reset_failures} consecutive times "
                    f"(task={self.env_cfg.task_name}); aborting instead of retrying forever."
                )
            return self.reset(seed=None)
        self._reset_failures = 0

        instruction = self._task_env.get_instruction()
        if instruction is None:
            instruction = self.env_cfg.task_name.replace("_", " ")
            self._task_env.set_instruction(instruction=instruction)
        self._instruction = instruction

        # 推第一段 chunk + cond_emb
        self._episode_pi_infer_time = 0.0
        self._episode_chunk_exec_time = 0.0
        self._episode_real_exec_time = 0.0
        chunk, cond, suffix = self._infer_pi()
        self._latest_chunk = chunk
        self._latest_cond_emb = cond
        self._latest_suffix = suffix
        self._chunk_idx = 0
        self._last_action = np.ones(self.action_dim, dtype=np.float32)
        self._last_fallback = 0.0
        self._episode_total_time = 0.0
        self._episode_step_count = 0
        # wall-clock 计时: 重置
        self._episode_wall_t0 = None
        self._episode_obs_time = 0.0
        self._episode_wall_time = 0.0
        self._episode_wall_time_no_obs = 0.0
        # 重置 RoboTwin 内部的 success obs 计时器
        self._task_env._last_success_obs_time = 0.0

        return self._assemble_state(cond), {"seed": next_seed, "instruction": instruction}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action.astype(np.float32), self.action_low, self.action_high)
        v = float(action[0])
        if self.action_dim >= 3:
            vel_scale, acc_scale = float(action[1]), float(action[2])
        else:
            vel_scale, acc_scale = 1.0, 1.0   # scalar_v (后端 A) 不用 vel/acc
        chunk = self._latest_chunk

        info: Dict[str, Any] = {"action": action.tolist()}

        # 开始计时 (第一次 step 时启动)
        if self._episode_wall_t0 is None:
            self._episode_wall_t0 = time.perf_counter()

        try:
            _t_exec = time.perf_counter()
            if self.exec_backend == "streaming":
                chunk_info = self._task_env.take_chunk_action_streaming(
                    chunk, v=v, hold_steps=self.stream_hold_steps,
                    max_actions=self.k_skip,
                )
            elif self.exec_backend == "per_action":
                chunk_info = self._task_env.take_chunk_action_per_action(
                    chunk, vel_scale=vel_scale, acc_scale=acc_scale, v=v,
                    max_actions=self.k_skip,
                )
            else:  # "whole_chunk" (整段 TOPPRA, k_skip 不适用, 整段执行)
                chunk_info = self._task_env.take_chunk_action(
                    chunk, vel_scale=vel_scale, acc_scale=acc_scale, v=v
                )
            self._episode_chunk_exec_time += time.perf_counter() - _t_exec
        except Exception as e:
            info["error"] = repr(e)
            self._finalize_wall_time()
            return (
                self._assemble_state(self._latest_cond_emb),
                self._crash_reward(),
                True,
                False,
                info,
            )

        info["chunk_info"] = chunk_info
        self._episode_step_count += 1
        self._episode_total_time += float(chunk_info.get("duration", 0.0))
        # 真实硬件执行时间: dense_steps 是这一 chunk 实际下发的 250Hz 物理步数
        self._episode_real_exec_time += float(chunk_info.get("dense_steps", 0)) / self._sim_hz
        # 累计本 chunk 内 success 分支的 get_obs 耗时 (从 take_chunk_action info 取)
        self._episode_obs_time += float(chunk_info.get("success_obs_time", 0.0))

        status = chunk_info["status"]
        is_fallback = status == "topp_fallback"
        is_truncated = status == "truncated"
        is_success = bool(self._task_env.eval_success)

        # ---- reward (reward_mode: knob | time, 见 _compute_reward) ----
        reward, r_v = self._compute_reward(action, chunk_info, is_success)
        info["r_v"] = r_v
        info["r_total"] = reward
        info["success"] = is_success

        # ---- terminal ----
        terminated = is_success
        truncated = is_truncated or (self._episode_step_count >= self.env_cfg.max_chunks_per_episode)

        self._last_action = action.copy()
        self._last_fallback = 1.0 if is_fallback else 0.0

        if terminated or truncated:
            self._finalize_wall_time()
            info["episode_wall_time"] = self._episode_wall_time
            info["episode_obs_time"] = self._episode_obs_time
            info["episode_wall_time_no_obs"] = self._episode_wall_time_no_obs
            info["episode_total_topp_time"] = self._episode_total_time
            info["episode_pi_infer_time"] = self._episode_pi_infer_time
            info["episode_chunk_exec_time"] = self._episode_chunk_exec_time
            info["episode_real_exec_time"] = self._episode_real_exec_time
            return self._assemble_state(self._latest_cond_emb), reward, terminated, truncated, info

        # ---- 推下一段 ----
        try:
            chunk, cond, suffix = self._infer_pi()
        except Exception as e:
            info["error"] = f"pi05 infer failed: {e!r}"
            self._finalize_wall_time()
            return (
                self._assemble_state(self._latest_cond_emb),
                self.reward_cfg.crash_penalty,
                True,
                False,
                info,
            )
        self._latest_chunk = chunk
        self._latest_cond_emb = cond
        self._latest_suffix = suffix
        self._chunk_idx += 1

        return self._assemble_state(cond), reward, False, False, info

    def _finalize_wall_time(self):
        """在 episode 结束时把 wall-clock 时间 / obs 排除值算清楚."""
        if self._episode_wall_t0 is not None:
            self._episode_wall_time = max(
                0.0, time.perf_counter() - self._episode_wall_t0
            )
        else:
            self._episode_wall_time = 0.0
        # 兜底: take_chunk_action 里 success 分支也会把当次的 obs_time 累到 self._episode_obs_time;
        # 但 take_action 路径 / per-frame 路径用的是 task_env._last_success_obs_time,
        # 这里再合并一次, 避免漏算.
        last_obs = float(getattr(self._task_env, "_last_success_obs_time", 0.0))
        if last_obs > self._episode_obs_time:
            self._episode_obs_time = last_obs
        self._episode_wall_time_no_obs = max(
            0.0, self._episode_wall_time - self._episode_obs_time
        )

    def close(self):
        if self._task_env is not None:
            try:
                self._task_env.close_env()
            except Exception:
                pass
            self._task_env = None
