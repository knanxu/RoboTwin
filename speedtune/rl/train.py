"""SAC 训练入口.

Usage:
    cd /home/xukainan/RoboTwin
    python -m speedtune.rl.train

可在命令行覆盖 config 字段:
    python -m speedtune.rl.train --task_name shake_bottle --total_env_steps 50000
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

from .config import FullConfig
from .env import ChunkSpeedupEnv
from .sac import SAC


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _override_config(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    """Allow flat overrides like --task_name X, --total_env_steps N."""
    flat = {}
    for sub in (cfg.action_space, cfg.reward, cfg.env, cfg.policy, cfg.network, cfg.sac, cfg.train):
        flat.update({k: (sub, k) for k in asdict(sub).keys()})
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in flat:
            sub, key = flat[k]
            cur_val = getattr(sub, key)
            try:
                cast = type(cur_val)(v) if cur_val is not None else v
            except Exception:
                cast = v
            setattr(sub, key, cast)
    return cfg


def _eval(env: ChunkSpeedupEnv, agent: SAC, n_episodes: int) -> dict:
    """Run ``n_episodes`` deterministic rollouts and aggregate metrics.

    Reports three things:
      - success rate over all eval episodes
      - mean episode reward (cumulative discounted-free sum)
      - mean wall-clock execution time of *successful* episodes only,
        with the success-branch get_obs() rendering cost subtracted out
        (so it reflects the controllable execution time, not Sapien's
        post-success snapshot work). Mirrors the wall_time_no_obs accounting
        in speedtune/tests/smoke_replay_expert.py.
    """
    successes = []
    ep_returns = []
    success_wall_times = []   # 仅成功 episode 的 (wall_time - obs_time)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        last_info: dict = {}
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            last_info = info
            done = terminated or truncated
        is_success = bool(last_info.get("success", False))
        successes.append(float(is_success))
        ep_returns.append(ret)
        if is_success:
            wall_no_obs = float(last_info.get("episode_wall_time_no_obs", 0.0))
            success_wall_times.append(wall_no_obs)

    metrics = {
        "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval/return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "eval/n_success": int(sum(successes)),
        "eval/n_episodes": int(len(successes)),
    }
    if success_wall_times:
        metrics["eval/success_wall_time_no_obs_s"] = float(np.mean(success_wall_times))
        metrics["eval/success_wall_time_no_obs_min_s"] = float(np.min(success_wall_times))
        metrics["eval/success_wall_time_no_obs_max_s"] = float(np.max(success_wall_times))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    # 暴露常改的字段; 其他维持默认
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--server_host", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--pi0_step", type=int, default=None)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = FullConfig()
    cfg = _override_config(cfg, args)

    _set_seed(cfg.train.seed)

    run_dir = os.path.join(cfg.train.log_dir, f"{cfg.train.run_name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run dir: {run_dir}")

    # tensorboard 可选; 没有就降级为 stdout
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=run_dir)
    except Exception:
        writer = None

    env = ChunkSpeedupEnv(
        env_cfg=cfg.env,
        action_cfg=cfg.action_space,
        reward_cfg=cfg.reward,
        policy_cfg=cfg.policy,
        seed=cfg.train.seed,
        device=cfg.train.device,
    )
    agent = SAC(
        state_dim=cfg.env.state_dim,
        action_dim=3,
        action_low=env.action_low,
        action_high=env.action_high,
        sac_cfg=cfg.sac,
        net_cfg=cfg.network,
        device=cfg.train.device,
    )

    obs, _ = env.reset(seed=cfg.train.seed)
    ep_return = 0.0
    ep_returns = deque(maxlen=20)
    ep_successes = deque(maxlen=20)
    # 只统计成功 episode 的 wall_time_no_obs (对齐 smoke_replay_expert 的口径)
    success_wall_times = deque(maxlen=20)

    start = time.time()
    for step in range(1, cfg.train.total_env_steps + 1):
        # warmup: 在 action 空间均匀采样
        if step <= cfg.sac.warmup_steps:
            action = np.random.uniform(env.action_low, env.action_high).astype(np.float32)
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done_for_buffer = float(terminated)  # truncated 不算 terminal -> 不切 bootstrap
        agent.buffer.add(obs, action, reward, next_obs, done_for_buffer)

        ep_return += reward
        obs = next_obs

        if terminated or truncated:
            ep_returns.append(ep_return)
            is_success = bool(info.get("success", False))
            ep_successes.append(float(is_success))
            if is_success:
                success_wall_times.append(
                    float(info.get("episode_wall_time_no_obs", 0.0))
                )
            ep_return = 0.0
            obs, _ = env.reset()

        # 学习
        if len(agent.buffer) >= cfg.sac.batch_size and step > cfg.sac.warmup_steps:
            for _ in range(cfg.sac.updates_per_step):
                metrics = agent.update()
            if writer is not None and step % cfg.train.log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

        if step % cfg.train.log_every == 0:
            elapsed = time.time() - start
            avg_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            avg_sr = float(np.mean(ep_successes)) if ep_successes else float("nan")
            avg_succ_wall = (
                float(np.mean(success_wall_times)) if success_wall_times else float("nan")
            )
            print(
                f"[step {step}/{cfg.train.total_env_steps}] "
                f"ep_ret={avg_ret:.3f} sr={avg_sr:.2f} "
                f"succ_wall(s)={avg_succ_wall:.2f} (n={len(success_wall_times)}) "
                f"buf={len(agent.buffer)} elapsed={elapsed:.0f}s",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("rollout/ep_return", avg_ret, step)
                writer.add_scalar("rollout/success_rate", avg_sr, step)
                if success_wall_times:
                    writer.add_scalar(
                        "rollout/success_wall_time_no_obs_s", avg_succ_wall, step
                    )

        if step % cfg.train.eval_every == 0 and step > cfg.sac.warmup_steps:
            print(f"[eval] step {step} ...", flush=True)
            metrics = _eval(env, agent, cfg.train.eval_episodes)
            print(f"[eval] {metrics}", flush=True)
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, step)

        if step % cfg.train.checkpoint_every == 0:
            ckpt_path = os.path.join(run_dir, f"sac_step{step}.pt")
            agent.save(ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}", flush=True)

    env.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
