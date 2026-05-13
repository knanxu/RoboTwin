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
    successes, ep_lens, ep_returns, ep_times = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            steps += 1
            done = terminated or truncated
        successes.append(float(info.get("success", False)))
        ep_lens.append(steps)
        ep_returns.append(ret)
        ep_times.append(float(env._episode_total_time))
    return {
        "eval/success_rate": float(np.mean(successes)),
        "eval/return": float(np.mean(ep_returns)),
        "eval/ep_chunks": float(np.mean(ep_lens)),
        "eval/ep_total_time_s": float(np.mean(ep_times)),
    }


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
    ep_len = 0
    ep_returns = deque(maxlen=20)
    ep_lens = deque(maxlen=20)
    ep_successes = deque(maxlen=20)

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
        ep_len += 1
        obs = next_obs

        if terminated or truncated:
            ep_returns.append(ep_return)
            ep_lens.append(ep_len)
            ep_successes.append(float(info.get("success", False)))
            ep_return = 0.0
            ep_len = 0
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
            avg_len = float(np.mean(ep_lens)) if ep_lens else float("nan")
            avg_sr = float(np.mean(ep_successes)) if ep_successes else float("nan")
            print(
                f"[step {step}/{cfg.train.total_env_steps}] "
                f"ep_ret={avg_ret:.3f} ep_len={avg_len:.1f} sr={avg_sr:.2f} "
                f"buf={len(agent.buffer)} elapsed={elapsed:.0f}s",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("rollout/ep_return", avg_ret, step)
                writer.add_scalar("rollout/ep_len", avg_len, step)
                writer.add_scalar("rollout/success_rate", avg_sr, step)

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
