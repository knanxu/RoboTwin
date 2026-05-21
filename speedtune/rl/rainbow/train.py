"""Rainbow DQN training entry point.

Usage:
    cd ~/RoboTwin
    python -m speedtune.rl.rainbow.train \
        --task_name shake_bottle --task_config demo_clean \
        --server_host 127.0.0.1 --server_port 8000

The env wrapper, reward shape, and timing accounting are reused from the
SAC variant (speedtune.rl.env.ChunkSpeedupEnv). Only the agent differs.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

from ..env import ChunkSpeedupEnv
from ..config import (
    ActionSpaceConfig as _SACActionSpaceConfig,
    EnvConfig as _SACEnvConfig,
    PolicyConfig as _SACPolicyConfig,
    RewardConfig as _SACRewardConfig,
)
from .agent import RainbowAgent
from .config import FullConfig


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _override_config(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    flat = {}
    for sub in (
        cfg.action_grid,
        cfg.reward,
        cfg.env,
        cfg.policy,
        cfg.network,
        cfg.c51,
        cfg.rainbow,
        cfg.train,
    ):
        flat.update({k: (sub, k) for k in asdict(sub).keys()})
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in flat:
            sub, key = flat[k]
            cur = getattr(sub, key)
            try:
                cast = type(cur)(v) if cur is not None else v
            except Exception:
                cast = v
            setattr(sub, key, cast)
    return cfg


def _to_sac_action_space(cfg: FullConfig):
    """Pull (low, high) bounds from the discrete grids so ChunkSpeedupEnv
    can still clamp / record `last_action` consistently."""
    grids = cfg.action_grid
    return _SACActionSpaceConfig(
        v_low=float(grids.v_grid[0]),
        v_high=float(grids.v_grid[-1]),
        vel_low=float(grids.vel_grid[0]),
        vel_high=float(grids.vel_grid[-1]),
        acc_low=float(grids.acc_grid[0]),
        acc_high=float(grids.acc_grid[-1]),
    )


def _to_sac_reward(cfg: FullConfig):
    rc = cfg.reward
    return _SACRewardConfig(
        alpha_v=rc.alpha_v, alpha_vs=rc.alpha_vs, alpha_as=rc.alpha_as,
        beta_v=rc.beta_v, beta_vs=rc.beta_vs, beta_as=rc.beta_as,
        fallback_penalty=rc.fallback_penalty, crash_penalty=rc.crash_penalty,
    )


def _to_sac_env(cfg: FullConfig):
    ec = cfg.env
    return _SACEnvConfig(
        task_name=ec.task_name,
        task_config=ec.task_config,
        instruction_type=ec.instruction_type,
        chunk_size=ec.chunk_size,
        max_chunks_per_episode=ec.max_chunks_per_episode,
        cond_emb_dim=ec.cond_emb_dim,
        state_dim=ec.state_dim,
    )


def _to_sac_policy(cfg: FullConfig):
    pc = cfg.policy
    return _SACPolicyConfig(
        server_host=pc.server_host,
        server_port=pc.server_port,
        api_key=pc.api_key,
        pi0_step=pc.pi0_step,
    )


def _eval(env: ChunkSpeedupEnv, agent: RainbowAgent, env_step: int, n_episodes: int) -> dict:
    """Greedy eval: ε=0, deterministic argmax."""
    successes = []
    ep_returns = []
    success_wall_times = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        last_info: dict = {}
        while not done:
            action, _idx = agent.select_action(obs, env_step=env_step, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            last_info = info
            done = terminated or truncated
        is_success = bool(last_info.get("success", False))
        successes.append(float(is_success))
        ep_returns.append(ret)
        if is_success:
            success_wall_times.append(
                float(last_info.get("episode_wall_time_no_obs", 0.0))
            )
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
    # Same overrides surface as the SAC trainer for muscle memory.
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
    # C51 / Rainbow specific
    parser.add_argument("--n_atoms", type=int, default=None)
    parser.add_argument("--v_min", type=float, default=None)
    parser.add_argument("--v_max", type=float, default=None)
    parser.add_argument("--n_step", type=int, default=None)
    args = parser.parse_args()

    cfg = FullConfig()
    cfg = _override_config(cfg, args)

    _set_seed(cfg.train.seed)

    run_dir = os.path.join(cfg.train.log_dir, f"{cfg.train.run_name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run dir: {run_dir}", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=run_dir)
    except Exception:
        writer = None

    env = ChunkSpeedupEnv(
        env_cfg=_to_sac_env(cfg),
        action_cfg=_to_sac_action_space(cfg),
        reward_cfg=_to_sac_reward(cfg),
        policy_cfg=_to_sac_policy(cfg),
        seed=cfg.train.seed,
        device=cfg.train.device,
    )

    grids = cfg.action_grid
    K_tuple = (len(grids.v_grid), len(grids.vel_grid), len(grids.acc_grid))
    print(
        f"[train] action grid sizes: v={K_tuple[0]} vs={K_tuple[1]} as={K_tuple[2]} "
        f"(total Q-logits = {sum(K_tuple)})",
        flush=True,
    )

    agent = RainbowAgent(
        state_dim=cfg.env.state_dim,
        num_actions_per_dim=K_tuple,
        action_grids=(
            np.asarray(grids.v_grid, dtype=np.float32),
            np.asarray(grids.vel_grid, dtype=np.float32),
            np.asarray(grids.acc_grid, dtype=np.float32),
        ),
        c51_cfg=cfg.c51,
        rainbow_cfg=cfg.rainbow,
        net_cfg=cfg.network,
        device=cfg.train.device,
    )

    obs, _ = env.reset(seed=cfg.train.seed)
    ep_return = 0.0
    ep_returns = deque(maxlen=20)
    ep_successes = deque(maxlen=20)
    success_wall_times = deque(maxlen=20)

    start = time.time()
    rcfg = cfg.rainbow
    beta_steps = max(1, rcfg.per_beta_steps)
    for step in range(1, cfg.train.total_env_steps + 1):
        # PER β annealing toward 1.0
        frac = min(1.0, step / beta_steps)
        beta = rcfg.per_beta_start + frac * (rcfg.per_beta_end - rcfg.per_beta_start)

        if step <= rcfg.warmup_steps:
            # Random discrete actions during warmup so the buffer sees
            # a diverse set of (v, vs, as) before learning starts.
            idx = np.array(
                [np.random.randint(K) for K in K_tuple],
                dtype=np.int64,
            )
            action = np.array(
                [agent.action_grids[i][idx[i]] for i in range(3)],
                dtype=np.float32,
            )
        else:
            action, idx = agent.select_action(obs, env_step=step, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer.add(obs, idx, reward, next_obs, float(terminated))

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

        if len(agent.buffer) >= rcfg.batch_size and step > rcfg.warmup_steps:
            for _ in range(rcfg.updates_per_step):
                metrics = agent.update(beta=beta)
            if writer is not None and step % cfg.train.log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/epsilon", agent.epsilon(step), step)
                writer.add_scalar("train/per_beta", beta, step)

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
                f"eps={agent.epsilon(step):.3f} buf={len(agent.buffer)} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("rollout/ep_return", avg_ret, step)
                writer.add_scalar("rollout/success_rate", avg_sr, step)
                if success_wall_times:
                    writer.add_scalar(
                        "rollout/success_wall_time_no_obs_s", avg_succ_wall, step
                    )

        if step % cfg.train.eval_every == 0 and step > rcfg.warmup_steps:
            print(f"[eval] step {step} ...", flush=True)
            metrics = _eval(env, agent, env_step=step, n_episodes=cfg.train.eval_episodes)
            print(f"[eval] {metrics}", flush=True)
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, step)

        if step % cfg.train.checkpoint_every == 0:
            ckpt_path = os.path.join(run_dir, f"rainbow_step{step}.pt")
            agent.save(ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}", flush=True)

    env.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
