"""Rainbow DQN training entry point with Weights & Biases logging.

Usage:
    cd ~/RoboTwin
    python -m speedtune.rl.rainbow.train \
        --task_name shake_bottle --task_config demo_clean \
        --server_host 127.0.0.1 --server_port 8000

    # Disable wandb (offline / no account):
    python -m speedtune.rl.rainbow.train ... --wandb_enabled false

The env wrapper, reward shape, and timing accounting are reused from the
SAC variant (speedtune.rl.env.ChunkSpeedupEnv). Only the agent differs.
"""
from __future__ import annotations

import argparse
import os
import re
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
from .config import FullConfig, c51_bounds_for


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
    grids = cfg.action_grid
    return _SACActionSpaceConfig(
        action_mode=grids.action_mode,
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
        reward_mode=rc.reward_mode,
        success_bonus=rc.success_bonus,
        alpha_v=rc.alpha_v, alpha_vs=rc.alpha_vs, alpha_as=rc.alpha_as,
        beta_v=rc.beta_v, beta_vs=rc.beta_vs, beta_as=rc.beta_as,
        fallback_penalty=rc.fallback_penalty, crash_penalty=rc.crash_penalty,
        alpha_time=rc.alpha_time, fallback_seconds=rc.fallback_seconds,
        crash_seconds=rc.crash_seconds,
    )


def _to_sac_env(cfg: FullConfig):
    ec = cfg.env
    return _SACEnvConfig(
        task_name=ec.task_name,
        task_config=ec.task_config,
        instruction_type=ec.instruction_type,
        chunk_size=ec.chunk_size,
        max_chunks_per_episode=ec.max_chunks_per_episode,
        exec_backend=ec.exec_backend,
        stream_hold_steps=ec.stream_hold_steps,
        k_skip=ec.k_skip,
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
        "evaluation/task_success_rate": float(np.mean(successes)) if successes else 0.0,
        "evaluation/episode_cumulative_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "evaluation/number_of_successful_episodes": int(sum(successes)),
        "evaluation/total_evaluation_episodes": int(len(successes)),
    }
    if success_wall_times:
        metrics["evaluation/successful_episode_execution_time_mean_seconds"] = float(np.mean(success_wall_times))
        metrics["evaluation/successful_episode_execution_time_min_seconds"] = float(np.min(success_wall_times))
        metrics["evaluation/successful_episode_execution_time_max_seconds"] = float(np.max(success_wall_times))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        choices=["paper_A", "ours_B", "ours_C"],
                        help="执行后端预设: paper_A (论文式流式 baseline) | "
                             "ours_B (逐 action TOPP) | ours_C (整段 TOPPRA). "
                             "不给则用 FullConfig 默认 (whole_chunk + knob).")
    parser.add_argument("--reward_mode", type=str, default=None,
                        choices=["knob", "time"], help="覆盖预设的 reward_mode")
    # 奖励 / k_skip 调参 (覆盖 preset; 名字与 config 字段一致, 经 _override_config 生效)
    parser.add_argument("--k_skip", type=int, default=None,
                        help="论文式 frame skip: A/B 每 env.step 执行的动作数 (C 忽略)")
    parser.add_argument("--success_bonus", type=float, default=None,
                        help="成功 terminal 奖励 (须远大于加速可省/可刷的奖励)")
    parser.add_argument("--alpha_time", type=float, default=None,
                        help="time 模式时间惩罚系数 (Pareto 旋钮; 调大→更激进加速)")
    parser.add_argument("--alpha_v", type=float, default=None,
                        help="knob 模式速度奖励系数 (须极小防 hack)")
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
    # wandb
    parser.add_argument("--wandb_project", type=str, default="speedtune-rainbow")
    parser.add_argument("--wandb_enabled", type=str, default="true",
                        help="Set to 'false' to disable wandb logging")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Existing wandb run id to resume into. Find it on the wandb "
                             "run page URL (the last path segment) or in "
                             "<run_dir>/wandb/latest-run/files/wandb-metadata.json.")
    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a previous rainbow_stepN.pt checkpoint. "
                             "Loads net/target/opt and continues from env step N "
                             "(replay buffer is NOT saved, will refill).")
    args = parser.parse_args()

    cfg = FullConfig.preset(args.mode) if args.mode else FullConfig()
    cfg = _override_config(cfg, args)

    # C51 支撑区间随最终 reward_mode 自动派生: 使 `--reward_mode knob` 消融无需手动给
    # --v_min/--v_max (time→[-20,16], knob→[0,16]). 用户显式 --v_min/--v_max 仍优先.
    if args.v_min is None and args.v_max is None:
        cfg.c51.v_min, cfg.c51.v_max = c51_bounds_for(cfg.reward.reward_mode)

    _set_seed(cfg.train.seed)

    run_dir = os.path.join(cfg.train.log_dir, f"{cfg.train.run_name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run dir: {run_dir}", flush=True)

    # ---- wandb init ----
    wandb_enabled = args.wandb_enabled.lower() not in ("false", "0", "no", "off")
    try:
        import wandb
    except ImportError:
        wandb = None
        wandb_enabled = False

    if wandb_enabled and wandb is not None:
        wandb_init_kwargs = dict(
            project=args.wandb_project,
            name=f"{cfg.train.run_name}_{cfg.env.task_name}",
            config=asdict(cfg),
            dir=run_dir,
        )
        if args.wandb_run_id is not None:
            wandb_init_kwargs["id"] = args.wandb_run_id
            wandb_init_kwargs["resume"] = "allow"
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run_id is not None:
            print(f"[wandb] resuming run id '{args.wandb_run_id}' "
                  f"in project '{args.wandb_project}'", flush=True)
        else:
            print(f"[wandb] logging to project '{args.wandb_project}'", flush=True)
    else:
        if wandb is not None:
            wandb.init(mode="disabled")
        print("[wandb] disabled", flush=True)

    # ---- tensorboard (保留作为本地备用) ----
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
    active = grids.active_grids()
    dim_names = grids.dim_names()
    K_tuple = tuple(len(g) for g in active)
    print(
        f"[train] exec_backend={cfg.env.exec_backend} reward_mode={cfg.reward.reward_mode} "
        f"action_mode={grids.action_mode} dims={dim_names} grid_sizes={K_tuple} "
        f"(total Q-logits = {sum(K_tuple)}) state_dim={env.state_dim}",
        flush=True,
    )

    agent = RainbowAgent(
        state_dim=env.state_dim,
        num_actions_per_dim=K_tuple,
        action_grids=tuple(np.asarray(g, dtype=np.float32) for g in active),
        c51_cfg=cfg.c51,
        rainbow_cfg=cfg.rainbow,
        net_cfg=cfg.network,
        device=cfg.train.device,
    )

    start_step = 0
    if args.resume_from is not None:
        print(f"[resume] loading checkpoint: {args.resume_from}", flush=True)
        agent.load(args.resume_from)
        m = re.search(r"step(\d+)", os.path.basename(args.resume_from))
        if m:
            start_step = int(m.group(1))
        print(
            f"[resume] continuing from env step {start_step}; "
            f"replay buffer is empty and will refill before next update",
            flush=True,
        )

    obs, _ = env.reset(seed=cfg.train.seed)
    ep_return = 0.0
    ep_returns = deque(maxlen=20)
    ep_successes = deque(maxlen=20)
    success_wall_times = deque(maxlen=20)

    start = time.time()
    rcfg = cfg.rainbow
    beta_steps = max(1, rcfg.per_beta_steps)
    for step in range(start_step + 1, cfg.train.total_env_steps + 1):
        frac = min(1.0, step / beta_steps)
        beta = rcfg.per_beta_start + frac * (rcfg.per_beta_end - rcfg.per_beta_start)

        if step <= rcfg.warmup_steps:
            idx = np.array(
                [np.random.randint(K) for K in K_tuple],
                dtype=np.int64,
            )
            action = np.array(
                [agent.action_grids[i][idx[i]] for i in range(len(K_tuple))],
                dtype=np.float32,
            )
        else:
            action, idx = agent.select_action(obs, env_step=step, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer.add(obs, idx, reward, next_obs, float(terminated))

        ep_return += reward
        obs = next_obs

        if terminated or truncated:
            # 任何形式的 episode 结束都要排空 n-step 窗口 (terminated 时 add 已排空,
            # 此调用为空操作; truncated 时不排空会把下个 episode 的 reward/state
            # 跨 reset 边界折进旧 episode 的尾部 transition).
            agent.buffer.flush_episode()
            ep_returns.append(ep_return)
            is_success = bool(info.get("success", False))
            ep_successes.append(float(is_success))
            if is_success:
                success_wall_times.append(
                    float(info.get("episode_wall_time_no_obs", 0.0))
                )
            ep_return = 0.0
            obs, _ = env.reset()

        # ---- learning ----
        if len(agent.buffer) >= rcfg.batch_size and step > rcfg.warmup_steps:
            for _ in range(rcfg.updates_per_step):
                train_metrics = agent.update(beta=beta)

            if step % cfg.train.log_every == 0:
                # Wandb: training losses and hyperparams
                # 全称命名:
                #   training/total_distributional_loss         - 三维 C51 cross-entropy 之和
                #   training/critic_loss_dimension_v           - v 维度的 C51 cross-entropy
                #   training/critic_loss_dimension_vel_scale   - vel_scale 维度
                #   training/critic_loss_dimension_acc_scale   - acc_scale 维度
                #   training/epsilon_greedy_probability        - 当前 ε
                #   training/prioritized_replay_beta           - PER importance-sampling β
                wandb_train = {
                    "training/total_distributional_loss": train_metrics["loss_total"],
                    "training/epsilon_greedy_probability": agent.epsilon(step),
                    "training/prioritized_replay_beta": beta,
                }
                for _i, _name in enumerate(dim_names):
                    wandb_train[f"training/critic_loss_dimension_{_name}"] = \
                        train_metrics[f"loss_dim{_i}"]
                if wandb_enabled and wandb is not None:
                    wandb.log(wandb_train, step=step)
                if writer is not None:
                    for k, v in wandb_train.items():
                        writer.add_scalar(k, v, step)

        # ---- stdout + rollout logging ----
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

            # Wandb: rollout metrics
            # 全称命名:
            #   rollout/episode_cumulative_return             - 最近 20 episode 的平均累计 reward
            #   rollout/task_success_rate                     - 最近 20 episode 成功率
            #   rollout/successful_episode_execution_time_seconds
            #       - 最近 20 个成功 episode 的平均执行时间 (去除渲染耗时)
            #   rollout/replay_buffer_size                    - buffer 当前条数
            rollout_log = {
                "rollout/episode_cumulative_return": avg_ret,
                "rollout/task_success_rate": avg_sr,
                "rollout/replay_buffer_size": float(len(agent.buffer)),
            }
            if success_wall_times:
                rollout_log["rollout/successful_episode_execution_time_seconds"] = avg_succ_wall
            if wandb_enabled and wandb is not None:
                wandb.log(rollout_log, step=step)
            if writer is not None:
                for k, v in rollout_log.items():
                    writer.add_scalar(k, v, step)

        # ---- evaluation ----
        if step % cfg.train.eval_every == 0 and step > rcfg.warmup_steps:
            print(f"[eval] step {step} ...", flush=True)
            eval_metrics = _eval(env, agent, env_step=step, n_episodes=cfg.train.eval_episodes)
            print(f"[eval] {eval_metrics}", flush=True)
            # eval 复用了训练 env (内部多次 reset), 训练上下文已失效: 旧 obs 是 stale 的,
            # env 停在 eval 末态 (可能 eval_success=True) —— 不 reset 的话下一个 train step
            # 会产生一条带虚假 +1 成功奖励的脏 transition, 且 PER 会优先采样它.
            agent.buffer.flush_episode()
            obs, _ = env.reset()
            ep_return = 0.0

            # Wandb: eval metrics
            # 全称命名:
            #   evaluation/task_success_rate
            #       - eval episodes 中任务成功的比例
            #   evaluation/episode_cumulative_return
            #       - eval episodes 的平均累计 reward
            #   evaluation/number_of_successful_episodes
            #       - eval 中成功的 episode 数量
            #   evaluation/total_evaluation_episodes
            #       - eval 总 episode 数量
            #   evaluation/successful_episode_execution_time_mean_seconds
            #       - 成功 episode 的平均真实执行时间 (排除成功帧渲染耗时)
            #   evaluation/successful_episode_execution_time_min_seconds
            #       - 成功 episode 中最短执行时间
            #   evaluation/successful_episode_execution_time_max_seconds
            #       - 成功 episode 中最长执行时间
            if wandb_enabled and wandb is not None:
                wandb.log(eval_metrics, step=step)
            if writer is not None:
                for k, v in eval_metrics.items():
                    writer.add_scalar(k, v, step)

        # ---- checkpoint ----
        if step % cfg.train.checkpoint_every == 0:
            ckpt_path = os.path.join(run_dir, f"rainbow_step{step}.pt")
            agent.save(ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}", flush=True)

    env.close()
    if wandb_enabled and wandb is not None:
        wandb.finish()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
