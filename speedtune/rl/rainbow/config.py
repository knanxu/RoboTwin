"""Rainbow DQN config for chunk speedup (SPEEDTUNING-style).

All hyperparameters live here so swapping between SAC and Rainbow only
needs to flip the imports in train_rainbow.py.

Defaults are picked to converge an order of magnitude faster than SAC on
the same task: small action grid (8/6/7), large C51 atoms (101), n-step
returns, PER, double + dueling. ε-greedy for exploration (linear decay).
"""
from dataclasses import dataclass, field
from typing import Tuple


# ----------------------------------------------------------------------
# Discrete grids for (v, vel_scale, acc_scale)
# ----------------------------------------------------------------------
# Conservative bounds (validated by smoke_replay_expert.py), with steps
# small enough to give the agent meaningful resolution but small enough
# to keep |A| tractable: 8 + 6 + 7 = 21 Q-logits in factored mode.
@dataclass
class ActionGridConfig:
    # action_mode: "scalar_v" (后端 A, 论文式, 只用 v) | "v_vel_acc" (后端 B/C)
    action_mode: str = "v_vel_acc"
    v_grid: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    # vel/acc 上界对应 base(1.0)×scale ≤ 物理天花板(3.0) (原 2.0/4.0; acc=4.0 超物理上限)
    vel_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    acc_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)

    def active_grids(self) -> Tuple[Tuple[float, ...], ...]:
        if self.action_mode == "scalar_v":
            return (self.v_grid,)
        return (self.v_grid, self.vel_grid, self.acc_grid)

    def dim_names(self) -> Tuple[str, ...]:
        if self.action_mode == "scalar_v":
            return ("v",)
        return ("v", "vel_scale", "acc_scale")


# ----------------------------------------------------------------------
# Reward (Plan A: non-negative reward only)
# ----------------------------------------------------------------------
# Per-chunk reward when TOPP solves and the chunk runs:
#   r_v   = α_v · v^β_v + α_vs · vs^β_vs + α_as · as^β_as
#   r_task = 1 if check_success() else 0
#   r_total = r_v + r_task                              ∈ [0.13, 1.51]
#
# When TOPP fallback or env crash:
#   r_total = 0   (no positive signal; episode budget keeps draining as
#                  an implicit cost, so the agent still learns to avoid it)
#   crash additionally terminates the episode.
#
# This keeps Q values ≥ 0 so C51 support can sit on [V_min=0, V_max] (见 C51Config).
@dataclass
class RewardConfig:
    # reward_mode: "knob" (论文式 α·v^β+r_task; 仅后端 A 安全) | "time" (真实时间惩罚, 防 hack)
    reward_mode: str = "knob"

    # --- knob mode ---
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0
    fallback_penalty: float = 0.0    # knob: 不给负 penalty
    crash_penalty: float = 0.0       # knob: 不给负 penalty (仅 terminal)

    # --- time mode ---
    alpha_time: float = 0.3          # 时间惩罚系数 (Pareto 旋钮)
    fallback_seconds: float = 4.0    # time: fallback 计为这么多秒 (防故意触发)
    crash_seconds: float = 8.0       # time: crash 计为这么多秒


# ----------------------------------------------------------------------
# Env (chunk-level wrapper, same as SAC)
# ----------------------------------------------------------------------
@dataclass
class EnvConfig:
    task_name: str = "shake_bottle"
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    chunk_size: int = 50
    max_chunks_per_episode: int = 100

    # exec_backend: "streaming" (A) | "per_action" (B) | "whole_chunk" (C)
    exec_backend: str = "whole_chunk"
    stream_hold_steps: int = 5            # 后端 A: 每目标 hold 物理步数 (250/5=50Hz)

    cond_emb_dim: int = 2048
    # state_dim 由 env 按 action_mode 实际计算 (cond_emb + action_dim + 2); 训练时
    # 用 env.state_dim 构建 agent. 这里默认值仅作 v_vel_acc 情形参考.
    state_dim: int = 2048 + 3 + 1 + 1


@dataclass
class PolicyConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    api_key: str | None = None
    pi0_step: int = 50


# ----------------------------------------------------------------------
# Rainbow DQN
# ----------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """Dueling + factored C51 head."""
    backbone_sizes: Tuple[int, ...] = (512, 512, 256)
    # Per-dim Q head MLP (after the shared backbone)
    head_hidden: int = 128


@dataclass
class C51Config:
    """Categorical distributional Q (C51 / Bellemare et al. 2017).

    Plan A reward is non-negative:
      single-chunk r ∈ [0.13, 1.51] (max only at terminal success step)
      max r_v per chunk = 0.05·1.5² + 0.05·2² + 0.05·4 ≈ 0.51 at full speed
      successful episode ~10-30 chunks, only the last gets r_task=1
      worst-case discounted return (full speed, 30 chunks, γ=0.99):
        0.51 · (1-0.99³⁰)/0.01 + 1 ≈ 14.3
      V_max must cover this — a smaller bound (the old 8.0) clamps the Q
      distribution exactly for sustained-high-speed behaviour, flattening the
      value signal the task is trying to reward.
      101 atoms → resolution 0.15/atom, still fine-grained for r ≈ 0.2~0.5.
    """
    v_min: float = 0.0
    v_max: float = 15.0
    n_atoms: int = 101


@dataclass
class RainbowConfig:
    gamma: float = 0.99
    n_step: int = 3              # multi-step return horizon
    tau_target: float = 1.0      # 1.0 == hard target update every target_update_every
    target_update_every: int = 500

    lr: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 1000
    updates_per_step: int = 1

    # PER (Schaul et al. 2015)
    per_alpha: float = 0.6           # priority exponent
    per_beta_start: float = 0.4      # IS-weight exponent (annealed to 1.0)
    per_beta_end: float = 1.0
    per_beta_steps: int = 30_000     # anneal over this many env steps
    per_eps: float = 1e-6            # min priority

    # ε-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000

    # Gradient clipping (per Rainbow paper Adam settings)
    grad_clip: float = 10.0
    adam_eps: float = 1.5e-4


@dataclass
class TrainConfig:
    total_env_steps: int = 30_000
    eval_every: int = 2000
    eval_episodes: int = 10
    checkpoint_every: int = 5000
    log_every: int = 50

    seed: int = 42
    device: str = "cuda"

    run_name: str = "rainbow_chunk_speedup"
    log_dir: str = "./speedtune/rl/runs"


@dataclass
class FullConfig:
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    c51: C51Config = field(default_factory=C51Config)
    rainbow: RainbowConfig = field(default_factory=RainbowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def preset(cls, mode: str) -> "FullConfig":
        """三个对比预设. 仍可被 CLI / 后续覆盖单个字段 (如 A 切 time 奖励).

          paper_A: 后端 A 论文式流式 + 标量 v + 论文 knob 奖励 (忠实 SpeedTune baseline)
          ours_B : 后端 B 逐 action TOPP + (v,vel,acc) + 时间奖励
          ours_C : 后端 C 整段 TOPPRA + (v,vel,acc) + 时间奖励 (本工作方法)
        """
        cfg = cls()
        if mode == "paper_A":
            cfg.env.exec_backend = "streaming"
            cfg.action_grid.action_mode = "scalar_v"
            # 论文式 baseline v 取更宽范围 (覆盖加速/减速), 因 A 里 v 线性控时间
            cfg.action_grid.v_grid = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
            cfg.reward.reward_mode = "knob"        # A 上安全且忠实 (beta≥1 防 hack)
            cfg.c51.v_min, cfg.c51.v_max = 0.0, 15.0
        elif mode == "ours_B":
            cfg.env.exec_backend = "per_action"
            cfg.action_grid.action_mode = "v_vel_acc"
            cfg.reward.reward_mode = "time"        # B/C 默认时间奖励 (防 hack)
            cfg.c51.v_min, cfg.c51.v_max = -20.0, 2.0
        elif mode == "ours_C":
            cfg.env.exec_backend = "whole_chunk"
            cfg.action_grid.action_mode = "v_vel_acc"
            cfg.reward.reward_mode = "time"
            cfg.c51.v_min, cfg.c51.v_max = -20.0, 2.0
        else:
            raise ValueError(
                f"unknown preset mode: {mode!r} (expect paper_A | ours_B | ours_C)"
            )
        return cfg
