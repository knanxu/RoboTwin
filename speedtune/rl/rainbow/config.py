"""Rainbow DQN config for chunk speedup (SPEEDTUNING-style).

All hyperparameters live here so swapping between SAC and Rainbow only
needs to flip the imports in train_rainbow.py.

Defaults are picked to converge an order of magnitude faster than SAC on
the same task: compact action grid (13/9/9 at step 0.25), large C51 atoms
(101), n-step returns, PER, double + dueling. ε-greedy (linear decay).
"""
from dataclasses import dataclass, field
from typing import Tuple

from ..config import PHYS_VEL_CEIL, PHYS_ACC_CEIL


# 离散动作网格的统一粒度旋钮 (用户选定 0.25). 同时作用于 v / vel_scale / acc_scale;
# 改这一个值即可整体调粗 (0.5) / 调细 (0.1). v∈[1,4], vel/acc∈[1, 物理天花板].
GRID_STEP: float = 0.25


def _lin_grid(lo: float, hi: float, step: float) -> Tuple[float, ...]:
    """构造闭区间 [lo, hi] 上步长为 step 的离散网格 (round 防浮点漂移)."""
    n = int(round((hi - lo) / step)) + 1
    return tuple(round(lo + i * step, 6) for i in range(n))


# ----------------------------------------------------------------------
# Discrete grids for (v, vel_scale, acc_scale)
# ----------------------------------------------------------------------
# 由 GRID_STEP 统一粒度构造 (factored 模式下 |A| = 各维网格长度之和):
#   v   ∈ [1, 4]                  chunk 压缩比率 (1=原速, 4=4× 加速; reconstruct_chunk)
#   vel ∈ [1, PHYS_VEL_CEIL=3.0]  TOPPRA 速度约束 scale (1=base mplib 1.0, 3=物理天花板)
#   acc ∈ [1, PHYS_ACC_CEIL=3.0]  TOPPRA 加速度约束 scale (3.0 = curobo 权威 max_acc)
# step=0.25 → 13 + 9 + 9 = 31 Q-logits (scalar_v 下 |V|=13).
@dataclass
class ActionGridConfig:
    # action_mode: "scalar_v" (后端 A, 论文式, 只用 v) | "v_vel_acc" (后端 B/C)
    action_mode: str = "v_vel_acc"
    v_grid: Tuple[float, ...] = _lin_grid(1.0, 4.0, GRID_STEP)
    vel_grid: Tuple[float, ...] = _lin_grid(1.0, PHYS_VEL_CEIL, GRID_STEP)
    acc_grid: Tuple[float, ...] = _lin_grid(1.0, PHYS_ACC_CEIL, GRID_STEP)

    def active_grids(self) -> Tuple[Tuple[float, ...], ...]:
        if self.action_mode == "scalar_v":
            return (self.v_grid,)
        return (self.v_grid, self.vel_grid, self.acc_grid)

    def dim_names(self) -> Tuple[str, ...]:
        if self.action_mode == "scalar_v":
            return ("v",)
        return ("v", "vel_scale", "acc_scale")


# ----------------------------------------------------------------------
# Reward —— 两种模式, 共同目标: 先保证成功, 再尽量快 (success-first then speed)
# ----------------------------------------------------------------------
# success_bonus 是一次性 terminal 奖励, 必须远大于"加速能省/能刷的奖励":
#
# knob (论文式): r = α_v·v^β_v (+vel/acc) + success_bonus·1{success}, 每个执行成功的 chunk
#   都领速度奖励. ⚠️ RoboTwin 失败 episode 跑到 step_lim 耗尽 (~80 步), 失败-高速可刷
#   80·α_v·v_max^β, 故 α_v 必须极小 (见下). 速度信号因此偏弱 —— 仅作论文复现 (后端 A).
# time (抗 hack, 推荐): r = -α_time·t_step + success_bonus·1{success}. 失败只累积负时间、
#   无正奖励, 唯一高分 = 又快又成功; 成功优先由 success_bonus > α_time·T_max 构造保证.
@dataclass
class RewardConfig:
    # reward_mode: "time" (真实时间惩罚, 抗 hack, 三 preset 统一默认) | "knob" (论文式 α·v^β,
    # 仅 paper_A 做论文复现消融用: --reward_mode knob). 见本文件顶部 Reward 说明.
    reward_mode: str = "time"

    # 成功奖励 (两模式共用, 统一 15). 须远大于"加速可省/可刷的奖励" 以保成功优先.
    success_bonus: float = 15.0

    # --- knob mode (论文式 α·v^β, β=2 为论文 ablation 最优; 仅消融) ---
    # ⚠️ alpha_v 极小是必须的: 成功优先要求 α_v < success_bonus/(N_max·v_max^β)
    #    = 15/(80·16) ≈ 0.0117 → 取 0.005 (失败-v4≈6.4 < 成功≈15.6). 速度信号弱, 加速请用 time.
    alpha_v: float = 0.005
    alpha_vs: float = 0.005
    alpha_as: float = 0.005
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0
    fallback_penalty: float = 0.0    # knob: 不给负 penalty
    crash_penalty: float = 0.0       # knob: 不给负 penalty (仅 terminal)

    # --- time mode (真实时间惩罚, 抗 hack; alpha_time 是 Pareto 旋钮) ---
    alpha_time: float = 0.2          # 0.2 给强速度信号且 success_bonus(15) 远超 α·T_max≈9.6
    fallback_seconds: float = 4.0    # time: fallback 计为这么多秒 (≥ 最慢正常 chunk, 防故意触发)
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
    # 后端 A: 每目标 hold 的物理步数. 取 15 = 对齐 pi0.5 训练数据采集步长 save_freq
    #   (250/15≈16.7Hz), 即 v=1 基线 = 数据原速. ⚠️ 不要用论文 ALOHA 的 50Hz(=5):
    #   pi0.5 在 16.7Hz 数据上训练, 用 5 会让 "v=1" 已比数据快 3×, 破坏 v∈[1,4] 的对比语义.
    stream_hold_steps: int = 15
    # k_skip (论文式 frame skip): A/B 每个 env.step 执行前 k_skip 个动作, 执行完即重推 pi0.5
    # (闭环), 速度策略每 k_skip 动作重决策一次. 论文取 10. C(整段 TOPPRA) 忽略此项.
    k_skip: int = 10

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

    支撑区间必须覆盖回报范围, 否则高回报尾部被钳掉 → 抹平正是要奖励的信号.
    由 c51_bounds_for(reward_mode) 按最终奖励范式自动派生 (preset 与 train.py 都调用它):

    time (三 preset 默认), α_time=0.2, success_bonus=15:
      快成功 ≈ 15-0.2·4.5 ≈ 14.1; 慢失败 ≈ -0.2·48 ≈ -9.6 (病态 fallback 风暴更负, 钳即可).
      → [-20, 16] (必须覆盖 +14 成功).
    knob (仅 paper_A 消融) 非负, α_v=0.005, success_bonus=15:
      失败-高速上界 80·0.005·16 = 6.4; 成功 ≈ 速度奖励(~0.6) + 15 ≈ 15.6 → [0, 16].
    101 atoms; time 跨度 36 → 0.36/atom, knob 跨度 16 → 0.16/atom.
    """
    v_min: float = -20.0
    v_max: float = 16.0
    n_atoms: int = 101


def c51_bounds_for(reward_mode: str) -> Tuple[float, float]:
    """C51 支撑区间随奖励范式自动派生 —— 避免 --reward_mode 覆盖后区间不匹配 (footgun).
      time: 含负时间惩罚 + 成功 +15 → [-20, 16]
      knob: 非负, success_bonus=15 → 上界 ~15.6 → [0, 16]
    """
    if reward_mode == "knob":
        return (0.0, 16.0)
    return (-20.0, 16.0)   # time (默认)


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
        """三个对比预设 —— 三者统一用 time 奖励 (公平对比执行后端, 抗 hack).
        仍可被 CLI 覆盖单个字段; 论文复现消融用 `--mode paper_A --reward_mode knob`
        (C51 区间会随 reward_mode 自动切换, 见 train.py / c51_bounds_for, 无需手动 v_min/v_max).

          paper_A: 后端 A 论文式流式 + 标量 v        (复现对象; 默认 time, knob 仅消融)
          ours_B : 后端 B 逐 action TOPP + (v,vel,acc)
          ours_C : 后端 C 整段 TOPPRA  + (v,vel,acc)  (本工作方法)
        奖励系数 (success_bonus=15, α_time=0.2, α_v=0.005...) 三者共用 RewardConfig 默认值.
        """
        cfg = cls()
        if mode == "paper_A":
            cfg.env.exec_backend = "streaming"
            cfg.env.k_skip = 10                    # 论文式 frame skip (speed 每 10 动作重决策)
            cfg.action_grid.action_mode = "scalar_v"
            # 论文式 baseline: v = chunk 压缩比率 ∈ [1,4] @ GRID_STEP (与 B/C 同源同粒度)
            cfg.action_grid.v_grid = _lin_grid(1.0, 4.0, GRID_STEP)
            cfg.reward.reward_mode = "time"        # 统一 time; 论文复现: --reward_mode knob
        elif mode == "ours_B":
            cfg.env.exec_backend = "per_action"
            cfg.env.k_skip = 10                    # 论文式 frame skip 也加入 B (逐 action TOPP)
            cfg.action_grid.action_mode = "v_vel_acc"
            cfg.reward.reward_mode = "time"
        elif mode == "ours_C":
            cfg.env.exec_backend = "whole_chunk"
            cfg.env.k_skip = None                  # 整段 TOPPRA: k_skip 不适用, 整段执行
            cfg.action_grid.action_mode = "v_vel_acc"
            cfg.reward.reward_mode = "time"
        else:
            raise ValueError(
                f"unknown preset mode: {mode!r} (expect paper_A | ours_B | ours_C)"
            )
        # C51 支撑区间随最终 reward_mode 自动派生 (三者 time → [-20,16]; knob 消融 → [0,16])
        cfg.c51.v_min, cfg.c51.v_max = c51_bounds_for(cfg.reward.reward_mode)
        return cfg
