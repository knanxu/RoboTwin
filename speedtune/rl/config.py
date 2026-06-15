"""Chunk-speedup 训练的共享配置 (动作空间 / 奖励 / env / pi0.5 server).

这些 dataclass 被 ChunkSpeedupEnv (rl/env.py) 直接消费. Rainbow 训练
(rl/rainbow/) 通过 adapter 把自己的 FullConfig 转成这里的形状再喂给 env;
算法专属超参 (C51 / Rainbow / 网络 / 训练循环) 见 rl/rainbow/config.py.

三个执行后端 (exec_backend), 一个 RL 加速模块:
  - "streaming"   后端 A: 论文式固定时长流式 (reconstruct(v) → 逐目标 hold H 步,
                  无 TOPP, 有限差分速度前馈, 连续非零速度). 忠实 SpeedTuning baseline.
                  动作只用标量 v; 不钳物理上限 (PD 跟踪能力即隐式约束, 忠实论文).
  - "per_action"  后端 B: RoboTwin 原生逐 action 点到点 TOPP (take_action), 外露
                  vel_scale/acc_scale. 每个 action 首尾零速 (stop-and-go).
  - "whole_chunk" 后端 C: 整段 TOPPRA (take_chunk_action). 本工作方法.

物理上限 (ARX5 / aloha-agilex): base mplib TOPP 约束 = 1.0 rad/s / 1.0 rad/s²
(mplib 默认 np.ones, = "1× 速度"基准). 不可超的物理天花板见下方常量; B/C 的
vel_scale/acc_scale 放大后被钳在此天花板内.
"""
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 物理天花板 (不可超). acc=3.0 来自 assets/embodiments/aloha-agilex/curobo_left.yml:86
# (RoboTwin 作者给 ARX5 的权威 max_acceleration); vel=3.0 为 ARX5 工程估计
# (≈180°/s, URDF 的 1000 是 Isaac 占位符不可用, 待数据手册逐关节精修).
# ---------------------------------------------------------------------------
PHYS_VEL_CEIL: float = 3.0   # rad/s
PHYS_ACC_CEIL: float = 3.0   # rad/s^2


@dataclass
class ActionSpaceConfig:
    # action_mode: "scalar_v" (后端 A, 论文式, 动作只有 v)
    #            | "v_vel_acc" (后端 B/C, 三参数 v/vel_scale/acc_scale)
    action_mode: str = "v_vel_acc"

    v_low: float = 0.8
    v_high: float = 1.5
    # vel/acc_scale 上界对应 base(1.0) × scale ≤ 物理天花板(3.0), 故上界 ≤ 3.0.
    # (原 vel_high=2.0 / acc_high=4.0; acc=4.0 已超物理上限, 下调到 3.0)
    vel_low: float = 1.0
    vel_high: float = 3.0
    acc_low: float = 1.0
    acc_high: float = 3.0


@dataclass
class RewardConfig:
    """两种可切换奖励 (reward_mode).

    "knob" (论文式, 奖励旋钮值):
        A → r = alpha_v·v^beta_v + r_task
        B/C → r = alpha_v·v^beta_v + alpha_vs·vel^beta_vs + alpha_as·acc^beta_as + r_task
      ⚠️ 仅后端 A 安全且忠实 (A 里 v 线性控时间、beta≥1 防 reward-hacking).
         B/C 用此为消融: 旋钮与真实省时/进度解耦 + chunk 粗粒度折扣 → 会 hack,
         "失败拖长"刷分 > "快速成功", 且调 alpha 救不了 (结构问题).

    "time" (真实时间惩罚, 防 hack, B/C 默认):
        r = -alpha_time · (dense_steps / sim_hz) + (1 if success else 0)
      刷分 = 累积负时间 = 自我限制; alpha_time 即 Pareto 旋钮 (扫它出前沿).
      vel/acc_scale 通过 dense_steps 自动"融合"进来 (省时间才得分, 饱和即止).
      fallback/crash 按一段"慢 chunk"计时, 防 agent 故意触发以逃避时间惩罚.
    """
    reward_mode: str = "knob"

    # --- knob mode ---
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0
    fallback_penalty: float = 0.0    # knob: fallback 不给负 penalty
    crash_penalty: float = 0.0       # knob: crash 不给负 penalty (仅 terminal)

    # --- time mode ---
    alpha_time: float = 0.3          # 时间惩罚系数 (Pareto 旋钮)
    fallback_seconds: float = 4.0    # time: fallback 计为这么多秒 (≥ 最慢正常 chunk)
    crash_seconds: float = 8.0       # time: crash 计为这么多秒


@dataclass
class EnvConfig:
    task_name: str = "shake_bottle"
    task_config: str = "smoke_test"
    instruction_type: str = "unseen"
    chunk_size: int = 50                    # pi0.5 action_horizon
    max_chunks_per_episode: int = 100       # 额外上限, 配合 step_lim

    # exec_backend: "streaming" (A) | "per_action" (B) | "whole_chunk" (C)
    exec_backend: str = "whole_chunk"
    stream_hold_steps: int = 15             # 后端 A: 每目标 hold 的物理步数, 对齐采集 save_freq=15 (250/15≈16.7Hz)

    # 状态维度: cond_emb + last_action(action_dim) + chunk_idx_norm(1) + last_topp_fallback(1)
    # action_dim 由 action_mode 决定 (scalar_v=1, v_vel_acc=3); env 实际计算并校验 state_dim,
    # 这里的默认值仅作 v_vel_acc 情形的参考.
    cond_emb_dim: int = 2048
    state_dim: int = 2048 + 3 + 1 + 1


@dataclass
class PolicyConfig:
    """pi0.5 服务器配置 (本机不加载权重, 通过 websocket 调云端 server).

    云端 openpi server 启动方式见 speedtune/rl/README.md.
    """
    server_host: str = "127.0.0.1"   # 本地通过 SSH 端口转发到云端
    server_port: int = 8000
    api_key: str | None = None
    # 每次从返回的 action chunk 中截取的前 pi0_step 帧 (对齐 pi_model.py:47)
    pi0_step: int = 50
