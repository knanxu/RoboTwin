"""
SAC for chunk-speedup 训练配置.

所有超参集中在这里, 方便手调. 默认值按用户指定:
  - action 空间: 保守边界
  - reward 系数: alpha=0.05 均匀, beta=[2, 2, 1]
  - topp fallback: 固定 -1 penalty, 屏蔽 r_v
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ActionSpaceConfig:
    # 保守边界: 在 smoke_replay_expert 验证过的安全区
    v_low: float = 0.8
    v_high: float = 1.5
    vel_low: float = 1.0
    vel_high: float = 2.0
    acc_low: float = 1.0
    acc_high: float = 4.0


@dataclass
class RewardConfig:
    """
    r_v = alpha_v * v^beta_v + alpha_vs * vel_scale^beta_vs + alpha_as * acc_scale^beta_as
    r_task ∈ {0, 1}  若 chunk 执行后 check_success == True 则为 1, 否则 0
    r_total = r_v + r_task

    TOPP fallback 时 r_v 被屏蔽, 只给固定 -1 penalty.
    """
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0

    fallback_penalty: float = -1.0   # topp fallback 固定 penalty (屏蔽 r_v)
    crash_penalty: float = -5.0      # 物理崩溃 / 推理失败 的 terminal penalty


@dataclass
class EnvConfig:
    task_name: str = "shake_bottle"
    task_config: str = "smoke_test"
    instruction_type: str = "unseen"
    chunk_size: int = 50                    # pi0.5 action_horizon
    max_chunks_per_episode: int = 100       # 额外上限, 配合 step_lim

    # 状态维度: cond_emb + last_action(3) + chunk_idx_norm(1) + last_topp_fallback(1)
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


@dataclass
class NetworkConfig:
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005              # target network soft update
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 0.2
    target_entropy: float = -3.0    # -action_dim
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 1000        # 纯随机 action 的 step 数
    updates_per_step: int = 1


@dataclass
class TrainConfig:
    total_env_steps: int = 50_000
    eval_every: int = 2000
    eval_episodes: int = 10
    checkpoint_every: int = 5000
    log_every: int = 50

    seed: int = 42
    device: str = "cuda"

    run_name: str = "sac_chunk_speedup"
    log_dir: str = "./speedtune/rl/runs"


@dataclass
class FullConfig:
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
