# Rainbow DQN for Chunk Speedup (SPEEDTUNING replication)

按 Yuan et al. *"SPEEDTUNING: Speeding Up Policy Execution with
Lightweight Reinforcement Learning"* 实现的 Rainbow DQN，扩展到
**3 维离散动作** `(v, vel_scale, acc_scale)`。比 `speedtune.rl.train`
（SAC）样本效率高一个量级，目标是把训练时间从几万步压到几千步，
让真机训练成为可能。

## 与论文的差异

| 维度 | 论文 SPEEDTUNING | 本实现 |
|---|---|---|
| 动作 | 1 维离散 `v` | **3 维离散** `(v, vel_scale, acc_scale)` (factored) |
| 状态 | proprio + 图像 encoder | pi0.5 prefix mean-pool (cond_emb, 2048D) |
| frame skip | 10（论文一个 step = 一帧） | 1（一个 env.step 已经是一个 chunk） |
| frame stack | 通常 4 | 1（cond_emb 已经融合了时序信息） |
| Reward | `α·v^β + r_task` | `α_v·v^β_v + α_vs·vs^β_vs + α_as·as^β_as + r_task` |
| 探索 | 论文未指定（多半 ε-greedy） | ε-greedy 线性退火 |

Rainbow 全 6 组件中实装的：
- ✅ **Double Q**：online 选 argmax、target 评估
- ✅ **Dueling**：value head + 3 个 advantage head
- ✅ **C51 distributional**：n_atoms=101，V_min=-5, V_max=10
- ✅ **PER**：SumTree proportional，α=0.6，β 0.4→1.0 退火
- ✅ **n-step**：n=3
- ❌ **Noisy Net**：用 ε-greedy 替代（实现简单、调参容易）

## 离散网格（默认）

由 `GRID_STEP`（默认 0.25）统一粒度构造，vel/acc 上界引用物理天花板 `PHYS_*_CEIL`：

| 维度 | 取值 | 档数 |
|---|---|---|
| v (chunk 压缩比率) | 1.0, 1.25, …, 4.0 (step 0.25) | 13 |
| vel_scale | 1.0, 1.25, …, 3.0 (step 0.25) | 9 |
| acc_scale | 1.0, 1.25, …, 3.0 (step 0.25) | 9 |

Factored Q heads → 共 **31 个 Q logits**（每个 logits 是 n_atoms=101 维分布）；
`scalar_v`（paper_A）下只有 v，**|V|=13**。

改粒度只动 `config.py:GRID_STEP` 一个值；改范围/上界改 `ActionGridConfig`。

## Reward 两范式（三 preset 统一默认 time）

详见 `rl/config.py` 顶部说明 + `rl/env.py:_compute_reward`。每个 env.step 算一次：

- **time（默认, 抗 hack, 推荐）**：`r = -α_time·(dense_steps/250) + success_bonus·1{success}`
  - α_time=0.2, success_bonus=15；fallback 按 `fallback_seconds=4` 计时、crash 按 `crash_seconds=8`。
  - 奖励实际省下的时间，唯一高分路径 = 又快又成功；失败只累积负时间 → 抗 hack。
- **knob（论文式 α·v^β, 仅 paper_A 论文复现消融: `--reward_mode knob`）**：
  `r = α_v·v^β_v (+α_vs·vel^β_vs+α_as·acc^β_as) + success_bonus·1{success}`
  - α_v=0.005（须极小, 否则失败 episode 跑满预算每步刷速度奖励）, β_v=2；fallback/crash penalty=0。

## C51 支持（区间随 reward_mode 自动派生）

`c51_bounds_for(reward_mode)` 在 preset 和 train.py 中调用；`--reward_mode knob` 时自动切区间，
无需手动 `--v_min/--v_max`（显式给则优先）：

| reward_mode | V_min | V_max | n_atoms | 回报范围估计 |
|---|---|---|---|---|
| time (默认) | -20.0 | 16.0 | 101 | 快成功 ~+14, 慢失败 ~-9.6 (病态 fallback 更负→钳) |
| knob (消融) | 0.0 | 16.0 | 101 | 非负, 成功 ~+15.6 |

改大 α / success_bonus 时, 相应调整 `c51_bounds_for` 的区间。

## 启动训练

云端拉最新代码后，在已经起好 openpi server 的环境里：

```bash
cd ~/RoboTwin
conda activate RoboTwin

python -m speedtune.rl.rainbow.train \
    --task_name shake_bottle \
    --task_config demo_clean \
    --server_host 127.0.0.1 \
    --server_port 8000 \
    --total_env_steps 30000 \
    --warmup_steps 1000
```

可调参数：
- `--n_atoms` 101 → 51 / 201
- `--v_min --v_max` 调 C51 support
- `--n_step` 3 → 5 (longer credit assignment)
- `--total_env_steps` 30k → 10k (更激进的早停)

## 与 SAC 切换

两个算法**共用同一个 env wrapper**（`speedtune.rl.env.ChunkSpeedupEnv`），
共用 reward 计算和 timing accounting。换算法只需切入口：

```bash
# SAC (3-D continuous, tanh squashed)
python -m speedtune.rl.train ...

# Rainbow (3-D discrete, factored C51)
python -m speedtune.rl.rainbow.train ...
```

TensorBoard 和 checkpoint 都各自落到 `speedtune/rl/runs/<run_name>_<ts>/`。

## 日志字段

训练日志（每 50 step 一行）：
```
[step 1500/30000] ep_ret=2.451 sr=0.40 succ_wall(s)=11.8 (n=6) eps=0.143 buf=1200 elapsed=520s
```

- `ep_ret`：最近 20 episode 平均累计 reward
- `sr`：最近 20 episode 成功率
- `succ_wall(s)`：最近 20 个 **成功 episode** 的 wall_time_no_obs 平均（核心指标）
- `eps`：当前 ε-greedy 概率
- `buf`：buffer 当前条数

TensorBoard 还会写 `train/loss_total`、`train/loss_dim{0,1,2}`、
`train/per_beta`、`eval/*` 等。

## 何时倾向用 Rainbow vs SAC

| 场景 | 推荐 |
|---|---|
| 实机训练 / 想要快速收敛 | **Rainbow** |
| 想观察连续 action 分布 | SAC |
| reward 范围会大幅变化 | SAC（C51 support 固定，重定标麻烦） |
| 离散网格能很好覆盖 sweet spot | **Rainbow** |
