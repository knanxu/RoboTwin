"""SPEEDTUNING-style Rainbow DQN for chunk speedup.

Reference: Yuan et al., "SPEEDTUNING: Speeding Up Policy Execution with
Lightweight Reinforcement Learning", 2024.

三个执行后端共用一个 RL 加速模块 (见 config.FullConfig.preset):
  - paper_A: 论文式固定时长流式 (无 TOPP) + 标量 v + 论文 knob 奖励 (忠实 baseline)
  - ours_B : RoboTwin 原生逐 action TOPP + (v, vel_scale, acc_scale) + 时间奖励
  - ours_C : 整段 TOPPRA + (v, vel_scale, acc_scale) + 时间奖励 (本工作方法)

Differences vs the paper:
  - 可选 1-D (v) 或 3-D factored 动作空间 (v, vel_scale, acc_scale); 每维一个
    dueling C51 head (factored Q heads)
  - VLM mean-pooled conditioning embedding as state (cheaper + richer than
    proprioception + image encoder)
  - 默认每 chunk 一次决策 (论文是帧级 k_skip=10; 见 plan 已知保真度差距)
"""
