# Chunk-Speedup RL (SAC)

用 SAC 训一个 agent, 看当前 pi0.5 推理出的 chunk 条件下, 选什么
`(v, vel_scale, acc_scale)` 三元组能最快把任务做完而不掉成功率.

- 动作: `(v, vel_scale, acc_scale)` 绝对值, tanh → linear map 到保守边界
  `v∈[0.8,1.5]`, `vel∈[1.0,2.0]`, `acc∈[1.0,4.0]` (可在 `config.py` 改)
- 状态: pi0.5 action expert 的 mean-pooled prefix features (2048D) +
  上一步动作 (3D) + cnt 进度 (1D) + 上一步是否 topp fallback (1D) = 2053D
- Reward:
  ```
  r_v = α_v * v^β_v + α_vs * vel_scale^β_vs + α_as * acc_scale^β_as
  r_task = 1 if chunk 执行后 check_success else 0
  r = r_v + r_task
  ```
  TOPP fallback 时 `r_v` 屏蔽, 给固定 `-1` penalty.


## 部署拓扑

训练全在云端 GPU 主机上跑, 本机只用来看日志.

```
云端 GPU 主机
└─ docker compose, 一个 image 起多个一次性/常驻容器:
     ├─ collect         采 RoboTwin 专家数据 (Sapien, headless, conda RoboTwin)
     ├─ convert         RoboTwin hdf5 → LeRobot v2.0  (openpi venv)
     ├─ sft             pi0.5 drift SFT (openpi venv, compute_norm_stats + train_pytorch)
     ├─ openpi_server   起 WebSocket 推理服务 (openpi venv)  [常驻]
     └─ robotwin_train  SAC 训练 agent (conda RoboTwin)     [常驻]

本机 (开发/监控)
└─ ssh 进去看 docker logs / 端口转发 tensorboard
```

同一个 image (`speedtune:latest`) 里同时装了:
- uv venv `/opt/venvs/openpi` (py3.11, flax/jax/torch/lerobot)
- conda env `RoboTwin` (py3.10, sapien/mplib/curobo/pytorch3d/websockets)

所有 service 共用这个 image, 各自激活各自的 venv.


## 一把梭 (从零到 ckpt 到 RL 训练)

**服务器上只需要装好 docker + nvidia-container-toolkit**, 其余全在容器里.

```bash
# 0) 克隆两个仓库 side-by-side
mkdir -p ~/work && cd ~/work
git clone <your-openpi-fork>.git openpi
git -C openpi checkout <your-drifting-branch>
git clone <your-RoboTwin-fork>.git RoboTwin

# 1) (可选) 如果你已经有 pi0.5 ckpt, 放到:
#      ~/.cache/openpi/checkpoints/<policy_config>/<exp>/<step>/
#    然后直接跑 runtime stage, 跳过 collect/convert/sft.
#
#    如果从零开始, 直接:
cd ~/work/RoboTwin
bash docker/run_pipeline.sh all
```

`bash docker/run_pipeline.sh all` 会依次执行:
1. `collect`  采 100 集 `shake_bottle` 专家 demo (几十分钟到几小时)
2. `convert`  转成 LeRobot 格式到 `~/.cache/huggingface/lerobot/shake_bottle_drifting_repo`
3. `sft`      在一张 GPU 上训 30k 步 pi0.5 drift (约 12-24 小时)
4. `link`     把 ckpt 软链到 server 挂载点
5. `runtime`  并发起 openpi_server + robotwin_train

每一步也可以单独跑, 见下面 "分阶段调试".


## 分阶段调试

```bash
cd ~/work/RoboTwin

# 采集: 默认 task=shake_bottle, config=demo_clean
bash docker/run_pipeline.sh collect shake_bottle demo_clean

# 数据转换 (hdf5 → LeRobot)
bash docker/run_pipeline.sh convert shake_bottle demo_clean shake_bottle_drifting_repo

# SFT (单卡)
bash docker/run_pipeline.sh sft pi05_aloha_robotwin_drifting_shake_bottle default

# SFT (多卡, 2 张)
NPROC_PER_NODE=2 SFT_GPU=0,1 \
    bash docker/run_pipeline.sh sft pi05_aloha_robotwin_drifting_shake_bottle default

# ckpt 软链 (把 openpi/checkpoints/<pol>/<exp>/<step> 链到 ~/.cache/openpi/checkpoints/)
bash docker/run_pipeline.sh link pi05_aloha_robotwin_drifting_shake_bottle default 29999

# 只起 server (前台, 方便看日志)
bash docker/run_pipeline.sh serve pi05_aloha_robotwin_drifting_shake_bottle default 29999

# 只起 RL 训练 (前提: server 已经起了)
bash docker/run_pipeline.sh rl shake_bottle smoke_test

# server + RL 一起起
bash docker/run_pipeline.sh runtime pi05_aloha_robotwin_drifting_shake_bottle default 29999
```


## 常用环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `TASK_NAME`         | `shake_bottle` | RoboTwin 任务名 |
| `TASK_CONFIG`       | `demo_clean` | collect/convert 阶段的 task_config |
| `RL_TASK_CONFIG`    | `smoke_test` | RL 训练/eval 阶段的 task_config |
| `REPO_ID`           | `shake_bottle_drifting_repo` | LeRobot dataset repo_id, 必须和 openpi config 匹配 |
| `EPISODE_NUM`       | `-1`  | convert 时取前 N 集, -1 = 全部 |
| `POLICY_CONFIG`     | `pi05_aloha_robotwin_drifting_shake_bottle` | openpi train config 名 |
| `EXP_NAME`          | `default` | SFT 实验名 (写到 ckpt 目录) |
| `SFT_STEP`          | `29999` | serve/link 用哪一个 ckpt step |
| `NPROC_PER_NODE`    | `1` | SFT 用几张 GPU (torchrun --nproc_per_node) |
| `OPENPI_GPU`        | `0` | server 容器用哪张 GPU |
| `ROBOTWIN_GPU`      | `0` | RL 训练容器用哪张 GPU |
| `SFT_GPU`           | `0` | SFT 容器用哪张 GPU (多卡: `0,1`) |
| `COLLECT_GPU`       | `0` | collect 容器用哪张 GPU |
| `OPENPI_DATA_HOME`  | `~/.cache/openpi` | 宿主放 ckpt/norm stats 的目录, 挂到 `/openpi_assets` |
| `LEROBOT_HOME`      | `~/.cache/huggingface/lerobot` | 宿主放 LeRobot dataset 的目录 |
| `TOTAL_ENV_STEPS`   | `50000` | RL 训练总步数 |
| `WARMUP_STEPS`      | `1000`  | RL 随机 warmup 步数 |


## 本机看日志 / TensorBoard (可选)

```bash
# 看某个 service 日志
ssh <user>@<云端IP> 'cd work/RoboTwin && docker compose -f docker/compose.yml logs -f robotwin_train'

# TensorBoard 端口转发 (在容器外宿主起, 直接读挂载的 runs/)
ssh -N -f -L 6006:127.0.0.1:6006 <user>@<云端IP>
ssh <user>@<云端IP> 'cd work/RoboTwin && tensorboard --logdir speedtune/rl/runs --port 6006'
# 本机浏览器打开 http://127.0.0.1:6006
```


## 如果要在本机小规模 dev (仍让 pi0.5 在云端)

保留 SSH 端口转发方案, 本机 train.py 不需要改任何东西
(默认 `server_host=127.0.0.1`):

```bash
# 云端
bash docker/run_pipeline.sh serve pi05_aloha_robotwin_drifting_shake_bottle default 29999

# 本机
ssh -N -f -L 8000:localhost:8000 <user>@<云端IP>
python -m speedtune.rl.train   # 本机 Sapien + 云端 pi0.5
```


## 常用命令

```bash
# 只覆盖连接参数, 走云端直连 (不用隧道)
python -m speedtune.rl.train --server_host 1.2.3.4 --server_port 8000

# 切换任务 (需要云端已加载对应 ckpt)
python -m speedtune.rl.train \
  --task_name open_microwave \
  --task_config demo_clean
```

## 关键代码位置

| 文件 | 作用 |
|---|---|
| `envs/_base_task.py:take_chunk_action` | 执行器, 加了 `v` 参数做 chunk reconstruct |
| `envs/utils/chunk_accel.py` | `reconstruct_chunk(chunk, v)` 实现 |
| `envs/robot/toppra_chunk_executor.py` | 整段 TOPPRA 时间重参数化 |
| `speedtune/rl/env.py` | Gym-like env, 包装 RoboTwin + 远端 pi0.5 |
| `speedtune/rl/networks.py` | SAC actor + twin critic |
| `speedtune/rl/sac.py` | SAC 主算法 (twin Q + 自动 alpha) |
| `speedtune/rl/config.py` | 所有超参集中管理 |
| `speedtune/rl/train.py` | 训练入口 |

openpi 侧:

| 文件 | 作用 |
|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py:_sample_actions_drifting` | `return_hidden=True` 时返回 `suffix_out` + `cond_emb` |
| `src/openpi/policies/policy.py:infer_with_hidden` | Policy 级别透出 hidden |
| `src/openpi/serving/websocket_policy_server.py` | server 根据 `_return_hidden` flag 分发 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py:infer_with_hidden` | client 发带 flag 的请求 |


## 已知的坑

1. **云端 checkpoint 必须和 `--policy.config` 对得上**. drift model 的
   config 在 `src/openpi/training/config.py` 里都是
   `pi05_aloha_robotwin_drifting_<task>` 形式, 不要拿 upstream 的
   `pi05_aloha` 去 serve.
2. **WebSocket max_size**. observation 包含 3 张 RGB 图像, 一个 chunk
   推理往返大概 1~3 MB, 已经通过 `max_size=None` 关掉限制.
3. **首 chunk 延迟**. 第一次 pi0.5 推理要 warmup JAX/PyTorch, 可能
   3-10s, 后续稳定在百 ms 级 (GPU 型号而定).
4. **训练侧 policy_client 不复位**. 每个 episode 不关 websocket, 保持
   长连接, 避免每次 handshake 开销.
5. **云端 Sapien headless**. 云端 GPU 机器通常没显示器, 需要 EGL 渲染:
   ```bash
   export PYOPENGL_PLATFORM=egl
   export DISPLAY=                # 清空 X display
   ```
   如果还是报 `Failed to open X display`, 检查
   `task_config/<task>.yml:render_freq`, 把它设成 0 (关 viewer) 再跑.
6. **GPU 分片**. pi0.5 server 会吃掉绝大多数显存, RoboTwin Sapien
   默认也拿 device 0. 如果单卡紧张, 要么:
   - server 用 `CUDA_VISIBLE_DEVICES=0 docker compose ...`, RL 用
     `CUDA_VISIBLE_DEVICES=1 python -m speedtune.rl.train`
   - 或者把 SAC 放 cpu (`--device cpu`), 只让 Sapien 和 pi0.5 抢 GPU0
7. **Docker + conda 同机**. docker 容器里 server 监听主机 8000,
   conda 宿主里 client 连 `127.0.0.1:8000`. 这条走的是 loopback, 不需要
   开外网端口. 但如果改 `network_mode: bridge`, 宿主连的就是容器 IP,
   那 config 里 `server_host` 也得改. 保持 `host` 模式最省心.
