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
├─ Docker 容器: openpi server (监听 0.0.0.0:8000)
└─ 宿主 conda env "RoboTwin":
     speedtune/rl/train.py
       └─ WebsocketClient(127.0.0.1:8000)   ← 走宿主回环, 无额外延迟
       └─ RoboTwin Sapien sim (headless)
       └─ SAC (actor/critic on cuda)

本机 (开发/监控)
└─ ssh 进去看日志 / 端口转发 tensorboard
```

本机完全不跑推理, pi0.5 权重也不用下到本机. 云端 GPU ≥ 24GB
(PaliGemma 2B bf16 大约占 15GB, 留 5-8GB 给 RoboTwin + SAC).


## Step 1: fork 仓库

- fork **RoboTwin** (本仓库)
- fork **openpi** 的带 drift model 和 `infer_with_hidden` 的分支

把两个 fork 都 push 上去, 云端直接从 GitHub 拉.


## Step 2: 云端环境准备

登录云端主机一次性装好两套环境:

```bash
# 2a. openpi server (docker 方式, 自带 uv 依赖)
git clone <your-openpi-fork>.git && cd openpi
git checkout <your-drifting-branch>
# 把 checkpoint 放到默认挂载位置 (compose.yml 映射 ~/.cache/openpi → /openpi_assets)
mkdir -p ~/.cache/openpi/checkpoints
# 从你的云盘/rsync 把 ckpt 同步到
#   ~/.cache/openpi/checkpoints/pi05_aloha_robotwin_drifting_shake_bottle/default/29999
cd ..

# 2b. RoboTwin conda env (宿主直接装, 不走 docker)
git clone <your-RoboTwin-fork>.git && cd RoboTwin
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh     # 装 sapien/mplib 并 patch 源码
pip install websockets msgpack msgpack-numpy   # SAC 侧调 server 用
# pytorch 按需装 cuda 对应版本
cd ..
```


## Step 3: 云端起 pi0.5 server

**一个单独的 terminal** (tmux 会话里更稳):

```bash
cd openpi
export SERVER_ARGS="--port 8000 policy:checkpoint \
  --policy.config=pi05_aloha_robotwin_drifting_shake_bottle \
  --policy.dir=/openpi_assets/checkpoints/pi05_aloha_robotwin_drifting_shake_bottle/default/29999"
docker compose -f scripts/docker/compose.yml up --build
```

等日志打出 `server listening on 0.0.0.0:8000` 就绪. `network_mode: host`
让容器直接复用主机的 8000 端口, 宿主里 `curl http://127.0.0.1:8000/healthz`
应该返回 `OK`.


## Step 4: 云端起训练

另开一个 terminal (也放 tmux 里):

```bash
cd RoboTwin
conda activate RoboTwin
python -m speedtune.rl.train \
  --task_name shake_bottle \
  --task_config smoke_test \
  --server_host 127.0.0.1 \
  --server_port 8000 \
  --total_env_steps 50000 \
  --warmup_steps 1000
```

日志和 tensorboard 都写到 `speedtune/rl/runs/<run_name>_<ts>/`.


## Step 5: 本机看日志 (可选)

```bash
# 5a. 直接 ssh 进去 tail 日志
ssh <user>@<云端IP> 'tmux a -t sac'

# 5b. 把云端 tensorboard 端口转发回本机
ssh -N -f -L 6006:127.0.0.1:6006 <user>@<云端IP>
# 云端先起 tensorboard
ssh <user>@<云端IP> 'cd RoboTwin && tensorboard --logdir speedtune/rl/runs --port 6006'
# 本机浏览器打开 http://127.0.0.1:6006
```


## 如果要在本机小规模 dev (仍然让 pi0.5 在云端)

保留原来那套 SSH 端口转发方案即可, 本机跑 train.py 不需要改任何东西
(`server_host=127.0.0.1` 会被转发到云端 8000):

```bash
ssh -N -f -L 8000:localhost:8000 <user>@<云端IP>
python -m speedtune.rl.train   # 本机 Sapien + 云端 pi0.5
```

> 注: 本机模式下 RoboTwin sim 依然占本机 GPU. RoboTwin 任务小, 消费
> GPU 主要是 Sapien 的渲染 (<2GB), 不推理 pi0.5, 本机 8GB 显存就够.


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
