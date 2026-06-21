#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 并行启动三个 SpeedTuning RL 训练 (三者统一 time 奖励, 公平对比执行后端):
#   paper_A : streaming   + 标量 v        + time 奖励  (k_skip=10; 论文复现消融: 加 --reward_mode knob)
#   ours_B  : per_action  + (v,vel,acc)   + time 奖励  (k_skip=10)
#   ours_C  : whole_chunk + (v,vel,acc)   + time 奖励  (整段 TOPPRA, 无 k_skip)
#
# 三训练默认共用同一个云端 pi0.5 server (你已在云端起的那个).
# ⚠️ 单 server 会串行处理三路推理 → 各自吞吐≈1/3. 若要满速, 起 3 个 server (不同端口/GPU)
#    并用下面的 PORT 数组分别指定.
#
# 用法:
#   bash speedtune/run_three_trainings.sh
#   TASK_CONFIG=demo_randomized SERVER_PORT=8000 bash speedtune/run_three_trainings.sh
#   WANDB=false TOTAL_STEPS=50000 bash speedtune/run_three_trainings.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/xukainan/RoboTwin}"
PY="${PY:-/home/xukainan/miniforge3/envs/RoboTwin/bin/python}"

# ---- 可调参数 (环境变量覆盖) ----
# server checkpoint = pi05_aloha_robotwin_drifting_stack_blocks_two → RoboTwin 任务类 stack_blocks_two
TASK_NAME="${TASK_NAME:-stack_blocks_two}"
# ⚠️ checkpoint 名含 "drifting": 若该策略是用域随机化数据训的, 把这里改成 demo_randomized
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"     # 云端 server (一般 SSH -L 端口转发到本地 127.0.0.1)
SERVER_PORT="${SERVER_PORT:-8000}"          # serve_policy 默认端口
TOTAL_STEPS="${TOTAL_STEPS:-30000}"
SEED="${SEED:-42}"
PI0_STEP="${PI0_STEP:-50}"
WANDB="${WANDB:-true}"                       # "false" 关闭 wandb
STAGGER="${STAGGER:-20}"                     # 错开启动的秒数 (避免三个 sapien sim 同时初始化)

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="${LOG_ROOT:-$REPO/speedtune/rl/runs/parallel_${STAMP}}"
mkdir -p "$LOG_ROOT"

MODES=(paper_A ours_B ours_C)
# 每个训练用不同 seed (探索去相关)
declare -A SEED_OFF=( [paper_A]=0 [ours_B]=1 [ours_C]=2 )
# 若起了多个 server, 在此为每个 mode 指定不同端口; 默认都用 SERVER_PORT
declare -A PORT=( [paper_A]=$SERVER_PORT [ours_B]=$SERVER_PORT [ours_C]=$SERVER_PORT )

echo "============================================================"
echo " SpeedTuning 三训练并行启动"
echo "   task        : $TASK_NAME ($TASK_CONFIG)"
echo "   pi0.5 server: $SERVER_HOST:$SERVER_PORT  (三训练共用 → 串行推理)"
echo "   total_steps : $TOTAL_STEPS   seed base: $SEED   wandb: $WANDB"
echo "   logs        : $LOG_ROOT"
echo "============================================================"

PIDS=()
cleanup() {
  echo ""
  echo "[run] 收到中断, 终止全部训练 ..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
  wait 2>/dev/null
  exit 130
}
trap cleanup INT TERM

cd "$REPO"
for m in "${MODES[@]}"; do
  run_seed=$(( SEED + ${SEED_OFF[$m]} ))
  logf="$LOG_ROOT/${m}.log"
  echo "[run] 启动 $m  (seed=$run_seed, port=${PORT[$m]})  → $logf"
  "$PY" -m speedtune.rl.rainbow.train \
      --mode "$m" \
      --task_name "$TASK_NAME" --task_config "$TASK_CONFIG" \
      --server_host "$SERVER_HOST" --server_port "${PORT[$m]}" \
      --pi0_step "$PI0_STEP" \
      --total_env_steps "$TOTAL_STEPS" \
      --seed "$run_seed" \
      --run_name "speedtune_${m}" \
      --log_dir "$LOG_ROOT" \
      --wandb_enabled "$WANDB" \
      > "$logf" 2>&1 &
  PIDS+=($!)
  sleep "$STAGGER"
done

echo ""
echo "[run] 三训练已启动 (PIDs: ${PIDS[*]})"
echo "[run] 实时日志:  tail -f $LOG_ROOT/paper_A.log"
echo "[run] 终止全部:  kill ${PIDS[*]}   (或在本终端 Ctrl-C)"
echo "[run] 等待全部结束 ..."

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[run] ⚠️  PID $pid 异常退出 (看对应 .log)"
    FAIL=1
  fi
done
echo "[run] 全部结束 (FAIL=$FAIL). 日志在 $LOG_ROOT"
exit $FAIL
