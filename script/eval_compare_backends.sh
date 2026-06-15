#!/usr/bin/env bash
#
# 对比三种「动作执行后端」在同一任务 / 同一 seed 上的 eval 效果.
#
#   streaming    后端 A: 论文式固定时长流式 (无 TOPP), 由 STREAM_HOLD_STEPS 控制每个目标 hold 的物理步数
#   per_action   后端 B: RoboTwin 原生逐 action 点到点 TOPP (外露 vel/acc)
#   whole_chunk  后端 C: 整段 TOPPRA 时间重参数化 (默认行为)
#
# 三种后端跑完全相同的任务 / 配置 / seed, 结果分别写到同一个对比文件夹下的三个子目录:
#   <COMPARE_ROOT>/streaming/
#   <COMPARE_ROOT>/per_action/
#   <COMPARE_ROOT>/whole_chunk/
# 每个子目录含: 评测视频 episode*.mp4, _result.txt (成功率), run.log (完整日志).
# 根目录额外生成 summary.txt 汇总三种后端的成功率.
#
# 前置: pi0_client 是 websocket 客户端, 需先在另一个终端启动 policy server, 例如:
#   uv run scripts/serve_policy.py policy:checkpoint \
#       --policy.config=<train_config> --policy.dir=<ckpt_dir> --port=8000
#
# 用法 (默认参数即对应需求: stack_blocks_two / demo_clean / seed 0 / pi0_client):
#   bash script/eval_compare_backends.sh
#
# 可用环境变量覆盖, 例如:
#   TASK_NAME=stack_blocks_two SEED=0 STREAM_HOLD_STEPS=5 bash script/eval_compare_backends.sh
#   BACKENDS="streaming whole_chunk" bash script/eval_compare_backends.sh   # 只跑指定后端

set -uo pipefail
cd "$(dirname "$0")/.."   # 切到仓库根目录 (eval_policy.py 依赖 cwd=仓库根)

# ---------------- 可调参数 (环境变量覆盖) ----------------
TASK_NAME="${TASK_NAME:-stack_blocks_two}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
POLICY_NAME="${POLICY_NAME:-pi0_client}"
CONFIG="${CONFIG:-policy/pi0_client/deploy_policy.yml}"
SEED="${SEED:-0}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-5}"
read -r -a BACKENDS <<< "${BACKENDS:-streaming per_action whole_chunk}"

STAMP="$(date +%Y%m%d_%H%M%S)"
COMPARE_ROOT="${COMPARE_ROOT:-eval_result/_compare/${TASK_NAME}_${STAMP}}"
mkdir -p "$COMPARE_ROOT"

# ---------------- policy server 连通性预检 (仅提示, 不阻塞) ----------------
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
if ! (exec 3<>"/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") 2>/dev/null; then
  echo -e "\033[93m[warn] 无法连接 policy server ${SERVER_HOST}:${SERVER_PORT}.\033[0m"
  echo -e "\033[93m       pi0_client 需要先启动 serve_policy; 若你用的是别的 host/port 可忽略本提示.\033[0m"
fi

echo "============================================================"
echo " Backend comparison"
echo "   task        = $TASK_NAME"
echo "   task_config = $TASK_CONFIG"
echo "   policy      = $POLICY_NAME"
echo "   seed        = $SEED"
echo "   backends    = ${BACKENDS[*]}"
echo "   output      = $COMPARE_ROOT"
echo "============================================================"

# ---------------- 依次运行三种后端 ----------------
for BACKEND in "${BACKENDS[@]}"; do
  OUT_DIR="${COMPARE_ROOT}/${BACKEND}"
  mkdir -p "$OUT_DIR"
  echo ""
  echo "############################################################"
  echo "# backend = $BACKEND  ->  $OUT_DIR"
  echo "############################################################"
  python script/eval_policy.py \
    --config "$CONFIG" \
    --exec_backend "$BACKEND" \
    --stream_hold_steps "$STREAM_HOLD_STEPS" \
    --save_dir "$OUT_DIR" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --seed "$SEED" \
    --policy_name "$POLICY_NAME" \
    2>&1 | tee "${OUT_DIR}/run.log"
  STATUS="${PIPESTATUS[0]}"
  if [[ "$STATUS" -ne 0 ]]; then
    echo -e "\033[91m[error] backend $BACKEND 退出码 $STATUS (跳过, 继续下一个后端).\033[0m"
  fi
done

# ---------------- 汇总成功率 ----------------
SUMMARY="${COMPARE_ROOT}/summary.txt"
{
  echo "Backend comparison summary"
  echo "task=$TASK_NAME  task_config=$TASK_CONFIG  policy=$POLICY_NAME  seed=$SEED"
  echo "stream_hold_steps=$STREAM_HOLD_STEPS"
  echo "------------------------------------------------------------"
  for BACKEND in "${BACKENDS[@]}"; do
    RES="${COMPARE_ROOT}/${BACKEND}/_result.txt"
    if [[ -f "$RES" ]]; then
      RATE="$(tail -n 1 "$RES")"
      printf "%-12s success_rate = %s\n" "$BACKEND" "$RATE"
    else
      printf "%-12s (no result, 见 %s/run.log)\n" "$BACKEND" "${COMPARE_ROOT}/${BACKEND}"
    fi
  done
} | tee "$SUMMARY"

echo ""
echo "Done. 对比目录: $COMPARE_ROOT"
echo "汇总文件: $SUMMARY"
