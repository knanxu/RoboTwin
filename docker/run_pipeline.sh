#!/usr/bin/env bash
# End-to-end driver for the speedtune Docker pipeline.
#
# 所有阶段都走 docker compose, 你的宿主机只需要 docker + nvidia-container-toolkit.
#
# Stages:
#   build     一次性 build image (强烈推荐先单独跑这一步, 见下文)
#   collect   采集 RoboTwin 专家数据 (hdf5)
#   convert   RoboTwin hdf5 -> LeRobot v2.0 (供 openpi 训练)
#   sft       pi0.5 drift SFT (先 compute_norm_stats, 再 train_pytorch)
#   link      把 openpi 训出的 ckpt 软链到 /openpi_assets 下, 供 server 吃
#   serve     起 pi0.5 WebSocket server
#   rl        跑 SAC 加速 agent
#   runtime   serve + rl 一起起 (等价 docker compose --profile runtime up)
#   all       collect + convert + sft + link + runtime 一条龙
#
# 重要: 第一次用先单独 build, 不要直接 `docker compose --profile all build`.
# 那样会让 6 个 service 并行 build 6 次, 网络更容易撞限流.
# 用:
#   bash docker/run_pipeline.sh build
# 或者直接:
#   docker compose -f docker/compose.yml build openpi_server
#
# Examples:
#   # 如果你已经有 pi0.5 ckpt, 想直接跑 RL:
#   bash docker/run_pipeline.sh runtime
#
#   # 从零开始跑单任务 (shake_bottle):
#   bash docker/run_pipeline.sh all
#
#   # 分步调试:
#   bash docker/run_pipeline.sh collect shake_bottle demo_clean 100
#   bash docker/run_pipeline.sh convert shake_bottle demo_clean shake_bottle_drifting_repo
#   NPROC_PER_NODE=2 SFT_GPU=0,1 bash docker/run_pipeline.sh sft \
#         pi05_aloha_robotwin_drifting_shake_bottle default
#   bash docker/run_pipeline.sh link pi05_aloha_robotwin_drifting_shake_bottle default 29999
#   bash docker/run_pipeline.sh runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/compose.yml"

DC="docker compose -f $COMPOSE_FILE"

# 默认值 (允许用环境变量覆盖)
: "${TASK_NAME:=shake_bottle}"
: "${TASK_CONFIG:=demo_clean}"            # collect/convert 阶段用, RL 阶段可不同
: "${REPO_ID:=shake_bottle_drifting_repo}"
: "${EPISODE_NUM:=-1}"
: "${TASK_PROMPT:=shake the bottle}"
: "${POLICY_CONFIG:=pi05_aloha_robotwin_drifting_shake_bottle}"
: "${EXP_NAME:=default}"
: "${SFT_STEP:=29999}"  # link / serve 用, 指向哪一个 ckpt step
: "${RL_TASK_CONFIG:=smoke_test}"  # RL eval 配置
: "${OPENPI_DATA_HOME:=$HOME/.cache/openpi}"

log() { echo -e "\033[1;34m[$(date +%H:%M:%S)] $*\033[0m"; }

usage() {
    grep -E '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

stage_build() {
    # 单独 build 一次 image. 复用一个 service 的 build spec 就够了
    # (所有 service 共用 speedtune:latest), 避免 --profile all 并行 build.
    log "build    speedtune:latest (single pass)"
    $DC build openpi_server "$@"
}

stage_collect() {
    local task="${1:-$TASK_NAME}"
    local cfg="${2:-$TASK_CONFIG}"
    log "collect  task=$task config=$cfg"
    TASK_NAME="$task" TASK_CONFIG="$cfg" \
        $DC --profile collect run --rm collect
}

stage_convert() {
    local task="${1:-$TASK_NAME}"
    local cfg="${2:-$TASK_CONFIG}"
    local repo="${3:-$REPO_ID}"
    log "convert  task=$task config=$cfg repo_id=$repo"
    TASK_NAME="$task" TASK_CONFIG="$cfg" REPO_ID="$repo" \
    EPISODE_NUM="$EPISODE_NUM" TASK_PROMPT="$TASK_PROMPT" \
        $DC --profile convert run --rm convert
}

stage_sft() {
    local pol="${1:-$POLICY_CONFIG}"
    local exp="${2:-$EXP_NAME}"
    log "sft      policy=$pol exp=$exp nproc=${NPROC_PER_NODE:-1}"
    POLICY_CONFIG="$pol" EXP_NAME="$exp" \
        $DC --profile sft run --rm sft
}

stage_link() {
    # 把 openpi 训出的 checkpoint (在 RoboTwin 仓外的 openpi/checkpoints/... )
    # 软链到 $OPENPI_DATA_HOME/checkpoints/ 下, 让 server 吃到
    local pol="${1:-$POLICY_CONFIG}"
    local exp="${2:-$EXP_NAME}"
    local step="${3:-$SFT_STEP}"
    local src="$(cd "$SCRIPT_DIR/../.." && pwd)/openpi/checkpoints/$pol/$exp/$step"
    local dst_dir="$OPENPI_DATA_HOME/checkpoints/$pol/$exp"
    local dst="$dst_dir/$step"
    if [ ! -d "$src" ]; then
        echo "[link] source ckpt not found: $src" >&2
        exit 2
    fi
    mkdir -p "$dst_dir"
    ln -sfn "$src" "$dst"
    log "link     $dst -> $src"
}

stage_serve() {
    local pol="${1:-$POLICY_CONFIG}"
    local exp="${2:-$EXP_NAME}"
    local step="${3:-$SFT_STEP}"
    log "serve    policy=$pol exp=$exp step=$step"
    POLICY_CONFIG="$pol" \
    POLICY_DIR="/openpi_assets/checkpoints/$pol/$exp/$step" \
        $DC --profile serve up --build openpi_server
}

stage_rl() {
    local task="${1:-$TASK_NAME}"
    local cfg="${2:-$RL_TASK_CONFIG}"
    log "rl       task=$task config=$cfg"
    TASK_NAME="$task" TASK_CONFIG="$cfg" \
        $DC --profile rl up --build robotwin_train
}

stage_runtime() {
    local pol="${1:-$POLICY_CONFIG}"
    local exp="${2:-$EXP_NAME}"
    local step="${3:-$SFT_STEP}"
    log "runtime  policy=$pol exp=$exp step=$step  (server + rl)"
    POLICY_CONFIG="$pol" \
    POLICY_DIR="/openpi_assets/checkpoints/$pol/$exp/$step" \
    TASK_CONFIG="$RL_TASK_CONFIG" \
        $DC --profile runtime up --build
}

stage_all() {
    stage_collect  "$TASK_NAME" "$TASK_CONFIG"
    stage_convert  "$TASK_NAME" "$TASK_CONFIG" "$REPO_ID"
    stage_sft      "$POLICY_CONFIG" "$EXP_NAME"
    stage_link     "$POLICY_CONFIG" "$EXP_NAME" "$SFT_STEP"
    stage_runtime  "$POLICY_CONFIG" "$EXP_NAME" "$SFT_STEP"
}

[ $# -ge 1 ] || usage
cmd="$1"; shift || true

case "$cmd" in
    build)   stage_build "$@" ;;
    collect) stage_collect "$@" ;;
    convert) stage_convert "$@" ;;
    sft)     stage_sft "$@" ;;
    link)    stage_link "$@" ;;
    serve)   stage_serve "$@" ;;
    rl)      stage_rl "$@" ;;
    runtime) stage_runtime "$@" ;;
    all)     stage_all ;;
    help|-h|--help) usage ;;
    *) echo "unknown stage: $cmd" >&2; usage ;;
esac
