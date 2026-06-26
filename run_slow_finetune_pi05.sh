#!/bin/bash
# =============================================================================
# run_slow_finetune_pi05.sh
#   慢速专家数据 → pi0.5 (drift) 微调 一键流程  (RoboTwin 采集 + openpi 训练)
#
#   [1] 采集慢速 demo (esf<1.0)  →  [2] 转中间 hdf5  →  [3] 转 LeRobot  →  [4] 微调 pi0.5(drift)
#
# 环境要求 (重要!):
#   * 在 RoboTwin 的 conda 环境下运行: 阶段 1/2 的 `python` 需要 sapien/h5py/cv2。
#   * 阶段 3/4 内部用 `uv run`, 自动使用 policy/pi05 与 openpi 各自的 .venv (不受当前 conda 影响)。
#   * 不需要起 RoboTwin env server —— openpi 是纯离线监督微调。
#
# 用法:
#   conda activate <robotwin_env>
#   bash run_slow_finetune_pi05.sh
#   # 后台长跑:   nohup bash run_slow_finetune_pi05.sh > slow_finetune.log 2>&1 &
#   # 断点重跑:   把已完成阶段的 DO_* 置 0 再跑 (例如数据已转好, 只重训: DO_COLLECT/PROCESS/CONVERT=0)
#   # 只跑单任务: 改 TASKS=(stack_blocks_two)
# =============================================================================
set -uo pipefail

# ===================== 可配置区 (按你的云端环境改) =====================
ROBOTWIN_ROOT=/home/chenlu/RoboTwin
OPENPI_ROOT=/home/chenlu/openpi
TASKS=(place_phone_stand stack_blocks_two place_empty_cup)
TASK_CONFIG=demo_clean            # 采集基础配置 (无域随机化; 要域随机化改 demo_randomized)
ESF=0.25                          # 速度倍率 <1.0 变慢 (0.25 ≈ 慢 4 倍)
GPU=0                             # 采集/转换用 (阶段1-3, 单卡够)
TRAIN_GPUS=0,1,2,3                # 训练用 (阶段4); 列出所有要用的卡
FSDP_DEVICES=4                    # 模型分片到几张卡 (FSDP 省显存)。openpi 默认=1 只复制不分片→单卡必 OOM。
                                  # 约束: 卡数 % FSDP_DEVICES==0, 且 batch_size(drift默认32) % 卡数==0 (32%4=0 ✓)
EXP_NAME=drifting_slow025         # 训练 exp 名 (ckpt 输出子目录名)
TRAIN_CFG_PREFIX=pi05_aloha_robotwin_drifting   # drift config 前缀 (在 openpi)
NUM_STEPS=20000                   # 训练步数 (覆盖 config 默认 30000); 最后 ckpt 在 step NUM_STEPS-1 = 19999
SAVE_INTERVAL=1000                # 每多少步存一次 (中途崩溃可 --resume; 最终仍只留 1 个, 见 KEEP_PERIOD)
KEEP_PERIOD=$NUM_STEPS            # ≥NUM_STEPS → 关闭周期保留; 配合 openpi 硬编码 max_to_keep=1 → 最终只留 19999 一个
                                  # 想"全程只存这 1 次"(省 I/O, 但崩溃要重来): 把 SAVE_INTERVAL 也设成 =NUM_STEPS

# 阶段开关 (1=执行, 0=跳过; 便于断点重跑)
DO_COLLECT=1
DO_PROCESS=1
DO_CONVERT=1
DO_TRAIN=1
# =====================================================================

SETTING="${TASK_CONFIG}_esf${ESF}"          # 与 collect_data.py 的 save_path 后缀一致
PI05_DIR="$ROBOTWIN_ROOT/policy/pi05"
FAILED=()

c()  { echo -e "\n\033[95m========== $* ==========\033[0m"; }
ts() { date '+%F %T'; }
skip_failed() { printf '%s\n' "${FAILED[@]:-}" | grep -qE "$1"; }   # 该任务前序阶段失败?
# 已采集 episode 数 (= process_data 的 expert_data_num, 也用于定位中间目录)
get_n() { ls "$ROBOTWIN_ROOT/data/$1/$SETTING/data/"episode*.hdf5 2>/dev/null | wc -l; }

# ---- 环境/路径硬检查 ----
[ -d "$ROBOTWIN_ROOT" ] || { echo "ERROR: ROBOTWIN_ROOT 不存在: $ROBOTWIN_ROOT"; exit 1; }
[ -d "$OPENPI_ROOT" ]   || { echo "ERROR: OPENPI_ROOT 不存在: $OPENPI_ROOT"; exit 1; }
[ -f "$ROBOTWIN_ROOT/collect_data.sh" ] || { echo "ERROR: 缺 collect_data.sh"; exit 1; }
grep -q "expert_speed_factor" "$ROBOTWIN_ROOT/collect_data.sh" \
  || { echo "ERROR: collect_data.sh 未含 --esf 改动 (需先同步含 esf 后缀的版本)"; exit 1; }

echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT  OPENPI_ROOT=$OPENPI_ROOT"
echo "TASKS=${TASKS[*]}"
echo "SETTING=$SETTING  GPU=$GPU  EXP_NAME=$EXP_NAME"
echo "stages: collect=$DO_COLLECT process=$DO_PROCESS convert=$DO_CONVERT train=$DO_TRAIN"

# ========================= 阶段 1: 采集慢速数据 =========================
if [ "$DO_COLLECT" = 1 ]; then
  for t in "${TASKS[@]}"; do
    c "[1/4] $(ts) 采集 $t  (config=$TASK_CONFIG, esf=$ESF)"
    cd "$ROBOTWIN_ROOT"
    bash collect_data.sh "$t" "$TASK_CONFIG" "$GPU" "$ESF" \
      || { echo "WARN: 采集 $t 失败, 跳过其后续阶段"; FAILED+=("collect:$t"); }
  done
fi

# ====================== 阶段 2: RoboTwin hdf5 → 中间 hdf5 ======================
if [ "$DO_PROCESS" = 1 ]; then
  for t in "${TASKS[@]}"; do
    skip_failed "collect:$t" && continue
    N=$(get_n "$t")
    if [ "$N" -le 0 ]; then echo "WARN: $t 无 episode (先跑阶段1), 跳过"; FAILED+=("process:$t"); continue; fi
    c "[2/4] $(ts) process_data $t  (N=$N)"
    cd "$PI05_DIR"
    bash process_data_pi05.sh "$t" "$SETTING" "$N" \
      || { echo "WARN: process_data $t 失败"; FAILED+=("process:$t"); }
  done
fi

# ====================== 阶段 3: 中间 hdf5 → LeRobot dataset ======================
if [ "$DO_CONVERT" = 1 ]; then
  for t in "${TASKS[@]}"; do
    skip_failed "(collect|process):$t" && continue
    N=$(get_n "$t")
    PROC="processed_data/${t}-${SETTING}-${N}"
    if [ ! -d "$PI05_DIR/$PROC" ]; then echo "WARN: 缺 $PROC (先跑阶段2), 跳过 $t"; FAILED+=("convert:$t"); continue; fi
    c "[3/4] $(ts) convert→LeRobot $t  ($PROC → ${t}_drifting_repo)"
    cd "$PI05_DIR"
    # 注意: repo_id=<task>_drifting_repo 会覆盖原速同名 LeRobot repo (慢速微调可接受)
    bash generate.sh "$PROC" "${t}_drifting_repo" \
      || { echo "WARN: convert $t 失败"; FAILED+=("convert:$t"); }
  done
fi

# ====================== 阶段 4: 微调 pi0.5 drift (在 openpi) ======================
if [ "$DO_TRAIN" = 1 ]; then
  # 护栏: 确认确实用 openpi 的 train.py (不是 policy/pi05 那个; drift config 只在 openpi)
  [ -f "$OPENPI_ROOT/scripts/train.py" ] || { echo "ERROR: 缺 $OPENPI_ROOT/scripts/train.py"; exit 1; }
  for t in "${TASKS[@]}"; do
    skip_failed "(collect|process|convert):$t" && continue
    c "[4/4] $(ts) 微调 $t  (config=${TRAIN_CFG_PREFIX}_$t, exp=$EXP_NAME, steps=$NUM_STEPS)"
    cd "$OPENPI_ROOT"
    echo "  进入 openpi 训练: cwd=$(pwd)  train.py=$OPENPI_ROOT/scripts/train.py"   # 一眼确认在 openpi
    # num_train_steps=20000 → 最后 ckpt 在 step 19999 (train.py:272 末步强制存)
    # keep_period≥steps + openpi 硬编码 max_to_keep=1 → 最终只留 19999 一个
    # norm_stats 复用 trossen(pi05_base) 不重算; 从 pi05_base 权重微调
    echo "  训练用 GPU=$TRAIN_GPUS  fsdp_devices=$FSDP_DEVICES (FSDP 分片省显存)"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
      uv run scripts/train.py "${TRAIN_CFG_PREFIX}_$t" --exp-name="$EXP_NAME" --overwrite \
        --num-train-steps="$NUM_STEPS" --save-interval="$SAVE_INTERVAL" --keep-period="$KEEP_PERIOD" \
        --fsdp-devices="$FSDP_DEVICES" \
      || { echo "WARN: 训练 $t 失败"; FAILED+=("train:$t"); }
  done
fi

# ============================== 汇总 ==============================
c "$(ts) 全部结束"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo -e "\033[91m失败项: ${FAILED[*]}\033[0m"
else
  echo -e "\033[92m全部成功\033[0m"
fi
echo "ckpt: $OPENPI_ROOT/checkpoints/${TRAIN_CFG_PREFIX}_<task>/$EXP_NAME/$((NUM_STEPS-1))/{params,assets/trossen}  (只此一个)"
echo "交 SpeedTune/expo-ft: export SPEEDTUNE_VLA_CKPT=<上面>/params ; SPEEDTUNE_VLA_ASSETS=<上面>/assets ; SPEEDTUNE_VLA_ASSET_ID=trossen"
