#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
expert_speed_factor=${4}   # 可选第4参: <1.0 放慢专家数据 (覆盖 task_config 同名值)

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

EXTRA=""
cache_setting="${task_config}"
if [ -n "${expert_speed_factor}" ]; then
    EXTRA="--expert_speed_factor ${expert_speed_factor}"
    cache_setting="${task_config}_esf${expert_speed_factor}"   # 与 collect_data.py 的 save_path 后缀一致
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config $EXTRA
rm -rf data/${task_name}/${cache_setting}/.cache
