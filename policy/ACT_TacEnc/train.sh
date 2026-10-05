#!/bin/bash
# bash train.sh <task_name> <data_version: isaac45|isaac51> <expert_data_num> <seed> <gpu_id> [train_config]
# checkpoints: <UniVTAC>/data/act_tacenc/act_ckpt/act-<task>/<data_version>-<N>/<train_config>   (paths.py)
# RESUME=1 : 그 폴더의 체크포인트에서 이어서 학습 (train_state.pt 가 있으면 정확히, 없으면 가중치만)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_name=${1}
data_version=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
train_config=${6:-"train_config_original"}

OUT="${ACT_TACENC_OUT:-${UNIVTAC_ROOT:-/workspace/UniVTAC}/data/act_tacenc}"
export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 "${SCRIPT_DIR}/imitate_episodes.py" \
    --task_name sim-${task_name}-${data_version}-${expert_data_num} \
    --ckpt_dir "${OUT}/act_ckpt/act-${task_name}/${data_version}-${expert_data_num}/${train_config}" \
    --config_path "${SCRIPT_DIR}/${train_config}.yml" \
    --seed ${seed} \
    $([[ "${RESUME:-0}" == 1 ]] && echo --resume)
