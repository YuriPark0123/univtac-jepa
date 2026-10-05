#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 평가 큐 — scripts/eval_queue.tsv 를 위에서부터 순서대로 평가한다 (컨테이너 안, tmux 에서 실행)
#
#   tmux new -s eval
#   bash /workspace/UniVTAC/univtac-jepa/scripts/eval_queue.sh                 # (다른 목록: 첫 인자로 tsv 경로)
#   (빠져나오기: Ctrl-b d,  다시 붙기: tmux attach -t eval,  중단: Ctrl-c — 다시 실행하면 남은 seed 부터 이어서)
#
# 한 항목 = 100 seed 를 PROCS 개 shard 로 나눠 Isaac 프로세스 PROCS 개로 동시에 돌린다 (shard i 는 GPUS 의 i 번째 GPU, 순환).
#   Isaac 평가 1개 ≈ GPU 10 GB. 다른 작업과 GPU 를 나눠 쓰면 reset 120 s 제한에 걸리기 쉽다 → 그 seed 는 같은 seed 로 재시도.
#   ★ 한 항목의 PROCS 는 바꾸지 말 것 (shard 파일 이름에 shard 수가 들어간다; 바꾸면 eval_run.py 가 거부한다)
# 환경변수: PROCS(2)  GPUS("0")  UNIVTAC_ROOT(/workspace/UniVTAC)
# 결과: results/evals/<task>/<method>/ (git 에 commit)   영상·시뮬레이터 로그: <UniVTAC>/data/act_tacenc/eval_raw/ (로컬)
# 로그: <UniVTAC>/data/act_tacenc/logs/eval_queue_<시각>.log, eval_<task>_<method>_shard<i>of<n>_<시각>.log
# 끝나면: python scripts/compare.py → git add results && git commit && git push
# -----------------------------------------------------------------------------
set -o pipefail
JEPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${JEPA_ROOT}/deps.lock"
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
OUT="${ACT_TACENC_OUT:-${ROOT}/data/act_tacenc}"
QUEUE_FILE="${1:-${JEPA_ROOT}/scripts/eval_queue.tsv}"
PROCS="${PROCS:-2}"; read -r -a GPU_LIST <<< "${GPUS:-0}"
LOGD="${OUT}/logs"; mkdir -p "${LOGD}"
export PATH="$(conda info --base)/envs/${UNIVTAC_JEPA_ENV}/bin:${PATH}"
QLOG="${LOGD}/eval_queue_$(date +%Y%m%d-%H%M%S).log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }
N_SEEDS="$(python -c "import json;print(len(json.load(open('${JEPA_ROOT}/results/protocol.json'))['seeds']))")"

done_count() {  # task method -> 지금까지 기록된 seed 수 (error 포함)
    cat "${JEPA_ROOT}/results/evals/$1/$2"/per_seed.shard*of"${PROCS}".csv 2>/dev/null | grep -c '^[0-9]'
}

PIDS=()
trap 'log "중단 (Ctrl-c) — 진행 중 평가 종료 (다시 실행하면 이어서 한다)"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM
log "평가 큐 시작: ${QUEUE_FILE}, PROCS=${PROCS}, GPUS=[${GPU_LIST[*]}], repo=$(git -C "${JEPA_ROOT}" rev-parse --short HEAD)"
while IFS=$'\t' read -r task method _; do
    [[ -z "${task}" || "${task}" == \#* ]] && continue
    n="$(done_count "${task}" "${method}")"
    if (( n >= N_SEEDS )); then log "건너뜀 (끝남 ${n}/${N_SEEDS}): ${task} ${method}"; continue; fi
    if [[ "${method}" == public_* && ! -f "${ROOT}/data/checkpoints/${task}/${method#public_}/policy_last.ckpt" ]]; then
        log "공개 체크포인트 다운로드: ${task}"
        (cd "${ROOT}" && bash data/download.sh --checkpoint "${task}") >> "${QLOG}" 2>&1 || { log "다운로드 실패: ${task}"; continue; }
    fi
    log "시작 ${task} ${method} (기록 ${n}/${N_SEEDS})"
    PIDS=()
    for ((i = 0; i < PROCS; i++)); do
        gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
        elog="${LOGD}/eval_${task}_${method}_shard${i}of${PROCS}_$(date +%Y%m%d-%H%M%S).log"
        CUDA_VISIBLE_DEVICES="${gpu}" python "${JEPA_ROOT}/scripts/eval_run.py" "${task}" "${method}" \
            --shard "${i}/${PROCS}" --headless > "${elog}" 2>&1 < /dev/null &
        PIDS+=($!)
        sleep 30   # Isaac 앱 기동이 겹치지 않게
    done
    rc=0; for p in "${PIDS[@]}"; do wait "${p}" || rc=$?; done
    n="$(done_count "${task}" "${method}")"
    log "종료 ${task} ${method}: 기록 ${n}/${N_SEEDS} (exit ${rc})"
done < "${QUEUE_FILE}"
log "평가 큐 종료 → python ${JEPA_ROOT}/scripts/compare.py"
