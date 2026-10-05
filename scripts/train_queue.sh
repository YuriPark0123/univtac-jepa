#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ACT_TacEnc 학습 큐 — scripts/train_queue.tsv 를 위에서부터 순서대로 학습한다. tmux 안에서 실행:
#
#   tmux new -s train
#   bash /workspace/UniVTAC/univtac-jepa/scripts/train_queue.sh            # (다른 목록: 첫 인자로 tsv 경로)
#   (빠져나오기: Ctrl-b d,  다시 붙기: tmux attach -t train,  중단: Ctrl-c — 진행 중 학습도 같이 종료)
#
# 규칙 (단순하게):
#   - 시작할 때 체크포인트 폴더가 이미 있는 항목은 건너뛴다 (끝났거나 다른 곳에서 학습 중). 다시 하려면 그 폴더를 지운다.
#   - 동시에 최대 MAX_JOBS(2)개. 하나 띄울 때마다 그 순간 여유 메모리가 가장 큰 GPU 를 고른다.
#   - CUDA out of memory 로 실패하면 큐 맨 뒤로 보내 나중에 다시 한다 (최대 OOM_RETRY(2)회). 다른 실패는 기록만 한다.
#   - 데이터가 없으면 학습 전에 다운로드(data/download.sh)·전처리(process_data.py)를 한다.
# 환경변수: MAX_JOBS(2)  OOM_RETRY(2)  LAUNCH_GAP(120초, 학습 하나 띄운 뒤 다음 GPU 선택까지 대기)  EP_NUM(50)  DATA_VERSION(isaac45)  UNIVTAC_ROOT(/workspace/UniVTAC)
# 로그: <UniVTAC>/data/act_tacenc/logs/queue_<시각>.log (큐), <task>_<config>_seed0_<시각>.log (학습별)
# -----------------------------------------------------------------------------
set -o pipefail
JEPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${JEPA_ROOT}/deps.lock"
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
OUT="${ACT_TACENC_OUT:-${ROOT}/data/act_tacenc}"
QUEUE_FILE="${1:-${JEPA_ROOT}/scripts/train_queue.tsv}"
MAX_JOBS="${MAX_JOBS:-2}"; OOM_RETRY="${OOM_RETRY:-2}"
EP_NUM="${EP_NUM:-50}"; DATA_VERSION="${DATA_VERSION:-isaac45}"
POL="${ROOT}/policy/ACT_TacEnc"; LOGD="${OUT}/logs"
mkdir -p "${LOGD}"
export PATH="$(conda info --base)/envs/${UNIVTAC_JEPA_ENV}/bin:${PATH}"
QLOG="${LOGD}/queue_$(date +%Y%m%d-%H%M%S).log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QLOG}"; }
ckpt_of() { echo "${OUT}/act_ckpt/act-$1/${DATA_VERSION}-${EP_NUM}/$2"; }

prepare() {  # task -> 원본 다운로드 + 전처리
    local task="$1" n
    n="$(ls "${ROOT}/data/${DATA_VERSION}/${task}/hdf5/"*.hdf5 2>/dev/null | wc -l)"
    if (( n < EP_NUM )); then
        log "  ${task}: 원본 다운로드 (data/download.sh --task ${task} --version ${DATA_VERSION#isaac})"
        (cd "${ROOT}" && bash data/download.sh --task "${task}" --version "${DATA_VERSION#isaac}") >> "${QLOG}" 2>&1 || return 1
    fi
    if (( $(ls "${OUT}/sim-${task}/${DATA_VERSION}-${EP_NUM}"/episode_*.hdf5 2>/dev/null | wc -l) < EP_NUM )); then
        log "  ${task}: 전처리 (${EP_NUM} 에피소드)"
        python "${POL}/process_data.py" "${task}" "${DATA_VERSION}" "${EP_NUM}" >> "${QLOG}" 2>&1 || return 1
    fi
}

freest_gpu() {  # 여유 메모리가 가장 큰 GPU 번호
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1
}

# 큐 만들기: 체크포인트 폴더가 이미 있는 항목은 제외
QUEUE=()
while IFS=$'\t' read -r task cfg _; do
    [[ -z "${task}" || "${task}" == \#* ]] && continue
    if [[ -d "$(ckpt_of "${task}" "${cfg}")" ]]; then log "건너뜀 (폴더 있음): ${task} ${cfg}"; continue; fi
    QUEUE+=("${task} ${cfg}")
done < "${QUEUE_FILE}"
log "큐 시작: ${#QUEUE[@]}개, 동시 ${MAX_JOBS}개, data=${DATA_VERSION}-${EP_NUM}, repo=$(git -C "${JEPA_ROOT}" rev-parse --short HEAD)"

declare -A JOB_ITEM JOB_LOG JOB_GPU OOMS
trap 'log "중단 (Ctrl-c) — 진행 중 학습 종료"; kill 0' INT TERM
DONE=0; FAILED=0

while (( ${#QUEUE[@]} > 0 || ${#JOB_ITEM[@]} > 0 )); do
    # 끝난 학습 정리
    for pid in "${!JOB_ITEM[@]}"; do
        kill -0 "${pid}" 2>/dev/null && continue
        wait "${pid}"; rc=$?
        read -r task cfg <<< "${JOB_ITEM[${pid}]}"; tlog="${JOB_LOG[${pid}]}"; key="${task}__${cfg}"
        if [[ ${rc} -eq 0 && -f "$(ckpt_of "${task}" "${cfg}")/policy_last.ckpt" ]]; then
            log "완료: ${task} ${cfg} (GPU${JOB_GPU[${pid}]})"; DONE=$((DONE + 1))
        elif grep -qE "CUDA out of memory|OutOfMemoryError" "${tlog}" && (( ${OOMS[${key}]:-0} < OOM_RETRY )); then
            OOMS[${key}]=$(( ${OOMS[${key}]:-0} + 1 ))
            log "OOM: ${task} ${cfg} (GPU${JOB_GPU[${pid}]}) → 큐 맨 뒤로 (재시도 ${OOMS[${key}]}/${OOM_RETRY})"
            QUEUE+=("${task} ${cfg}")
        else
            log "실패: ${task} ${cfg} (exit ${rc}) — ${tlog}"; FAILED=$((FAILED + 1))
        fi
        unset "JOB_ITEM[${pid}]" "JOB_LOG[${pid}]" "JOB_GPU[${pid}]"
    done

    # 자리가 있으면 다음 항목 시작
    if (( ${#JOB_ITEM[@]} < MAX_JOBS && ${#QUEUE[@]} > 0 )); then
        read -r task cfg <<< "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
        if ! prepare "${task}"; then log "실패: ${task} 데이터 준비 — ${cfg} 건너뜀"; FAILED=$((FAILED + 1)); continue; fi
        gpu="$(freest_gpu)"
        tlog="${LOGD}/${task}_${cfg}_seed0_$(date +%Y%m%d-%H%M%S).log"
        (
            echo "repo=univtac-jepa@$(git -C "${JEPA_ROOT}" rev-parse --short HEAD) task=${task} data=${DATA_VERSION}-${EP_NUM} cfg=${cfg} seed=0 gpu=${gpu} start=$(date '+%F %T')"
            bash "${POL}/train.sh" "${task}" "${DATA_VERSION}" "${EP_NUM}" 0 "${gpu}" "${cfg}"
            rc=$?; echo "exit=${rc} end=$(date '+%F %T')"; exit ${rc}
        ) > "${tlog}" 2>&1 < /dev/null &
        JOB_ITEM[$!]="${task} ${cfg}"; JOB_LOG[$!]="${tlog}"; JOB_GPU[$!]="${gpu}"
        log "시작: ${task} ${cfg} → GPU${gpu}  ($(basename "${tlog}"), 남은 큐 ${#QUEUE[@]}개)"
        sleep "${LAUNCH_GAP:-120}"   # 방금 띄운 학습이 메모리를 잡은 뒤에 다음 GPU 를 고르도록
    else
        sleep 30
    fi
done
log "큐 종료: 완료 ${DONE}, 실패 ${FAILED}"
