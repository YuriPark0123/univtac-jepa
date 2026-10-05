#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ACT_TacEnc 학습 큐 — 컨테이너 안에서 tmux 로 실행 (Claude 세션·터미널을 닫아도 계속 돈다)
#
#   tmux new -s train
#   bash /workspace/UniVTAC/univtac-jepa/scripts/train_queue.sh            # scripts/train_queue.tsv 순서대로
#   bash .../train_queue.sh my_queue.tsv                                    # 다른 목록
#   bash .../train_queue.sh --status                                        # 상태만 출력 (아무것도 실행 안 함)
#   (tmux 에서 나오기: Ctrl-b d,  다시 붙기: tmux attach -t train,  중단: Ctrl-c — 진행 중 학습도 같이 종료)
#
# 동작: GPU 마다 작업자 1개. 각 작업자는 그 GPU 에서 이미 돌고 있는 imitate_episodes.py 가 끝날 때까지 기다린 뒤
#       목록 위에서부터 "아직 안 된" 항목을 하나씩 가져가 학습한다. 건너뛰는 것:
#         done    : policy_last.ckpt 가 있음      running : 다른 프로세스가 같은 체크포인트 폴더로 학습 중
#         failed  : 이전에 실패 (.queue_failed — 다시 하려면 그 파일을 지울 것)
#       데이터가 없으면 학습 전에 다운로드(data/download.sh)·전처리(process_data.py)를 한다 (두 작업자가 겹치지 않게 잠금).
# 환경변수: GPUS("0 1")  EP_NUM(50)  DATA_VERSION(isaac45)  UNIVTAC_ROOT(/workspace/UniVTAC)
# 로그: <UniVTAC>/data/act_tacenc/logs/  (queue_<시각>.log = 큐 진행, <task>_<config>_seed0_<시각>.log = 학습별)
# -----------------------------------------------------------------------------
set -Eo pipefail
JEPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${JEPA_ROOT}/deps.lock"
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
OUT="${ACT_TACENC_OUT:-${ROOT}/data/act_tacenc}"
GPUS="${GPUS:-0 1}"
EP_NUM="${EP_NUM:-50}"
DATA_VERSION="${DATA_VERSION:-isaac45}"
POL="${ROOT}/policy/ACT_TacEnc"
LOGD="${OUT}/logs"; CLAIMS="${OUT}/.queue_claims"
mkdir -p "${LOGD}" "${CLAIMS}"
export PATH="$(conda info --base)/envs/${UNIVTAC_JEPA_ENV}/bin:${PATH}"

STATUS_ONLY=0; QUEUE="${JEPA_ROOT}/scripts/train_queue.tsv"
case "${1:-}" in --status) STATUS_ONLY=1 ;; "") ;; *) QUEUE="$1" ;; esac
[[ -f "${QUEUE}" ]] || { echo "!! 큐 파일 없음: ${QUEUE}"; exit 1; }

QLOG="${LOGD}/queue_$(date +%Y%m%d-%H%M%S).log"
log() { echo "[$(date '+%F %T')] $*" | { if [[ ${STATUS_ONLY} -eq 1 ]]; then cat; else tee -a "${QLOG}"; fi; }; }
items() { grep -vE '^\s*(#|$)' "${QUEUE}" | awk -F'\t' '{print $1, $2}'; }
ckpt_of() { echo "${OUT}/act_ckpt/act-$1/${DATA_VERSION}-${EP_NUM}/$2"; }

state_of() {  # task cfg -> done | running | failed | claimed | todo
    local ckpt; ckpt="$(ckpt_of "$1" "$2")"
    if [[ -f "${ckpt}/policy_last.ckpt" ]]; then echo done
    elif pgrep -f -- "--ckpt_dir ${ckpt} " >/dev/null; then echo running
    elif [[ -f "${ckpt}/.queue_failed" ]]; then echo failed
    elif [[ -e "${CLAIMS}/$1__$2" ]]; then echo claimed
    else echo todo; fi
}

gpu_busy() {  # 이 GPU 를 쓰는 imitate_episodes.py 가 있는가 (CUDA_VISIBLE_DEVICES 로 판별)
    local pid
    for pid in $(pgrep -f imitate_episodes.py); do
        tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0
    done
    return 1
}

prepare() {  # task -> 원본 다운로드 + 전처리 (전역 잠금 안에서)
    local task="$1" raw n key
    raw="${ROOT}/data/${DATA_VERSION}/${task}/hdf5"
    n="$(ls "${raw}"/*.hdf5 2>/dev/null | wc -l)"
    if (( n < EP_NUM )); then
        log "  ${task}: 원본 ${n}개 < ${EP_NUM} → 다운로드 (data/download.sh --task ${task} --version ${DATA_VERSION#isaac})"
        (cd "${ROOT}" && bash data/download.sh --task "${task}" --version "${DATA_VERSION#isaac}") >> "${QLOG}" 2>&1 || return 1
        n="$(ls "${raw}"/*.hdf5 2>/dev/null | wc -l)"
        (( n >= EP_NUM )) || { log "  ${task}: 다운로드 후에도 ${n}개"; return 1; }
    fi
    key="sim-${task}-${DATA_VERSION}-${EP_NUM}"
    if ! python -c "import json,sys; sys.exit(0 if '${key}' in json.load(open('${OUT}/SIM_TASK_CONFIGS.json')) else 1)" 2>/dev/null \
       || (( $(ls "${OUT}/sim-${task}/${DATA_VERSION}-${EP_NUM}"/episode_*.hdf5 2>/dev/null | wc -l) < EP_NUM )); then
        log "  ${task}: 전처리 (${EP_NUM} 에피소드)"
        python "${POL}/process_data.py" "${task}" "${DATA_VERSION}" "${EP_NUM}" >> "${QLOG}" 2>&1 || return 1
    fi
}

claim_next() {  # 잠금 안에서 다음 todo 항목을 찜하고 "task cfg" 출력
    (
        flock 9
        while read -r task cfg; do
            if [[ "$(state_of "${task}" "${cfg}")" == todo ]]; then
                touch "${CLAIMS}/${task}__${cfg}"; echo "${task} ${cfg}"; break
            fi
        done < <(items)
    ) 9> "${OUT}/.queue.lock"
}

worker() {
    local gpu="$1" task cfg ckpt tlog rc git_rev
    if gpu_busy "${gpu}"; then
        log "GPU${gpu}: 이미 학습 중인 프로세스가 끝날 때까지 대기"
        while gpu_busy "${gpu}"; do sleep 60; done
    fi
    while true; do
        read -r task cfg <<< "$(claim_next)"
        [[ -z "${task}" ]] && { log "GPU${gpu}: 남은 항목 없음 → 종료"; return 0; }
        ckpt="$(ckpt_of "${task}" "${cfg}")"
        if ! ( flock 8; prepare "${task}" ) 8> "${OUT}/.prep.lock"; then
            log "GPU${gpu}: ${task} 데이터 준비 실패 → ${cfg} 건너뜀"
            mkdir -p "${ckpt}"; touch "${ckpt}/.queue_failed"; rm -f "${CLAIMS}/${task}__${cfg}"; continue
        fi
        tlog="${LOGD}/${task}_${cfg}_seed0_$(date +%Y%m%d-%H%M%S).log"
        git_rev="$(git -C "${JEPA_ROOT}" rev-parse --short HEAD 2>/dev/null)$(git -C "${JEPA_ROOT}" diff --quiet HEAD 2>/dev/null || echo '+dirty')"
        log "GPU${gpu}: 시작 ${task} ${cfg}  → $(basename "${tlog}")"
        (
            echo "repo=univtac-jepa@${git_rev} task=${task} data=${DATA_VERSION}-${EP_NUM} cfg=${cfg} seed=0 gpu=${gpu} start=$(date '+%F %T')"
            bash "${POL}/train.sh" "${task}" "${DATA_VERSION}" "${EP_NUM}" 0 "${gpu}" "${cfg}"
            rc=$?
            echo "exit=${rc} end=$(date '+%F %T')"
            exit ${rc}
        ) > "${tlog}" 2>&1 < /dev/null
        rc=$?
        if [[ ${rc} -eq 0 && -f "${ckpt}/policy_last.ckpt" ]]; then
            log "GPU${gpu}: 완료 ${task} ${cfg}"
        else
            log "GPU${gpu}: 실패 ${task} ${cfg} (exit ${rc}) — 로그 확인: ${tlog}"
            mkdir -p "${ckpt}"; touch "${ckpt}/.queue_failed"
        fi
        rm -f "${CLAIMS}/${task}__${cfg}"
    done
}

print_status() {
    printf '%-22s %-32s %s\n' task train_config state
    while read -r task cfg; do printf '%-22s %-32s %s\n' "${task}" "${cfg}" "$(state_of "${task}" "${cfg}")"; done < <(items)
}

if [[ ${STATUS_ONLY} -eq 1 ]]; then print_status; exit 0; fi

# 큐는 한 번에 하나만 (두 번 띄우면 같은 항목을 두 번 학습할 수 있음)
exec 7> "${OUT}/.queue_instance.lock"
flock -n 7 || { echo "!! 다른 train_queue.sh 가 이미 실행 중입니다 (--status 로 확인)"; exit 1; }
rm -f "${CLAIMS}"/*

log "큐 시작: ${QUEUE}  GPUS=[${GPUS}]  data=${DATA_VERSION}-${EP_NUM}  repo=$(git -C "${JEPA_ROOT}" rev-parse --short HEAD 2>/dev/null)"
log "큐 로그: ${QLOG}"
print_status | tee -a "${QLOG}"
trap 'log "중단 (Ctrl-c) — 진행 중 학습 종료"; kill 0' INT TERM
for g in ${GPUS}; do worker "${g}" & sleep 2; done
wait
log "큐 종료"
print_status | tee -a "${QLOG}"
