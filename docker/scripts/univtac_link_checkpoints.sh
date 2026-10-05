#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 공개 checkpoint(data/checkpoints/…) 를 UniVTAC 정책 코드가 기대하는 경로에 심볼릭 링크로 연결
#
#   사용 (컨테이너 안, /workspace/UniVTAC 에서):
#     bash data/download.sh --checkpoint [task]      # 먼저 다운로드 (modelscope)
#     univtac_link_checkpoints.sh                    # 내려온 task 전부 연결 (idempotent)
#
# 매핑 근거 (2026-09-22, data/checkpoints/<task>/*/log.log 의 Eval Config 로 확인):
#   data/checkpoints/<task>/univtac/      = ACT, train_config=train_config        (vision + tactile, demo, 50 ep)
#   data/checkpoints/<task>/vision_only/  = ACT, train_config=train_config_vision (vision only,      demo, 50 ep)
#   data/checkpoints/encoder.pth          = 공유 촉각 인코더 = policy/*/train_config*.yml 의 tactile_ckpt 경로
#   ACT/Ablation deploy_policy.py 가 찾는 경로: policy/ACT/act_ckpt/act-<task>/<task_config>-<EP_NUM>/<TRAIN_CONFIG>/
#     (<task_config>=demo, EP_NUM=50 이 기본값; 다른 값은 환경변수 EP_NUM / TRAIN_CONFIG 로 바꾼다)
# -----------------------------------------------------------------------------
set -Eeo pipefail
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"; cd "${ROOT}"
CK="${ROOT}/data/checkpoints"
[[ -d "${CK}" ]] || { echo "!! ${CK} 없음 — 먼저: bash data/download.sh --checkpoint"; exit 1; }

TASK_CONFIG="${TASK_CONFIG:-demo}"; EP_NUM="${EP_NUM:-50}"
n=0
for tdir in "${CK}"/*/; do
    task="$(basename "${tdir}")"
    for pair in univtac:train_config vision_only:train_config_vision; do
        src="${tdir}${pair%%:*}"; cfg="${pair##*:}"
        [[ -f "${src}/policy_last.ckpt" ]] || continue
        dst="policy/ACT/act_ckpt/act-${task}/${TASK_CONFIG}-${EP_NUM}/${cfg}"
        mkdir -p "$(dirname "${dst}")"; ln -sfn "${src}" "${dst}"; n=$((n+1))
        echo "  ${dst}  ->  ${src#${ROOT}/}"
    done
done
if [[ -f "${CK}/encoder.pth" ]]; then
    enc="$(grep -hoE '^tactile_ckpt:\s*\S+' policy/ACT/train_config.yml | awk '{print $2}')"
    mkdir -p "$(dirname "${enc}")"; ln -sfn "${CK}/encoder.pth" "${enc}"
    echo "  ${enc}  ->  data/checkpoints/encoder.pth   (공유 촉각 인코더)"
fi
echo "연결 ${n}개.  평가:  HEADLESS=1 python scripts/eval_policy.py <task> ${TASK_CONFIG} ACT/deploy --total_num 100 --headless"
echo "               vision-only 는  TRAIN_CONFIG=train_config_vision  을 앞에 붙인다"
