#!/usr/bin/env bash
# univtac-jepa 테스트 (컨테이너 안). 실측 ~3분, GPU 메모리 ~8GB. 출력은 임시 폴더에만 쓰고 지운다.
#   1) test_encoders.py : Sparsh 전처리 == 공식 repo, encoder 6종 strict 로드, original == encoder.pth
#   2) 파이프라인       : isaac45 grasp_classify 2 에피소드 전처리 → ACT_TacEnc(sparsh_ijepa) 학습 1 step
#   3) test_deploy.py   : 학습된 체크포인트를 deploy 경로로 strict 재로드, 촉각 프레임 기록 간격, action shape
# 환경변수: UNIVTAC_ROOT(/workspace/UniVTAC)  TEST_GPU(0)  KEEP_TEST_OUT(1 이면 임시 폴더 유지)
set -Eeo pipefail
JEPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${JEPA_ROOT}/deps.lock"
export UNIVTAC_ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
export CUDA_VISIBLE_DEVICES="${TEST_GPU:-0}"
PY="$(conda info --base)/envs/${UNIVTAC_JEPA_ENV}/bin/python"
POL="${UNIVTAC_ROOT}/policy/ACT_TacEnc"
CFG=train_config_sparsh_ijepa

OUT="$(mktemp -d /tmp/univtac_jepa_test.XXXXXX)"
export ACT_TACENC_OUT="${OUT}"
[[ "${KEEP_TEST_OUT:-0}" == 1 ]] || trap 'rm -rf "${OUT}"' EXIT
quiet() { "$@" > "${OUT}/last.log" 2>&1 || { tail -30 "${OUT}/last.log"; return 1; }; }

echo "[test 1/3] encoders"
"${PY}" "${JEPA_ROOT}/tests/test_encoders.py"

echo "[test 2/3] process 2 episodes + train ${CFG} for 1 step"
quiet "${PY}" "${POL}/process_data.py" grasp_classify isaac45 2
sed -e 's/^num_steps: .*/num_steps: 1/' "${POL}/${CFG}.yml" > "${OUT}/${CFG}.yml"
quiet "${PY}" "${POL}/imitate_episodes.py" --task_name sim-grasp_classify-isaac45-2 \
    --ckpt_dir "${OUT}/act_ckpt/act-grasp_classify/isaac45-2/${CFG}" --config_path "${OUT}/${CFG}.yml" --seed 0
[[ -f "${OUT}/act_ckpt/act-grasp_classify/isaac45-2/${CFG}/policy_last.ckpt" ]]
echo "[ok] checkpoint written"

echo "[test 3/3] deploy reload"
TRAIN_CONFIG="${CFG}" EP_NUM=2 DATA_VERSION=isaac45 "${PY}" "${JEPA_ROOT}/tests/test_deploy.py"
echo "all tests passed"
