#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# univtac-jepa 추가 설치 — 컨테이너 안에서 실행 (기본 UniVTAC 설치 = univtac_install_main.sh 가 끝난 뒤)
#
#   bash scripts/univtac_install_jepa.sh            # 설치 (이미 된 단계는 건너뜀) + 테스트
#   bash scripts/univtac_install_jepa.sh --no-test  # 설치만
#   bash scripts/univtac_install_jepa.sh --check    # 변경 없이 상태만 검사 (GPU 안 씀)
#
# 하는 일 (전부 로컬 디스크, 코드 repo 에는 아무것도 쓰지 않는다):
#   [1] UniVTAC 리포/기본 env 확인        [2] third_party/sparsh @ SPARSH_COMMIT
#   [3] conda env UniVTAC-jepa = UniVTAC 복제 + requirements-jepa.txt
#   [4] assets.tsv 의 큰 파일 다운로드 + sha256 검증 (Sparsh 가중치, encoder.pth, 테스트용 에피소드 2개)
#   [5] 코드 연결: <UniVTAC>/policy/{ACT_TacEnc,tactile_encoders} -> 이 repo   [6] import 검사   [7] 테스트
# 환경변수: UNIVTAC_ROOT(/workspace/UniVTAC)  TEST_GPU(0)
# -----------------------------------------------------------------------------
set -Eeo pipefail

JEPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${JEPA_ROOT}/deps.lock"
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
MODE=install; RUN_TEST=1
case "${1:-}" in
    --check) MODE=check; RUN_TEST=0 ;;
    --no-test) RUN_TEST=0 ;;
    "") ;;
    *) echo "사용법: $0 [--check|--no-test]"; exit 2 ;;
esac
trap 'echo "!! 실패: ${BASH_SOURCE[0]}:${LINENO} — 위 로그를 확인하세요"' ERR

CONDA_BASE="$(conda info --base)"
BASE_PY="${CONDA_BASE}/envs/${UNIVTAC_BASE_ENV}/bin/python"
JEPA_PY="${CONDA_BASE}/envs/${UNIVTAC_JEPA_ENV}/bin/python"
FAILED=0
step() { echo; echo "================ $* ================"; }
ok() { echo "[ok] $*"; }
bad() { echo "[$1] ${*:2}"; FAILED=$((FAILED + 1)); }
need_install() { [[ "${MODE}" == "install" ]]; }

# ---------------------------------------------------------------- [1] base
step "[1/7] 기본 UniVTAC 설치 확인"
[[ -f "${ROOT}/scripts/install.sh" && -d "${ROOT}/third_party/TacEx" ]] || { echo "!! ${ROOT} 가 UniVTAC 리포가 아닙니다"; exit 1; }
cur="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${cur}" == "${UNIVTAC_COMMIT}" ]] && ok "UniVTAC @ ${cur:0:7}" || bad wrong "UniVTAC @ ${cur:0:7} (기대 ${UNIVTAC_COMMIT:0:7})"
[[ -x "${BASE_PY}" ]] || { echo "!! 기본 env '${UNIVTAC_BASE_ENV}' 가 없습니다 — 먼저 univtac_install_main.sh"; exit 1; }
ok "기본 env ${UNIVTAC_BASE_ENV}"

# ---------------------------------------------------------------- [2] sparsh
step "[2/7] third_party/sparsh @ ${SPARSH_COMMIT:0:7}"
SP="${ROOT}/third_party/sparsh"
if [[ ! -d "${SP}/.git" ]]; then
    if need_install; then
        git init -q "${SP}" && git -C "${SP}" remote add origin "${SPARSH_REPO}"
    else
        bad missing "sparsh 없음"
    fi
fi
if [[ -d "${SP}/.git" ]]; then
    have="$(git -C "${SP}" rev-parse HEAD 2>/dev/null || echo none)"
    if [[ "${have}" != "${SPARSH_COMMIT}" ]] && need_install; then
        git -C "${SP}" fetch -q --depth 1 origin "${SPARSH_COMMIT}"
        git -C "${SP}" checkout -q "${SPARSH_COMMIT}"
        have="$(git -C "${SP}" rev-parse HEAD)"
    fi
    [[ "${have}" == "${SPARSH_COMMIT}" ]] && ok "sparsh @ ${have:0:7}" || bad wrong "sparsh @ ${have:0:7}"
fi

# ---------------------------------------------------------------- [3] env
step "[3/7] conda env ${UNIVTAC_JEPA_ENV}"
if [[ ! -x "${JEPA_PY}" ]]; then
    if need_install; then
        echo "기본 env 복제 (실측 ~2분, 디스크 ~24GB)"
        conda create -y -q --clone "${UNIVTAC_BASE_ENV}" -n "${UNIVTAC_JEPA_ENV}"
    else
        bad missing "env 없음"
    fi
fi
if [[ -x "${JEPA_PY}" ]]; then
    freeze="$("${JEPA_PY}" -m pip freeze 2>/dev/null)"
    missing_pins="$(grep -vE '^\s*(#|$)' "${JEPA_ROOT}/requirements-jepa.txt" | while read -r pin; do grep -qxiF "${pin}" <<< "${freeze}" || echo "${pin}"; done)"
    if [[ -n "${missing_pins}" ]] && need_install; then
        "${JEPA_PY}" -m pip install -q --root-user-action=ignore -r "${JEPA_ROOT}/requirements-jepa.txt"
        freeze="$("${JEPA_PY}" -m pip freeze 2>/dev/null)"
        missing_pins="$(grep -vE '^\s*(#|$)' "${JEPA_ROOT}/requirements-jepa.txt" | while read -r pin; do grep -qxiF "${pin}" <<< "${freeze}" || echo "${pin}"; done)"
    fi
    [[ -z "${missing_pins}" ]] && ok "requirements-jepa.txt 12개 일치" || bad wrong "버전 불일치: $(echo ${missing_pins})"
    t_base="$("${BASE_PY}" -c 'import torch;print(torch.__version__)')"; t_jepa="$("${JEPA_PY}" -c 'import torch;print(torch.__version__)')"
    [[ "${t_base}" == "${t_jepa}" ]] && ok "torch ${t_jepa} (기본 env 와 동일)" || bad wrong "torch ${t_jepa} ≠ 기본 env ${t_base}"
fi

# ---------------------------------------------------------------- [4] assets
step "[4/7] 큰 파일 (assets.tsv → ${ROOT}/data, sha256 검증)"
while IFS=$'\t' read -r path sha src repo file; do
    [[ -z "${path}" || "${path}" == \#* ]] && continue
    dst="${ROOT}/data/${path}"
    if [[ ! -f "${dst}" ]] || [[ "$(sha256sum "${dst}" | cut -d' ' -f1)" != "${sha}" ]]; then
        if need_install; then
            echo "다운로드: ${path}"
            mkdir -p "$(dirname "${dst}")"
            if [[ "${src}" == hf ]]; then
                "${JEPA_PY}" -c "from huggingface_hub import hf_hub_download as d; d('${repo}', '${file}', local_dir='$(dirname "${dst}")')"
            else
                "${CONDA_BASE}/envs/${UNIVTAC_BASE_ENV}/bin/modelscope" download "${repo}" --repo-type dataset \
                    --include "${file}" --local-dir "${ROOT}/data" >/dev/null
            fi
        else
            bad missing "${path}"; continue
        fi
    fi
    [[ "$(sha256sum "${dst}" | cut -d' ' -f1)" == "${sha}" ]] && ok "${path}" || bad wrong "${path} sha256 불일치"
done < "${JEPA_ROOT}/assets.tsv"

# ---------------------------------------------------------------- [5] link code
step "[5/7] 코드 연결: ${ROOT}/policy -> ${JEPA_ROOT}/policy"
EXCLUDE="${ROOT}/.git/info/exclude"
for m in ACT_TacEnc tactile_encoders; do
    link="${ROOT}/policy/${m}"; target="${JEPA_ROOT}/policy/${m}"
    if [[ -e "${link}" && ! -L "${link}" ]]; then
        bad wrong "${link} 가 실제 폴더입니다 (repo 로 옮기기 전 사본?) — 확인 후 지우거나 옮기세요"; continue
    fi
    if [[ "$(readlink "${link}" 2>/dev/null)" != "${target}" ]]; then
        need_install && ln -sfn "${target}" "${link}" || { bad missing "${link}"; continue; }
    fi
    ok "${link} -> ${target}"
    # UniVTAC 원본 git 상태를 깨끗하게 유지 (tracked 파일은 건드리지 않음)
    need_install && { grep -qx "/policy/${m}" "${EXCLUDE}" 2>/dev/null || echo "/policy/${m}" >> "${EXCLUDE}"; }
done
if need_install; then
    case "${JEPA_ROOT}/" in "${ROOT}/"*) rel="/${JEPA_ROOT#${ROOT}/}/"; grep -qx "${rel}" "${EXCLUDE}" || echo "${rel}" >> "${EXCLUDE}";; esac
    mkdir -p "${ROOT}/data/act_tacenc"
fi

# ---------------------------------------------------------------- [6] import
step "[6/7] import 검사 (${UNIVTAC_JEPA_ENV}, CPU)"
if [[ -x "${JEPA_PY}" ]]; then
    if UNIVTAC_ROOT="${ROOT}" "${JEPA_PY}" - <<EOF
import sys; sys.path[:0] = ["${ROOT}", "${ROOT}/policy", "${SP}"]
import lightning, safetensors, tactile_encoders
from tactile_encoders import ENCODERS
from tactile_ssl.model.vision_transformer import vit_base
print("encoders:", ", ".join(ENCODERS))
EOF
    then ok "tactile_encoders / sparsh / lightning import"; else bad wrong "import 실패"; fi
fi

# ---------------------------------------------------------------- [7] tests
step "[7/7] 테스트 (GPU ${TEST_GPU:-0}, ~3분, GPU 메모리 ~8GB)"
if [[ ${RUN_TEST} -eq 1 && ${FAILED} -eq 0 ]]; then
    if UNIVTAC_ROOT="${ROOT}" bash "${JEPA_ROOT}/tests/run_tests.sh"; then ok "tests"; else bad wrong "tests 실패"; fi
else
    echo "(건너뜀: $([[ ${RUN_TEST} -eq 0 ]] && echo "--check/--no-test" || echo "앞 단계 실패"))"
fi

echo
if [[ ${FAILED} -eq 0 ]]; then echo "univtac-jepa: 전부 ok"; else echo "!! univtac-jepa: 문제 ${FAILED}건"; exit 1; fi
