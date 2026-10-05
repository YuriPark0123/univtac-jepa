#!/bin/bash
# -----------------------------------------------------------------------------
# univtac-jepa 환경을 한 번에 만드는 호스트 스크립트
#
#   git clone https://github.com/YuriPark0123/univtac-jepa.git <로컬디스크>/univtac/univtac-jepa
#   bash <로컬디스크>/univtac/univtac-jepa/setup_host.sh          # 전체 (처음 1~2시간, 이미 됐으면 수 분)
#   bash .../setup_host.sh --check                                 # 실행 중인 컨테이너에서 상태만 검사
#
# 순서: docker/univtac_setup_host.sh (UniVTAC clone@deps.lock → 이미지 → 컨테이너 → 기본 설치 → 검증)
#       → 컨테이너 안 scripts/univtac_install_jepa.sh (sparsh, UniVTAC-jepa env, 가중치, 코드 연결, 테스트)
# 위치: 이 repo 와 UniVTAC 리포는 로컬 디스크에 둔다 (NAS/CIFS 불가 — 심볼릭 링크·빌드).
#   UNIVTAC_LOCAL  UniVTAC 리포의 상위 폴더 (기본: 이 repo 의 상위 폴더 → <상위>/UniVTAC)
#   ISAACLAB_LOCAL conda env·캐시 상위 폴더 (기본: $HOME/.isaaclab_vnc)
#   CONTAINER_NAME 컨테이너 이름 (기본: univtac)  ★ 같은 이름의 컨테이너는 지우고 다시 만든다
#   TEST_GPU       테스트에 쓸 GPU 번호 (기본: 0)
# -----------------------------------------------------------------------------
set -Eeo pipefail
JEPA_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${JEPA_SRC}/deps.lock"
export UNIVTAC_LOCAL="${UNIVTAC_LOCAL:-$(dirname "${JEPA_SRC}")}"
export UNIVTAC_COMMIT
export UNIVTAC_JEPA_SRC="${JEPA_SRC}"
CONTAINER="${CONTAINER_NAME:-univtac}"
JEPA_IN="/workspace/univtac_jepa/scripts/univtac_install_jepa.sh"

case "${JEPA_SRC}/" in /mnt/nas/*|/mnt/cifs/*)
    echo "!! 이 repo 가 NAS(CIFS) 경로에 있습니다. 로컬 디스크에 clone 하세요."; exit 1;; esac

if [[ "${1:-}" == "--check" ]]; then
    docker exec "${CONTAINER}" bash -c 'univtac_install_main.sh --check 2>&1 | grep -E "^\[|!!"'
    docker exec "${CONTAINER}" bash "${JEPA_IN}" --check
    exit 0
fi

echo "[univtac-jepa] repo         : ${JEPA_SRC}  (git $(git -C "${JEPA_SRC}" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[univtac-jepa] UniVTAC 리포 : ${UNIVTAC_LOCAL}/UniVTAC @ ${UNIVTAC_COMMIT:0:7}"
echo "[univtac-jepa] 컨테이너     : ${CONTAINER} (같은 이름이 있으면 다시 만든다)"

# 1) 기본 UniVTAC 환경 (기존 절차 그대로; UNIVTAC_JEPA_SRC 가 run_container 에서 repo 를 마운트한다)
bash "${JEPA_SRC}/docker/univtac_setup_host.sh"

# 2) univtac-jepa 추가분 (컨테이너 안)
LOG="${UNIVTAC_LOCAL}/UniVTAC/log/install_jepa.log"
echo; echo "[univtac-jepa] JEPA 설치 + 테스트. 로그: ${LOG}"
docker exec -e TEST_GPU="${TEST_GPU:-0}" "${CONTAINER}" bash -o pipefail -c "bash ${JEPA_IN} 2>&1 | tee /workspace/UniVTAC/log/install_jepa.log"

cat <<MSG

완료.  접속:   docker exec -it ${CONTAINER} bash
       학습:   conda activate ${UNIVTAC_JEPA_ENV} && cd /workspace/UniVTAC
               bash data/download.sh --task insert_hole --version 45
               bash policy/ACT_TacEnc/process_data.sh insert_hole isaac45 50
               bash policy/ACT_TacEnc/train.sh insert_hole isaac45 50 0 0 train_config_sparsh_ijepa
       상태:   bash ${JEPA_SRC}/setup_host.sh --check
       설명:   ${JEPA_SRC}/README.md
MSG
