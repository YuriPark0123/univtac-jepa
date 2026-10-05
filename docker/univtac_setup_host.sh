#!/bin/bash
# -----------------------------------------------------------------------------
# UniVTAC (main = Isaac Sim 4.5.0, 논문 환경) 컨테이너를 "처음부터 끝까지" 한 번에 재현하는 호스트 스크립트
#
#   bash univtac_setup_host.sh            # clone → 이미지 빌드 → 컨테이너 기동 → 환경 설치(1~2h) → 검증
#   bash univtac_setup_host.sh --no-install   # 설치 단계는 빼고 컨테이너만 띄움
#
# 이미 설치된 서버에서 다시 실행해도 안전하다 (모든 단계가 idempotent):
#   - 리포가 있으면 clone 생략, 커밋만 확인
#   - 이미지는 캐시로 수 초
#   - 컨테이너는 지우고 다시 만들지만, conda env·리포·빌드 산출물·출력은 전부 host 마운트라 그대로 남는다
#   - 설치 스크립트는 이미 된 단계를 건너뛴다 (--check 만 도는 수준, ~1분)
#
# 이 스크립트가 있는 폴더(로컬: …/isaaclab_vnc/univtac/docker, 또는 NAS 사본)의 Dockerfile.univtac / run_container_univtac.sh / scripts/ 를 사용한다.
# -----------------------------------------------------------------------------
set -Eeo pipefail
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UNIVTAC_LOCAL="${UNIVTAC_LOCAL:-/mnt/data/yuri_docker/data/isaaclab_vnc/univtac}"
UNIVTAC_SRC="${UNIVTAC_SRC:-${UNIVTAC_LOCAL}/UniVTAC}"
UNIVTAC_COMMIT="${UNIVTAC_COMMIT:-0dafa10262e22f486f160a55d6f11aeab12d8e7b}"   # main, 2026-09-07 "fix readme" — 스펙 문서와 동일
IMAGE="${IMAGE:-univtac:isaac45}"
CONTAINER="${CONTAINER_NAME:-univtac}"
DO_INSTALL=1; [[ "${1:-}" == "--no-install" ]] && DO_INSTALL=0

log() { echo; echo "[univtac-setup] $*"; }

# ── 전제조건 (다른 서버에서 처음 돌릴 때 여기서 걸리면 그것부터 설치) ──────────
for c in docker git; do command -v "$c" >/dev/null || { echo "!! '$c' 가 필요합니다"; exit 1; }; done
docker info >/dev/null 2>&1 || { echo "!! docker 데몬에 접근 불가 (docker 그룹 권한 확인: sudo usermod -aG docker \$USER)"; exit 1; }
# `docker info` 에 nvidia 런타임이 안 보여도 toolkit 이 있으면 --gpus 가 동작한다 (이 서버가 그 경우) → 바이너리 존재로 검사
command -v nvidia-container-cli >/dev/null || command -v nvidia-ctk >/dev/null || command -v nvidia-container-runtime >/dev/null \
  || { echo "!! nvidia-container-toolkit 이 없습니다 (apt install nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker && systemctl restart docker)"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || { echo "!! NVIDIA 드라이버/GPU 를 찾을 수 없습니다 (driver ≥ 535, 검증은 550.107)"; exit 1; }
case "${UNIVTAC_LOCAL}" in /mnt/nas/*|/mnt/cifs/*) echo "!! UNIVTAC_LOCAL 이 NAS(CIFS) 경로입니다 — 심볼릭 링크/빌드 불가. 로컬 디스크로 지정하세요: UNIVTAC_LOCAL=/data/univtac bash $0"; exit 1;; esac

# ── 0) 로컬 작업 폴더 (root 소유 상위 폴더라면 docker 로 만들고 내 uid 로 넘긴다) ──
if [ ! -d "${UNIVTAC_LOCAL}" ]; then
  log "로컬 폴더 생성: ${UNIVTAC_LOCAL}"
  if ! mkdir -p "${UNIVTAC_LOCAL}" 2>/dev/null; then
    parent="$(dirname "${UNIVTAC_LOCAL}")"
    docker run --rm -v "${parent}":/w ubuntu:22.04 bash -c "mkdir -p /w/$(basename "${UNIVTAC_LOCAL}") && chown $(id -u):$(id -g) /w/$(basename "${UNIVTAC_LOCAL}")"
  fi
fi

# 출력 폴더는 리포 안에 둔다 (코드가 ./data, ./eval_result, ./log 상대경로를 사용)

# ── 1) UniVTAC clone (로컬 디스크 필수 — NAS/CIFS 는 심볼릭 링크·빌드 불가) ─────
if [ ! -f "${UNIVTAC_SRC}/scripts/install.sh" ]; then
  log "clone UniVTAC → ${UNIVTAC_SRC}"
  git clone https://github.com/univtac/UniVTAC.git "${UNIVTAC_SRC}"
fi
cur="$(git -C "${UNIVTAC_SRC}" rev-parse HEAD)"
if [ "${cur}" != "${UNIVTAC_COMMIT}" ]; then
  log "커밋 고정: ${cur:0:7} → ${UNIVTAC_COMMIT:0:7} (main)"
  git -C "${UNIVTAC_SRC}" fetch -q origin main isaac51
  git -C "${UNIVTAC_SRC}" checkout -q -B main "${UNIVTAC_COMMIT}"
fi
# 에셋 복원(7b)에 isaac51 브랜치 blob 이 필요하다
git -C "${UNIVTAC_SRC}" rev-parse --verify -q origin/isaac51 >/dev/null || git -C "${UNIVTAC_SRC}" fetch -q origin isaac51:refs/remotes/origin/isaac51
echo "UniVTAC @ $(git -C "${UNIVTAC_SRC}" rev-parse --short HEAD) ($(git -C "${UNIVTAC_SRC}" rev-parse --abbrev-ref HEAD))"

# ── 2) 이미지 빌드 ──────────────────────────────────────────────────────────────
log "docker build ${IMAGE}"
docker build -f "${DATA_DIR}/Dockerfile.univtac" -t "${IMAGE}" "${DATA_DIR}"

# ── 3) 컨테이너 기동 (백그라운드) ───────────────────────────────────────────────
log "컨테이너 기동: ${CONTAINER}"
DETACH=1 IMAGE="${IMAGE}" CONTAINER_NAME="${CONTAINER}" UNIVTAC_SRC="${UNIVTAC_SRC}" UNIVTAC_LOCAL="${UNIVTAC_LOCAL}" \
  bash "${DATA_DIR}/run_container_univtac.sh"

# ── 4) 환경 설치 + 검증 (컨테이너 안) ──────────────────────────────────────────
if [ "${DO_INSTALL}" = 1 ]; then
  mkdir -p "${UNIVTAC_SRC}"/{data,eval_result,log}
  log "환경 설치 (처음이면 1~2시간, 이미 됐으면 ~1분). 로그: ${UNIVTAC_SRC}/log/install_isaac45.log"
  docker exec "${CONTAINER}" bash -c 'cd /workspace/UniVTAC && mkdir -p log && univtac_install_main.sh 2>&1 | tee log/install_isaac45.log | grep -E "^={5,}|^\[univtac-install|^\[(ok|wrong|missing)\]|^\[toolchain\]|!!|설치 완료"'
  log "검증"
  docker exec "${CONTAINER}" bash -c 'univtac_install_main.sh --check 2>&1 | grep -E "^\[|!!"'
fi

cat <<MSG

완료.  접속:        docker exec -it ${CONTAINER} bash        (안에서: conda activate UniVTAC)
       헤드리스 수집: HEADLESS=1 ENABLE_CAMERAS=1 python scripts/collect_data.py grasp_classify demo --start_seed 0 --max_seed 0 --episode_num 1 --gpu 0
       정책 평가:    HEADLESS=1 python scripts/eval_policy.py grasp_classify demo ACT/deploy --total_num 10 --headless   (먼저 bash data/download.sh --checkpoint grasp_classify && univtac_link_checkpoints.sh)
       GUI(VNC):     start_vnc.sh  → SSH 터널로 5901 접속 후 bash collect_data.sh grasp_classify demo 0
수집·데이터셋/checkpoint 다운로드·학습·평가 전체 절차: ${DATA_DIR}/README_univtac.md
MSG
