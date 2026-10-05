#!/bin/bash
# -----------------------------------------------------------------------------
# UniVTAC 벤치마크 컨테이너 실행 스크립트
# 사용법:  bash run_container_univtac.sh
#          CONTAINER_NAME=univtac2 UNIVTAC_LOCAL=/data/univtac bash run_container_univtac.sh
#
# 기존 run_container.sh 의 규칙을 그대로 따른다:
#   ★ 캐시 / tailscale 상태 / conda env 는 "그 서버 로컬" ($HOME/.isaaclab_vnc) 에 둔다
#   ★ 데이터셋·수집 hdf5·평가 영상·로그는 리포 안(data/, eval_result/, log/)에 쌓인다 — 코드가 그 상대경로를 쓴다.
#     따라서 마운트는 리포 1개(+재현 파일 ro 1개)면 충분하다.
#   ★ 대상: UniVTAC main 브랜치 = 논문 환경 (Isaac Sim 4.5.0 / Isaac Lab 2.1.1). 이미지 univtac:isaac45
#   ★ 호스트 경로는 자유롭게 옮겨도 된다 (UNIVTAC_LOCAL / ISAACLAB_LOCAL 로 지정). 빌드 산출물·editable 설치·심볼릭
#     링크에 박힌 경로는 전부 컨테이너 쪽 /workspace/UniVTAC/... 라서, 마운트 대상 경로만 그대로면 깨지지 않는다.
#   ★ NAS(CIFS) 는 심볼릭 링크를 못 만들므로, 빌드가 필요한 UniVTAC 리포 자체는 로컬 디스크에 둔다
#     (IsaacLab/UniVTAC_INSTALL_NOTES.md §0, §3.1 — 리포에 tracked symlink 도 있음).
#   ★ DexVerse / lerobot 출력 / datasets 오버레이는 이 컨테이너에 필요 없어 마운트하지 않는다.
# -----------------------------------------------------------------------------
set -e

# ── 작업 루트 = 이 스크립트가 있는 폴더 (…/univtac/docker) ────────────────────────────────
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 참고용 코드 마운트(선택). 예전에는 "스크립트 폴더 옆의 IsaacLab" 을 자동으로 걸었는데, 파일을 옮기면 조용히 끊기고
# /workspace/isaaclab 이라는 이름 때문에 "Isaac Lab 이 없다" 는 오해를 부른다 → 명시적으로 지정할 때만 마운트한다.
#   예:  ISAACLAB_REF_SRC=/mnt/data/yuri_docker/data/isaaclab_vnc/IsaacLab bash run_container_univtac.sh
# ★ UniVTAC 이 실제로 쓰는 Isaac Lab 은 /workspace/UniVTAC/third_party/IsaacLab (v2.1.1, conda env 에 editable 설치) 이다.
ISAACLAB_SRC="${ISAACLAB_REF_SRC:-}"

# ── 서버 로컬: 컨테이너를 지워도 남아야 하는 것 ─────────────────────────────
LOCAL_DIR="${ISAACLAB_LOCAL:-$HOME/.isaaclab_vnc}"
UV_LOCAL="${LOCAL_DIR}/univtac"              # UniVTAC 전용 (기존 isaaclab 컨테이너 것과 섞이지 않게 분리)
CONDA_ENVS_DIR="${UV_LOCAL}/conda_envs"      # → /root/miniconda3/envs   UniVTAC env (15~25GB)
CONDA_PKGS_DIR="${UV_LOCAL}/conda_pkgs"      # → /root/miniconda3/pkgs   conda 패키지 캐시
PIP_CACHE_DIR="${UV_LOCAL}/pip_cache"        # → /root/.cache/pip        Isaac Sim pip 휠(~10GB) 재다운로드 방지
OV_CACHE_DIR="${UV_LOCAL}/cache/ov"          # → /root/.cache/ov         Kit 셰이더/텍스처 캐시 (GPU/드라이버별)
OV_DATA_DIR="${UV_LOCAL}/cache/ov_data"      # → /root/.local/share/ov   Kit 데이터/확장 캐시
TS_STATE_DIR="${UV_LOCAL}/tailscale"         # → /var/lib/tailscale      기존 컨테이너와 다른 노드여야 하므로 분리
VCPKG_DIR="${UV_LOCAL}/vcpkg"                # → /root/Toolchain/vcpkg   libuipc 빌드용 vcpkg (TacEx 문서 경로)
VSCODE_DIR="${UV_LOCAL}/vscode-server"       # → /root/.vscode-server    VS Code Remote 서버+확장(1.1GB). 컨테이너를 다시 만들 때마다 재다운로드·재색인되는 것을 막는다
HF_CACHE_DIR="${LOCAL_DIR}/hf_cache"         # → /root/.cache/huggingface 기존 컨테이너와 공유 (HF 허브 캐시)

# ── UniVTAC 리포 (로컬 디스크, 빌드 산출물 .cache/ third_party/ 포함) ────────
UNIVTAC_LOCAL="${UNIVTAC_LOCAL:-/mnt/data/yuri_docker/data/isaaclab_vnc/univtac}"
UNIVTAC_SRC="${UNIVTAC_SRC:-${UNIVTAC_LOCAL}/UniVTAC}"

# ── 출력 폴더 (로컬) — 무거운 IO 는 전부 여기로. 리포 안의 원래 자리에 마운트한다 ──
#   data/        : data/download.sh 결과(데이터셋·checkpoint) + 수집 hdf5 (task_config: save_root_dir: data)
#   eval_result/ : 평가 영상·성공률 (task_config: replay_settings.save_root_dir: eval_result)
#   log/         : scripts/*.py 의 ./log
#   (policy/<이름>/ 아래 학습 체크포인트(.ckpt/.pkl)는 리포 안에 남는다 — 리포도 로컬이라 성능 문제는 없음)

# ── 이미지 / 컨테이너 이름 / 빌드 파라미터 ──────────────────────────────────
IMAGE="${IMAGE:-univtac:isaac45}"
CONTAINER="${CONTAINER_NAME:-univtac}"
# GPU 아키텍처(sm)를 nvidia-smi 로 자동 감지 (A6000/3090=86, 4090/L40=89). 실패 시 86.
UNIVTAC_CUDA_ARCH="${UNIVTAC_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')}"
UNIVTAC_CUDA_ARCH="${UNIVTAC_CUDA_ARCH:-86}"
UNIVTAC_BUILD_JOBS="${UNIVTAC_BUILD_JOBS:-16}"   # libuipc/cuRobo 빌드 병렬도 (RAM 부족하면 4)

# ── Tailscale(VPN) 인증키: 로컬 우선, 없으면 작업 루트, 그것도 없으면 수동 로그인 ─
TS_AUTHKEY="$(cat "${LOCAL_DIR}/tailscale.authkey" 2>/dev/null \
           || cat "${DATA_DIR}/tailscale.authkey"  2>/dev/null || true)"

# ── 사전 점검 ────────────────────────────────────────────────────────────────
if [ ! -f "${UNIVTAC_SRC}/scripts/install.sh" ]; then
  echo "[에러] UniVTAC 리포가 ${UNIVTAC_SRC} 에 없습니다. 로컬 디스크에 먼저 clone 하세요:"
  echo "       mkdir -p ${UNIVTAC_LOCAL}"
  echo "       git clone https://github.com/univtac/UniVTAC.git ${UNIVTAC_SRC}"
  echo "       git -C ${UNIVTAC_SRC} checkout main"
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[에러] 이미지 ${IMAGE} 가 없습니다. 먼저 빌드하세요:"
  echo "       cd ${DATA_DIR} && docker build -f Dockerfile.univtac -t ${IMAGE} ."
  exit 1
fi

mkdir -p "${CONDA_ENVS_DIR}" "${CONDA_PKGS_DIR}" "${PIP_CACHE_DIR}" "${OV_CACHE_DIR}" "${OV_DATA_DIR}" \
         "${TS_STATE_DIR}" "${HF_CACHE_DIR}" "${VCPKG_DIR}" "${VSCODE_DIR}"
mkdir -p "${UNIVTAC_SRC}"/{data,eval_result,log}

# ── 참고용 코드 마운트 (선택) ────────────────────────────────────────────────────
ISAACLAB_MOUNT=()
if [ -f "${ISAACLAB_SRC}/isaaclab.sh" ]; then
  ISAACLAB_MOUNT=(-v "${ISAACLAB_SRC}":/workspace/isaaclab_ref:ro)
else
  echo "[경고] ISAACLAB_REF_SRC=${ISAACLAB_SRC} 가 없어 /workspace/isaaclab_ref 마운트를 생략합니다."
fi

# ── univtac-jepa 코드 repo (선택). 최상위 setup_host.sh 가 UNIVTAC_JEPA_SRC 를 넘긴다 ──────────
#   코드만 담긴 git repo 를 /workspace/univtac_jepa 에 rw 로 건다. 무거운 출력은 repo 가 아니라 UniVTAC/data 아래로 간다.
JEPA_MOUNT=()
if [ -n "${UNIVTAC_JEPA_SRC:-}" ]; then
  [ -f "${UNIVTAC_JEPA_SRC}/deps.lock" ] || { echo "[에러] UNIVTAC_JEPA_SRC=${UNIVTAC_JEPA_SRC} 가 univtac-jepa repo 가 아닙니다"; exit 1; }
  JEPA_MOUNT=(-v "${UNIVTAC_JEPA_SRC}":/workspace/univtac_jepa:rw)
fi

# ── GPU 장치 명시 (--gpus all 만 쓰면 host 의 systemd daemon-reload 뒤 NVML Unknown Error 가 남 —
#    run_container.sh 의 TODO / 노트 §3.5. --device 로 넘기면 docker 가 systemd scope 에 등록해 유지된다)
DEV_ARGS=()
for d in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
  [ -e "${d}" ] && DEV_ARGS+=(--device "${d}:${d}")
done

BR="$(git -C "${UNIVTAC_SRC}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
[ "${BR}" = "main" ] || echo "[경고] UniVTAC 브랜치가 '${BR}' 입니다. 이 이미지는 main(Isaac Sim 4.5) 용입니다."
echo "[정보] 이미지     : ${IMAGE}"
echo "[정보] 컨테이너   : ${CONTAINER}"
echo "[정보] UniVTAC    : ${UNIVTAC_SRC}  ->  /workspace/UniVTAC   (branch: $(git -C "${UNIVTAC_SRC}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
echo "[정보] 출력       : ${UNIVTAC_SRC}/{data,eval_result,log}   (리포 안. 코드가 ./data 등 상대경로를 쓰므로 여기여야 한다)"
echo "[정보] conda env  : ${CONDA_ENVS_DIR}  ->  /root/miniconda3/envs"
echo "[정보] 캐시       : ${UV_LOCAL}/{conda_pkgs,pip_cache,cache}   (이 서버 로컬)"
echo "[정보] 재현 파일  : ${DATA_DIR}  ->  /workspace/univtac_docker (읽기전용, README_univtac.md 포함)"
echo "[정보] Isaac Lab  : /workspace/UniVTAC/third_party/IsaacLab  (v2.1.1, conda env UniVTAC 에 editable 설치)"
[ -n "${ISAACLAB_SRC}" ] && [ -d "${ISAACLAB_SRC}" ] && echo "[정보] 참고 코드  : ${ISAACLAB_SRC}  ->  /workspace/isaaclab_ref (ro)"
[ ${#JEPA_MOUNT[@]} -gt 0 ] && echo "[정보] univtac-jepa: ${UNIVTAC_JEPA_SRC}  ->  /workspace/univtac_jepa (rw)"
echo "[정보] GPU        : ${#DEV_ARGS[@]} 장치 인자, CUDA_ARCH=${UNIVTAC_CUDA_ARCH}, BUILD_JOBS=${UNIVTAC_BUILD_JOBS}"

# 기존 동일 이름 컨테이너 제거
#   (conda env·리포·출력·vcpkg 는 전부 host 마운트라, 컨테이너를 다시 만들어도 잃는 것이 없다)
docker rm -f "${CONTAINER}" 2>/dev/null || true

# 실행 모드:
#   기본       → -it 로 셸에 바로 들어간다 (터미널에서 사용)
#   DETACH=1   → 백그라운드로 띄우고 곧바로 반환 (TTY 없는 환경/장시간 설치용).
#                이후 접속:  docker exec -it <이름> bash
RUN_MODE=(-it); TRAILING_CMD=()
if [ -n "${DETACH:-}" ]; then
  RUN_MODE=(-d); TRAILING_CMD=(sleep infinity)
  echo "[정보] DETACH=1 → 백그라운드 기동. 접속:  docker exec -it ${CONTAINER} bash"
fi

# NVIDIA_DISABLE_REQUIRE=1 : cuda 베이스 이미지의 NVIDIA_REQUIRE_CUDA(드라이버 브랜드/버전 매트릭스) 검사를 끈다.
#   12.4 는 driver 550 으로 충족되지만, 브랜드 매칭 실패로 기동이 막히는 일을 피하기 위한 안전장치.
docker run "${RUN_MODE[@]}" --gpus all \
  --name "${CONTAINER}" \
  --shm-size=16g \
  --network=host \
  --cap-add=NET_ADMIN \
  --device /dev/net/tun:/dev/net/tun \
  "${DEV_ARGS[@]}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e TZ="${TZ:-Asia/Seoul}" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e DISPLAY=":${VNC_DISPLAY:-1}" \
  -e VNC_DISPLAY="${VNC_DISPLAY:-1}" \
  -e VNC_PASSWORD=isaaclab \
  -e TS_HOSTNAME="$(hostname)-univtac" \
  -e TS_AUTHKEY="${TS_AUTHKEY}" \
  -e UNIVTAC_CUDA_ARCH="${UNIVTAC_CUDA_ARCH}" \
  -e UNIVTAC_BUILD_JOBS="${UNIVTAC_BUILD_JOBS}" \
  -v "${UNIVTAC_SRC}":/workspace/UniVTAC:rw \
  "${ISAACLAB_MOUNT[@]}" \
  "${JEPA_MOUNT[@]}" \
  -v "${CONDA_ENVS_DIR}":/root/miniconda3/envs:rw \
  -v "${CONDA_PKGS_DIR}":/root/miniconda3/pkgs:rw \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:rw \
  -v "${OV_CACHE_DIR}":/root/.cache/ov:rw \
  -v "${OV_DATA_DIR}":/root/.local/share/ov:rw \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:rw \
  -v "${TS_STATE_DIR}":/var/lib/tailscale:rw \
  -v "${VCPKG_DIR}":/root/Toolchain/vcpkg:rw \
  -v "${VSCODE_DIR}":/root/.vscode-server:rw \
  -v "${DATA_DIR}":/workspace/univtac_docker:ro \
  "${IMAGE}" "${TRAILING_CMD[@]}"

# 컨테이너 안에서 (entrypoint 가 안내를 출력함):
#   설치(1회, 1~2시간):  setsid nohup univtac_install_main.sh > log/install_isaac45.log 2>&1 &   (원본 install.sh 는 쓰지 않음, 노트 §5)
#   검증:                conda activate UniVTAC && univtac_install_main.sh --check
#   스모크:              bash collect_data.sh grasp_classify demo 0
#   VNC:                 start_vnc.sh   (포트 5901, SSH 터널로 접속)
# 컨테이너에 다시 붙기:  docker exec -it univtac bash
