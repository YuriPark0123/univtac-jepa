#!/bin/bash
# -----------------------------------------------------------------------------
# UniVTAC 컨테이너 entrypoint
#   1) 상태 점검 출력 (GPU / conda env / 리포 브랜치)  — 설치 여부에 따라 다음 할 일을 안내
#   2) TS_ENABLE=1 이면 tailscale 자동 시작 (기존 컨테이너와 동일한 opt-in 방식)
#   3) 전달받은 명령(기본 /bin/bash) 실행
# -----------------------------------------------------------------------------
ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
ENV_NAME="${UNIVTAC_CONDA_ENV:-UniVTAC}"
ENV_PY="/root/miniconda3/envs/${ENV_NAME}/bin/python"

# host 에서 guest 계정으로 clone 한 리포를 root 로 다루므로 git 의 dubious-ownership 경고를 끈다
git config --global --add safe.directory '*' 2>/dev/null || true
# host 마운트(uid 1001) 인 pip 캐시는 소유자가 달라 pip 이 캐시를 끄므로 root 로 맞춘다
[ -d /root/.cache/pip ] && chown root:root /root/.cache/pip 2>/dev/null || true

echo "──────────────────────────────────────────────────────────────"
echo "[entrypoint] GPU"
if ! nvidia-smi -L 2>/dev/null; then
    echo "  !! nvidia-smi 실패 — NVML/cgroup 문제면 host 에서: docker restart $(hostname)"
fi

echo "[entrypoint] UniVTAC 리포: ${ROOT}  (Isaac Sim 4.5.0 / main 브랜치용 이미지)"
if [ -f "${ROOT}/scripts/install.sh" ]; then
    echo "  branch: $(git -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')" \
         " @ $(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?')"
else
    echo "  !! ${ROOT}/scripts/install.sh 가 없습니다. host 에서 clone 후 다시 실행하세요."
fi

if [ -x "${ENV_PY}" ]; then
    echo "[entrypoint] conda env '${ENV_NAME}' 있음.  →  conda activate ${ENV_NAME}"
    echo "             검증:        univtac_install_main.sh --check"
    echo "             수집(헤드리스): HEADLESS=1 ENABLE_CAMERAS=1 python scripts/collect_data.py <task> demo --start_seed 0 --max_seed 0 --episode_num 1 --gpu 0"
    echo "             평가:        bash data/download.sh --checkpoint <task> && univtac_link_checkpoints.sh"
    echo "                          HEADLESS=1 python scripts/eval_policy.py <task> demo ACT/deploy --total_num 100 --headless"
    echo "             전체 절차:   /workspace/univtac_docker/README_univtac.md  (호스트의 univtac/docker/ 읽기전용 마운트)"
else
    echo "[entrypoint] conda env '${ENV_NAME}' 없음.  →  설치(1~2시간, 로그 남기기):"
    echo "  mkdir -p log && setsid nohup univtac_install_main.sh > log/install_isaac45.log 2>&1 &   # tail -f log/install_isaac45.log"
    echo "  (UNIVTAC_CUDA_ARCH=${UNIVTAC_CUDA_ARCH:-86}, UNIVTAC_BUILD_JOBS=${UNIVTAC_BUILD_JOBS:-16} — 환경변수로 이미 전달됨)"
fi
echo "[entrypoint] VNC: start_vnc.sh  (display :${VNC_DISPLAY:-1}, 포트 $((5900 + ${VNC_DISPLAY:-1})))"
echo "──────────────────────────────────────────────────────────────"

if [ -n "${TS_ENABLE}" ] && command -v tailscaled >/dev/null 2>&1; then
    /usr/local/bin/start_tailscale.sh || echo "[entrypoint] tailscale 시작 실패 — 무시하고 계속합니다."
fi

exec "$@"
