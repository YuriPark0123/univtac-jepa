#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# UniVTAC main 브랜치 (Isaac Sim 4.5.0 / Isaac Lab 2.1.1) 환경 설치 — 컨테이너 안에서 실행
#   UniVTAC_INSTALL_NOTES.md §5 의 "수동 절차"를 idempotent 스크립트로 옮긴 것.
#   원본 scripts/install.sh 를 쓰지 않는 이유(§5 표): vcpkg clone 조건 반전, sudo apt, livestream 테스트로 멈춤,
#   설치 직후 수집 실행, 그리고 env.yaml 의 `gcc=11.4` 가 pkgs/main 에 없어 첫 단계에서 실패.
#
# 사용:  univtac_install_main.sh            # 전체 설치 (이미 된 단계는 건너뜀)
#        univtac_install_main.sh --check    # 설치 없이 버전 검증만
# 환경변수: UNIVTAC_ROOT(/workspace/UniVTAC) UNIVTAC_CONDA_ENV(UniVTAC) UNIVTAC_CUDA_ARCH(86)
#           UNIVTAC_BUILD_JOBS(16) UNIVTAC_VCPKG_ROOT(/root/Toolchain/vcpkg)
# 로그 남기며 백그라운드로:  setsid nohup univtac_install_main.sh > log/install_isaac45.log 2>&1 &
# -----------------------------------------------------------------------------
# nounset(-u) 은 쓰지 않는다: conda 활성/비활성 훅(특히 cuda-toolkit 이 끌고 오는 gcc_linux-64 의
# deactivate-gcc_linux-64.sh)이 unbound variable 로 죽는다. 모든 변수는 ${VAR:-default} 로 참조한다.
set -Eeo pipefail

ROOT="${UNIVTAC_ROOT:-/workspace/UniVTAC}"
ENV_NAME="${UNIVTAC_CONDA_ENV:-UniVTAC}"
PY_VER="${UNIVTAC_PYTHON:-3.10.12}"     # 프로젝트 스펙 문서(2026-09-18)와 동일한 패치 버전
CUDA_ARCH="${UNIVTAC_CUDA_ARCH:-86}"; CUDA_ARCH="${CUDA_ARCH//./}"
JOBS="${UNIVTAC_BUILD_JOBS:-16}"
VCPKG_ROOT="${UNIVTAC_VCPKG_ROOT:-/root/Toolchain/vcpkg}"
# vcpkg 체크아웃 커밋. libuipc(main) 의 manifest 는 builtin-baseline=b2cb0da 인데, 체크아웃을 그 커밋에 맞추면
# 버전 DB 가 너무 오래돼 `cpptrace>=0.8.3` 을 못 찾는다(실측: "no version database entry for cpptrace at 0.8.3").
# baseline 은 "조상 커밋"이기만 하면 되므로, upstream isaac51 브랜치가 고정한 dd3097e(b2cb0da 포함, cpptrace 1.0.2,
# tbb 2022.1, libigl 2.6) 를 쓴다. upstream main 은 아예 고정 없이 최신 vcpkg 를 쓴다.
VCPKG_COMMIT="dd3097e305afa53f7b4312371f62058d2e665320"
ISAACLAB_TAG="v2.1.1"
CUROBO_COMMIT="0a50de1ba72db304195d59d9d0b1ed269696047f"   # cuRobo v0.7.7 (원본 install.sh 와 동일)
CHECK_ONLY=0; [[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
trap 'echo "!! 설치 실패: ${BASH_SOURCE[0]}:${LINENO} — 위 로그를 확인하세요"' ERR

TORCH_ARCH="${CUDA_ARCH:0:${#CUDA_ARCH}-1}.${CUDA_ARCH: -1}"   # 86 → 8.6
log() { echo "[univtac-install $(date +%H:%M:%S)] $*"; }
step() { echo; echo "================ $* ================"; }

# pip 은 캐시 디렉토리 소유자가 다르면(host 마운트, uid 1001) 캐시를 꺼버린다 → root 로 맞춘다
[[ -d /root/.cache/pip ]] && chown root:root /root/.cache/pip 2>/dev/null || true
cd "${ROOT}"
[[ -f scripts/install.sh && -d third_party/TacEx ]] || { echo "!! ${ROOT} 가 UniVTAC 리포가 아닙니다"; exit 1; }
BR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
[[ "${BR}" == "main" ]] || log "경고: 현재 브랜치가 '${BR}' 입니다 (이 스크립트는 main = Isaac Sim 4.5 용)"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
MAIN=https://repo.anaconda.com/pkgs/main
R=https://repo.anaconda.com/pkgs/r

# ---------------------------------------------------------------- [1/9] conda env
step "[1/9] conda env '${ENV_NAME}' (python ${PY_VER})"
if [[ ! -x "${CONDA_BASE}/envs/${ENV_NAME}/bin/python" ]]; then
    [[ ${CHECK_ONLY} -eq 1 ]] && { echo "!! env 없음"; exit 1; }
    conda create -y -n "${ENV_NAME}" --override-channels -c "${MAIN}" -c "${R}" "python=${PY_VER}" pip
fi
conda activate "${ENV_NAME}"
PY="${CONDA_PREFIX}/bin/python"
"${PY}" -c 'import sys; assert sys.version_info[:2]==(3,10), sys.version' || { echo "!! python 3.10 이 아님"; exit 1; }
[[ "$("${PY}" -c 'import platform;print(platform.python_version())')" == "${PY_VER}" ]] || log "경고: python 패치 버전이 ${PY_VER} 가 아님 (기존 env 재사용 중)"

# ---------------------------------------------------------------- [2/9] toolchain
#   원본 env.yaml: cmake=3.26, gcc=11.4, cuda-toolkit=12.4 (pkgs/main + pkgs/r)
#   → gcc=11.4 는 그 채널에 없음(11.2 만). Ubuntu 22.04 시스템 gcc 가 정확히 11.4.0 이라 그걸 쓴다.
step "[2/9] toolchain: conda cmake 3.26 + cuda-toolkit 12.4, 시스템 gcc 11.4"
if [[ ${CHECK_ONLY} -eq 0 ]] && ! { [[ -x "${CONDA_PREFIX}/bin/nvcc" ]] && [[ -x "${CONDA_PREFIX}/bin/cmake" ]]; }; then
    conda install -y -n "${ENV_NAME}" --override-channels -c "${MAIN}" -c "${R}" cmake=3.26 cuda-toolkit=12.4
fi
export CUDA_HOME="${CONDA_PREFIX}" CUDA_PATH="${CONDA_PREFIX}" CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# 컴파일러: 시스템 gcc/g++ 11.4 + 시스템 링커(-B/usr/bin).
#   PATH 앞쪽이 conda bin 이라 그냥 두면 conda 의 ld 가 잡힐 수 있어, upstream isaac51 브랜치와 같은 방식으로 고정한다.
for w in cc:gcc cxx:g++; do
    printf '#!/usr/bin/env bash\nexec /usr/bin/%s -B/usr/bin "$@"\n' "${w#*:}" > "/usr/local/bin/univtac-${w%%:*}"
    chmod +x "/usr/local/bin/univtac-${w%%:*}"
done
export CC=/usr/local/bin/univtac-cc CXX=/usr/local/bin/univtac-cxx CUDAHOSTCXX=/usr/local/bin/univtac-cxx
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" MAX_JOBS="${JOBS}"
nvcc --version | grep -q "release 12\.4" || { echo "!! nvcc 12.4 아님: $(nvcc --version | tail -1)"; exit 1; }
/usr/bin/gcc --version | head -1 | grep -q " 11\." || { echo "!! 시스템 gcc 11.x 아님: $(/usr/bin/gcc --version | head -1)"; exit 1; }
echo "[toolchain] CC=${CC} ($(/usr/bin/gcc -dumpversion)), ld=$(/usr/bin/ld --version | head -1)"
cmake --version | head -1

# ---------------------------------------------------------------- check
check_environment() {
    "${PY}" - <<'PY'
from importlib.metadata import version, PackageNotFoundError
want = {"isaacsim": "4.5.0", "isaaclab": "0.41.3", "numpy": "1.26.4", "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
        "warp-lang": "1.0.0", "torch-scatter": "2.1.2+pt25cu124", "tacex": None, "tacex-assets": None,
        "tacex-tasks": None, "tacex_uipc": None, "nvidia-curobo": None, "transforms3d": None, "trimesh": None, "tetgen": None, "modelscope": None}
bad = False
for p, w in want.items():
    try: v = version(p)
    except PackageNotFoundError: print(f"[missing] {p}"); bad = True; continue
    ok = (w is None) or v.startswith(w)
    print(f"[{'ok' if ok else 'wrong'}] {p}=={v}" + ("" if ok else f" (expected {w})")); bad |= not ok
import torch; print(f"[{'ok' if torch.version.cuda=='12.4' else 'wrong'}] torch cuda runtime {torch.version.cuda}, available={torch.cuda.is_available()}")
import pathlib; _sc = pathlib.Path("third_party/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py").read_text()
print(("[ok] " if "UNIVTAC_SCENE_QUERY" in _sc else "[missing] ") + "IsaacLab headless scene-query 패치"); bad |= "UNIVTAC_SCENE_QUERY" not in _sc
import uipc, curobo  # noqa  (컴파일된 모듈 import. tacex 는 omni.kit(Isaac Sim 앱) 기동 후에만 import 가능 → 스모크 테스트로 확인)
print("[ok] import uipc / curobo")
raise SystemExit(1 if bad else 0)
PY
}
if [[ ${CHECK_ONLY} -eq 1 ]]; then check_environment; exit; fi

UV="${CONDA_PREFIX}/bin/uv"
[[ -x "${UV}" ]] || "${PY}" -m pip install -q uv
"${PY}" -m pip install -q --upgrade pip
"${UV}" pip install --python "${PY}" 'setuptools<82' wheel toml pybind11 mypy 'numpy<2'

# ---------------------------------------------------------------- [3/9] torch
step "[3/9] torch 2.5.1 + torchvision 0.20.1 (cu124)"
ensure_torch() {
    local v; v="$("${PY}" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
    if [[ "${v}" != "2.5.1+cu124" ]]; then
        log "torch=${v:-none} → 2.5.1+cu124 로 (재)설치"
        "${UV}" pip uninstall --python "${PY}" torch torchvision torchaudio 2>/dev/null || true
        "${UV}" pip install --python "${PY}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    else log "torch 2.5.1+cu124 이미 설치됨"; fi
}
ensure_torch

# ---------------------------------------------------------------- [4/9] Isaac Sim 4.5
step "[4/9] isaacsim[all,extscache]==4.5.0 (pypi.nvidia.com, ~10GB)"
if "${PY}" -m pip show isaacsim >/dev/null 2>&1; then log "isaacsim 이미 설치됨"; else
    "${UV}" pip install --python "${PY}" 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
fi

# ---------------------------------------------------------------- [5/9] Isaac Lab 2.1.1
step "[5/9] Isaac Lab ${ISAACLAB_TAG} → third_party/IsaacLab (editable)"
if "${PY}" -m pip show isaaclab >/dev/null 2>&1; then log "isaaclab 이미 설치됨"; else
    if [[ ! -d third_party/IsaacLab/.git ]]; then git clone https://github.com/isaac-sim/IsaacLab third_party/IsaacLab; fi
    git -C third_party/IsaacLab checkout -q "${ISAACLAB_TAG}"
    "${UV}" pip install --python "${PY}" flatdict==4.0.1 --no-build-isolation
    ( cd third_party/IsaacLab && ./isaaclab.sh --install )     # CONDA_PREFIX 활성 상태 → conda python 에 설치
    ensure_torch                                               # isaaclab 이 torch 를 바꿨으면 되돌림
fi

# ---------------------------------------------------------------- [5b/9] Isaac Lab 로컬 패치: 헤드리스 scene query
#   Isaac Lab 은 GUI 일 때만 sim.enable_scene_query_support 를 True 로 강제한다. TacEx 의 gel pad 부착은 PhysX sweep
#   쿼리를 쓰므로 헤드리스(collect/eval/parallel 전부)에서 이 플래그가 꺼지면 부착점이 NaN → 촉각 깨짐·grasp 전부 실패.
#   → 로컬 clone 의 simulation_context.py 한 곳을 고쳐 기본 ON (UNIVTAC_SCENE_QUERY=0 이면 원래 동작). idempotent.
step "[5b/9] Isaac Lab 로컬 패치 (헤드리스 scene query 강제)"
SC=third_party/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py
if grep -q "UNIVTAC_SCENE_QUERY" "${SC}"; then log "이미 패치됨"; else
    "${PY}" - "${SC}" <<'PYEOF'
import sys; p=sys.argv[1]; s=open(p).read()
old = "        if self._has_gui:\n            self.cfg.enable_scene_query_support = True\n"
new = ("        # [UniVTAC 로컬 패치] TacEx 의 gel pad 부착(PhysX sweep 쿼리)은 헤드리스에서도 scene query 가 필요하다.\n"
       "        # 기본 강제 ON, 끄려면 UNIVTAC_SCENE_QUERY=0 (원본: GUI 일 때만 True).\n"
       "        if self._has_gui or os.environ.get(\"UNIVTAC_SCENE_QUERY\", \"1\") != \"0\":\n"
       "            self.cfg.enable_scene_query_support = True\n")
assert old in s, "patch site not found (IsaacLab version changed?)"
open(p, "w").write(s.replace(old, new)); print("patched", p)
PYEOF
fi

# ---------------------------------------------------------------- [6/9] cuRobo
step "[6/9] cuRobo ${CUROBO_COMMIT:0:7} (v0.7.7) → third_party/curobo, sm_${CUDA_ARCH}"
if "${PY}" -m pip show nvidia_curobo >/dev/null 2>&1; then log "curobo 이미 설치됨"; else
    if [[ ! -d third_party/curobo/.git ]]; then git clone https://github.com/NVlabs/curobo.git third_party/curobo; fi
    git -C third_party/curobo checkout -q "${CUROBO_COMMIT}"
    "${UV}" pip install --python "${PY}" warp-lang==1.0.0 --no-build-isolation
    "${UV}" pip install --python "${PY}" -e third_party/curobo --no-build-isolation
fi

# ---------------------------------------------------------------- [7/9] TacEx core (번들 소스만!)
step "[7/9] TacEx core (tacex, tacex_assets, tacex_tasks) + torch_scatter pt25cu124"
if "${PY}" -m pip show tacex >/dev/null 2>&1; then log "tacex 이미 설치됨"; else
    ( cd third_party/TacEx && ./tacex.sh -i )
fi
TS="$("${PY}" -c 'from importlib.metadata import version;print(version("torch-scatter"))' 2>/dev/null || true)"
if [[ "${TS}" != 2.1.2+pt25cu124 ]]; then
    log "torch_scatter=${TS:-none} → 2.1.2+pt25cu124 (tacex/setup.py 는 pt28cu128 휠을 가리켜 교체 필요)"
    "${UV}" pip uninstall --python "${PY}" torch_scatter 2>/dev/null || true
    "${UV}" pip install --python "${PY}" torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
fi
ensure_torch

# ---------------------------------------------------------------- [7b/9] TacEx 에셋 복원 (LFS 포인터 → 실제 파일)
#   main 브랜치는 커밋 a9abc0d "remove git lfs due to quota" 이후 third_party/TacEx 의 USD/메시 114개가
#   131바이트 LFS 포인터 파일로만 남아 있다 (LFS 서버 quota 초과로 받을 수도 없음). 실제 바이너리는 isaac51 브랜치에
#   "Restore TacEx assets as non-LFS files" 로 들어 있고, 포인터의 sha256 과 blob 해시가 전부 일치함을 확인했다(2026-09-19).
#   → 포인터마다 isaac51 의 같은 경로 blob 을 sha256 검증 후 덮어쓴다. (없으면 Isaac Sim 이 Robot/gelpad_left prim 을 못 찾음)
step "[7b/9] TacEx 에셋 복원 — LFS 포인터를 isaac51 브랜치의 실제 파일로"
ASSET_REF="origin/isaac51"
git rev-parse --verify -q "${ASSET_REF}" >/dev/null || git fetch -q origin isaac51:refs/remotes/origin/isaac51
restored=0; failed=0
while IFS= read -r -d "" f; do
    oid="$(sed -n "s/^oid sha256://p" "$f")"; want="$(sed -n "s/^size //p" "$f")"
    if git cat-file -e "${ASSET_REF}:${f}" 2>/dev/null; then
        tmp="$(mktemp)"; git show "${ASSET_REF}:${f}" > "${tmp}"
        if [[ "$(stat -c%s "${tmp}")" == "${want}" && "$(sha256sum "${tmp}" | cut -d" " -f1)" == "${oid}" ]]; then
            mv -f "${tmp}" "$f"; restored=$((restored+1))
        else rm -f "${tmp}"; failed=$((failed+1)); echo "  !! 해시 불일치: $f"; fi
    else failed=$((failed+1)); echo "  !! ${ASSET_REF} 에 없음: $f"; fi
done < <(grep -rlZ --exclude-dir=.git -m1 "^version https://git-lfs.github.com/spec/v1" third_party/TacEx 2>/dev/null || true)
log "TacEx 에셋 복원: ${restored}개 (실패 ${failed}개; 0 이면 이미 복원됨)"
[[ ${failed} -eq 0 ]] || { echo "!! 에셋 복원 실패 — 위 목록 확인"; exit 1; }

# ---------------------------------------------------------------- [8/9] vcpkg
UIPC=third_party/TacEx/source/tacex_uipc
step "[8/9] vcpkg @ ${VCPKG_COMMIT:0:7} → ${VCPKG_ROOT}"
if [[ ! -d "${VCPKG_ROOT}/.git" ]]; then mkdir -p "$(dirname "${VCPKG_ROOT}")"; git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"; fi
if [[ "$(git -C "${VCPKG_ROOT}" rev-parse HEAD)" != "${VCPKG_COMMIT}" || ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
    git -C "${VCPKG_ROOT}" fetch -q origin "${VCPKG_COMMIT}" || true
    git -C "${VCPKG_ROOT}" checkout -q --detach "${VCPKG_COMMIT}"
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics      # 체크아웃에 맞는 vcpkg 바이너리로 갱신
    rm -rf "${UIPC:-third_party/TacEx/source/tacex_uipc}/build"   # 이전 vcpkg 로 만든 manifest/캐시 폐기
else log "vcpkg 이미 ${VCPKG_COMMIT:0:7} — 건너뜀"; fi
export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"

# ── 포트 소스 아카이브 해시 드리프트 대응 ──────────────────────────────────────
#   GitHub 이 archive/*.tar.gz 를 재생성하면 바이트가 바뀌어 vcpkg 포트의 SHA512 와 불일치한다
#   (실측 2026-09-19: tinygltf v2.9.3/v2.9.6 모두 "had an unexpected hash"). vcpkg 커밋을 바꿔도 소용없는
#   외부 요인이라, 문제 포트만 baseline 커밋의 포트 파일을 overlay 로 복사하고 SHA512 를 "지금 실제 아카이브"의
#   해시로 갱신한다. 버전은 upstream 이 빌드한 것과 동일하게 유지된다(내용은 같고 압축 바이트만 다름).
#   형식: "<port>:<baseline commit>:<archive URL>"  — 드리프트가 또 생기면 여기에 한 줄 추가.
VCPKG_BASELINE="b2cb0da531c2f1f740045bfe7c4dac59f0b2b69c"     # libuipc(main) gen_vcpkg_json.py 의 builtin-baseline
OVERLAY="${VCPKG_ROOT}/univtac-overlay-ports"
HASH_FIX_PORTS=(
    "tinygltf:${VCPKG_BASELINE}:https://github.com/syoyo/tinygltf/archive/v2.9.3.tar.gz"
)
mkdir -p "${OVERLAY}"
for spec in "${HASH_FIX_PORTS[@]}"; do
    port="${spec%%:*}"; rest="${spec#*:}"; commit="${rest%%:*}"; url="${rest#*:}"
    if [[ ! -f "${OVERLAY}/${port}/.hash-fixed" ]]; then
        rm -rf "${OVERLAY}/${port}"
        git -C "${VCPKG_ROOT}" archive "${commit}" "ports/${port}" | tar -x -C "${OVERLAY}" --strip-components=1
        actual="$(curl -fsSL "${url}" | sha512sum | cut -d" " -f1)"
        [[ ${#actual} -eq 128 ]] || { echo "!! ${port}: 아카이브 다운로드/해시 계산 실패 (${url})"; exit 1; }
        sed -i -E "s/(SHA512[[:space:]]+)[0-9a-f]{128}/\1${actual}/" "${OVERLAY}/${port}/portfile.cmake"
        echo "${actual}" > "${OVERLAY}/${port}/.hash-fixed"
        log "overlay 포트 ${port}@$(grep -oE '"version[^"]*": "[^"]+"' "${OVERLAY}/${port}/vcpkg.json" | head -1 | cut -d'"' -f4): SHA512 → ${actual:0:16}…"
    fi
done
export VCPKG_OVERLAY_PORTS="${OVERLAY}"       # vcpkg CLI 와 vcpkg.cmake 툴체인 모두 이 env 를 읽는다

# ---------------------------------------------------------------- [9/9] libuipc + tacex_uipc
step "[9/9] libuipc 빌드 + tacex_uipc (가장 오래 걸림: 30분~1시간+, -j${JOBS})"
# main 의 setup.py 는 -j4 로 하드코딩 → CMAKE_BUILD_PARALLEL_LEVEL(기본 8) 을 따르도록 한 줄만 고친다 (로컬 clone, idempotent)
if grep -q 'build_args += \["-j4"\]' "${UIPC}/setup.py"; then
    sed -i 's|build_args += \["-j4"\]|build_args += ["-j" + os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")]|' "${UIPC}/setup.py"
    log "setup.py: -j4 → -j\$CMAKE_BUILD_PARALLEL_LEVEL 로 패치 (git diff ${UIPC}/setup.py 로 확인 가능)"
fi
if "${PY}" -c 'import uipc' >/dev/null 2>&1 && "${PY}" -m pip show tacex_uipc >/dev/null 2>&1; then log "tacex_uipc/uipc 이미 설치됨"; else
    # build/ 는 항상 비우고 시작한다. libuipc 의 uipc_config_vcpkg_install() 은 build/vcpkg.json 이 이미 있으면
    # "설치 불필요"로 보고 vcpkg install 을 건너뛰는데, 이전 시도가 중간에 실패했으면 패키지가 없는 채로
    # find_package(urdfdom 등) 에서 죽는다. 완성된 포트는 vcpkg 바이너리 캐시에 있어 재빌드 비용은 작다.
    [[ -d "${UIPC}/build" ]] && { log "이전 build/ 제거 (vcpkg manifest 재설치 강제)"; rm -rf "${UIPC}/build"; }
    "${UV}" pip install --python "${PY}" -e "${UIPC}" -v --no-build-isolation
fi
"${UV}" pip install --python "${PY}" transforms3d trimesh tetgen modelscope    # modelscope: data/download.sh (데이터셋/checkpoint)

step "검증"
check_environment
echo
echo "설치 완료.  사용:  conda activate ${ENV_NAME}"
echo "  스모크:  HEADLESS=1 ENABLE_CAMERAS=1 python scripts/collect_data.py grasp_classify demo --start_seed 0 --max_seed 0 --episode_num 1 --gpu 0"
echo "  전체 사용법: /workspace/univtac_docker/README_univtac.md — 수집 / 데이터셋·checkpoint 다운로드 / 학습 / 평가"
