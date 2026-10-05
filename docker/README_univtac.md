# UniVTAC 벤치마크 컨테이너 운영 가이드 (main = Isaac Sim 4.5.0, 논문 환경)

갱신: 2026-09-22 (로컬 이전, 마운트 2개로 단순화, 스펙 검증 반영). 이 문서 하나로 ① 컨테이너가 지워졌을 때 다시 띄우기, ② 무엇이 어디에 마운트돼 있는지,
③ 다른 서버에 똑같이 구축하기, ④ 컨테이너 안에서 벤치마크(수집 → 학습 → 평가) 돌리기를 설명한다.

---

## 0. 한눈에 보기

```
[로컬] /mnt/data/yuri_docker/data/isaaclab_vnc/univtac/docker/   ← 재현에 필요한 "코드" 전부 (이 폴더만 있으면 됨)
       (NAS 사본: /mnt/nas/yuri/yuri_docker/data/isaaclab_vnc/ — 2026-09-22 NAS 접근 불가로 로컬이 원본. 기존 isaaclab:vnc 용
        Dockerfile/run_container.sh/scripts 와 섞이지 않도록 univtac/docker/ 하위에 분리)
   ├─ univtac_setup_host.sh        호스트 원샷: clone → build → run → 설치 → 검증
   ├─ Dockerfile.univtac           이미지 univtac:isaac45
   ├─ run_container_univtac.sh     컨테이너 기동 (마운트 규칙)
   ├─ README_univtac.md            이 문서
   └─ scripts/
       ├─ univtac_install_main.sh      컨테이너 안 환경 설치 (9단계, idempotent, --check)
       ├─ univtac_link_checkpoints.sh  공개 checkpoint 를 정책 코드 경로에 연결
       ├─ univtac_entrypoint.sh        기동 시 상태·다음 할 일 안내
       ├─ univtac_start_vnc.sh         VNC 시작 → start_vnc.sh (GUI 필요 시)
       ├─ univtac_stop_vnc.sh          VNC 종료 → stop_vnc.sh
       └─ start_tailscale.sh           VPN (TS_ENABLE=1 일 때만)

[서버 로컬]  ← 컨테이너를 지워도 남는 "상태" (이 두 곳이 있으면 설치 1~2시간을 건너뜀)
   ├─ /mnt/data/yuri_docker/data/isaaclab_vnc/univtac/UniVTAC   리포 + third_party 빌드 + 데이터셋/결과/로그
   │     └─ data/  eval_result/  log/   ← 코드가 ./data 등 상대경로로 쓰므로 리포 안에 있어야 한다
   └─ $HOME/.isaaclab_vnc/univtac/                              conda env UniVTAC, vcpkg, Kit 캐시 (36GB)
```

---

## 1. 컨테이너가 삭제됐을 때 다시 띄우기 (같은 서버)

컨테이너는 "일회용"이다. 환경·코드·데이터는 전부 호스트 마운트(§2)에 있으므로 컨테이너를 지워도 잃는 것이 없다.

```bash
cd /mnt/data/yuri_docker/data/isaaclab_vnc/univtac/docker
bash univtac_setup_host.sh              # ~1분: 이미지(캐시) → 컨테이너 재생성 → 설치 스크립트가 "이미 설치됨" 확인 → 검증
docker exec -it univtac bash            # 접속
```

- 이미지(`univtac:isaac45`)까지 지워졌어도 같은 명령이면 된다 (빌드 ~10분, 네트워크 필요).
- 컨테이너만 빠르게: `DETACH=1 bash run_container_univtac.sh` (백그라운드) 또는 `bash run_container_univtac.sh` (바로 셸 진입).
  이 스크립트는 같은 이름의 기존 컨테이너를 **지우고 새로 만든다** — 안에서 돌던 작업이 있으면 끝나고 실행할 것.
- 컨테이너가 살아 있는데 멈춘 것 같으면: `docker restart univtac` (NVML/cgroup 오류일 때도 이걸로 해결).
- 기동 시 entrypoint 가 GPU·env·브랜치 상태와 다음 할 일을 출력한다.

---

## 2. 마운트 — 컨테이너 ↔ 호스트

`run_container_univtac.sh` 가 매번 동일하게 건다. **컨테이너 안 경로는 항상 같다**(코드가 절대 경로를 가정하므로 바꾸지 말 것).

| 컨테이너 안 | 호스트 | 내용 / 왜 밖에 두나 |
|---|---|---|
| `/workspace/UniVTAC` (마운트 1개로 전부) | `…/isaaclab_vnc/univtac/UniVTAC` (로컬 디스크) | UniVTAC 리포 `main@0dafa10` + `third_party/{IsaacLab,curobo,TacEx}` 빌드 + **출력 전부**(`data/` 데이터셋·수집 hdf5·checkpoint, `eval_result/`, `log/`). 코드가 `./data`·`./eval_result`·`./log` 상대경로를 쓰므로 리포 안이어야 한다. **NAS(CIFS) 불가** — 심볼릭 링크·빌드가 안 됨 |
| `/root/miniconda3/envs` | `$HOME/.isaaclab_vnc/univtac/conda_envs` | conda env `UniVTAC` (Isaac Sim 포함 ~20GB) |
| `/root/miniconda3/pkgs`, `/root/.cache/pip` | `$HOME/.isaaclab_vnc/univtac/{conda_pkgs,pip_cache}` | 재설치 시 재다운로드 방지 |
| `/root/.cache/ov`, `/root/.local/share/ov` | `$HOME/.isaaclab_vnc/univtac/cache/{ov,ov_data}` | Isaac Sim 셰이더/확장 캐시 (첫 기동 2분 → 이후 30초) |
| `/root/Toolchain/vcpkg` | `$HOME/.isaaclab_vnc/univtac/vcpkg` | libuipc 빌드용 vcpkg + overlay 포트 |
| `/root/.cache/huggingface` | `$HOME/.isaaclab_vnc/hf_cache` | HF 캐시 (기존 isaaclab 컨테이너와 공유) |
| `/var/lib/tailscale` | `$HOME/.isaaclab_vnc/univtac/tailscale` | `TS_ENABLE=1` 로 띄울 때만 사용 |
| `/root/.vscode-server` | `$HOME/.isaaclab_vnc/univtac/vscode-server` | VS Code Remote(Attach to Container) 서버+확장 1.1GB. 안 걸면 컨테이너 재생성 때마다 재다운로드·재색인 → "파일 하나 여는 데 몇 분" (2026-09-23 추가) |
| `/workspace/univtac_docker` (ro) | `…/univtac/docker` | **이 문서와 재현 스크립트 일체** — 컨테이너 안에서 `less /workspace/univtac_docker/README_univtac.md` |
| `/workspace/isaaclab_ref` (ro) | `ISAACLAB_REF_SRC` 로 지정한 폴더 | 참고용 코드(선택). 지정하지 않으면 마운트하지 않음. **UniVTAC 이 쓰는 Isaac Lab 은 여기가 아니라 `/workspace/UniVTAC/third_party/IsaacLab` (v2.1.1)** |

데이터 마운트는 **리포 1개 + 재현 파일(ro) 1개** 뿐이다(나머지는 conda env·캐시). 예전에는 출력을 `univtac/outputs/` 로 분리해 3개를 더 걸었지만, 코드가 어차피 리포 안 상대경로를 쓰므로 2026-09-22 리포 안으로 합쳤다.

### 호스트 경로를 옮겨도 되는가 — 된다 (검증 2026-09-22)

빌드 산출물·설치 정보에 박혀 있는 경로는 **전부 컨테이너 쪽 경로**(`/workspace/UniVTAC/...`)다. 호스트 경로는 어디에도 기록되지 않는다:

| 경로가 박히는 곳 | 박힌 값 |
|---|---|
| conda env 의 editable 설치 (`__editable___*_finder.py`) — isaaclab·isaaclab_tasks·tacex·tacex_uipc·curobo 등 10개 | `/workspace/UniVTAC/third_party/...` |
| libuipc CMake 캐시 (`tacex_uipc/build/CMakeCache.txt`) | `/workspace/UniVTAC/third_party/...` |
| checkpoint·encoder 심볼릭 링크 (`univtac_link_checkpoints.sh` 가 만든 것) | `/workspace/UniVTAC/data/checkpoints/...` |

conda env 전체를 뒤져도 `/mnt/...` 같은 호스트 경로는 나오지 않는다. 따라서 **마운트 대상(컨테이너 쪽) 경로만 그대로면** 호스트에서 폴더를 옮기거나 다른 서버에 복사해도 깨지지 않는다. 옮긴 뒤에는 `UNIVTAC_LOCAL`(리포 상위) 과 `ISAACLAB_LOCAL`(env·캐시 상위)만 새 위치로 지정하면 된다.

실제 검증: 같은 환경을 다른 호스트 경로(`/home/guest/univtac_altpath`)로 마운트해 컨테이너를 띄운 뒤 `univtac_install_main.sh --check` 전 항목 ok, checkpoint 심볼릭 링크도 정상 해석됨.

주의 — 호스트 쪽에서 보면 위 심볼릭 링크들은 **깨진 것처럼 보인다**(`/workspace/...` 는 호스트에 없는 경로). 정상이며, 컨테이너 안에서만 해석된다. 그리고 NAS(CIFS)로 옮길 때는 심볼릭 링크·권한이 보존되지 않으므로 반드시 `tar` 로 묶어 옮긴다(§3-3).

기타 실행 옵션: `--gpus all --network=host --shm-size=16g`, `VNC_DISPLAY`(기본 1 → 포트 5901; 기존 isaaclab 컨테이너와 겹치면 `VNC_DISPLAY=2`),
`UNIVTAC_CUDA_ARCH`(기본 86 = A6000; RTX 4090 은 89), `UNIVTAC_BUILD_JOBS`(기본 16).
컨테이너 안에서 만든 파일은 호스트에서 root 소유가 된다.

호스트 경로를 바꾸려면 환경변수로: `UNIVTAC_LOCAL=/data/univtac ISAACLAB_LOCAL=/data/.isaaclab_vnc bash univtac_setup_host.sh`
(`UNIVTAC_LOCAL` → 리포의 상위 폴더, `ISAACLAB_LOCAL` → conda env·캐시의 상위 폴더).

---

## 3. 다른 서버에 똑같이 구축하기

### 3-1. 필요한 것

| 구분 | 내용 |
|---|---|
| 파일 | `univtac/docker/` 폴더 통째로: `univtac_setup_host.sh`, `Dockerfile.univtac`, `run_container_univtac.sh`, `.dockerignore`, `README_univtac.md`, `scripts/` (univtac_* 5개 + `start_tailscale.sh`) |
| 호스트 SW | docker ≥ 24 + **nvidia-container-toolkit**, git, NVIDIA 드라이버 ≥ 535 (검증: 550.107.02 / Ubuntu 22.04 / docker 28.1 / toolkit 1.17.6) |
| HW | NVIDIA GPU ≥ 24GB 권장 (수집 시 ~8GB, 평가 시 ~10GB 사용). sm_86(A6000)·sm_89(4090) 검증. 다른 아키텍처면 `UNIVTAC_CUDA_ARCH` 지정 |
| 디스크 | 로컬 디스크 ~50GB (env 36GB + 리포 2.3GB + 이미지 9.7GB) + 데이터셋(task 당 ~4GB) |
| 네트워크 (첫 설치 시) | github.com, pypi.nvidia.com(Isaac Sim 휠 ~10GB), pypi.org, download.pytorch.org, repo.anaconda.com, modelscope.cn(데이터셋) |

### 3-2. 절차 (처음부터, 1~2시간)

```bash
scp -r <이서버>:/mnt/data/yuri_docker/data/isaaclab_vnc/univtac/docker ./univtac_docker
cd univtac_docker
UNIVTAC_LOCAL=/data/univtac bash univtac_setup_host.sh      # 경로는 그 서버의 로컬 디스크로. 전제조건은 스크립트가 먼저 검사
```

스크립트가 하는 일: 로컬 폴더 생성 → UniVTAC clone + 커밋 고정(`0dafa10`, `isaac51` ref 도 fetch — 에셋 복원용) → 이미지 빌드 →
컨테이너 기동 → 컨테이너 안 `univtac_install_main.sh`(9단계) → `--check` 검증. 로그: `<UNIVTAC_LOCAL>/UniVTAC/log/install_isaac45.log`.
중간에 끊겨도 같은 명령을 다시 실행하면 된 단계는 건너뛴다.

### 3-3. 빠른 복제 (설치 1~2시간 생략)

설치가 끝난 서버의 두 폴더를 새 서버의 같은 역할 위치에 복사하면 설치 스크립트가 전부 "이미 설치됨"으로 통과한다(컨테이너 안 경로가 동일하므로 그대로 동작).

```bash
# 원본 서버
tar -C $HOME/.isaaclab_vnc -cf univtac_env.tar univtac                      # 36GB: conda env, vcpkg, 캐시
tar -C /mnt/data/yuri_docker/data/isaaclab_vnc/univtac \
    --exclude=UniVTAC/data --exclude=UniVTAC/eval_result --exclude=UniVTAC/log \
    -cf univtac_repo.tar UniVTAC                                             # 2.3GB: 리포 + 빌드 (데이터셋·결과 제외)
# 새 서버 (경로는 §2 의 두 상위 폴더)
mkdir -p $HOME/.isaaclab_vnc && tar -C $HOME/.isaaclab_vnc -xf univtac_env.tar
mkdir -p /data/univtac && tar -C /data/univtac -xf univtac_repo.tar
UNIVTAC_LOCAL=/data/univtac bash univtac_setup_host.sh                      # ~2분
```
NAS(CIFS) 를 거쳐 옮길 때는 반드시 tar 로 묶는다(심볼릭 링크·권한이 CIFS 에 저장되지 않음). GPU 아키텍처가 다르면(A6000→4090) cuRobo/libuipc 만 재빌드가 필요하므로 복제 대신 3-2 로 설치할 것.

---

## 4. 컨테이너 안에서 벤치마크 실행

모든 명령은 `docker exec -it univtac bash` 로 들어가서, `/workspace/UniVTAC` 에서, `conda activate UniVTAC` 후 실행한다.
헤드리스 서버이므로 **`HEADLESS=1`** 을 항상 앞에 붙인다(§6 #9·#10 참고 — 이 환경은 헤드리스에서도 촉각이 정상 동작하도록 패치돼 있음).

### 4-0. 상태 확인
```bash
univtac_install_main.sh --check        # 버전 검증 (전 항목 [ok] 여야 함)
less /workspace/univtac_docker/README_univtac.md   # 이 문서 (컨테이너 안 경로)
nvidia-smi                             # GPU 0/1 사용량 — 다른 컨테이너와 공유 중이면 빈 GPU 를 --gpu 로 지정
```

### 4-1. 태스크·설정
- 태스크 (`envs/*.py`): `lift_bottle` `lift_can` `insert_HDMI` `insert_hole` `insert_tube` `pull_out_key` `put_bottle_in_shelf` `grasp_classify` (+ `collect`: 접촉 사전학습용)
- 태스크 설정 (`task_config/*.yml`): `demo`(GUI 창 렌더, 논문 checkpoint 학습에 쓰인 설정), `clean`(헤드리스 전용, render 0), `contact`(접촉 수집). 필드 설명은 `docs/Collection.md`
- 센서: GelSight Mini(`gsmini`)만 지원 (upstream TODO)

### 4-2. 데이터 수집 (스크립트 전문가 정책 + cuRobo)
```bash
# 단일 프로세스 — 인자: task config --start_seed --max_seed --episode_num(성공 에피소드 수) --gpu
HEADLESS=1 ENABLE_CAMERAS=1 python scripts/collect_data.py grasp_classify demo --start_seed 0 --max_seed 200 --episode_num 50 --gpu 0
# 병렬 (Isaac Sim 앱을 워커 수만큼 띄움, GPU 메모리 워커당 ~8GB)
HEADLESS=1 python scripts/parallel_collect_data.py grasp_classify clean --workers 3 --episodes 100 --gpu 0,1
# (GUI/VNC 로 볼 때만) bash collect_data.sh grasp_classify demo 0
```
결과: `data/<task>/<config>/hdf5/<seed>.hdf5`, `video/<seed>_success.mp4`, `metadata.json`, `suc_map.txt`. 에피소드당 ~70초(A6000), 초기화 ~1분.
`collect_data.py` 는 `--headless` 플래그를 받지 않으므로 환경변수로 켠다. 이 서버에서 seed 0·1·5·7·9 성공(5/5) 확인.
접촉 사전학습 데이터: `HEADLESS=1 python scripts/collect_contact.py collect contact --headless`.

### 4-3. 공개 데이터셋 / checkpoint 받기 (modelscope, 설치돼 있음)
```bash
bash data/download.sh --task grasp_classify --version 45    # 100 에피소드 ≈ 4GB → data/isaac45/<task>/hdf5/   (main = 45)
bash data/download.sh --task --version 45                   # 8개 태스크 전부
bash data/download.sh --contact                             # 접촉 사전학습 데이터 (14 shape)
bash data/download.sh --checkpoint grasp_classify           # → data/checkpoints/<task>/{univtac,vision_only}/ + encoder.pth (≈0.9GB/task)
univtac_link_checkpoints.sh                                 # ★ 정책 코드가 찾는 경로로 링크 (아래 4-5)
```

### 4-4. 정책 학습 (ACT 계열, 같은 env 에서 실행 가능 — import 확인됨)
```bash
cd policy/ACT
# 1) hdf5 → ACT 학습 포맷.  인자: task task_config expert_data_num
#    입력: /workspace/UniVTAC/data/<task>/<task_config>/**/*.hdf5  (직접 수집한 것이 그 자리에 쌓임)
#    공개 데이터셋(data/isaac45/<task>/hdf5)으로 학습하려면 그 자리에 링크 (직접 수집한 demo 데이터가 없을 때):
#      mkdir -p ../../data/grasp_classify/demo && ln -sfn /workspace/UniVTAC/data/isaac45/grasp_classify/hdf5 ../../data/grasp_classify/demo/hdf5
bash process_data.sh grasp_classify demo 50            # → policy/ACT/data/sim-grasp_classify/demo-50/
# 2) 학습.  인자: task task_config expert_data_num seed gpu [train_config]
bash train.sh grasp_classify demo 50 0 0 train_config              # vision+tactile (논문 'univtac')
bash train.sh grasp_classify demo 50 0 0 train_config_vision       # vision only
#    → act_ckpt/act-<task>/<config>-<ep>/<train_config>/{policy_last.ckpt,dataset_stats.pkl}
```
`train_config*.yml` 의 `tactile_ckpt` 는 공유 촉각 인코더(`encoder/checkpoints/resnet18/20251128-125750/best.pth`) — `univtac_link_checkpoints.sh` 가 다운로드한 `encoder.pth` 를 그 자리에 링크한다. 인코더를 직접 학습하려면 `encoder/train.py`(접촉 데이터 필요).
Ablation(`policy/Ablation`, 모달리티 조합)·ViTAL(`policy/ViTAL`, CLIP 사전학습 인코더)도 같은 구조. smolvla/UniT 는 별도 env 가 필요할 수 있음(각 폴더의 `conda_env*.yaml`).

### 4-5. 정책 평가 = 벤치마크 점수
```bash
# 인자: task task_config <policy>/<deploy.yml 이름> [--total_num N] [--start_seed S] [--max_seed M] [--expert_check]
HEADLESS=1 python scripts/eval_policy.py grasp_classify demo ACT/deploy --total_num 100 --headless
TRAIN_CONFIG=train_config_vision HEADLESS=1 python scripts/eval_policy.py grasp_classify demo ACT/deploy --total_num 100 --headless   # vision-only checkpoint
# 병렬 (인자: task task_config policy_config --workers --total_num --gpu)
HEADLESS=1 python scripts/parallel_eval_policy.py grasp_classify demo ACT/deploy --workers 3 --total_num 100 --gpu 0,1
```
- checkpoint 경로 규칙: `policy/ACT/act_ckpt/act-<task>/<task_config>-<EP_NUM>/<TRAIN_CONFIG>/` (기본 `EP_NUM=50`, `TRAIN_CONFIG=train_config`; 환경변수로 변경). 공개 checkpoint 는 `demo-50` 으로 학습됐으므로 **task_config 를 `demo` 로** 평가해야 경로가 맞는다.
- 평가 seed 는 기본 `1000000` 부터(공개 checkpoint 의 `log.log` 와 같은 구간 → 논문 수치와 직접 비교 가능).
- 결과: `eval_result/<policy>/<task>/<deploy>/<시각>/log.log`(에피소드별 성공/실패 + 누적 성공률), `metadata.json`, `video/`. 마지막 줄 `Total k/N (xx%) success.` 가 벤치마크 점수.
- 공개 checkpoint 의 원저자 결과(`data/checkpoints/<task>/*/log.log`)와 비교하면 환경이 논문과 동일한지 검증할 수 있다.
  **이 서버 검증(2026-09-22)**: `grasp_classify demo ACT/deploy --total_num 2` → seed 1000000/1000001 **2/2 성공(277/277 step)**, 원저자 기록 2/2 성공(273/269 step).
- 시간: 첫 에피소드까지 ~6분(모듈 import ~4분 + Task init 80s — 로그가 멈춘 듯 보여도 정상), 이후 에피소드당 35~70초. 100회 ≈ 1.5시간.

### 4-6. 자체 래퍼(워커)를 짤 때 알아둘 실제 동작 — 스펙 문서 절차로 검증(2026-09-22, `log/verify_spec.py`)
- `AppLauncher(headless=True, enable_cameras=True, livestream=0, device="cuda:0")` → `TaskCfg()` 오버라이드 → `Task(cfg, mode="eval")` → `reset(seed, instructions=[…])` → 촉각 검증 6항목 → `_get_observations()` → `take_action()` 까지 스펙과 전부 일치(40항목 PASS).
- **`check_early_stop()` 은 `True` 아니면 `None`** 을 돌려준다(insert_tube·insert_hole·lift_can·lift_bottle·pull_out_key·put_bottle_in_shelf 가 조건 불충족 시 `return` 없이 끝남). `check_success()` 는 `numpy.bool_`. → 소켓으로 보내기 전 `bool(...)` 로 정규화할 것.
- **`simulation_app.close()` 는 반환하지 않고 프로세스를 종료**시킨다(Kit shutdown 이 exit, `atexit` 도 안 돎, exit code 0). 종료 메시지 전송·파일 flush 등은 `close()` 호출 **전에** 끝낼 것.
- 첫 에피소드까지 ~5분 로그가 멈춘 듯 보이는 구간(cuRobo/warp import) 은 정상. `py-spy` 로 스택을 보려면 `docker exec --privileged` 가 필요하다.

### 4-7. 그 외
```bash
HEADLESS=1 python scripts/replay.py grasp_classify demo --gpu 0 --headless   # 수집 데이터 리플레이
python scripts/visualize.py grasp_classify demo 0 --task video              # 수집 hdf5 → 영상 (--task frame: 프레임 데이터 출력)
start_vnc.sh   # GUI 필요 시: 호스트에서 ssh -L 5901:localhost:5901 → VNC 뷰어 localhost:5901 (비밀번호 isaaclab)
stop_vnc.sh    # VNC 종료 (start_vnc.sh 를 다시 실행하면 알아서 죽이고 새로 띄우므로 평소엔 불필요)
```

#### VNC 로 "보이는" 것과 안 보이는 것 (2026-10-05 확인)
- **`eval_policy.py` 는 창을 띄우지 않는다** — 65행 `args_cli.livestream = 2` 가 upstream 에 하드코딩돼 있고, Isaac Lab 은 livestream 이 켜지면 headless 로 강제한다. VNC 터미널에서 평가를 돌려도 화면엔 아무것도 안 나오는 것이 정상.
- 창이 뜨는 것은 **`collect_data.py` + `render_frequency: 1` 인 설정(`demo`)** 뿐 (`clean` 은 render 0 → livestream=2 → 창 없음). `replay.py` 도 livestream=2 고정.
- 컨테이너 기본 `DISPLAY=:1` 이므로 VNC 를 안 띄운 채 GUI 모드(`demo`, HEADLESS 없이)로 돌리면 X 연결 실패로 죽는다 → 헤드리스 서버에선 항상 `HEADLESS=1`.
- VNC 는 수동 시작이다(컨테이너 재생성 시 자동으로 뜨지 않음). VNC 를 끄면(`stop_vnc.sh`, 로그아웃) 그 X 위에서 돌던 GUI 작업은 "X connection to :1 broken" 으로 함께 죽는다.
- GPU 를 다른 컨테이너가 포화 상태로 쓰고 있으면 reset 이 120 s 제한을 넘겨 `Timeout: reset exceed time limit` 으로 실패한다(실측: GPU0/1 모두 100% 일 때 188 s). 평가 전 `nvidia-smi` 로 빈 GPU 를 고를 것.
- 컨테이너 시간대는 `TZ`(기본 Asia/Seoul) 로 호스트와 맞춘다(2026-10-05 추가; 그 전 로그는 UTC = KST−9h).

---

## 5. 고정 버전 (스펙 문서 2026-09-18 과 동일)

| 구성 | 버전 | 설치 단계 |
|---|---|---|
| UniVTAC | `main` @ `0dafa10262e22f486f160a55d6f11aeab12d8e7b` (2026-09-07) | setup_host 가 checkout |
| Isaac Sim | 4.5.0.0 (pip `isaacsim[all,extscache]`) | 4 |
| Isaac Lab | v2.1.1 (= `isaaclab` 0.41.3), `third_party/IsaacLab` editable + 로컬 패치 1줄 | 5, 5b |
| Python | 3.10.12 (conda env `UniVTAC`) | 1 |
| PyTorch | 2.5.1+cu124, torchvision 0.20.1 | 3 (Isaac Lab 이 바꿔놓으면 되돌림) |
| NumPy | 1.26.4 | |
| cuRobo | v0.7.7 (`0a50de1`), warp-lang 1.0.0 | 6 |
| TacEx (번들, 수정본) | tacex / tacex_assets / tacex_tasks / tacex_uipc 0.1.0, torch_scatter 2.1.2+pt25cu124 | 7, 7b(에셋 복원) |
| libuipc 툴체인 | conda cmake 3.26 + cuda-toolkit 12.4.1, **시스템 gcc 11.4.0 + 시스템 ld** | 2 |
| vcpkg | `dd3097e` (upstream isaac51 핀), libuipc baseline `b2cb0da` | 8 |
| 기타 | transforms3d, trimesh, tetgen, modelscope | 9 |
| 베이스 이미지 | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` + Miniconda + TigerVNC/XFCE + Tailscale | |
| 검증 HW | RTX A6000 ×2 (sm_86), driver 550.107.02, Ubuntu 22.04 host | |

---

## 6. 이 환경을 만들며 해결한 문제 (전부 스크립트에 반영됨 — 재현 시 자동 처리)

| # | 증상 | 원인 | 반영 위치 |
|---|---|---|---|
| 1 | 컨테이너에서 Vulkan 이 GPU 를 못 잡음 (`vk_icdGetInstanceProcAddr` 오류) | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`·`/etc/vulkan/icd.d/nvidia_icd.json` 없음 — 공식 isaac-lab 이미지엔 구워져 있고 toolkit 은 주입 안 함 | Dockerfile 1-b, 1-c |
| 2 | 원본 `scripts/install.sh` 1단계 실패 | env.yaml 의 `gcc=11.4` 가 conda 채널에 없음(11.2 만) + sudo/livestream/vcpkg 조건 반전 버그 | 자체 설치 스크립트, 시스템 gcc 11.4 |
| 3 | conda deactivate 훅 `unbound variable` | `set -u` 와 gcc_linux-64 훅 충돌 | nounset 미사용 |
| 4 | vcpkg `no version database entry for cpptrace at 0.8.3` | vcpkg 를 baseline 커밋에 고정하면 버전 DB 가 너무 오래됨 | vcpkg `dd3097e` |
| 5 | vcpkg tinygltf `unexpected hash` | GitHub 이 아카이브를 재생성해 바이트가 바뀜(외부 요인) | overlay 포트 + 실제 해시 자동 갱신(`HASH_FIX_PORTS`) |
| 6 | libuipc `find_package(urdfdom)` 실패 | 실패한 이전 시도의 `build/vcpkg.json` 때문에 vcpkg 설치 건너뜀 | 9단계 시작 시 `build/` 초기화 |
| 7 | `Could not find prim …/Robot/gelpad_left` | **main 의 TacEx 에셋 114개가 Git LFS 포인터(131B)**(`a9abc0d remove git lfs due to quota`), LFS 서버에서도 못 받음 | 7b: `isaac51` 브랜치 blob 을 sha256 검증 후 복원 |
| 8 | pip 캐시 비활성 | host 마운트 uid 불일치 | entrypoint chown |
| 9 | 헤드리스에서 촉각 마커가 중앙으로 뭉치고 depth 상수, `SVD did not converge`, grasp 100% 실패(Plan True/Check False) | Isaac Lab 은 GUI 일 때만 `sim.enable_scene_query_support=True`. TacEx 의 gel pad 부착이 PhysX sweep 쿼리를 쓰므로 헤드리스에선 부착점이 NaN | 5b: `third_party/IsaacLab/…/simulation_context.py` 로컬 패치(기본 강제 ON, `UNIVTAC_SCENE_QUERY=0` 으로 해제). collect/eval/parallel 전부에 적용 |
| 10 | `collect_data.py --headless` → unrecognized argument | AppLauncher 인자 추가 전에 `parse_args()` | `HEADLESS=1 ENABLE_CAMERAS=1` 환경변수 사용 (eval/replay 는 `--headless` 가능) |
| 11 | 공개 checkpoint 를 받아도 `eval_policy.py` 가 못 찾음 | 다운로드 경로(`data/checkpoints/<task>/univtac`)와 코드 경로(`policy/ACT/act_ckpt/act-<task>/demo-50/train_config`)가 다름, 매핑 문서 없음 | `univtac_link_checkpoints.sh` |
| 12 | 재현 파일을 NAS→로컬로 옮기자 "Isaac Lab 이 없다" | run 스크립트가 "스크립트 폴더 옆의 IsaacLab" 을 `/workspace/isaaclab` 에 자동 마운트하던 암묵 규칙이 조용히 끊김 (참고용이라 실제론 불필요) | 자동 마운트 제거, `ISAACLAB_REF_SRC` 지정 시에만 `/workspace/isaaclab_ref:ro`. 기동 시 진짜 Isaac Lab 위치를 출력 |
| 13 | `univtac_setup_host.sh` 가 "nvidia-container-toolkit 없음" 으로 중단 | 이 서버는 `docker info` 에 nvidia 런타임이 안 보여도 `--gpus` 가 동작 (toolkit 1.17 CDI 방식) | 바이너리(`nvidia-container-cli`) 존재로 검사 |
| 14 | 마운트가 4개(리포 + 출력 3개)라 호스트에서 결과물 위치가 헷갈림 | 출력을 `univtac/outputs/` 로 분리했었으나 코드가 리포 안 상대경로를 쓰므로 굳이 분리할 이유가 없었음 | 출력을 리포 안으로 합침 → 데이터 마운트 2개 (§2) |
