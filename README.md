# univtac-jepa

[UniVTAC](https://github.com/univtac/UniVTAC)(Isaac Sim 4.5, `main@0dafa10`)에서 ACT policy의 **촉각 encoder만 바꿔**
성공률을 비교하기 위한 코드와 환경 설치 스크립트입니다. 비교 대상 encoder는 기존 UniVTAC encoder, ImageNet ResNet18,
Sparsh MAE / DINO / I-JEPA / V-JEPA입니다.

이 repo에는 **코드만** 들어 있습니다. UniVTAC 원본, conda env, 가중치, 데이터셋, 체크포인트, 영상은
각 서버의 로컬 디스크에 설치 스크립트가 받거나 학습·평가 중에 생성합니다.

## 새 서버에 설치 (한 번 실행)

```bash
git clone https://github.com/YuriPark0123/univtac-jepa.git /data/univtac/univtac-jepa   # 로컬 디스크 (NAS 불가)
bash /data/univtac/univtac-jepa/setup_host.sh
```

- **순서:** UniVTAC clone(`/data/univtac/UniVTAC`) → docker 이미지 → 컨테이너 `univtac` → 기본 설치(처음 1~2시간)
  → JEPA 추가 설치 → 테스트(약 3분, GPU 약 8GB)
- **다시 실행해도 안전합니다.** 이미 된 단계는 건너뜁니다. 단, **같은 이름의 컨테이너는 지우고 다시 만듭니다.**
- **상태만 검사:** `bash setup_host.sh --check`
- **환경변수:**
  - `UNIVTAC_LOCAL`: UniVTAC 상위 폴더. 기본값은 이 repo의 상위 폴더
  - `ISAACLAB_LOCAL`: env·캐시 상위 폴더. 기본값은 `$HOME/.isaaclab_vnc`
  - `CONTAINER_NAME`, `TEST_GPU`
- **호스트 전제조건:** NVIDIA 드라이버 ≥ 535, docker ≥ 24, nvidia-container-toolkit, GPU sm_86 검증(A6000·3090).
  자세한 내용은 [docker/README_univtac.md](docker/README_univtac.md) 참고.

## 무엇이 어디에 있나

| 구분 | 위치 (컨테이너 안) | 출처 |
|---|---|---|
| 이 repo (코드) | `/workspace/univtac_jepa` | git |
| UniVTAC 원본 | `/workspace/UniVTAC` | `deps.lock`의 commit |
| 우리 policy 코드 | `/workspace/UniVTAC/policy/{ACT_TacEnc,tactile_encoders}` → 이 repo로 심볼릭 링크 | 설치 스크립트 |
| Sparsh 코드 | `/workspace/UniVTAC/third_party/sparsh` | `deps.lock`의 commit |
| conda env | `UniVTAC`(기본), `UniVTAC-jepa`(= 기본 복제 + [requirements-jepa.txt](requirements-jepa.txt)) | 설치 스크립트 |
| 가중치·테스트 데이터 | `/workspace/UniVTAC/data/...` ([assets.tsv](assets.tsv), sha256 검증) | HF / modelscope |
| 전처리 데이터·체크포인트 | `/workspace/UniVTAC/data/act_tacenc/` | 학습 시 생성 |

## 사용

```bash
docker exec -it univtac bash
conda activate UniVTAC-jepa && cd /workspace/UniVTAC

bash data/download.sh --task insert_hole --version 45                       # 원본 데모 (task당 4~22GB)
bash policy/ACT_TacEnc/process_data.sh insert_hole isaac45 50               # → data/act_tacenc/sim-insert_hole/isaac45-50
bash policy/ACT_TacEnc/train.sh insert_hole isaac45 50 0 0 train_config_sparsh_ijepa
#    → data/act_tacenc/act_ckpt/act-insert_hole/isaac45-50/train_config_sparsh_ijepa

TRAIN_CONFIG=train_config_sparsh_ijepa EP_NUM=50 DATA_VERSION=isaac45 \
  python scripts/eval_policy.py insert_hole demo ACT_TacEnc/deploy --total_num 100 --headless
```

학습 설정(`policy/ACT_TacEnc/train_config_*.yml`)은 encoder 관련 키를 빼면 모두 공개 ACT `train_config.yml`과 같습니다.

| 설정 | `tactile_encoder` | 고정 | 비고 |
|---|---|---|---|
| `original` | UniVTAC `encoder.pth` (ResNet18) | 아니오 (lr 1e-5) | 공개 "univtac" 체크포인트와 같은 구성 |
| `original_frozen` | 〃 | 예 | |
| `vision` | (촉각 입력 없음) | — | 공개 "vision_only"와 같은 구성 |
| `resnet18_imagenet` | ImageNet ResNet18 | 아니오 (lr 1e-5) | |
| `sparsh_{mae,dino,ijepa,vjepa}` | Sparsh ViT-B | 예 | 768→512 projection만 추가 |

- **Sparsh 입력:** 공식 사전학습 파이프라인을 그대로 따릅니다(배경 차분, 90° 회전, 320×240, [0,1]).
  - MAE·DINO·I-JEPA는 [t, t-5] 6채널, V-JEPA는 [t, t-2, t-4, t-6] 프레임입니다.
  - 배경 프레임은 `tactile_background: first_frame | fixed`로 정합니다. 현재 기본값은 `first_frame`이고, 아직 미결정입니다.
- **프레임 간격:** 데이터는 2 sim step마다 저장되고 평가는 매 sim step마다 policy를 호출합니다.
  그래서 `deploy.yml`의 `eval_steps_per_data_frame: 2`로 과거 프레임 간격을 맞춥니다.

## 공개 ACT 코드 대비 수정 사항 (`policy/ACT`는 수정하지 않음)

- **학습 진입점:** `train.sh` → `imitate_episodes.py`가 argparse 에러로 시작되지 않던 문제를 고쳤습니다(`num_epochs` 전달).
- **전처리:**
  - 원본 경로를 `data/<isaac45|isaac51>/<task>`로 바꿨습니다.
  - head 카메라를 `cam_high`로 저장하고, joint는 앞 8차원만 씁니다(설정의 `state_dim` 8과 공개 통계에 맞춤).
  - 촉각 이미지는 원본 해상도 그대로 저장합니다.
- **경로:** encoder 체크포인트 경로를 실행 위치가 아니라 UniVTAC 루트 기준으로 해석합니다.
  기존 코드는 경로가 없으면 경고 없이 로딩을 건너뛰었습니다.

## 학습 서버 ↔ 평가 서버

학습과 평가를 다른 서버에서 합니다. 코드와 결과(작은 텍스트)는 git으로, 체크포인트(학습 1회당 약 383MB)는 공유 폴더(NAS 등)로 옮깁니다.

```
[학습 서버]  scripts/train_queue.sh → python scripts/record_runs.py → git add results/runs && git commit && git push
             python scripts/ckpt_transfer.py export <공유폴더>
[평가 서버]  git pull → python scripts/ckpt_transfer.py import <공유폴더>   (sha256이 results/runs 기록과 같아야 통과)
             tmux에서 scripts/eval_queue.sh → python scripts/compare.py → git add results && git commit && git push
```

- **공유 폴더:** 컨테이너 안에 공유 폴더가 마운트되어 있지 않으면, `ckpt_transfer.py`를 호스트에서
  `UNIVTAC_ROOT=<호스트의 UniVTAC 경로>`로 실행합니다.
- **평가 큐:** `scripts/eval_queue.sh`는 `scripts/eval_queue.tsv`를 위에서부터 처리합니다.
  - 한 항목의 100 seed를 `PROCS`개(기본 2) Isaac 프로세스로 나눠 `GPUS`(기본 `"0"`)에서 돌립니다. 프로세스 1개에 GPU 약 10GB가 필요합니다.
  - 공개 체크포인트(`public_univtac`, `public_vision_only`)는 없으면 자동으로 받습니다.
  - 중단돼도 다시 실행하면 남은 seed부터 이어서 합니다. **한 항목의 `PROCS`는 바꾸지 마세요**(바꾸면 `eval_run.py`가 거부합니다).
  - 예: `PROCS=4 GPUS="0 1" bash scripts/eval_queue.sh`

### 저장하는 것 (`results/`, git에 commit)

| 경로 | 내용 | 만드는 곳 |
|---|---|---|
| `protocol.json` | 평가 조건 고정: seed 1000000–1000099, `demo`, isaac45, 데모 50개, 같은 seed 재시도 2회, 기준선 `original` | 고정 |
| `runs/<task>/<train_config>.json` | 체크포인트·통계 파일 sha256, 학습 설정 전체, 사용한 원본 에피소드, repo commit, 학습 로그(시작·끝, 마지막 loss, val loss), 호스트 | 학습 서버 `record_runs.py` |
| `evals/<task>/<method>/per_seed.shard<i>of<n>.csv` | seed, 결과(success/failed/error), step, action 수, 시간, **시도 횟수**, error 내용 | 평가 서버 `eval_run.py` |
| `evals/<task>/<method>/eval.shard<i>of<n>.json` | 평가한 체크포인트 sha256, protocol sha256, repo·UniVTAC commit, Isaac 버전, 호스트, 시작 시 GPU 상태, 영상 위치 | 〃 |
| `reference/<task>/public_*/per_seed.csv` | 공개 체크포인트에 딸린 평가 로그를 같은 형식으로 변환한 것 (참고값) | `make_reference.py` (완료) |
| `tables/summary.md`, `summary.csv` | 비교표 | `compare.py` |

영상과 시뮬레이터 로그는 무거워서 각 서버의 `<UniVTAC>/data/act_tacenc/eval_raw/`(로컬)에만 남습니다.

### 비교 방법 (`scripts/compare.py`)

- **유효성:** 다음 중 하나라도 어긋나면 표에 ❌와 이유가 표시되고 Δ·p 계산에서 빠집니다.
  - protocol이 같은지, seed 100개가 모두 있고 중복이 없는지
  - 재시도 후에도 error인 seed가 없는지
  - shard 간 체크포인트가 같은지, 체크포인트 sha256이 `results/runs` 기록과 같은지
- **지표 (task별):**
  - 성공/n과 Wilson 95% 신뢰구간
  - 기준선 `original`(같은 파이프라인 재학습) 대비 Δ와 exact McNemar p (같은 seed 짝 비교)
  - 성공한 episode의 평균 action 수, 재시도한 seed 수
  - 공개 체크포인트를 다시 평가한 경우: 공개 로그와 seed별 성공/실패 일치 수
- **표 형식:** 행 = 방법, 열 = 위 지표 + 데이터 버전, 학습 step 수, encoder 고정 여부. 공개 로그 값은 "참고" 행으로 함께 보여 줍니다.
- **결론:** task가 5개뿐이라 task별로 내립니다.

## 테스트

```bash
bash scripts/univtac_install_jepa.sh --check      # 설치 상태 (GPU 안 씀)
TEST_GPU=0 bash tests/run_tests.sh                # encoder 6종, Sparsh 전처리 == 공식, 학습 1 step, deploy 재로드
```

## 주의

- **라이선스:** Sparsh 코드와 가중치는 **CC-BY-NC 4.0**(비상업 연구용)입니다. 이 repo에는 포함하지 않고 설치 시 받습니다.
- **비밀번호:** `docker/`의 VNC 기본 비밀번호(`isaaclab`)가 하드코딩되어 있으므로 repo를 **private으로 유지**하세요.
- **평가 seed:** `scripts/eval_policy.py`는 reset이 120초(실제 경과 시간)를 넘으면 그 seed를 error로 처리하고 **다음 seed로 넘어갑니다.**
  GPU를 다른 작업과 나눠 쓰면 이런 일이 생기고, 그러면 비교 seed 집합이 달라집니다.
