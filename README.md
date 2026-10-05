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
