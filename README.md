# PracticalDG 재현: Hybrid Domain Generalization (CVPR 2024)

논문 **"PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization"**
(Zining Chen et al., CVPR 2024, [arXiv:2404.09011](https://arxiv.org/abs/2404.09011)) 을
저자 공식 코드 [znchen666/HDG](https://github.com/znchen666/HDG) 를 바탕으로 **재현·검증**한 저장소입니다.

- 논문이 제안한 **SCI-PD** (Score / Class / Instance 수준 perturbation distillation) 와 비교 기준 **CLIPBase** 를
  PACS 데이터셋에서 직접 학습·평가하여 성능을 비교했습니다.
- 공식 코드를 **다른 환경에서도 바로 실행되도록** 정리하고(절대 경로 제거, 클래스 이름 분리, 실행 스크립트·환경 점검 스크립트 추가),
  결과 집계·시각화 코드를 Python 으로 작성했습니다.
- 실험 로그와 결과(CSV, 그림)는 [`results/pacs/`](results/pacs) 에 포함되어 있습니다.

---

## 1. 문제 설정: Hybrid Domain Generalization

- **Domain Generalization (DG)**: 여러 소스 도메인(예: 사진, 그림, 스케치)으로 학습한 모델을 **학습에 쓰지 않은 타깃 도메인**에서 평가합니다.
- **Hybrid DG**: 여기에 더해 (1) 소스 도메인마다 **클래스 공간이 다를 수 있고**, (2) 타깃 도메인에는 학습 때 보지 못한 **unknown 클래스**가 섞여 있습니다.
  소스 도메인 간 클래스 공간이 얼마나 다른지는 **hybridness H** 로 표현합니다 (코드의 `--is_different_class_space 0/1/2/3` = H 0, 1/6, 1/3, 1).
- **평가 지표**
  - `target acc`: 타깃 도메인 known 클래스 정확도
  - `target auroc`: known / unknown 구분 성능 (AUROC)
  - `H-score`: known 정확도와 unknown 정확도의 조화평균 (threshold 를 바꿔 가며 구한 최댓값)
- **비교 방법**
  - `CLIPBase`: CLIP ViT-B/16 (teacher) 의 zero-shot 유사도를 soft label 로 사용해 ResNet-18 (student) 을 distillation
  - `SCIPD`: 논문 제안 방법. 위 distillation 에 Score / Class / Instance 수준의 perturbation 을 더해 unknown 에 강인하게 학습

## 2. 재현 결과 (PACS, target = art_painting)

소스 3개 도메인(cartoon, photo, sketch) 으로 학습하고 art_painting 으로 평가했습니다.
PACS 7개 클래스 중 6개(dog, elephant, giraffe, guitar, horse, house) 를 known, `person` 을 unknown 으로 둡니다.

설정: ResNet-18, lr 1e-3, **30 epoch × 100 step**, seed 0, 단일 GPU. (논문·공식 스크립트의 기본값보다 짧게 줄인 설정이므로 절대 수치는 논문과 다를 수 있습니다.)

| hybridness H | CLIPBase acc | **SCIPD acc** | CLIPBase AUROC | **SCIPD AUROC** | CLIPBase H-score | **SCIPD H-score** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0   | 0.4053 | **0.4271** | 0.6724 | **0.7004** | 0.3875 | **0.4108** |
| 1/6 | 0.5485 | **0.5916** | 0.5614 | **0.6151** | **0.4921** | 0.4915 |
| 1/3 | 0.7567 | **0.7955** | 0.7065 | **0.7849** | 0.6412 | **0.6747** |
| 1   | 0.8449 | **0.8705** | 0.7918 | **0.8491** | 0.7221 | **0.7439** |

<p align="center"><img src="results/pacs/domain_0.png" width="520" alt="CLIPBase vs SCIPD H-score on PACS (target: art_painting)"></p>

- **모든 hybridness 에서 SCIPD 가 CLIPBase 보다 target acc 와 AUROC 가 높고**, H-score 도 4개 구간 중 3개에서 높아
  "perturbation distillation 이 unknown 이 섞인 타깃 도메인에서 CLIP distillation 베이스라인보다 강인하다"는 논문의 주장과 같은 방향의 결과를 얻었습니다.
- H = 1/6 에서 H-score 차이(0.0006)는 단일 seed 의 오차 범위로, 유의미한 차이로 보기 어렵습니다.
- 집계 결과: [`results/pacs/pacs_compare_test_envs_0.csv`](results/pacs/pacs_compare_test_envs_0.csv). 학습 로그와 체크포인트(`output/`, run 당 약 800MB)는 용량 문제로 저장소에 포함하지 않았고, 각 run 의 `done.txt` / `eval_h_score.txt` 요약을 위 표에 옮겼습니다.

## 3. 저장소 구성

### 3-1. 직접 작성·수정한 코드

| 파일 | 내용 |
|---|---|
| `scripts/run_pacs.sh` | PACS 실험 1회 (train → eval → H-score). 알고리즘 / hybridness / 타깃 도메인을 인자로 받음 |
| `scripts/run_pacs_all.sh` | CLIPBase vs SCIPD × hybridness 4단계 8회 실행 후 CSV·그림 자동 생성 |
| `scripts/smoke_test.sh` | 설치·데이터·코드가 끝까지 도는지 1 epoch 로 빠르게 확인 |
| `check_setup.py` | 패키지 버전, GPU, 데이터셋 폴더와 이미지 리스트 존재 여부를 점검 |
| `utils/class_names.py` | 알고리즘 코드에 하드코딩돼 있던 OfficeHome 클래스 이름을 분리하고 PACS 지원 추가. CLIP 텍스트 프롬프트 토큰화 함수 |
| `dalib/vision/datasets/pacs.py`, `officehome.py` | 절대 경로(`/home/user/...`) 를 저장소 루트 기준 상대 경로로 바꾸고, 도메인 라벨을 경로에서 추출하도록 정리 |
| `alg/algs/CLIPBase.py`, `alg/algs/SCIPD.py` | 공통 모듈(`utils/class_names.py`) 을 쓰도록 리팩터링 (손실 계산 로직은 원본과 동일) |
| `visualize/compare_clipbase_scipd.py` | `done.txt`, `eval_h_score.txt` 를 모아 hybridness 별 acc / H-score 비교 CSV·그래프 생성 (논문 Fig. 4) |
| `visualize/visualize_hyper.py`, `scripts/run_hyper.sh` | 하이퍼파라미터(α, β, τ) 민감도 실험·시각화 (논문 Fig. 5) |
| `visualize/summarize_loss.py`, `scripts/run_loss.sh` | 손실 구성요소(IP / SI / SCI) ablation 집계 (논문 Table 5) |
| `eval_hscore_val.py`, `scripts/run_val.sh`, `visualize/threshold_val_*.py` | 테스트 셋이 아닌 **검증 셋**으로 unknown threshold 를 정하는 현실적 설정 평가 |
| `visualize/compute_h2_cv.py` | hybridness 별 H-score 평균과 변동계수 계산 |
| `image_list/OfficeHome/split.py` | OfficeHome train / val 이미지 리스트 분할 |
| `requirements.txt`, `.gitignore`, `README.md` | 환경 명세 및 문서 |

### 3-2. 원본(HDG) 코드

`train.py`, `eval.py`, `eval_hscore.py`, `alg/`, `network/`, `datautil/`, `dalib/`, `utils/util.py` 등 학습·평가 파이프라인은
[znchen666/HDG](https://github.com/znchen666/HDG) 의 코드이며, 아래 "원본 코드 대비 수정 사항"에 적은 부분 외에는 그대로 사용했습니다. 라이선스는 원본과 같은 GPL-3.0 입니다.

## 4. 환경 설정

검증한 환경: Python 3.10, PyTorch 2.9.1 (CUDA 12.8), torchvision 0.24.1, numpy 1.26.4, 단일 NVIDIA GPU.

```bash
conda create -n hdg python=3.10 -y
conda activate hdg
# PyTorch 는 CUDA 버전에 맞게 https://pytorch.org 안내대로 먼저 설치
pip install -r requirements.txt     # OpenAI CLIP 포함
```

## 5. 데이터 준비

PACS: [DomainDrop 저장소 안내](https://github.com/lingeringlight/DomainDrop?tab=readme-ov-file) /
OfficeHome: [공식 페이지](https://www.hemanthdv.org/officeHomeDataset.html) 에서 내려받아 저장소 루트의 `Dataset/` 아래에 둡니다
(다른 위치에 있으면 `ln -s /path/to/PACS Dataset/PACS` 처럼 심볼릭 링크로 연결해도 됩니다).

```
Dataset/
├── PACS/
│   ├── art_painting/{dog,elephant,...}/*.jpg
│   ├── cartoon/ ...
│   ├── photo/ ...
│   └── sketch/ ...
└── OfficeHome/
    ├── Art/ ...  Clipart/ ...  Product/ ...  Real_World/ ...
```

이미지 목록은 `image_list/{PACS,OfficeHome}/` 에 포함되어 있습니다. 준비가 끝나면 아래로 점검합니다.

```bash
python check_setup.py                       # PACS
python check_setup.py --dataset office-home
```

## 6. 실행 방법

### 6-1. 빠른 동작 확인 (수 분)

```bash
bash scripts/smoke_test.sh
```

환경 점검 후 CLIPBase / SCIPD 를 1 epoch 씩 돌려 `output/smoke/` 에 결과를 남깁니다.

### 6-2. PACS 재현 실험 (위 2절의 결과)

```bash
# 전체: CLIPBase vs SCIPD x hybridness 0~3, target = art_painting (GPU 1장 기준 약 1시간 20분)
bash scripts/run_pacs_all.sh 0

# 1회만: <ALGO> <CLASS_SPACE 0~3> <TEST_ENV 0~3>
EPOCHS=30 bash scripts/run_pacs.sh SCIPD 3 0
```

| TEST_ENV | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 타깃 도메인 | art_painting | cartoon | photo | sketch |

결과는 `output/pacs/PACS_class_space{H}/{ALGO}/test_envs_{E}/seed_{S}/` 에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `done.txt` | valid acc, target acc, target AUROC, 학습 시간 |
| `eval/eval.txt` | 정확도 상세 |
| `eval/eval_h_score.txt` | threshold 별 H-score, max / mean H-score |

`run_pacs_all.sh` 는 끝나면 `visualize/compare_clipbase_scipd.py` 를 실행해
`results/pacs/pacs_compare_test_envs_0.csv` 와 `results/pacs/domain_0.png` 를 만듭니다. 따로 실행하려면:

```bash
python visualize/compare_clipbase_scipd.py --root_dir ./output/pacs --plot_dir ./results/pacs \
    --csv_path ./results/pacs/pacs_compare_test_envs_0.csv
```

### 6-3. OfficeHome 추가 실험 스크립트

아래 스크립트는 OfficeHome(50 epoch) 기준으로 작성되어 있으며, 이 저장소에는 결과를 포함하지 않았습니다.
`dataset` / `data_dir` 변수를 바꾸면 PACS 에도 사용할 수 있습니다.

```bash
bash scripts/run.sh               # 전체 도메인 성능 (논문 Table 1, 2 / Fig. 4) - 스크립트 안의 알고리즘·class space 배열 조정
bash scripts/run_loss.sh IP       # 손실 ablation: IP | SI | SCI (논문 Table 5)  -> visualize/summarize_loss.py
bash scripts/run_hyper.sh alpha   # 하이퍼파라미터 민감도: alpha | beta | tau (논문 Fig. 5) -> visualize/visualize_hyper.py
bash scripts/run_val.sh           # 검증 셋 기반 threshold 평가 -> visualize/threshold_val_test_user.py
```

## 7. 원본 코드 대비 수정 사항

| 항목 | 원본 | 수정 |
|---|---|---|
| 데이터 경로 | `dalib/vision/datasets/*.py` 에 저자 PC 절대 경로가 하드코딩 | 저장소 루트 기준 `resolve_path()` 로 통일. `--data_dir /Dataset/PACS/` 는 `<repo>/Dataset/PACS` 로 해석 |
| 클래스 이름 | `CLIPBase.py`, `SCIPD.py` 안에 OfficeHome 65개 클래스가 직접 나열되어 PACS 에서 동작하지 않음 | `utils/class_names.py` 로 분리, 데이터셋별 known 클래스 제공 |
| GPU 지정 | `os.environ["CUDA_VISIBLE_DEVICS"]` (오타로 무효) | 제거하고 `--gpu_id` 로 `cuda:{id}` 를 직접 선택 |
| 학습 요약 | `done.txt` 에 epoch / epoch 당 시간이 음수로 기록 | 실제 수행 epoch 수 기준으로 계산 |
| 실행 편의 | 긴 argparse 명령을 매번 입력 | `scripts/run_pacs*.sh`, `smoke_test.sh`, `check_setup.py` 추가 |
| 저장소 위생 | `__pycache__`, 체크포인트가 섞임 | `.gitignore` 로 `Dataset/`, `output/`, `*.pkl` 등 제외 |

## License / Citation

원본 코드와 동일하게 [GPL-3.0](LICENSE) 을 따릅니다.

```bibtex
@inproceedings{chen2024practicaldg,
  title={PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization},
  author={Chen, Zining and Wang, Weiqiu and Zhao, Zhicheng and Su, Fei and Men, Aidong and Meng, Hongying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23501--23511},
  year={2024}
}
```
