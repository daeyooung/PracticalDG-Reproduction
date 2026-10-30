#!/bin/bash
# PACS 하이브리드 DG 실험 1회: 학습(train.py) -> acc/AUROC 평가(eval.py) -> H-score 평가(eval_hscore.py)
#
# 사용법: bash scripts/run_pacs.sh <ALGO> <CLASS_SPACE> <TEST_ENV>
#   ALGO        : CLIPBase | SCIPD                      (기본 SCIPD)
#   CLASS_SPACE : 0 | 1 | 2 | 3  -> hybridness H = 0, 1/6, 1/3, 1 (기본 3)
#   TEST_ENV    : 0 art_painting | 1 cartoon | 2 photo | 3 sketch (기본 0)
# 환경변수(선택): EPOCHS(30) STEPS(100) GPU(0) SEED(0) LOSS(SCI: IP|SI|SCI) OUT_ROOT(./output/pacs)
#
# 결과: $OUT_ROOT/PACS_class_space{CS}/{ALGO}/test_envs_{TEST_ENV}/seed_{SEED}/
#   done.txt              valid acc / target acc / target auroc
#   eval/eval.txt         acc 상세
#   eval/eval_h_score.txt max_h-score / mean_h-score
set -e
cd "$(dirname "$0")/.."

ALGO=${1:-SCIPD}
CS=${2:-3}
TEST_ENV=${3:-0}
EPOCHS=${EPOCHS:-30}
STEPS=${STEPS:-100}
GPU=${GPU:-0}
SEED=${SEED:-0}
LOSS=${LOSS:-SCI}
OUT_ROOT=${OUT_ROOT:-./output/pacs}

dataset=PACS
data_dir=/Dataset/PACS/      # 저장소 루트 기준 경로 (<repo>/Dataset/PACS), 끝의 '/' 필요
output=${OUT_ROOT}/${dataset}_class_space${CS}/${ALGO}/test_envs_${TEST_ENV}/seed_${SEED}

ARGS="--seed $SEED --lr 1e-3 --data_dir $data_dir --domain_num 4 --max_epoch $EPOCHS --steps_per_epoch $STEPS \
 --net resnet18 --output $output --test_envs $TEST_ENV --dataset $dataset --algorithm $ALGO --gpu_id $GPU \
 --loader_name daml_loader --is_data_aug yes --is_different_class_space $CS --loss_type $LOSS"

echo "[run_pacs] $ALGO class_space=$CS test_env=$TEST_ENV epochs=$EPOCHS -> $output"
python train.py $ARGS
python eval.py $ARGS
python eval_hscore.py $ARGS
echo "[run_pacs] done: $output"
