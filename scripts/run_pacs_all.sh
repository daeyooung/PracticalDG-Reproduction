#!/bin/bash
# PACS 재현 실험 전체: CLIPBase vs SCIPD x hybridness 4단계 (class_space 0~3) 를 한 target 도메인에서 비교하고,
# 결과를 CSV / 그래프로 정리한다 (논문 Fig. 4 에 해당).
#
# 사용법: bash scripts/run_pacs_all.sh [TEST_ENV]   (기본 0 = art_painting)
# 환경변수(선택): EPOCHS(30) STEPS(100) GPU(0) SEED(0)
set -e
cd "$(dirname "$0")/.."

TEST_ENV=${1:-0}
export OUT_ROOT=./output/pacs

for ALGO in CLIPBase SCIPD; do
    for CS in 0 1 2 3; do
        bash scripts/run_pacs.sh "$ALGO" "$CS" "$TEST_ENV"
    done
done

mkdir -p results/pacs
python visualize/compare_clipbase_scipd.py \
    --root_dir ./output/pacs \
    --plot_dir ./results/pacs \
    --csv_path ./results/pacs/pacs_compare_test_envs_${TEST_ENV}.csv

echo "[run_pacs_all] results -> ./results/pacs"
