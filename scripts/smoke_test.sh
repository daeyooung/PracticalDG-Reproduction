#!/bin/bash
# 설치 / 데이터 / 코드가 끝까지 동작하는지 빠르게 확인한다 (1 epoch x 5 step, GPU 기준 수 분).
# 결과는 ./output/smoke 아래에 저장되며 실제 실험 결과(./output/pacs)와 섞이지 않는다.
set -e
cd "$(dirname "$0")/.."

python check_setup.py

export EPOCHS=1 STEPS=5 OUT_ROOT=./output/smoke
bash scripts/run_pacs.sh CLIPBase 3 0
bash scripts/run_pacs.sh SCIPD 3 0

echo
echo "===== smoke test passed ====="
echo "CLIPBase:"; cat ./output/smoke/PACS_class_space3/CLIPBase/test_envs_0/seed_0/done.txt
echo "SCIPD:";    cat ./output/smoke/PACS_class_space3/SCIPD/test_envs_0/seed_0/done.txt
