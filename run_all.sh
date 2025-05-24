#!/usr/bin/env bash
cd "$(dirname "$0")"
set -e
DATA_DIR="data/EuroSAT_RGB"

echo "=== 1. Train ==="
python src/train.py --root $DATA_DIR

echo "=== 2. Evaluate ==="
python src/evaluate.py --root $DATA_DIR \
       --weights results/best_eurosat_resnet50.pt

echo "=== 3. Grad-CAM ==="
python src/cam.py --root $DATA_DIR \
       --weights results/best_eurosat_resnet50.pt --num 5
