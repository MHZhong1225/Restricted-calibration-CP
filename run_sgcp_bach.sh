#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/stoCP_bach_${TS}.log"

PYTHONWARNINGS=ignore python -u run.py \
  --methods sgcp \
  --dataset-mode bach \
  --grid-json json/grid_bach.json \
  --outdir results_vis/sgcp_bach \
  --alpha 0.1 \
  --cuda 1 \
  > "$LOG"

echo "stdout log saved to: $LOG"