#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/stoCP_${TS}.log"

PYTHONWARNINGS=ignore python -u run.py \
  --methods sgcp \
  --dataset-mode syn \
  --grid-json json/grid_syn.json \
  --outdir results_vis/sgcp_syn_run \
  --alpha 0.1 \
  --cuda 0 \
  > "$LOG"

echo "stdout log saved to: $LOG"
