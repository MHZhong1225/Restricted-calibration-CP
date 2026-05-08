#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/stoCP_${TS}.log"

PYTHONWARNINGS=ignore python -u run.py \
  --methods sgcp \
  --dataset-mode mimic \
  --grid-json json/grid_mimic.json \
  --sweep-name sgcp_mimic_sweep \
  --outdir results_vis/sgcp_mimic_delay \
  --mimic-label-col icu_delay_class \
  --alpha 0.1 \
  --cuda 1 \
  > "$LOG"

echo "stdout log saved to: $LOG"
