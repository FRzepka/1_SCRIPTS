#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
TRAIN_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/1_training"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$TRAIN_ROOT/retrain_logs/$TS"
mkdir -p "$LOG_DIR"

runs=(
  "LSTM 0.1.2.4"
  "TCN 0.2.2.2"
  "GRU 0.3.1.2"
  "CNN 0.4.1.3"
)

echo "timestamp: $TS" > "$LOG_DIR/SESSIONS.txt"
echo "split: C19 in train | test: C11,C23,C29" >> "$LOG_DIR/SESSIONS.txt"

for entry in "${runs[@]}"; do
  family="${entry%% *}"
  ver="${entry##* }"
  low_family="$(echo "$family" | tr '[:upper:]' '[:lower:]')"
  session="soh_retrain_${low_family}_${ver//./_}_${TS}"
  cfg="$TRAIN_ROOT/$ver/config/train_soh.yaml"
  script="$TRAIN_ROOT/$ver/scripts/train_soh.py"
  log="$LOG_DIR/${family}_${ver}.log"
  cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$script' --config '$cfg' --run-id 'retrain_${TS}' |& tee -a '$log'"
  screen -dmS "$session" bash -lc "$cmd"
  echo "$session -> $log" | tee -a "$LOG_DIR/SESSIONS.txt"
done

echo
echo "Started sessions:"
screen -ls | grep "soh_retrain_.*_${TS}" || true
echo "Logs: $LOG_DIR"
