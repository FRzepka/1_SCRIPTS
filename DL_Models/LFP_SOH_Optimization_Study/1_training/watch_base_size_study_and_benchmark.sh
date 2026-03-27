#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
STUDY_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study"
TRAIN_ROOT="$STUDY_ROOT/1_training"
BENCH_ROOT="$STUDY_ROOT/5_benchmark/SOH_Comparison_Base"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"

TS="${1:?missing training timestamp}"
POST_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$TRAIN_ROOT/base_size_study/watch_logs/$TS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/watch_$POST_TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[info] waiting for size-study screens with ts=$TS"
while screen -ls | grep "soh_sizes_.*_${TS}" | grep -vq "soh_sizes_watch_${TS}"; do
  date '+[wait] %F %T size-study screens still running'
  screen -ls | grep "soh_sizes_.*_${TS}" | grep -v "soh_sizes_watch_${TS}" || true
  sleep 180
done

echo "[info] all size-study screens finished"
"$PY" "$TRAIN_ROOT/finalize_base_size_study.py" --ts "$TS"
"$PY" "$BENCH_ROOT/run_benchmark_base_size_study.py" --study-ts "$TS"
echo "[done] finalize + benchmark completed for ts=$TS"
