#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/florianr/MG_Farm/1_Scripts"
LOG="$ROOT/tools/model_optimization/logs/soc1710_v5_after_v4_2026-03-29.log"
mkdir -p "$(dirname "$LOG")"
log(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1" >> "$LOG"; }
screen_exists(){ screen -ls 2>/dev/null | grep -q "$1"; }
log "waiting for v4 GPU work to clear"
while screen_exists ft_struct30_paper || screen_exists benchv4_pipeline; do
  log "still waiting: ft_struct30_paper or benchv4_pipeline active"
  sleep 60
done
log "starting larger SOC workflow (treat as v5)"
screen -S soc1710_v6 -X quit >/dev/null 2>&1 || true
screen -dmS soc1710_v5 bash -lc 'cd /home/florianr/MG_Farm/1_Scripts && PYTHONUNBUFFERED=1 /home/florianr/anaconda3/envs/ml1/bin/python tools/model_optimization/run_soc1710_hpt_train_optimize_v6.py >> /home/florianr/MG_Farm/1_Scripts/tools/model_optimization/logs/soc1710_v5_2026-03-29.log 2>&1'
log "launched soc1710_v5"
