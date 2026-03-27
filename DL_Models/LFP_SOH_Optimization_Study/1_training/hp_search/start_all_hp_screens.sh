#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
HP_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/1_training/hp_search"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"

N_TRIALS="${1:-30}"
MAX_EPOCHS="${2:-60}"
EARLY_STOP="${3:-12}"
VAL_INTERVAL="${4:-3}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$HP_DIR/logs/$TS"
mkdir -p "$LOG_DIR"

TARGET_MAE="$($PY - <<'PY'
import re
from pathlib import Path
import torch

root = Path('/home/florianr/MG_Farm/1_Scripts')
bases = [
    root / 'DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Base/0.1.2.3/checkpoints',
    root / 'DL_Models/LFP_SOH_Optimization_Study/2_models/TCN/Base/0.2.2.1/checkpoints',
    root / 'DL_Models/LFP_SOH_Optimization_Study/2_models/GRU/Base/0.3.1.1/checkpoints',
    root / 'DL_Models/LFP_SOH_Optimization_Study/2_models/CNN/Base/0.4.1.1/checkpoints',
]
maes = []
for ckpt_dir in bases:
    if not ckpt_dir.exists():
        continue
    cands = sorted(ckpt_dir.glob('best_epoch*_rmse*.pt'))
    if not cands:
        continue
    def score(p):
        m = re.search(r'rmse([0-9]+(?:\.[0-9]+)?)', p.name)
        return float(m.group(1)) if m else float('inf')
    best = sorted(cands, key=score)[0]
    state = torch.load(best, map_location='cpu')
    metrics = state.get('metrics', {})
    mae = metrics.get('mae')
    if mae is not None:
        maes.append(float(mae))
print(f"{sum(maes)/len(maes):.8f}" if maes else "0.02")
PY
)"

echo "Target MAE (from current base checkpoints): $TARGET_MAE"

declare -a KEYS=(
  "LSTM_0.1.2.3"
  "TCN_0.2.2.1"
  "GRU_0.3.1.1"
  "CNN_0.4.1.1"
)

for KEY in "${KEYS[@]}"; do
  SHORT="$(echo "$KEY" | tr '[:upper:].' '[:lower:]_')"
  SESSION="soh_hp_${SHORT}_${TS}"
  LOG="$LOG_DIR/${KEY}.log"
  CMD="cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$HP_DIR/run_optuna_hpo.py' \
    --model-key '$KEY' \
    --n-trials '$N_TRIALS' \
    --max-epochs '$MAX_EPOCHS' \
    --early-stopping '$EARLY_STOP' \
    --val-interval '$VAL_INTERVAL' \
    --device cuda \
    --num-workers 2 \
    --prefetch-factor 2 \
    --target-mae '$TARGET_MAE' \
    --mae-penalty-weight 0.5 \
    |& tee -a '$LOG'"
  screen -dmS "$SESSION" bash -lc "$CMD"
  echo "$SESSION -> $LOG"
done

cat > "$LOG_DIR/SESSIONS.txt" <<EOF
timestamp: $TS
n_trials: $N_TRIALS
max_epochs: $MAX_EPOCHS
early_stopping: $EARLY_STOP
val_interval: $VAL_INTERVAL
target_mae: $TARGET_MAE
EOF

screen -ls | grep "soh_hp_.*_${TS}" || true
echo "Logs: $LOG_DIR"
