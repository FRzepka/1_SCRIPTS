#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"

PRUNE_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/3_pruning"
QUANT_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/4_quantize"
LOG_ROOT="$PRUNE_DIR/logs"

PRUNE_AMOUNT="${1:-0.5}"
PRUNE_PCT="$($PY - <<PY
amt = float("${PRUNE_AMOUNT}")
print(int(round(amt * 100.0)))
PY
)"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$TS"
mkdir -p "$LOG_DIR"

declare -a FAMILIES=("LSTM" "TCN" "GRU" "CNN")
declare -a VERSIONS=("0.1.2.3" "0.2.2.1" "0.3.1.1" "0.4.1.1")

for i in "${!FAMILIES[@]}"; do
  FAM="${FAMILIES[$i]}"
  VER="${VERSIONS[$i]}"

  BASE_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models/$FAM/Base/${VER}_hp"
  CKPT="$BASE_DIR/checkpoints/best_model.pt"
  PRUNED_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models/$FAM/Pruned/${VER}_hp_p${PRUNE_PCT}"
  PRUNED_CKPT="$PRUNED_DIR/checkpoints/best_model_pruned.pt"
  QUANT_OUT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models/$FAM/Quantized/${VER}_hp_p${PRUNE_PCT}_int8"

  LOG="$LOG_DIR/${FAM}_${VER}.log"
  SESSION="soh_pq_${FAM,,}_${VER//./_}_${TS}"
  CMD="cd '$ROOT' && \
    '$PY' '$PRUNE_DIR/prune_soh_model.py' \
      --model-dir '$BASE_DIR' \
      --out-dir '$PRUNED_DIR' \
      --amount '$PRUNE_AMOUNT' \
      --ckpt '$CKPT' && \
    '$PY' '$QUANT_DIR/quantize_soh_model.py' \
      --model-dir '$PRUNED_DIR' \
      --out-dir '$QUANT_OUT' \
      --ckpt '$PRUNED_CKPT' \
    |& tee -a '$LOG'"

  screen -dmS "$SESSION" bash -lc "$CMD"
  echo "$SESSION -> $LOG"
done

cat > "$LOG_DIR/SESSIONS.txt" <<EOF
timestamp: $TS
prune_amount: $PRUNE_AMOUNT
prune_pct: $PRUNE_PCT
EOF

screen -ls | grep "soh_pq_.*_${TS}" || true
echo "Logs: $LOG_DIR"
