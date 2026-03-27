#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"
FT_SCRIPT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/3_pruning/finetune_pruned_soh_model.py"
LOG_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/3_pruning/logs_finetune"

PRUNE_TAG="${1:-p50}"          # p20 or p50
EPOCHS="${2:-8}"
LR="${3:-0.0002}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/${TS}_${PRUNE_TAG}"
mkdir -p "$LOG_DIR"

declare -a FAMILIES=("LSTM" "TCN" "GRU" "CNN")
declare -a VERSIONS=("0.1.2.3" "0.2.2.1" "0.3.1.1" "0.4.1.1")

for i in "${!FAMILIES[@]}"; do
  FAM="${FAMILIES[$i]}"
  VER="${VERSIONS[$i]}"

  MODEL_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models/$FAM/Pruned/${VER}_hp_${PRUNE_TAG}"
  CKPT="$MODEL_DIR/checkpoints/best_model_pruned.pt"
  OUT_DIR="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models/$FAM/PrunedFT/${VER}_hp_${PRUNE_TAG}_ft"
  LOG_FILE="$LOG_DIR/${FAM}_${VER}.log"
  SESSION="soh_ft_${FAM,,}_${VER//./_}_${PRUNE_TAG}_${TS}"

  CMD="cd '$ROOT' && \
    '$PY' '$FT_SCRIPT' \
      --model-dir '$MODEL_DIR' \
      --out-dir '$OUT_DIR' \
      --ckpt '$CKPT' \
      --epochs '$EPOCHS' \
      --lr '$LR' \
      --weight-decay 0.0 \
      --device cuda \
      --val-interval 1 \
      --early-stopping 4 \
      --num-workers 2 \
      --prefetch-factor 2 \
      --head-only \
      |& tee -a '$LOG_FILE'"

  screen -dmS "$SESSION" bash -lc "$CMD"
  echo "$SESSION -> $LOG_FILE"
done

cat > "$LOG_DIR/SESSIONS.txt" <<EOF
timestamp: $TS
prune_tag: $PRUNE_TAG
epochs: $EPOCHS
lr: $LR
mode: head_only
EOF

screen -ls | grep "soh_ft_.*_${PRUNE_TAG}_${TS}" || true
echo "Logs: $LOG_DIR"
