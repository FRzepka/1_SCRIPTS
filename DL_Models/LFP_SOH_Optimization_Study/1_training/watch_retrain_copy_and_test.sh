#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
TRAIN_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/1_training"
MODELS_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/2_models"
TEST_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study/6_test"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"

RETRAIN_TS="${1:-20260216_090801}"
POST_TS="$(date +%Y%m%d_%H%M%S)"
POST_LOG_DIR="$TRAIN_ROOT/retrain_logs/$RETRAIN_TS/postprocess_$POST_TS"
TEST_OUT_DIR="$TEST_ROOT/BASE_MODELS_RETRAIN_${RETRAIN_TS}_$POST_TS"
mkdir -p "$POST_LOG_DIR" "$TEST_OUT_DIR"

POST_LOG="$POST_LOG_DIR/postprocess.log"
exec > >(tee -a "$POST_LOG") 2>&1

echo "[INFO] Retrain TS: $RETRAIN_TS"
echo "[INFO] Test output: $TEST_OUT_DIR"

declare -a SPECS=(
  "LSTM|0.1.2.4|0.1.2.3"
  "TCN|0.2.2.2|0.2.2.1"
  "GRU|0.3.1.2|0.3.1.1"
  "CNN|0.4.1.3|0.4.1.2"
)

# 1) Wait until all retrain screen sessions are gone
while screen -ls | grep -q "soh_retrain_.*_${RETRAIN_TS}"; do
  echo "[WAIT] $(date '+%F %T') retrain screens still running..."
  screen -ls | grep "soh_retrain_.*_${RETRAIN_TS}" || true
  sleep 120
done
echo "[INFO] All retrain screens finished."

# 2) Copy best artifacts into 2_models/Base/<new_version>
COPIED_LIST="$POST_LOG_DIR/copied_models.csv"
echo "family,version,src_run,best_ckpt,dst_base" > "$COPIED_LIST"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r FAMILY VER TEST_SRC_VER <<< "$spec"
  echo "[COPY] Processing $FAMILY $VER"

  SRC_VER_DIR="$TRAIN_ROOT/$VER"
  OUT_ROOT="$SRC_VER_DIR/outputs/soh"
  PREFERRED_RUN="$OUT_ROOT/retrain_${RETRAIN_TS}"

  SRC_RUN="$($PY - <<PY
from pathlib import Path
out_root = Path(r'''$OUT_ROOT''')
pref = Path(r'''$PREFERRED_RUN''')
if pref.exists():
    print(pref)
else:
    runs = [p for p in out_root.iterdir() if p.is_dir()] if out_root.exists() else []
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(runs[0] if runs else '')
PY
)"

  if [[ -z "$SRC_RUN" || ! -d "$SRC_RUN" ]]; then
    echo "[WARN] Missing run dir for $FAMILY $VER, skipping."
    continue
  fi

  BEST_CKPT="$($PY - <<PY
from pathlib import Path
import re
run_dir = Path(r'''$SRC_RUN''')
ckpt_dir = run_dir / 'checkpoints'
if not ckpt_dir.exists():
    print('')
    raise SystemExit(0)
best = list(ckpt_dir.glob('best_epoch*_rmse*.pt'))
if best:
    def rmse(p):
        m = re.search(r'rmse([0-9]+(?:\.[0-9]+)?)', p.name)
        return float(m.group(1)) if m else float('inf')
    best.sort(key=rmse)
    print(best[0])
else:
    final = ckpt_dir / 'final_model.pt'
    if final.exists():
        print(final)
    else:
        any_ckpt = sorted(ckpt_dir.glob('*.pt'))
        print(any_ckpt[0] if any_ckpt else '')
PY
)"

  if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "[WARN] Missing best checkpoint for $FAMILY $VER, skipping."
    continue
  fi

  SCALER="$SRC_RUN/scaler_robust.joblib"
  if [[ ! -f "$SCALER" ]]; then
    echo "[WARN] Missing scaler for $FAMILY $VER ($SCALER), skipping."
    continue
  fi

  DST_BASE="$MODELS_ROOT/$FAMILY/Base/$VER"
  mkdir -p "$DST_BASE/checkpoints" "$DST_BASE/config" "$DST_BASE/scripts" "$DST_BASE/test"

  cp -f "$SRC_VER_DIR/config/train_soh.yaml" "$DST_BASE/config/train_soh.yaml"
  cp -f "$SRC_VER_DIR/scripts/train_soh.py" "$DST_BASE/scripts/train_soh.py"
  cp -f "$TRAIN_ROOT/$TEST_SRC_VER/test/plot_true_vs_pred.py" "$DST_BASE/test/plot_true_vs_pred.py"
  cp -f "$SCALER" "$DST_BASE/scaler_robust.joblib"
  cp -f "$BEST_CKPT" "$DST_BASE/checkpoints/$(basename "$BEST_CKPT")"
  cp -f "$BEST_CKPT" "$DST_BASE/checkpoints/best_model.pt"

  cat > "$DST_BASE/source_run.txt" <<EOF
family=$FAMILY
version=$VER
source_run=$SRC_RUN
best_ckpt=$BEST_CKPT
copied_at=$(date '+%F %T')
EOF

  echo "$FAMILY,$VER,$SRC_RUN,$BEST_CKPT,$DST_BASE" >> "$COPIED_LIST"
  echo "[COPY] Done $FAMILY $VER -> $DST_BASE"
done

# 3) Run per-model tests (same style as before: timeseries+scatter+metrics per cell/group)
TEST_RUNS_CSV="$POST_LOG_DIR/test_runs.csv"
echo "family,version,test_out,run_dir" > "$TEST_RUNS_CSV"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r FAMILY VER TEST_SRC_VER <<< "$spec"
  DST_BASE="$MODELS_ROOT/$FAMILY/Base/$VER"
  PLOT_PY="$DST_BASE/test/plot_true_vs_pred.py"
  CFG="$DST_BASE/config/train_soh.yaml"

  if [[ ! -f "$PLOT_PY" || ! -f "$CFG" ]]; then
    echo "[WARN] Missing test/config for $FAMILY $VER, skipping tests."
    continue
  fi

  # Use the run dir recorded in source_run.txt (fallback to retrain_<ts>)
  RUN_DIR="$TRAIN_ROOT/$VER/outputs/soh/retrain_${RETRAIN_TS}"
  if [[ -f "$DST_BASE/source_run.txt" ]]; then
    SRC_LINE="$(grep '^source_run=' "$DST_BASE/source_run.txt" || true)"
    if [[ -n "$SRC_LINE" ]]; then
      RUN_DIR="${SRC_LINE#source_run=}"
    fi
  fi

  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[WARN] Run dir missing for $FAMILY $VER ($RUN_DIR), skipping tests."
    continue
  fi

  OUT_SUB="$TEST_OUT_DIR/${FAMILY}_${VER}_base"
  mkdir -p "$OUT_SUB"

  echo "[TEST] $FAMILY $VER -> $OUT_SUB"
  "$PY" "$PLOT_PY" \
    --config "$CFG" \
    --run-dir "$RUN_DIR" \
    --group all \
    --device cuda \
    --out-dir "$OUT_SUB" \
    > "$OUT_SUB/run.log" 2>&1

  echo "$FAMILY,$VER,$OUT_SUB,$RUN_DIR" >> "$TEST_RUNS_CSV"
done

# 4) Build summary tables (architecture + size + metrics by group)
"$PY" - <<PY
from pathlib import Path
import pandas as pd
import yaml
import json
import torch
import re

root = Path(r'''$ROOT''')
test_out = Path(r'''$TEST_OUT_DIR''')
specs = [
  ('LSTM','0.1.2.4'),
  ('TCN','0.2.2.2'),
  ('GRU','0.3.1.2'),
  ('CNN','0.4.1.3'),
]

rows = []
group_rows = []

for family, ver in specs:
    model_tag = f"{family}_{ver}_base"
    mdir = root / 'DL_Models' / 'LFP_SOH_Optimization_Study' / '2_models' / family / 'Base' / ver
    tdir = test_out / model_tag
    metrics_csv = tdir / 'metrics.csv'
    cfg_path = mdir / 'config' / 'train_soh.yaml'
    ckpt_path = mdir / 'checkpoints' / 'best_model.pt'

    if not (metrics_csv.exists() and cfg_path.exists() and ckpt_path.exists()):
        continue

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get('model_state_dict', state)
    if isinstance(sd, dict):
        params = int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel')))
    else:
        params = 0

    float32_kb = (params * 4.0) / 1024.0
    ckpt_kb = ckpt_path.stat().st_size / 1024.0

    mdf = pd.read_csv(metrics_csv)
    if mdf.empty:
        continue

    by_group = mdf.groupby('group', dropna=False)[['rmse','mae','r2']].mean().reset_index()
    grp = {r['group']: r for _, r in by_group.iterrows()}

    if family == 'LSTM':
        arch = (
            f"LSTM(embed={cfg['model'].get('embed_size')}, hidden={cfg['model'].get('hidden_size')}, "
            f"layers={cfg['model'].get('num_layers')}, res={cfg['model'].get('res_blocks')}, "
            f"drop={float(cfg['model'].get('dropout',0.0)):.3f})"
        )
    elif family == 'GRU':
        arch = (
            f"GRU(embed={cfg['model'].get('embed_size')}, hidden={cfg['model'].get('hidden_size')}, "
            f"layers={cfg['model'].get('num_layers')}, res={cfg['model'].get('res_blocks')}, "
            f"drop={float(cfg['model'].get('dropout',0.0)):.3f})"
        )
    elif family == 'TCN':
        arch = (
            f"TCN(hidden={cfg['model'].get('hidden_size')}, mlp={cfg['model'].get('mlp_hidden')}, "
            f"k={cfg['model'].get('kernel_size')}, dil={cfg['model'].get('dilations')}, "
            f"drop={float(cfg['model'].get('dropout',0.0)):.3f})"
        )
    else:
        arch = (
            f"CNN(hidden={cfg['model'].get('hidden_size')}, mlp={cfg['model'].get('mlp_hidden')}, "
            f"k={cfg['model'].get('kernel_size')}, dil={cfg['model'].get('dilations')}, "
            f"drop={float(cfg['model'].get('dropout',0.0)):.3f})"
        )

    rows.append({
        'model': model_tag,
        'params': params,
        'float32_weights_kb': float32_kb,
        'ckpt_kb': ckpt_kb,
        'test_mae': float(grp.get('test', {}).get('mae', float('nan'))),
        'test_rmse': float(grp.get('test', {}).get('rmse', float('nan'))),
        'test_r2': float(grp.get('test', {}).get('r2', float('nan'))),
        'arch': arch,
    })

    group_rows.append({
        'model': model_tag,
        'train_rmse': float(grp.get('train', {}).get('rmse', float('nan'))),
        'val_rmse': float(grp.get('val', {}).get('rmse', float('nan'))),
        'test_rmse': float(grp.get('test', {}).get('rmse', float('nan'))),
        'train_mae': float(grp.get('train', {}).get('mae', float('nan'))),
        'val_mae': float(grp.get('val', {}).get('mae', float('nan'))),
        'test_mae': float(grp.get('test', {}).get('mae', float('nan'))),
    })

if rows:
    sdf = pd.DataFrame(rows).sort_values('test_mae', na_position='last')
    gdf = pd.DataFrame(group_rows)
    sdf.to_csv(test_out / 'architecture_and_test_summary_by_group.csv', index=False)
    gdf.to_csv(test_out / 'group_means.csv', index=False)

    md = []
    md.append('# Retrain Model Summary (Architecture + Size + Metrics)')
    md.append('')
    md.append('| model | params | float32 KB | ckpt KB | test MAE | test RMSE | test R2 | arch |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for _, r in sdf.iterrows():
        md.append(
            f"| {r['model']} | {int(r['params'])} | {float(r['float32_weights_kb']):.1f} | {float(r['ckpt_kb']):.1f} | "
            f"{float(r['test_mae']):.6f} | {float(r['test_rmse']):.6f} | {float(r['test_r2']):.4f} | {r['arch']} |"
        )
    md.append('')
    md.append('## Group Means')
    md.append('')
    md.append('| model | train RMSE | val RMSE | test RMSE | train MAE | val MAE | test MAE |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for _, r in gdf.iterrows():
        md.append(
            f"| {r['model']} | {float(r['train_rmse']):.6f} | {float(r['val_rmse']):.6f} | {float(r['test_rmse']):.6f} | "
            f"{float(r['train_mae']):.6f} | {float(r['val_mae']):.6f} | {float(r['test_mae']):.6f} |"
        )

    (test_out / 'architecture_and_test_summary_by_group.md').write_text('\n'.join(md), encoding='utf-8')

readme = [
    '# Retrain Copy + Test',
    '',
    '- Source retrain timestamp: `$RETRAIN_TS`',
    '- Models copied to `2_models/<Family>/Base/<version>`:',
    '  - LSTM/0.1.2.4',
    '  - TCN/0.2.2.2',
    '  - GRU/0.3.1.2',
    '  - CNN/0.4.1.3',
    '- Tests run with `plot_true_vs_pred.py --group all` per model.',
]
(test_out / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
print(f'[INFO] Summary written to: {test_out}')
PY

echo "[DONE] Postprocess complete."
echo "[DONE] Artifacts in: $TEST_OUT_DIR"
