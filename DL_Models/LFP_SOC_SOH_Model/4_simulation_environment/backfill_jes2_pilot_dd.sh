#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/florianr/MG_Farm/1_Scripts
ROOT="$WORKSPACE/DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
CAMPAIGN_ROOT="$ROOT/campaigns"
PYTHON=/home/florianr/anaconda3/envs/ml1/bin/python
CELLS=(C09 C13 C15 C25 C27 C29)
DATE_TAG=20260825
POST_LOG="$CAMPAIGN_ROOT/jes2_six_cell_pilot_dd_backfill_${DATE_TAG}.log"

cd "$WORKSPACE"
export LD_LIBRARY_PATH="/home/florianr/anaconda3/envs/ml1/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# Preserve each complete original shard before the DD-only runner rewrites its manifest.
while screen -ls 2>/dev/null | grep -Eq '[.]jes2_pilot_C(09|13|15|25|27|29)_20260825'; do
    sleep 60
done
while screen -ls 2>/dev/null | grep -q '.jes2_pilot_merge_plot_20260825'; do
    sleep 30
done

for cell in "${CELLS[@]}"; do
    campaign="$CAMPAIGN_ROOT/jes2_six_cell_pilot_${cell}_${DATE_TAG}"
    if ! grep -q 'PILOT_EXIT_STATUS=0' "$campaign/pilot.log"; then
        printf 'Original shard %s did not finish successfully.\n' "$cell" >> "$POST_LOG"
        exit 1
    fi
    cp "$campaign/jes2_manifest.json" "$campaign/jes2_manifest_pre_dd.json"
    screen -dmS "jes2_pilot_dd_${cell}_${DATE_TAG}" bash -lc \
        "cd '$WORKSPACE' && export LD_LIBRARY_PATH='/home/florianr/anaconda3/envs/ml1/lib':\${LD_LIBRARY_PATH:-} && export CUDA_VISIBLE_DEVICES=0 && '$PYTHON' '$ROOT/run_jes2_benchmark.py' --cells '$cell' --models DD --aliases initial_soc_error --stochastic_repeats 3 --tag 'jes2_six_cell_pilot_${cell}_${DATE_TAG}' --trace_device cuda --model_device cuda --skip_existing >> '$campaign/dd_initial_backfill.log' 2>&1; status=\$?; printf '\nDD_BACKFILL_EXIT_STATUS=%s\n' \"\$status\" >> '$campaign/dd_initial_backfill.log'"
done

while screen -ls 2>/dev/null | grep -Eq '[.]jes2_pilot_dd_C(09|13|15|25|27|29)_20260825'; do
    sleep 60
done

manifests=()
for cell in "${CELLS[@]}"; do
    campaign="$CAMPAIGN_ROOT/jes2_six_cell_pilot_${cell}_${DATE_TAG}"
    if ! grep -q 'DD_BACKFILL_EXIT_STATUS=0' "$campaign/dd_initial_backfill.log"; then
        printf 'DD backfill %s did not finish successfully.\n' "$cell" >> "$POST_LOG"
        exit 1
    fi
    manifests+=("$campaign/jes2_manifest.json")
done
for cell in "${CELLS[@]}"; do
    manifests+=("$CAMPAIGN_ROOT/jes2_six_cell_pilot_${cell}_${DATE_TAG}/jes2_manifest_pre_dd.json")
done

"$PYTHON" "$ROOT/merge_jes2_manifests.py" \
    --manifests "${manifests[@]}" \
    --out "$CAMPAIGN_ROOT/jes2_six_cell_pilot_merged_${DATE_TAG}.json" \
    --tag "jes2_six_cell_pilot_merged_${DATE_TAG}" > "$POST_LOG" 2>&1

"$PYTHON" "$ROOT/results/build_jes2_paper_results.py" \
    --manifest "$CAMPAIGN_ROOT/jes2_six_cell_pilot_merged_${DATE_TAG}.json" \
    --out_dir "$WORKSPACE/LATEX/JES/paper_robustness_benchmark/JES_2.0/pilot_results" \
    --figures_dir "$WORKSPACE/LATEX/JES/paper_robustness_benchmark/figures/Results/JES2_Pilot" \
    --bootstrap_samples 5000 >> "$POST_LOG" 2>&1

printf '\nDD_BACKFILL_POSTPROCESS_EXIT_STATUS=0\n' >> "$POST_LOG"
