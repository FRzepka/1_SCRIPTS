#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/florianr/MG_Farm/1_Scripts
ROOT="$WORKSPACE/DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
CAMPAIGN_ROOT="$ROOT/campaigns"
PYTHON=/home/florianr/anaconda3/envs/ml1/bin/python
CELLS=(C09 C13 C15 C25 C27 C29)
DATE_TAG=20260825
SCHEDULER_LOG="$CAMPAIGN_ROOT/jes2_full_scheduler_${DATE_TAG}.log"
POST_LOG="$CAMPAIGN_ROOT/jes2_full_postprocess_${DATE_TAG}.log"
WINDOW_MANIFEST="$WORKSPACE/LATEX/JES/paper_robustness_benchmark/JES_2.0/tables/jes2_evaluation_windows.csv"

cd "$WORKSPACE"
export LD_LIBRARY_PATH="/home/florianr/anaconda3/envs/ml1/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
exec >> "$SCHEDULER_LOG" 2>&1

printf 'Waiting for the DD all-scenario frozen-window pilot.\n'
while screen -ls 2>/dev/null | grep -q '.jes2_dd_windows_alltests_pilot_20260825'; do
    sleep 60
done

if ! grep -q 'DD_ALLTESTS_PILOT_WALL=' \
    "$CAMPAIGN_ROOT/jes2_dd_windows_alltests_pilot_${DATE_TAG}.screen.log"; then
    printf 'DD frozen-window pilot did not finish successfully; full campaign aborted.\n'
    exit 1
fi

printf 'Re-running automated tests before full campaign.\n'
"$PYTHON" -m pytest -q "$ROOT/tests"

AUDIT_TAG="jes2_full_protocol_prelaunch_audit_${DATE_TAG}"
printf 'Expanding and auditing the complete protocol.\n'
"$PYTHON" "$ROOT/run_jes2_benchmark.py" \
    --cells C29 \
    --tag "$AUDIT_TAG" \
    --lstm_publish_intervals 1 6 24 \
    --window_manifest "$WINDOW_MANIFEST" \
    --trace_device cuda \
    --model_device cuda \
    --dry_run > "$CAMPAIGN_ROOT/${AUDIT_TAG}.log" 2>&1

"$PYTHON" - "$CAMPAIGN_ROOT/$AUDIT_TAG/jes2_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
runs = manifest["runs"]
keys = [
    (run["cell"], run["window_id"], run["alias"], run["seed"], run.get("soh_condition", "none"), run["model"])
    for run in runs
]
aliases = manifest["protocol"]["scenarios"]
initial_models = {run["model"] for run in runs if run["alias"] == "initial_soc_error"}
if len(aliases) != 19:
    raise SystemExit(f"Expected 19 scenarios, found {len(aliases)}")
if len(runs) != 1260 or len(keys) != len(set(keys)):
    raise SystemExit(f"Invalid per-cell plan: runs={len(runs)}, unique={len(set(keys))}")
if initial_models != {"DM", "HDM", "HECM", "DD"}:
    raise SystemExit(f"Invalid initialization models: {sorted(initial_models)}")
if any(run["status"] != "dry_run" for run in runs):
    raise SystemExit("Protocol audit contains a non-dry-run record")
print("Protocol audit passed: 19 scenarios, three C29 windows, 1260 unique model runs.")
PY

SMOKE_TAG="jes2_full_all_scenario_smoke_${DATE_TAG}"
printf 'Running all 19 scenarios through every estimator on the frozen C27 window.\n'
"$PYTHON" "$ROOT/run_jes2_benchmark.py" \
    --cells C27 \
    --tag "$SMOKE_TAG" \
    --stochastic_repeats 1 \
    --secondary_stochastic_repeats 1 \
    --window_manifest "$WINDOW_MANIFEST" \
    --trace_device cuda \
    --model_device cuda \
    --skip_existing > "$CAMPAIGN_ROOT/${SMOKE_TAG}.log" 2>&1

"$PYTHON" "$ROOT/results/build_jes2_paper_results.py" \
    --manifest "$CAMPAIGN_ROOT/$SMOKE_TAG/jes2_manifest.json" \
    --out_dir "$CAMPAIGN_ROOT/$SMOKE_TAG/results" \
    --figures_dir "$CAMPAIGN_ROOT/$SMOKE_TAG/figures" \
    --bootstrap_samples 100 >> "$CAMPAIGN_ROOT/${SMOKE_TAG}.log" 2>&1
printf 'All-scenario real-data smoke and strict result build passed.\n'

available_kb=$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')
if (( available_kb < 20 * 1024 * 1024 )); then
    printf 'Less than 20 GiB free; full campaign aborted.\n'
    exit 1
fi

printf 'Launching six full cell shards.\n'
for cell in "${CELLS[@]}"; do
    tag="jes2_full_${cell}_${DATE_TAG}"
    campaign="$CAMPAIGN_ROOT/$tag"
    mkdir -p "$campaign"
    screen -dmS "jes2_full_${cell}_${DATE_TAG}" bash -lc \
        "cd '$WORKSPACE' && export LD_LIBRARY_PATH='/home/florianr/anaconda3/envs/ml1/lib':\${LD_LIBRARY_PATH:-} && export CUDA_VISIBLE_DEVICES=0 && '$PYTHON' '$ROOT/run_jes2_benchmark.py' --cells '$cell' --tag '$tag' --stochastic_repeats 10 --secondary_stochastic_repeats 5 --lstm_publish_intervals 1 6 24 --window_manifest '$WINDOW_MANIFEST' --trace_device cuda --model_device cuda --skip_existing >> '$campaign/full.log' 2>&1; status=\$?; printf '\nFULL_EXIT_STATUS=%s\n' \"\$status\" >> '$campaign/full.log'"
done

while screen -ls 2>/dev/null | grep -Eq '[.]jes2_full_C(09|13|15|25|27|29)_20260825'; do
    sleep 120
done

manifests=()
for cell in "${CELLS[@]}"; do
    campaign="$CAMPAIGN_ROOT/jes2_full_${cell}_${DATE_TAG}"
    if ! grep -q 'FULL_EXIT_STATUS=0' "$campaign/full.log"; then
        printf 'Full shard %s did not finish successfully; merge aborted.\n' "$cell"
        exit 1
    fi
    manifests+=("$campaign/jes2_manifest.json")
done

printf 'Merging complete shards and building publication outputs.\n'
"$PYTHON" "$ROOT/merge_jes2_manifests.py" \
    --manifests "${manifests[@]}" \
    --out "$CAMPAIGN_ROOT/jes2_full_holdout_merged_${DATE_TAG}.json" \
    --tag "jes2_full_holdout_${DATE_TAG}" > "$POST_LOG" 2>&1

"$PYTHON" "$ROOT/results/build_jes2_paper_results.py" \
    --manifest "$CAMPAIGN_ROOT/jes2_full_holdout_merged_${DATE_TAG}.json" \
    --out_dir "$WORKSPACE/LATEX/JES/paper_robustness_benchmark/JES_2.0/results" \
    --figures_dir "$WORKSPACE/LATEX/JES/paper_robustness_benchmark/figures/Results" \
    --bootstrap_samples 10000 >> "$POST_LOG" 2>&1

printf '\nFULL_POSTPROCESS_EXIT_STATUS=0\n' >> "$POST_LOG"
printf 'Full JES2 campaign and publication result build completed successfully.\n'
