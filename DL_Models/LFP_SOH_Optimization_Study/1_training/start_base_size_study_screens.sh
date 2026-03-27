#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florianr/MG_Farm/1_Scripts"
STUDY_ROOT="$ROOT/DL_Models/LFP_SOH_Optimization_Study"
TRAIN_ROOT="$STUDY_ROOT/1_training"
PY="/home/florianr/anaconda3/envs/ml1/bin/python"
SPECS_JSON="$STUDY_ROOT/base_size_study_specs.json"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$TRAIN_ROOT/base_size_study/logs/$TS"
mkdir -p "$LOG_DIR"

"$PY" "$TRAIN_ROOT/prepare_base_size_study.py"

export STUDY_ROOT TRAIN_ROOT PY SPECS_JSON TS LOG_DIR

"$PY" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

study_root = Path(os.environ["STUDY_ROOT"])
train_root = Path(os.environ["TRAIN_ROOT"])
py = os.environ["PY"]
specs_json = Path(os.environ["SPECS_JSON"])
ts = os.environ["TS"]
log_dir = Path(os.environ["LOG_DIR"])

with open(specs_json, "r", encoding="utf-8") as f:
    specs = json.load(f)

sessions = []
for idx, family_spec in enumerate(specs["families"]):
    family = family_spec["family"]
    family_lower = family.lower()
    family_log = log_dir / f"{family}.log"
    commands = ["set -euo pipefail", f"cd '{Path('/home/florianr/MG_Farm/1_Scripts')}'", f"sleep {idx * 20}"]
    for variant in family_spec["variants"]:
        if variant["role"] == "base":
            continue
        version = variant["version"]
        tag = variant["tag"]
        cfg = train_root / version / "config" / "train_soh.yaml"
        script = train_root / version / "scripts" / "train_soh.py"
        run_id = f"size_{ts}_{tag}"
        commands.append(
            f"echo '[start] {family} {version} {tag}' | tee -a '{family_log}'"
        )
        commands.append(
            f"CUDA_VISIBLE_DEVICES=0 '{py}' '{script}' --config '{cfg}' --run-id '{run_id}' |& tee -a '{family_log}'"
        )
    session = f"soh_sizes_{family_lower}_{ts}"
    full_cmd = " && ".join(commands)
    subprocess.run(["screen", "-dmS", session, "bash", "-lc", full_cmd], check=True)
    sessions.append({"family": family, "session": session, "log": str(family_log)})

with open(log_dir / "SESSIONS.txt", "w", encoding="utf-8") as f:
    f.write(f"timestamp: {ts}\n")
    f.write(f"specs: {specs_json}\n")
    f.write(f"test_cell: {specs['test_cell']}\n")
    for item in sessions:
        f.write(f"{item['session']} -> {item['log']}\n")

for item in sessions:
    print(f"{item['session']} -> {item['log']}")
print(f"Logs: {log_dir}")
PY

screen -ls | grep "soh_sizes_.*_${TS}" || true
