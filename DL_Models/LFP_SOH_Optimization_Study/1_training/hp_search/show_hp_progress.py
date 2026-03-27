#!/usr/bin/env python3
"""Show latest HP progress snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", default="/home/florianr/MG_Farm/1_Scripts")
    args = ap.parse_args()

    root = Path(args.workspace_root).resolve()
    runs_root = root / "DL_Models" / "LFP_SOH_Optimization_Study" / "1_training" / "hp_search" / "runs"
    files = sorted(runs_root.glob("*/progress.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("No progress files found.")
        return

    latest_by_model = {}
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("model_key")
            if key and key not in latest_by_model:
                latest_by_model[key] = (p, data)
        except Exception:
            continue

    for key in sorted(latest_by_model.keys()):
        p, d = latest_by_model[key]
        print(
            f"{key}: trial={d.get('last_trial_number')} state={d.get('last_trial_state')} "
            f"best_trial={d.get('best_trial_number')} best_score={d.get('best_value')} "
            f"best_rmse={d.get('best_val_rmse')} best_mae={d.get('best_val_mae')}"
        )
        print(f"  {p}")


if __name__ == "__main__":
    main()
