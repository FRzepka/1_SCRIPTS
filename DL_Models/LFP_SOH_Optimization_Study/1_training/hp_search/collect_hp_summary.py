#!/usr/bin/env python3
"""Collect comparison table for all *_hp base models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", default="/home/florianr/MG_Farm/1_Scripts")
    args = ap.parse_args()

    root = Path(args.workspace_root).resolve()
    models_root = root / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models"
    out_root = root / "DL_Models" / "LFP_SOH_Optimization_Study" / "1_training" / "hp_search" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary in sorted(models_root.glob("*/Base/*_hp/hpo_summary.json")):
        with open(summary, "r", encoding="utf-8") as f:
            d = json.load(f)
        p = d.get("best_params", {})
        rows.append(
            {
                "model_key": d.get("model_key"),
                "family": d.get("family"),
                "version_hp": d.get("version_hp"),
                "best_val_mae": d.get("best_val_mae"),
                "best_val_rmse": d.get("best_val_rmse"),
                "best_objective": d.get("best_objective"),
                "target_mae": d.get("target_mae"),
                "params": d.get("params"),
                "param_size_kb_float32": d.get("param_size_kb_float32"),
                "checkpoint_size_kb": d.get("checkpoint_size_kb"),
                "seq_chunk_size": p.get("seq_chunk_size"),
                "window_stride": p.get("window_stride"),
                "batch_size": p.get("batch_size"),
                "lr": p.get("lr"),
                "weight_decay": p.get("weight_decay"),
                "dropout": p.get("dropout"),
                "architecture": d.get("architecture"),
                "summary_path": str(summary),
            }
        )

    rows.sort(key=lambda x: (x["family"] or "", x["version_hp"] or ""))
    fields = [
        "model_key",
        "family",
        "version_hp",
        "best_val_mae",
        "best_val_rmse",
        "best_objective",
        "target_mae",
        "params",
        "param_size_kb_float32",
        "checkpoint_size_kb",
        "seq_chunk_size",
        "window_stride",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "architecture",
        "summary_path",
    ]

    csv_path = out_root / "hp_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_root / "hp_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HP Search Summary\n\n")
        f.write("| model_key | MAE | RMSE | params | float32 KB | ckpt KB | seq | stride | batch |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model_key']} | {r['best_val_mae']:.6f} | {r['best_val_rmse']:.6f} | "
                f"{int(r['params']) if r['params'] is not None else 'nan'} | "
                f"{float(r['param_size_kb_float32']):.1f} | {float(r['checkpoint_size_kb']):.1f} | "
                f"{r['seq_chunk_size']} | {r['window_stride']} | {r['batch_size']} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if rows:
        print("\nTop models by RMSE:")
        for r in sorted(rows, key=lambda x: x["best_val_rmse"])[:4]:
            print(
                f"  {r['model_key']}: MAE={float(r['best_val_mae']):.6f}, "
                f"RMSE={float(r['best_val_rmse']):.6f}, params={r['params']}"
            )


if __name__ == "__main__":
    main()
