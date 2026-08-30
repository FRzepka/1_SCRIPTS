from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robustness_common import COMMON_EVALUATION_START_SAMPLE, compute_common_recovery_metrics


MODEL_FILES = {
    "DM": ("no_soh", "soc_cc_fullcell_{cell}.csv", "soc_cc"),
    "HDM": ("lstm_h1", "soc_cc_soh_fullcell_{cell}.csv", "soc_cc"),
    "HECM": ("lstm_h1", "ecm_soc_fullcell_{cell}.csv", "soc_ecm"),
    "DD": ("lstm_h1", "soc_pred_fullcell_{cell}.csv", "soc_pred"),
}


def load_prediction(path: Path, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["index", "time_s", prediction_column])
    return frame.rename(columns={prediction_column: "prediction"})


def infer_event_time(frame: pd.DataFrame) -> float:
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    source_index = frame["index"].to_numpy(dtype=np.float64)
    dt = np.diff(time_s) / np.diff(source_index)
    finite = dt[np.isfinite(dt) & (dt > 0.0)]
    nominal_dt = float(np.median(finite)) if len(finite) else 1.0
    return float(time_s[0] - source_index[0] * nominal_dt)


def hierarchical_cell_ci(frame: pd.DataFrame, column: str, samples: int, seed: int) -> dict:
    cell_means = frame.groupby("cell", sort=True)[column].mean()
    values = cell_means.to_numpy(dtype=np.float64)
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_cells": int(len(values)),
        "n_windows": int(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical paired JES2 initial-state recovery analysis.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--sustain_seconds", type=float, default=300.0)
    parser.add_argument("--horizon_seconds", type=float, default=86_400.0)
    parser.add_argument(
        "--evaluation_start_sample", type=int, default=COMMON_EVALUATION_START_SAMPLE
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for cell_dir in sorted(path for path in args.runs.iterdir() if path.is_dir()):
        cell = cell_dir.name
        for window_dir in sorted(path for path in cell_dir.iterdir() if path.is_dir()):
            for model, (condition, filename, prediction_column) in MODEL_FILES.items():
                baseline_path = window_dir / "baseline" / "seed_42" / condition / model / filename.format(cell=cell)
                perturbed_path = window_dir / "initial_soc_error" / "seed_42" / condition / model / filename.format(cell=cell)
                if not baseline_path.is_file() or not perturbed_path.is_file():
                    continue
                baseline = load_prediction(baseline_path, prediction_column)
                perturbed = load_prediction(perturbed_path, prediction_column)
                event_time_s = infer_event_time(baseline)
                baseline = baseline[baseline["index"] >= args.evaluation_start_sample]
                perturbed = perturbed[perturbed["index"] >= args.evaluation_start_sample]
                paired = baseline.merge(
                    perturbed, on=["index", "time_s"], suffixes=("_baseline", "_perturbed")
                )
                metrics = compute_common_recovery_metrics(
                    paired["time_s"].to_numpy(dtype=np.float64),
                    paired["prediction_baseline"].to_numpy(dtype=np.float64),
                    paired["prediction_perturbed"].to_numpy(dtype=np.float64),
                    start_index=0,
                    threshold=args.threshold,
                    sustain_seconds=args.sustain_seconds,
                    horizon_seconds=args.horizon_seconds,
                    event_time_s=event_time_s,
                )
                rows.append({
                    "cell": cell,
                    "window_id": window_dir.name,
                    "model": model,
                    "paired_samples": int(len(paired)),
                    "evaluation_start_sample": int(args.evaluation_start_sample),
                    "initial_trajectory_difference": metrics["common_recovery_initial_abs_err"],
                    "recovery_time_h": metrics["common_recovery_time_h"],
                    "recovery_or_censor_time_h": metrics["common_recovery_or_censor_time_h"],
                    "recovery_censored": metrics["common_recovery_censored"],
                    "recovery_excess_auc_soc_h": metrics["common_recovery_excess_auc_soc_h"],
                    "recovery_relapsed_after_first_hold": metrics["common_recovery_relapsed"],
                })

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise RuntimeError("No paired recovery trajectories were found")
    aggregate_rows = []
    for model, group in runs.groupby("model", sort=False):
        for metric in [
            "recovery_or_censor_time_h",
            "recovery_excess_auc_soc_h",
            "recovery_censored",
            "recovery_relapsed_after_first_hold",
        ]:
            aggregate_rows.append({
                "model": model,
                "metric": metric,
                **hierarchical_cell_ci(group, metric, args.bootstrap_samples, seed=7310 + len(aggregate_rows)),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "jes2_paired_initial_recovery_runs.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(
        args.out_dir / "jes2_paired_initial_recovery_statistics.csv", index=False
    )
    (args.out_dir / "jes2_paired_initial_recovery_method.txt").write_text(
        "Endpoint: absolute difference between a perturbed estimator trajectory and its cell-, window-, "
        "model-, SOH-, and seed-matched correctly initialized trajectory. Recovery is the first entry into "
        "a 0.02 SOC band that remains inside for at least 300 seconds. Runs without recovery within 24 hours "
        "are right-censored at 24 hours. A later departure from the band is retained as a separate relapse "
        "endpoint. All estimators use source samples 2023 onward, while reported recovery time remains "
        "referenced to the initialization intervention at source sample 0. Windows are averaged within "
        "cells before the cell bootstrap.\n",
        encoding="utf-8",
    )
    print(runs.groupby("model")["recovery_or_censor_time_h"].mean().to_string())


if __name__ == "__main__":
    main()
