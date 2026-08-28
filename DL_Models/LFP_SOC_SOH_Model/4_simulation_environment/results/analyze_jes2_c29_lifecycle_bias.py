from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MODELS = ("DM", "HDM", "HECM", "DD")
ALIASES = ("positive_3pct", "negative_3pct")
STATES = ("fresh", "mid_life", "aged")


def metrics_by_state(summary: dict) -> dict[str, dict]:
    states = {
        row["stratum"]: row
        for row in summary["stratified_metrics"] if row["dimension"] == "soh_state"
    }
    states["all"] = summary
    return states


def cv_reset_statistics(data_root: Path, cell: str = "C29") -> dict:
    path = data_root / f"df_FE_{cell}.parquet"
    table = pq.read_table(path, columns=["Testtime[s]", "Voltage[V]"])
    frame = table.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
    time_s = frame["Testtime[s]"].to_numpy(float)
    high = frame["Voltage[V]"].to_numpy(float) >= 3.63
    starts = np.flatnonzero(high & np.r_[True, ~high[:-1]])
    stops = np.flatnonzero(high & np.r_[~high[1:], True])
    durations = time_s[stops] - time_s[starts]
    eligible = durations >= 300.0
    reset_times = time_s[starts[eligible]] + 300.0
    intervals_h = np.diff(reset_times) / 3600.0
    return {
        "cell": cell,
        "voltage_threshold_v": 3.63,
        "sustain_seconds": 300.0,
        "eligible_cv_episodes": int(eligible.sum()),
        "median_inter_reset_h": float(np.median(intervals_h)) if len(intervals_h) else None,
        "max_inter_reset_h": float(np.max(intervals_h)) if len(intervals_h) else None,
        "lifecycle_duration_h": float((time_s[-1] - time_s[0]) / 3600.0),
    }


def main() -> None:
    simulation = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze the continuous C29 lifecycle current-bias test.")
    parser.add_argument(
        "--campaign", type=Path,
        default=simulation / "campaigns/jes2_c29_lifecycle_bias_20260828",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir or args.campaign / "results"
    out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for alias in ("baseline",) + ALIASES:
        for model in MODELS:
            path = args.campaign / "runs" / alias / model / "summary.json"
            summaries[(alias, model)] = metrics_by_state(json.loads(path.read_text(encoding="utf-8")))

    rows = []
    for state in ("all",) + STATES:
        for model in MODELS:
            baseline = summaries[("baseline", model)][state]
            for alias in ALIASES:
                biased = summaries[(alias, model)][state]
                rows.append({
                    "cell": "C29", "soh_state": state, "model": model, "alias": alias,
                    "n_samples": int(biased.get("n_samples", 0)),
                    "baseline_mae": float(baseline["mae"]),
                    "biased_mae": float(biased["mae"]),
                    "delta_mae": float(biased["mae"] - baseline["mae"]),
                    "baseline_rmse": float(baseline["rmse"]),
                    "biased_rmse": float(biased["rmse"]),
                    "delta_rmse": float(biased["rmse"] - baseline["rmse"]),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "c29_lifecycle_bias_metrics.csv", index=False)
    adverse = metrics.loc[metrics.groupby(["soh_state", "model"]).delta_mae.idxmax()].copy()
    adverse.to_csv(out / "c29_lifecycle_bias_adverse.csv", index=False)
    reset_stats = cv_reset_statistics(args.data_root)
    (out / "c29_lifecycle_reset_protocol.json").write_text(
        json.dumps(reset_stats, indent=2), encoding="utf-8"
    )

    print(adverse[[
        "soh_state", "model", "alias", "baseline_mae", "biased_mae", "delta_mae"
    ]].sort_values(["model", "soh_state"]).to_string(index=False))
    print(json.dumps(reset_stats, indent=2))


if __name__ == "__main__":
    main()
