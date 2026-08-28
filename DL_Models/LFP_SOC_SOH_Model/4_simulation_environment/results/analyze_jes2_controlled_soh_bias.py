from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("HDM", "HECM", "DD")
LEVELS = ("fresh", "mid_life", "aged")
ALIASES = ("positive_3pct", "negative_3pct")
PREDICTIONS = {"HDM": "soc_cc", "HECM": "soc_ecm", "DD": "soc_pred"}
PATTERNS = {
    "HDM": "soc_cc_soh_fullcell_*.csv",
    "HECM": "ecm_soc_fullcell_*.csv",
    "DD": "soc_pred_fullcell_*.csv",
}


def load_run(root: Path, level: str, alias: str, model: str) -> pd.DataFrame:
    path = next((root / "runs" / level / alias / model).glob(PATTERNS[model]))
    frame = pd.read_csv(path)
    return frame.rename(columns={PREDICTIONS[model]: "soc_pred"})


def pair_runs(baseline: pd.DataFrame, biased: pd.DataFrame) -> pd.DataFrame:
    columns = ["index", "time_s", "soc_true", "soc_pred", "abs_err"]
    pair = baseline[columns].merge(
        biased[columns], on="index", suffixes=("_baseline", "_biased"), how="inner"
    )
    pair["prediction_divergence"] = np.abs(pair.soc_pred_biased - pair.soc_pred_baseline)
    pair["delta_abs_error"] = pair.abs_err_biased - pair.abs_err_baseline
    pair["elapsed_h"] = (pair.time_s_baseline - pair.time_s_baseline.iloc[0]) / 3600.0
    dt_h = pair.time_s_baseline.diff().fillna(0).clip(lower=0).to_numpy(float) / 3600.0
    pair["cumulative_divergence_soc_h"] = np.cumsum(pair.prediction_divergence.to_numpy(float) * dt_h)
    pair["cumulative_delta_error_soc_h"] = np.cumsum(pair.delta_abs_error.to_numpy(float) * dt_h)
    return pair


def main() -> None:
    simulation = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze the fixed-trace, controlled-SOH current-bias replay.")
    parser.add_argument(
        "--campaign", type=Path,
        default=simulation / "campaigns/jes2_controlled_soh_bias_C29_20260828",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir or args.campaign / "results"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.campaign / "jes2_manifest.json").read_text(encoding="utf-8"))
    soh_values = {
        row["soh_level"]: float(row["soh_value"])
        for row in manifest["runs"] if row.get("status") in {"completed", "skipped_existing"}
    }

    rows = []
    pairs: dict[tuple[str, str, str], pd.DataFrame] = {}
    for level in LEVELS:
        for model in MODELS:
            baseline = load_run(args.campaign, level, "baseline", model)
            baseline_mae = float(baseline.abs_err.mean())
            for alias in ALIASES:
                biased = load_run(args.campaign, level, alias, model)
                pair = pair_runs(baseline, biased)
                pairs[(level, model, alias)] = pair
                rows.append({
                    "cell": "C29",
                    "measurement_window": "C29_mid_life",
                    "soh_level": level,
                    "controlled_soh": soh_values[level],
                    "model": model,
                    "alias": alias,
                    "baseline_mae": baseline_mae,
                    "biased_mae": float(biased.abs_err.mean()),
                    "delta_mae": float(biased.abs_err.mean() - baseline_mae),
                    "mean_prediction_divergence": float(pair.prediction_divergence.mean()),
                    "cumulative_divergence_soc_h": float(pair.cumulative_divergence_soc_h.iloc[-1]),
                    "cumulative_delta_error_soc_h": float(pair.cumulative_delta_error_soc_h.iloc[-1]),
                    "samples": len(pair),
                })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "controlled_soh_bias_metrics.csv", index=False)
    adverse = metrics.loc[metrics.groupby(["soh_level", "model"]).delta_mae.idxmax()].copy()
    adverse = adverse.sort_values(["model", "controlled_soh"], ascending=[True, False])
    adverse.to_csv(out / "controlled_soh_bias_adverse.csv", index=False)

    trajectories = []
    for row in adverse.itertuples(index=False):
        pair = pairs[(row.soh_level, row.model, row.alias)]
        indices = np.unique(np.r_[np.arange(0, len(pair), 60), len(pair) - 1])
        sampled = pair.iloc[indices][
            ["elapsed_h", "prediction_divergence", "delta_abs_error",
             "cumulative_divergence_soc_h", "cumulative_delta_error_soc_h"]
        ].copy()
        sampled.insert(0, "alias", row.alias)
        sampled.insert(0, "model", row.model)
        sampled.insert(0, "controlled_soh", row.controlled_soh)
        sampled.insert(0, "soh_level", row.soh_level)
        trajectories.append(sampled)
    pd.concat(trajectories, ignore_index=True).to_csv(
        out / "controlled_soh_bias_trajectories_1min.csv", index=False
    )

    print(adverse[
        ["soh_level", "controlled_soh", "model", "alias", "baseline_mae", "biased_mae",
         "delta_mae", "mean_prediction_divergence"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
