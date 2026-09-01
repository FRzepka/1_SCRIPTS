from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "current_offset_neg_50mA": -0.05,
    "current_offset_pos_50mA": 0.05,
}
MODELS = ["DM", "HDM", "HECM", "DD"]
EXPECTED_CELLS = {"C09", "C13", "C15", "C25", "C27", "C29"}
EXPECTED_WINDOWS = 16


def cell_bootstrap(values: pd.Series, samples: int, seed: int) -> dict[str, float]:
    data = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def load_offset_runs(manifests: list[Path]) -> pd.DataFrame:
    rows = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["runs"]:
            if record["alias"] not in ALIASES:
                continue
            summary_path = Path(record["out_dir"]) / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing completed summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            offset = float(summary["scenario_meta"]["current_offset_a"])
            if not np.isclose(offset, ALIASES[record["alias"]]):
                raise ValueError(f"Offset mismatch in {summary_path}: {offset}")
            rows.append(
                {
                    "cell": record["cell"],
                    "window_id": record["window_id"],
                    "soh_state": record["soh_state"],
                    "cell_load_class": record["cell_load_class"],
                    "model": record["model"],
                    "alias": record["alias"],
                    "current_offset_a": offset,
                    "mae": float(summary["mae"]),
                    "rmse": float(summary["rmse"]),
                    "p95_error": float(summary["p95_error"]),
                    "max_error": float(summary["max_error"]),
                    "bias": float(summary["bias"]),
                    "evaluation_start_sample": int(summary["evaluation_start_sample"]),
                    "evaluation_samples": int(summary["evaluation_samples"]),
                    "summary_path": str(summary_path.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    expected_rows = EXPECTED_WINDOWS * len(MODELS) * len(ALIASES)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} offset runs, found {len(frame)}")
    if set(frame["cell"]) != EXPECTED_CELLS:
        raise ValueError(f"Unexpected cell coverage: {sorted(set(frame['cell']))}")
    if set(frame["model"]) != set(MODELS) or set(frame["alias"]) != set(ALIASES):
        raise ValueError("Current-offset model or sign coverage is incomplete")
    if set(frame["evaluation_start_sample"]) != {2023}:
        raise ValueError("Current-offset runs do not use the common evaluation mask")
    if set(frame["evaluation_samples"]) != {84377}:
        raise ValueError("Current-offset runs do not have the expected matched sample count")
    if frame.duplicated(["cell", "window_id", "model", "alias"]).any():
        raise ValueError("Duplicate current-offset run keys detected")
    return frame


def add_baseline_deltas(offset: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["alias"].eq("baseline")]
    baseline = baseline[
        ((baseline["model"] == "DM") & baseline["soh_condition"].fillna("none").eq("none"))
        | ((baseline["model"] != "DM") & baseline["soh_condition"].eq("lstm_h1"))
    ]
    keys = ["cell", "window_id", "model"]
    if len(baseline) != EXPECTED_WINDOWS * len(MODELS) or baseline.duplicated(keys).any():
        raise ValueError("Nominal baseline selection is not one-to-one")
    baseline = baseline[keys + ["mae", "rmse", "p95_error", "bias"]].rename(
        columns={name: f"baseline_{name}" for name in ["mae", "rmse", "p95_error", "bias"]}
    )
    merged = offset.merge(baseline, on=keys, how="left", validate="many_to_one")
    if merged["baseline_mae"].isna().any():
        raise ValueError("At least one current-offset run has no matched nominal baseline")
    for metric in ["mae", "rmse", "p95_error"]:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]
    return merged


def aggregate(
    runs: pd.DataFrame, bootstrap_samples: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [
        "mae",
        "rmse",
        "p95_error",
        "bias",
        "baseline_mae",
        "baseline_rmse",
        "baseline_p95_error",
        "baseline_bias",
        "delta_mae",
        "delta_rmse",
        "delta_p95_error",
    ]
    cells = (
        runs.groupby(["cell", "cell_load_class", "model", "alias", "current_offset_a"], as_index=False)[numeric]
        .mean()
        .sort_values(["model", "alias", "cell"])
    )

    statistics = []
    for index, ((model, alias), group) in enumerate(cells.groupby(["model", "alias"], sort=False)):
        row = {
            "model": model,
            "alias": alias,
            "current_offset_a": float(group["current_offset_a"].iloc[0]),
            "n_cells": len(group),
            "n_windows": int((runs["model"].eq(model) & runs["alias"].eq(alias)).sum()),
        }
        for metric_index, metric in enumerate(["mae", "rmse", "delta_mae", "delta_rmse"]):
            values = cell_bootstrap(group[metric], bootstrap_samples, 9200 + 100 * index + metric_index)
            row[metric] = values["mean"]
            row[f"{metric}_ci_low"] = values["ci_low"]
            row[f"{metric}_ci_high"] = values["ci_high"]
        statistics.append(row)
    statistics_frame = pd.DataFrame(statistics)

    adverse_cells = (
        cells.sort_values("delta_mae", ascending=False)
        .groupby(["cell", "model"], as_index=False)
        .first()
        .sort_values(["model", "cell"])
    )
    adverse = []
    for index, (model, group) in enumerate(adverse_cells.groupby("model", sort=False)):
        delta = cell_bootstrap(group["delta_mae"], bootstrap_samples, 19300 + index)
        total = cell_bootstrap(group["mae"], bootstrap_samples, 19400 + index)
        adverse.append(
            {
                "model": model,
                "n_cells": len(group),
                "n_windows": int((runs["model"].eq(model)).sum() / len(ALIASES)),
                "adverse_delta_mae": delta["mean"],
                "adverse_delta_mae_ci_low": delta["ci_low"],
                "adverse_delta_mae_ci_high": delta["ci_high"],
                "adverse_mae": total["mean"],
                "adverse_mae_ci_low": total["ci_low"],
                "adverse_mae_ci_high": total["ci_high"],
                "positive_adverse_cells": int((group["current_offset_a"] > 0).sum()),
                "negative_adverse_cells": int((group["current_offset_a"] < 0).sum()),
            }
        )
    return cells, statistics_frame, pd.DataFrame(adverse)


def write_note(path: Path, statistics: pd.DataFrame, adverse: pd.DataFrame) -> None:
    signed = statistics.pivot(index="model", columns="alias", values="delta_mae")
    adverse_by_model = adverse.set_index("model")
    lines = [
        "# JES2 additive current-offset results",
        "",
        "Generated from the completed six-cell, 16-window JES2 extension. The measured current is",
        "changed additively by -50 mA or +50 mA. This is distinct from the multiplicative",
        "current-gain error already present in the benchmark. All four estimators use the same",
        "sign-matched causal SOH trace, common sample mask, and matched nominal baseline.",
        "",
        "## Cell-macro result",
        "",
        "| Model | Delta MAE at -50 mA | Delta MAE at +50 mA | Adverse Delta MAE [95% CI] | Adverse sign by cell (+/-) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = adverse_by_model.loc[model]
        lines.append(
            f"| {model} | {signed.loc[model, 'current_offset_neg_50mA']:.5f} | "
            f"{signed.loc[model, 'current_offset_pos_50mA']:.5f} | "
            f"{row.adverse_delta_mae:.5f} [{row.adverse_delta_mae_ci_low:.5f}, "
            f"{row.adverse_delta_mae_ci_high:.5f}] | "
            f"{int(row.positive_adverse_cells)}/{int(row.negative_adverse_cells)} |"
        )
    best = adverse.loc[adverse["adverse_delta_mae"].idxmin(), "model"]
    worst = adverse.loc[adverse["adverse_delta_mae"].idxmax(), "model"]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"The smallest adverse additive-offset penalty is observed for {best}, while {worst} has the largest.",
            "The result represents one matched signed stress level. It does not claim complete coverage of",
            "all current-sensor offset magnitudes, time variation, calibration drift, or combined sensor errors.",
            "The manuscript PDF was intentionally not rebuilt from these results.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the signed JES2 additive current-offset extension.")
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline_runs", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = add_baseline_deltas(load_offset_runs(args.manifests), args.baseline_runs)
    cells, statistics, adverse = aggregate(runs, args.bootstrap_samples)
    runs.to_csv(args.out_dir / "jes2_current_offset_runs.csv", index=False)
    cells.to_csv(args.out_dir / "jes2_current_offset_cells.csv", index=False)
    statistics.to_csv(args.out_dir / "jes2_current_offset_statistics.csv", index=False)
    adverse.to_csv(args.out_dir / "jes2_current_offset_adverse_statistics.csv", index=False)
    protocol = {
        "analysis": "JES2 signed additive current-offset extension",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "current_offset_a": [-0.05, 0.05],
        "current_offset_mA": [-50, 50],
        "runs": len(runs),
        "cells": sorted(runs["cell"].unique()),
        "windows": sorted(runs["window_id"].unique()),
        "models": MODELS,
        "evaluation_start_sample": 2023,
        "evaluation_samples_per_run": 84377,
        "aggregation": "windows averaged within cell; equal-weight six-cell macro; 10000-draw cell bootstrap",
        "adverse_definition": "larger signed delta MAE selected within each cell before equal-weight aggregation",
        "interpretation_boundary": "One matched signed additive stress level, separate from multiplicative current gain. No combined sensor errors.",
    }
    (args.out_dir / "jes2_current_offset_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    write_note(args.out_dir / "JES2_CURRENT_OFFSET_RESULTS.md", statistics, adverse)
    print(json.dumps({"runs": len(runs), "out_dir": str(args.out_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
