from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jes2_plot_style import MODEL_COLORS, MODEL_HATCHES, clean_axes, save_figure, setup_style


MODELS = ["DM", "HDM", "HECM", "DD"]
POSITIVE = {
    "current_bias_0p5pct": 0.5,
    "current_bias_1p5pct": 1.5,
    "current_bias_3p0pct": 3.0,
}
NEGATIVE = {
    "current_bias_neg_0p5pct": -0.5,
    "current_bias_neg_1p5pct": -1.5,
    "current_bias_neg_3p0pct": -3.0,
}


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[((frame.model == "DM") & (frame.soh_condition == "none")) |
                 ((frame.model != "DM") & (frame.soh_condition == "lstm_h1"))].copy()


def bootstrap_cell_macro(values: pd.Series, samples: int, seed: int) -> tuple[float, float, float]:
    cell_values = values.dropna().to_numpy(dtype=float)
    point = float(np.mean(cell_values))
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(cell_values, size=(samples, len(cell_values)), replace=True), axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the symmetric signed-current-bias JES2 Figure 05.")
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    base = primary(pd.read_csv(args.base_results))
    base = base[base.alias.isin(["baseline", *POSITIVE])]
    base["offset_pct"] = base.alias.map({"baseline": 0.0, **POSITIVE})
    base = base[["cell", "window_id", "model", "offset_pct", "mae"]]

    manifest = json.loads(args.negative_manifest.read_text(encoding="utf-8"))
    negative_rows = []
    for record in manifest["runs"]:
        summary = json.loads((Path(record["out_dir"]) / "summary.json").read_text(encoding="utf-8"))
        negative_rows.append({
            "cell": record["cell"], "window_id": record["window_id"], "model": record["model"],
            "offset_pct": 100.0 * float(record["current_offset_pct"]), "mae": float(summary["mae"]),
        })
    runs = pd.concat([base, pd.DataFrame(negative_rows)], ignore_index=True)
    baseline = runs[runs.offset_pct == 0].set_index(["cell", "window_id", "model"])["mae"]
    runs["delta_mae"] = runs.apply(
        lambda row: row.mae - baseline.loc[(row.cell, row.window_id, row.model)], axis=1
    )
    cells = runs.groupby(["model", "offset_pct", "cell"], as_index=False).delta_mae.mean()

    rows = []
    for index, ((model, offset), group) in enumerate(cells.groupby(["model", "offset_pct"])):
        mean, low, high = bootstrap_cell_macro(group.delta_mae, args.bootstrap_samples, 2608 + index)
        rows.append({"model": model, "offset_pct": offset, "mean": mean, "ci_low": low, "ci_high": high,
                     "n_cells": group.cell.nunique()})
    statistics = pd.DataFrame(rows).sort_values(["model", "offset_pct"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.out_csv, index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for model in MODELS:
        part = statistics[statistics.model == model].sort_values("offset_pct")
        ax.errorbar(
            part.offset_pct, part["mean"],
            yerr=np.vstack([part["mean"] - part.ci_low, part.ci_high - part["mean"]]),
            color=MODEL_COLORS[model], marker="o", linewidth=2.0, capsize=3, label=model,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Signed current-sensor bias [%]")
    ax.set_ylabel(r"Cell-macro $\Delta$MAE [SOC]")
    ax.set_title("Symmetric current-bias sensitivity across independent holdout cells")
    ax.set_xticks([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
    ax.legend(ncol=4, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, args.figures_dir / "Figure_05_Current_Bias.png")


if __name__ == "__main__":
    main()
