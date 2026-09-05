#!/usr/bin/env python3
"""Render the three-cell candidate replacement for the capacity study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "CURRENT_MODELS_CELL_MACRO"
ORDER = ("CNN", "GRU", "LSTM", "TCN")
COLORS = {
    "CNN": "#59C7C2",
    "GRU": "#59E83A",
    "LSTM": "#E76B91",
    "TCN": "#294862",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--paper-output-dir",
        type=Path,
        help="Optional destination for the four publication-ready PDF panels.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 17,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_display_variants(data: pd.DataFrame) -> pd.DataFrame:
    """Recenter visible size labels around the model used in Figure 3."""
    labeled = []
    for family in ORDER:
        subset = (
            data[data["architecture"] == family]
            .sort_values("parameters")
            .reset_index(drop=True)
            .copy()
        )
        selected_index = int(
            np.flatnonzero(subset["selected_for_figure_3"].to_numpy())[0]
        )
        display_variants = []
        for index, row in subset.iterrows():
            if index < selected_index:
                role = f"s{selected_index - index}"
            elif index > selected_index:
                role = f"l{index - selected_index}"
            else:
                role = "base"
            width = str(row["variant"]).rsplit("_h", 1)[-1]
            display_variants.append(f"{role}_h{width}")
        subset["display_variant"] = display_variants
        labeled.append(subset)
    return pd.concat(labeled, ignore_index=True)


def draw_panel(ax: plt.Axes, family: str, data: pd.DataFrame) -> None:
    subset = data[data["architecture"] == family].sort_values("parameters")
    color = COLORS[family]
    ax.plot(
        subset["parameters"],
        subset["mae"],
        color=color,
        marker="o",
        markersize=8,
        linewidth=2.4,
        zorder=3,
    )
    selected = subset[subset["selected_for_figure_3"]]
    ax.scatter(
        selected["parameters"],
        selected["mae"],
        s=155,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.8,
        zorder=4,
        label="Baseline used in Fig. 3",
    )
    ax.scatter(
        selected["parameters"],
        selected["mae"],
        s=55,
        facecolor=color,
        edgecolor=color,
        zorder=5,
    )
    for row in subset.itertuples(index=False):
        label = row.display_variant.replace("base_", "base\n").replace("_h", "\nh")
        offset = (0, 9)
        alignment = "center"
        if family == "GRU" and row.display_variant == "s4_h48":
            offset = (-5, 9)
            alignment = "right"
        elif family == "GRU" and row.display_variant == "s3_h64":
            offset = (5, 9)
            alignment = "left"
        ax.annotate(
            label,
            (row.parameters, row.mae),
            textcoords="offset points",
            xytext=offset,
            ha=alignment,
            fontsize=14,
        )
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Cell-macro MAE")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.grid(color="#C8C8C8", alpha=0.45, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.01, 0.039)
    ax.margins(x=0.09)


def render_figures(
    data: pd.DataFrame, output_dir: Path, paper_output_dir: Path | None = None
) -> None:
    if paper_output_dir is not None:
        paper_output_dir.mkdir(parents=True, exist_ok=True)
    for family in ORDER:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
        fig.subplots_adjust(left=0.17, right=0.975, top=0.97, bottom=0.19)
        draw_panel(ax, family, data)
        stem = output_dir / f"candidate_capacity_sensitivity_{family.lower()}_three_cell"
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        if paper_output_dir is not None:
            fig.savefig(
                paper_output_dir
                / f"baseline_capacity_sensitivity_{family.lower()}_three_cell.pdf",
                bbox_inches="tight",
            )
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.4), dpi=180)
    for ax, family, label in zip(axes.flat, ORDER, ("(a)", "(b)", "(c)", "(d)")):
        draw_panel(ax, family, data)
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.08, wspace=0.20, hspace=0.30)
    stem = output_dir / "candidate_figure_5_capacity_sensitivity_three_cell"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def local_minimum(subset: pd.DataFrame, selected_index: int) -> bool:
    selected_mae = float(subset.iloc[selected_index]["mae"])
    neighbors = []
    if selected_index > 0:
        neighbors.append(float(subset.iloc[selected_index - 1]["mae"]))
    if selected_index + 1 < len(subset):
        neighbors.append(float(subset.iloc[selected_index + 1]["mae"]))
    return all(selected_mae < value for value in neighbors)


def write_report(
    data: pd.DataFrame, consistency: pd.DataFrame, output_dir: Path
) -> None:
    lines = [
        "# Three-cell Capacity Sensitivity Check",
        "",
        "The 28 trained size variants were evaluated without retraining on C11, C23, and C29. Metrics were calculated per cell and then macro-averaged so that every test cell has equal weight.",
        "",
        "## Figure 3 consistency",
        "",
        "| Architecture | New MAE | Figure 3 MAE | New RMSE | Figure 3 RMSE | Match at 4 decimals |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in consistency.itertuples(index=False):
        lines.append(
            f"| {row.architecture} | {row.mae_capacity_study:.6f} | "
            f"{row.mae_figure_3:.6f} | {row.rmse_capacity_study:.6f} | "
            f"{row.rmse_figure_3:.6f} | {'yes' if row.reported_4dp_match else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Capacity result",
            "",
            "| Architecture | Displayed Figure 3 point | Internal model ID | C11 MAE | Three-cell MAE | Strict local minimum | Lowest tested point | Lowest MAE |",
            "|---|---|---|---:|---:|:---:|---|---:|",
        ]
    )
    per_cell = pd.read_csv(output_dir / "metrics_by_cell.csv")
    for family in ORDER:
        subset = data[data["architecture"] == family].sort_values("parameters").reset_index(drop=True)
        selected_index = int(np.flatnonzero(subset["selected_for_figure_3"].to_numpy())[0])
        selected = subset.iloc[selected_index]
        best = subset.loc[subset["mae"].idxmin()]
        c11_mae = float(
            per_cell[
                (per_cell["architecture"] == family)
                & (per_cell["variant"] == selected["variant"])
                & (per_cell["cell"] == "C11")
            ]["mae"].iloc[0]
        )
        lines.append(
            f"| {family} | {selected['display_variant']} | {selected['variant']} | "
            f"{c11_mae:.6f} | "
            f"{selected['mae']:.6f} | "
            f"{'yes' if local_minimum(subset, selected_index) else 'no'} | "
            f"{best['display_variant']} | {best['mae']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The three-cell result confirms a non-monotonic relationship between parameter count and test MAE. It does not support the stronger claim that every baseline used in Figure 3 is a local test-MAE minimum. Only the selected CNN is a strict local minimum in this sweep.",
            "",
            "The single-cell C11 figure should not be retained merely because it produces a cleaner minimum. The defensible interpretation is that the reference configurations were fixed through the model-development and validation pipeline before this test-set sweep. The three-cell result is a post-selection sensitivity analysis, not a second model-selection step. Since every variant represents one trained initialization, the curves must not be interpreted as a seed-averaged causal effect of model capacity.",
        ]
    )
    (output_dir / "THREE_CELL_CAPACITY_STUDY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.expanduser().resolve()
    summary = pd.read_csv(output_dir / "metrics_summary.csv")
    data = add_display_variants(
        summary[summary["aggregation"] == "cell_macro"].copy()
    )
    data.to_csv(output_dir / "capacity_sensitivity_display_data.csv", index=False)
    consistency = pd.read_csv(output_dir / "reference_consistency.csv")
    configure_style()
    paper_output_dir = (
        args.paper_output_dir.expanduser().resolve()
        if args.paper_output_dir is not None
        else None
    )
    render_figures(data, output_dir, paper_output_dir)
    write_report(data, consistency, output_dir)
    print(f"Candidate Figure 5 and report written to {output_dir}")


if __name__ == "__main__":
    main()
