from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


RESULTS_CODE = (
    Path(__file__).resolve().parents[4]
    / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results"
)
sys.path.insert(0, str(RESULTS_CODE))
from jes2_plot_style import MODEL_COLORS, clean_axes, save_figure, setup_style


MODELS = ("DM", "HDM", "HECM", "DD")
STATES = ("fresh", "mid_life", "aged")
STATE_LABELS = {"fresh": "Fresh", "mid_life": "Mid-life", "aged": "Aged"}


def aging_ratio(frame: pd.DataFrame, value: str, state_column: str) -> pd.Series:
    pivot = frame.pivot(index="model", columns=state_column, values=value)
    ratio = pivot["aged"] / pivot["fresh"]
    return ratio.where((pivot["aged"] > 0) & (pivot["fresh"] > 0))


def main() -> None:
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the continuous current-bias lifecycle figure.")
    parser.add_argument("--results-dir", type=Path, default=paper / "JES_2.0/results")
    parser.add_argument("--lifecycle-results", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path,
        default=paper / "figures/Results/All Cells/Figure_05d_Current_Bias_Lifecycle.png",
    )
    args = parser.parse_args()
    lifecycle_results = args.lifecycle_results or args.results_dir

    lifecycle = pd.read_csv(lifecycle_results / "c29_lifecycle_bias_adverse.csv")
    lifecycle = lifecycle[lifecycle.soh_state.isin(STATES)].copy()
    natural = pd.read_csv(args.results_dir / "jes2_bias_aging_natural_summary.csv")
    controlled = pd.read_csv(args.results_dir / "controlled_soh_bias_adverse.csv")
    reset = json.loads((lifecycle_results / "c29_lifecycle_reset_protocol.json").read_text())

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": (1.15, 1.0)})
    x = np.arange(len(STATES), dtype=float)
    offsets = np.linspace(-0.10, 0.10, len(MODELS))
    for offset, model in zip(offsets, MODELS):
        part = lifecycle[lifecycle.model == model].set_index("soh_state").reindex(STATES)
        axes[0].plot(
            x + offset, part.delta_mae, color=MODEL_COLORS[model], marker="o",
            markersize=6.0, linewidth=1.9, label=model,
        )
    axes[0].axhline(0.0, color="#555555", linewidth=0.85)
    axes[0].set_xticks(x, [STATE_LABELS[state] for state in STATES])
    axes[0].set_ylabel(r"Larger signed-pair $\Delta$MAE [SOC]")
    axes[0].set_title("(a) Continuous 1,344-h C29 trajectory")
    axes[0].legend(ncol=4, frameon=False, loc="best")
    axes[0].text(
        0.99, 0.96,
        f"Active CV reset: {reset['eligible_cv_episodes']} episodes; "
        f"median interval {reset['median_inter_reset_h']:.1f} h",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=8.7, color="#666666",
    )
    clean_axes(axes[0])

    ratios = {
        "Natural 24-h windows\n(5 cells)": aging_ratio(
            natural, "mean_delta_mae", "window_soh_state"
        ),
        "Controlled SOH replay\n(same C29 trace)": aging_ratio(
            controlled, "delta_mae", "soh_level"
        ),
        "Continuous C29\n(active resets)": aging_ratio(
            lifecycle, "delta_mae", "soh_state"
        ),
    }
    group_x = np.arange(len(MODELS), dtype=float)
    width = 0.24
    alphas = (0.48, 0.28, 0.12)
    edge_alphas = (0.95, 0.72, 0.48)
    for index, ((label, values), alpha, edge_alpha) in enumerate(
        zip(ratios.items(), alphas, edge_alphas)
    ):
        positions = group_x + (index - 1) * width
        heights = values.reindex(MODELS).to_numpy(float)
        for position, value, model in zip(positions, heights, MODELS):
            if not np.isfinite(value):
                axes[1].text(
                    position, 0.04, "N/A", rotation=90, ha="center", va="bottom",
                    fontsize=8, color="#777777",
                )
                continue
            color = to_rgb(MODEL_COLORS[model])
            axes[1].bar(
                position, value, width=width * 0.9, facecolor=(*color, alpha),
                edgecolor=(*color, edge_alpha), linewidth=1.6,
            )
    axes[1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(group_x, MODELS)
    axes[1].set_ylabel("Aged / fresh bias sensitivity ratio")
    axes[1].set_title("(b) Does lower SOH amplify current-gain bias?")
    legend_handles = [
        Patch(
            facecolor=(0.35, 0.35, 0.35, alpha),
            edgecolor=(0.35, 0.35, 0.35, edge_alpha), linewidth=1.5, label=label,
        )
        for (label, _), alpha, edge_alpha in zip(ratios.items(), alphas, edge_alphas)
    ]
    axes[1].legend(handles=legend_handles, frameon=False, fontsize=8.3, loc="best")
    axes[1].text(
        0.01, 0.02, r"N/A: fresh or aged $\Delta$MAE $\leq 0$",
        transform=axes[1].transAxes, ha="left", va="bottom", fontsize=8.1, color="#666666",
    )
    clean_axes(axes[1])

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
