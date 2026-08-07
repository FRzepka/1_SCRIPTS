from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parents[4]
TABLE_DIR = (
    SCRIPTS_ROOT
    / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_tables_v4"
)
OUTPUT = PROJECT_ROOT / "pictures/eaai_palette/robustness_signal_integrity.png"

MODELS = [
    "Direct measurement",
    "Hybrid direct measurement",
    "Hybrid ECM",
    "Data-driven",
]
SHORT = {
    "Direct measurement": "DM",
    "Hybrid direct measurement": "HDM",
    "Hybrid ECM": "HECM",
    "Data-driven": "DD",
}
COLORS = {
    "Direct measurement": "#2CA02C",
    "Hybrid direct measurement": "#9467BD",
    "Hybrid ECM": "#1F77B4",
    "Data-driven": "#D62728",
}


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha)


def _read_local_metrics(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|:"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if fields[0] == "class":
            continue
        rows.append(fields)
    return pd.DataFrame(
        rows,
        columns=["class", "focus_scenario", "local_metric", "value", "threshold"],
    )


def render() -> None:
    key = pd.read_csv(TABLE_DIR / "table_key_results.md")
    local = _read_local_metrics(TABLE_DIR / "table_local_behaviour.md")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.8,
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "hatch.linewidth": 1.35,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))
    x = np.arange(len(MODELS))
    width = 0.22
    scenarios = [
        ("Missing samples", "//"),
        ("Irregular sampling", ".."),
        ("Burst dropout", ""),
    ]

    for scenario_index, (scenario, hatch) in enumerate(scenarios):
        scenario_rows = key[key["scenario_label"] == scenario].set_index("class")
        values = scenario_rows.reindex(MODELS)["delta_mae"].to_numpy(dtype=float)
        edgecolors = [COLORS[model] for model in MODELS]
        if scenario == "Missing samples":
            facecolors = [(1.0, 1.0, 1.0, 1.0)] * len(MODELS)
        elif scenario == "Irregular sampling":
            facecolors = [_rgba(COLORS[model], 0.13) for model in MODELS]
        else:
            facecolors = [_rgba(COLORS[model], 0.28) for model in MODELS]

        axes[0].bar(
            x + (scenario_index - 1) * width,
            values,
            width=width,
            color=facecolors,
            edgecolor=edgecolors,
            linewidth=2.0,
            hatch=hatch,
            zorder=3,
        )

    axes[0].axhline(0.0, color="#777777", linewidth=0.8, zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT[model] for model in MODELS])
    axes[0].set_ylabel(r"$\Delta$MAE")
    axes[0].set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[0].legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="#666666",
                linewidth=1.8,
                hatch="//",
                label="Missing samples",
            ),
            Patch(
                facecolor=_rgba("#999999", 0.13),
                edgecolor="#666666",
                linewidth=1.8,
                hatch="..",
                label="Irregular sampling",
            ),
            Patch(
                facecolor=_rgba("#999999", 0.28),
                edgecolor="#666666",
                linewidth=1.8,
                label="Burst dropout",
            ),
        ],
        frameon=True,
        loc="upper right",
    )

    recovery_rows = local[
        (local["focus_scenario"] == "missing_gap")
        & (local["local_metric"] == "recovery_time_h")
    ].copy()
    recovery_rows["value"] = pd.to_numeric(recovery_rows["value"], errors="coerce")
    recovery = recovery_rows.set_index("class").reindex(MODELS)["value"].to_numpy()
    bar_colors = [_rgba(COLORS[model], 0.43) for model in MODELS]
    bars = axes[1].bar(
        x,
        recovery,
        color=bar_colors,
        edgecolor=[COLORS[model] for model in MODELS],
        linewidth=1.5,
        zorder=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT[model] for model in MODELS])
    axes[1].set_ylabel("Recovery time [h]")
    axes[1].set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    max_value = float(np.nanmax(recovery))
    for model, bar, value in zip(MODELS, bars, recovery):
        text_color = "black" if model == "Hybrid ECM" else "white"
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value - max_value * 0.04,
            f"{value:.1f}",
            ha="center",
            va="top",
            color=text_color,
            fontsize=12,
        )

    fig.subplots_adjust(wspace=0.18)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    render()
