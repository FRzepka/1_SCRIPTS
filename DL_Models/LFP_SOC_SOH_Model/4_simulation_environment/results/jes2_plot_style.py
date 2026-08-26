from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# Palette used by the existing dissertation robustness figures in
# pictures/eaai_palette: green DM, purple HDM, blue HECM, and red DD.
MODEL_COLORS = {
    "DM": "#2ca02c",
    "HDM": "#9467bd",
    "HECM": "#1f77b4",
    "DD": "#d62728",
}
MODEL_HATCHES = {"DM": "//", "HDM": "..", "HECM": "xx", "DD": "\\\\"}
MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
NEUTRAL_DARK = "#434343"
NEUTRAL_MID = "#777777"
NEUTRAL_LIGHT = "#d9d9d9"
TU_RED = "#d62728"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": NEUTRAL_DARK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": NEUTRAL_LIGHT,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "font.family": "Nimbus Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
