from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = DISS_ROOT.parents[2]
OUT_DIR = DISS_ROOT / "pictures" / "red_muted"

TABLE_DIR = SCRIPTS_ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_tables_v4"

CLASS_ORDER = [
    "Direct measurement",
    "Hybrid direct measurement",
    "Hybrid ECM",
    "Data-driven",
]

CLASS_SHORT = {
    "Direct measurement": "DM",
    "Hybrid direct measurement": "HDM",
    "Hybrid ECM": "HECM",
    "Data-driven": "DD",
}

CLASS_COLORS = {
    "DM": "#b6302d",
    "HDM": "#d1887e",
    "HECM": "#8b6763",
    "DD": "#566b78",
}

ROBUSTNESS_SCENARIOS = [
    "Current noise (high)",
    "Current bias",
    "Irregular sampling",
    "Burst dropout",
    "Missing samples",
    "Voltage spikes",
    "Temperature noise",
    "Voltage noise",
]

PROFILE_WEIGHTS = {
    "Accuracy-weighted": {"Accuracy": 0.60, "Robustness": 0.20, "Recovery": 0.20},
    "Robustness-weighted": {"Accuracy": 0.20, "Robustness": 0.60, "Recovery": 0.20},
    "Recovery-weighted": {"Accuracy": 0.20, "Robustness": 0.20, "Recovery": 0.60},
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def read_markdown_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    headers: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {":", "-"} for cell in cells):
            continue
        if headers is None:
            headers = cells
            continue
        rows.append(cells)
    if headers is None:
        raise ValueError(f"No markdown table found in {path}")
    return pd.DataFrame(rows, columns=headers)


def lower_better_scores(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    vmax = values.max()
    vmin = values.min()
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (vmax - values) / (vmax - vmin)


def penalized_lower_better_scores(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    vmin = finite.min()
    vmax = finite.max()
    penalty = max(vmax * 1.25, vmin + 1e-6)
    filled = values.fillna(penalty)
    if np.isclose(penalty, vmin):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (penalty - filled) / (penalty - vmin)


def build_meta_scores() -> pd.DataFrame:
    baseline = pd.read_csv(TABLE_DIR / "table_baseline.md")
    key = pd.read_csv(TABLE_DIR / "table_key_results.md")
    local = read_markdown_table(TABLE_DIR / "table_local_behaviour.md")

    baseline = baseline.set_index("class").loc[CLASS_ORDER]
    key = key[key["scenario_label"].isin(ROBUSTNESS_SCENARIOS)].copy()
    key["class"] = pd.Categorical(key["class"], CLASS_ORDER, ordered=True)
    key = key.sort_values(["scenario_label", "class"])

    local = local[local["local_metric"].isin(["recovery_time_to_baseline_band_strict_h"])].copy()
    local["value"] = pd.to_numeric(local["value"], errors="coerce")

    accuracy_parts = []
    for metric in ["mae", "rmse", "p95_error"]:
        part = lower_better_scores(baseline[metric])
        part.name = metric
        accuracy_parts.append(part)
    accuracy_scores = pd.concat(accuracy_parts, axis=1).mean(axis=1).rename("Accuracy")

    robustness_raw = key.pivot(index="class", columns="scenario_label", values="delta_mae").loc[CLASS_ORDER]
    robustness_score_matrix = robustness_raw.apply(lower_better_scores, axis=0)
    robustness_score_matrix = robustness_score_matrix[ROBUSTNESS_SCENARIOS]
    robustness_scores = robustness_score_matrix.mean(axis=1).rename("Robustness")

    recovery_raw = local.pivot(index="class", columns="local_metric", values="value").reindex(CLASS_ORDER)
    recovery_score_matrix = recovery_raw.apply(penalized_lower_better_scores, axis=0)
    recovery_scores = recovery_score_matrix["recovery_time_to_baseline_band_strict_h"].rename("Recovery")

    meta_scores = pd.concat([accuracy_scores, robustness_scores, recovery_scores], axis=1).reset_index()
    meta_scores = meta_scores.rename(columns={"class": "Class"})
    meta_scores["Model"] = meta_scores["Class"].map(CLASS_SHORT)
    return meta_scores[["Model", "Class", "Accuracy", "Robustness", "Recovery"]]


def render_decision_synthesis() -> None:
    meta_scores = build_meta_scores()
    labels = ["Accuracy", "Robustness", "Recovery"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    profile_scores = meta_scores[["Model"]].copy()
    for profile, weights in PROFILE_WEIGHTS.items():
        profile_scores[profile] = (
            meta_scores["Accuracy"] * weights["Accuracy"]
            + meta_scores["Robustness"] * weights["Robustness"]
            + meta_scores["Recovery"] * weights["Recovery"]
        )

    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.32)
    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels)
    ax_radar.tick_params(axis="x", pad=10)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_facecolor("white")
    for _, row in meta_scores.iterrows():
        short = row["Model"]
        values = row[labels].to_numpy(dtype=float)
        values = np.concatenate([values, [values[0]]])
        color = CLASS_COLORS[short]
        ax_radar.plot(angles, values, linewidth=2.2, color=color, label=short)
        ax_radar.fill(angles, values, color=color, alpha=0.10)

    x = np.arange(len(PROFILE_WEIGHTS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(profile_scores))
    for i, (_, row) in enumerate(profile_scores.iterrows()):
        short = row["Model"]
        ax_bar.bar(
            x + offsets[i],
            row[list(PROFILE_WEIGHTS.keys())].to_numpy(dtype=float),
            width=width,
            label=short,
            color=CLASS_COLORS[short],
            alpha=0.95,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(list(PROFILE_WEIGHTS.keys()))
    ax_bar.set_ylim(0, 1.02)
    ax_bar.set_ylabel("Composite score")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_radar.set_title("(a) Relative decision dimensions", pad=26, fontsize=12, fontweight="bold")
    ax_bar.set_title("(b) Priority-weighted composite scores", fontsize=12, fontweight="bold")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    handles, labels_legend = ax_bar.get_legend_handles_labels()
    ax_bar.legend(handles, labels_legend, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "robustness_decision.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    render_decision_synthesis()
    print(f"Rendered JES red-muted figures into {OUT_DIR}")


if __name__ == "__main__":
    main()
