from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
TABLE_DIR = ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_tables_v4"
OUT_DIR = ROOT / "LATEX/EAAI/paper_robustness_benchmark/Decision_Diagrams/v4"
PAPER_FIG_DIR = ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_figures_v4"

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
    "DM": "#6e2fc4",
    "HDM": "#08bdba",
    "HECM": "#d4bbff",
    "DD": "#4589ff",
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
    "Accuracy-first": {"Accuracy": 0.60, "Robustness": 0.25, "Recovery": 0.15},
    "Robustness-first": {"Accuracy": 0.20, "Robustness": 0.65, "Recovery": 0.15},
    "Recovery-first": {"Accuracy": 0.20, "Robustness": 0.20, "Recovery": 0.60},
}

SCENARIO_SHORT = {
    "Current noise (high)": "Current\nnoise",
    "Current bias": "Current\nbias",
    "Irregular sampling": "Irregular\nsampling",
    "Burst dropout": "Burst\ndropout",
    "Missing samples": "Missing\nsamples",
    "Voltage spikes": "Voltage\nspikes",
    "Temperature noise": "Temperature\nnoise",
    "Voltage noise": "Voltage\nnoise",
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


def save_markdown(
    scores: pd.DataFrame,
    robustness_raw: pd.DataFrame,
    recovery_raw: pd.DataFrame,
) -> None:
    lines = [
        "# Decision Diagram Notes (v4)",
        "",
        "These diagrams are exploratory decision aids built from the existing `paper_tables_v4` results.",
        "They do not replace the raw benchmark results and are not meant as a universal ranking.",
        "",
        "## Inputs",
        "",
        "- Baseline table: `table_baseline.csv`",
        "- Disturbed-scenario table: `table_key_results.csv`",
        "- Local behaviour table: `table_local_behaviour.csv`",
        "",
        "## Meta-scores",
        "",
        "All scores are normalized to `[0, 1]` across the four estimator classes, with higher being better.",
        "",
        "### Accuracy",
        "",
        "Average of min-max normalized baseline `MAE`, `RMSE`, and `P95`.",
        "",
        "### Robustness",
        "",
        "Average of min-max normalized disturbed-scenario `delta_MAE` over:",
    ]
    lines.extend([f"- {name}" for name in ROBUSTNESS_SCENARIOS])
    lines.extend(
        [
            "",
            "### Recovery",
            "",
            "Average of penalized lower-is-better scores for:",
            "- `recovery_time_to_baseline_band_strict_h`",
            "- `recovery_time_to_baseline_band_fair_h`",
            "",
            "Missing recovery times are treated as non-recovery and mapped to a zero score using a penalty larger than the slowest finite recovery time.",
            "",
            "## Decision profiles",
            "",
            "These are not presented as a single universal truth. They are only alternative weighting views:",
        ]
    )
    for profile, weights in PROFILE_WEIGHTS.items():
        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
        lines.append(f"- `{profile}`: {weights_str}")
    lines.extend(
        [
            "",
            "## Final normalized scores",
            "",
            scores.round(4).to_markdown(index=False),
            "",
            "## Robustness raw inputs (`delta_MAE`)",
            "",
            robustness_raw.round(6).to_markdown(index=False),
            "",
            "## Recovery raw inputs",
            "",
            recovery_raw.round(6).to_markdown(index=False),
        ]
    )
    (OUT_DIR / "decision_method_v4.md").write_text("\n".join(lines), encoding="utf-8")


def build_radar(meta_scores: pd.DataFrame) -> None:
    labels = ["Accuracy", "Robustness", "Recovery"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7.8, 6.8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.35)

    for _, row in meta_scores.iterrows():
        short = row["Model"]
        values = row[labels].to_numpy(dtype=float)
        values = np.concatenate([values, [values[0]]])
        color = CLASS_COLORS[short]
        ax.plot(angles, values, linewidth=2.2, color=color, label=short)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_facecolor("white")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decision_radar_v4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_profiles(meta_scores: pd.DataFrame) -> None:
    profile_scores = meta_scores[["Model"]].copy()
    for profile, weights in PROFILE_WEIGHTS.items():
        profile_scores[profile] = (
            meta_scores["Accuracy"] * weights["Accuracy"]
            + meta_scores["Robustness"] * weights["Robustness"]
            + meta_scores["Recovery"] * weights["Recovery"]
        )

    profile_scores.to_csv(OUT_DIR / "decision_profile_scores_v4.csv", index=False)

    x = np.arange(len(PROFILE_WEIGHTS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(profile_scores))

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for i, (_, row) in enumerate(profile_scores.iterrows()):
        short = row["Model"]
        ax.bar(
            x + offsets[i],
            row[list(PROFILE_WEIGHTS.keys())].to_numpy(dtype=float),
            width=width,
            label=short,
            color=CLASS_COLORS[short],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(PROFILE_WEIGHTS.keys()))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Composite score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decision_profiles_v4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_heatmap(robustness_score_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    im = ax.imshow(robustness_score_matrix.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(robustness_score_matrix.columns)))
    ax.set_xticklabels(robustness_score_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(robustness_score_matrix.index)))
    ax.set_yticklabels(robustness_score_matrix.index)
    for i in range(len(robustness_score_matrix.index)):
        for j in range(len(robustness_score_matrix.columns)):
            value = robustness_score_matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized robustness score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness_heatmap_v4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_delta_mae_heatmap(robustness_raw: pd.DataFrame) -> None:
    shown = robustness_raw.rename(index=CLASS_SHORT, columns=SCENARIO_SHORT)
    vmax = float(np.nanmax(np.abs(shown.to_numpy(dtype=float))))
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue_purple",
        [
            (0.00, "#2563eb"),
            (0.42, "#e8f1ff"),
            (0.50, "#ffffff"),
            (0.70, "#d8c3ff"),
            (1.00, "#5b21b6"),
        ],
    )
    norm = SymLogNorm(linthresh=0.005, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    im = ax.imshow(shown.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(shown.columns)))
    ax.set_xticklabels(shown.columns)
    ax.set_yticks(np.arange(len(shown.index)))
    ax.set_yticklabels(shown.index)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            val = shown.iloc[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.5, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$MAE")
    cbar.set_ticks([-0.30, -0.10, -0.03, -0.01, -0.003, 0.0, 0.003, 0.01, 0.03, 0.10, 0.30])
    fig.tight_layout()
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "Figure_9_delta_mae_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_combined_decision_figure(meta_scores: pd.DataFrame) -> None:
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

    fig = plt.figure(figsize=(13.8, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.24)
    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=12)
    ax_radar.tick_params(axis="x", pad=2)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
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
    handles, labels_legend = ax_bar.get_legend_handles_labels()
    fig.legend(handles, labels_legend, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    fig.subplots_adjust(top=0.88, left=0.04, right=0.98, bottom=0.11, wspace=0.24)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "Figure_10_decision_synthesis.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    baseline = pd.read_csv(TABLE_DIR / "table_baseline.csv")
    key = pd.read_csv(TABLE_DIR / "table_key_results.csv")
    local = pd.read_csv(TABLE_DIR / "table_local_behaviour.csv")

    baseline = baseline.set_index("class").loc[CLASS_ORDER]
    key = key[key["scenario_label"].isin(ROBUSTNESS_SCENARIOS)].copy()
    key["class"] = pd.Categorical(key["class"], CLASS_ORDER, ordered=True)
    key = key.sort_values(["scenario_label", "class"])

    local = local[
        local["local_metric"].isin(
            [
                "recovery_time_to_baseline_band_strict_h",
                "recovery_time_to_baseline_band_fair_h",
            ]
        )
    ].copy()

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
    recovery_scores = recovery_score_matrix.mean(axis=1).rename("Recovery")

    meta_scores = pd.concat([accuracy_scores, robustness_scores, recovery_scores], axis=1).reset_index()
    meta_scores = meta_scores.rename(columns={"class": "Class"})
    meta_scores["Model"] = meta_scores["Class"].map(CLASS_SHORT)
    meta_scores = meta_scores[["Model", "Class", "Accuracy", "Robustness", "Recovery"]]
    meta_scores.to_csv(OUT_DIR / "decision_scores_v4.csv", index=False)

    build_radar(meta_scores)
    build_profiles(meta_scores)
    build_heatmap(robustness_score_matrix.rename(index=CLASS_SHORT, columns=SCENARIO_SHORT))
    build_delta_mae_heatmap(robustness_raw)
    build_combined_decision_figure(meta_scores)
    save_markdown(
        scores=meta_scores,
        robustness_raw=robustness_raw.reset_index().rename(columns={"class": "Class"}),
        recovery_raw=recovery_raw.reset_index().rename(columns={"class": "Class"}),
    )


if __name__ == "__main__":
    main()
