#!/usr/bin/env python3
"""Build revised Six-Cell supplements for Figures 11--13.

The script deliberately keeps the detailed dissertation figures untouched.  It
creates complementary views whose local and aggregate statements use the same
metric and replaces the current-bias heatmap entries with the signed paired
sweep. The available six-cell burst-dropout macro remains part of the analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "JES_2.0" / "results"
FIGURES = Path(__file__).resolve().parent / "Results"
INITIAL_RUNS = (
    Path(__file__).resolve().parents[4]
    / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns"
    / "jes2_initial_state_paired_sixcell_20260827_cuda/runs"
)
CAMPAIGNS = (
    Path(__file__).resolve().parents[4]
    / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns"
)

MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
PRIMARY_CONDITION = {"DM": "none", "HDM": "lstm_h1", "HECM": "lstm_h1", "DD": "lstm_h1"}
CELL_ORDER = ["C09", "C13", "C15", "C25", "C27", "C29"]
STATE_ORDER = ["fresh", "mid_life", "aged"]

ALIAS_ORDER = [
    "current_noise_low",
    "current_noise_high",
    "voltage_noise",
    "temperature_noise",
    "current_bias_0p5pct",
    "current_bias_1p5pct",
    "current_bias_3p0pct",
    "voltage_offset",
    "temperature_offset",
    "adc_quantization",
    "missing_samples_periodic",
    "missing_samples_random",
    "irregular_sampling_0p1s",
    "irregular_sampling_0p5s",
    "irregular_sampling_0p9s",
    "missing_gap_1h",
    "voltage_spikes",
]

ALIAS_LABEL = {
    "current_noise_low": "Current noise\n(0.02 A)",
    "current_noise_high": "Current noise\n(0.10 A)",
    "voltage_noise": "Voltage noise\n(0.01 V)",
    "temperature_noise": "Temperature noise\n(1.0 °C)",
    "current_bias_0p5pct": "Gain error\n(±0.5%)",
    "current_bias_1p5pct": "Gain error\n(±1.5%)",
    "current_bias_3p0pct": "Gain error\n(±3.0%)",
    "voltage_offset": "Voltage offset\n(0.02 V)",
    "temperature_offset": "Temperature offset\n(3 °C)",
    "adc_quantization": "ADC\nquantization",
    "missing_samples_periodic": "Periodic missing\n(1/50)",
    "missing_samples_random": "Random missing\n(2%)",
    "irregular_sampling_0p1s": "Timing jitter\n(±0.1 s)",
    "irregular_sampling_0p5s": "Timing jitter\n(±0.5 s)",
    "irregular_sampling_0p9s": "Timing jitter\n(±0.9 s)",
    "missing_gap_1h": "Burst dropout\n(1 h)",
    "voltage_spikes": "Voltage spikes\n(±0.20 V)",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keep = np.zeros(len(frame), dtype=bool)
    for model, condition in PRIMARY_CONDITION.items():
        keep |= (frame["model"] == model) & (frame["soh_condition"] == condition)
    return frame.loc[keep].copy()


def metric_rows(aggregate: pd.DataFrame, metric: str) -> pd.DataFrame:
    return primary_rows(aggregate[aggregate["metric"] == metric])


def lower_better(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        return pd.Series(1.0, index=values.index)
    return (high - values) / (high - low)


def revised_delta_matrix(aggregate: pd.DataFrame, signed_bias: pd.DataFrame) -> pd.DataFrame:
    rows = metric_rows(aggregate, "delta_mae")
    pivot = rows.pivot(index="model", columns="alias", values="mean").reindex(MODEL_ORDER)
    pivot = pivot.reindex(columns=ALIAS_ORDER)

    magnitude_alias = {0.5: "current_bias_0p5pct", 1.5: "current_bias_1p5pct", 3.0: "current_bias_3p0pct"}
    for magnitude, alias in magnitude_alias.items():
        replacement = signed_bias[np.isclose(signed_bias["bias_magnitude_pct"], magnitude)].set_index("model")["mean"]
        pivot.loc[:, alias] = replacement.reindex(MODEL_ORDER)

    return pivot


def load_spike_event_effects() -> pd.DataFrame:
    rows = []
    pattern = "jes2_full_C*_20260825/runs/*/*/voltage_spikes/seed_*/*/*/summary.json"
    for path in CAMPAIGNS.glob(pattern):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = path.parent.name
        condition = path.parent.parent.name
        expected_condition = "no_soh" if model == "DM" else PRIMARY_CONDITION[model]
        if condition != expected_condition:
            continue
        window = next(part for part in path.parts if part[:3] in CELL_ORDER and "_" in part)
        rows.append({
            "cell": window[:3],
            "window_soh_state": window[4:],
            "model": model,
            "event_error_penalty": float(summary["disturbed_mae"] - summary["calm_mae"]),
        })
    if not rows:
        raise FileNotFoundError("No voltage-spike summaries found in full-cell campaigns")
    return pd.DataFrame(rows)


def figure_11_spike_susceptibility(run_metrics: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    del run_metrics, aggregate
    event_effects = load_spike_event_effects()
    cell_values = event_effects.groupby(["cell", "model"], as_index=False)["event_error_penalty"].mean()
    state_values = event_effects.groupby(
        ["cell", "window_soh_state", "model"], as_index=False
    )["event_error_penalty"].mean()
    cell_values.to_csv(RESULTS / "jes2_spike_cell_susceptibility.csv", index=False)
    state_values.to_csv(RESULTS / "jes2_spike_state_susceptibility.csv", index=False)

    fig = plt.figure(figsize=(13.6, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.0), wspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        values = cell_values[cell_values["model"] == model]["event_error_penalty"].to_numpy(float)
        mean = float(np.mean(values))
        rng = np.random.default_rng(1127 + index)
        draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        ax.bar(index, mean, width=0.62, color=MODEL_COLORS[model], alpha=0.32,
               edgecolor=MODEL_COLORS[model], linewidth=1.8)
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]],
                    color="#111111", capsize=3, linewidth=1.2)
    ax.set_xticks(x, MODEL_ORDER)
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_ylabel(r"Spike-sample MAE penalty")
    ax.set_title("(a) Six-cell direct spike effect\n(bars: mean and 95% bootstrap CI)")
    ax.spines[["top", "right"]].set_visible(False)

    maximum = max(1e-6, float(np.nanmax(np.abs(state_values["event_error_penalty"]))))
    cmap = LinearSegmentedColormap.from_list("paired_effect", ["#b2182b", "#ffffff", "#2166ac"])
    for column, model in enumerate(["HECM", "DD"], start=1):
        panel = state_values[state_values["model"] == model].pivot(
            index="cell", columns="window_soh_state", values="event_error_penalty"
        ).reindex(index=CELL_ORDER, columns=STATE_ORDER)
        panel.columns = ["Fresh", "Mid-life", "Aged"]
        axis = fig.add_subplot(grid[0, column])
        image = axis.imshow(panel.to_numpy(float), aspect="auto", cmap=cmap, vmin=-maximum, vmax=maximum)
        axis.grid(False)
        axis.set_xticks(np.arange(3), panel.columns)
        axis.set_yticks(np.arange(len(CELL_ORDER)), CELL_ORDER)
        axis.set_title(
            f"({'b' if model == 'HECM' else 'c'}) {model} by cell and SOH state\n"
            r"cell values [$10^{-3}$ SOC]"
        )
        for row in range(panel.shape[0]):
            for col in range(panel.shape[1]):
                value = panel.iloc[row, col]
                label = "—" if not np.isfinite(value) else f"{1000 * value:.1f}"
                color = "white" if np.isfinite(value) and abs(value) > 0.55 * maximum else "#222222"
                axis.text(col, row, label, ha="center", va="center", color=color, fontsize=9)
        axis.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=fig.axes[1:3], fraction=0.022, pad=0.035)
    cbar.set_label("Spike-sample MAE penalty [SOC]")
    fig.suptitle("Voltage-spike response: direct event penalty across cells and aging states", fontsize=12)
    fig.subplots_adjust(top=0.78, left=0.06, right=0.90, bottom=0.12)
    save(fig, "Figure_11_Voltage_Spike_Response_REVISED")


def figure_12_heatmap(matrix: pd.DataFrame, output_path: Path | None = None) -> None:
    matrix.to_csv(RESULTS / "jes2_revised_delta_mae_matrix.csv")
    values = matrix.to_numpy(float)
    finite_limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    cmap = LinearSegmentedColormap.from_list(
        "diss_diverging", ["#1f77b4", "#f7f7f7", "#d62728"]
    )
    cmap.set_bad("#dedede")

    fig, ax = plt.subplots(figsize=(13.8, 4.4))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-finite_limit, vmax=finite_limit)
    ax.set_xticks(np.arange(len(matrix.columns)), [ALIAS_LABEL[a] for a in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.grid(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 * finite_limit else "#222222")
            else:
                ax.text(col, row, "n/a", ha="center", va="center", fontsize=6.5, color="#555555")
    fig.colorbar(image, ax=ax, label=r"Cell-macro $\Delta$MAE [SOC]", shrink=0.82)
    ax.set_title("Cross-scenario robustness across six holdout cells")
    fig.subplots_adjust(left=0.055, right=0.96, bottom=0.30, top=0.86)
    if output_path is None:
        save(fig, "Figure_12_Cross_Scenario_Heatmap_REVISED")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def initial_recovery_cell_means() -> pd.DataFrame:
    rows = []
    files = {
        "DM": ("no_soh", "soc_cc_fullcell_*.csv", "soc_cc"),
        "HDM": ("lstm_h1", "soc_cc_soh_fullcell_*.csv", "soc_cc"),
        "HECM": ("lstm_h1", "ecm_soc_fullcell_*.csv", "soc_ecm"),
        "DD": ("lstm_h1", "soc_pred_fullcell_*.csv", "soc_pred"),
    }
    for cell_dir in sorted(INITIAL_RUNS.glob("*")):
        if not cell_dir.is_dir():
            continue
        for window in sorted(cell_dir.glob("*")):
            if not window.is_dir():
                continue
            for model, (mode, pattern, prediction) in files.items():
                baseline_path = next((window / "baseline" / "seed_42" / mode / model).glob(pattern))
                shifted_path = next((window / "initial_soc_error" / "seed_42" / mode / model).glob(pattern))
                baseline = pd.read_csv(baseline_path, usecols=["time_s", prediction]).rename(columns={prediction: "correct"})
                shifted = pd.read_csv(shifted_path, usecols=["time_s", prediction]).rename(columns={prediction: "shifted"})
                paired = baseline.merge(shifted, on="time_s", how="inner")
                error = (paired["shifted"] - paired["correct"]).abs().to_numpy(float)
                elapsed_h = (paired["time_s"].to_numpy(float) - float(paired["time_s"].iloc[0])) / 3600.0
                violations = np.flatnonzero(error > 0.02)
                if len(violations) == 0:
                    stable_time_h = 0.0
                elif violations[-1] + 1 < len(error):
                    stable_time_h = float(elapsed_h[violations[-1] + 1])
                else:
                    stable_time_h = float(elapsed_h[-1])
                rows.append({"cell": cell_dir.name, "model": model, "stable_recovery_time_h": stable_time_h})
    frame = pd.DataFrame(rows)
    cell_means = frame.groupby(["cell", "model"], as_index=False)["stable_recovery_time_h"].mean()
    cell_means.to_csv(RESULTS / "jes2_initial_state_stable_recovery_cell_means.csv", index=False)
    return cell_means


def decision_scores(aggregate: pd.DataFrame, revised_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy_parts = []
    for metric in ["mae", "rmse", "p95_error"]:
        rows = metric_rows(aggregate, metric)
        values = rows[rows["alias"] == "baseline"].set_index("model")["mean"].reindex(MODEL_ORDER)
        accuracy_parts.append(lower_better(values))
    accuracy = pd.concat(accuracy_parts, axis=1).mean(axis=1)

    robust_aliases = [
        "current_noise_high",
        "voltage_noise",
        "temperature_noise",
        "current_bias_3p0pct",
        "missing_samples_random",
        "irregular_sampling_0p9s",
        "missing_gap_1h",
        "voltage_spikes",
        "adc_quantization",
    ]
    robust_parts = []
    for alias in robust_aliases:
        # Tiny negative deltas are stochastic variation, not a robustness bonus.
        penalty = revised_matrix[alias].clip(lower=0.0)
        robust_parts.append(lower_better(penalty))
    robustness = pd.concat(robust_parts, axis=1).mean(axis=1)

    recovery_cells = initial_recovery_cell_means()
    initial_recovery_raw = (
        recovery_cells.groupby("model")["stable_recovery_time_h"]
        .mean()
        .reindex(MODEL_ORDER)
    )
    initial_recovery = lower_better(initial_recovery_raw)

    burst_parts = []
    burst_raw = None
    for metric in [
        "common_recovery_or_censor_time_h",
        "common_recovery_censored_fraction",
        "common_recovery_excess_auc_soc_h",
    ]:
        rows = metric_rows(aggregate, metric)
        values = (
            rows[rows["alias"] == "missing_gap_1h"]
            .set_index("model")["mean"]
            .reindex(MODEL_ORDER)
        )
        if metric == "common_recovery_or_censor_time_h":
            burst_raw = values
        burst_parts.append(lower_better(values))
    if burst_raw is None or burst_raw.isna().any():
        raise ValueError("Complete six-cell burst-dropout recovery data are required for Figure 13.")
    burst_recovery = pd.concat(burst_parts, axis=1).mean(axis=1)
    recovery = pd.concat([initial_recovery, burst_recovery], axis=1).mean(axis=1)

    scores = pd.DataFrame(
        {
            "Model": MODEL_ORDER,
            "Accuracy": accuracy.reindex(MODEL_ORDER).to_numpy(float),
            "Robustness": robustness.reindex(MODEL_ORDER).to_numpy(float),
            "Recovery": recovery.reindex(MODEL_ORDER).to_numpy(float),
            "Initial stable recovery time [h]": initial_recovery_raw.to_numpy(float),
            "Burst recovery/censor time [h]": burst_raw.to_numpy(float),
            "Burst recovery score": burst_recovery.reindex(MODEL_ORDER).to_numpy(float),
        }
    )
    weights = {
        "Accuracy-weighted": (0.60, 0.20, 0.20),
        "Robustness-weighted": (0.20, 0.60, 0.20),
        "Recovery-weighted": (0.20, 0.20, 0.60),
    }
    profiles = scores[["Model"]].copy()
    for name, (wa, wr, wc) in weights.items():
        profiles[name] = wa * scores["Accuracy"] + wr * scores["Robustness"] + wc * scores["Recovery"]
    scores.to_csv(RESULTS / "jes2_revised_decision_dimensions.csv", index=False)
    profiles.to_csv(RESULTS / "jes2_revised_decision_profiles.csv", index=False)
    return scores, profiles


def figure_13_decision(
    scores: pd.DataFrame,
    profiles: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    labels = ["Accuracy", "Robustness", "Recovery"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(13.6, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.45), wspace=0.32)
    radar = fig.add_subplot(grid[0, 0], projection="polar")
    bars = fig.add_subplot(grid[0, 1])

    radar.set_theta_offset(np.pi / 2)
    radar.set_theta_direction(-1)
    radar.set_xticks(angles, labels)
    radar.tick_params(axis="x", pad=10)
    radar.set_yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"])
    radar.set_ylim(0.0, 1.0)
    for row in scores.itertuples(index=False):
        values = np.array([row.Accuracy, row.Robustness, row.Recovery], dtype=float)
        values = np.r_[values, values[0]]
        radar.plot(
            closed_angles, values, color=MODEL_COLORS[row.Model], lw=2.0,
            marker="o", markersize=5.0, markerfacecolor=MODEL_COLORS[row.Model],
            markeredgecolor=MODEL_COLORS[row.Model], markeredgewidth=0.8, label=row.Model,
        )
        radar.fill(closed_angles, values, color=MODEL_COLORS[row.Model], alpha=0.12)
    radar.set_rlabel_position(0)
    radar.grid(color="#d8d8d8", linewidth=0.8)
    radar.spines["polar"].set_color("#555555")
    radar.set_title("(a) Relative decision dimensions", pad=24)

    profile_names = [c for c in profiles.columns if c != "Model"]
    x = np.arange(len(profile_names), dtype=float) * 1.18
    width = 0.14
    offsets = np.linspace(-0.285, 0.285, len(MODEL_ORDER))
    for index, row in profiles.iterrows():
        model = row["Model"]
        bars.bar(x + offsets[index], row[profile_names].to_numpy(float), width=width,
                 color=MODEL_COLORS[model], alpha=0.42, edgecolor=MODEL_COLORS[model],
                 linewidth=1.5, label=model, zorder=3)
    bars.set_xticks(x, profile_names)
    bars.set_ylim(0.0, 1.02)
    bars.set_yticks(np.linspace(0.0, 1.0, 6))
    bars.set_ylabel("Composite score")
    bars.set_title("(b) Priority-weighted composite scores")
    bars.grid(axis="y", color="#d8d8d8", linewidth=0.8, zorder=0)
    bars.grid(axis="x", visible=False)
    bars.spines[["top", "right"]].set_visible(False)
    bars.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.86)
    if output_path is None:
        save(fig, "Figure_13_Decision_Synthesis_REVISED")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build revised six-cell JES figures.")
    parser.add_argument("--figure-12-only", action="store_true")
    parser.add_argument("--figure-12-output", type=Path, default=None)
    parser.add_argument("--figure-13-only", action="store_true")
    parser.add_argument("--figure-13-output", type=Path, default=None)
    args = parser.parse_args()

    setup_style()
    aggregate = pd.read_csv(RESULTS / "jes2_macro_statistics.csv")
    signed_bias = pd.read_csv(RESULTS / "jes2_signed_current_bias_statistics.csv")
    matrix = revised_delta_matrix(aggregate, signed_bias)
    if args.figure_12_only:
        figure_12_heatmap(matrix, args.figure_12_output)
        return
    if args.figure_13_only:
        scores, profiles = decision_scores(aggregate, matrix)
        figure_13_decision(scores, profiles, args.figure_13_output)
        print(scores.round(4).to_string(index=False))
        print(profiles.round(4).to_string(index=False))
        return

    run_metrics = pd.read_csv(RESULTS / "jes2_run_metrics.csv")
    figure_11_spike_susceptibility(run_metrics, aggregate)
    figure_12_heatmap(matrix)
    scores, profiles = decision_scores(aggregate, matrix)
    figure_13_decision(scores, profiles)
    print(scores.round(4).to_string(index=False))
    print(profiles.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
