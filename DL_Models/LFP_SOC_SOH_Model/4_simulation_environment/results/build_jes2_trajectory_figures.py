from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_jes2_paper_results import draw_model_bars, metric_slice
from jes2_plot_style import MODEL_COLORS, MODEL_ORDER, clean_axes, save_figure, setup_style


PREDICTION_COLUMNS = {"DM": "soc_cc", "HDM": "soc_cc", "HECM": "soc_ecm", "DD": "soc_pred"}
FILE_PATTERNS = {
    "DM": "soc_cc_fullcell_*.csv",
    "HDM": "soc_cc_soh_fullcell_*.csv",
    "HECM": "ecm_soc_fullcell_*.csv",
    "DD": "soc_pred_fullcell_*.csv",
}


def load_run(root: Path, alias: str, model: str) -> tuple[pd.DataFrame, dict]:
    directory = root / alias / model
    path = next(directory.glob(FILE_PATTERNS[model]))
    frame = pd.read_csv(path)
    frame = frame.rename(columns={PREDICTION_COLUMNS[model]: "soc_pred"})
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return frame, summary


def downsample(frame: pd.DataFrame, maximum: int = 5000) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def trajectory_panel(ax, runs: dict[str, pd.DataFrame], start: float, end: float, title: str) -> None:
    reference = next(iter(runs.values()))
    mask = (reference.time_s >= start) & (reference.time_s <= end)
    truth = downsample(reference.loc[mask])
    ax.plot((truth.time_s - start) / 3600, truth.soc_true, color="#111111", linestyle="--", linewidth=1.5,
            label="Reference SOC")
    for model in MODEL_ORDER:
        frame = runs[model]
        part = downsample(frame[(frame.time_s >= start) & (frame.time_s <= end)])
        ax.plot((part.time_s - start) / 3600, part.soc_pred, color=MODEL_COLORS[model], linewidth=1.4, label=model)
    ax.set_xlabel("Time from panel start [h]")
    ax.set_ylabel("SOC")
    ax.set_title(title)
    clean_axes(ax)


def plot_bias(root: Path, statistics: Path, out: Path) -> None:
    baseline = {model: load_run(root, "baseline", model)[0] for model in MODEL_ORDER}
    biased = {model: load_run(root, "current_bias_3p0pct", model)[0] for model in MODEL_ORDER}
    stats = pd.read_csv(statistics)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    hecm_base, _ = load_run(root, "baseline", "HECM")
    hecm_bias, _ = load_run(root, "current_bias_3p0pct", "HECM")
    sample = downsample(hecm_base.iloc[:3600], 1800)
    sample_bias = hecm_bias.iloc[sample.index]
    axes[0].plot((sample.time_s - sample.time_s.iloc[0]) / 60, sample.I, color="#444444", label="Baseline")
    axes[0].plot((sample_bias.time_s - sample_bias.time_s.iloc[0]) / 60, sample_bias.I,
                 color="#d62728", linestyle="--", label="+3% gain error")
    axes[0].set(xlabel="Time [min]", ylabel="Measured current [A]", title="(a) Applied gain error")
    axes[0].legend(frameon=False)
    clean_axes(axes[0])

    for model in MODEL_ORDER:
        part = stats[stats.model == model].sort_values("bias_magnitude_pct")
        axes[1].errorbar(part.bias_magnitude_pct, part["mean"],
                         yerr=np.vstack([part["mean"] - part.ci_low, part.ci_high - part["mean"]]),
                         color=MODEL_COLORS[model], marker="o", capsize=3, linewidth=1.8, label=model)
    axes[1].set(xlabel="Current-gain error magnitude [%]", ylabel=r"Worst-case $\Delta$MAE [SOC]",
                title="(b) Six-cell worst-case sensitivity")
    axes[1].set_xticks([0, 0.5, 1.5, 3.0])
    clean_axes(axes[1])

    start = float(baseline["DM"].time_s.iloc[0])
    end = start + 12 * 3600
    reference = downsample(baseline["DM"][(baseline["DM"].time_s >= start) & (baseline["DM"].time_s <= end)])
    axes[2].plot((reference.time_s - start) / 3600, reference.soc_true, "k--", linewidth=1.3, label="Reference")
    for model in MODEL_ORDER:
        part = downsample(biased[model][(biased[model].time_s >= start) & (biased[model].time_s <= end)])
        axes[2].plot((part.time_s - start) / 3600, part.soc_pred, color=MODEL_COLORS[model], linewidth=1.25, label=model)
    axes[2].set(xlabel="Time [h]", ylabel="SOC", title="(c) C29 example at +3%")
    clean_axes(axes[2])
    axes[2].legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "Figure_05_Current_Bias.png")


def plot_initial(root: Path, aggregate: pd.DataFrame, out: Path) -> None:
    runs = {model: load_run(root, "initial_soc_error", model)[0] for model in MODEL_ORDER}
    start = float(runs["DM"].time_s.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.2))
    trajectory_panel(axes[0], runs, start, start + 6 * 3600, "(a) C29 recovery trajectory")
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    draw_model_bars(axes[1], metric_slice(aggregate, "initial_soc_error", "common_recovery_or_censor_time_h"),
                    "Recovery/censor time [h]", "(b) Six-cell recovery")
    draw_model_bars(axes[2], metric_slice(aggregate, "initial_soc_error", "common_recovery_excess_auc_soc_h"),
                    "Excess-error AUC [SOC h]", "(c) Six-cell recovery burden")
    fig.tight_layout()
    save_figure(fig, out / "Figure_07_Initial_State_Recovery.png")


def plot_gap(root: Path, aggregate: pd.DataFrame, out: Path) -> None:
    loaded = {model: load_run(root, "missing_gap_1h", model) for model in MODEL_ORDER}
    runs = {model: value[0] for model, value in loaded.items()}
    summary = loaded["DM"][1]
    gap_start, gap_end = float(summary["gap_start_time_s"]), float(summary["gap_end_time_s"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    trajectory_panel(axes[0], runs, gap_start - 3600, gap_end + 2 * 3600, "(a) C29 dropout transition")
    axes[0].axvspan(1.0, 2.0, color="#999999", alpha=0.18, label="Missing input")
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    draw_model_bars(axes[1], metric_slice(aggregate, "missing_gap_1h", "common_recovery_initial_abs_err"),
                    "Absolute SOC error", "(b) Six-cell error after resume")
    fig.tight_layout()
    save_figure(fig, out / "Figure_09_Burst_Dropout_Transition.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for model in MODEL_ORDER:
        frame = runs[model]
        part = downsample(frame[(frame.time_s >= gap_end) & (frame.time_s <= gap_end + 6 * 3600)])
        axes[0].plot((part.time_s - gap_end) / 3600, part.abs_err, color=MODEL_COLORS[model], label=model)
    axes[0].set(xlabel="Time after data resume [h]", ylabel="Absolute SOC error", title="(a) C29 recovery path")
    axes[0].legend(ncol=2, frameon=False)
    clean_axes(axes[0])
    draw_model_bars(axes[1], metric_slice(aggregate, "missing_gap_1h", "common_recovery_or_censor_time_h"),
                    "Recovery/censor time [h]", "(b) Six-cell recovery")
    fig.tight_layout()
    save_figure(fig, out / "Figure_10_Burst_Dropout_Recovery.png")


def plot_spikes(root: Path, aggregate: pd.DataFrame, out: Path) -> None:
    baseline = {model: load_run(root, "baseline", model)[0] for model in MODEL_ORDER}
    spikes = {model: load_run(root, "voltage_spikes", model)[0] for model in MODEL_ORDER}
    dd_delta = np.abs(spikes["DD"].soc_pred.to_numpy() - baseline["DD"].soc_pred.to_numpy())
    center = float(spikes["DD"].time_s.iloc[int(np.nanargmax(dd_delta))])
    start, end = center - 60, center + 180
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), gridspec_kw={"height_ratios": [1.05, 1.0]})
    reference = spikes["DM"]
    mask = (reference.time_s >= start) & (reference.time_s <= end)
    axes[0, 0].plot(reference.loc[mask, "time_s"] - center, reference.loc[mask, "soc_true"],
                    "k--", linewidth=1.4, label="Reference SOC")
    for model in MODEL_ORDER:
        part = spikes[model]
        part = part[(part.time_s >= start) & (part.time_s <= end)]
        axes[0, 0].plot(part.time_s - center, part.soc_pred, color=MODEL_COLORS[model], label=model)
    axes[0, 0].set(xlabel="Seconds relative to spike", ylabel="SOC", title="(a) C29 SOC response around spike")
    clean_axes(axes[0, 0])
    axes[0, 0].axvline(0, color="#111111", linestyle=":", linewidth=1.0)
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)
    for model in MODEL_ORDER:
        frame = spikes[model]
        base = baseline[model]
        mask = (frame.time_s >= start) & (frame.time_s <= end)
        part = frame.loc[mask]
        indices = part.index
        output_deviation = np.abs(part.soc_pred.to_numpy() - base.loc[indices, "soc_pred"].to_numpy())
        axes[0, 1].plot((part.time_s - center), output_deviation, color=MODEL_COLORS[model], label=model)
    axes[0, 1].axvline(0, color="#111111", linestyle="--", linewidth=1.0)
    axes[0, 1].set(xlabel="Seconds relative to spike", ylabel="Absolute output deviation from baseline",
                   title="(b) C29 transient model response")
    clean_axes(axes[0, 1])
    draw_model_bars(axes[1, 0], metric_slice(aggregate, "voltage_spikes", "delta_mae"),
                    r"$\Delta$MAE [SOC]", "(c) Six-cell global penalty")
    draw_model_bars(axes[1, 1], metric_slice(aggregate, "voltage_spikes", "jump_count_gt_5pct"),
                    "Output jumps >5%", "(d) Six-cell transient susceptibility")
    fig.tight_layout()
    save_figure(fig, out / "Figure_11_Voltage_Spike_Response.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JES2 trajectory-plus-aggregate figures.")
    parser.add_argument("--trajectory_dir", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--bias_statistics", type=Path, required=True)
    parser.add_argument("--figures_dir", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    aggregate = pd.read_csv(args.aggregate)
    plot_bias(args.trajectory_dir, args.bias_statistics, args.figures_dir)
    plot_initial(args.trajectory_dir, aggregate, args.figures_dir)
    plot_gap(args.trajectory_dir, aggregate, args.figures_dir)
    plot_spikes(args.trajectory_dir, aggregate, args.figures_dir)


if __name__ == "__main__":
    main()
