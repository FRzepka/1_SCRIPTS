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


def plot_gap(root: Path, out: Path) -> None:
    loaded = {model: load_run(root, "missing_gap_1h", model) for model in MODEL_ORDER}
    runs = {model: value[0] for model, value in loaded.items()}
    baseline = {model: load_run(root, "baseline", model)[0] for model in MODEL_ORDER}
    summary = loaded["DM"][1]
    gap_start, gap_end = float(summary["gap_start_time_s"]), float(summary["gap_end_time_s"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    start, end = gap_start - 3600, gap_end + 2 * 3600
    for model in MODEL_ORDER:
        base = downsample(baseline[model][(baseline[model].time_s >= start) & (baseline[model].time_s <= end)])
        drop = downsample(runs[model][(runs[model].time_s >= start) & (runs[model].time_s <= end)])
        axes[0].plot((base.time_s - gap_start) / 3600, base.soc_pred, color=MODEL_COLORS[model],
                     linestyle="--", linewidth=1.1, alpha=0.75)
        axes[0].plot((drop.time_s - gap_start) / 3600, drop.soc_pred, color=MODEL_COLORS[model],
                     linewidth=1.5, label=model)
    axes[0].axvspan(0.0, 1.0, color="#999999", alpha=0.18, label="Missing input")
    axes[0].set(xlabel="Time from dropout start [h]", ylabel="SOC",
                title="(a) C29 baseline (dashed) vs dropout (solid)")
    clean_axes(axes[0])
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    for model in MODEL_ORDER:
        frame = runs[model]
        part = downsample(frame[(frame.time_s >= gap_end) & (frame.time_s <= gap_end + 4 * 3600)])
        axes[1].plot((part.time_s - gap_end) / 3600, part.abs_err, color=MODEL_COLORS[model], label=model)
    axes[1].axhline(0.02, color="#222222", linestyle="--", linewidth=1.2, label="2% error threshold")
    axes[1].set(xlabel="Time after measurements resume [h]", ylabel="Absolute SOC error",
                title="(b) C29 error after resume")
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    clean_axes(axes[1])
    fig.tight_layout()
    save_figure(fig, out / "Figure_09_Burst_Dropout_Transition.png")


def spike_jump_deltas(run_metrics: Path) -> pd.DataFrame:
    raw = pd.read_csv(run_metrics)
    raw = raw[((raw.model == "DM") & raw.soh_condition.fillna("none").eq("none")) |
              ((raw.model != "DM") & raw.soh_condition.eq("lstm_h1"))]
    rows = []
    for index, model in enumerate(MODEL_ORDER):
        baseline = raw[(raw.model == model) & (raw.alias == "baseline")].drop_duplicates(
            ["cell", "window_id"]
        ).set_index(["cell", "window_id"])["jump_count_gt_5pct"]
        spikes = raw[(raw.model == model) & (raw.alias == "voltage_spikes")].groupby(
            ["cell", "window_id"]
        ).jump_count_gt_5pct.mean()
        paired = (spikes - baseline).groupby(level="cell").mean().dropna().to_numpy()
        rng = np.random.default_rng(1127 + index)
        draws = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
        rows.append({"model": model, "mean": paired.mean(), "ci_low": np.percentile(draws, 2.5),
                     "ci_high": np.percentile(draws, 97.5)})
    return pd.DataFrame(rows)


def plot_spikes(root: Path, aggregate: pd.DataFrame, run_metrics: Path, out: Path) -> None:
    del aggregate, run_metrics
    baseline = {model: load_run(root, "baseline", model)[0] for model in MODEL_ORDER}
    spikes = {model: load_run(root, "voltage_spikes", model)[0] for model in MODEL_ORDER}

    # Locate real injection samples from the measured voltage, then select the
    # event with the largest immediate causal HECM output change.
    hecm_voltage = spikes["HECM"]["U"].to_numpy()
    base_voltage = baseline["HECM"]["U"].to_numpy()
    event_indices = np.flatnonzero(np.abs(hecm_voltage - base_voltage) > 0.1)
    event_indices = event_indices[(event_indices > 0) & (event_indices < len(hecm_voltage) - 1)]
    hecm_delta = spikes["HECM"].soc_pred.to_numpy() - baseline["HECM"].soc_pred.to_numpy()
    causal_jumps = np.abs(hecm_delta[event_indices] - hecm_delta[event_indices - 1])
    event_index = int(event_indices[int(np.nanargmax(causal_jumps))])
    center = float(spikes["HECM"].time_s.iloc[event_index])
    start, end = center - 12, center + 40

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    hecm_spike = spikes["HECM"]
    hecm_base = baseline["HECM"]
    mask = (hecm_spike.time_s >= start) & (hecm_spike.time_s <= end)
    time = hecm_spike.loc[mask, "time_s"] - center
    axes[0].plot(time, hecm_base.loc[mask, "U"], color="#444444", linestyle="--",
                 linewidth=1.5, label="Baseline voltage")
    axes[0].plot(time, hecm_spike.loc[mask, "U"], color=MODEL_COLORS["HECM"],
                 linewidth=1.5, label="Voltage with spike")
    axes[0].set(xlabel="Seconds relative to spike", ylabel="Measured voltage [V]",
                title="(a) Injected C15 voltage spike")
    axes[0].legend(frameon=False, fontsize=8)
    clean_axes(axes[0])

    axes[1].plot(time, hecm_base.loc[mask, "soc_pred"], color=MODEL_COLORS["HECM"],
                 linestyle="--", linewidth=1.6, label="HECM baseline")
    axes[1].plot(time, hecm_spike.loc[mask, "soc_pred"], color=MODEL_COLORS["HECM"],
                 linewidth=1.6, label="HECM with spike")
    axes[1].plot(time, hecm_spike.loc[mask, "soc_true"], color="#222222", linewidth=1.1,
                 alpha=0.75, label="Reference SOC")
    axes[1].set(xlabel="Seconds relative to spike", ylabel="SOC",
                title="(b) HECM baseline vs disturbed output")
    axes[1].legend(frameon=False, fontsize=8)
    clean_axes(axes[1])

    for model in MODEL_ORDER:
        frame = spikes[model]
        base = baseline[model]
        mask = (frame.time_s >= start) & (frame.time_s <= end)
        part = frame.loc[mask]
        indices = part.index
        deviation = part.soc_pred.to_numpy() - base.loc[indices, "soc_pred"].to_numpy()
        before = np.flatnonzero(part.time_s.to_numpy() < center)
        offset = deviation[before[-1]] if len(before) else deviation[0]
        axes[2].plot(part.time_s - center, deviation - offset,
                     color=MODEL_COLORS[model], linewidth=1.5, label=model)
    axes[2].set(xlabel="Seconds relative to spike", ylabel="Output change from pre-spike offset",
                title="(c) Causal transient response")
    axes[2].legend(frameon=False, fontsize=8)
    clean_axes(axes[2])
    for axis in axes:
        axis.axvline(0, color="#111111", linestyle=":", linewidth=1.0)
    fig.tight_layout()
    save_figure(fig, out / "Figure_11_Voltage_Spike_Response.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JES2 trajectory-plus-aggregate figures.")
    parser.add_argument("--trajectory_dir", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--run_metrics", type=Path, required=True)
    parser.add_argument("--spike_trajectory_dir", type=Path, required=True)
    parser.add_argument("--bias_statistics", type=Path, required=True)
    parser.add_argument("--figures_dir", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    aggregate = pd.read_csv(args.aggregate)
    plot_bias(args.trajectory_dir, args.bias_statistics, args.figures_dir)
    plot_initial(args.trajectory_dir, aggregate, args.figures_dir)
    plot_gap(args.trajectory_dir, args.figures_dir)
    plot_spikes(args.spike_trajectory_dir, aggregate, args.run_metrics, args.figures_dir)


if __name__ == "__main__":
    main()
