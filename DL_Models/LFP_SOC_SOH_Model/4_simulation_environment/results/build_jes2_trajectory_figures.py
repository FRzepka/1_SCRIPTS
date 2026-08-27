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


def spike_event_penalties(campaigns: Path) -> pd.DataFrame:
    rows = []
    pattern = "jes2_full_C*_20260825/runs/*/*/voltage_spikes/seed_*/*/*/summary.json"
    for path in campaigns.glob(pattern):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = path.parent.name
        condition = path.parent.parent.name
        expected = "no_soh" if model == "DM" else "lstm_h1"
        if model not in MODEL_ORDER or condition != expected:
            continue
        window = next(part for part in path.parts if part[:3] in {"C09", "C13", "C15", "C25", "C27", "C29"}
                      and "_" in part)
        rows.append({"cell": window[:3], "model": model,
                     "penalty": float(summary["disturbed_mae"] - summary["calm_mae"])})
    raw = pd.DataFrame(rows)
    result = []
    for index, model in enumerate(MODEL_ORDER):
        values = raw[raw.model == model].groupby("cell").penalty.mean().to_numpy()
        rng = np.random.default_rng(1127 + index)
        draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        result.append({"model": model, "mean": values.mean(), "ci_low": np.percentile(draws, 2.5),
                       "ci_high": np.percentile(draws, 97.5)})
    return pd.DataFrame(result)


def plot_spikes(root: Path, aggregate: pd.DataFrame, run_metrics: Path, out: Path) -> None:
    del aggregate, run_metrics
    baseline = {model: load_run(root, "baseline", model)[0] for model in MODEL_ORDER}
    spikes = {model: load_run(root, "voltage_spikes", model)[0] for model in MODEL_ORDER}

    # Locate real injection samples and select a typical DD response rather than
    # centering the panel on a later model deviation.
    hecm_voltage = spikes["HECM"]["U"].to_numpy()
    base_voltage = baseline["HECM"]["U"].to_numpy()
    event_indices = np.flatnonzero(np.abs(hecm_voltage - base_voltage) > 0.1)
    event_indices = event_indices[(event_indices > 0) & (event_indices < len(hecm_voltage) - 1)]
    common_start = max(float(frame.time_s.iloc[0]) for frame in baseline.values()) + 60
    event_indices = event_indices[spikes["HECM"].time_s.iloc[event_indices].to_numpy() >= common_start]
    maximum_baseline_errors = []
    for candidate in event_indices:
        event_time = float(spikes["HECM"].time_s.iloc[candidate])
        errors = []
        for model in MODEL_ORDER:
            index = int(np.searchsorted(baseline[model].time_s.to_numpy(), event_time))
            errors.append(abs(float(baseline[model].soc_pred.iloc[index] - baseline[model].soc_true.iloc[index])))
        maximum_baseline_errors.append(max(errors))
    event_index = int(event_indices[int(np.nanargmin(maximum_baseline_errors))])
    center = float(spikes["HECM"].time_s.iloc[event_index])
    start, end = center - 60, center + 180
    rel_grid = np.arange(-60, 181, dtype=float)
    fig = plt.figure(figsize=(13.6, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.26)
    ax_soc = fig.add_subplot(grid[0, :])
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_local = fig.add_subplot(grid[1, 1])
    reference = spikes["DM"]
    ref_mask = (reference.time_s >= start) & (reference.time_s <= end)
    ax_soc.plot(reference.loc[ref_mask, "time_s"] - center, reference.loc[ref_mask, "soc_true"],
                color="#111111", linestyle="--", linewidth=1.7, label="Reference SOC")
    for model in MODEL_ORDER:
        frame = spikes[model]
        base = baseline[model]
        mask = (frame.time_s >= start) & (frame.time_s <= end)
        part = frame.loc[mask]
        indices = part.index
        base_part = base.loc[indices]
        pre_mask = (part.time_s >= center - 60) & (part.time_s <= center - 10)
        offset = float((base_part.loc[pre_mask, "soc_pred"] - base_part.loc[pre_mask, "soc_true"]).mean())
        ax_soc.plot(part.time_s - center, base_part.soc_pred - offset, color=MODEL_COLORS[model],
                    linestyle="--", linewidth=1.3, alpha=0.65)
        ax_soc.plot(part.time_s - center, part.soc_pred - offset, color=MODEL_COLORS[model],
                    linewidth=1.8, label=model)

        aligned = []
        model_times = frame.time_s.to_numpy()
        for event_time in spikes["HECM"].time_s.iloc[event_indices[:40]].to_numpy():
            event = int(np.searchsorted(model_times, event_time))
            if event < 60 or event + 180 >= len(frame):
                continue
            window = frame.iloc[event - 60:event + 181]
            pre_error = float(window.abs_err.iloc[50:60].mean())
            aligned.append(window.abs_err.to_numpy() - pre_error)
        if aligned:
            values = np.vstack(aligned)
            mean = np.nanmean(values, axis=0)
            low, high = np.nanpercentile(values, [25, 75], axis=0)
            ax_local.plot(rel_grid, mean, color=MODEL_COLORS[model], linewidth=1.8, label=model)
            ax_local.fill_between(rel_grid, low, high, color=MODEL_COLORS[model], alpha=0.14)

    penalties = spike_event_penalties(root.parent)
    draw_model_bars(ax_bar, penalties, r"Spike-sample $\Delta$MAE [SOC]",
                    "(b) Direct spike effect across six cells")
    ax_soc.set(xlabel="Seconds relative to representative voltage spike", ylabel="Offset-aligned SOC",
               title="(a) C15 mid-life response after pre-spike offset alignment")
    ax_local.set(xlabel="Seconds relative to voltage spike", ylabel="Excess absolute error",
                 title="(c) C15 aligned response over 40 spikes")
    for axis in (ax_soc, ax_local):
        axis.axvline(0, color="#111111", linestyle="--", linewidth=1.1)
        axis.legend(frameon=False, ncol=3, fontsize=8)
        clean_axes(axis)
    clean_axes(ax_bar)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.98)
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
