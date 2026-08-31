from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from jes2_plot_style import MODEL_COLORS, MODEL_ORDER, clean_axes, save_figure, setup_style


PREDICTION_COLUMNS = {"DM": "soc_cc", "HDM": "soc_cc", "HECM": "soc_ecm", "DD": "soc_pred"}
FILE_PATTERNS = {
    "DM": "soc_cc_fullcell_*.csv",
    "HDM": "soc_cc_soh_fullcell_*.csv",
    "HECM": "ecm_soc_fullcell_*.csv",
    "DD": "soc_pred_fullcell_*.csv",
}
NOISE_LEVELS = {
    "current_noise_low": 0.02,
    "current_noise_high": 0.10,
}


def load_run(root: Path, alias: str, model: str) -> pd.DataFrame:
    directory = root / alias / model
    path = next(directory.glob(FILE_PATTERNS[model]))
    frame = pd.read_csv(path)
    return frame.rename(columns={PREDICTION_COLUMNS[model]: "soc_pred"})


def align_pair(baseline: pd.DataFrame, noisy: pd.DataFrame) -> pd.DataFrame:
    columns = ["time_s", "soc_pred"]
    return baseline[columns].merge(noisy[columns], on="time_s", suffixes=("_baseline", "_noise"))


def select_reset_free_window(
    baseline: dict[str, pd.DataFrame],
    noisy: dict[str, pd.DataFrame],
    current: pd.DataFrame,
    duration_s: float = 1800.0,
) -> tuple[float, float]:
    common_start = max(float(frame.time_s.min()) for frame in [*baseline.values(), *noisy.values()])
    common_end = min(float(frame.time_s.max()) for frame in [*baseline.values(), *noisy.values()])
    starts = np.arange(common_start + 1800.0, common_end - duration_s, 600.0)
    best: tuple[float, float] | None = None

    reference = baseline["HECM"]
    for start in starts:
        end = start + duration_s
        ref = reference[(reference.time_s >= start) & (reference.time_s <= end)]
        cur = current[(current.time_s >= start) & (current.time_s <= end)]
        if len(ref) < 1200 or len(cur) < 1200:
            continue

        largest_step = 0.0
        for model in MODEL_ORDER:
            for frame in (baseline[model], noisy[model]):
                part = frame[(frame.time_s >= start) & (frame.time_s <= end)]
                if len(part) < 1200:
                    largest_step = np.inf
                    break
                largest_step = max(largest_step, float(part.soc_pred.diff().abs().max()))
        if largest_step > 0.004:
            continue

        soc_span = float(ref.soc_true.max() - ref.soc_true.min())
        current_activity = float(cur.I.std()) + 0.25 * float(cur.I.abs().mean())
        score = 4.0 * soc_span + current_activity
        if best is None or score > best[0]:
            best = (score, float(start))

    if best is None:
        raise RuntimeError("No reset-free 30 min noise window satisfies the selection criteria")
    return best[1], best[1] + duration_s


def primary_noise_statistics(macro: pd.DataFrame) -> pd.DataFrame:
    selected = macro[
        macro.alias.isin(NOISE_LEVELS)
        & macro.metric.eq("delta_mae")
        & (
            (macro.model.eq("DM") & macro.soh_condition.fillna("none").eq("none"))
            | (~macro.model.eq("DM") & macro.soh_condition.eq("lstm_h1"))
        )
    ].copy()
    selected["noise_std"] = selected.alias.map(NOISE_LEVELS)
    return selected


def local_step_sensitivity(
    baseline: dict[str, pd.DataFrame],
    noise_runs: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for alias, level in NOISE_LEVELS.items():
        for model in MODEL_ORDER:
            paired = align_pair(baseline[model], noise_runs[alias][model])
            output_difference = paired.soc_pred_noise - paired.soc_pred_baseline
            rows.append(
                {
                    "model": model,
                    "noise_std": level,
                    "p95_step": float(output_difference.diff().abs().quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def downsample(frame: pd.DataFrame, maximum: int = 1800) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected JES2 current-noise mechanism figure.")
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--noise_dir", type=Path, required=True)
    parser.add_argument("--macro_statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_style()
    baseline = {model: load_run(args.baseline_dir, "baseline", model) for model in MODEL_ORDER}
    noise_runs = {
        alias: {model: load_run(args.noise_dir, alias, model) for model in MODEL_ORDER}
        for alias in NOISE_LEVELS
    }
    current_baseline = baseline["HECM"][["time_s", "I"]]
    current_noise = noise_runs["current_noise_high"]["HECM"][["time_s", "I"]]
    start, end = select_reset_free_window(
        baseline,
        noise_runs["current_noise_high"],
        current_baseline,
    )

    fig = plt.figure(figsize=(14.2, 11.7))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.15, 1.0], hspace=0.34, wspace=0.25)
    ax_current = fig.add_subplot(grid[0, :])
    ax_soc = fig.add_subplot(grid[1, :])
    ax_global = fig.add_subplot(grid[2, 0])
    ax_local = fig.add_subplot(grid[2, 1])

    base_part = downsample(current_baseline[(current_baseline.time_s >= start) & (current_baseline.time_s <= end)])
    noise_part = downsample(current_noise[(current_noise.time_s >= start) & (current_noise.time_s <= end)])
    ax_current.plot((base_part.time_s - start) / 60.0, base_part.I, color="#225ea8", linewidth=2.0,
                    label="Baseline current")
    ax_current.plot((noise_part.time_s - start) / 60.0, noise_part.I, color="#d62728", linewidth=1.4,
                    label=r"Current with noise ($\sigma_I=0.10$ A)")
    ax_current.set(xlabel="Time [min]", ylabel="Current [A]", title="(a)")
    ax_current.legend(frameon=True, ncol=2, loc="upper right")
    clean_axes(ax_current)

    truth = baseline["HECM"]
    truth = downsample(truth[(truth.time_s >= start) & (truth.time_s <= end)])
    ax_soc.plot((truth.time_s - start) / 60.0, truth.soc_true, color="#111111", linewidth=1.3,
                label="Reference SOC")
    for model in MODEL_ORDER:
        base = baseline[model]
        base = downsample(base[(base.time_s >= start) & (base.time_s <= end)])
        noisy = noise_runs["current_noise_high"][model]
        noisy = downsample(noisy[(noisy.time_s >= start) & (noisy.time_s <= end)])
        ax_soc.plot((base.time_s - start) / 60.0, base.soc_pred, color=MODEL_COLORS[model],
                    linestyle="--", linewidth=1.2, alpha=0.55)
        ax_soc.plot((noisy.time_s - start) / 60.0, noisy.soc_pred, color=MODEL_COLORS[model],
                    linewidth=1.8, label=model)
    model_legend = ax_soc.legend(frameon=True, ncol=2, loc="upper left", title="Model colors")
    ax_soc.add_artist(model_legend)
    ax_soc.legend(
        handles=[
            Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label="Baseline prediction"),
            Line2D([0], [0], color="#666666", linewidth=1.8, label="Noise prediction"),
            Line2D([0], [0], color="#111111", linewidth=1.3, label="Reference SOC"),
        ],
        frameon=True,
        loc="lower right",
    )
    ax_soc.set(xlabel="Time [min]", ylabel="SOC [-]", title="(b)")
    clean_axes(ax_soc)

    statistics = primary_noise_statistics(pd.read_csv(args.macro_statistics))
    for model in MODEL_ORDER:
        part = statistics[statistics.model.eq(model)].sort_values("noise_std")
        lower = part["mean"] - part.ci_low
        upper = part.ci_high - part["mean"]
        ax_global.errorbar(
            part.noise_std,
            part["mean"],
            yerr=np.vstack([lower, upper]),
            color=MODEL_COLORS[model],
            marker="o",
            markersize=5.5,
            linewidth=1.8,
            capsize=3,
            label=model,
        )
    ax_global.axhline(0.0, color="#555555", linewidth=0.9)
    ax_global.set(
        xlabel=r"Current-noise std $\sigma_I$ [A]",
        ylabel=r"$\Delta$MAE [SOC]",
        title="(c) Six-cell global effect (95% CI)",
    )
    ax_global.set_xticks(sorted(NOISE_LEVELS.values()))
    ax_global.legend(frameon=True, ncol=2, loc="upper left")
    clean_axes(ax_global)

    local = local_step_sensitivity(baseline, noise_runs)
    for model in MODEL_ORDER:
        part = local[local.model.eq(model)].sort_values("noise_std")
        ax_local.plot(part.noise_std, part.p95_step, color=MODEL_COLORS[model], marker="o",
                      markersize=5.5, linewidth=1.8, label=model)
    ax_local.set(
        xlabel=r"Current-noise std $\sigma_I$ [A]",
        ylabel=r"p95 $|\Delta \hat{y}_k-\Delta \hat{y}_{k-1}|$ [SOC]",
        title="(d) C29 local output transmission",
    )
    ax_local.set_xticks(sorted(NOISE_LEVELS.values()))
    ax_local.legend(frameon=True, ncol=2, loc="upper left")
    clean_axes(ax_local)

    save_figure(fig, args.output)
    print(f"Selected reset-free window: {start:.1f}--{end:.1f} s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
