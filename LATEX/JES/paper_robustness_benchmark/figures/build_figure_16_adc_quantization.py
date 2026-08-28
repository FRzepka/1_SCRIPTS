#!/usr/bin/env python3
"""Combine ADC signal detail with the current six-cell performance result."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
RESULTS = PAPER / "JES_2.0/results"
OUTPUT = PAPER / "figures/Results/All Cells/Figure_16_ADC_Quantization.png"
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
CELL = "MGFarm_18650_C07"

sys.path.insert(0, str(SIMULATION))
from robustness_common import apply_measurement_scenario, load_cell_dataframe  # noqa: E402


MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
PRIMARY_CONDITION = {"DM": "none", "HDM": "lstm_h1", "HECM": "lstm_h1", "DD": "lstm_h1"}


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


def pick_dynamic_window(frame: pd.DataFrame, seconds: int = 300, after_s: int = 600) -> tuple[float, float]:
    current = frame["Current[A]"].to_numpy(float)
    time_s = frame["Testtime[s]"].to_numpy(float)
    eligible = np.flatnonzero(time_s >= after_s)
    if len(eligible) < seconds:
        return float(time_s[0]), float(time_s[min(len(time_s) - 1, seconds)])

    candidates = eligible[: len(eligible) - seconds + 1]
    sampled = candidates[::10]
    scores = np.asarray([np.std(current[index : index + seconds]) for index in sampled])
    best = int(sampled[int(np.argmax(scores))])
    return float(time_s[best]), float(time_s[best + seconds - 1])


def metric_rows(aggregate: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = aggregate[(aggregate["alias"] == "adc_quantization") & (aggregate["metric"] == metric)]
    keep = np.zeros(len(rows), dtype=bool)
    for model, condition in PRIMARY_CONDITION.items():
        keep |= (rows["model"] == model) & (rows["soh_condition"] == condition)
    return rows.loc[keep].set_index("model").reindex(MODEL_ORDER)


def clean_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    setup_style()
    raw = load_cell_dataframe(DATA_ROOT, CELL)
    quantized, _ = apply_measurement_scenario(
        raw.copy(),
        "adc_quantization",
        SimpleNamespace(
            quantize_current_a=0.01,
            quantize_voltage_v=0.005,
            quantize_temp_c=0.5,
            seed=42,
        ),
    )
    start_s, end_s = pick_dynamic_window(raw)
    raw_segment = raw[(raw["Testtime[s]"] >= start_s) & (raw["Testtime[s]"] <= end_s)]
    quant_segment = quantized[
        (quantized["Testtime[s]"] >= start_s) & (quantized["Testtime[s]"] <= end_s)
    ]
    relative_s = raw_segment["Testtime[s]"].to_numpy(float) - start_s

    aggregate = pd.read_csv(RESULTS / "jes2_macro_statistics.csv")
    delta = metric_rows(aggregate, "delta_mae")
    adc_mae = metric_rows(aggregate, "mae")
    if delta[["mean", "ci_low", "ci_high"]].isna().any().any() or adc_mae["mean"].isna().any():
        raise ValueError("Complete six-cell ADC statistics are required.")

    figure = plt.figure(figsize=(13.5, 7.5))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.22), hspace=0.30, wspace=0.25)
    current_axis = figure.add_subplot(grid[0, 0])
    voltage_axis = figure.add_subplot(grid[1, 0], sharex=current_axis)
    bars_axis = figure.add_subplot(grid[:, 1])

    current_axis.plot(relative_s, raw_segment["Current[A]"], color="#222222", lw=1.8, label="Raw")
    current_axis.step(
        relative_s,
        quant_segment["Current[A]"],
        where="post",
        color="#d62728",
        alpha=0.68,
        lw=1.5,
        label="Quantized",
    )
    current_axis.set_ylabel("Current [A]")
    current_axis.set_title("(a) Current Quantization")
    current_axis.legend(frameon=False, loc="upper right")
    current_axis.tick_params(labelbottom=False)

    voltage_axis.plot(relative_s, raw_segment["Voltage[V]"], color="#222222", lw=1.8, label="Raw")
    voltage_axis.step(
        relative_s,
        quant_segment["Voltage[V]"],
        where="post",
        color="#d62728",
        alpha=0.68,
        lw=1.5,
        label="Quantized",
    )
    voltage_axis.set_ylabel("Voltage [V]")
    voltage_axis.set_xlabel("Seconds within Representative Local Window")
    voltage_axis.set_title("(b) Voltage Quantization")
    voltage_axis.legend(frameon=False, loc="upper right")

    x = np.arange(len(MODEL_ORDER), dtype=float)
    values = delta["mean"].to_numpy(float)
    low = delta["ci_low"].to_numpy(float)
    high = delta["ci_high"].to_numpy(float)
    bars = bars_axis.bar(
        x,
        values,
        width=0.68,
        color=[to_rgba(MODEL_COLORS[model], 0.34) for model in MODEL_ORDER],
        edgecolor=[MODEL_COLORS[model] for model in MODEL_ORDER],
        linewidth=1.6,
        zorder=3,
    )
    bars_axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="none",
        ecolor="#111111",
        elinewidth=1.35,
        capsize=4.0,
        capthick=1.35,
        zorder=4,
    )
    span = float(max(high.max(), 0.0) - min(low.min(), 0.0))
    label_offset = 0.035 * span
    for index, (bar, model) in enumerate(zip(bars, MODEL_ORDER)):
        bars_axis.text(
            bar.get_x() + bar.get_width() / 2,
            high[index] + label_offset,
            f"{values[index]:+.4f}\nMAE {adc_mae.loc[model, 'mean']:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#222222",
        )
    bars_axis.axhline(0.0, color="#555555", lw=0.9, zorder=2)
    bars_axis.set_xticks(x, MODEL_ORDER)
    bars_axis.set_ylabel(r"$\Delta$MAE [SOC]")
    bars_axis.set_title("(c) Six-Cell ADC Performance Impact")
    bars_axis.text(
        0.98,
        0.98,
        "Cell-macro mean and hierarchical 95% CI (n=6 cells)",
        transform=bars_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#666666",
    )
    lower_limit = min(float(low.min()) - 0.10 * span, -0.0055)
    upper_limit = max(float(high.max()) + 0.25 * span, 0.0115)
    bars_axis.set_ylim(lower_limit, upper_limit)

    for axis in (current_axis, voltage_axis, bars_axis):
        clean_axes(axis)
    figure.suptitle("ADC Quantization: Signal Detail and Six-Cell Impact", y=0.985)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.91)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
