from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
SIM_ROOT = RESULTS_DIR.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from jes2_protocol import DEFAULT_REFERENCE_ALIASES, SCENARIOS, SCENARIO_LABELS, STOCHASTIC_ALIASES  # noqa: E402
from jes2_plot_style import NEUTRAL_DARK, NEUTRAL_LIGHT, save_figure, setup_style  # noqa: E402


CELLS = ["C25", "C27", "C09", "C13", "C15", "C29"]
LOAD_CLASS = {
    "C25": "low",
    "C27": "low",
    "C09": "middle",
    "C13": "middle",
    "C15": "middle",
    "C29": "high",
}
LOAD_COLORS = {"low": "#2ca02c", "middle": "#9467bd", "high": "#d62728"}
STATE_COLORS = {"fresh": "#2ca02c", "mid_life": "#9467bd", "aged": "#d62728"}
STATE_LABELS = {"fresh": "Fresh (SOH >= 0.90)", "mid_life": "Mid-life (0.80-0.90)", "aged": "Aged (SOH < 0.80)"}


def cell_id(value: str) -> str:
    return str(value).rsplit("_", 1)[-1]


def load_characteristics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["cell"] = frame["cell"].map(cell_id)
    frame = frame.set_index("cell").reindex(CELLS).reset_index()
    frame["cell_load_class"] = frame["cell"].map(LOAD_CLASS)
    return frame


def collect_state_coverage(
    data_root: Path,
    sample_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, float | str]] = []
    samples: list[pd.DataFrame] = []
    columns = ["Testtime[s]", "Temperature[°C]", "SOH", "SOC", "EFC", "C_Rate"]

    for cell in CELLS:
        source = data_root / f"df_FE_{cell}.parquet"
        frame = pd.read_parquet(source, columns=columns)
        soh = frame["SOH"].to_numpy(dtype=np.float64)
        temperature = frame["Temperature[°C]"].to_numpy(dtype=np.float64)
        c_rate = np.abs(frame["C_Rate"].to_numpy(dtype=np.float64))
        soc = frame["SOC"].to_numpy(dtype=np.float64)

        masks = {
            "soh_fresh": soh >= 0.90,
            "soh_mid_life": (soh >= 0.80) & (soh < 0.90),
            "soh_aged": soh < 0.80,
            "temperature_nominal": temperature <= 30.0,
            "temperature_elevated": (temperature > 30.0) & (temperature <= 35.0),
            "temperature_hot": temperature > 35.0,
            "load_low": c_rate < 0.5,
            "load_medium": (c_rate >= 0.5) & (c_rate < 1.5),
            "load_high": c_rate >= 1.5,
            "soc_low": soc < 0.20,
            "soc_middle": (soc >= 0.20) & (soc <= 0.80),
            "soc_high": soc > 0.80,
        }
        row: dict[str, float | str] = {
            "cell": cell,
            "cell_load_class": LOAD_CLASS[cell],
        }
        row.update({name: float(np.mean(mask)) for name, mask in masks.items()})
        coverage_rows.append(row)

        count = min(sample_points, len(frame))
        indices = np.linspace(0, len(frame) - 1, count, dtype=np.int64)
        sampled = frame.iloc[indices].copy()
        sampled.insert(0, "cell", cell)
        sampled["abs_c_rate"] = np.abs(sampled["C_Rate"].to_numpy(dtype=np.float64))
        sampled["life_fraction"] = np.linspace(0.0, 1.0, count)
        samples.append(sampled)
        del frame, sampled, soh, temperature, c_rate, soc
        gc.collect()

    return pd.DataFrame(coverage_rows), pd.concat(samples, ignore_index=True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)


def plot_holdout_overview(cells: pd.DataFrame, out: Path) -> None:
    y = np.arange(len(cells))
    labels = cells["cell"].tolist()
    colors = [LOAD_COLORS[value] for value in cells["cell_load_class"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))

    ax = axes[0, 0]
    ax.axvspan(0.58, 0.80, color="#b6302d", alpha=0.07)
    ax.axvspan(0.80, 0.90, color="#d1887e", alpha=0.10)
    ax.axvspan(0.90, 1.005, color="#566b78", alpha=0.08)
    for idx, row in cells.iterrows():
        ax.plot([row["soh_min"], 1.0], [idx, idx], color=colors[idx], linewidth=3.3, solid_capstyle="round")
        ax.scatter(row["soh_mean"], idx, s=54, color="white", edgecolor=colors[idx], linewidth=2.0, zorder=3)
    ax.axvline(0.80, color=NEUTRAL_DARK, linewidth=0.8, linestyle="--")
    ax.axvline(0.90, color=NEUTRAL_DARK, linewidth=0.8, linestyle="--")
    ax.set_xlim(0.58, 1.01)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reference SOH (line: observed range; circle: mean)")
    ax.set_title("Aging envelope")
    clean_axis(ax)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    values = cells["abs_c_rate_p95"].to_numpy(dtype=float)
    bars = ax.barh(y, values, color=colors, edgecolor=NEUTRAL_DARK, linewidth=0.7, height=0.62)
    for idx, (bar, load_class) in enumerate(zip(bars, cells["cell_load_class"])):
        ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f} C  |  {load_class}", va="center", fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(values) * 1.38)
    ax.set_xlabel("95th percentile absolute C-rate")
    ax.set_title("Frozen load-class assignment")
    clean_axis(ax)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.axvspan(35.0, 50.5, color="#b6302d", alpha=0.06)
    for idx, row in cells.iterrows():
        ax.plot([row["temperature_min_c"], row["temperature_max_c"]], [idx, idx],
                color=colors[idx], linewidth=3.3, solid_capstyle="round")
        ax.scatter(row["temperature_mean_c"], idx, s=54, color="white", edgecolor=colors[idx], linewidth=2.0, zorder=3)
    ax.axvline(35.0, color="#b6302d", linewidth=0.9, linestyle="--")
    ax.set_xlim(24.0, 50.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cell temperature [degC] (line: range; circle: mean)")
    ax.set_title("Thermal envelope")
    clean_axis(ax)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    left = np.zeros(len(cells), dtype=float)
    for state, column in [("fresh", "soh_fresh_fraction"), ("mid_life", "soh_mid_life_fraction"), ("aged", "soh_aged_fraction")]:
        values = 100.0 * cells[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=STATE_COLORS[state], edgecolor="white", linewidth=0.7,
                height=0.62, label=STATE_LABELS[state])
        for idx, (start, value) in enumerate(zip(left, values)):
            if value >= 9.0:
                ax.text(start + value / 2, idx, f"{value:.0f}%", ha="center", va="center",
                        color="white" if state != "mid_life" else NEUTRAL_DARK, fontsize=8, fontweight="bold")
        left += values
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Fraction of trajectory [%]")
    ax.set_title("SOH-state coverage")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False, fontsize=8)
    clean_axis(ax)
    panel_label(ax, "(d)")

    fig.suptitle("Six independent LFP holdout cells: aging, load, thermal, and state coverage", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.12, hspace=0.46, wspace=0.32)
    save_figure(fig, out / "Figure_16_Holdout_Cell_Coverage.png")


def plot_aging_trajectories(samples: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharey=True)
    norm = Normalize(vmin=25.0, vmax=45.0)
    cmap = LinearSegmentedColormap.from_list("temperature", ["#566b78", "#d1887e", "#b6302d"])
    scatter = None
    for ax, cell in zip(axes.flat, CELLS):
        part = samples[samples["cell"] == cell]
        ax.axhspan(0.60, 0.80, color="#b6302d", alpha=0.055)
        ax.axhspan(0.80, 0.90, color="#d1887e", alpha=0.075)
        ax.axhspan(0.90, 1.00, color="#566b78", alpha=0.055)
        scatter = ax.scatter(part["EFC"], part["SOH"], c=part["Temperature[°C]"], s=4,
                             alpha=0.62, cmap=cmap, norm=norm, rasterized=True, linewidths=0)
        ax.axhline(0.80, color=NEUTRAL_DARK, linewidth=0.65, linestyle="--")
        ax.axhline(0.90, color=NEUTRAL_DARK, linewidth=0.65, linestyle="--")
        ax.set_title(f"{cell}  |  {LOAD_CLASS[cell]} load", color=LOAD_COLORS[LOAD_CLASS[cell]], fontweight="bold")
        ax.set_xlabel("Equivalent full cycles")
        clean_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Reference SOH")
    axes[0, 0].set_ylim(0.59, 1.005)
    colorbar_axis = fig.add_axes([0.925, 0.20, 0.014, 0.58])
    colorbar = fig.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label("Measured temperature [degC]")
    fig.suptitle("Observed aging trajectories and thermal exposure of all holdout cells", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.89, top=0.90, bottom=0.08, hspace=0.30, wspace=0.24)
    save_figure(fig, out / "Figure_24_Holdout_Aging_Trajectories.png")


def plot_state_coverage_matrix(coverage: pd.DataFrame, out: Path) -> None:
    columns = [
        "soh_fresh", "soh_mid_life", "soh_aged",
        "temperature_nominal", "temperature_elevated", "temperature_hot",
        "load_low", "load_medium", "load_high",
        "soc_low", "soc_middle", "soc_high",
    ]
    labels = [
        "Fresh", "Mid", "Aged", "<=30", "30-35", ">35",
        "<0.5C", "0.5-1.5C", ">=1.5C", "<20%", "20-80%", ">80%",
    ]
    matrix = coverage.set_index("cell").reindex(CELLS)[columns].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("coverage", ["#f7f7f7", "#edc6c1", "#b6302d"])
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(CELLS)), CELLS)
    ax.tick_params(axis="x", rotation=0)
    for boundary in [2.5, 5.5, 8.5]:
        ax.axvline(boundary, color="white", linewidth=4.0)
        ax.axvline(boundary, color=NEUTRAL_DARK, linewidth=0.55)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if value >= 0.62 else NEUTRAL_DARK,
                    fontweight="bold" if value >= 0.62 else "normal")
    group_centers = [1.0, 4.0, 7.0, 10.0]
    for x, title in zip(group_centers, ["SOH state", "Temperature [degC]", "Instantaneous load", "SOC state"]):
        ax.text(x, -1.00, title, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Trajectory coverage fraction")
    ax.set_title("Measured state coverage by independent holdout cell (cell values in %)", fontsize=14, fontweight="bold", pad=34)
    ax.set_xlabel("Predeclared reference-state strata")
    ax.set_ylabel("Holdout cell")
    fig.tight_layout()
    save_figure(fig, out / "Figure_25_State_Coverage_Matrix.png")


def plot_operating_envelopes(samples: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharex=True, sharey=True)
    cmap = LinearSegmentedColormap.from_list("soh", ["#b6302d", "#d1887e", "#566b78"])
    image = None
    for ax, cell in zip(axes.flat, CELLS):
        part = samples[samples["cell"] == cell]
        image = ax.hexbin(part["abs_c_rate"], part["Temperature[°C]"], C=part["SOH"],
                          reduce_C_function=np.mean, gridsize=(39, 28), mincnt=2,
                          cmap=cmap, vmin=0.60, vmax=1.0, linewidths=0.0, rasterized=True)
        ax.axvline(0.5, color=NEUTRAL_DARK, linewidth=0.6, linestyle="--")
        ax.axvline(1.5, color=NEUTRAL_DARK, linewidth=0.6, linestyle="--")
        ax.axhline(30.0, color=NEUTRAL_DARK, linewidth=0.6, linestyle="--")
        ax.axhline(35.0, color="#b6302d", linewidth=0.7, linestyle="--")
        ax.set_title(f"{cell}  |  {LOAD_CLASS[cell]} load", color=LOAD_COLORS[LOAD_CLASS[cell]], fontweight="bold")
        ax.set_xlabel("Absolute C-rate")
        clean_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature [degC]")
    axes[0, 0].set_xlim(0.0, 3.15)
    axes[0, 0].set_ylim(24.0, 50.5)
    colorbar_axis = fig.add_axes([0.925, 0.20, 0.014, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean reference SOH in occupied bin")
    fig.suptitle("Joint load-temperature operating envelope and associated aging state", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.89, top=0.90, bottom=0.08, hspace=0.30, wspace=0.24)
    save_figure(fig, out / "Figure_26_Load_Temperature_Operating_Envelope.png")


def scenario_axes(alias: str) -> set[str]:
    if alias == "baseline":
        return {"baseline"}
    if alias.startswith("current_") or alias == "adc_quantization":
        return {"current"}
    if alias.startswith("voltage_"):
        return {"voltage"}
    if alias.startswith("temperature_"):
        return {"temperature"}
    if alias.startswith("irregular_sampling"):
        return {"timing"}
    if alias.startswith("missing_"):
        return {"availability"}
    if alias == "initial_soc_error":
        return {"initialization"}
    return set()


def plot_scenario_matrix(out: Path) -> None:
    columns = ["Baseline", "Current", "Voltage", "Temperature", "Timing", "Availability", "Initialization", "Repeated", "Ref. SOH"]
    keys = ["baseline", "current", "voltage", "temperature", "timing", "availability", "initialization", "stochastic", "reference"]
    matrix = np.zeros((len(SCENARIOS), len(columns)), dtype=int)
    row_labels = []
    for row, (alias, _scenario, _args) in enumerate(SCENARIOS):
        active = scenario_axes(alias)
        if alias in STOCHASTIC_ALIASES:
            active.add("stochastic")
        if alias in DEFAULT_REFERENCE_ALIASES:
            active.add("reference")
        for col, key in enumerate(keys):
            matrix[row, col] = int(key in active)
        row_labels.append(SCENARIO_LABELS[alias])

    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(len(SCENARIOS) - 0.4, -0.6)
    for row in range(len(SCENARIOS)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#f3f3f3", zorder=0)
        for col in range(len(columns)):
            if not matrix[row, col]:
                continue
            key = keys[col]
            color = "#b6302d" if key not in {"stochastic", "reference"} else ("#566b78" if key == "stochastic" else "#d1887e")
            ax.scatter(col, row, s=145, marker="s", color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(col, row, "x", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=27, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=6)
    ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("JES 2.0 measurement-only disturbance matrix", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.61, 0.025,
             "Red: manipulated signal/state   |   Blue-gray: 10/5 seeded repetitions   |   Rose: paired reference-SOH ablation",
             ha="center", fontsize=9, color=NEUTRAL_DARK)
    fig.subplots_adjust(left=0.255, right=0.98, top=0.87, bottom=0.07)
    save_figure(fig, out / "Figure_27_JES2_Test_Matrix.png")


def draw_flow_box(ax, xy: tuple[float, float], size: tuple[float, float], title: str, body: str,
                  facecolor: str, edgecolor: str = NEUTRAL_DARK) -> None:
    x, y = xy
    width, height = size
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.018,rounding_size=0.018",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1)
    ax.add_patch(box)
    ax.text(x + 0.04 * width, y + 0.68 * height, title, fontsize=10, fontweight="bold", va="center")
    ax.text(x + 0.04 * width, y + 0.30 * height, body, fontsize=8.2, va="center", color=NEUTRAL_DARK, linespacing=1.25)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.2, color="#666666", connectionstyle="arc3,rad=0.0"))


def plot_statistical_workflow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    top = [
        ((0.03, 0.64), "Independent holdouts", "6 cells\nclass sizes: 2 / 3 / 1", "#e8edf0"),
        ((0.27, 0.64), "Scenario execution", "19 cases\n10/5 seeds if random", "#f5e3df"),
        ((0.51, 0.64), "Run-level metrics", "MAE, RMSE, bias, P95\ncoverage + recovery", "#f3e9e7"),
        ((0.75, 0.64), "Fixed state strata", "SOH, load, temperature\nand SOC states", "#e8edf0"),
    ]
    bottom = [
        ((0.75, 0.19), "Cell-level pairing", "window/seed mean per cell\nscenario - baseline", "#e8edf0"),
        ((0.51, 0.19), "Equal cell macro", "no trajectory-length\nweighting", "#f3e9e7"),
        ((0.27, 0.19), "Uncertainty + tests", "hierarchical bootstrap\nsign-flip + effect dz", "#f5e3df"),
        ((0.03, 0.19), "Paper evidence", "95% CI + raw effects\nHolm-adjusted p-values", "#e8edf0"),
    ]
    box_size = (0.19, 0.20)
    for position, title, body, color in top + bottom:
        draw_flow_box(ax, position, box_size, title, body, color)
    for left, right in zip(top[:-1], top[1:]):
        draw_arrow(ax, (left[0][0] + box_size[0], left[0][1] + 0.10), (right[0][0] - 0.012, right[0][1] + 0.10))
    draw_arrow(ax, (0.845, 0.64), (0.845, 0.41))
    for left, right in zip(bottom[:-1], bottom[1:]):
        draw_arrow(ax, (left[0][0] - 0.012, left[0][1] + 0.10), (right[0][0] + box_size[0], right[0][1] + 0.10))

    ax.text(0.5, 0.965, "JES 2.0 statistical analysis: cells are the independent unit",
            ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.90, "Per-second samples are used to calculate errors, never as independent statistical replicates.",
            ha="center", va="top", fontsize=10, color=NEUTRAL_DARK)
    ax.text(0.5, 0.075, "Primary interpretation: paired effect sizes and confidence intervals; p-values are secondary at n = 6 cells.",
            ha="center", va="center", fontsize=9.5, color="#8c2725", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "Figure_28_Statistical_Analysis_Workflow.png")


def plot_evaluation_windows(windows: pd.DataFrame, out: Path) -> None:
    frame = windows.copy()
    frame["cell"] = frame["cell"].map(cell_id)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ax.axvspan(0.60, 0.80, color=STATE_COLORS["aged"], alpha=0.07)
    ax.axvspan(0.80, 0.90, color=STATE_COLORS["mid_life"], alpha=0.09)
    ax.axvspan(0.90, 1.00, color=STATE_COLORS["fresh"], alpha=0.07)
    for y, cell in enumerate(CELLS):
        part = frame[frame["cell"] == cell]
        for row in part.itertuples(index=False):
            ax.scatter(
                row.soh_median, y, s=95, color=STATE_COLORS[row.soh_state],
                edgecolor="white", linewidth=1.1, zorder=3,
            )
            ax.text(row.soh_median, y - 0.23, row.soh_state.replace("mid_life", "mid"),
                    ha="center", va="top", fontsize=7.5)
    ax.axvline(0.80, color=NEUTRAL_DARK, linewidth=0.7, linestyle="--")
    ax.axvline(0.90, color=NEUTRAL_DARK, linewidth=0.7, linestyle="--")
    ax.set_xlim(0.60, 1.005)
    ax.set_ylim(len(CELLS) - 0.45, -0.55)
    ax.set_yticks(np.arange(len(CELLS)), CELLS)
    ax.set_xlabel("Median reference SOH in selected 24 h window")
    ax.set_title("16 frozen cell/SOH evaluation windows")
    clean_axis(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.27, 0.22, "SOH context", "192 h, causal\nnot scored", "#e8edf0"),
        (0.35, 0.58, 0.25, 0.22, "Primary tests", "24 h\n86,400 rows", "#f3e9e7"),
        (0.66, 0.58, 0.31, 0.22, "Dropout test", "48 h; central 1 h gap\n~24 h recovery", "#f5e3df"),
    ]
    for x, y, width, height, title, body, color in boxes:
        draw_flow_box(ax, (x, y), (width, height), title, body, color)
    draw_arrow(ax, (0.29, 0.69), (0.34, 0.69))
    draw_arrow(ax, (0.60, 0.69), (0.65, 0.69))
    ax.scatter(0.325, 0.69, marker="D", s=75, color="#b6302d", edgecolor="white", zorder=5)
    ax.text(0.325, 0.47, "Measured full-charge anchor\nSOC >= 0.98; U >= 3.58 V",
            ha="center", va="top", fontsize=8.5, color=NEUTRAL_DARK)
    ax.text(0.50, 0.21,
            "Window choice uses measured-feature medoids only;\nmodel predictions and errors are never selection inputs.",
            ha="center", va="center", fontsize=9.3, fontweight="bold", color="#8c2725")
    ax.set_title("Causal and model-independent execution protocol", pad=12)
    panel_label(ax, "(b)")

    fig.suptitle("JES 2.0 representative-window protocol", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "Figure_29_Evaluation_Window_Protocol.png")


def plot_figure_overview(figures_dir: Path, out_path: Path) -> None:
    files = sorted(
        figures_dir.glob("Figure_*.png"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    columns = 3
    rows = int(np.ceil(len(files) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15.0, 3.6 * rows))
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, path in zip(flat_axes, files):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem.replace("_", " "), fontsize=9, fontweight="bold", pad=7)
        ax.axis("off")
    for ax in flat_axes[len(files):]:
        ax.axis("off")
    fig.suptitle("JES paper figure inventory", fontsize=16, fontweight="bold", y=0.998)
    fig.text(
        0.5,
        0.002,
        "Figures 04-14: legacy single-cell results pending JES2 replacement | "
        "Figures 16 and 24-28: real six-cell coverage/design evidence",
        ha="center",
        fontsize=9,
        color="#8c2725",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.975, bottom=0.025, hspace=0.18, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-data JES2 coverage and design figures.")
    parser.add_argument("--data_root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument(
        "--characteristics",
        type=Path,
        default=Path("LATEX/JES/paper_robustness_benchmark/JES_2.0/tables/jes2_cell_characteristics.csv"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tables_dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("LATEX/JES/paper_robustness_benchmark/JES_2.0/tables/jes2_evaluation_windows.csv"),
    )
    parser.add_argument("--sample_points", type=int, default=30000)
    args = parser.parse_args()

    setup_style()
    plt.rcParams.update({
        "font.family": "Nimbus Sans",
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.5,
        "figure.titlesize": 14,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    cells = load_characteristics(args.characteristics)
    coverage, samples = collect_state_coverage(args.data_root, args.sample_points)
    coverage.to_csv(args.tables_dir / "jes2_state_coverage.csv", index=False)
    samples.to_csv(args.tables_dir / "jes2_trajectory_plot_samples.csv.gz", index=False, compression="gzip")

    plot_holdout_overview(cells, args.out_dir)
    plot_aging_trajectories(samples, args.out_dir)
    plot_state_coverage_matrix(coverage, args.out_dir)
    plot_operating_envelopes(samples, args.out_dir)
    plot_scenario_matrix(args.out_dir)
    plot_statistical_workflow(args.out_dir)
    plot_evaluation_windows(pd.read_csv(args.windows), args.out_dir)
    plot_figure_overview(args.out_dir, args.tables_dir.parent / "JES2_Figure_Overview.png")
    print(f"Generated 7 publication figures in {args.out_dir}")


if __name__ == "__main__":
    main()
