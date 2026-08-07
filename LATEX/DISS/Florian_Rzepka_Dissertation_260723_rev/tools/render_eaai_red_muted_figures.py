from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, MaxNLocator


DISS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = DISS_ROOT.parents[2]
OUT_DIR = DISS_ROOT / "pictures" / "red_muted"

SOC_NPZ = (
    SCRIPTS_ROOT
    / "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC/bench_v_soc_full/soc_streaming_base_quant_pruned_data.npz"
)
SOH_NPZ = (
    SCRIPTS_ROOT
    / "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH/BENCH_SOH_FULL_FINAL_20251124/benchmark_results.npz"
)
PARQUET_PATH = (
    SCRIPTS_ROOT
    / "3_Projekte/MG_Farm/5_Data/01_LFP/00_Data/Versuch_18650_standart/MGFarm_18650_FE/df_FE_C07.parquet"
)

# Dissertation palette. For the three embedded variants we use the high-contrast
# red-muted subset M1, M2, M4 so the original EAAI layout remains readable.
COLORS = {
    "Base": "#b6302d",
    "Pruned": "#d1887e",
    "Quantized": "#566b78",
}

FS_TITLE = 20
FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 13


def apply_boxplot_style(bplot, colors_list: list[str]) -> None:
    for i, color in enumerate(colors_list):
        r, g, b, _ = to_rgba(color)
        edge_color = (r, g, b, 1.0)
        face_color = (r, g, b, 0.4)

        patch = bplot["boxes"][i]
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(2.0)

        median = bplot["medians"][i]
        median.set_color("black")
        median.set_linewidth(1.5)

        bplot["whiskers"][i * 2].set_color(edge_color)
        bplot["whiskers"][i * 2].set_linewidth(1.5)
        bplot["whiskers"][i * 2 + 1].set_color(edge_color)
        bplot["whiskers"][i * 2 + 1].set_linewidth(1.5)

        bplot["caps"][i * 2].set_color(edge_color)
        bplot["caps"][i * 2].set_linewidth(1.5)
        bplot["caps"][i * 2 + 1].set_color(edge_color)
        bplot["caps"][i * 2 + 1].set_linewidth(1.5)


def plot_mae_hist_styled(preds: dict[str, np.ndarray], y_true: np.ndarray, title_suffix: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    order = ["Base", "Pruned", "Quantized"]
    sorted_names = [name for name in order if name in preds]
    scale = 100.0

    data_to_plot = []
    colors_list = []
    for name in sorted_names:
        abs_err = np.abs(preds[name] - y_true) * scale
        data_to_plot.append(abs_err)
        colors_list.append(COLORS[name])

    bplot = ax1.boxplot(
        data_to_plot,
        tick_labels=sorted_names,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
    apply_boxplot_style(bplot, colors_list)

    ax1.set_ylabel("Absolute Error [%]", fontsize=FS_LABEL)
    ax1.set_title(f"Error Distribution (Boxplot) - {title_suffix}", fontsize=FS_TITLE)
    ax1.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax1.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch

    legend_elements = []
    for name, color in zip(sorted_names, colors_list):
        mae = np.mean(np.abs(preds[name] - y_true)) * scale
        label_text = f"{name}\nMAE: {mae:.2f}"
        r, g, b, _ = to_rgba(color)
        legend_elements.append(
            Patch(
                facecolor=(r, g, b, 0.4),
                edgecolor=(r, g, b, 1.0),
                linewidth=2.0,
                label=label_text,
            )
        )

    ax1.legend(handles=legend_elements, loc="upper left", fontsize=FS_LEGEND)

    for name in sorted_names:
        err = (preds[name] - y_true) * scale
        ax2.hist(
            err,
            bins=100,
            range=(-10.0, 10.0),
            alpha=0.4,
            label=name,
            color=COLORS[name],
            histtype="stepfilled",
        )

    ax2.set_xlabel("Error (pred - GT) [%]", fontsize=FS_LABEL)
    ax2.set_ylabel("Count [k]", fontsize=FS_LABEL)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:g}"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.set_title(f"Error Histogram (Counts) - {title_suffix}", fontsize=FS_TITLE)
    ax2.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax2.legend(loc="upper right", fontsize=FS_LEGEND)
    ax2.grid(alpha=0.2)

    fig.tight_layout(w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_soc_error() -> None:
    data = np.load(SOC_NPZ)
    s = slice(1000, None, 50)
    y_true = data["y"][s]
    preds = {
        "Base": data["base"][s],
        "Pruned": data["pruned"][s],
        "Quantized": data["quant"][s],
    }
    plot_mae_hist_styled(preds, y_true, "SOC", OUT_DIR / "embedded_soc_error.png")


def load_soc_full() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    data = np.load(SOC_NPZ)
    y_true = data["y"]
    preds = {
        "Base": data["base"],
        "Pruned": data["pruned"],
        "Quantized": data["quant"],
    }
    time_axis = np.arange(len(y_true), dtype=float)
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH, columns=["Testtime[s]"])
            if len(df) >= len(y_true):
                time_axis = df["Testtime[s]"].to_numpy(dtype=float)[: len(y_true)]
        except Exception as exc:
            print(f"Warning: could not load {PARQUET_PATH}: {exc}")
    return y_true, preds, time_axis


def plot_soc_zoom(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    time_axis: np.ndarray,
    start: float,
    end: float,
    out_path: Path,
) -> None:
    mask = (time_axis >= start) & (time_axis <= end)
    if not np.any(mask):
        raise ValueError(f"No SOC samples found in range {start}-{end} s")

    t_slice = time_axis[mask]
    y_slice = y_true[mask] * 100.0

    fig, (ax_s, ax_e) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_s.plot(t_slice, y_slice, "k-", alpha=0.5, label="GT")

    for name in ["Base", "Pruned", "Quantized"]:
        pred = preds[name][mask] * 100.0
        ax_s.plot(t_slice, pred, label=name, color=COLORS[name], linewidth=1, alpha=0.8)
        ax_e.plot(t_slice, pred - y_slice, label=name, color=COLORS[name], linewidth=0.8, alpha=0.7)

    ax_s.set_title(f"SOC comparison {int(start)}-{int(end)} s")
    ax_s.set_ylabel("SOC [%]")
    ax_s.legend(loc="lower left")
    ax_s.grid(alpha=0.3)

    ax_e.set_ylabel("Error (Pred - GT) [%]")
    ax_e.set_xlabel("Time [s]")
    ax_e.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_soc_zooms() -> None:
    y_true, preds, time_axis = load_soc_full()
    plot_soc_zoom(
        y_true,
        preds,
        time_axis,
        30000,
        40000,
        OUT_DIR / "embedded_soc_zoom_pulse.png",
    )
    plot_soc_zoom(
        y_true,
        preds,
        time_axis,
        657000,
        897000,
        OUT_DIR / "embedded_soc_zoom_checkup.png",
    )


def render_soh_error() -> None:
    data = np.load(SOH_NPZ)
    stride = 10 if len(data["y_gt"]) > 50000 else 1
    s = slice(100, None, stride)
    y_true = data["y_gt"][s]
    preds = {
        "Base": data["C_Base"][s],
        "Pruned": data["C_Pruned"][s],
        "Quantized": data["C_Quant"][s],
    }
    plot_mae_hist_styled(preds, y_true, "SOH", OUT_DIR / "embedded_soh_error.png")


def main() -> None:
    render_soc_error()
    render_soh_error()
    render_soc_zooms()
    print(f"Rendered EAAI red-muted figures into {OUT_DIR}")


if __name__ == "__main__":
    main()
