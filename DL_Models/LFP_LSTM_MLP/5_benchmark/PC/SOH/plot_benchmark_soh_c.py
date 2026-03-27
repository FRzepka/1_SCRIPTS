#!/usr/bin/env python3
"""
SOH Benchmark (C-only) styled like SOC bench_v_soc_full:
- Uses full C-only arrays (Base/Pruned/Quant) from arrays.npz
- Produces:
  * soh_streaming_dashboard.png      (Overlay + Errors, downsampled, skip initial)
  * soh_mae_hist_combined.png        (MAE boxplot + error histogram)
  * soh_model_sizes.png              (Params, est. flash, .so size)
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Match SOC color scheme: Base=Green, Pruned=Red, Quantized=Blue
COLORS = {
    "Base": "#2ca02c",
    "Pruned": "#d62728",
    "Quantized": "#1f77b4",
}

STRIDE = 50
SKIP_INITIAL = 1000


def load_arrays(npz_path: Path):
    z = np.load(npz_path)
    # Handle both old (benchmark_results.npz) and new (arrays.npz) formats
    if 'y_gt' in z:
        y = z['y_gt']
        base = z['C_Base']
        pruned = z['C_Pruned']
        quant = z['C_Quant']
    else:
        y = z["y"]
        base = z["base_c"]
        pruned = z["pruned_c"]
        quant = z["quant_c"]
        
    n = min(len(y), len(base), len(pruned), len(quant))
    return y[:n], base[:n], pruned[:n], quant[:n]


def outline_bars(bars):
    """SOC-style bar outline (semi-transparent fill, strong edge)."""
    for bar in bars:
        c = bar.get_facecolor()
        if len(c) == 4:
            r, g, b, _ = c
        else:
            r, g, b = to_rgba(c)[:3]
        bar.set_alpha(None)
        bar.set_edgecolor((r, g, b, 1.0))
        bar.set_facecolor((r, g, b, 0.4))
        bar.set_linewidth(1.5)


def compute_error_metrics(y_true, pred):
    err = pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    std = float(np.std(abs_err))
    rmse = float(math.sqrt(float(np.mean(err**2))))
    max_abs = float(np.max(abs_err))
    return {"MAE": mae, "STD": std, "RMSE": rmse, "MAX": max_abs}


def plot_mae_and_hist_combined(metrics, y, preds, out_png, bins=100):
    """Adapted from SOC plot_mae_and_hist_combined, but for SOH."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    FS_TITLE = 16
    FS_LABEL = 14
    FS_TICK = 12
    FS_LEGEND = 11

    names = list(metrics.keys())
    order = ["Base", "Pruned", "Quantized"]
    sorted_names = [n for n in order if n in names]

    data_to_plot = []
    colors_list = []
    for name in sorted_names:
        pred = preds[name]
        abs_err = np.abs(pred - y)
        data_to_plot.append(abs_err)
        colors_list.append(COLORS[name])

    bplot = ax1.boxplot(
        data_to_plot,
        tick_labels=sorted_names,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )

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

    ax1.set_ylabel("Absolute Error", fontsize=FS_LABEL)
    ax1.set_title("SOH Error Distribution (Boxplot)", fontsize=FS_TITLE)
    ax1.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax1.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch

    legend_elements = []
    for name, c in zip(sorted_names, colors_list):
        m = metrics[name]["MAE"]
        label_text = f"{name}\nMAE: {m:.4f}"
        r, g, b, _ = to_rgba(c)
        legend_elements.append(
            Patch(
                facecolor=(r, g, b, 0.4),
                edgecolor=(r, g, b, 1.0),
                linewidth=2.0,
                label=label_text,
            )
        )
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=FS_LEGEND)

    # Right: histogram
    for name, pred in preds.items():
        err = pred - y
        c = COLORS[name]
        ax2.hist(
            err,
            bins=bins,
            alpha=0.4,
            label=name,
            color=c,
            histtype="stepfilled",
        )
    ax2.set_xlabel("Error (pred - GT)", fontsize=FS_LABEL)
    ax2.set_ylabel("Count", fontsize=FS_LABEL)
    ax2.set_title("SOH Error Histogram (Counts)", fontsize=FS_TITLE)
    ax2.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", fontsize=FS_LEGEND)

    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_streaming_dashboard(y, preds, out_png):
    """2x1 Dashboard: overlay + errors, with skip_initial & stride like SOC."""
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    s = slice(SKIP_INITIAL, None, STRIDE)
    x_axis = np.arange(len(y))[s]

    y_s = y[s]
    base_s = preds["Base"][s]
    pruned_s = preds["Pruned"][s]
    quant_s = preds["Quantized"][s]

    ax_top.plot(x_axis, y_s, label="GT SOH", linewidth=1.5, alpha=0.7, color="black")
    ax_top.plot(x_axis, base_s, label="Base", linewidth=1.2, alpha=1.0, color=COLORS["Base"])
    ax_top.plot(x_axis, pruned_s, label="Pruned", linewidth=1.2, alpha=1.0, color=COLORS["Pruned"])
    ax_top.plot(x_axis, quant_s, label="Quantized", linewidth=1.2, alpha=1.0, color=COLORS["Quantized"])
    ax_top.set_ylabel("SOH")
    ax_top.set_title(f"SOH Prediction (skipped first {SKIP_INITIAL} steps)")
    ax_top.legend(loc="lower left")
    ax_top.grid(alpha=0.2)

    err_base = base_s - y_s
    err_pruned = pruned_s - y_s
    err_quant = quant_s - y_s

    ax_err.plot(x_axis, err_base, label="Err Base", linewidth=1.0, alpha=0.9, color=COLORS["Base"])
    ax_err.plot(x_axis, err_pruned, label="Err Pruned", linewidth=1.0, alpha=0.9, color=COLORS["Pruned"])
    ax_err.plot(x_axis, err_quant, label="Err Quantized", linewidth=1.0, alpha=0.9, color=COLORS["Quantized"])
    ax_err.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_err.set_ylabel("Error (Pred - GT)")
    ax_err.set_xlabel("step")
    ax_err.legend(loc="lower left")
    ax_err.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_model_sizes(info: dict, out_path: Path):
    names = list(info.keys())
    params = [info[n]["params_m"] for n in names]
    flash = [info[n]["flash_kb"] for n in names]
    so = [info[n]["so_kb"] for n in names]
    
    colors = [COLORS[n] for n in names]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Params
    bars1 = ax1.bar(names, params, color=colors, width=0.5)
    outline_bars(bars1)
    ax1.set_title("Parameters [M]")
    ax1.set_ylabel("Million Parameters")
    ax1.grid(axis='y', alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.3f}", ha='center', va='bottom')

    # Flash
    bars2 = ax2.bar(names, flash, color=colors, width=0.5)
    outline_bars(bars2)
    ax2.set_title("Est. Flash Usage [KB]")
    ax2.set_ylabel("KB")
    ax2.grid(axis='y', alpha=0.3)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.1f}", ha='center', va='bottom')

    # SO Size
    bars3 = ax3.bar(names, so, color=colors, width=0.5)
    outline_bars(bars3)
    ax3.set_title("Actual .so Size [KB]")
    ax3.set_ylabel("KB")
    ax3.grid(axis='y', alpha=0.3)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.1f}", ha='center', va='bottom')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def est_params(hidden: int, mlp_hidden: int = 64, in_f: int = 6):
    lstm_params = 4 * hidden * in_f + 4 * hidden * hidden + 8 * hidden
    mlp_params = mlp_hidden * hidden + mlp_hidden + mlp_hidden + 1
    return lstm_params + mlp_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        type=Path,
        default=Path(
            "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH/BENCH_SOH_C_20251124_143243/arrays.npz"
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    y, base_c, pruned_c, quant_c = load_arrays(args.npz)
    # Map to SOC-style keys
    preds = {
        "Base": base_c,
        "Pruned": pruned_c,
        "Quantized": quant_c,
    }

    out_dir = args.out_dir or args.npz.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics on sliced region (skip initial), like SOC
    y_sliced = y[SKIP_INITIAL:]
    preds_sliced = {k: v[SKIP_INITIAL:] for k, v in preds.items()}
    metrics = {name: compute_error_metrics(y_sliced, p) for name, p in preds_sliced.items()}

    # Sizes
    repo = Path(__file__).resolve().parents[3]
    base_so = repo / "2_models/base/soh_2.1.0.0_base/c_implementation/liblstm_soh_base.so"
    pruned_so = repo / "2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142/c_implementation/liblstm_soh_pruned.so"
    quant_so = repo / "2_models/quantized/soh_2.1.0.0_quantized/c_implementation/liblstm_soh_quant.so"
    size_info = {
        "Base": {
            "params_m": est_params(128) / 1e6,
            "flash_kb": est_params(128) * 4 / 1024.0,
            "so_kb": base_so.stat().st_size / 1024.0 if base_so.exists() else 0.0,
        },
        "Pruned": {
            "params_m": est_params(90) / 1e6,
            "flash_kb": est_params(90) * 4 / 1024.0,
            "so_kb": pruned_so.stat().st_size / 1024.0 if pruned_so.exists() else 0.0,
        },
        "Quantized": {
            "params_m": est_params(128) / 1e6,
            # rough: int8 LSTM + fp32 MLP
            "flash_kb": ((4 * 128 * 6 + 4 * 128 * 128) / 1024.0)
            + ((64 * 128 + 64 + 64 + 1) * 4 / 1024.0),
            "so_kb": quant_so.stat().st_size / 1024.0 if quant_so.exists() else 0.0,
        },
    }

    # Plots
    plot_streaming_dashboard(y, preds, out_dir / "soh_streaming_dashboard.png")
    plot_mae_and_hist_combined(metrics, y_sliced, preds_sliced, out_dir / "soh_mae_hist_combined.png")
    plot_model_sizes(size_info, out_dir / "soh_model_sizes.png")

    # Save metrics table
    rows = []
    for name, m in metrics.items():
        rows.append({"Model": name, "MAE": m["MAE"], "STD": m["STD"], "RMSE": m["RMSE"], "MAX": m["MAX"]})
    with open(out_dir / "soh_plot_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[done] SOH plots (SOC style) written to {out_dir}")


if __name__ == "__main__":
    main()
