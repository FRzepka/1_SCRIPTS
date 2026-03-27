import numpy as np
import matplotlib.pyplot as plt
import os
import math
import torch
import pandas as pd
from pathlib import Path

# Settings
FILE_PATH = r'C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_LSTM_MLP\5_benchmark\PC\SOC\bench_v_soc_full\soc_streaming_base_quant_pruned_data.npz'
PARQUET_PATH = r'C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet'
OUT_DIR = os.path.dirname(FILE_PATH)

# Checkpoints for model size calculation
BASE_CKPT = r'C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_LSTM_MLP\2_models\base\soc_1.5.0.0_base\1.5.0.0_soc_epoch0001_rmse0.02897.pt'
PRUNED_CKPT = r'C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_LSTM_MLP\2_models\pruned\soc_1.5.0.0_pruned\prune_30pct_20250916_140404\soc_pruned_hidden45.pt'

# Colors requested by user
COLORS = {
    'Base': '#2ca02c',      # Green
    'Quantized': '#1f77b4', # Blue
    'Pruned': '#d62728'     # Red
}
# Map keys from NPZ/Script to these keys
KEY_MAP = {
    'base': 'Base',
    'quant': 'Quantized',
    'pruned': 'Pruned',
    'BASE': 'Base',
    'QUANT': 'Quantized',
    'PRUNED': 'Pruned'
}

STRIDE = 50

def get_color(name):
    # Try direct match
    if name in COLORS: return COLORS[name]
    # Try mapped match
    if name in KEY_MAP: return COLORS[KEY_MAP[name]]
    return 'gray'

def plot_combined_dashboard(y, base, quant, pruned, soh, out_png, skip_initial=100000):
    # Create figure with 2 subplots (SOC/SOH on top, Errors on bottom)
    fig, (ax_soc, ax_err) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Data Preparation (Skip Initial + Downsample) ---
    s = slice(skip_initial, None, STRIDE)
    x_axis = np.arange(len(y))[s]
    
    y_s = y[s]
    base_s = base[s] if base is not None else None
    quant_s = quant[s] if quant is not None else None
    pruned_s = pruned[s] if pruned is not None else None
    soh_s = soh[s] if soh is not None else None

    # --- Top Plot: SOC (Left Axis) & SOH (Right Axis) ---
    # SOC
    ax_soc.plot(x_axis, y_s, label="GT SOC", linewidth=1.0, alpha=0.9, color="black")
    if base_s is not None:
        ax_soc.plot(x_axis, base_s, label="BASE", linewidth=0.9, alpha=0.9, color=COLORS['Base'])
    if quant_s is not None:
        ax_soc.plot(x_axis, quant_s, label="QUANT", linewidth=0.9, alpha=0.9, color=COLORS['Quantized'])
    if pruned_s is not None:
        ax_soc.plot(x_axis, pruned_s, label="PRUNED", linewidth=0.9, alpha=0.9, color=COLORS['Pruned'])
    
    ax_soc.set_ylabel("SOC")
    ax_soc.set_title(f"SOC & SOH Prediction + Error Analysis (Skipped first {skip_initial})")
    ax_soc.legend(loc='lower left')
    ax_soc.grid(alpha=0.2)

    # SOH (Twin Axis)
    if soh_s is not None:
        ax_soh = ax_soc.twinx()
        ax_soh.plot(x_axis, soh_s, label="SOH", linewidth=1.0, color="purple", linestyle='--')
        ax_soh.set_ylabel("SOH")
        # Combine legends? Or just put SOH legend separately
        ax_soh.legend(loc='lower right')
    
    # --- Bottom Plot: Errors ---
    if base_s is not None:
        err_base = base_s - y_s
        ax_err.plot(x_axis, err_base, label="Err BASE", linewidth=0.8, alpha=0.8, color=COLORS['Base'])
    
    if quant_s is not None:
        err_quant = quant_s - y_s
        ax_err.plot(x_axis, err_quant, label="Err QUANT", linewidth=0.8, alpha=0.8, color=COLORS['Quantized'])
        
    if pruned_s is not None:
        err_pruned = pruned_s - y_s
        ax_err.plot(x_axis, err_pruned, label="Err PRUNED", linewidth=0.8, alpha=0.8, color=COLORS['Pruned'])

    ax_err.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_err.set_ylabel("Error (Pred - GT)")
    ax_err.set_xlabel("Step")
    ax_err.legend(loc='upper right')
    ax_err.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_error(y, pred, label, color_key, out_png):
    if pred is None:
        return
    
    # Downsample
    s = slice(None, None, STRIDE)
    x_axis = np.arange(len(y))[s]
    err = (pred - y)[s]

    plt.figure(figsize=(12, 3))
    plt.plot(x_axis, err, linewidth=0.7, color=COLORS[color_key])
    plt.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.title(f"{label} - GT (Streaming, first N steps)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def compute_error_metrics(y_true, pred):
    err = pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(float(np.mean(err ** 2))))
    max_abs = float(np.max(np.abs(err)))
    return {"MAE": mae, "RMSE": rmse, "MAX": max_abs}

def plot_metrics_bars(metrics, out_png):
    names = list(metrics.keys())
    mae = [metrics[n]["MAE"] for n in names]
    rmse = [metrics[n]["RMSE"] for n in names]
    mxe = [metrics[n]["MAX"] for n in names]

    x = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    
    bar_kwargs = dict(alpha=0.6, edgecolor="#222222", linewidth=1.2)
    ax.bar(x - width, mae, width, label="MAE", **bar_kwargs)
    ax.bar(x, rmse, width, label="RMSE", **bar_kwargs)
    ax.bar(x + width, mxe, width, label="MAX |err|", **bar_kwargs)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Error")
    ax.set_title("SOC Error-Metriken vs. Groundtruth")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_error_histograms(y, preds, out_png, bins=100):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, pred in preds.items():
        err = pred - y
        # Map name to color key
        c_key = KEY_MAP.get(name, 'Base')
        ax.hist(
            err,
            bins=bins,
            alpha=0.4,
            label=name,
            color=COLORS[c_key],
            histtype="stepfilled",
        )
    ax.set_xlabel("Error (pred - GT)")
    ax.set_ylabel("Count")
    ax.set_title("SOC error distribution (streaming, full run)")
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_parity(y, preds, out_png, max_points=20000):
    n = len(y)
    # Use stride for parity plot as well if n is large, but respect max_points
    # If we just take every 50th point, we might get fewer than max_points, which is fine.
    # Or we can just use the stride logic requested.
    
    s = slice(None, None, STRIDE)
    y_s = y[s]
    preds_s = {k: v[s] for k, v in preds.items()}
    
    # If still too many points, subsample further
    if len(y_s) > max_points:
        idx = np.linspace(0, len(y_s) - 1, max_points).astype(int)
        y_s = y_s[idx]
        preds_s = {k: v[idx] for k, v in preds_s.items()}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
    for name, pred in preds_s.items():
        c_key = KEY_MAP.get(name, 'Base')
        ax.scatter(
            y_s,
            pred,
            s=2,
            alpha=0.5,
            label=name,
            color=COLORS[c_key]
        )
    ax.set_xlabel("GT SOC")
    ax.set_ylabel("Pred SOC")
    ax.set_title("Parity-Plot SOC (Streaming, full run)")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=3, loc='lower right')
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def calculate_model_sizes():
    # Load checkpoints
    if not os.path.exists(BASE_CKPT):
        print(f"Warning: Base checkpoint not found at {BASE_CKPT}")
        return None
    if not os.path.exists(PRUNED_CKPT):
        print(f"Warning: Pruned checkpoint not found at {PRUNED_CKPT}")
        return None
        
    sd_base = torch.load(BASE_CKPT, map_location='cpu')
    if 'model_state_dict' in sd_base: sd_base = sd_base['model_state_dict']
    
    sd_pruned = torch.load(PRUNED_CKPT, map_location='cpu')
    if 'model_state_dict' in sd_pruned: sd_pruned = sd_pruned['model_state_dict']
    
    def count_params(sd):
        return sum(p.numel() for p in sd.values())

    base_params = count_params(sd_base)
    pruned_params = count_params(sd_pruned)
    
    base_size_kb = (base_params * 4) / 1024
    pruned_size_kb = (pruned_params * 4) / 1024
    
    # Quantized size estimation
    lstm_keys = ["lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0"]
    base_lstm_bytes = sum(sd_base[k].numel() * 4 for k in lstm_keys if k in sd_base)
    base_rest_bytes = (base_params * 4) - base_lstm_bytes
    
    w_ih = sd_base["lstm.weight_ih_l0"]
    w_hh = sd_base["lstm.weight_hh_l0"]
    b_ih = sd_base.get("lstm.bias_ih_l0")
    b_hh = sd_base.get("lstm.bias_hh_l0")
    
    int8_bytes = w_ih.numel() + w_hh.numel()
    hidden_size = w_hh.shape[1]
    scales_bytes = (4 * hidden_size + 4 * hidden_size) * 4 
    
    bias_bytes = 0
    if b_ih is not None: bias_bytes += b_ih.numel() * 4
    if b_hh is not None: bias_bytes += b_hh.numel() * 4
    
    quant_lstm_bytes = int8_bytes + scales_bytes + bias_bytes
    quant_total_kb = (base_rest_bytes + quant_lstm_bytes) / 1024
    
    return {
        'Base': {'Params': base_params, 'Size_KB': base_size_kb},
        'Quantized': {'Params': base_params, 'Size_KB': quant_total_kb},
        'Pruned': {'Params': pruned_params, 'Size_KB': pruned_size_kb}
    }

def plot_model_size_bars(sizes, out_png):
    if sizes is None: return
    
    names = list(sizes.keys())
    params_m = [sizes[n]['Params'] / 1e6 for n in names]
    size_kb = [sizes[n]['Size_KB'] for n in names]
    colors = [COLORS[n] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, params_m, width, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Parameter [Mio.]")
    ax1.set_title("Parameteranzahl pro Modell")
    ax1.grid(axis="y", alpha=0.2)

    ax2.bar(x, size_kb, width, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Größe [KB]")
    ax2.set_title("Gewichtsspeicher (geschätzt)")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    print(f"Loading data from {FILE_PATH}...")
    data = np.load(FILE_PATH)
    y_true = data['y']
    
    # Extract predictions
    base = data['base'] if 'base' in data else None
    quant = data['quant'] if 'quant' in data else None
    pruned = data['pruned'] if 'pruned' in data else None

    # Load SOH
    soh = None
    if os.path.exists(PARQUET_PATH):
        print(f"Loading SOH from {PARQUET_PATH}...")
        try:
            df = pd.read_parquet(PARQUET_PATH, columns=['SOH'])
            # Slice SOH to match y_true length
            if len(df) >= len(y_true):
                soh = df['SOH'].values[:len(y_true)]
            else:
                print(f"Warning: SOH data length ({len(df)}) is shorter than SOC data ({len(y_true)}). Padding with NaN.")
                soh = np.full(len(y_true), np.nan)
                soh[:len(df)] = df['SOH'].values
        except Exception as e:
            print(f"Error loading SOH: {e}")
    else:
        print(f"Warning: Parquet file not found at {PARQUET_PATH}")
    
    print("Generating plots...")
    
    # Overlay
    plot_combined_dashboard(y_true, base, quant, pruned, soh, os.path.join(OUT_DIR, "soc_streaming_dashboard.png"), skip_initial=500)
    
    # Errors (Separate plots still useful? User said "jetzt werden die error... extra geplottet... ich fände es aber eigentlich auch gut wenn die auch unterhalb geplottet werden". 
    # This implies the dashboard replaces the need for separate error plots, OR adds to it. 
    # I'll keep the separate ones for now but maybe skip them if user wants only the dashboard.
    # But user said "die errors alle in einem plot". The dashboard does that.
    
    # Let's keep the separate ones as backup but maybe rename them or just leave them.
    if base is not None:
        plot_error(y_true, base, "BASE (fp32)", 'Base', os.path.join(OUT_DIR, "soc_streaming_error_base_firstN.png"))
    if quant is not None:
        plot_error(y_true, quant, "QUANT (int8)", 'Quantized', os.path.join(OUT_DIR, "soc_streaming_error_quant_firstN.png"))
    if pruned is not None:
        plot_error(y_true, pruned, "PRUNED (fp32)", 'Pruned', os.path.join(OUT_DIR, "soc_streaming_error_pruned_firstN.png"))
        
    # Metrics
    preds = {}
    if base is not None: preds['BASE'] = base
    if quant is not None: preds['QUANT'] = quant
    if pruned is not None: preds['PRUNED'] = pruned
    
    metrics = {name: compute_error_metrics(y_true, p) for name, p in preds.items()}
    plot_metrics_bars(metrics, os.path.join(OUT_DIR, "soc_metrics_bar.png"))
    
    # Histograms & Parity
    plot_error_histograms(y_true, preds, os.path.join(OUT_DIR, "soc_error_hist.png"))
    plot_parity(y_true, preds, os.path.join(OUT_DIR, "soc_parity_plot.png"))
    
    # Model Sizes
    sizes = calculate_model_sizes()
    plot_model_size_bars(sizes, os.path.join(OUT_DIR, "soc_model_sizes.png"))
    
    print("Done! Plots regenerated.")

if __name__ == "__main__":
    main()
