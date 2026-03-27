import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, MaxNLocator
import math
import glob
import json

# --- Configuration ---
# Detect OS and set BASE_DIR accordingly
if os.name == 'nt': # Windows
    BASE_DIR = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts"
else: # Linux/Other
    BASE_DIR = "/home/florianr/MG_Farm/1_Scripts"

OUT_DIR = os.path.join(BASE_DIR, "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC_SOH_Combined_Results")

# Data Paths
SOC_NPZ = os.path.join(BASE_DIR, "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC/bench_v_soc_full/soc_streaming_base_quant_pruned_data.npz")
SOH_NPZ = os.path.join(BASE_DIR, "DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH/BENCH_SOH_FULL_FINAL_20251124/benchmark_results.npz")
PARQUET_PATH = os.path.join(BASE_DIR, "3_Projekte/MG_Farm/5_Data/01_LFP/00_Data/Versuch_18650_standart/MGFarm_18650_FE/df_FE_C07.parquet")

# STM32 Results Paths
SOC_STM32_DIR = os.path.join(BASE_DIR, "DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOC") # Path to new JSONs
SOH_STM32_DIR = os.path.join(BASE_DIR, "DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOH") # Path to new JSONs

# Model Sizes Data (Hardcoded from previous analysis)
# Format: {Model: {Flash: bytes, RAM: bytes, Params: count}}
# SOC Values (from regenerate_plots_styled.py and reports)
SOC_SIZES = {
    # RAM values updated to measured Static+Stack from STM32 JSON benchmarks
    'Base':      {'Flash': 107844, 'RAM': 5036, 'Params': 22785},
    'Pruned':    {'Flash': 63768,  'RAM': 4112, 'Params': 13500},  # Approx params
    'Quantized': {'Flash': 53744,  'RAM': 3576, 'Params': 22785}
}

# SOH Values (Initial defaults, will be updated from JSONs)
SOH_SIZES = {
    'Base':      {'Flash': 343044, 'RAM': 7680, 'Params': 85761},
    'Pruned':    {'Flash': 186788, 'RAM': 5552, 'Params': 46697},
    'Quantized': {'Flash': 141316, 'RAM': 7680, 'Params': 85761}
}

# Colors
# Pipeline-aligned palette, jetzt etwas kräftiger:
# Base = warmes Gelb/Orange, Pruned = kräftiges Blau, Quantized = sattes Grün.
BASE_HEX = "#2ca02c"      # green  (Base)
PRUNED_HEX = "#d62728"    # red    (Pruned)
QUANT_HEX = "#1f77b4"     # blue   (Quantized)

def lighten_color(color, amount=0.3):
    """Blend the given color with white by the given fraction."""
    r, g, b, a = to_rgba(color)
    r = (1 - amount) * r + amount * 1.0
    g = (1 - amount) * g + amount * 1.0
    b = (1 - amount) * b + amount * 1.0
    return (r, g, b, a)

# Base, Pruned, Quantized colours for generic plots (SOC-focused)
STD_COLORS = {
    'Base': BASE_HEX,
    'Pruned': PRUNED_HEX,
    'Quantized': QUANT_HEX,
}

# For combined SOC/SOH bar charts we use the same hues, but SOH is drawn
# slightly lighter to preserve the "darker SOC / lighter SOH" convention.
COLORS = {
    'SOC_Base':      BASE_HEX,
    'SOH_Base':      lighten_color(BASE_HEX, amount=0.4),
    'SOC_Pruned':    PRUNED_HEX,
    'SOH_Pruned':    lighten_color(PRUNED_HEX, amount=0.4),
    'SOC_Quantized': QUANT_HEX,
    'SOH_Quantized': lighten_color(QUANT_HEX, amount=0.4),
}

# Default font sizes (used by most plots; MAE/histograms override locally)
FS_TITLE = 16
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 11

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def get_color(name, type_prefix=None):
    if type_prefix:
        key = f"{type_prefix}_{name}"
        if key in COLORS:
            return COLORS[key]
    if name in STD_COLORS:
        return STD_COLORS[name]
    return 'gray'

def apply_boxplot_style(bplot, colors_list):
    """Apply consistent styling to boxplots."""
    for i, color in enumerate(colors_list):
        r, g, b, _ = to_rgba(color)
        edge_color = (r, g, b, 1.0)
        face_color = (r, g, b, 0.4)
        
        # Box
        patch = bplot['boxes'][i]
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(2.0)
        
        # Median - make it black or dark gray for visibility
        median = bplot['medians'][i]
        median.set_color('black')
        median.set_linewidth(1.5)
        
        # Whiskers
        bplot['whiskers'][i*2].set_color(edge_color)
        bplot['whiskers'][i*2].set_linewidth(1.5)
        bplot['whiskers'][i*2+1].set_color(edge_color)
        bplot['whiskers'][i*2+1].set_linewidth(1.5)
        
        # Caps
        bplot['caps'][i*2].set_color(edge_color)
        bplot['caps'][i*2].set_linewidth(1.5)
        bplot['caps'][i*2+1].set_color(edge_color)
        bplot['caps'][i*2+1].set_linewidth(1.5)

def plot_mae_hist_styled(preds, y_true, title_suffix, out_path):
    """Generates the styled MAE Boxplot + Histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # These figures are placed as single-column images in the paper, so we use
    # slightly larger in-figure typography than the multi-panel plots.
    fs_title = FS_TITLE + 4
    fs_label = FS_LABEL + 4
    fs_tick = FS_TICK + 3
    fs_legend = FS_LEGEND + 2
    
    names = list(preds.keys())
    # Sort names: Base, Pruned, Quantized
    order = ['Base', 'Pruned', 'Quantized']
    sorted_names = [n for n in order if n in names]
    
    SCALE = 100.0  # convert absolute error from fraction to percentage points

    # Prepare data for boxplot (Absolute Errors in %)
    data_to_plot = []
    colors_list = []
    for name in sorted_names:
        pred = preds[name]
        abs_err = np.abs(pred - y_true) * SCALE
        data_to_plot.append(abs_err)
        colors_list.append(STD_COLORS[name])

    # Plot boxplot
    bplot = ax1.boxplot(
        data_to_plot,
        tick_labels=sorted_names,
        patch_artist=True,
        showfliers=False,
        widths=0.5
    )
    
    apply_boxplot_style(bplot, colors_list)
    
    ax1.set_ylabel("Absolute Error [%]", fontsize=fs_label)
    ax1.set_title(f"Error Distribution (Boxplot) - {title_suffix}", fontsize=fs_title)
    ax1.tick_params(axis='both', which='major', labelsize=fs_tick)
    ax1.grid(axis="y", alpha=0.2)
    
    # Create custom legend with MAE values
    from matplotlib.patches import Patch
    legend_elements = []
    for name, c in zip(sorted_names, colors_list):
        mae = np.mean(np.abs(preds[name] - y_true)) * SCALE
        label_text = f"{name}\nMAE: {mae:.2f}"
        r, g, b, _ = to_rgba(c)
        legend_elements.append(Patch(
            facecolor=(r, g, b, 0.4),
            edgecolor=(r, g, b, 1.0),
            linewidth=2.0,
            label=label_text
        ))
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=fs_legend)

    # --- Right Plot: Histogram ---
    for name in sorted_names:
        pred = preds[name]
        err = (pred - y_true) * SCALE
        c = STD_COLORS[name]
        ax2.hist(
            err,
            bins=100,
            range=(-10.0, 10.0),  # Fixed range in percentage points
            alpha=0.4,
            label=name,
            color=c,
             histtype="stepfilled",
         )

    ax2.set_xlabel("Error (pred - GT) [%]", fontsize=fs_label)

    # Reduce tick-label crowding on the count axis by expressing counts in 10^3.
    ax2.set_ylabel("Count [k]", fontsize=fs_label)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000:g}"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax2.set_title(f"Error Histogram (Counts) - {title_suffix}", fontsize=fs_title)
    ax2.tick_params(axis='both', which='major', labelsize=fs_tick)
    ax2.legend(loc='upper right', fontsize=fs_legend)
    ax2.grid(alpha=0.2)

    fig.tight_layout(w_pad=2.0)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_dashboard_styled(time_axis, y_true, preds, title_suffix, out_path, order_override=None):
    """Generates the styled Dashboard (Predictions + Errors).

    The time axis is provided in seconds. For improved interpretability we convert it
    to days or hours depending on the total duration: full-stream SOC/SOH plots span
    several days and are therefore shown in days, while zoomed windows are rendered
    in hours. Short snippets remain in seconds.
    """
    time_axis = np.asarray(time_axis)
    max_t = float(time_axis.max()) if time_axis.size > 0 else 0.0
    if max_t >= 24 * 3600.0:
        x = time_axis / (24.0 * 3600.0)
        x_label = "Time [d]"
    elif max_t >= 3600.0:
        x = time_axis / 3600.0
        x_label = "Time [h]"
    else:
        x = time_axis
        x_label = "Time [s]"

    SCALE = 100.0  # convert SOC/SOH and errors to percentage units

    fig, (ax_main, ax_err) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Top Plot: Predictions ---
    ax_main.plot(x, y_true * SCALE, label=f"GT {title_suffix}", linewidth=1.0, alpha=0.9, color="black")

    if order_override is not None:
        order = list(order_override)
    else:
        # Requested plot order for SOC dashboard (visibility experiment):
        # GT -> Quantized -> Pruned -> Base (Base drawn last).
        if str(title_suffix).strip().upper() == "SOC":
            order = ['Quantized', 'Pruned', 'Base']
        else:
            order = ['Base', 'Pruned', 'Quantized']
    for name in order:
        if name in preds:
            ax_main.plot(x, preds[name] * SCALE, label=name, linewidth=1.2, alpha=1.0, color=STD_COLORS[name])

    ax_main.set_ylabel(f"{title_suffix} [%]")
    ax_main.set_title(f"{title_suffix} Prediction + Error Analysis")
    ax_main.legend(loc='lower left')
    ax_main.grid(alpha=0.2)
    ax_main.tick_params(axis='both')

    # --- Bottom Plot: Errors ---
    for name in order:
        if name in preds:
            err = (preds[name] - y_true) * SCALE
            ax_err.plot(x, err, label=f"Err {name}", linewidth=1.0, alpha=0.9, color=STD_COLORS[name])

    ax_err.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_err.set_ylabel("Error (Pred - GT) [%]")
    ax_err.set_xlabel(x_label)
    ax_err.legend(loc='lower left')
    ax_err.grid(alpha=0.2)
    ax_err.tick_params(axis='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# --- Plotting Functions ---

def plot_soc_individual():
    print("Generating SOC Individual Plots...")
    if not os.path.exists(SOC_NPZ):
        print(f"Error: SOC NPZ not found at {SOC_NPZ}")
        return

    data = np.load(SOC_NPZ)
    y_true = data['y']
    base = data['base'] if 'base' in data else None
    quant = data['quant'] if 'quant' in data else None
    pruned = data['pruned'] if 'pruned' in data else None
    
    # Load Time Axis
    time_axis = None
    if os.path.exists(PARQUET_PATH):
        try:
            df = pd.read_parquet(PARQUET_PATH, columns=['Testtime[s]'])
            if len(df) >= len(y_true):
                time_axis = df['Testtime[s]'].values[:len(y_true)]
        except Exception as e:
            print(f"Warning: Could not load parquet: {e}")
    
    if time_axis is None:
        time_axis = np.arange(len(y_true))

    # Prepare Data (Skip initial)
    skip = 1000
    stride = 50
    s = slice(skip, None, stride)
    
    preds_sliced = {}
    if base is not None: preds_sliced['Base'] = base[s]
    if pruned is not None: preds_sliced['Pruned'] = pruned[s]
    if quant is not None: preds_sliced['Quantized'] = quant[s]
    y_sliced = y_true[s]
    t_sliced = time_axis[s]

    # 1. Dashboard
    plot_dashboard_styled(t_sliced, y_sliced, preds_sliced, "SOC", os.path.join(OUT_DIR, "soc_streaming_dashboard.png"))

    # Alternate dashboard ordering for visual comparison: Pruned drawn last (on top).
    plot_dashboard_styled(
        t_sliced,
        y_sliced,
        preds_sliced,
        "SOC",
        os.path.join(OUT_DIR, "soc_streaming_dashboard_pruned_last.png"),
        order_override=["Quantized", "Base", "Pruned"],
    )
    
    # 2. MAE & Hist Combined
    # Use full data (minus skip) for histogram/boxplot for better stats, or sliced? 
    # Reference used sliced data for plots but maybe full for hist? 
    # Let's use sliced to match dashboard visual.
    plot_mae_hist_styled(preds_sliced, y_sliced, "SOC", os.path.join(OUT_DIR, "soc_mae_hist_combined.png"))
    
    # 3. Zoomed Plots (PC vs STM32 comparison excerpts)
    ranges = [
        (30000, 40000, "soc_pc_comparison_zoomed_30k_40k.png"),
        (657000, 897000, "soc_pc_comparison_checkup_657k_897k.png")
    ]
    
    for start, end, fname in ranges:
        mask = (time_axis >= start) & (time_axis <= end)
        if not np.any(mask):
            continue

        t_slice = time_axis[mask]
        y_slice = y_true[mask] * 100.0

        fig, (ax_s, ax_e) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        ax_s.plot(t_slice, y_slice, 'k-', alpha=0.5, label='GT')

        for k in ['Base', 'Pruned', 'Quantized']:
            if k == 'Base' and base is not None:
                p = base[mask] * 100.0
            elif k == 'Pruned' and pruned is not None:
                p = pruned[mask] * 100.0
            elif k == 'Quantized' and quant is not None:
                p = quant[mask] * 100.0
            else:
                continue

            ax_s.plot(t_slice, p, label=k, color=STD_COLORS[k], linewidth=1, alpha=0.8)
            ax_e.plot(t_slice, p - y_slice, label=k, color=STD_COLORS[k], linewidth=0.8, alpha=0.7)

        ax_s.set_title(f"SOC comparison {start}-{end} s")
        ax_s.set_ylabel("SOC [%]")
        ax_s.legend(loc="lower left")
        ax_s.grid(alpha=0.3)

        ax_e.set_ylabel("Error (Pred - GT) [%]")
        ax_e.set_xlabel("Time [s]")
        ax_e.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close()

def plot_soh_individual():
    print("Generating SOH Individual Plots...")
    if not os.path.exists(SOH_NPZ):
        print(f"Error: SOH NPZ not found at {SOH_NPZ}")
        return

    data = np.load(SOH_NPZ)
    # Keys: ['y_gt', 'Py_Base', 'Py_Pruned', 'C_Base', 'C_Pruned', 'C_Quant']
    if 'y_gt' not in data:
        print("Error: 'y_gt' not found in SOH NPZ")
        return
        
    y_true = data['y_gt']
    
    # Map keys
    base = data['C_Base'] if 'C_Base' in data else None
    pruned = data['C_Pruned'] if 'C_Pruned' in data else None
    quant = data['C_Quant'] if 'C_Quant' in data else None
    
    # SOH usually doesn't have a separate time axis in the NPZ, we might need to infer or load.
    # Assuming same length as SOC or just index.
    # For now, use index as time if no other info.
    time_axis = np.arange(len(y_true))

    # Prepare Data (Skip initial)
    skip = 100 # SOH settles faster or we just skip less
    stride = 1 # SOH is usually shorter, maybe no stride needed?
    # If data is huge, stride. If it's the 20k benchmark, no stride.
    if len(y_true) > 50000:
        stride = 10
    
    s = slice(skip, None, stride)
    
    preds_sliced = {}
    if base is not None: preds_sliced['Base'] = base[s]
    if pruned is not None: preds_sliced['Pruned'] = pruned[s]
    if quant is not None: preds_sliced['Quantized'] = quant[s]
    y_sliced = y_true[s]
    t_sliced = time_axis[s]

    # 1. Dashboard
    plot_dashboard_styled(t_sliced, y_sliced, preds_sliced, "SOH", os.path.join(OUT_DIR, "soh_streaming_dashboard.png"))
    
    # 2. MAE & Hist Combined
    plot_mae_hist_styled(preds_sliced, y_sliced, "SOH", os.path.join(OUT_DIR, "soh_mae_hist_combined.png"))

def load_stm32_soh_json_results(dir_path):
    """Loads benchmark results from JSON files."""
    results = {}
    for m in ['Base', 'Pruned', 'Quantized']:
        fpath = os.path.join(dir_path, f"result_{m}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r') as f:
                    results[m] = json.load(f)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return results

def load_stm32_soc_json_results(dir_path):
    """Loads benchmark results from JSON files (SOC)."""
    results = {}
    for m in ['Base', 'Pruned', 'Quantized']:
        fpath = os.path.join(dir_path, f"result_{m}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r') as f:
                    results[m] = json.load(f)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return results

def plot_model_sizes_combined():
    print("Generating Combined Model Sizes Plot...")
    
    # SOC RAM: keep values from SOC_SIZES (from map/static analysis).
    # For SOH we have reliable on-device RAM measurements in the JSONs
    # (static_ram_bytes + max_stack_bytes), so we update only SOH here.

    # Load SOH Real RAM Data
    soh_results = load_stm32_soh_json_results(SOH_STM32_DIR)
    for m, res in soh_results.items():
        if m in SOH_SIZES:
            static_val = res.get('static_ram_bytes', 0)
            stack_val = res.get('max_stack_bytes', 0)
            if static_val and stack_val:
                total = static_val + stack_val
                SOH_SIZES[m]['RAM'] = total
                print(f"Updated SOH {m} RAM to {SOH_SIZES[m]['RAM']} bytes")

    # Prepare Data
    models = ['Base', 'Pruned', 'Quantized']
    
    # 1. Parameters
    soc_params = [SOC_SIZES[m]['Params'] for m in models]
    soh_params = [SOH_SIZES[m]['Params'] for m in models]
    
    # 2. Estimated Flash
    # Base: FP32 -> 4 bytes
    # Pruned: FP32 -> 4 bytes (but fewer params)
    # Quantized: INT8 -> 1 byte
    
    def calc_est_flash(params, model_type):
        if model_type == 'Quantized':
            return params * 1.0 / 1024.0 # 1 Byte per param
        else:
            return params * 4.0 / 1024.0 # 4 Bytes per param

    soc_est = [calc_est_flash(SOC_SIZES[m]['Params'], m) for m in models]
    soh_est = [calc_est_flash(SOH_SIZES[m]['Params'], m) for m in models]
    
    # 3. Actual Flash
    soc_flash = [SOC_SIZES[m]['Flash'] / 1024.0 for m in models]
    soh_flash = [SOH_SIZES[m]['Flash'] / 1024.0 for m in models]
    
    # 4. RAM
    soc_ram = [SOC_SIZES[m]['RAM'] / 1024.0 for m in models]
    soh_ram = [SOH_SIZES[m]['RAM'] / 1024.0 for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Helper to plot grouped bars
    def plot_grouped(ax, data_soc, data_soh, title, ylabel):
        x = np.arange(len(models))
        width = 0.35
        
        # SOC Bars
        bars_soc = ax.bar(x - width/2, data_soc, width, label='SOC')
        # SOH Bars
        bars_soh = ax.bar(x + width/2, data_soh, width, label='SOH')
        
        # Color them
        for i, m in enumerate(models):
            # SOC
            c_soc = get_color(m, 'SOC')
            bars_soc[i].set_facecolor(to_rgba(c_soc, 0.4))
            bars_soc[i].set_edgecolor(to_rgba(c_soc, 1.0))
            bars_soc[i].set_linewidth(1.5)
            
            # SOH
            c_soh = get_color(m, 'SOH')
            bars_soh[i].set_facecolor(to_rgba(c_soh, 0.4))
            bars_soh[i].set_edgecolor(to_rgba(c_soh, 1.0))
            bars_soh[i].set_linewidth(1.5)
            
            # Add labels
            ax.text(x[i] - width/2, data_soc[i], f'{data_soc[i]:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(x[i] + width/2, data_soh[i], f'{data_soh[i]:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        
        # Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', label='Left: SOC (Darker)'),
            Patch(facecolor='lightgray', edgecolor='black', label='Right: SOH (Lighter)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Plot 1: Parameters
    plot_grouped(axes[0, 0], soc_params, soh_params, "Parameter Count", "Count")
    
    # Plot 2: Estimated Flash
    plot_grouped(axes[0, 1], soc_est, soh_est, "Estimated Flash (Weights Only)", "Size [KB]")
    
    # Plot 3: Actual Flash
    plot_grouped(axes[1, 0], soc_flash, soh_flash, "Actual Flash Usage (Binary)", "Size [KB]")
    
    # Plot 4: RAM
    plot_grouped(axes[1, 1], soc_ram, soh_ram, "RAM Usage (Static + Stack)", "Size [KB]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_model_sizes.png"), dpi=300)
    plt.close()

def plot_latency_combined():
    print("Generating Combined Latency Plot...")
    
    # Load JSON results (preferred source)
    soc_results = load_stm32_soc_json_results(SOC_STM32_DIR)
    soh_results = load_stm32_soh_json_results(SOH_STM32_DIR)

    def extract_latencies_ms(results_dict):
        """Return concatenated latency array in ms from raw_metrics.time_us (fallback to avg_time_us)."""
        all_lat = []
        for m in ['Base', 'Pruned', 'Quantized']:
            if m not in results_dict:
                continue
            res = results_dict[m]
            raw = res.get('raw_metrics', {}).get('time_us', [])
            if raw:
                arr = np.array(raw, dtype=float) / 1000.0
            else:
                avg_us = res.get('avg_time_us', 0.0)
                arr = np.array([avg_us / 1000.0], dtype=float)
            all_lat.append(arr)
        if all_lat:
            return np.concatenate(all_lat)
        return np.array([], dtype=float)

    soc_lat_all = extract_latencies_ms(soc_results)
    soh_lat_all = extract_latencies_ms(soh_results)

    # Optional fallback to CSVs if JSONs not present
    if soc_lat_all.size == 0 or soh_lat_all.size == 0:
        def load_latencies_csv(d):
            vals = []
            for m in ['Base', 'Pruned', 'Quantized']:
                f = os.path.join(d, f"results_{m}.csv")
                if os.path.exists(f):
                    df = pd.read_csv(f)
                    if 'latency_ms' in df.columns:
                        vals.append(df['latency_ms'].values)
            if vals:
                return np.concatenate(vals)
            return np.array([], dtype=float)

        if soc_lat_all.size == 0:
            print("Falling back to SOC CSV latency results...")
            soc_lat_all = load_latencies_csv(SOC_STM32_DIR)
        if soh_lat_all.size == 0:
            print("Falling back to SOH CSV latency results...")
            soh_lat_all = load_latencies_csv(SOH_STM32_DIR)

    if soc_lat_all.size == 0 or soh_lat_all.size == 0:
        print("Warning: Could not find latency data for SOC or SOH. Skipping combined latency plot.")
        return

    # --- Figure with Histogram (left) + Boxplot (right) ---
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram: SOC vs SOH distributions in one axis
    all_lat = np.concatenate([soc_lat_all, soh_lat_all])
    lat_min, lat_max = float(all_lat.min()), float(all_lat.max())
    if lat_min < 0:
        lat_min = 0.0
    # Add a small margin
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0
    bins = np.linspace(lat_min, lat_max + 0.1 * lat_range, 80)

    c_soc = get_color('Base', 'SOC')
    c_soh = get_color('Base', 'SOH')

    ax_hist.hist(
        soc_lat_all,
        bins=bins,
        alpha=0.5,
        label='SOC',
        color=c_soc,
        density=False,
        histtype='stepfilled'
    )
    ax_hist.hist(
        soc_lat_all,
        bins=bins,
        alpha=1.0,
        color=c_soc,
        density=False,
        histtype='step',
        linewidth=1.5
    )
 
    ax_hist.hist(
        soh_lat_all,
        bins=bins,
        alpha=0.5,
        label='SOH',
        color=c_soh,
        density=False,
        histtype='stepfilled'
    )
    ax_hist.hist(
        soh_lat_all,
        bins=bins,
        alpha=1.0,
        color=c_soh,
        density=False,
        histtype='step',
        linewidth=1.5
    )

    ax_hist.set_title("SOC vs SOH Latency Distribution (STM32)", fontsize=FS_TITLE)
    ax_hist.set_xlabel("Latency [ms]", fontsize=FS_LABEL)
    ax_hist.set_ylabel("Count", fontsize=FS_LABEL)
    ax_hist.legend(fontsize=FS_LEGEND)
    ax_hist.grid(alpha=0.3)
    ax_hist.tick_params(axis='both', which='major', labelsize=FS_TICK)

    # Boxplot: SOC vs SOH (with outliers)
    data_to_plot = [soc_lat_all, soh_lat_all]
    labels = ['SOC', 'SOH']
    colors_list = [c_soc, c_soh]

    bplot = ax_box.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.6
    )

    apply_boxplot_style(bplot, colors_list)

    ax_box.set_title("Latency Summary (Boxplot)", fontsize=FS_TITLE)
    ax_box.set_ylabel("Latency [ms]", fontsize=FS_LABEL)
    ax_box.grid(axis='y', alpha=0.3)
    ax_box.tick_params(axis='both', which='major', labelsize=FS_TICK)

    # Annotate means above each box
    means = [float(np.mean(soc_lat_all)), float(np.mean(soh_lat_all))]
    y_min, y_max = ax_box.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    for idx, mean_val in enumerate(means, start=1):
        ax_box.text(
            idx,
            y_max - 0.05 * y_span,
            f"mean={mean_val:.3f} ms",
            ha='center',
            va='top',
            fontsize=FS_LEGEND
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_inference_hist.png"), dpi=150)
    plt.close()

def plot_latency_combined_per_model():
    """Inference-time plot styled like MAE plots, per model and per task (SOC/SOH)."""
    print("Generating Per-Model Inference-Time Plot...")
    
    soc_results = load_stm32_soc_json_results(SOC_STM32_DIR)
    soh_results = load_stm32_soh_json_results(SOH_STM32_DIR)

    def extract_latencies_ms_per_model(results_dict):
        lat = {}
        for m in ['Base', 'Pruned', 'Quantized']:
            res = results_dict.get(m)
            if not res:
                continue
            raw = res.get('raw_metrics', {}).get('time_us', [])
            if raw:
                arr = np.array(raw, dtype=float) / 1000.0
            else:
                avg_us = res.get('avg_time_us', 0.0)
                arr = np.array([avg_us / 1000.0], dtype=float)
            lat[m] = arr
        return lat

    soc_lat = extract_latencies_ms_per_model(soc_results)
    soh_lat = extract_latencies_ms_per_model(soh_results)

    if not soc_lat and not soh_lat:
        print("Warning: No latency data found for SOC or SOH. Skipping per-model latency plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_soc_box, ax_soc_hist = axes[0]
    ax_soh_box, ax_soh_hist = axes[1]

    def plot_latency_row(lat_dict, title_prefix, ax_box, ax_hist):
        if not lat_dict:
            ax_box.set_visible(False)
            ax_hist.set_visible(False)
            return

        order = ['Base', 'Pruned', 'Quantized']
        names = [n for n in order if n in lat_dict]
        data = [lat_dict[n] for n in names]
        colors_list = [STD_COLORS[n] for n in names]

        # Boxplot
        bplot = ax_box.boxplot(
            data,
            tick_labels=names,
            patch_artist=True,
            showfliers=True,
            widths=0.5
        )
        apply_boxplot_style(bplot, colors_list)

        ax_box.set_ylabel("Inference Time [ms]", fontsize=FS_LABEL)
        ax_box.set_title(f"{title_prefix} Inference Time (Boxplot)", fontsize=FS_TITLE)
        ax_box.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax_box.grid(axis="y", alpha=0.2)

        # Legend mit Mittelwerten
        from matplotlib.patches import Patch
        legend_elements = []
        for name, c in zip(names, colors_list):
            mean_ms = float(np.mean(lat_dict[name]))
            r, g, b, _ = to_rgba(c)
            legend_elements.append(Patch(
                facecolor=(r, g, b, 0.4),
                edgecolor=(r, g, b, 1.0),
                linewidth=2.0,
                label=f"{name}\nmean={mean_ms:.3f} ms"
            ))
        ax_box.legend(handles=legend_elements, loc='upper right', fontsize=FS_LEGEND)

        # Histogram
        all_lat = np.concatenate(data)
        lat_min, lat_max = float(all_lat.min()), float(all_lat.max())
        if lat_min < 0:
            lat_min = 0.0
        lat_range = lat_max - lat_min if lat_max > lat_min else 1.0
        bins = np.linspace(lat_min, lat_max + 0.05 * lat_range, 80)

        for name in names:
            c = STD_COLORS[name]
            arr = lat_dict[name]
            ax_hist.hist(
                arr,
                bins=bins,
                alpha=0.4,
                label=name,
                color=c,
                histtype="stepfilled",
                density=False
            )
        ax_hist.set_xlabel("Inference Time [ms]", fontsize=FS_LABEL)
        ax_hist.set_ylabel("Count", fontsize=FS_LABEL)
        ax_hist.set_title(f"{title_prefix} Inference Time Distribution", fontsize=FS_TITLE)
        ax_hist.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax_hist.grid(alpha=0.2)
        ax_hist.legend(loc='upper right', fontsize=FS_LEGEND)

    plot_latency_row(soc_lat, "SOC", ax_soc_box, ax_soc_hist)
    plot_latency_row(soh_lat, "SOH", ax_soh_box, ax_soh_hist)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_latency_boxplot.png"), dpi=150)
    plt.close()

def plot_latency_histograms():
    """Inference-time histograms based on on-device DWT measurements."""
    print("Generating Inference-Time Histograms (per model, on-device)...")
    
    soc_results = load_stm32_soc_json_results(SOC_STM32_DIR)
    soh_results = load_stm32_soh_json_results(SOH_STM32_DIR)

    def extract_latencies_ms_per_model(results_dict):
        lat = {}
        for m in ['Base', 'Pruned', 'Quantized']:
            res = results_dict.get(m)
            if not res:
                continue
            raw = res.get('raw_metrics', {}).get('time_us', [])
            if raw:
                arr = np.array(raw, dtype=float) / 1000.0
            else:
                avg_us = res.get('avg_time_us', 0.0)
                arr = np.array([avg_us / 1000.0], dtype=float)
            lat[m] = arr
        return lat

    soc_lat = extract_latencies_ms_per_model(soc_results)
    soh_lat = extract_latencies_ms_per_model(soh_results)

    if not soc_lat and not soh_lat:
        print("Warning: No latency data found for SOC or SOH. Skipping latency histogram plot.")
        return

    fig, (ax_soc, ax_soh) = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)

    def plot_latency_side(ax, lat_dict, title_prefix):
        order = ['Base', 'Pruned', 'Quantized']
        names = [n for n in order if n in lat_dict]
        if not names:
            ax.set_visible(False)
            return

        data = [lat_dict[n] for n in names]

        # Determine sensible binning range from data and clamp to [0, 45] ms
        all_lat = np.concatenate(data)
        lat_min, lat_max = float(all_lat.min()), float(all_lat.max())
        if lat_min < 0:
            lat_min = 0.0
        # Clamp hard to the range of interest for the paper
        lat_min = max(0.0, lat_min)
        lat_max = min(45.0, lat_max)
        if lat_max <= lat_min:
            lat_min, lat_max = 0.0, 45.0
        bins = np.linspace(lat_min, lat_max, 80)

        for name in names:
            arr = lat_dict[name]
            # Clip to the visible window so rare outliers do not dominate the scale
            arr = arr[(arr >= lat_min) & (arr <= lat_max)]
            c = STD_COLORS[name]
            ax.hist(
                arr,
                bins=bins,
                alpha=0.4,
                label=name,
                color=c,
                histtype="stepfilled",
                density=False,
            )
            ax.hist(
                arr,
                bins=bins,
                alpha=1.0,
                color=c,
                histtype="step",
                density=False,
                linewidth=1.5,
            )

        ax.set_xlim(lat_min, lat_max)
        ax.set_xlabel("Inference Time [ms]", fontsize=FS_LABEL)
        ax.set_ylabel("Count", fontsize=FS_LABEL)
        ax.set_title(f"{title_prefix} Inference Time Distribution (STM32)", fontsize=FS_TITLE)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=FS_LEGEND)

    plot_latency_side(ax_soc, soc_lat, "SOC")
    plot_latency_side(ax_soh, soh_lat, "SOH")

    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_inference_hist.png"), dpi=150)
    plt.close()

def plot_host_latency_histograms():
    """Host-side latency histograms (end-to-end: PC send -> PC receive)."""
    print("Generating Host Latency Histograms (per model)...")
    
    soc_results = load_stm32_soc_json_results(SOC_STM32_DIR)
    soh_results = load_stm32_soh_json_results(SOH_STM32_DIR)

    def extract_host_lat_ms_per_model(results_dict):
        lat = {}
        for m in ['Base', 'Pruned', 'Quantized']:
            res = results_dict.get(m)
            if not res:
                continue
            raw = res.get('raw_metrics', {}).get('host_latency_ms', [])
            if raw:
                arr = np.array(raw, dtype=float)
                lat[m] = arr
        return lat

    soc_lat = extract_host_lat_ms_per_model(soc_results)
    soh_lat = extract_host_lat_ms_per_model(soh_results)

    if not soc_lat and not soh_lat:
        print("Warning: No host latency data found for SOC or SOH. Skipping host-latency histogram plot.")
        return

    # Use (12, 5) to match the aspect ratio/scaling of the MAE plots
    fig, (ax_soc, ax_soh) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def plot_latency_side(ax, lat_dict, title_prefix, legend_loc="upper right"):
        order = ['Base', 'Pruned', 'Quantized']
        names = [n for n in order if n in lat_dict]
        if not names:
            ax.set_visible(False)
            return

        data = [lat_dict[n] for n in names]

        # Use a fixed window [0, 45] ms for all host-latency plots
        lat_min, lat_max = 0.0, 45.0
        bins = np.linspace(lat_min, lat_max, 80)

        for name in names:
            arr = lat_dict[name]
            # Clip to visible window so outliers do not affect the shape
            arr = arr[(arr >= lat_min) & (arr <= lat_max)]
            c = STD_COLORS[name]
            ax.hist(
                arr,
                bins=bins,
                alpha=0.4,
                label=name,
                color=c,
                histtype="stepfilled",
                density=False,
            )
            ax.hist(
                arr,
                bins=bins,
                alpha=1.0,
                color=c,
                histtype="step",
                density=False,
                linewidth=1.5,
            )

        ax.set_xlim(lat_min, lat_max)
        # Integer tick labels: 0, 5, 10, ..., 45
        ax.set_xticks(np.arange(0, 46, 5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

        # Use even larger fonts for single-column figure to match user request
        fs_title = FS_TITLE + 6
        fs_label = FS_LABEL + 6
        fs_tick = FS_TICK + 5
        fs_legend = FS_LEGEND + 4

        ax.set_xlabel("Host Latency [ms]", fontsize=fs_label)
        ax.set_ylabel("Count", fontsize=fs_label)
        ax.set_title(f"{title_prefix} Host Latency Distribution", fontsize=fs_title)
        ax.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax.grid(alpha=0.2)
        ax.legend(loc=legend_loc, fontsize=fs_legend)

    plot_latency_side(ax_soc, soc_lat, "SOC", legend_loc="upper right")
    plot_latency_side(ax_soh, soh_lat, "SOH", legend_loc="upper left")

    fig.tight_layout()
    # Host-side latency is the overall \"latency\" figure for the paper
    plt.savefig(os.path.join(OUT_DIR, "combined_latency_hist.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    
    # 1. SOC Plots
    plot_soc_individual()
    
    # 2. SOH Plots
    plot_soh_individual()
    
    # 3. Combined Model Sizes
    plot_model_sizes_combined()
    
    # 4. Combined Latency (per model histograms, on-device inference time)
    plot_latency_histograms()

    # 5. Host-side Latency (PC <-> STM32)
    plot_host_latency_histograms()
    
    print("All plots generated successfully.")
