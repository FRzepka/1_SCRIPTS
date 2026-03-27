#!/usr/bin/env python3
"""
STM32 SOC Host Benchmark (Combined)
- Connects to flashed SOC firmware over serial
- Supports switching between Base, Pruned, and Quantized models at runtime
- Generates realistic inputs using scaler centers/scales from firmware
- Sends samples as fast as possible (or at user-specified rate)
- Parses `SOC:` and `METRICS:` lines produced by firmware
- Saves CSV results and plots for all models

Usage example (Windows cmd):
  conda activate ml1
  python STM32\benchmark\benchmark_stm32_soc_combined.py --port COM7 --samples 5000 --start_idx 0
"""
import argparse
import serial
import time
import numpy as np
import os
import csv
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# These values were copied from firmware scaler_params.h (ROBUST scaler center/scale)
SCALER_CENTER = np.array([
    3.3605999947,
    0.6542999744,
    27.3999996185,
    -0.5109897852,
    0.0,
    0.0
], dtype=np.float32)
SCALER_SCALE = np.array([
    0.2009000778,
    2.6982000470,
    1.1000003815,
    0.5354322791,
    1.0,
    1.0
], dtype=np.float32)
INPUT_SIZE = 6

METRICS_RE = re.compile(r"METRICS:\s*cycles=(\d+)\s+us=([0-9.]+)\s+E_uJ=([0-9.]+)")
SOC_RE = re.compile(r"SOC:\s*([-]?\d+)\.(\d{1,3})")

MODELS = {
    'Base': 0,
    'Pruned': 1,
    'Quantized': 2
}

COLORS = {
    'Base': '#2ca02c',      # Green
    'Quantized': '#1f77b4', # Blue
    'Pruned': '#d62728'     # Red
}

def plot_bar_comparison(all_results, metric_key, title, ylabel, filename, output_dir, scale=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = list(all_results.keys())
    values = []
    colors = []
    
    for v in variants:
        df = all_results[v]
        if metric_key in df.columns:
            val = df[metric_key].mean()
        else:
            val = 0
        values.append(val * scale)
        colors.append(COLORS.get(v, 'gray'))
        
    bars = ax.bar(
        variants,
        values,
        color=colors,
        alpha=0.40,
        edgecolor=colors,
        linewidth=1.3,
        width=0.42,
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
                
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_predictions_comparison(all_results, y_true, t_true, output_dir, skip_initial=100):
    """
    Plots Ground Truth vs Predictions for all models.
    Bottom subplot shows the error (Prediction - Ground Truth).
    skip_initial: Number of initial samples to exclude from the plot (to hide settling transients).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    variants = list(all_results.keys())
    
    # Determine X axis
    if t_true is not None:
        x_axis = t_true
        xlabel = 'Test Time (s)'
    else:
        # Fallback if t_true is not provided, try to get it from the first result df
        first_df = all_results[variants[0]]
        if 'test_time' in first_df.columns:
            x_axis = first_df['test_time'].values
            xlabel = 'Test Time (s)'
        else:
            x_axis = np.arange(len(first_df))
            xlabel = 'Sample Index'

    # Apply skipping
    if len(x_axis) > skip_initial:
        x_axis = x_axis[skip_initial:]
        if y_true is not None:
            y_true = y_true[skip_initial:]
    else:
        print(f"Warning: Data length ({len(x_axis)}) <= skip_initial ({skip_initial}). Skipping disabled.")
        skip_initial = 0

    # Plot 1: Full Sequence Predictions
    if y_true is not None:
        ax1.plot(x_axis, y_true, label='Ground Truth', color='black', linewidth=1.5, alpha=0.6)
    
    for v in variants:
        df = all_results[v]
        # Ensure df matches length of original x_axis before slicing
        if len(df) > skip_initial:
            # Slice df
            pred_soc = df['pred_soc'].values[skip_initial:]
            # Ensure lengths match after slicing (in case of mismatch)
            min_len = min(len(x_axis), len(pred_soc))
            ax1.plot(x_axis[:min_len], pred_soc[:min_len], label=f'{v} Model', color=COLORS.get(v, 'gray'), linewidth=1, alpha=0.7, linestyle='--')

    ax1.set_title(f'SOC Prediction Comparison (Skipping first {skip_initial} samples)')
    ax1.set_ylabel('SOC')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    for v in variants:
        df = all_results[v]
        if 'error' in df.columns and len(df) > skip_initial:
            error = df['error'].values[skip_initial:]
            min_len = min(len(x_axis), len(error))
            ax2.plot(x_axis[:min_len], error[:min_len], label=f'{v} Error', color=COLORS.get(v, 'gray'), linewidth=1, alpha=0.8)

    ax2.set_title('Prediction Error (Model - Ground Truth)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Error (SOC)')
    ax2.grid(True, alpha=0.3)
    
    # Add zero line for error
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_predictions_combined.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_latency_stability_combined(all_results, output_dir):
    """
    Plots latency time series for all models in one plot.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    variants = list(all_results.keys())
    
    for v in variants:
        df = all_results[v]
        if 'test_time' in df.columns:
            x_axis = df['test_time']
            xlabel = 'Test Time (s)'
        else:
            x_axis = df.index
            xlabel = 'Sample Index'
            
        # Use rolling mean to smooth it slightly for readability if dense
        ax.plot(x_axis, df['latency_ms'], label=f'{v} Model', color=COLORS.get(v, 'gray'), linewidth=0.8, alpha=0.7)
            
    ax.set_ylabel('Host Latency (ms)')
    ax.set_xlabel(xlabel)
    ax.set_title('Inference Latency Stability Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_latency_stability.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_latency_histogram_combined(all_results, output_dir):
    """
    Overlaid histogram of latencies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(all_results.keys())
    
    for v in variants:
        df = all_results[v]
        ax.hist(df['latency_ms'], bins=100, range=(0, 20), 
                label=f'{v} Model', color=COLORS.get(v, 'gray'), 
                alpha=0.4, density=True, edgecolor=COLORS.get(v, 'gray'), histtype='stepfilled')
        # Add a step line on top for clarity
        ax.hist(df['latency_ms'], bins=100, range=(0, 20), 
                color=COLORS.get(v, 'gray'), density=True, histtype='step', linewidth=1.5)

    ax.set_xlabel('Host Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Latency Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_latency_hist_combined.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_mae_zoomed(all_results, output_dir):
    """
    Bar chart for MAE, but zoomed in to show differences.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = list(all_results.keys())
    values = []
    colors = []
    
    for v in variants:
        df = all_results[v]
        if 'error' in df.columns:
            mae = df['error'].abs().mean()
        else:
            mae = 0
        values.append(mae)
        colors.append(COLORS.get(v, 'gray'))
        
    bars = ax.bar(
        variants,
        values,
        color=colors,
        alpha=0.40,
        edgecolor=colors,
        linewidth=1.3,
        width=0.42,
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Zoom Y-axis
    if values and max(values) > 0:
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 2.0 if max_val != min_val else 0.1
        # If values are very close (e.g. 0.198 vs 0.199), zoom in
        ax.set_ylim(max(0, min_val - margin), max_val + margin)
                
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Model Accuracy Comparison (Zoomed)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_mae_zoomed.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_inference_time_stability_combined(all_results, output_dir):
    """
    Plots device inference time (us) time series for all models.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    variants = list(all_results.keys())
    
    for v in variants:
        df = all_results[v]
        if 'test_time' in df.columns:
            x_axis = df['test_time']
            xlabel = 'Test Time (s)'
        else:
            x_axis = df.index
            xlabel = 'Sample Index'
            
        ax.plot(x_axis, df['inference_us'], label=f'{v} Model', color=COLORS.get(v, 'gray'), linewidth=0.8, alpha=0.8)
            
    ax.set_ylabel('Device Inference Time (µs)')
    ax.set_xlabel(xlabel)
    ax.set_title('On-Device Computation Time Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_inference_time_stability.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_inference_time_dist_combined(all_results, output_dir):
    """
    Overlaid histogram of device inference times.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(all_results.keys())
    
    for v in variants:
        df = all_results[v]
        ax.hist(df['inference_us'], bins=50, 
                label=f'{v} Model', color=COLORS.get(v, 'gray'), 
                alpha=0.5, density=True)

    ax.set_xlabel('Device Inference Time (µs)')
    ax.set_ylabel('Density')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_inference_time_dist.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_latency_boxplot(all_results, output_dir):
    """
    Boxplot of latencies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(all_results.keys())
    data = [all_results[v]['latency_ms'] for v in variants]
    
    # Create boxplot
    bplot = ax.boxplot(data, patch_artist=True, labels=variants)
    
    # Color the boxes
    for patch, v in zip(bplot['boxes'], variants):
        patch.set_facecolor(COLORS.get(v, 'gray'))
        patch.set_alpha(0.6)
        
    ax.set_ylabel('Host Latency (ms)')
    ax.set_title('Latency Distribution Comparison (Boxplot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'latency_boxplot.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_inference_time_boxplot(all_results, output_dir):
    """
    Boxplot of inference times.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(all_results.keys())
    data = [all_results[v]['inference_us'] for v in variants]
    
    # Create boxplot
    bplot = ax.boxplot(data, patch_artist=True, labels=variants)
    
    # Color the boxes
    for patch, v in zip(bplot['boxes'], variants):
        patch.set_facecolor(COLORS.get(v, 'gray'))
        patch.set_alpha(0.6)
        
    ax.set_ylabel('Device Inference Time (µs)')
    ax.set_title('Inference Time Distribution (Boxplot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'inference_time_boxplot.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def load_test_data(path, start_idx=0, samples=None):
    """
    Load test data from CSV, Parquet or NPZ.
    """
    print(f"Loading test data from {path}...")
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.npz':
        try:
            data = np.load(path, allow_pickle=True)
            if 'features' in data and 'soc' in data:
                X = data['features']
                y = data['soc']
            elif 'x' in data and 'y' in data:
                X = data['x']
                y = data['y']
            else:
                print(f"ERROR: NPZ file {path} missing required keys.")
                return None, None
        except Exception as e:
            print(f"ERROR loading NPZ: {e}")
            return None, None
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if ext != '.npz':
        # Map columns to standard 6 features
        col_map = {
            'Voltage[V]': ['Voltage[V]', 'voltage', 'V'],
            'Current[A]': ['Current[A]', 'current', 'I'],
            'Temperature[degC]': ['Temperature[degC]', 'Temperature', 'temp', 'T', 'Temperature[°C]'],
            'Q_c': ['Q_c', 'capacity', 'Ah'],
            'dU_dt': ['dU_dt', 'dV_dt', 'dU_dt[V/s]'],
            'dI_dt': ['dI_dt', 'dI_dt[A/s]']
        }
        
        X_cols = []
        for std_col, candidates in col_map.items():
            found = False
            for c in candidates:
                if c in df.columns:
                    X_cols.append(c)
                    found = True
                    break
            if not found:
                print(f"WARNING: Column {std_col} not found in dataset!")
        
        y_col = None
        for c in ['SOC', 'soc', 'y']:
            if c in df.columns:
                y_col = c
                break
        
        t_col = None
        for c in ['Testtime[s]', 'Time[s]', 'time', 'Time']:
            if c in df.columns:
                t_col = c
                break

        X = df[X_cols].values.astype(np.float32)
        y = df[y_col].values.astype(np.float32) if y_col else None
        t = df[t_col].values.astype(np.float32) if t_col else None

    # Slice
    if samples is not None:
        X = X[start_idx:start_idx+samples]
        y = y[start_idx:start_idx+samples] if y is not None else None
        t = t[start_idx:start_idx+samples] if t is not None else None
    else:
        X = X[start_idx:]
        y = y[start_idx:] if y is not None else None
        t = t[start_idx:] if t is not None else None
        
    return X, y, t

def send_command(ser, cmd):
    ser.write((cmd + "\n").encode())
    time.sleep(0.1)
    # Flush input
    while ser.in_waiting:
        print(f"Response: {ser.readline().decode().strip()}")

def run_benchmark(ser, model_name, model_id, X, y, t, output_dir):
    print(f"\n=== Running Benchmark for Model: {model_name} ===")
    
    # Switch Model
    print(f"Switching to model {model_id}...")
    ser.write(f"MODEL:{model_id}\n".encode())
    time.sleep(0.5)
    # Read response
    resp = ser.read_all().decode(errors='ignore')
    if "MOD:OK" not in resp:
        print(f"WARNING: Model switch might have failed. Response: {resp}")
    else:
        print("Model switch confirmed.")

    # Reset State
    ser.write(b"RESET\n")
    time.sleep(0.2)
    ser.read_all() # Clear buffer

    results = []
    latencies = []
    
    pbar = tqdm(total=len(X), desc=f"Inferencing ({model_name})")
    
    for i in range(len(X)):
        feat = X[i]
        # Send features: "V I T Qc dV dI"
        msg = f"{feat[0]:.6f} {feat[1]:.6f} {feat[2]:.6f} {feat[3]:.6f} {feat[4]:.6f} {feat[5]:.6f}\n"
        
        t_start = time.time()
        ser.write(msg.encode())
        
        # Read response
        line_soc = ""
        line_metrics = ""
        
        # We expect 2 lines: METRICS and SOC (order may vary)
        got_soc = False
        got_metrics = False
        
        metrics_data = {}
        pred_soc = 0.0
        
        while not (got_soc and got_metrics):
            try:
                line = ser.readline().decode().strip()
                if not line:
                    break
                
                if line.startswith("SOC:"):
                    m = SOC_RE.search(line)
                    if m:
                        pred_soc = float(m.group(1) + "." + m.group(2))
                        got_soc = True
                elif line.startswith("METRICS:"):
                    m = METRICS_RE.search(line)
                    if m:
                        metrics_data = {
                            'cycles': int(m.group(1)),
                            'us': float(m.group(2)),
                            'e_uj': float(m.group(3))
                        }
                        got_metrics = True
                elif line.startswith("ERR:"):
                    print(f"Firmware Error: {line}")
                    break
            except Exception as e:
                print(f"Serial Error: {e}")
                break
        
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000.0
        latencies.append(latency_ms)
        
        res = {
            'sample_idx': i,
            'pred_soc': pred_soc,
            'latency_ms': latency_ms,
            'cycles': metrics_data.get('cycles', 0),
            'inference_us': metrics_data.get('us', 0.0),
            'energy_uj': metrics_data.get('e_uj', 0.0)
        }
        if y is not None:
            res['true_soc'] = y[i]
            res['error'] = pred_soc - y[i]
        
        if t is not None:
            res['test_time'] = t[i]
            
        results.append(res)
        pbar.update(1)
        
    pbar.close()
    
    # Save Results
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"results_{model_name}.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Calculate Metrics
    if y is not None:
        mae = mean_absolute_error(df_res['true_soc'], df_res['pred_soc'])
        rmse = np.sqrt(mean_squared_error(df_res['true_soc'], df_res['pred_soc']))
        print(f"Metrics ({model_name}): MAE={mae:.6f}, RMSE={rmse:.6f}")
        
    print(f"Avg Latency: {np.mean(latencies):.2f} ms")
    print(f"Avg Inference: {df_res['inference_us'].mean():.2f} us")
    
    return df_res

def generate_markdown_table(all_results, output_dir):
    """
    Generates a markdown summary table of the results.
    """
    table_lines = []
    table_lines.append("| Model | MAE | RMSE | Avg Latency (ms) | Avg Inference (µs) | Avg Energy (µJ) |")
    table_lines.append("|---|---|---|---|---|---|")
    
    for v, df in all_results.items():
        mae = df['error'].abs().mean() if 'error' in df.columns else 0.0
        rmse = np.sqrt((df['error']**2).mean()) if 'error' in df.columns else 0.0
        lat = df['latency_ms'].mean()
        inf = df['inference_us'].mean()
        eng = df['energy_uj'].mean()
        
        table_lines.append(f"| {v} | {mae:.6f} | {rmse:.6f} | {lat:.2f} | {inf:.2f} | {eng:.2f} |")
        
    md_content = "\n".join(table_lines)
    print("\nSummary Table:")
    print(md_content)
    
    with open(os.path.join(output_dir, 'summary_table.md'), 'w') as f:
        f.write(md_content)

def generate_plots(all_results, y, t, out_dir, skip_initial=100):
    """
    Helper function to generate all plots from results.
    """
    print(f"\nGenerating Comparison Plots (skipping first {skip_initial} samples)...")
    
    # 1. Bar Charts
    plot_bar_comparison(all_results, 'energy_uj', 'Average Energy Consumption per Inference', 'Energy (µJ)', 'comparison_energy.png', out_dir)
    plot_bar_comparison(all_results, 'inference_us', 'Average Inference Time (Device)', 'Time (µs)', 'comparison_inference_time_bar.png', out_dir)
    plot_bar_comparison(all_results, 'latency_ms', 'Average Latency (Host)', 'Time (ms)', 'comparison_latency_bar.png', out_dir)
    
    # 2. Predictions
    plot_predictions_comparison(all_results, y, t, out_dir, skip_initial=skip_initial)
    
    # 3. Stability & Distributions
    plot_latency_stability_combined(all_results, out_dir)
    plot_latency_histogram_combined(all_results, out_dir)
    plot_inference_time_stability_combined(all_results, out_dir)
    plot_inference_time_dist_combined(all_results, out_dir)
    
    # 4. Boxplots
    plot_latency_boxplot(all_results, out_dir)
    plot_inference_time_boxplot(all_results, out_dir)
    
    # 5. Accuracy Zoomed
    if y is not None:
        plot_mae_zoomed(all_results, out_dir)
        
    # 6. Summary Table
    generate_markdown_table(all_results, out_dir)

def load_results_from_dir(input_dir):
    """
    Loads results from CSV files in a directory.
    """
    all_results = {}
    variants = ['Base', 'Pruned', 'Quantized']
    
    y_true = None
    t_true = None
    
    for v in variants:
        csv_path = os.path.join(input_dir, f"results_{v}.csv")
        if os.path.exists(csv_path):
            print(f"Loading {csv_path}...")
            df = pd.read_csv(csv_path)
            all_results[v] = df
            
            # Extract ground truth from the first available file
            if y_true is None and 'true_soc' in df.columns:
                y_true = df['true_soc'].values
            if t_true is None and 'test_time' in df.columns:
                t_true = df['test_time'].values
        else:
            print(f"Warning: {csv_path} not found.")
            
    return all_results, y_true, t_true

def main():
    parser = argparse.ArgumentParser(description="STM32 Combined SOC Benchmark")
    parser.add_argument("--mode", choices=['benchmark', 'plot'], default='benchmark', help="Mode: benchmark (run on hardware) or plot (regenerate plots from CSV)")
    parser.add_argument("--port", help="Serial port (e.g., COM7) - Required for benchmark mode")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to test")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--skip", type=int, default=100, help="Number of initial samples to skip in plots")
    parser.add_argument("--data", default=r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet", help="Path to test data")
    parser.add_argument("--output", default="DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOC/combined_results", help="Output directory (for benchmark mode) or Input directory (for plot mode)")
    
    args = parser.parse_args()
    
    if args.mode == 'plot':
        # Plot Mode
        if not os.path.exists(args.output):
            print(f"Error: Directory {args.output} does not exist.")
            return
            
        all_results, y, t = load_results_from_dir(args.output)
        if not all_results:
            print("No results found to plot.")
            return
            
        generate_plots(all_results, y, t, args.output, skip_initial=args.skip)
        print(f"Plots regenerated in {args.output}")
        
    else:
        # Benchmark Mode
        if not args.port:
            print("Error: --port is required for benchmark mode.")
            return
            
        # Create output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.output, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        
        # Load Data
        X, y, t = load_test_data(args.data, start_idx=args.start_idx, samples=args.samples)
        if X is None:
            return

        # Connect Serial
        try:
            ser = serial.Serial(args.port, args.baud, timeout=2)
            print(f"Connected to {args.port}")
            time.sleep(2) # Wait for reset
        except Exception as e:
            print(f"Failed to connect: {e}")
            return

        # Run Benchmarks
        all_results = {}
        
        for name, mid in MODELS.items():
            df = run_benchmark(ser, name, mid, X, y, t, out_dir)
            all_results[name] = df
            time.sleep(1)

        ser.close()
        
        # Generate Plots
        generate_plots(all_results, y, t, out_dir, skip_initial=args.skip)
        print(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
