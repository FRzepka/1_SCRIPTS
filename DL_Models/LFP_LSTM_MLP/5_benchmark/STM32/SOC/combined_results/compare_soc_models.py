#!/usr/bin/env python3
"""
STM32 SOC Model Comparison Script
- Scans for the latest benchmark results for Base, Quantized, and Pruned models.
- Generates comparative plots (Inference Time, Energy, Accuracy, Latency).
- Outputs a combined Markdown summary table.

Usage:
  python STM32/benchmark/compare_soc_models.py
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Configuration
VARIANTS = ['base', 'quantized', 'pruned']
# Path to results relative to this script
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'DL_Models', 'LFP_LSTM_MLP', '5_benchmark', 'STM32', 'SOC')
OUTPUT_DIR = os.path.join(RESULTS_ROOT, 'comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

COLORS = {
    'base': '#2ca02c',      # Green
    'quantized': '#1f77b4', # Blue
    'pruned': '#d62728'     # Red
}

LABELS = {
    'base': 'Base (FP32)',
    'quantized': 'Quantized (INT8)',
    'pruned': 'Pruned (FP32)'
}

def find_latest_result(variant):
    variant_dir = os.path.join(RESULTS_ROOT, variant)
    if not os.path.exists(variant_dir):
        return None
    
    # Find all results_* folders
    result_dirs = glob.glob(os.path.join(variant_dir, 'results_*'))
    if not result_dirs:
        return None
    
    # Sort by name (timestamp) descending
    result_dirs.sort(reverse=True)
    return result_dirs[0]

def load_data():
    data = {}
    for v in VARIANTS:
        path = find_latest_result(v)
        if path:
            print(f"Found latest {v} result: {os.path.basename(path)}")
            summary_path = os.path.join(path, 'summary.json')
            csv_path = os.path.join(path, 'stm32_soc_bench.csv')
            
            if os.path.exists(summary_path) and os.path.exists(csv_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                df = pd.read_csv(csv_path)
                data[v] = {'summary': summary, 'df': df, 'path': path}
            else:
                print(f"Warning: Missing files in {path}")
        else:
            print(f"Warning: No results found for {v}")
    return data

def plot_bar_comparison(data, metric_key, title, ylabel, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = [v for v in VARIANTS if v in data]
    values = []
    colors = []
    
    for v in variants:
        val = data[v]['summary']['stats'].get(metric_key)
        if val is None:
            val = 0
        values.append(val * scale)
        colors.append(COLORS[v])
        
    bars = ax.bar([LABELS[v] for v in variants], values, color=colors, alpha=0.8, edgecolor='black')
    
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
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_predictions_comparison(data):
    """
    Plots Ground Truth vs Predictions for all models.
    Includes a zoomed-in subplot to show details.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    variants = [v for v in VARIANTS if v in data]
    
    # Get Ground Truth from the first available dataset (assuming all use same test data)
    first_v = variants[0]
    df_ref = data[first_v]['df']
    
    if 'soc_true' not in df_ref.columns:
        print("No Ground Truth data found in CSV.")
        return

    # Plot 1: Full Sequence
    ax1.plot(df_ref['idx'], df_ref['soc_true'], label='Ground Truth', color='black', linewidth=1.5, alpha=0.6)
    
    for v in variants:
        df = data[v]['df']
        if 'soc_pred' in df.columns:
            ax1.plot(df['idx'], df['soc_pred'], label=LABELS[v], color=COLORS[v], linewidth=1, alpha=0.7, linestyle='--')

    ax1.set_title('SOC Prediction Comparison (Full Sequence)')
    ax1.set_ylabel('SOC')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(df_ref))

    # Plot 2: Zoomed In (e.g., samples 1000 to 1200 or a dynamic region)
    # Try to find a dynamic region (high variance)
    start_idx = 1000
    end_idx = 1300
    if len(df_ref) > 1300:
        pass # use default
    else:
        start_idx = 0
        end_idx = min(300, len(df_ref))

    ax2.plot(df_ref['idx'], df_ref['soc_true'], label='Ground Truth', color='black', linewidth=2, alpha=0.5)
    
    for v in variants:
        df = data[v]['df']
        if 'soc_pred' in df.columns:
            ax2.plot(df['idx'], df['soc_pred'], label=LABELS[v], color=COLORS[v], linewidth=1.5, linestyle='--')

    ax2.set_title(f'Zoomed Detail (Samples {start_idx}-{end_idx})')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('SOC')
    ax2.set_xlim(start_idx, end_idx)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_predictions_combined.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_latency_stability_combined(data):
    """
    Plots latency time series for all models in one plot.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    variants = [v for v in VARIANTS if v in data]
    
    for v in variants:
        df = data[v]['df']
        if 'rtt_ms' in df.columns:
            # Use rolling mean to smooth it slightly for readability if dense
            ax.plot(df['idx'], df['rtt_ms'], label=LABELS[v], color=COLORS[v], linewidth=0.8, alpha=0.7)
            
    ax.set_ylabel('Host Latency (ms)')
    ax.set_xlabel('Sample Index')
    ax.set_title('Inference Latency Stability Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_latency_stability.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_latency_histogram_combined(data):
    """
    Overlaid histogram of latencies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = [v for v in VARIANTS if v in data]
    
    for v in variants:
        df = data[v]['df']
        if 'rtt_ms' in df.columns:
            ax.hist(df['rtt_ms'], bins=100, range=(0, 20), 
                    label=LABELS[v], color=COLORS[v], 
                    alpha=0.4, density=True, edgecolor=COLORS[v], histtype='stepfilled')
            # Add a step line on top for clarity
            ax.hist(df['rtt_ms'], bins=100, range=(0, 20), 
                    color=COLORS[v], density=True, histtype='step', linewidth=1.5)

    ax.set_xlabel('Host Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Latency Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_latency_hist_combined.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_mae_zoomed(data):
    """
    Bar chart for MAE, but zoomed in to show differences.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = [v for v in VARIANTS if v in data]
    values = []
    colors = []
    
    for v in variants:
        val = data[v]['summary']['stats'].get('mae', 0)
        values.append(val)
        colors.append(COLORS[v])
        
    bars = ax.bar([LABELS[v] for v in variants], values, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Zoom Y-axis
    if values:
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 2.0 if max_val != min_val else 0.1
        # If values are very close (e.g. 0.198 vs 0.199), zoom in
        ax.set_ylim(max(0, min_val - margin), max_val + margin)
                
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Model Accuracy Comparison (Zoomed)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_mae_zoomed.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_inference_time_stability_combined(data):
    """
    Plots device inference time (us) time series for all models.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    variants = [v for v in VARIANTS if v in data]
    
    for v in variants:
        df = data[v]['df']
        if 'metric_us' in df.columns:
            # Filter out None/NaN
            valid_us = df['metric_us'].dropna()
            if not valid_us.empty:
                ax.plot(df.loc[valid_us.index, 'idx'], valid_us, label=LABELS[v], color=COLORS[v], linewidth=0.8, alpha=0.8)
            
    ax.set_ylabel('Device Inference Time (µs)')
    ax.set_xlabel('Sample Index')
    ax.set_title('On-Device Computation Time Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_inference_time_stability.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def plot_inference_time_dist_combined(data):
    """
    Overlaid histogram of device inference times.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = [v for v in VARIANTS if v in data]
    
    for v in variants:
        df = data[v]['df']
        if 'metric_us' in df.columns:
            valid_us = df['metric_us'].dropna()
            if not valid_us.empty:
                ax.hist(valid_us, bins=50, 
                        label=LABELS[v], color=COLORS[v], 
                        alpha=0.5, density=True)

    ax.set_xlabel('Device Inference Time (µs)')
    ax.set_ylabel('Density')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_inference_time_dist.png')
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()

def generate_markdown_table(data):
    md = "# STM32 SOC Model Comparison\n\n"
    md += "| Metric | Base (FP32) | Quantized (INT8) | Pruned (FP32) | Unit |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    metrics = [
        ('Inference Time', 'metrics_us_mean', '{:.2f}', 'µs'),
        ('Energy (Est.)', 'metrics_E_uJ_mean', '{:.2f}', 'µJ'),
        ('MAE', 'mae', '{:.4f}', '-'),
        ('RMSE', 'rmse', '{:.4f}', '-'),
        ('Throughput', 'throughput_samples_per_s', '{:.1f}', 'Hz'),
        ('Flash (.text)', 'build.elf_size_bytes', '{}', 'Bytes'), # Note: This is ELF size, approx
        ('RAM (.data+.bss)', 'ram_usage', '{}', 'Bytes')
    ]
    
    for name, key, fmt, unit in metrics:
        row = f"| **{name}** |"
        for v in VARIANTS:
            val = 'N/A'
            if v in data:
                stats = data[v]['summary']['stats']
                
                # Special handling for nested keys or computed values
                if key == 'ram_usage':
                    build = stats.get('build', {})
                    if build:
                        val_num = build.get('data_bytes', 0) + build.get('bss_bytes', 0)
                        val = fmt.format(val_num)
                    else:
                        val = 'N/A'
                elif '.' in key:
                    # Handle build.elf_size_bytes
                    parts = key.split('.')
                    curr = stats
                    for p in parts:
                        curr = curr.get(p, {})
                    if isinstance(curr, (int, float)):
                        val = fmt.format(curr)
                    else:
                        val = 'N/A'
                else:
                    val_num = stats.get(key)
                    if val_num is not None:
                        val = fmt.format(val_num)
            
            row += f" {val} |"
        row += f" {unit} |"
        md += row + "\n"
        
    path = os.path.join(OUTPUT_DIR, 'comparison_summary.md')
    with open(path, 'w') as f:
        f.write(md)
    print(f"Saved summary table to {path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating comparison in {OUTPUT_DIR}...")
    
    data = load_data()
    
    if not data:
        print("No data found to compare.")
        return

    # 1. Inference Time Comparison
    plot_bar_comparison(data, 'metrics_us_mean', 'On-Device Inference Time', 'Time (µs)', 'comparison_inference_time.png')
    
    # 2. Energy Comparison
    plot_bar_comparison(data, 'metrics_E_uJ_mean', 'Estimated Inference Energy', 'Energy (µJ)', 'comparison_energy.png')
    
    # 3. Accuracy Comparison (MAE)
    plot_mae_zoomed(data)
    
    # 4. Predictions Comparison (Combined)
    plot_predictions_comparison(data)
    
    # 5. Latency Stability (Combined)
    plot_latency_stability_combined(data)
    
    # 6. Latency Histogram (Combined)
    plot_latency_histogram_combined(data)

    # 7. Inference Time Stability (Combined)
    plot_inference_time_stability_combined(data)

    # 8. Inference Time Distribution (Combined)
    plot_inference_time_dist_combined(data)
    
    # 9. Markdown Table
    generate_markdown_table(data)
    
    print("Comparison complete.")

if __name__ == '__main__':
    main()
