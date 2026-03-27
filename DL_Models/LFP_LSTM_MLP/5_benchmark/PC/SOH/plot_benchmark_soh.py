#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH")

def main():
    npz_path = OUTPUT_DIR / "benchmark_results.npz"
    if not npz_path.exists():
        print("Results not found.")
        return
        
    data = np.load(npz_path)
    y_gt = data['y_gt']
    
    # Downsample for plotting
    N = len(y_gt)
    step = max(1, N // 100000)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_gt[::step], label='Ground Truth', color='black', alpha=0.5, linewidth=1)
    
    colors = {'Py_Base': 'blue', 'Py_Pruned': 'cyan', 'C_Base': 'red', 'C_Pruned': 'orange', 'C_Quant': 'green'}
    styles = {'Py_Base': '-', 'Py_Pruned': '--', 'C_Base': ':', 'C_Pruned': '-.', 'C_Quant': '-'}
    
    for k in data.files:
        if k == 'y_gt': continue
        plt.plot(data[k][::step], label=k, color=colors.get(k, 'gray'), linestyle=styles.get(k, '-'), alpha=0.8)
        
    plt.title("SOH Benchmark: Base vs Pruned vs Quantized (Python vs C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_overlay.png", dpi=150)
    
    # Zoom
    zoom_n = min(5000, N)
    plt.figure(figsize=(12, 6))
    plt.plot(y_gt[:zoom_n], label='Ground Truth', color='black', alpha=0.5)
    for k in data.files:
        if k == 'y_gt': continue
        plt.plot(data[k][:zoom_n], label=k, color=colors.get(k, 'gray'), linestyle=styles.get(k, '-'))
    plt.title(f"SOH Benchmark (First {zoom_n} steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_zoom.png", dpi=150)
    
    # Metrics Bar Chart
    with open(OUTPUT_DIR / "metrics.json", 'r') as f:
        metrics = json.load(f)
        
    names = list(metrics.keys())
    rmses = [metrics[n]['RMSE'] for n in names]
    maes = [metrics[n]['MAE'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rmses, width, label='RMSE')
    rects2 = ax.bar(x + width/2, maes, width, label='MAE')
    
    ax.set_ylabel('Error')
    ax.set_title('Model Accuracy Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.5f', rotation=90)
    ax.bar_label(rects2, padding=3, fmt='%.5f', rotation=90)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_metrics.png", dpi=150)
    
    print("Plots saved.")

if __name__ == '__main__':
    main()
