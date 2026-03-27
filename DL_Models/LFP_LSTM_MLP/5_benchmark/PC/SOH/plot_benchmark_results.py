
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH")
RESULTS_PATH = OUTPUT_DIR / "benchmark_results.npz"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

def main():
    if not RESULTS_PATH.exists():
        print("Results not found.")
        return

    data = np.load(RESULTS_PATH)
    y_gt = data['y_gt']
    
    # Load metrics
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot GT
    plt.plot(y_gt, label='Ground Truth', color='black', linewidth=1.5, alpha=0.8)
    
    # Plot Predictions
    colors = {'Py_Base': 'blue', 'C_Base': 'cyan', 
              'Py_Pruned': 'green', 'C_Pruned': 'lime', 
              'C_Quant': 'red'}
    
    for name in data.files:
        if name == 'y_gt': continue
        if name not in colors: continue
        
        preds = data[name]
        mae = metrics[name]['MAE']
        plt.plot(preds, label=f'{name} (MAE={mae:.4f})', color=colors[name], linewidth=1, alpha=0.7)

    plt.title("SOH Benchmark: Python vs C Implementation")
    plt.xlabel("Sample")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = OUTPUT_DIR / "benchmark_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
