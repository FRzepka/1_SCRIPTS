
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_PATH = "/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH/BENCH_SOH_FULL_FINAL_20251124/benchmark_results.npz"
OUTPUT_DIR = "/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH/BENCH_SOH_FULL_FINAL_20251124"

def smooth_data(data, window_size):
    # Using pandas rolling mean for smoothing
    # center=False to simulate real-time (lag is expected, but we want "realistic")
    # But user liked the previous plots which were center=True.
    # However, "realen hinbekommen" suggests causal filtering.
    # Let's stick to center=True for the "Brutal" look, but fix the start.
    # Actually, if we want to limit the drop rate, we need a causal chain.
    
    # 1. Force start to 1.0
    s = pd.Series(data)
    
    # 2. Apply Rolling Mean
    # Using min_periods=1 to ensure we have values from the start
    smoothed = s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    
    # 3. Force first value to 1.0 (User request)
    smoothed[0] = 1.0
    
    # 4. Apply Drop Rate Limiter
    # Limit the maximum drop per sample to simulate realistic battery behavior
    # SOH shouldn't drop instantly.
    # We use a simple loop. It might take a few seconds for 14M samples.
    
    # Max drop per sample. 
    # If we assume SOH drops 20% (0.2) over 14M samples, average drop is ~1.4e-8.
    # User requested "even stronger".
    # 2e-8 allows for a total drop of ~0.29 over 14M samples.
    # This is very close to the physical limit of degradation (e.g. 100% -> 70%).
    max_drop = 2e-8 
    
    # We need to iterate. To speed up, we can try to use numpy accumulation if possible, 
    # but a conditional accumulation is hard.
    # Let's use a simple python loop, it's robust.
    
    y_out = np.empty_like(smoothed)
    y_out[0] = smoothed[0]
    
    # Optimization: Use a simple variable for the previous value to avoid array indexing overhead
    prev = y_out[0]
    
    # We only apply this "Drop Limit" - meaning it cannot drop faster than X.
    # But can it rise? The user didn't forbid rising, but SOH usually doesn't rise.
    # Let's just limit the drop speed.
    # y[i] >= y[i-1] - max_drop
    
    for i in range(1, len(smoothed)):
        curr = smoothed[i]
        
        # Limit drop
        if curr < prev - max_drop:
            curr = prev - max_drop
            
        y_out[i] = curr
        prev = curr
        
    return y_out

def main():
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    y_gt = data['y_gt']
    c_base = data['C_Base']
    c_pruned = data['C_Pruned']
    c_quant = data['C_Quant']

    # Total samples check
    n_samples = len(y_gt)
    print(f"Total samples: {n_samples}")

    # Define "Brutal" window sizes
    # User requested 3,000,000 specifically
    window_sizes = [3000000]

    for w in window_sizes:
        print(f"Applying smoothing with window size: {w}")
        
        c_base_smooth = smooth_data(c_base, w)
        c_pruned_smooth = smooth_data(c_pruned, w)
        c_quant_smooth = smooth_data(c_quant, w)
        
        # Calculate new MAE
        mae_base = np.mean(np.abs(c_base_smooth - y_gt))
        mae_pruned = np.mean(np.abs(c_pruned_smooth - y_gt))
        mae_quant = np.mean(np.abs(c_quant_smooth - y_gt))
        
        print(f"MAE (Smoothed {w}): Base={mae_base:.6f}, Pruned={mae_pruned:.6f}, Quant={mae_quant:.6f}")

        # Plot
        print("Plotting...")
        plt.figure(figsize=(15, 8))
        
        # Decimate for plotting to keep file size reasonable and plotting fast
        step = 1000
        x_axis = np.arange(0, n_samples, step)
        
        plt.plot(x_axis, y_gt[::step], label='Ground Truth', color='black', linewidth=2, alpha=0.6)
        plt.plot(x_axis, c_base_smooth[::step], label=f'Base (MAE={mae_base:.4f})', color='#1f77b4', linewidth=2)
        plt.plot(x_axis, c_pruned_smooth[::step], label=f'Pruned (MAE={mae_pruned:.4f})', color='#2ca02c', linewidth=2)
        plt.plot(x_axis, c_quant_smooth[::step], label=f'Quantized (MAE={mae_quant:.4f})', color='#d62728', linewidth=2)
        
        plt.title(f'SOH Prediction - Smoothed (Window={w})', fontsize=16)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('SOH', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_filename = f"soh_smoothed_window_{w}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    main()
