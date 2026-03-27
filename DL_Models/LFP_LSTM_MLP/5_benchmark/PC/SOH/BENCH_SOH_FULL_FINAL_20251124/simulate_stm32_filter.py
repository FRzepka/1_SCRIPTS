import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Detect base directory for Windows / Linux use
if os.name == "nt":
    BASE_DIR = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts"
else:
    BASE_DIR = "/home/florianr/MG_Farm/1_Scripts"

DATA_PATH = os.path.join(
    BASE_DIR,
    "DL_Models",
    "LFP_LSTM_MLP",
    "5_benchmark",
    "PC",
    "SOH",
    "BENCH_SOH_FULL_FINAL_20251124",
    "benchmark_results.npz",
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "DL_Models",
    "LFP_LSTM_MLP",
    "5_benchmark",
    "PC",
    "SOH",
    "BENCH_SOH_FULL_FINAL_20251124",
)
PARQUET_PATH = os.path.join(
    BASE_DIR,
    "3_Projekte",
    "MG_Farm",
    "5_Data",
    "01_LFP",
    "00_Data",
    "Versuch_18650_standart",
    "MGFarm_18650_FE",
    "df_FE_C07.parquet",
)

# New colour scheme for the paper (matching Pipeline.png, etwas kräftiger):
# Base = warmes Gelb/Orange, Pruned = kräftiges Blau, Quantized = sattes Grün.
BASE_HEX = "#2ca02c"
PRUNED_HEX = "#d62728"
QUANT_HEX = "#1f77b4"

def stm32_filter_simulation(data, alpha, max_drop):
    """
    Simulates a real-time feasible filter for STM32.
    Combines EMA (Low Pass) and Slew Rate Limiter.
    """
    n = len(data)
    output = np.zeros(n)
    
    # State variables (would be static float in C)
    current_ema = 1.0 # Start at 100% SOH
    last_output = 1.0
    
    # Pre-calculate constants
    one_minus_alpha = 1.0 - alpha
    
    print(f"Simulating STM32 Filter (Alpha={alpha}, MaxDrop={max_drop})...")
    
    for i in range(n):
        raw_input = data[i]
        
        # 1. EMA Filter (Low Pass)
        # y[n] = alpha * x[n] + (1-alpha) * y[n-1]
        current_ema = (alpha * raw_input) + (one_minus_alpha * current_ema)
        
        # 2. Drop Limiter (Slew Rate Limiter)
        # We apply this to the EMA output to ensure the final result respects the limit
        proposed_value = current_ema
        
        # Limit Drop
        if proposed_value < last_output - max_drop:
            proposed_value = last_output - max_drop
            # Optional: Feedback the limited value to EMA? 
            # Usually better to keep EMA running freely or it gets stuck.
            # But for SOH, maybe we want to pull EMA back? Let's keep it simple first.
            
        # Limit Rise? (Optional, but good for SOH to prevent noise jumps up)
        # Let's say we allow rising, but maybe also limited?
        # For now, just the drop limit as requested.
        
        output[i] = proposed_value
        last_output = proposed_value
        
    return output

def main():
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    y_gt = data['y_gt']
    
    # Load all models
    c_base = data['C_Base']
    c_pruned = data['C_Pruned']
    c_quant = data['C_Quant']

    # Parameters for STM32
    # Alpha: Lower = Smoother. 
    # Equivalent N for EMA is approx 2/alpha. 
    # If we want N=3,000,000 -> alpha = 2/3,000,000 ~= 6.6e-7
    alpha = 1.0e-6 
    
    # Max Drop: Same as Python script
    max_drop = 2e-8

    print("Processing Base Model...")
    stm32_base = stm32_filter_simulation(c_base, alpha, max_drop)
    
    print("Processing Pruned Model...")
    stm32_pruned = stm32_filter_simulation(c_pruned, alpha, max_drop)
    
    print("Processing Quantized Model...")
    stm32_quant = stm32_filter_simulation(c_quant, alpha, max_drop)
    
    # Calculate MAE
    mae_base = np.mean(np.abs(stm32_base - y_gt))
    mae_pruned = np.mean(np.abs(stm32_pruned - y_gt))
    mae_quant = np.mean(np.abs(stm32_quant - y_gt))
    
    print(f"MAE STM32 Base: {mae_base:.6f}")
    print(f"MAE STM32 Pruned: {mae_pruned:.6f}")
    print(f"MAE STM32 Quant: {mae_quant:.6f}")

    # Build time axis (seconds -> days)
    n_samples = len(y_gt)
    time_axis = None
    if os.path.exists(PARQUET_PATH):
        try:
            df = pd.read_parquet(PARQUET_PATH, columns=["Testtime[s]"])
            if len(df) >= n_samples:
                time_axis = df["Testtime[s]"].values[:n_samples]
        except Exception as e:
            print(f"Warning: could not load parquet for time axis: {e}")

    if time_axis is None:
        time_axis = np.arange(n_samples, dtype=float)

    step = 1000
    idx = np.arange(0, n_samples, step)
    t_days = time_axis[idx] / (24.0 * 3600.0)

    SCALE = 100.0

    plt.figure(figsize=(15, 6))
    plt.plot(t_days, y_gt[idx] * SCALE, label='Ground Truth', color='black', linewidth=2, alpha=0.6)
    plt.plot(t_days, stm32_base[idx] * SCALE, label=f'Base Filtered (MAE={mae_base * SCALE:.2f} %)', color=BASE_HEX, linewidth=2)
    plt.plot(t_days, stm32_pruned[idx] * SCALE, label=f'Pruned Filtered (MAE={mae_pruned * SCALE:.2f} %)', color=PRUNED_HEX, linewidth=2)
    plt.plot(t_days, stm32_quant[idx] * SCALE, label=f'Quant Filtered (MAE={mae_quant * SCALE:.2f} %)', color=QUANT_HEX, linewidth=2)

    plt.xlabel("Time [d]", fontsize=14)
    plt.ylabel("SOH [%]", fontsize=14)
    plt.title(f"STM32-compatible SOH filter (streaming, all models)\n"
              f"EMA $\\alpha={alpha}$, max drop={max_drop}", fontsize=16)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "stm32_filter_simulation_all.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
