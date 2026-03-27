"""
STM32 Hybrid INT8 Model Test: 1000 samples
Ground Truth vs STM32 Hardware predictions only
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import serial
import time
import argparse
from pathlib import Path

# ==================== UART Config ====================
SERIAL_PORT = "COM7"  # Adjust if needed!
BAUD_RATE = 115200
TIMEOUT = 0.1

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=1000, help='Number of samples to test')
args = parser.parse_args()

NUM_SAMPLES = args.samples

print("="*80)
print(f"STM32 Hybrid INT8 Model Test: {NUM_SAMPLES} samples")
print("="*80)

# ============================================================================
# Load test data
# ============================================================================
print("\n[1/3] Loading test data...")

# Load scaler
scaler_path = Path(r'C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\outputs\scaler_robust.joblib')
print(f"   Scaler: {scaler_path.name}")
scaler = joblib.load(scaler_path)
print("   ✓ Scaler loaded")

# Load data
data_path = r'C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE'
test_cell = 'df_FE_C07'
parquet_file = f'{data_path}\\{test_cell}.parquet'

print(f"   Loading: {test_cell}.parquet")
df = pd.read_parquet(parquet_file)
print(f"   ✓ Loaded {len(df)} total samples")

# Extract features
features = ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'Q_c', 'dU_dt[V/s]', 'dI_dt[A/s]']
feature_cols = []
for feat in features:
    if feat in df.columns:
        feature_cols.append(feat)
    else:
        for col in df.columns:
            if 'Temperature' in feat and 'Temperature' in col:
                feature_cols.append(col)
                break

# Get first 1000 samples
X_raw = df[feature_cols].values[:NUM_SAMPLES].astype(np.float32)
y_true = df['SOC'].values[:NUM_SAMPLES].astype(np.float32)

print(f"   ✓ Using first {NUM_SAMPLES} samples")
print(f"   Ground Truth SOC range: {y_true.min():.4f} to {y_true.max():.4f}")

# ============================================================================
# STM32 UART Communication
# ============================================================================
print(f"\n[2/3] STM32 Hardware Test via UART")
print(f"   Port: {SERIAL_PORT}")
print(f"   Baud: {BAUD_RATE}")
print(f"   Connecting...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print("   ✓ Connected!")
    print("   Waiting for STM32 reset...")
    time.sleep(2)
    
    # Clear startup messages
    while ser.in_waiting > 0:
        ser.readline()
    
    print(f"\n   Sending {NUM_SAMPLES} samples to STM32...")
    print("   (Progress updates every 100 samples)\n")
    
    stm32_preds = []
    errors = 0
    start_time = time.time()
    
    for i in range(NUM_SAMPLES):
        # Send raw features (STM32 scales internally)
        sample = X_raw[i]
        cmd = f"{sample[0]:.6f} {sample[1]:.6f} {sample[2]:.6f} {sample[3]:.6f} {sample[4]:.6f} {sample[5]:.6f}\n"
        
        ser.write(cmd.encode('utf-8'))
        ser.flush()
        
        # Wait for "SOC: X.XXX" response
        soc_value = None
        timeout_start = time.time()
        
        while time.time() - timeout_start < TIMEOUT:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("SOC:"):
                    try:
                        soc_value = float(line.split("SOC:")[1].strip())
                        break
                    except (ValueError, IndexError):
                        pass
        
        if soc_value is not None:
            stm32_preds.append(soc_value)
        else:
            stm32_preds.append(np.nan)
            errors += 1
        
        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (NUM_SAMPLES - i - 1) / rate
            print(f"   {i+1}/{NUM_SAMPLES} | {rate:.1f} samples/s | ETA: {eta:.0f}s | Errors: {errors}")
    
    elapsed_total = time.time() - start_time
    print(f"\n   ✓ STM32 communication completed!")
    print(f"     Time: {elapsed_total:.1f}s")
    print(f"     Rate: {NUM_SAMPLES/elapsed_total:.1f} samples/s")
    print(f"     Errors: {errors}/{NUM_SAMPLES}")
    
    ser.close()
    print("   ✓ Port closed")

except serial.SerialException as e:
    print(f"\n❌ ERROR: Could not open {SERIAL_PORT}")
    print(f"   {e}")
    print("\n   Check:")
    print("   1. STM32 connected via USB")
    print("   2. Correct COM port (check Device Manager)")
    print("   3. No other program using the port")
    exit(1)

stm32_preds = np.array(stm32_preds)

# ============================================================================
# Analysis
# ============================================================================
print("\n[3/3] Analyzing results...")
print("="*80)

# Remove failed samples
valid_mask = ~np.isnan(stm32_preds)
n_valid = np.sum(valid_mask)

print(f"\nValid samples: {n_valid}/{NUM_SAMPLES} ({100*n_valid/NUM_SAMPLES:.1f}%)")

if n_valid < 10:
    print("❌ Too few valid samples!")
    exit(1)

stm32_valid = stm32_preds[valid_mask]
y_true_valid = y_true[valid_mask]

# Calculate error
mae = np.mean(np.abs(stm32_valid - y_true_valid))
max_error = np.max(np.abs(stm32_valid - y_true_valid))
rmse = np.sqrt(np.mean((stm32_valid - y_true_valid)**2))

print(f"\nGround Truth: min={y_true_valid.min():.6f}, max={y_true_valid.max():.6f}")
print(f"STM32:        min={stm32_valid.min():.6f}, max={stm32_valid.max():.6f}")

print(f"\n📊 STM32 Hybrid INT8 vs Ground Truth:")
print(f"  MAE:      {mae:.6f}")
print(f"  RMSE:     {rmse:.6f}")
print(f"  Max Err:  {max_error:.6f}")

# ============================================================================
# Plotting
# ============================================================================
print("\n📊 Generating plot...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Predictions vs Ground Truth
ax1 = axes[0]
indices = np.arange(n_valid)
ax1.plot(indices, y_true_valid, 'k-', linewidth=1.5, alpha=0.8, label='Ground Truth', zorder=3)
ax1.plot(indices, stm32_valid, 'r-', linewidth=1.0, alpha=0.7, label='STM32 Hybrid INT8')
ax1.set_xlabel('Sample Index', fontsize=11)
ax1.set_ylabel('SOC', fontsize=11)
ax1.set_title(f'STM32 Hybrid INT8 Model: {n_valid} Samples | MAE={mae:.6f}', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Absolute Error
ax2 = axes[1]
error = np.abs(stm32_valid - y_true_valid)
ax2.plot(indices, error, 'r-', linewidth=0.8, alpha=0.7)
ax2.fill_between(indices, 0, error, color='red', alpha=0.2)
ax2.set_xlabel('Sample Index', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
ax2.set_title(f'Prediction Error | Mean={mae:.6f}, Max={max_error:.6f}', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=mae, color='orange', linestyle='--', linewidth=1.5, label=f'Mean={mae:.6f}')
ax2.legend(fontsize=10)

plt.tight_layout()

output_path = Path(__file__).parent / f'stm32_hybrid_int8_test_{NUM_SAMPLES}.png'
print(f"   Saving plot: {output_path.name}")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Plot saved!")

print("\n   Opening plot window...")
plt.show()

print("\n" + "="*80)
print("✅ TEST COMPLETED!")
print("="*80)
print(f"\nResults Summary:")
print(f"  Samples:   {n_valid}/{NUM_SAMPLES}")
print(f"  MAE:       {mae:.6f}")
print(f"  RMSE:      {rmse:.6f}")
print(f"  Max Error: {max_error:.6f}")
print(f"  Rate:      {NUM_SAMPLES/elapsed_total:.1f} samples/s")
print("="*80)
