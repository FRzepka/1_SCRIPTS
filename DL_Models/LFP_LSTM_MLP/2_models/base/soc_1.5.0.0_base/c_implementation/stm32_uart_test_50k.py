"""
STM32 Hardware Test via UART: Compare 50k samples with PyTorch reference
Uses the WORKING plot_comparison_50k.py as base and adds UART communication
"""
import numpy as np
import torch
import joblib
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import serial
import time
from pathlib import Path

# ==================== UART Config ====================
SERIAL_PORT = "COM7"  # STM32 STLink Virtual COM Port
BAUD_RATE = 115200
TIMEOUT = 0.1  # 100ms timeout (faster!)

NUM_SAMPLES = 50000  # Test with 50000 samples

print("="*80)
print(f"STM32 UART Test: {NUM_SAMPLES} samples")
print("="*80)
print("\n[1/6] Loading configuration...")

# ============================================================================
# Load PyTorch model (COPY from working plot_comparison_50k.py)
# ============================================================================
config_path = Path(__file__).parent.parent.parent.parent / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'
print(f"   Config: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print("   ✓ Config loaded")

print("\n[2/6] Loading PyTorch model...")

class LSTM_MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, mlp_hidden_size, output_size, num_layers=1):
        super(LSTM_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(mlp_hidden_size, output_size),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.mlp(lstm_out[:, -1, :])
        return out, hn, cn

# Load model
model_path = Path(__file__).parent.parent / '1.5.0.0_soc_epoch0001_rmse0.02897.pt'
print(f"   Model: {model_path.name}")
print("   Loading checkpoint...")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print("   Creating model architecture...")
model = LSTM_MLP(
    input_size=len(config['model']['features']),
    hidden_size=config['model']['hidden_size'],
    mlp_hidden_size=config['model']['mlp_hidden'],
    output_size=1,
    num_layers=config['model']['num_layers']
)
print("   Loading weights...")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("   ✓ PyTorch model loaded")

# ============================================================================
# Load test data
# ============================================================================
print("\n[3/6] Loading test data...")

# Load scaler
scaler_path = Path(__file__).parent.parent.parent.parent / '1_training' / '1.5.0.0' / 'outputs' / 'scaler_robust.joblib'
print(f"   Scaler: {scaler_path.name}")
scaler = joblib.load(scaler_path)
print("   ✓ Scaler loaded")

# Load data from MG_Farm directory
data_path = r'C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE'
test_cell = 'df_FE_C07'

# Find parquet file
import os
parquet_file = os.path.join(data_path, f'{test_cell}.parquet')
print(f"   Data file: {test_cell}.parquet")
if not os.path.exists(parquet_file):
    raise FileNotFoundError(f"File not found: {parquet_file}")

print("   Reading parquet file (this may take a moment)...")
df = pd.read_parquet(parquet_file)
print(f"   ✓ Loaded {len(df)} total samples from file")

# Get features (handle encoding)
print("   Extracting features...")
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

print(f"   Features: {feature_cols}")

# Limit to NUM_SAMPLES
X_raw = df[feature_cols].values[:NUM_SAMPLES].astype(np.float32)
y_true = df['SOC'].values[:NUM_SAMPLES].astype(np.float32)  # Ground truth!

print(f"   ✓ Using first {len(X_raw)} samples for test")
print(f"   Ground Truth SOC range: {y_true.min():.4f} to {y_true.max():.4f}")

# ============================================================================
# Get PyTorch predictions (Reference)
# ============================================================================
print("\n[4/6] Running PyTorch predictions (reference)...")
print(f"   Processing {len(X_raw)} samples with PyTorch...")
pytorch_preds = []
X_scaled = scaler.transform(X_raw)
X_tensor = torch.from_numpy(X_scaled).float()

with torch.no_grad():
    h = torch.zeros(1, 1, 64)
    c = torch.zeros(1, 1, 64)
    for i in range(len(X_tensor)):
        x_t = X_tensor[i:i+1].unsqueeze(0)  # [1, 1, 6]
        y_pred, h, c = model(x_t, h, c)
        pytorch_preds.append(y_pred.squeeze().item())
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(X_raw)} samples")

pytorch_preds = np.array(pytorch_preds)
print(f"   ✓ PyTorch done!")
print(f"     SOC range: {pytorch_preds.min():.4f} to {pytorch_preds.max():.4f}")

# ============================================================================
# STM32 UART Communication
# ============================================================================
print(f"\n[5/6] STM32 Hardware Test via UART")
print(f"   Port: {SERIAL_PORT}")
print(f"   Baud: {BAUD_RATE}")
print(f"   Connecting...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print("   ✓ Connected!")
    print("   Waiting for STM32 reset...")
    time.sleep(2)  # Wait for STM32 reset
    
    # Read startup messages
    print("   STM32 startup messages:")
    for _ in range(10):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"      {line}")
        time.sleep(0.1)
    
    print(f"\n   Sending {len(X_raw)} samples to STM32...")
    print("   (Progress updates every 100 samples)\n")
    stm32_preds = []
    errors = 0
    
    start_time = time.time()
    
    for i in range(len(X_raw)):
        # Send raw features (STM32 will scale internally)
        sample = X_raw[i]
        cmd = f"{sample[0]:.6f} {sample[1]:.6f} {sample[2]:.6f} {sample[3]:.6f} {sample[4]:.6f} {sample[5]:.6f}\n"
        
        # DEBUG: Print first 3 samples
        if i < 3:
            print(f"\n   [DEBUG Sample {i}] Sending: {cmd.strip()}")
        
        ser.write(cmd.encode('utf-8'))
        ser.flush()
        
        # Wait for "SOC: X.XXX" response
        soc_value = None
        timeout_start = time.time()
        responses = []
        
        while time.time() - timeout_start < TIMEOUT:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                responses.append(line)
                
                if line.startswith("SOC:"):
                    try:
                        soc_value = float(line.split("SOC:")[1].strip())
                        break
                    except (ValueError, IndexError):
                        pass
        
        # DEBUG: Print first 3 responses
        if i < 3:
            print(f"   [DEBUG Sample {i}] Received: {responses}")
            print(f"   [DEBUG Sample {i}] Parsed SOC: {soc_value}")
        
        if soc_value is not None:
            stm32_preds.append(soc_value)
        else:
            stm32_preds.append(np.nan)
            errors += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(X_raw) - i - 1) / rate
            print(f"   {i+1}/{len(X_raw)} | {rate:.1f} samples/s | ETA: {eta:.0f}s | Errors: {errors}")
    
    elapsed_total = time.time() - start_time
    print(f"\n   ✓ STM32 communication completed!")
    print(f"     Time: {elapsed_total:.1f}s")
    print(f"     Rate: {len(X_raw)/elapsed_total:.1f} samples/s")
    print(f"     Errors: {errors}/{len(X_raw)}")
    
    print("   Closing serial port...")
    ser.close()
    print("   ✓ Port closed")

except serial.SerialException as e:
    print(f"\n❌ ERROR: Could not open {SERIAL_PORT}")
    print(f"   {e}")
    print("\n   Check:")
    print("   1. STM32 connected via USB")
    print("   2. Correct COM port (Device Manager)")
    print("   3. No other program using the port")
    exit(1)

stm32_preds = np.array(stm32_preds)

# ============================================================================
# Analysis
# ============================================================================
print("\n[6/6] Analyzing results...")
print("="*80)

# Remove failed samples
valid_mask = ~np.isnan(stm32_preds)
n_valid = np.sum(valid_mask)
n_failed = len(stm32_preds) - n_valid

print(f"\nValid samples: {n_valid}/{len(stm32_preds)} ({100*n_valid/len(stm32_preds):.1f}%)")

if n_valid < 10:
    print("❌ Too few valid samples!")
    exit(1)

pytorch_valid = pytorch_preds[valid_mask]
stm32_valid = stm32_preds[valid_mask]
y_true_valid = y_true[valid_mask]

# Calculate differences
diff_pytorch_stm32 = np.abs(pytorch_valid - stm32_valid)
diff_pytorch_true = np.abs(pytorch_valid - y_true_valid)
diff_stm32_true = np.abs(stm32_valid - y_true_valid)

print(f"\nGround Truth: min={y_true_valid.min():.6f}, max={y_true_valid.max():.6f}")
print(f"PyTorch:      min={pytorch_valid.min():.6f}, max={pytorch_valid.max():.6f}")
print(f"STM32:        min={stm32_valid.min():.6f}, max={stm32_valid.max():.6f}")

print(f"\n📊 PyTorch vs Ground Truth:")
print(f"  MAE:    {diff_pytorch_true.mean():.6f}")
print(f"  Max:    {diff_pytorch_true.max():.6f}")

print(f"\n📊 STM32 vs Ground Truth:")
print(f"  MAE:    {diff_stm32_true.mean():.6f}")
print(f"  Max:    {diff_stm32_true.max():.6f}")

print(f"\n📊 STM32 vs PyTorch (Implementation Difference):")
print(f"  Mean:   {diff_pytorch_stm32.mean():.6e}")
print(f"  Median: {np.median(diff_pytorch_stm32):.6e}")
print(f"  Max:    {diff_pytorch_stm32.max():.6e}")

# Tolerance counts (STM32 vs PyTorch)
tol_001 = np.sum(diff_pytorch_stm32 < 0.001)
tol_0001 = np.sum(diff_pytorch_stm32 < 0.0001)
tol_00001 = np.sum(diff_pytorch_stm32 < 0.00001)

print(f"\nSTM32 vs PyTorch - Samples within tolerance:")
print(f"  < 0.001:   {tol_001:5d} ({100*tol_001/n_valid:5.1f}%)")
print(f"  < 0.0001:  {tol_0001:5d} ({100*tol_0001/n_valid:5.1f}%)")
print(f"  < 0.00001: {tol_00001:5d} ({100*tol_00001/n_valid:5.1f}%)")

# ============================================================================
# Plotting
# ============================================================================
print("\n📊 Generating comparison plots...")
print("   Creating 3 subplots...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Predictions vs Ground Truth
ax1 = axes[0]
indices = np.arange(n_valid)
ax1.plot(indices, y_true_valid, 'k-', linewidth=1.0, alpha=0.8, label='Ground Truth', zorder=3)
ax1.plot(indices, pytorch_valid, 'b-', linewidth=0.7, alpha=0.7, label='PyTorch (CPU)')
ax1.plot(indices, stm32_valid, 'r--', linewidth=0.7, alpha=0.7, label='STM32 (Hardware)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('SOC')
ax1.set_title(f'STM32 Hardware Test: {n_valid} Valid Samples')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Errors vs Ground Truth
ax2 = axes[1]
ax2.plot(indices, diff_pytorch_true, 'b-', linewidth=0.5, alpha=0.7, label='PyTorch Error')
ax2.plot(indices, diff_stm32_true, 'r-', linewidth=0.5, alpha=0.7, label='STM32 Error')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Absolute Error')
ax2.set_title(f'Prediction Errors (vs Ground Truth)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: STM32 vs PyTorch difference
ax3 = axes[2]
ax3.plot(indices, diff_pytorch_stm32, 'g-', linewidth=0.5, alpha=0.7)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('|PyTorch - STM32|')
ax3.set_title(f'Implementation Difference: Mean={diff_pytorch_stm32.mean():.6e}, Max={diff_pytorch_stm32.max():.6e}')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, which='both')
ax3.axhline(y=0.001, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='0.001')
ax3.axhline(y=0.0001, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.0001')
ax3.legend()

plt.tight_layout()
output_path = Path(__file__).parent / 'stm32_hardware_comparison_1k.png'
print(f"   Saving plot to: {output_path.name}")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Plot saved!")
print(f"   Full path: {output_path}")

print("\n   Opening plot window...")
plt.show()

print("\n" + "="*80)
print("✅ TEST COMPLETED SUCCESSFULLY!")
print("="*80)
