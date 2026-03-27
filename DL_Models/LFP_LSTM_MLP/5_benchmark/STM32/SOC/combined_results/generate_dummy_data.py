import numpy as np
import os

# Output path
output_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_LSTM_MLP\5_benchmark\soc_streaming_base_quant_pruned_data.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Scaler params (from firmware/host script)
SCALER_CENTER = np.array([3.3606, 0.6543, 27.4000, -0.5110, 0.0, 0.0], dtype=np.float32)
SCALER_SCALE = np.array([0.2009, 2.6982, 1.1000, 0.5354, 1.0, 1.0], dtype=np.float32)

# Generate 5000 samples
N = 5000
# Generate random data in "standard normal" distribution (scaled space)
features_scaled = np.random.normal(loc=0.0, scale=1.0, size=(N, 6)).astype(np.float32)

# Unscale to get "raw" values (Physical units: V, A, degC, Ah, V/s, A/s)
# This assumes the firmware/model expects RAW inputs and scales them, OR the host script sends raw inputs.
# If the model expects scaled inputs and there is no scaler in firmware, we should send scaled inputs.
# However, usually test data is stored in raw format.
features_raw = features_scaled * SCALER_SCALE + SCALER_CENTER

# SOC (Ground Truth) - just random for dummy
soc = np.random.uniform(0, 100, size=(N,)).astype(np.float32)

np.savez(output_path, features=features_raw, soc=soc)
print(f"Generated dummy data at {output_path}")
