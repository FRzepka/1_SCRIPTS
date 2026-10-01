import serial
import numpy as np
import pandas as pd
import struct
import time
import joblib
from sklearn.preprocessing import RobustScaler
import pyarrow.parquet as pq

# ====== USER SETTINGS ======
PORT = "/dev/ttyACM0"
BAUD = 115200
PRINT_EVERY = 20
FRAME_HDR = b"BATT"
SEQ_LEN = 32
STEP = 16                # stride between sequence starts (= SEQLEN => non-overlapping)
READ_TIMEOUT = 10.0

DATA_PATH = "df_FE_C13.parquet"
SCALER_PATH = "scaler_robust.joblib"

# ====== FEATURES ======
SCALER_FEATURES = [
    "Testtime[s]",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "EFC",
    "Q_c",
]
FULL_FEATURES = SCALER_FEATURES + ["Capacity[Ah]"]

##################################################################
# Helper: read exact bytes with timeout
##################################################################
def read_exact(ser, n_bytes, timeout=READ_TIMEOUT):
    start = time.time()
    buf = b""
    while len(buf) < n_bytes:
        if ser.in_waiting:
            buf += ser.read(n_bytes - len(buf))
        else:
            time.sleep(0.03)
        if time.time() - start > timeout:
            print(f"##### Timeout ({len(buf)}/{n_bytes} bytes received)")
            return None
    return bytes(buf)


##################################################################
# Load and preprocess data
##################################################################
print(f"Loading parquet dataset: {DATA_PATH}")
table = pq.read_table(DATA_PATH, columns=FULL_FEATURES)
df = table.to_pandas().dropna().reset_index(drop=True)
print(f"Data shape: {df.shape}")

# Load the robust scaler used in training
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded.")

# Apply scaling
print("Scaling numeric features...")

scaled = scaler.transform(df[SCALER_FEATURES].values.astype(np.float32))

# Normalize time manually
time_col = df["Testtime[s]"].values.astype(np.float32)
time_norm = (time_col - time_col.min()) / (time_col.max() - time_col.min())
scaled[:, 0] = time_norm

# Robust-scale Capacity[Ah]
cap_scaler = RobustScaler()
capacity_scaled = cap_scaler.fit_transform(df["Capacity[Ah]"].values.reshape(-1, 1))

# Combine
data_scaled = np.concatenate([scaled, capacity_scaled], axis=1).astype(np.float32)
print("Scaling complete. Final feature shape:", data_scaled.shape)

##################################################################
# Shape into sliding windows (stride = STEP)
##################################################################
n_rows = len(data_scaled)
total_sequences = (n_rows - SEQ_LEN) // STEP
print(f"Generating {total_sequences:,} sequences (SEQ_LEN={SEQ_LEN}, STEP={STEP})")

# Prepare overlapping windows
windows = np.lib.stride_tricks.sliding_window_view(data_scaled, (SEQ_LEN, data_scaled.shape[1]))
windows = windows.reshape(-1, SEQ_LEN, len(FULL_FEATURES))[::STEP]
print(f"Effective sequences prepared: {len(windows)}")

##################################################################
# Serial connection
##################################################################
print(f"Connecting to {PORT} @ {BAUD}...")
ser = serial.Serial(PORT, BAUD, timeout=0.1)

# Allow CDC re-enumeration
ser.dtr = False
ser.rts = False
time.sleep(0.1)
ser.dtr = True
ser.rts = True
time.sleep(2.0)
ser.reset_input_buffer()

# Handshake
print("Performing handshake...")
ser.write(b"PING")
ser.flush()
resp = read_exact(ser, 4, timeout=2.0)
if resp == b"PONG":
    print("Handshake OK")
else:
    print(f"##### No proper handshake (got: {resp})")

ser.reset_input_buffer()
time.sleep(0.5)

##################################################################
# Streaming loop
##################################################################
start_time = time.time()
sent = 0
predictions = []
latencies = []  # record per-sequence latency

batch_start_time = start_time
bytes_sent_batch = 0
bytes_recv_batch = 0
lat_batch = []

print("\nStarting stream...")
for i in range(len(windows)):
    seq = np.asarray(windows[i], dtype="<f4")
    seq_bytes = seq.tobytes(order="C")

    # Start timer for latency
    t0 = time.time()

    # Send
    ser.write(FRAME_HDR + seq_bytes)
    ser.flush()
    bytes_sent_batch += len(FRAME_HDR) + len(seq_bytes)

    # Receive prediction
    pred_bytes = read_exact(ser, 4, timeout=READ_TIMEOUT)
    t1 = time.time()  # stop timer here

    # Handle timeout
    if pred_bytes is None:
        print(f"##### Timeout waiting for response at sequence {i}")
        break

    latency = t1 - t0
    latencies.append(latency)
    lat_batch.append(latency)
    bytes_recv_batch += 4

    pred = struct.unpack("<f", pred_bytes)[0]
    predictions.append(pred)
    sent += 1

    # Print progress + stats
    if (i + 1) % PRINT_EVERY == 0 or i == 0:
        now = time.time()
        dt = now - batch_start_time
        total_bytes = bytes_sent_batch + bytes_recv_batch
        throughput = total_bytes / dt if dt > 0 else 0.0
        avg_latency = np.mean(lat_batch) if lat_batch else 0.0

        print(f"[{i+1}/{len(windows)}] Pred SOH: {pred:.4f} | "
              f"{PRINT_EVERY/dt:.2f} seq/s | {throughput/1024:.1f} kB/s | "
              f"avg latency {avg_latency*1000:.1f} ms")

        # reset batch timers and counters
        batch_start_time = now
        bytes_sent_batch = 0
        bytes_recv_batch = 0
        lat_batch = []

ser.close()
elapsed = time.time() - start_time

print("\n===== STREAM COMPLETE =====")
print(f"Sequences processed: {sent}/{len(windows)}")
print(f"Total elapsed: {elapsed:.2f} s")
print(f"Avg latency per sequence: {np.mean(latencies)*1000:.2f} ms")
print(f"Avg time per sequence: {elapsed / max(sent, 1) * 1000:.2f} ms")


##################################################################
# Save predictions
##################################################################
efc_values = df["EFC"].iloc[np.arange(0, sent * STEP, STEP)].to_numpy()
out_df = pd.DataFrame({"EFC": efc_values[:sent], "predicted_SOH": predictions})
out_df.to_csv("stm32_predictions.csv", index=False)
print(f"Saved {sent} predictions to stm32_predictions.csv")
