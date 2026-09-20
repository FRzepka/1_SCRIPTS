# Recreated based on TUB files and interpretation
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

# === Parameters ===
SEQ_LEN = 2048
BATCH_SIZE = 256
STEP = 2048  # stride between sequence starts
PARQUET_PATH = "df_FE_C13.parquet"

############################################################
# MODEL
############################################################
class LSTM_SOHPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),  # mlp.0
            nn.ReLU(),                    # mlp.1
            nn.ReLU(),                    # mlp.2
            nn.Linear(128, 1),            # mlp.3
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.mlp(last)


############################################################
# INFERENCE
############################################################
def main():
    checkpoint_path = "2.1.0.0_soh_epoch0005_mae0.00369.pt"
    scaler_path = "scaler_robust.joblib"

    # Feature order for scaler
    scaler_order = [
        "Testtime[s]",
        "Voltage[V]",
        "Current[A]",
        "Temperature[°C]",
        "EFC",
        "Q_c",
    ]

    # Model expects these 7 features (time unscaled, capacity unscaled)
    model_features = scaler_order + ["Capacity[Ah]"]

    # === Load resources ===
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)

    print("Loading model...")
    model = LSTM_SOHPredictor()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    print(f"Loading parquet: {PARQUET_PATH}")
    table = pq.read_table(PARQUET_PATH, columns=model_features)
    df = table.to_pandas()
    n_rows = len(df)
    print(f"Data shape: {n_rows:,} rows × {len(df.columns)} columns")

    # === Apply scaler correctly ===
    print("Scaling numeric features...")
    scaled = scaler.transform(df[scaler_order].values.astype(np.float32))

    # Scale time
    time_col = df["Testtime[s]"].values.astype(np.float32)
    time_norm = (time_col - time_col.min()) / (time_col.max() - time_col.min())
    scaled[:, 0] = time_norm

    # Add unscaled Capacity[Ah] as last feature
    capacity_raw = df["Capacity[Ah]"].values.astype(np.float32).reshape(-1, 1)
    cap_scaler = RobustScaler()
    capacity_scaled = cap_scaler.fit_transform(capacity_raw)
    data_scaled = np.concatenate([scaled, capacity_scaled], axis=1).astype(np.float32)

    print("Scaling complete. Final feature shape:", data_scaled.shape)

    print(data_scaled[:10])
    print(data_scaled[-10:])

    # === Prepare output ===
    out_file = "predictions.csv"
    with open(out_file, "w") as f:
        f.write("EFC,predicted_SOH\n")

    print("Starting batched inference...")

    total_sequences = (n_rows - SEQ_LEN) // STEP
    processed = 0

    for start in range(0, n_rows - SEQ_LEN, STEP * BATCH_SIZE):
        end = min(start + SEQ_LEN + STEP * (BATCH_SIZE - 1), n_rows)
        block = data_scaled[start:end]

        # Build sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(block, (SEQ_LEN, 7))
        windows = windows.reshape(-1, SEQ_LEN, 7)[::STEP]

        # Convert to tensor
        windows_tensor = torch.from_numpy(windows)

        # Run model inference
        with torch.no_grad():
            batch_pred = model(windows_tensor).squeeze().numpy()

        # Save results
        with open(out_file, "a") as f:
            for j, p in enumerate(batch_pred):
                seq_start = start + j * STEP
                efc_val = df["EFC"].iloc[seq_start]
                f.write(f"{efc_val:.6f},{p}\n")


        processed += len(batch_pred)
        if processed % (BATCH_SIZE * 10) == 0 or start == 0:
            pct = 100.0 * processed / total_sequences
            print(f"{processed:,}/{total_sequences:,} sequences processed ({pct:.2f}%)")

    print(f"Finished. Saved {processed:,} predictions to {out_file}")


if __name__ == "__main__":
    main()
