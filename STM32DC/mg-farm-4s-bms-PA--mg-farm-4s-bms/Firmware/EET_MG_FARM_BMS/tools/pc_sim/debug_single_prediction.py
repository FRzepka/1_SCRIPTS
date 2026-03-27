#!/usr/bin/env python3
"""
DEBUG: Test single prediction from PC model to verify correctness.
"""
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent))

# Import model (same as compare script)
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        out = self.drop(out)
        return self.norm(x + out)

class SOH_LSTM_Seq2Seq(nn.Module):
    def __init__(self, in_features, embed_size, hidden_size, mlp_hidden, num_layers, res_blocks, dropout):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.res_blocks = nn.ModuleList([ResidualMLPBlock(hidden_size, mlp_hidden, dropout) for _ in range(res_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        x = self.feature_proj(x)
        out, _ = self.lstm(x)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        return self.head(out).squeeze(-1)


def main():
    # Paths
    model_dir = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_SOH_Optimization_Study\2_models\LSTM\Base\0.1.2.3")
    csv_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_SOH_Optimization_Study\6_test\STM32DC\LSTM_0.1.2.3\HW_C11_20260120_105233\stm32_hw_c11.csv")
    
    print("=" * 60)
    print("DEBUG: Single Prediction Test")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading Base model...")
    checkpoint = torch.load(model_dir / "checkpoints" / "best_epoch0093_rmse0.02165.pt", map_location='cpu', weights_only=False)
    scaler = joblib.load(model_dir / "scaler_robust.joblib")
    
    model = SOH_LSTM_Seq2Seq(20, 128, 192, 160, 3, 2, 0.20)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    print("   Model loaded successfully")
    
    # Load data
    print("\n2. Loading CSV and aggregating...")
    df_raw = pd.read_csv(csv_path)
    print(f"   Rows: {len(df_raw)}")
    
    # Aggregate to hourly
    df_raw["hour"] = (df_raw["t_s"] / 3600.0).astype(int)
    base_features = ["pack_v", "current_a", "temp_c", "efc", "q_c"]
    
    agg_dict = {}
    for feat in base_features:
        agg_dict[f"{feat}_mean"] = (feat, "mean")
        agg_dict[f"{feat}_std"] = (feat, "std")
        agg_dict[f"{feat}_min"] = (feat, "min")
        agg_dict[f"{feat}_max"] = (feat, "max")
    
    df_hourly = df_raw.groupby("hour").agg(**agg_dict).reset_index()
    print(f"   Hourly samples: {len(df_hourly)}")
    
    # Take first 48 hours as sequence
    seq_len = 48
    df_seq = df_hourly.iloc[:seq_len].copy()
    
    # Feature columns
    feature_cols = []
    for feat in base_features:
        for stat in ["mean", "std", "min", "max"]:
            feature_cols.append(f"{feat}_{stat}")
    
    X = df_seq[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n3. Input sequence shape: {X.shape}")
    print(f"   Features: {feature_cols[:3]}...")
    print(f"   First hour values: {X[0, :5]}")
    
    # Scale
    X_scaled = scaler.transform(X).astype(np.float32)
    print(f"   Scaled values: {X_scaled[0, :5]}")
    
    # Predict
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(0)  # [1, 48, 20]
    
    print("\n4. Running inference...")
    with torch.no_grad():
        y_seq = model(X_tensor)  # [1, 48]
    
    predictions = y_seq[0].cpu().numpy()
    
    # Get ground truth from CSV
    soh_true_hourly = []
    for h in range(seq_len):
        rows = df_raw[df_raw["hour"] == h]
        if len(rows) > 0:
            soh_true_hourly.append(rows.iloc[-1]["soh_true"])
        else:
            soh_true_hourly.append(np.nan)
    soh_true_hourly = np.array(soh_true_hourly)
    
    print("\n5. Results:")
    print("   Hour | Predicted | Ground Truth | Error")
    print("   -----|-----------|--------------|-------")
    for i in [0, 1, 5, 10, 20, 30, 40, 47]:
        if i < len(predictions):
            pred = predictions[i]
            true_val = soh_true_hourly[i]
            err = pred - true_val if not np.isnan(true_val) else np.nan
            print(f"   {i:4d} | {pred:9.6f} | {true_val:12.6f} | {err:7.4f}")
    
    # Summary stats
    valid_mask = ~np.isnan(soh_true_hourly)
    if np.sum(valid_mask) > 0:
        mae = np.abs(predictions[valid_mask] - soh_true_hourly[valid_mask]).mean()
        pred_mean = predictions[valid_mask].mean()
        true_mean = soh_true_hourly[valid_mask].mean()
        
        print(f"\n6. Summary:")
        print(f"   Predictions mean: {pred_mean:.6f}")
        print(f"   Ground truth mean: {true_mean:.6f}")
        print(f"   MAE: {mae:.6f} ({mae*100:.2f}%)")
        print(f"   Pred range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"   True range: [{np.nanmin(soh_true_hourly):.6f}, {np.nanmax(soh_true_hourly):.6f}]")
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    if predictions[0] > 1.0:
        print("❌ ERROR: Predictions start > 1.0! Model output is incorrectly scaled!")
    elif pred_mean < 0.95:
        print("❌ ERROR: Predictions too low compared to ground truth!")
    elif mae > 0.05:
        print("⚠️  WARNING: High MAE - model may not be working correctly")
    else:
        print("✅ Predictions look reasonable")
    print("=" * 60)


if __name__ == "__main__":
    main()
