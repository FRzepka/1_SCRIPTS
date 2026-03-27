#!/usr/bin/env python3
"""Simple plot: PC predictions + STM32 predictions vs ground truth."""
import sys
import math
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import load as joblib_load

def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location('train_soh_module', str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stateful_predict(model, X: np.ndarray, chunk: int, device: torch.device):
    model.eval()
    preds = []
    h = None
    c = None
    with torch.inference_mode():
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            xb = torch.from_numpy(X[start:end]).unsqueeze(0).to(device)
            if h is None:
                y_seq, (h, c) = model(xb, state=None, return_state=True)
            else:
                y_seq, (h, c) = model(xb, state=(h, c), return_state=True)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    # Paths
    model_dir = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_SOH_Optimization_Study\2_models\LSTM\Base\0.1.2.3")
    train_py = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_SOH_Optimization_Study\1_training\0.1.2.3\scripts\train_soh.py")
    csv_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\DL_Models\LFP_SOH_Optimization_Study\6_test\STM32DC\LSTM_0.1.2.3\HW_C11_20260120_105233\stm32_hw_c11.csv")
    
    print("Loading model...")
    train_mod = load_train_module(train_py)
    
    device = torch.device('cpu')
    ckpt_path = model_dir / 'checkpoints' / 'best_epoch0093_rmse0.02165.pt'
    scaler_path = model_dir / 'scaler_robust.joblib'
    
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get('model_state_dict', state)
    scaler = joblib_load(scaler_path)
    
    model = train_mod.SOH_LSTM_Seq2Seq(
        in_features=20,
        embed_size=128,
        hidden_size=192,
        mlp_hidden=160,
        num_layers=3,
        res_blocks=2,
        bidirectional=False,
        dropout=0.20,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Loading CSV...")
    df_raw = pd.read_csv(csv_path)
    
    # Aggregate to hourly
    print("Aggregating hourly...")
    df_raw["hour"] = (df_raw["t_s"] / 3600.0).astype(int)
    base_features = ["pack_v", "current_a", "temp_c", "efc", "q_c"]
    
    agg_dict = {}
    for feat in base_features:
        agg_dict[f"{feat}_mean"] = (feat, "mean")
        agg_dict[f"{feat}_std"] = (feat, "std")
        agg_dict[f"{feat}_min"] = (feat, "min")
        agg_dict[f"{feat}_max"] = (feat, "max")
    agg_dict["soh_true"] = ("soh_true", "last")
    agg_dict["soh_stm32"] = ("soh_stm32", "last")
    
    df_hourly = df_raw.groupby("hour").agg(**agg_dict).reset_index()
    for col in df_hourly.columns:
        if "_std" in col:
            df_hourly[col] = df_hourly[col].fillna(0.0)
    
    print(f"Hourly samples: {len(df_hourly)}")
    
    # Extract features
    feature_cols = []
    for feat in base_features:
        for stat in ["mean", "std", "min", "max"]:
            feature_cols.append(f"{feat}_{stat}")
    
    X = df_hourly[feature_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X).astype(np.float32)
    
    y_true = df_hourly["soh_true"].to_numpy(dtype=np.float32)
    y_stm32 = df_hourly["soh_stm32"].to_numpy(dtype=np.float32)
    
    # PC prediction (stateful)
    print("Running PC inference...")
    y_pc = stateful_predict(model, X_scaled, chunk=168, device=device)
    
    if len(y_pc) != len(y_true):
        min_len = min(len(y_pc), len(y_true), len(y_stm32))
        y_pc = y_pc[:min_len]
        y_true = y_true[:min_len]
        y_stm32 = y_stm32[:min_len]
    
    # Metrics
    valid_mask = ~np.isnan(y_stm32)
    if np.sum(valid_mask) > 0:
        mae_pc = np.abs(y_pc - y_true).mean()
        mae_stm32 = np.abs(y_stm32[valid_mask] - y_true[valid_mask]).mean()
        mae_pc_stm32 = np.abs(y_pc[valid_mask] - y_stm32[valid_mask]).mean()
        print(f"\nMetrics:")
        print(f"  PC MAE vs GT:    {mae_pc:.6f} ({mae_pc*100:.2f}%)")
        print(f"  STM32 MAE vs GT: {mae_stm32:.6f} ({mae_stm32*100:.2f}%)")
        print(f"  PC vs STM32 MAE: {mae_pc_stm32:.6f} ({mae_pc_stm32*100:.2f}%)")
    
    # Plot
    print("Creating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Timeseries - all together
    ax = axes[0, 0]
    ax.plot(y_true, label='Ground Truth', linewidth=1.2, color='black', alpha=0.7)
    ax.plot(y_pc, label='PC Prediction', linewidth=0.9, color='blue', alpha=0.7)
    ax.plot(y_stm32, label='STM32 Prediction', linewidth=0.9, color='red', alpha=0.7, linestyle='--')
    ax.set_xlabel('Hour')
    ax.set_ylabel('SOH')
    ax.set_title('SOH Predictions: PC + STM32 vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.05)
    
    # PC scatter
    ax = axes[0, 1]
    ax.scatter(y_true, y_pc, s=20, alpha=0.5, color='blue')
    ax.plot([0.85, 1.0], [0.85, 1.0], 'k--', alpha=0.5)
    ax.set_xlabel('Ground Truth SOH')
    ax.set_ylabel('PC Predicted SOH')
    ax.set_title(f'PC Accuracy (MAE={mae_pc*100:.2f}%)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.85, 1.05)
    ax.set_ylim(0.85, 1.05)
    ax.set_aspect('equal')
    
    # STM32 scatter
    ax = axes[1, 0]
    mask = ~np.isnan(y_stm32)
    ax.scatter(y_true[mask], y_stm32[mask], s=20, alpha=0.5, color='red')
    ax.plot([0.85, 1.0], [0.85, 1.0], 'k--', alpha=0.5)
    ax.set_xlabel('Ground Truth SOH')
    ax.set_ylabel('STM32 Predicted SOH')
    ax.set_title(f'STM32 Accuracy (MAE={mae_stm32*100:.2f}%)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.85, 1.05)
    ax.set_ylim(0.85, 1.05)
    ax.set_aspect('equal')
    
    # PC vs STM32
    ax = axes[1, 1]
    ax.scatter(y_stm32[mask], y_pc[mask], s=20, alpha=0.5, color='purple')
    ax.plot([0.85, 1.0], [0.85, 1.0], 'k--', alpha=0.5)
    ax.set_xlabel('STM32 Predicted SOH')
    ax.set_ylabel('PC Predicted SOH')
    ax.set_title(f'PC vs STM32 (MAE={mae_pc_stm32*100:.2f}%)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.85, 1.05)
    ax.set_ylim(0.85, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    out_path = csv_path.parent / 'pc_stm32_comparison_FINAL.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {out_path}")
    
    plt.close()


if __name__ == '__main__':
    main()
