#!/usr/bin/env python3
"""
Compare PC-based SOH inference vs STM32 hardware SOH predictions.

Loads the original PyTorch LSTM model (Base, non-quantized) and runs inference
on the same input data that was sent to the STM32. Compares:
- PC predictions (original PyTorch model)
- STM32 predictions (quantized C implementation)
- Ground truth (from dataset)

Note: The 0.1.2.3 model expects hourly aggregated features with mean/std/min/max.
Since the STM32 hardware test sends raw 1Hz samples, we'll aggregate on-the-fly
to match the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import joblib
    from sklearn.preprocessing import RobustScaler
except Exception as exc:
    raise SystemExit(
        f"Missing dependencies: {exc}\n"
        "Install with: pip install torch joblib scikit-learn"
    ) from exc


def find_repo_root(start: Path) -> Path:
    """Find repository root by locating DL_Models folder."""
    for p in [start] + list(start.parents):
        if (p / "DL_Models").exists() and (p / "STM32DC").exists():
            return p
    return start


def load_base_model(model_dir: Path) -> tuple[torch.nn.Module, RobustScaler, dict]:
    """Load the Base (non-quantized) PyTorch LSTM model."""
    scaler_path = model_dir / "scaler_robust.joblib"
    checkpoint_dir = model_dir / "checkpoints"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints not found: {checkpoint_dir}")
    
    # Find best checkpoint
    checkpoints = sorted(checkpoint_dir.glob("best_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
    checkpoint_path = checkpoints[0]
    print(f"  Loading checkpoint: {checkpoint_path.name}")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Build model architecture (SOH_LSTM_Seq2Seq)
    class ResidualMLPBlock(torch.nn.Module):
        def __init__(self, dim: int, hidden: int, dropout: float = 0.15):
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, hidden)
            self.act = torch.nn.GELU()
            self.fc2 = torch.nn.Linear(hidden, dim)
            self.drop = torch.nn.Dropout(dropout)
            self.norm = torch.nn.LayerNorm(dim)

        def forward(self, x):
            out = self.fc2(self.act(self.fc1(x)))
            out = self.drop(out)
            return self.norm(x + out)

    class SOH_LSTM_Seq2Seq(torch.nn.Module):
        def __init__(
            self,
            in_features: int = 20,
            embed_size: int = 128,
            hidden_size: int = 192,
            mlp_hidden: int = 160,
            num_layers: int = 3,
            res_blocks: int = 2,
            dropout: float = 0.20,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.feature_proj = torch.nn.Sequential(
                torch.nn.Linear(in_features, embed_size),
                torch.nn.LayerNorm(embed_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout * 0.5),
                torch.nn.Linear(embed_size, embed_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout * 0.5),
            )

            self.lstm = torch.nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            self.post_norm = torch.nn.LayerNorm(hidden_size)
            self.res_blocks = torch.nn.ModuleList(
                [ResidualMLPBlock(hidden_size, mlp_hidden, dropout) for _ in range(res_blocks)]
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, mlp_hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(mlp_hidden, mlp_hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(mlp_hidden, 1),
            )

        def forward(self, x, state=None, return_state=False):
            x = self.feature_proj(x)
            out, new_state = self.lstm(x, state)
            out = self.post_norm(out)
            for blk in self.res_blocks:
                out = blk(out)
            y_seq = self.head(out).squeeze(-1)
            if return_state:
                return y_seq, new_state
            return y_seq
    
    # Model config from train_soh.yaml
    in_features = 20  # 5 base features × 4 aggregations (mean/std/min/max)
    embed_size = 128
    hidden_size = 192
    mlp_hidden = 160
    num_layers = 3
    res_blocks = 2
    dropout = 0.20
    
    model = SOH_LSTM_Seq2Seq(
        in_features=in_features,
        embed_size=embed_size,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        res_blocks=res_blocks,
        dropout=dropout,
    )
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    meta = {
        "in_features": in_features,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "checkpoint": str(checkpoint_path),
    }
    
    return model, scaler, meta


def aggregate_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw 1Hz samples into hourly features with mean/std/min/max.
    
    This matches the training pipeline for model 0.1.2.3.
    """
    # Convert t_s to hours
    df = df.copy()
    df["hour"] = (df["t_s"] / 3600.0).astype(int)
    
    # Base features to aggregate
    base_features = ["pack_v", "current_a", "temp_c", "efc", "q_c"]
    
    # Aggregate per hour
    agg_dict = {}
    for feat in base_features:
        if feat in df.columns:
            agg_dict[f"{feat}_mean"] = (feat, "mean")
            agg_dict[f"{feat}_std"] = (feat, "std")
            agg_dict[f"{feat}_min"] = (feat, "min")
            agg_dict[f"{feat}_max"] = (feat, "max")
    
    # Also keep SOH true/stm32 (take last value per hour)
    if "soh_true" in df.columns:
        agg_dict["soh_true"] = ("soh_true", "last")
    if "soh_stm32" in df.columns:
        agg_dict["soh_stm32"] = ("soh_stm32", "last")
    
    df_hourly = df.groupby("hour").agg(**agg_dict).reset_index()
    
    # Fill NaN std with 0 (constant hours)
    for col in df_hourly.columns:
        if "_std" in col:
            df_hourly[col] = df_hourly[col].fillna(0.0)
    
    return df_hourly


def predict_soh_pc(
    model: torch.nn.Module,
    scaler: RobustScaler,
    df: pd.DataFrame,
    seq_len: int = 168,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run STATEFUL PC-based SOH inference (matching training test script).
    
    Returns: predictions [N] aligned with hourly dataframe
    """
    # Feature columns (20 aggregated features)
    feature_cols = []
    for base in ["pack_v", "current_a", "temp_c", "efc", "q_c"]:
        for agg in ["mean", "std", "min", "max"]:
            col = f"{base}_{agg}"
            if col in df.columns:
                feature_cols.append(col)
    
    if len(feature_cols) == 0:
        raise ValueError("No aggregated feature columns found in dataframe")
    
    print(f"  Using {len(feature_cols)} features: {feature_cols[:3]}...")
    
    # Extract and scale features
    X_raw = df[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    # Stateful prediction (like training test script)
    model.eval()
    preds = []
    h_state = None
    c_state = None
    
    with torch.no_grad():
        for start in range(0, len(X_scaled), seq_len):
            end = min(start + seq_len, len(X_scaled))
            xb = torch.from_numpy(X_scaled[start:end]).unsqueeze(0)  # [1, chunk_len, features]
            
            if h_state is None:
                y_seq, (h_state, c_state) = model(xb, state=None, return_state=True)
            else:
                y_seq, (h_state, c_state) = model(xb, state=(h_state, c_state), return_state=True)
            
            preds.append(y_seq.squeeze(0).cpu().numpy())
    
    result = np.concatenate(preds, axis=0)
    return result


def map_hourly_to_raw(df_raw: pd.DataFrame, df_hourly: pd.DataFrame, soh_pc_hourly: np.ndarray) -> np.ndarray:
    """Map hourly PC predictions back to raw 1Hz samples (forward-fill)."""
    df_raw = df_raw.copy()
    df_raw["hour"] = (df_raw["t_s"] / 3600.0).astype(int)
    
    # Create hour -> soh_pc mapping
    hour_to_soh = {}
    for i, row in df_hourly.iterrows():
        if np.isfinite(soh_pc_hourly[i]):
            hour_to_soh[int(row["hour"])] = soh_pc_hourly[i]
    
    # Map to raw samples
    soh_pc_raw = np.full(len(df_raw), np.nan, dtype=np.float32)
    for i, row in df_raw.iterrows():
        hour = int(row["hour"])
        if hour in hour_to_soh:
            soh_pc_raw[i] = hour_to_soh[hour]
    
    return soh_pc_raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PC (Base model) vs STM32 SOH predictions")
    ap.add_argument("--csv", type=str, required=True, help="Path to stm32_hw_c11.csv")
    ap.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Path to Base model directory (default: auto-detect)",
    )
    ap.add_argument("--seq-len", type=int, default=168, help="LSTM sequence length (hours)")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: same as CSV)")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = ap.parse_args()
    
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Auto-detect model directory
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    else:
        repo_root = find_repo_root(csv_path)
        model_dir = (
            repo_root
            / "DL_Models"
            / "LFP_SOH_Optimization_Study"
            / "2_models"
            / "LSTM"
            / "Base"
            / "0.1.2.3"
        )
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Avoid matplotlib writing to temp
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    
    print(f"Loading Base model from: {model_dir}")
    model, scaler, meta = load_base_model(model_dir)
    
    print(f"\nReading CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Rows: {len(df_raw)}")
    print(f"  Columns: {list(df_raw.columns)}")
    
    # Check required columns
    required = ["pack_v", "current_a", "temp_c", "soh_stm32"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Optional features
    if "efc" not in df_raw.columns:
        print("Warning: 'efc' column missing, filling with zeros")
        df_raw["efc"] = 0.0
    if "q_c" not in df_raw.columns:
        print("Warning: 'q_c' column missing, filling with zeros")
        df_raw["q_c"] = 0.0
    
    # Aggregate to hourly features
    print("\nAggregating to hourly features (matching training pipeline)...")
    df_hourly = aggregate_hourly_features(df_raw)
    print(f"  Hourly samples: {len(df_hourly)}")
    
    # Run PC inference on hourly data
    print(f"\nRunning PC inference (seq_len={args.seq_len} hours)...")
    soh_pc_hourly = predict_soh_pc(model, scaler, df_hourly, seq_len=args.seq_len)
    df_hourly["soh_pc"] = soh_pc_hourly
    
    # Map hourly PC predictions back to raw 1Hz samples
    print("Mapping hourly predictions to raw samples...")
    soh_pc_raw = map_hourly_to_raw(df_raw, df_hourly, soh_pc_hourly)
    df_raw["soh_pc"] = soh_pc_raw
    
    # Extract STM32 predictions and ground truth
    soh_stm32 = df_raw["soh_stm32"].values
    soh_pc = soh_pc_raw
    has_true = "soh_true" in df_raw.columns
    soh_true = df_raw["soh_true"].values if has_true else None
    
    # Filter valid predictions
    valid_pc_stm32 = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
    n_valid_pc_stm32 = int(np.sum(valid_pc_stm32))
    
    if n_valid_pc_stm32 == 0:
        print("ERROR: No valid overlapping PC-STM32 predictions found!")
        sys.exit(1)
    
    print(f"\nValid PC-STM32 predictions: {n_valid_pc_stm32}/{len(df_raw)}")
    
    # Compute PC vs STM32 metrics
    soh_pc_cmp = soh_pc[valid_pc_stm32]
    soh_stm32_cmp = soh_stm32[valid_pc_stm32]
    
    diff_pc_stm32 = soh_pc_cmp - soh_stm32_cmp
    mae_pc_stm32 = float(np.mean(np.abs(diff_pc_stm32)))
    rmse_pc_stm32 = float(np.sqrt(np.mean(diff_pc_stm32**2)))
    max_abs_err_pc_stm32 = float(np.max(np.abs(diff_pc_stm32)))
    
    metrics = {
        "n_total_raw": len(df_raw),
        "n_hourly": len(df_hourly),
        "n_valid_pc_stm32": n_valid_pc_stm32,
        "mae_pc_stm32": mae_pc_stm32,
        "rmse_pc_stm32": rmse_pc_stm32,
        "max_abs_error_pc_stm32": max_abs_err_pc_stm32,
        "pc_mean": float(np.mean(soh_pc_cmp)),
        "pc_std": float(np.std(soh_pc_cmp)),
        "stm32_mean": float(np.mean(soh_stm32_cmp)),
        "stm32_std": float(np.std(soh_stm32_cmp)),
    }
    
    # Also compare against ground truth if available
    if has_true and soh_true is not None:
        valid_all = valid_pc_stm32 & np.isfinite(soh_true)
        if int(np.sum(valid_all)) > 0:
            soh_true_cmp = soh_true[valid_all]
            soh_pc_true_cmp = soh_pc[valid_all]
            soh_stm32_true_cmp = soh_stm32[valid_all]
            
            diff_pc_true = soh_pc_true_cmp - soh_true_cmp
            diff_stm32_true = soh_stm32_true_cmp - soh_true_cmp
            
            metrics["n_valid_all"] = int(np.sum(valid_all))
            metrics["mae_pc_true"] = float(np.mean(np.abs(diff_pc_true)))
            metrics["mae_stm32_true"] = float(np.mean(np.abs(diff_stm32_true)))
            metrics["true_mean"] = float(np.mean(soh_true_cmp))
            metrics["true_std"] = float(np.std(soh_true_cmp))
    
    print("\n" + "="*60)
    print("PC vs STM32 SOH Comparison (Original PyTorch Base Model)")
    print("="*60)
    print(f"Valid predictions:     {n_valid_pc_stm32}/{len(df_raw)}")
    print(f"\nPC vs STM32:")
    print(f"  MAE:                 {mae_pc_stm32:.6f}")
    print(f"  RMSE:                {rmse_pc_stm32:.6f}")
    print(f"  Max absolute error:  {max_abs_err_pc_stm32:.6f}")
    print(f"  PC   mean±std:       {metrics['pc_mean']:.4f} ± {metrics['pc_std']:.4f}")
    print(f"  STM32 mean±std:      {metrics['stm32_mean']:.4f} ± {metrics['stm32_std']:.4f}")
    
    if "mae_pc_true" in metrics:
        print(f"\nVs Ground Truth:")
        print(f"  PC   MAE:            {metrics['mae_pc_true']:.6f}")
        print(f"  STM32 MAE:           {metrics['mae_stm32_true']:.6f}")
        print(f"  True mean±std:       {metrics['true_mean']:.4f} ± {metrics['true_std']:.4f}")
    
    print("="*60 + "\n")
    
    # Save results
    out_csv = out_dir / "pc_vs_stm32_soh_full.csv"
    df_raw.to_csv(out_csv, index=False)
    print(f"Saved augmented CSV: {out_csv}")
    
    out_json = out_dir / "pc_vs_stm32_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out_json}")
    
    # Plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # Plot ENTIRE dataset (not just valid predictions)
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            
            # Full timeseries (all samples, including NaN predictions)
            ax = axes[0, 0]
            t_hours = df_raw["t_s"].values / 3600.0
            ax.plot(t_hours, soh_pc, label="SOH PC (Base)", alpha=0.8, linewidth=0.8, color='blue')
            ax.plot(t_hours, soh_stm32, label="SOH STM32 (Quantized)", alpha=0.8, linewidth=0.8, color='red')
            if has_true:
                ax.plot(t_hours, soh_true, label="SOH True", alpha=0.6, linewidth=0.6, linestyle='--', color='green')
            ax.set_xlabel("Time [hours]")
            ax.set_ylabel("SOH")
            ax.set_title(f"Complete Dataset: PC vs STM32 SOH ({len(df_raw)} samples, {len(df_hourly)} hours)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, t_hours[-1])
            
            # Scatter plot PC vs STM32 (valid only)
            ax = axes[0, 1]
            ax.scatter(soh_stm32_cmp, soh_pc_cmp, alpha=0.3, s=2, color='purple')
            lim_min = min(soh_stm32_cmp.min(), soh_pc_cmp.min())
            lim_max = max(soh_stm32_cmp.max(), soh_pc_cmp.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label="Perfect match", linewidth=1)
            ax.set_xlabel("SOH STM32")
            ax.set_ylabel("SOH PC")
            ax.set_title(f"PC vs STM32 Correlation (MAE={mae_pc_stm32:.6f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Error histogram
            ax = axes[1, 0]
            ax.hist(diff_pc_stm32, bins=100, alpha=0.7, edgecolor='black', color='orange')
            ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
            ax.set_xlabel("Error (PC - STM32)")
            ax.set_ylabel("Count")
            ax.set_title(f"Error Distribution (RMSE={rmse_pc_stm32:.6f})")
            ax.grid(True, alpha=0.3)
            
            # Error over time (full)
            ax = axes[1, 1]
            idx_valid = np.where(valid_pc_stm32)[0]
            t_valid = t_hours[idx_valid]
            ax.plot(t_valid, diff_pc_stm32, alpha=0.7, linewidth=0.5, color='red')
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_xlabel("Time [hours]")
            ax.set_ylabel("Error (PC - STM32)")
            ax.set_title(f"Error Over Time (max={max_abs_err_pc_stm32:.6f})")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, t_hours[-1])
            
            # vs Ground Truth comparison (if available)
            if has_true and "mae_pc_true" in metrics:
                # PC vs True
                ax = axes[2, 0]
                valid_all = valid_pc_stm32 & np.isfinite(soh_true)
                idx_all = np.where(valid_all)[0]
                t_all = t_hours[idx_all]
                soh_true_cmp = soh_true[valid_all]
                soh_pc_true_cmp = soh_pc[valid_all]
                diff_pc_true = soh_pc_true_cmp - soh_true_cmp
                ax.plot(t_all, diff_pc_true, alpha=0.7, linewidth=0.5, color='blue', label=f'PC Error (MAE={metrics["mae_pc_true"]:.6f})')
                ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
                ax.set_xlabel("Time [hours]")
                ax.set_ylabel("Error (Pred - True)")
                ax.set_title("PC vs Ground Truth Error")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, t_hours[-1])
                
                # STM32 vs True
                ax = axes[2, 1]
                soh_stm32_true_cmp = soh_stm32[valid_all]
                diff_stm32_true = soh_stm32_true_cmp - soh_true_cmp
                ax.plot(t_all, diff_stm32_true, alpha=0.7, linewidth=0.5, color='red', label=f'STM32 Error (MAE={metrics["mae_stm32_true"]:.6f})')
                ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
                ax.set_xlabel("Time [hours]")
                ax.set_ylabel("Error (Pred - True)")
                ax.set_title("STM32 vs Ground Truth Error")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, t_hours[-1])
            else:
                # No ground truth: show hourly aggregation info
                ax = axes[2, 0]
                ax.text(0.5, 0.5, 
                       f"Hourly Aggregation:\n"
                       f"Raw samples: {len(df_raw)}\n"
                       f"Hours: {len(df_hourly)}\n"
                       f"Seq length: {args.seq_len}h\n"
                       f"Valid predictions: {n_valid_pc_stm32}",
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.axis('off')
                
                ax = axes[2, 1]
                ax.text(0.5, 0.5, 
                       f"Model Config:\n"
                       f"Hidden: 192\n"
                       f"Layers: 3\n"
                       f"Embed: 128\n"
                       f"Dropout: 0.20",
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.axis('off')
            
            plt.tight_layout()
            out_png = out_dir / "pc_vs_stm32_soh_comparison.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {out_png}")
            
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
