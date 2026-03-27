#!/usr/bin/env python3
"""
PC vs STM32 SOH Comparison - Full Test Dataset

Uses the COMPLETE C11 parquet (like training) to:
1. Run PC inference with Base model (168h seq, hourly agg)
2. Compare against STM32 hardware predictions from latest HW test
3. Plot EVERYTHING from start to end

This is the PROPER way to evaluate - matching training pipeline exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import joblib
    import pyarrow.parquet as pq
    from sklearn.preprocessing import RobustScaler
except Exception as exc:
    raise SystemExit(f"Missing dependencies: {exc}")


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE"
)


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "DL_Models").exists() and (p / "STM32DC").exists():
            return p
    return start


def load_base_model(model_dir: Path):
    """Load Base LSTM model."""
    scaler_path = model_dir / "scaler_robust.joblib"
    checkpoint_dir = model_dir / "checkpoints"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints not found: {checkpoint_dir}")
    
    checkpoints = sorted(checkpoint_dir.glob("best_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
    checkpoint_path = checkpoints[0]
    print(f"  Loading: {checkpoint_path.name}")
    
    scaler = joblib.load(scaler_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Model architecture
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
        def __init__(self, in_features=20, embed_size=128, hidden_size=192,
                     mlp_hidden=160, num_layers=3, res_blocks=2, dropout=0.20):
            super().__init__()
            self.feature_proj = torch.nn.Sequential(
                torch.nn.Linear(in_features, embed_size),
                torch.nn.LayerNorm(embed_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout * 0.5),
                torch.nn.Linear(embed_size, embed_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout * 0.5),
            )
            self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers,
                                      batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
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

        def forward(self, x):
            x = self.feature_proj(x)
            out, _ = self.lstm(x)
            out = self.post_norm(out)
            for blk in self.res_blocks:
                out = blk(out)
            return self.head(out).squeeze(-1)
    
    model = SOH_LSTM_Seq2Seq()
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    return model, scaler


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to hourly features (mean/std/min/max)."""
    df = df.copy()
    df["hour"] = (df["Testtime[s]"] / 3600.0).astype(int)
    
    agg_dict = {}
    for feat in ["Voltage[V]", "Current[A]", "Temperature[°C]", "EFC", "Q_c"]:
        if feat in df.columns:
            for stat in ["mean", "std", "min", "max"]:
                agg_dict[f"{feat}_{stat}"] = (feat, stat)
    
    if "SOH" in df.columns:
        agg_dict["SOH"] = ("SOH", "last")
    
    df_hourly = df.groupby("hour").agg(**agg_dict).reset_index()
    
    # Fill NaN std with 0
    for col in df_hourly.columns:
        if "_std" in col:
            df_hourly[col] = df_hourly[col].fillna(0.0)
    
    return df_hourly


def predict_soh_pc(model, scaler, df_hourly: pd.DataFrame, seq_len: int = 168):
    """Run PC inference on hourly data."""
    feature_cols = []
    for base in ["Voltage[V]", "Current[A]", "Temperature[°C]", "EFC", "Q_c"]:
        for stat in ["mean", "std", "min", "max"]:
            col = f"{base}_{stat}"
            if col in df_hourly.columns:
                feature_cols.append(col)
    
    print(f"  Features: {len(feature_cols)}")
    
    X_raw = df_hourly[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_samples = len(X_raw)
    if n_samples < seq_len:
        print(f"Warning: Only {n_samples}h available, need {seq_len}h. Using shorter sequences.")
        seq_len = min(seq_len, n_samples)
    
    n_seqs = n_samples - seq_len + 1
    X_seqs = np.zeros((n_seqs, seq_len, len(feature_cols)), dtype=np.float32)
    for i in range(n_seqs):
        X_seqs[i] = X_raw[i:i+seq_len]
    
    # Scale
    X_flat = X_seqs.reshape(-1, len(feature_cols))
    X_scaled = scaler.transform(X_flat).astype(np.float32)
    X_scaled = X_scaled.reshape(n_seqs, seq_len, len(feature_cols))
    
    # Predict
    X_tensor = torch.from_numpy(X_scaled)
    preds = []
    model.eval()
    batch_size = 64
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch = X_tensor[i:i+batch_size]
            out = model(batch)
            preds.append(out[:, -1].cpu().numpy())  # Last timestep
    
    preds = np.concatenate(preds, axis=0).flatten()
    
    # Align with hourly dataframe
    result = np.full(n_samples, np.nan, dtype=np.float32)
    result[seq_len-1:] = preds
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="C11", help="Cell ID")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--model-dir", default="", help="Base model dir (auto-detect)")
    ap.add_argument("--stm32-csv", default="", help="Latest STM32 HW test CSV (auto-detect)")
    ap.add_argument("--max-hours", type=int, default=0, help="Limit hours (0=all)")
    ap.add_argument("--seq-len", type=int, default=168, help="Sequence length (hours)")
    ap.add_argument("--out-dir", default="", help="Output directory")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    
    data_root = Path(args.data_root)
    parquet = data_root / f"df_FE_{args.cell}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")
    
    # Auto-detect model
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        repo_root = find_repo_root(Path(__file__).resolve())
        model_dir = repo_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.3"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    
    # Auto-detect latest STM32 test
    if args.stm32_csv:
        stm32_csv = Path(args.stm32_csv)
    else:
        test_root = repo_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "6_test" / "STM32DC" / "LSTM_0.1.2.3"
        hw_dirs = sorted([d for d in test_root.glob("HW_C11_*") if d.is_dir()])
        if not hw_dirs:
            print("Warning: No STM32 HW test found, PC-only mode")
            stm32_csv = None
        else:
            stm32_csv = hw_dirs[-1] / "stm32_hw_c11.csv"
            print(f"Using latest STM32 test: {stm32_csv.parent.name}")
    
    # Output
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if stm32_csv:
            out_dir = stm32_csv.parent
        else:
            out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    
    print(f"\nLoading Base model from: {model_dir}")
    model, scaler = load_base_model(model_dir)
    
    print(f"\nReading parquet: {parquet}")
    pf = pq.ParquetFile(parquet)
    total_rows = pf.metadata.num_rows
    print(f"  Total rows: {total_rows:,}")
    
    # Read data
    df_full = pq.read_table(parquet).to_pandas()
    print(f"  Loaded: {len(df_full):,} rows")
    
    # Limit if requested
    if args.max_hours > 0:
        max_s = args.max_hours * 3600
        df_full = df_full[df_full["Testtime[s]"] <= max_s].reset_index(drop=True)
        print(f"  Limited to {args.max_hours}h: {len(df_full):,} rows")
    
    # Aggregate to hourly
    print("\nAggregating to hourly features...")
    df_hourly = aggregate_hourly(df_full)
    print(f"  Hourly samples: {len(df_hourly)}")
    
    # PC inference
    print(f"\nRunning PC inference (seq_len={args.seq_len}h)...")
    soh_pc_hourly = predict_soh_pc(model, scaler, df_hourly, seq_len=args.seq_len)
    df_hourly["SOH_PC"] = soh_pc_hourly
    
    # Prepare for plotting
    has_stm32 = False
    stm32_data = None
    
    if stm32_csv and stm32_csv.exists():
        print(f"\nLoading STM32 test: {stm32_csv}")
        df_stm32 = pd.read_csv(stm32_csv)
        print(f"  STM32 samples: {len(df_stm32):,}")
        has_stm32 = True
        stm32_data = df_stm32
    
    # Save results
    out_csv = out_dir / "pc_full_test_hourly.csv"
    df_hourly.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    
    # Metrics
    soh_pc_valid = soh_pc_hourly[np.isfinite(soh_pc_hourly)]
    metrics = {
        "total_hours": len(df_hourly),
        "valid_pc_predictions": len(soh_pc_valid),
        "pc_mean": float(np.mean(soh_pc_valid)),
        "pc_std": float(np.std(soh_pc_valid)),
        "pc_min": float(np.min(soh_pc_valid)),
        "pc_max": float(np.max(soh_pc_valid)),
    }
    
    if "SOH" in df_hourly.columns:
        soh_true_hourly = df_hourly["SOH"].values
        valid_both = np.isfinite(soh_pc_hourly) & np.isfinite(soh_true_hourly)
        if np.sum(valid_both) > 0:
            diff = soh_pc_hourly[valid_both] - soh_true_hourly[valid_both]
            metrics["mae_pc_true"] = float(np.mean(np.abs(diff)))
            metrics["rmse_pc_true"] = float(np.sqrt(np.mean(diff**2)))
            metrics["true_mean"] = float(np.mean(soh_true_hourly[valid_both]))
    
    out_json = out_dir / "pc_full_test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_json}")
    
    print("\n" + "="*60)
    print("PC Full Test Results")
    print("="*60)
    print(f"Hours: {metrics['total_hours']}")
    print(f"Valid PC: {metrics['valid_pc_predictions']}")
    print(f"PC: {metrics['pc_mean']:.4f} ± {metrics['pc_std']:.4f}")
    if "mae_pc_true" in metrics:
        print(f"MAE vs True: {metrics['mae_pc_true']:.6f}")
    print("="*60)
    
    # Plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(16, 10))
            
            # Main plot: full timeline
            ax1 = plt.subplot(2, 1, 1)
            hours = df_hourly["hour"].values
            ax1.plot(hours, soh_pc_hourly, label="SOH PC", linewidth=1, color='blue')
            if "SOH" in df_hourly.columns:
                ax1.plot(hours, df_hourly["SOH"].values, label="SOH True", linewidth=0.8, alpha=0.7, color='green', linestyle='--')
            ax1.set_xlabel("Time [hours]")
            ax1.set_ylabel("SOH")
            ax1.set_title(f"PC Full Test ({len(df_hourly)} hours, seq_len={args.seq_len}h)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, hours[-1])
            
            # Error plot
            if "SOH" in df_hourly.columns:
                ax2 = plt.subplot(2, 1, 2)
                valid_both = np.isfinite(soh_pc_hourly) & np.isfinite(df_hourly["SOH"].values)
                h_valid = hours[valid_both]
                err = soh_pc_hourly[valid_both] - df_hourly["SOH"].values[valid_both]
                ax2.plot(h_valid, err, linewidth=0.7, color='red', alpha=0.7)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2.set_xlabel("Time [hours]")
                ax2.set_ylabel("Error (PC - True)")
                if "mae_pc_true" in metrics:
                    ax2.set_title(f"PC Error (MAE={metrics['mae_pc_true']:.6f})")
                else:
                    ax2.set_title("PC Error")
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(0, hours[-1])
            
            plt.tight_layout()
            out_png = out_dir / "pc_full_test.png"
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"\nPlot: {out_png}")
            
        except Exception as e:
            print(f"Plot error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
