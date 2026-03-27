#!/usr/bin/env python3
"""
Plot PC vs STM32 SOH predictions on HOURLY aggregated data (like training).
This is the correct comparison - not the expanded 1Hz data!
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description="Plot hourly PC vs STM32 comparison")
    parser.add_argument("--csv", required=True, help="Path to pc_vs_stm32_soh_full.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Total raw samples: {len(df_raw)}")

    # Aggregate to hourly - take LAST value of each hour for predictions
    df_raw["hour"] = (df_raw["t_s"] / 3600.0).astype(int)
    df_hourly = df_raw.groupby("hour").agg({
        "t_s": "last",
        "soh_pc": "last",
        "soh_stm32": "last",
        "soh_true": "last",
    }).reset_index()

    print(f"  Hourly samples: {len(df_hourly)}")
    
    # Calculate metrics on hourly data (where both valid)
    mask = pd.notna(df_hourly["soh_pc"]) & pd.notna(df_hourly["soh_stm32"])
    df_valid = df_hourly[mask].copy()
    n_valid = len(df_valid)
    
    if n_valid == 0:
        print("ERROR: No valid overlapping predictions!")
        return
    
    mae = np.abs(df_valid["soh_pc"] - df_valid["soh_stm32"]).mean()
    rmse = np.sqrt(((df_valid["soh_pc"] - df_valid["soh_stm32"])**2).mean())
    max_err = np.abs(df_valid["soh_pc"] - df_valid["soh_stm32"]).max()
    
    # Metrics vs ground truth
    mae_pc_true = np.abs(df_valid["soh_pc"] - df_valid["soh_true"]).mean()
    mae_stm32_true = np.abs(df_valid["soh_stm32"] - df_valid["soh_true"]).mean()
    
    print(f"\n{'='*60}")
    print(f"HOURLY Aggregated Comparison (Like Training)")
    print(f"{'='*60}")
    print(f"Valid hourly predictions: {n_valid}/{len(df_hourly)}")
    print(f"\nPC vs STM32:")
    print(f"  MAE:  {mae:.6f} ({mae*100:.2f}%)")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Max:  {max_err:.6f}")
    print(f"\nVs Ground Truth:")
    print(f"  PC MAE:    {mae_pc_true:.6f} ({mae_pc_true*100:.2f}%)")
    print(f"  STM32 MAE: {mae_stm32_true:.6f} ({mae_stm32_true*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"HOURLY PC vs STM32 SOH Comparison (n={n_valid} hours)\nMAE={mae*100:.2f}%", 
                 fontsize=14, fontweight='bold')
    
    t_h = df_hourly["t_s"] / 3600.0
    
    # 1. Timeseries - PC
    ax = axes[0, 0]
    ax.plot(t_h, df_hourly["soh_true"], 'k-', alpha=0.6, label='Ground Truth', linewidth=1.5)
    ax.plot(t_h, df_hourly["soh_pc"], 'b-', alpha=0.8, label='PC Prediction', linewidth=1.0)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("SOH")
    ax.set_title("PC Predictions (Hourly)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Timeseries - STM32
    ax = axes[0, 1]
    ax.plot(t_h, df_hourly["soh_true"], 'k-', alpha=0.6, label='Ground Truth', linewidth=1.5)
    ax.plot(t_h, df_hourly["soh_stm32"], 'r-', alpha=0.8, label='STM32 Prediction', linewidth=1.0)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("SOH")
    ax.set_title("STM32 Predictions (Hourly)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PC vs STM32 overlay
    ax = axes[1, 0]
    ax.plot(t_h, df_hourly["soh_true"], 'k-', alpha=0.6, label='Ground Truth', linewidth=1.5)
    ax.plot(t_h, df_hourly["soh_pc"], 'b-', alpha=0.7, label='PC', linewidth=1.0)
    ax.plot(t_h, df_hourly["soh_stm32"], 'r--', alpha=0.7, label='STM32', linewidth=1.0)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("SOH")
    ax.set_title("PC vs STM32 Overlay (Hourly)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error timeseries
    ax = axes[1, 1]
    error = df_hourly["soh_pc"] - df_hourly["soh_stm32"]
    ax.plot(t_h, error, 'purple', alpha=0.7, linewidth=1.0)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.fill_between(t_h, 0, error, alpha=0.3, color='purple')
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Error (PC - STM32)")
    ax.set_title(f"Prediction Error (MAE={mae*100:.2f}%)")
    ax.grid(True, alpha=0.3)
    
    # 5. Scatter PC vs ground truth
    ax = axes[2, 0]
    ax.scatter(df_valid["soh_true"], df_valid["soh_pc"], alpha=0.5, s=20, c='blue')
    lims = [df_valid[["soh_true", "soh_pc"]].min().min() - 0.01,
            df_valid[["soh_true", "soh_pc"]].max().max() + 0.01]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    ax.set_xlabel("Ground Truth SOH")
    ax.set_ylabel("PC Predicted SOH")
    ax.set_title(f"PC Accuracy (MAE={mae_pc_true*100:.2f}%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 6. Scatter STM32 vs ground truth
    ax = axes[2, 1]
    ax.scatter(df_valid["soh_true"], df_valid["soh_stm32"], alpha=0.5, s=20, c='red')
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    ax.set_xlabel("Ground Truth SOH")
    ax.set_ylabel("STM32 Predicted SOH")
    ax.set_title(f"STM32 Accuracy (MAE={mae_stm32_true*100:.2f}%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save
    out_dir = csv_path.parent
    out_png = out_dir / "pc_vs_stm32_HOURLY_comparison.png"
    out_json = out_dir / "pc_vs_stm32_HOURLY_metrics.json"
    
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {out_png}")
    
    # Save metrics
    metrics = {
        "n_hourly_total": len(df_hourly),
        "n_hourly_valid": n_valid,
        "mae_pc_stm32": float(mae),
        "rmse_pc_stm32": float(rmse),
        "max_error_pc_stm32": float(max_err),
        "mae_pc_true": float(mae_pc_true),
        "mae_stm32_true": float(mae_stm32_true),
    }
    
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out_json}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
