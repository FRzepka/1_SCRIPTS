#!/usr/bin/env python3
"""
Live-Plot: Real SOC vs. Predicted SOC from STM32
Shows a rolling window of the last 120 samples
"""
import argparse
import time
from typing import List
from collections import deque

import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Default features from train_soc.yaml (1.5.0.0)
DEFAULT_FEATURES = [
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "Q_c",
    "dU_dt[V/s]",
    "dI_dt[A/s]",
]


def _parse_cols_arg(df: pd.DataFrame, cols_arg: str, need: int) -> List[str]:
    parts = [c.strip() for c in cols_arg.split(",") if c.strip()]
    cols: List[str] = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if idx < 0 or idx >= len(df.columns):
                raise ValueError(f"Column index {idx} out of range (0..{len(df.columns)-1})")
            cols.append(df.columns[idx])
        else:
            if p not in df.columns:
                raise ValueError(f"Column '{p}' not found in parquet columns")
            cols.append(p)
    if len(cols) != need:
        raise ValueError(f"Please provide exactly {need} columns (got {len(cols)})")
    bad = [c for c in cols if not is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Selected non-numeric columns: {bad}")
    return cols


def _try_yaml_features(yaml_path: str | None) -> List[str] | None:
    if not yaml_path:
        return None
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        feats = data.get("model", {}).get("features")
        if isinstance(feats, list) and len(feats) == 6:
            return [str(x) for x in feats]
        return None
    except Exception:
        return None


def pick_columns(df: pd.DataFrame, need: int, cols_arg: str | None, yaml_path: str | None) -> List[str]:
    feats = _try_yaml_features(yaml_path)
    if feats:
        missing = [c for c in feats if c not in df.columns]
        if not missing:
            return feats
    if cols_arg:
        return _parse_cols_arg(df, cols_arg, need)
    if all(c in df.columns for c in DEFAULT_FEATURES):
        return DEFAULT_FEATURES
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if len(num_cols) >= need:
        return num_cols[:need]
    raise ValueError(
        "Could not determine 6 feature columns. Provide --cols or --yaml.\n"
        f"Available columns: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser(description="Live-Plot: Real vs. Predicted SOC from STM32")
    ap.add_argument("parquet", help="Path to parquet file")
    ap.add_argument("--port", required=True, help="Serial port e.g. COM9")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cols", help="Comma-separated column names or indices (exactly 6)")
    ap.add_argument("--yaml", help="Path to YAML config (reads model.features if present)")
    ap.add_argument("--start", type=int, default=0, help="Start row index")
    ap.add_argument("--n", type=int, default=None, help="Number of rows to send (default: all remaining rows)")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay after write (seconds, 0=no delay)")
    ap.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout (seconds)")
    ap.add_argument("--window", type=int, default=120, help="Rolling window size for plot")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    try:
        cols = pick_columns(df, need=6, cols_arg=args.cols, yaml_path=args.yaml)
    except Exception as e:
        print(str(e))
        print("Hint: Use --yaml to point to your train_soc.yaml or --cols to specify columns explicitly.")
        return
    
    # If --n not specified, go to end of file
    if args.n is None:
        end = len(df)
    else:
        end = min(len(df), args.start + max(args.n, 0))
    
    if args.start >= len(df):
        raise ValueError(f"Start index {args.start} is beyond end of file (max: {len(df)-1})")
    if args.start >= end:
        raise ValueError("Empty selection: check --start/--n")

    # Check if SOC column exists in parquet
    soc_col = None
    for candidate in ["SOC", "soc", "SOC[%]", "SoC"]:
        if candidate in df.columns:
            soc_col = candidate
            break

    if not soc_col:
        print("ERROR: No SOC column found in parquet (tried: SOC, soc, SOC[%], SoC)")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"Using columns: {cols}")
    print(f"Real SOC column: '{soc_col}'")
    print(f"Total rows in file: {len(df)}")
    print(f"Streaming rows: {args.start} .. {end-1} (total {end-args.start} samples)")
    print("Starting live plot... (Press Ctrl+C to stop)")

    # Data buffers
    indices = deque(maxlen=args.window)
    real_socs = deque(maxlen=args.window)
    pred_socs = deque(maxlen=args.window)
    errors = deque(maxlen=args.window)

    # Serial connection
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.2)
    # Drain boot messages
    boot = ser.read(ser.in_waiting or 1).decode(errors="ignore")

    # Setup plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('STM32 SOC Estimation - Real vs. Predicted', fontsize=14, fontweight='bold')
    
    line_real, = ax1.plot([], [], 'b-', linewidth=2, label='Real SOC')
    line_pred, = ax1.plot([], [], 'r--', linewidth=2, label='Predicted SOC')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('SOC')
    ax1.set_title('State of Charge Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    line_error, = ax2.plot([], [], 'g-', linewidth=1.5, label='Error (Real - Pred)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Streaming loop
    current_idx = args.start
    try:
        while current_idx < end:
            idx = current_idx
            row_vals = df.loc[idx, cols].astype("float32").values.tolist()
            real_soc = float(df.loc[idx, soc_col])
            line = " ".join(f"{v:.6f}" for v in row_vals) + "\n"

            # Send to STM32
            ser.write(line.encode())
            ser.flush()
            if args.delay > 0:
                time.sleep(args.delay)

            # Read response
            deadline = time.time() + args.timeout
            predicted_soc = None
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode(errors="ignore").strip()
                if "SOC:" in text:
                    soc_idx = text.find("SOC:")
                    resp = text[soc_idx:]
                    try:
                        predicted_soc = float(resp.split()[1])
                        break
                    except (IndexError, ValueError):
                        pass

            if predicted_soc is not None:
                # Add to buffers
                indices.append(idx)
                real_socs.append(real_soc)
                pred_socs.append(predicted_soc)
                errors.append(real_soc - predicted_soc)

                # Update plot
                line_real.set_data(list(indices), list(real_socs))
                line_pred.set_data(list(indices), list(pred_socs))
                line_error.set_data(list(indices), list(errors))
                
                # Sliding window: always show the last 'window' samples
                if len(indices) > 0:
                    # X-axis: scroll with the data (show last window_size samples)
                    if len(indices) < args.window:
                        # Still filling up - show from start
                        ax1.set_xlim(args.start, args.start + args.window)
                        ax2.set_xlim(args.start, args.start + args.window)
                    else:
                        # Sliding window - show last window samples
                        ax1.set_xlim(max(indices) - args.window + 1, max(indices) + 1)
                        ax2.set_xlim(max(indices) - args.window + 1, max(indices) + 1)
                    
                    # Y-axis: auto-scale based on visible data
                    all_socs = list(real_socs) + list(pred_socs)
                    if all_socs:
                        soc_min, soc_max = min(all_socs), max(all_socs)
                        margin = (soc_max - soc_min) * 0.1 or 0.01
                        ax1.set_ylim(soc_min - margin, soc_max + margin)
                    
                    if errors:
                        err_min, err_max = min(errors), max(errors)
                        err_margin = max(abs(err_min), abs(err_max)) * 0.1 or 0.01
                        ax2.set_ylim(err_min - err_margin, err_max + err_margin)

                # Only update plot every 10 samples for speed
                if idx % 10 == 0:
                    plt.pause(0.001)
                
                # Console output
                print(f"[{idx}] Real={real_soc:.3f} | Pred={predicted_soc:.3f} | Error={real_soc - predicted_soc:+.3f}")
            else:
                print(f"[{idx}] TIMEOUT - no response")

            current_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        ser.close()
        plt.ioff()
        plt.show()
        print("\nStream complete. Close plot window to exit.")

    # Show final statistics
    if len(errors) > 0:
        import numpy as np
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        print(f"\n=== Statistics (last {len(errors)} samples) ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
