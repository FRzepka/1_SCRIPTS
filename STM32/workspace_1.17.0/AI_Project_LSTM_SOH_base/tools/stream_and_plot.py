#!/usr/bin/env python3
"""
Live plot: real SOH vs. predicted SOH from STM32.
Shows a rolling window of recent samples.
"""
import argparse
import re
import time
import unicodedata
from collections import deque
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import serial
from pandas.api.types import is_numeric_dtype


# Default features from train_soh.yaml (2.1.0.0).
DEFAULT_FEATURES = [
    "Testtime[s]",
    "Voltage[V]",
    "Current[A]",
    "Temperature[C]",
    "EFC",
    "Q_c",
]


def _column_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    key = re.sub(r"[^a-z0-9]+", "", text)
    if key.startswith("temperature"):
        return "temperature"
    return key


def _resolve_column_name(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested

    req_key = _column_key(requested)
    matches = [str(c) for c in df.columns if _column_key(str(c)) == req_key]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Column '{requested}' not found in parquet columns")
    raise ValueError(f"Column '{requested}' is ambiguous after normalization: {matches}")


def _resolve_requested_columns(df: pd.DataFrame, requested: List[str], need: int) -> List[str]:
    if len(requested) != need:
        raise ValueError(f"Please provide exactly {need} columns (got {len(requested)})")

    cols = [_resolve_column_name(df, c) for c in requested]
    bad = [c for c in cols if not is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Selected non-numeric columns: {bad}")
    return cols


def _parse_cols_arg(df: pd.DataFrame, cols_arg: str, need: int) -> List[str]:
    parts = [c.strip() for c in cols_arg.split(",") if c.strip()]
    cols: List[str] = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if idx < 0 or idx >= len(df.columns):
                raise ValueError(f"Column index {idx} out of range (0..{len(df.columns)-1})")
            cols.append(str(df.columns[idx]))
        else:
            cols.append(_resolve_column_name(df, p))
    return _resolve_requested_columns(df, cols, need)


def _try_yaml_features(yaml_path: Optional[str]) -> Optional[List[str]]:
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


def pick_columns(df: pd.DataFrame, need: int, cols_arg: Optional[str], yaml_path: Optional[str]) -> List[str]:
    feats = _try_yaml_features(yaml_path)
    if feats:
        try:
            return _resolve_requested_columns(df, feats, need)
        except ValueError:
            pass

    if cols_arg:
        return _parse_cols_arg(df, cols_arg, need)

    try:
        return _resolve_requested_columns(df, DEFAULT_FEATURES, need)
    except ValueError:
        pass

    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if len(num_cols) >= need:
        return num_cols[:need]
    raise ValueError(
        "Could not determine 6 feature columns. Provide --cols or --yaml.\n"
        f"Available columns: {list(df.columns)}"
    )


def pick_target_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["SOH", "SOH[%]", "soh", "StateOfHealth", "state_of_health"]:
        if candidate in df.columns:
            return candidate
    for candidate in ["SOC", "soc", "SOC[%]", "SoC"]:
        if candidate in df.columns:
            return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Live plot: real vs. predicted SOH from STM32")
    ap.add_argument("parquet", help="Path to parquet file")
    ap.add_argument("--port", required=True, help="Serial port e.g. COM8")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cols", help="Comma-separated column names or indices (exactly 6)")
    ap.add_argument("--yaml", help="Path to YAML config (reads model.features if present)")
    ap.add_argument("--start", type=int, default=0, help="Start row index")
    ap.add_argument("--n", type=int, default=None, help="Number of rows to send (default: all remaining rows)")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay after write (seconds, 0=no delay)")
    ap.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout (seconds)")
    ap.add_argument("--window", type=int, default=120, help="Rolling window size for plot")
    ap.add_argument("--window-seconds", type=float, help="Rolling window duration in seconds, converted with --delay")
    ap.add_argument("--fixed-soh-axis", action="store_true", help="Fix SOH y-axis instead of auto-scaling")
    ap.add_argument("--soh-ymin", type=float, default=0.0, help="Minimum SOH y-axis value")
    ap.add_argument("--soh-ymax", type=float, default=1.0, help="Maximum SOH y-axis value")
    args = ap.parse_args()

    window_size = max(args.window, 1)
    if args.window_seconds is not None:
        if args.delay <= 0:
            raise ValueError("--window-seconds needs --delay > 0 so seconds can be converted to samples")
        window_size = max(1, int(round(args.window_seconds / args.delay)))

    df = pd.read_parquet(args.parquet)
    try:
        cols = pick_columns(df, need=6, cols_arg=args.cols, yaml_path=args.yaml)
    except Exception as e:
        print(str(e))
        print("Hint: Use --yaml to point to train_soh.yaml or --cols to specify columns explicitly.")
        return

    if args.n is None:
        end = len(df)
    else:
        end = min(len(df), args.start + max(args.n, 0))

    if args.start >= len(df):
        raise ValueError(f"Start index {args.start} is beyond end of file (max: {len(df)-1})")
    if args.start >= end:
        raise ValueError("Empty selection: check --start/--n")

    target_col = pick_target_column(df)
    if not target_col:
        print("ERROR: No SOH/SOC target column found in parquet")
        print(f"Available columns: {list(df.columns)}")
        return
    target_label = "SOH" if "soh" in target_col.lower() else "SOC"

    print(f"Using columns: {cols}")
    print(f"Real {target_label} column: '{target_col}'")
    print(f"Total rows in file: {len(df)}")
    print(f"Streaming rows: {args.start} .. {end-1} (total {end-args.start} samples)")
    print(f"Rolling window: {window_size} samples")
    print("Starting live plot... (Press Ctrl+C to stop)")

    indices = deque(maxlen=window_size)
    real_values = deque(maxlen=window_size)
    pred_values = deque(maxlen=window_size)
    errors = deque(maxlen=window_size)

    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.2)
    ser.read(ser.in_waiting or 1).decode(errors="ignore")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"STM32 {target_label} Estimation - Real vs. Predicted", fontsize=14, fontweight="bold")

    line_real, = ax1.plot([], [], "b-", linewidth=2, label=f"Real {target_label}")
    line_pred, = ax1.plot([], [], "r--", linewidth=2, label=f"Predicted {target_label}")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel(target_label)
    ax1.set_title(f"{target_label} Comparison")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    if args.fixed_soh_axis:
        ax1.set_ylim(args.soh_ymin, args.soh_ymax)

    line_error, = ax2.plot([], [], "g-", linewidth=1.5, label="Error (Real - Pred)")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error")
    ax2.set_title("Prediction Error")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    current_idx = args.start
    try:
        while current_idx < end:
            idx = current_idx
            row_vals = df.loc[idx, cols].astype("float32").values.tolist()
            real_value = float(df.loc[idx, target_col])
            line = " ".join(f"{v:.6f}" for v in row_vals) + "\n"

            ser.write(line.encode("ascii"))
            ser.flush()
            if args.delay > 0:
                time.sleep(args.delay)

            deadline = time.time() + args.timeout
            predicted_value = None
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode(errors="ignore").strip()
                tag = None
                if "SOH:" in text:
                    tag = "SOH:"
                elif "SOC:" in text:
                    tag = "SOC:"
                if tag:
                    tag_idx = text.find(tag)
                    resp = text[tag_idx:]
                    try:
                        predicted_value = float(resp.split()[1])
                        break
                    except (IndexError, ValueError):
                        pass

            if predicted_value is not None:
                indices.append(idx)
                real_values.append(real_value)
                pred_values.append(predicted_value)
                errors.append(real_value - predicted_value)

                line_real.set_data(list(indices), list(real_values))
                line_pred.set_data(list(indices), list(pred_values))
                line_error.set_data(list(indices), list(errors))

                if len(indices) > 0:
                    if len(indices) < window_size:
                        ax1.set_xlim(args.start, args.start + window_size)
                        ax2.set_xlim(args.start, args.start + window_size)
                    else:
                        ax1.set_xlim(max(indices) - window_size + 1, max(indices) + 1)
                        ax2.set_xlim(max(indices) - window_size + 1, max(indices) + 1)

                    if args.fixed_soh_axis:
                        ax1.set_ylim(args.soh_ymin, args.soh_ymax)
                    else:
                        all_values = list(real_values) + list(pred_values)
                        if all_values:
                            val_min, val_max = min(all_values), max(all_values)
                            margin = (val_max - val_min) * 0.1 or 0.01
                            ax1.set_ylim(val_min - margin, val_max + margin)

                    if errors:
                        err_min, err_max = min(errors), max(errors)
                        err_margin = max(abs(err_min), abs(err_max)) * 0.1 or 0.01
                        ax2.set_ylim(err_min - err_margin, err_max + err_margin)

                if idx % 10 == 0:
                    plt.pause(0.001)

                print(
                    f"[{idx}] Real={real_value:.3f} | "
                    f"Pred={predicted_value:.3f} | "
                    f"Error={real_value - predicted_value:+.3f}"
                )
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

    if errors:
        import numpy as np

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        print(f"\n=== Statistics (last {len(errors)} samples) ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
