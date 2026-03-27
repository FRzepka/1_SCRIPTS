#!/usr/bin/env python3
import argparse
import time
from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial

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
        import yaml  # type: ignore
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        feats = data.get("model", {}).get("features")
        if isinstance(feats, list) and len(feats) == 6:
            return [str(x) for x in feats]
        return None
    except Exception:
        # fallback silently; user can still pass --cols
        return None


def pick_columns(df: pd.DataFrame, need: int, cols_arg: str | None, yaml_path: str | None) -> List[str]:
    # 1) If YAML provided, try to use its model.features
    feats = _try_yaml_features(yaml_path)
    if feats:
        missing = [c for c in feats if c not in df.columns]
        if not missing:
            return feats
        # if YAML features not found, continue other strategies
    # 2) If --cols provided, honor it
    if cols_arg:
        return _parse_cols_arg(df, cols_arg, need)
    # 3) Try default features from config
    if all(c in df.columns for c in DEFAULT_FEATURES):
        return DEFAULT_FEATURES
    # 4) Fallback: first N numeric columns
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if len(num_cols) >= need:
        return num_cols[:need]
    # 5) Nothing worked – raise with diagnostics
    raise ValueError(
        "Could not determine 6 feature columns. Provide --cols or --yaml.\n"
        f"Available columns: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser(description="Stream 6 numeric features from a Parquet file to STM32 and print SOC response")
    ap.add_argument("parquet", help="Path to parquet file")
    ap.add_argument("--port", required=True, help="Serial port e.g. COM9")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cols", help="Comma-separated column names or indices (exactly 6). Example: 'Voltage[V],Current[A],Temperature[°C],Q_c,dU_dt[V/s],dI_dt[A/s]' or '0,1,2,3,4,5'")
    ap.add_argument("--yaml", help="Path to YAML config (reads model.features if present)")
    ap.add_argument("--start", type=int, default=0, help="Start row index")
    ap.add_argument("--n", type=int, default=50, help="Number of rows to send")
    ap.add_argument("--delay", type=float, default=0.05, help="Delay after write (seconds)")
    ap.add_argument("--timeout", type=float, default=2.0, help="Serial read timeout (seconds)")
    ap.add_argument("--step", action="store_true", help="Step-by-step mode: press Enter to send next row, 'q' to quit")
    ap.add_argument("--echo", action="store_true", help="Print all lines read from device (in addition to SOC)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    try:
        cols = pick_columns(df, need=6, cols_arg=args.cols, yaml_path=args.yaml)
    except Exception as e:
        print(str(e))
        print("Hint: Use --yaml to point to your train_soc.yaml or --cols to specify columns explicitly.")
        return
    end = min(len(df), args.start + max(args.n, 0))
    if args.start >= end:
        raise ValueError("Empty selection: check --start/--n")

    # Check if SOC column exists in parquet
    soc_col = None
    for candidate in ["SOC", "soc", "SOC[%]", "SoC"]:
        if candidate in df.columns:
            soc_col = candidate
            break

    print(f"Using columns: {cols}")
    if soc_col:
        print(f"Real SOC column: '{soc_col}' (type: {df[soc_col].dtype})")
        # Show first few values for debugging
        first_vals = df[soc_col].iloc[args.start:min(args.start+3, len(df))].tolist()
        print(f"First SOC values: {first_vals}")
    else:
        print("Warning: No SOC column found in parquet (tried: SOC, soc, SOC[%], SoC)")
        print(f"Available columns: {list(df.columns)}")
    print(f"Rows: {args.start} .. {end-1} (total {end-args.start})")

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)
        # Drain any boot messages
        boot = ser.read(ser.in_waiting or 1).decode(errors="ignore")
        if boot:
            print(boot.strip())

        for idx in range(args.start, end):
            row_vals = df.loc[idx, cols].astype("float32").values.tolist()
            real_soc = float(df.loc[idx, soc_col]) if soc_col else None
            line = " ".join(f"{v:.6f}" for v in row_vals) + "\n"

            if args.step:
                cmd = input(f"[{idx}] Send: {row_vals}? Press Enter, type 'q' to quit, or input 6 space-separated numbers to override: ").strip()
                if cmd.lower() == 'q':
                    print("Aborted by user.")
                    break
                if cmd:
                    try:
                        parts = [float(x) for x in cmd.split()]
                        if len(parts) != 6:
                            print("Please enter exactly 6 numbers or just press Enter.")
                            continue
                        row_vals = parts
                        line = " ".join(f"{v:.6f}" for v in row_vals) + "\n"
                    except ValueError:
                        print("Invalid numbers. Try again.")
                        continue

            ser.write(line.encode())
            ser.flush()
            time.sleep(args.delay)

            deadline = time.time() + args.timeout
            resp = None
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode(errors="ignore").strip()
                if args.echo and text:
                    print(f"[DEV] {text}")
                # Check if SOC: is embedded in the line (e.g., "...0.000000SOC: 0.411")
                if "SOC:" in text:
                    # Extract just the SOC part
                    soc_idx = text.find("SOC:")
                    resp = text[soc_idx:]
                    break
                if text.startswith("ERR"):
                    resp = text
                    break
            
            # Parse SOC value if found
            predicted_soc = None
            if resp and resp.startswith("SOC:"):
                try:
                    predicted_soc = float(resp.split()[1])
                except (IndexError, ValueError):
                    predicted_soc = resp
            elif resp:
                predicted_soc = resp
            
            # Format output with real vs predicted SOC
            if real_soc is not None:
                print(f"IN[{idx}]: Real SOC={real_soc:.3f} | Predicted SOC={predicted_soc} | Features={row_vals}")
            else:
                print(f"IN[{idx}]: Predicted SOC={predicted_soc} | Features={row_vals}")
            
            if predicted_soc is None:
                print("WARN: No SOC response within timeout")
                if args.step:
                    k = input("Continue? (Enter=yes, q=quit) ").strip().lower()
                    if k == 'q':
                        break

if __name__ == "__main__":
    main()
