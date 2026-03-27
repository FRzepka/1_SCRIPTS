#!/usr/bin/env python3
import argparse
import time
from typing import List

import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial
from tqdm.auto import tqdm

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
    ap = argparse.ArgumentParser(description="Stream 6 numeric features from a Parquet file to STM32 and print SOC/SOH response")
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
    ap.add_argument("--echo", action="store_true", help="Print all lines read from device (in addition to SOC/SOH)")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars for send/receive")
    ap.add_argument("--out-dir", default="", help="Optional directory to save CSV log and device raw output")
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

    # Check if SOH/SOC column exists in parquet
    y_col = None
    for candidate in ["SOH", "SOH[%]", "soh", "SOC", "soc", "SOC[%]", "SoC"]:
        if candidate in df.columns:
            y_col = candidate
            break

    print(f"Using columns: {cols}")
    if y_col:
        print(f"Real Y column (SOC/SOH): '{y_col}' (type: {df[y_col].dtype})")
        # Show first few values for debugging
        first_vals = df[y_col].iloc[args.start:min(args.start+3, len(df))].tolist()
        print(f"First Y values: {first_vals}")
    else:
        print("Warning: No SOC/SOH column found in parquet (tried common names)")
        print(f"Available columns: {list(df.columns)}")
    print(f"Rows: {args.start} .. {end-1} (total {end-args.start})")

    with serial.Serial(args.port, args.baud, timeout=args.timeout, write_timeout=args.timeout) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)
        # Drain any boot messages
        boot = ser.read(ser.in_waiting or 1).decode(errors="ignore")
        if boot:
            print(boot.strip())

        pbar = tqdm(total=(end - args.start), unit="row", desc="send", disable=not args.progress)
        rbar = tqdm(total=(end - args.start), unit="resp", desc="recv", disable=not args.progress)
        raw_lines = []
        records = []

        for idx in range(args.start, end):
            row_vals = df.loc[idx, cols].astype("float32").values.tolist()
            real_y = float(df.loc[idx, y_col]) if y_col else None
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

            try:
                ser.write(line.encode("ascii"))
                ser.flush()
            except Exception as e:
                print(f"[serial] write failed at idx={idx}: {e}")
                break
            time.sleep(args.delay)
            if args.progress:
                pbar.update(1)

            deadline = time.time() + args.timeout
            resp = None
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode(errors="ignore").strip()
                if args.echo and text:
                    print(f"[DEV] {text}")
                if text:
                    raw_lines.append(text)
                # Check if SOC:/SOH: embedded in the line (e.g., "... SOC: 0.411" or "SOH: 0.98")
                int_tag = None
                if "SOC:" in text:
                    int_tag = "SOC:"
                elif "SOH:" in text:
                    int_tag = "SOH:"
                if int_tag:
                    idx = text.find(int_tag)
                    resp = text[idx:]
                    break
                if text.startswith("ERR"):
                    resp = text
                    break
            
            # Parse predicted value if found
            predicted_val = None
            if resp and (resp.startswith("SOC:") or resp.startswith("SOH:")):
                try:
                    predicted_val = float(resp.split()[1])
                except (IndexError, ValueError):
                    predicted_val = resp
            elif resp:
                predicted_val = resp
            
            # Format output with real vs predicted SOC
            if real_y is not None:
                print(f"IN[{idx}]: Real Y={real_y:.3f} | Pred={predicted_val} | Features={row_vals}")
            else:
                print(f"IN[{idx}]: Pred={predicted_val} | Features={row_vals}")

            # persist record in memory
            rec = {"idx": int(idx), "pred": predicted_val}
            if real_y is not None:
                rec["y_true"] = real_y
            for i, v in enumerate(row_vals):
                rec[f"x{i}"] = float(v)
            records.append(rec)

            if predicted_val is None:
                print("WARN: No SOC response within timeout")
                if args.step:
                    k = input("Continue? (Enter=yes, q=quit) ").strip().lower()
                    if k == 'q':
                        break
            else:
                if args.progress:
                    rbar.update(1)

        if args.progress:
            pbar.close(); rbar.close()

        # optional saving
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            # write device log
            with open(os.path.join(args.out_dir, 'device_log.txt'), 'w', encoding='utf-8') as f:
                for ln in raw_lines:
                    f.write(ln+"\n")
            # write records csv
            try:
                pd.DataFrame.from_records(records).to_csv(os.path.join(args.out_dir, 'stream_records.csv'), index=False)
                print(f"Saved stream to {os.path.join(args.out_dir, 'stream_records.csv')}")
            except Exception as e:
                print(f"[save] warn: could not write CSV: {e}")

if __name__ == "__main__":
    main()
