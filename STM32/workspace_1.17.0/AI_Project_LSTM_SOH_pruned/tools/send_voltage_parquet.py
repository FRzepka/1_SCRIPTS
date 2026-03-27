import sys
import time
import pathlib
import argparse

import serial
import pandas as pd

# Default config (can be overridden with CLI args)
PARQUET_PATH = r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C01.parquet"
PORT = "COM9"
BAUD = 115200
N_SAMPLES = 50           # how many rows to send
SLEEP_BETWEEN = 0.05     # seconds between sends
READ_TIMEOUT = 2.0      # seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default=PARQUET_PATH)
    parser.add_argument('--port', default=PORT)
    parser.add_argument('--baud', type=int, default=BAUD)
    parser.add_argument('--n', type=int, default=N_SAMPLES)
    parser.add_argument('--delay', type=float, default=SLEEP_BETWEEN)
    args = parser.parse_args()

    p = pathlib.Path(args.parquet)
    if not p.exists():
        print(f"Parquet not found: {p}")
        sys.exit(1)

    print(f"Loading parquet: {p}")
    df = pd.read_parquet(p)

    # Try to detect column name for voltage
    col_candidates = [
        "Voltage[V]", "Voltage", "voltage", "U", "u", "V",
    ]
    for c in col_candidates:
        if c in df.columns:
            voltage_col = c
            break
    else:
        print(f"No voltage column found in parquet. Available columns: {list(df.columns)[:10]} ...")
        sys.exit(2)

    ser = serial.Serial(args.port, args.baud, timeout=READ_TIMEOUT)
    time.sleep(0.2)

    # Drain any existing bytes
    ser.reset_input_buffer()

    print(f"Connected {args.port} @ {args.baud}. Sending {args.n} samples from column '{voltage_col}'.")
    print("Press RESET on the board once to see 'READY' in a terminal if desired.")

    # Read back any initial line
    try:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"<< {line}")
    except Exception:
        pass

    sent = 0
    for v in df[voltage_col].astype(float).head(args.n):
        msg = f"{v:.6f}\n"
        print(f"SENDING: {msg.strip()}")
        ser.write(msg.encode())
        ser.flush()
        # small delay to allow MCU to process
        time.sleep(0.05)

        # try to read response (may take up to READ_TIMEOUT)
        resp = ser.readline().decode(errors='ignore').strip()
        if not resp:
            # try again quickly
            time.sleep(0.05)
            resp = ser.readline().decode(errors='ignore').strip()

        print(f"-> {v:.6f} | <- {resp}")
        sent += 1
        time.sleep(args.delay)

    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()
