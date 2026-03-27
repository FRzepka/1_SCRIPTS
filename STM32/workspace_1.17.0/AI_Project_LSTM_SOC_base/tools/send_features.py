#!/usr/bin/env python3
import argparse
import serial
import time
import random


def main():
    p = argparse.ArgumentParser(description="Send 6 float features to STM32 over UART and print SOC reply")
    p.add_argument("port", help="COM port, e.g. COM9")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--n", type=int, default=10, help="number of lines to send")
    p.add_argument("--delay", type=float, default=0.05, help="delay after write (s)")
    p.add_argument("--features", nargs=6, type=float, metavar=("F0","F1","F2","F3","F4","F5"),
                   help="six space-separated floats to send each line")
    p.add_argument("--random", action="store_true", help="send random features in [0,1)")
    args = p.parse_args()

    if not args.features and not args.random:
        print("Provide --features f0 f1 f2 f3 f4 f5 or use --random")
        return

    with serial.Serial(args.port, args.baud, timeout=2) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        # read any READY/AI:OK
        time.sleep(0.2)
        try:
            boot = ser.read(ser.in_waiting or 1).decode(errors="ignore")
            if boot:
                print(boot.strip())
        except Exception:
            pass
        for i in range(args.n):
            if args.random:
                feats = [random.random() for _ in range(6)]
            else:
                feats = args.features
            line = " ".join(f"{v:.6f}" for v in feats) + "\n"
            ser.write(line.encode())
            ser.flush()
            time.sleep(args.delay)
            resp = ser.readline().decode(errors="ignore").strip()
            print(f"-> {line.strip()} | <- {resp}")

if __name__ == "__main__":
    main()
