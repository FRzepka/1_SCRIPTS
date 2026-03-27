import serial
import time
import json
import os
import re
import numpy as np
import sys
import serial.tools.list_ports

# Configuration
COM_PORT = "COM7"
BAUD_RATE = 115200
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_com_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STLink" in p.description or "STM32" in p.description:
            return p.device
    return COM_PORT


def run_benchmark(model_name, n_samples=1000):
    port = find_com_port()
    print(f"Connecting to {port} for model: {model_name}...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    print(f"\n--- Benchmarking {model_name} ---")

    # Auto-Reset via DTR (ASCII only)
    print("Triggering Auto-Reset via DTR...")
    ser.dtr = False
    time.sleep(0.1)
    ser.dtr = True
    time.sleep(0.1)

    metrics = {
        "static_ram": 0,
        "stack_ram": [],
        "cycles": [],
        "time_us": [],
        "host_latency_ms": [],
        "energy_uj": [],
        "predictions": [],
    }

    boot_captured = False
    timeout_start = time.time()

    # Wait for boot message
    print("Waiting for boot message...")
    while not boot_captured:
        if time.time() - timeout_start > 10:
            print("Warning: No boot message received. Using default Static RAM = 0.")
            boot_captured = True
            break

        try:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line and "RAM_MEASURE" in line:
                    print(f"STM32: {line}")
                    m = re.search(r"Static=(\d+)", line)
                    if m:
                        metrics["static_ram"] = int(m.group(1))
                        print(f"Captured Static RAM: {metrics['static_ram']} bytes")
                        boot_captured = True
        except Exception as e:
            print(f"Serial error: {e}")
            time.sleep(0.1)

    print(f"Starting inference loop ({n_samples} samples)...")

    for i in range(n_samples):
        inputs = np.random.rand(6).astype(np.float32)
        input_str = " ".join([f"{x:.4f}" for x in inputs]) + "\n"

        # Host-side latency start: from sending the sample until both
        # METRICS and SOC prediction have been received.
        t_start = time.time()
        ser.write(input_str.encode())

        got_metrics = False
        got_pred = False
        start_wait = time.time()

        while (not got_metrics or not got_pred) and (time.time() - start_wait < 2.0):
            if ser.in_waiting:
                line = ser
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if "METRICS:" in line:
                    m_cyc = re.search(r"cycles=(\d+)", line)
                    m_us = re.search(r"us=([\d\.]+)", line)
                    m_e = re.search(r"E_uJ=([\d\.]+)", line)
                    m_stk = re.search(r"Stack=(\d+)", line)
                    m_static = re.search(r"Static=(\d+)", line)

                    if m_cyc:
                        metrics["cycles"].append(int(m_cyc.group(1)))
                    if m_us:
                        metrics["time_us"].append(float(m_us.group(1)))
                    if m_e:
                        metrics["energy_uj"].append(float(m_e.group(1)))
                    if m_stk:
                        metrics["stack_ram"].append(int(m_stk.group(1)))
                    if m_static:
                        # Prefer Static from METRICS lines (emitted every inference),
                        # fallback to boot-time RAM_MEASURE if necessary.
                        static_val = int(m_static.group(1))
                        if static_val > 0:
                            metrics["static_ram"] = static_val
                    got_metrics = True

                if "SOC:" in line:
                    metrics["predictions"].append(line)
                    got_pred = True

        # Record host-side latency only if we received both metrics and prediction.
        if got_metrics and got_pred:
            latency_ms = (time.time() - t_start) * 1000.0
            metrics["host_latency_ms"].append(latency_ms)

        if i % max(1, n_samples // 10) == 0:
            print(f"Sample {i}/{n_samples}...")

    ser.close()

    summary = {
        "model": model_name,
        "static_ram_bytes": metrics["static_ram"],
        "max_stack_bytes": max(metrics["stack_ram"]) if metrics["stack_ram"] else 0,
        "total_ram_bytes": metrics["static_ram"]
        + (max(metrics["stack_ram"]) if metrics["stack_ram"] else 0),
        "avg_cycles": float(np.mean(metrics["cycles"])) if metrics["cycles"] else 0.0,
        "avg_time_us": float(np.mean(metrics["time_us"])) if metrics["time_us"] else 0.0,
        "avg_host_latency_ms": float(np.mean(metrics["host_latency_ms"])) if metrics["host_latency_ms"] else 0.0,
        "avg_energy_uj": float(np.mean(metrics["energy_uj"])) if metrics["energy_uj"] else 0.0,
        "raw_metrics": metrics,
    }

    print(f"\n--- Results for {model_name} ---")
    # Print only a compact summary to avoid flooding the console for large n_samples
    compact = {
        "model": summary["model"],
        "n_samples": len(metrics["time_us"]),
        "avg_time_us": summary["avg_time_us"],
        "avg_host_latency_ms": summary.get("avg_host_latency_ms", 0.0),
        "avg_energy_uj": summary["avg_energy_uj"],
    }
    print(json.dumps(compact, indent=2))

    outfile = os.path.join(OUTPUT_DIR, f"result_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_benchmark.py <ModelName> [NSamples]")
        sys.exit(1)

    model_name = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            n_samples = int(sys.argv[2])
        except ValueError:
            print(f"Invalid sample count '{sys.argv[2]}', using default 1000")
            n_samples = 1000
    else:
        n_samples = 1000

    run_benchmark(model_name, n_samples)
