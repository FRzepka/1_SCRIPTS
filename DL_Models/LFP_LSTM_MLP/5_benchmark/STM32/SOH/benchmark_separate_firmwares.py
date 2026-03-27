import serial
import time
import json
import os
import re
import numpy as np
import sys

# Configuration
COM_PORT = 'COM7' # Default, can be changed
BAUD_RATE = 115200
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = ['Base', 'Pruned', 'Quantized']

def find_com_port():
    """Helper to find STM32 COM port automatically or ask user"""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    print("Available COM ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    
    # Try to guess or use default
    for p in ports:
        if "STLink" in p.description or "STM32" in p.description:
            return p.device
    
    return input(f"Enter COM port (default {COM_PORT}): ") or COM_PORT

def run_benchmark(model_name, ser):
    print(f"\n--- Benchmarking {model_name} Model ---")
    print("1. Please flash the firmware for this model if not already done.")
    print("2. Press the RESET button on the STM32 board to capture boot metrics.")
    print("Waiting for boot message...")

    metrics = {
        'static_ram': 0,
        'stack_ram': [],
        'cycles': [],
        'time_us': [],
        'energy_uj': [],
        'predictions': []
    }

    boot_captured = False
    
    # Wait for boot message and static RAM info
    while not boot_captured:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"STM32: {line}")
                if "RAM_MEASURE" in line:
                    # Parse: RAM_MEASURE: Static=1234 (Data=..., BSS=...), StackTop=..., SP=...
                    m = re.search(r'Static=(\d+)', line)
                    if m:
                        metrics['static_ram'] = int(m.group(1))
                        print(f"✅ Captured Static RAM: {metrics['static_ram']} bytes")
                        boot_captured = True
        except serial.SerialException:
            print("Serial error, retrying...")
            time.sleep(1)

    print("Starting inference loop (100 samples)...")
    
    # Send random test data
    # Input size is 6 (Base/Pruned) or 6 (Quantized) - actually Quantized uses 6 too.
    # Format: "v1 v2 v3 v4 v5 v6"
    
    for i in range(100):
        # Generate random inputs (normalized-ish range)
        inputs = np.random.rand(6).astype(np.float32)
        input_str = " ".join([f"{x:.4f}" for x in inputs]) + "\n"
        
        ser.write(input_str.encode())
        
        # Read response
        # Expecting: METRICS: ... and SOH: ...
        got_metrics = False
        got_pred = False
        
        start_wait = time.time()
        while (not got_metrics or not got_pred) and (time.time() - start_wait < 2.0):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            # print(f"RX: {line}") # Debug
            
            if "METRICS:" in line:
                # METRICS: cycles=123 us=456.7 E_uJ=89.0 Stack=100
                m_cyc = re.search(r'cycles=(\d+)', line)
                m_us = re.search(r'us=([\d\.]+)', line)
                m_e = re.search(r'E_uJ=([\d\.]+)', line)
                m_stk = re.search(r'Stack=(\d+)', line)
                
                if m_cyc: metrics['cycles'].append(int(m_cyc.group(1)))
                if m_us: metrics['time_us'].append(float(m_us.group(1)))
                if m_e: metrics['energy_uj'].append(float(m_e.group(1)))
                if m_stk: metrics['stack_ram'].append(int(m_stk.group(1)))
                got_metrics = True
                
            if "SOH:" in line:
                metrics['predictions'].append(line)
                got_pred = True
        
        if i % 10 == 0:
            print(f"Sample {i}/100 done...")

    # Aggregate results
    summary = {
        'model': model_name,
        'static_ram_bytes': metrics['static_ram'],
        'max_stack_bytes': max(metrics['stack_ram']) if metrics['stack_ram'] else 0,
        'total_ram_bytes': metrics['static_ram'] + (max(metrics['stack_ram']) if metrics['stack_ram'] else 0),
        'avg_cycles': float(np.mean(metrics['cycles'])) if metrics['cycles'] else 0,
        'avg_time_us': float(np.mean(metrics['time_us'])) if metrics['time_us'] else 0,
        'avg_energy_uj': float(np.mean(metrics['energy_uj'])) if metrics['energy_uj'] else 0
    }
    
    print(f"\n--- Results for {model_name} ---")
    print(json.dumps(summary, indent=2))
    
    return summary

def main():
    port = find_com_port()
    print(f"Opening {port}...")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    all_results = {}
    
    for model in MODELS:
        while True:
            print(f"\n\nReady to benchmark: {model}")
            print("Action required: Flash the STM32 with the " + model.upper() + " firmware.")
            resp = input("Press ENTER when ready (or 's' to skip, 'q' to quit): ").lower()
            if resp == 'q':
                ser.close()
                return
            if resp == 's':
                print(f"Skipping {model}...")
                break
            
            try:
                result = run_benchmark(model, ser)
                all_results[model] = result
                
                # Save individual result
                with open(os.path.join(OUTPUT_DIR, f"result_{model.lower()}.json"), 'w') as f:
                    json.dump(result, f, indent=2)
                
                break # Move to next model
            except KeyboardInterrupt:
                print("\nInterrupted.")
                ser.close()
                return
            except Exception as e:
                print(f"Error during benchmark: {e}")
                retry = input("Retry? (y/n): ")
                if retry.lower() != 'y':
                    break

    ser.close()
    
    # Save combined results
    if all_results:
        print("\n\n=== FINAL SUMMARY ===")
        print(f"{'Model':<12} | {'Static RAM':<10} | {'Stack RAM':<10} | {'Total RAM':<10} | {'Time (us)':<10} | {'Energy (uJ)':<10}")
        print("-" * 80)
        for name, res in all_results.items():
            print(f"{name:<12} | {res['static_ram_bytes']:<10} | {res['max_stack_bytes']:<10} | {res['total_ram_bytes']:<10} | {res['avg_time_us']:<10.1f} | {res['avg_energy_uj']:<10.3f}")
        
        with open(os.path.join(OUTPUT_DIR, "benchmark_summary.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
