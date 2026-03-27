#!/usr/bin/env python3
"""
STM32 SOC Host Benchmark (Quantized)
- Connects to flashed SOC firmware over serial
- Generates realistic inputs using scaler centers/scales from firmware
- Sends samples as fast as possible (or at user-specified rate)
- Parses `SOC:` and `METRICS:` lines produced by firmware
- Saves CSV results and a couple of plots for a paper-ready figure

Usage example (Windows cmd):
  conda activate ml1
  pip install -r STM32\benchmark\SOC_quantized\requirements.txt
  python STM32\benchmark\SOC_quantized\benchmark_stm32_soc_quantized.py --port COM7 --samples 5000

"""
import argparse
import serial
import time
import numpy as np
import os
import csv
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# These values were copied from firmware scaler_params.h (ROBUST scaler center/scale)
SCALER_CENTER = np.array([
    3.3605999947,
    0.6542999744,
    27.3999996185,
    -0.5109897852,
    0.0,
    0.0
], dtype=np.float32)
SCALER_SCALE = np.array([
    0.2009000778,
    2.6982000470,
    1.1000003815,
    0.5354322791,
    1.0,
    1.0
], dtype=np.float32)
INPUT_SIZE = 6

METRICS_RE = re.compile(r"METRICS:\s*cycles=(\d+)\s+us=([0-9.]+)\s+E_uJ=([0-9.]+)")
SOC_RE = re.compile(r"SOC:\s*([-]?\d+)\.(\d{1,3})")


def load_test_data(path, samples=None):
    """
    Load test data from CSV, Parquet or NPZ.
    Expected columns/keys: 'Voltage[V]', 'Current[A]', 'Temperature[degC]', 'Q_c', 'dU_dt', 'dI_dt'
    Target: 'SOC' or 'y'
    """
    print(f"Loading test data from {path}...")
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.npz':
        # NPZ handling is tricky if it doesn't have standard keys
        try:
            data = np.load(path, allow_pickle=True)
            # Check for known keys
            keys = list(data.keys())
            if 'features' in data and 'soc' in data:
                X = data['features']
                y = data['soc']
                return X[:samples], y[:samples]
            elif 'x' in data and 'y' in data:
                X = data['x']
                y = data['y']
                return X[:samples], y[:samples]
            else:
                print(f"ERROR: NPZ file {path} missing required keys ('features'/'soc' or 'x'/'y'). Found: {keys}")
                print("This file likely contains only results, not input features.")
                return None, None
        except Exception as e:
            print(f"ERROR loading NPZ: {e}")
            return None, None
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Map columns to standard 6 features
    # Try to find columns flexibly
    col_map = {
        'Voltage[V]': ['Voltage[V]', 'voltage', 'V'],
        'Current[A]': ['Current[A]', 'current', 'I'],
        'Temperature[degC]': ['Temperature[degC]', 'Temperature', 'temp', 'T', 'Temperature[°C]'],
        'Q_c': ['Q_c', 'capacity', 'Ah'],
        'dU_dt': ['dU_dt', 'dV_dt', 'dU_dt[V/s]'],
        'dI_dt': ['dI_dt', 'dI_dt[A/s]']
    }
    
    selected_cols = []
    for target_feat, candidates in col_map.items():
        found = False
        for c in candidates:
            if c in df.columns:
                selected_cols.append(c)
                found = True
                break
        if not found:
            # If dU_dt or dI_dt missing, fill with 0
            if target_feat in ['dU_dt', 'dI_dt']:
                print(f"Warning: {target_feat} not found, filling with 0")
                df[target_feat] = 0.0
                selected_cols.append(target_feat)
            else:
                raise ValueError(f"Missing required feature: {target_feat}. Available: {list(df.columns)}")

    # Target
    y = None
    for c in ['SOC', 'soc', 'y', 'SOC_ZHU']:
        if c in df.columns:
            y = df[c].values.astype(np.float32)
            break
            
    X = df[selected_cols].values.astype(np.float32)
    
    if samples:
        X = X[:samples]
        if y is not None:
            y = y[:samples]
            
    return X, y


def generate_samples(n, seed=None, center=SCALER_CENTER, scale=SCALER_SCALE):
    rng = np.random.default_rng(seed)
    # produce raw features as center + scale * N(0,1)
    raw = center + scale * rng.standard_normal((n, INPUT_SIZE), dtype=np.float32)
    # optional clipping to realistic ranges (Voltage 2.5-4.3, SOH 0-1, Temp maybe 0-60 etc.)
    raw[:,0] = np.clip(raw[:,0], 2.5, 4.3)   # Voltage
    raw[:,1] = np.clip(raw[:,1], -20.0, 20.0) # Current (A)
    raw[:,2] = np.clip(raw[:,2], -40.0, 85.0) # Temperature degC
    raw[:,3] = np.clip(raw[:,3], -2000.0, 2000.0) # Q_c like coulomb
    # dU/dt and dI/dt keep as-is
    return raw


def parse_soc_line(s):
    m = SOC_RE.search(s)
    if m:
        ones = int(m.group(1))
        frac = int(m.group(2))
        val = ones + (frac / 1000.0 if ones >= 0 else - (frac / 1000.0))
        return val
    # fallback: try float after colon
    if ':' in s:
        try:
            return float(s.split(':',1)[1].strip())
        except Exception:
            return None
    return None


def parse_metrics_line(s):
    m = METRICS_RE.search(s)
    if not m:
        return None
    return {
        'cycles': int(m.group(1)),
        'us': float(m.group(2)),
        'E_uJ': float(m.group(3))
    }


def extract_build_info(map_path, elf_path):
    info = {}
    try:
        info['elf_size_bytes'] = os.path.getsize(elf_path) if os.path.exists(elf_path) else None
    except Exception:
        info['elf_size_bytes'] = None
    # try to parse simple symbols from map: .data and .bss totals
    try:
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            # simple heuristics
            m_data = re.search(r"\.data\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)", txt)
            m_bss = re.search(r"\.bss\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)", txt)
            if m_data:
                info['data_bytes'] = int(m_data.group(1), 16)
            if m_bss:
                info['bss_bytes'] = int(m_bss.group(1), 16)
    except Exception:
        pass
    return info


def run_benchmark(port, baud, samples, rate_delay, timeout, outdir, seed=None, data_path=None):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'stm32_soc_bench.csv')
    summary_path = os.path.join(outdir, 'summary.json')

    y_true = None
    raw = None
    
    if data_path and os.path.exists(data_path):
        try:
            raw, y_true = load_test_data(data_path, samples)
        except Exception as e:
            print(f"Error loading data: {e}")
            raw = None
            
        if raw is not None:
            samples = len(raw) # Update samples count
            print(f"Loaded {samples} samples from {data_path}")
        else:
            print("Falling back to synthetic data generation.")
    
    if raw is None:
        if data_path:
            print(f"Warning: Data file {data_path} could not be loaded or missing features. Using synthetic data.")
        print("Generating synthetic samples...")
        raw = generate_samples(samples, seed=seed)

    # output containers
    rows = []
    metrics_records = []

    print(f"Connecting to {port} @ {baud}... (timeout={timeout}s)")
    with serial.Serial(port, baud, timeout=timeout) as ser:
        # small delay for device reset
        time.sleep(2.0)
        # drain startup lines until BOOT is seen or timeout
        boot_deadline = time.time() + 3.0
        found_boot = False
        while time.time() < boot_deadline:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print('<DEVICE>', line)
                if 'BOOT' in line:
                    found_boot = True
                    break
        if not found_boot:
            print('Warning: BOOT line not observed; proceeding anyway.')

        last_metrics = None
        sent_count = 0
        recv_count = 0
        lost_count = 0
        start_global = time.perf_counter()

        for i in tqdm(range(samples), desc="Benchmarking"):
            sample = raw[i]
            to_send = ' '.join(f'{v:.6f}' for v in sample) + '\n'
            t_send = time.perf_counter()
            try:
                ser.write(to_send.encode())
                ser.flush()
                sent_count += 1
            except Exception as e:
                print('Serial write error:', e)
                break

            # read lines up to timeout until SOC or timeout
            recv_ts = None; soc_val = None; parsed_metrics = None
            t_dead = time.perf_counter() + timeout
            while time.perf_counter() < t_dead:
                try:
                    s = ser.readline().decode(errors='ignore').strip()
                except Exception:
                    s = ''
                if not s:
                    continue
                # parse
                if s.startswith('SOC:'):
                    soc_val = parse_soc_line(s)
                    recv_ts = time.perf_counter()
                    recv_count += 1
                    # attach any last seen metrics as metadata if present
                    if last_metrics:
                        parsed_metrics = last_metrics
                    break
                elif s.startswith('METRICS:'):
                    m = parse_metrics_line(s)
                    if m:
                        # store metrics and also keep as last_metrics
                        metrics_records.append({'idx': i, 't': time.time(), **m})
                        last_metrics = m
                else:
                    # other device messages
                    pass

            if recv_ts is None:
                lost_count += 1
                rtt_ms = None
            else:
                rtt_ms = (recv_ts - t_send) * 1000.0

            row = {
                'idx': i,
                'send_ts': t_send,
                'recv_ts': recv_ts,
                'rtt_ms': rtt_ms,
                'soc_pred': soc_val,
                'metric_cycles': (parsed_metrics['cycles'] if parsed_metrics else None),
                'metric_us': (parsed_metrics['us'] if parsed_metrics else None),
                'metric_E_uJ': (parsed_metrics['E_uJ'] if parsed_metrics else None),
            }
            if y_true is not None:
                row['soc_true'] = float(y_true[i])
                if soc_val is not None:
                    row['error'] = soc_val - float(y_true[i])
                    row['abs_error'] = abs(row['error'])
            
            rows.append(row)

            # optional pacing
            if rate_delay and rate_delay > 0:
                time.sleep(rate_delay)

    total_time = time.perf_counter() - start_global
    print(f"Sent {sent_count} samples, received {recv_count}, lost {lost_count}, total_time={total_time:.3f}s")

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # compute stats
    rtts = df['rtt_ms'].dropna().astype(float)
    stats = {
        'samples_sent': int(sent_count),
        'samples_received': int(recv_count),
        'lost': int(lost_count),
        'total_time_s': float(total_time),
        'throughput_samples_per_s': float(recv_count) / total_time if total_time>0 else None,
        'rtt_mean_ms': float(rtts.mean()) if len(rtts)>0 else None,
        'rtt_median_ms': float(rtts.median()) if len(rtts)>0 else None,
        'rtt_p95_ms': float(rtts.quantile(0.95)) if len(rtts)>0 else None,
    }

    # Accuracy stats
    if y_true is not None and 'soc_pred' in df.columns:
        valid_preds = df.dropna(subset=['soc_pred', 'soc_true'])
        if not valid_preds.empty:
            mae = mean_absolute_error(valid_preds['soc_true'], valid_preds['soc_pred'])
            rmse = np.sqrt(mean_squared_error(valid_preds['soc_true'], valid_preds['soc_pred']))
            stats['mae'] = float(mae)
            stats['rmse'] = float(rmse)
            print(f"Accuracy: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # metrics aggregation
    metrics_df = pd.DataFrame(metrics_records)
    if not metrics_df.empty:
        stats['metrics_cycles_mean'] = float(metrics_df['cycles'].mean())
        stats['metrics_us_mean'] = float(metrics_df['us'].mean())
        stats['metrics_E_uJ_mean'] = float(metrics_df['E_uJ'].mean())
        # Total Energy Estimation for the sequence
        # Assuming each sample represents a time step. If we knew dt, we could integrate power.
        # Here we just sum the energy per inference.
        stats['total_inference_energy_mJ'] = float(metrics_df['E_uJ'].sum() / 1000.0)
    else:
        stats['metrics_cycles_mean'] = None
        stats['metrics_us_mean'] = None
        stats['metrics_E_uJ_mean'] = None

    # extract build info if available
    # Path relative to STM32/benchmark/SOC_quantized/
    map_path = os.path.join('..', '..', 'workspace_1.17.0', 'AI_Project_LSTM_SOC_quantized', 'Debug', 'AI_Project_LSTM_SOC_quantized.map')
    map_path = os.path.normpath(os.path.join(os.path.dirname(__file__), map_path))
    elf_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'workspace_1.17.0', 'AI_Project_LSTM_SOC_quantized', 'Debug', 'AI_Project_LSTM_SOC_quantized.elf'))
    build_info = extract_build_info(map_path, elf_path)
    stats['build'] = build_info

    # save summary
    with open(summary_path, 'w') as f:
        json.dump({'stats': stats, 'timestamp': datetime.utcnow().isoformat(), 'samples_file': os.path.basename(csv_path)}, f, indent=2)

    # produce simple plots
    # 1. Histogram
    fig1 = plt.figure(figsize=(6,4))
    if len(rtts)>0:
        plt.hist(rtts, bins=80, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.xlabel('Round-trip Latency (ms)')
        plt.ylabel('Count')
        plt.title(f'STM32 SOC Inference Latency (N={len(rtts)})')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig1_path = os.path.join(outdir, 'latency_hist.png')
        fig1.savefig(fig1_path, dpi=300)
        print('Saved', fig1_path)

    # 2. Time Series
    fig2 = plt.figure(figsize=(10,4))
    if len(rtts)>0:
        plt.plot(rtts.values, linewidth=0.5, color='#1f77b4')
        plt.xlabel('Sample Index')
        plt.ylabel('Latency (ms)')
        plt.title('Inference Latency Stability')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig2_path = os.path.join(outdir, 'latency_timeseries.png')
        fig2.savefig(fig2_path, dpi=300)
        print('Saved', fig2_path)

    # 3. Boxplot (Paper style)
    fig3 = plt.figure(figsize=(4,5))
    if len(rtts)>0:
        plt.boxplot(rtts, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#aec7e8', color='black'),
                    medianprops=dict(color='red'))
        plt.ylabel('Latency (ms)')
        plt.title('Latency Distribution')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig3_path = os.path.join(outdir, 'latency_boxplot.png')
        fig3.savefig(fig3_path, dpi=300)
        print('Saved', fig3_path)

    # 4. Device Metrics (if available)
    if not metrics_df.empty:
        fig4 = plt.figure(figsize=(8,4))
        ax1 = fig4.add_subplot(111)
        ax1.plot(metrics_df['us'].values, color='#ff7f0e', marker='.', linestyle='-', markersize=3, label='Inference Time (us)')
        ax1.set_xlabel('Metric Sample')
        ax1.set_ylabel('Device Time (us)', color='#ff7f0e')
        ax1.tick_params(axis='y', labelcolor='#ff7f0e')
        
        ax2 = ax1.twinx()
        ax2.plot(metrics_df['E_uJ'].values, color='#9467bd', marker='x', linestyle='--', markersize=3, label='Energy (uJ)')
        ax2.set_ylabel('Est. Energy (uJ)', color='#9467bd')
        ax2.tick_params(axis='y', labelcolor='#9467bd')
        
        plt.title('On-Device Performance Metrics')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig4_path = os.path.join(outdir, 'device_metrics.png')
        fig4.savefig(fig4_path, dpi=300)
        print('Saved', fig4_path)

    # 5. Pred vs GT (if available)
    if y_true is not None and 'soc_pred' in df.columns:
        valid_preds = df.dropna(subset=['soc_pred', 'soc_true'])
        if not valid_preds.empty:
            fig5 = plt.figure(figsize=(10,5))
            plt.plot(valid_preds['idx'], valid_preds['soc_true'], label='Ground Truth', color='black', linewidth=1.0)
            plt.plot(valid_preds['idx'], valid_preds['soc_pred'], label='STM32 Prediction', color='red', linestyle='--', linewidth=1.0, alpha=0.8)
            plt.xlabel('Sample Index')
            plt.ylabel('SOC')
            plt.title(f'Prediction vs Ground Truth (MAE={stats.get("mae",0):.4f})')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            fig5_path = os.path.join(outdir, 'pred_vs_gt.png')
            fig5.savefig(fig5_path, dpi=300)
            print('Saved', fig5_path)

            # Error distribution
            fig6 = plt.figure(figsize=(6,4))
            plt.hist(valid_preds['error'], bins=50, color='purple', alpha=0.7)
            plt.xlabel('Prediction Error (Pred - True)')
            plt.ylabel('Count')
            plt.title('Error Distribution')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            fig6_path = os.path.join(outdir, 'error_hist.png')
            fig6.savefig(fig6_path, dpi=300)
            print('Saved', fig6_path)

    # Generate Paper Markdown Table
    md_table = f"""
### Benchmark Results Summary (Quantized)

| Metric | Value | Unit |
| :--- | :--- | :--- |
| **Host Latency (Mean)** | {stats['rtt_mean_ms']:.3f} | ms |
| **Host Latency (P95)** | {stats['rtt_p95_ms']:.3f} | ms |
| **Device Inference Time** | {stats['metrics_us_mean'] if stats['metrics_us_mean'] is not None else 'N/A'} | µs |
| **Device Energy (Est.)** | {stats['metrics_E_uJ_mean'] if stats['metrics_E_uJ_mean'] is not None else 'N/A'} | µJ |
| **Throughput** | {stats['throughput_samples_per_s']:.1f} | Hz |
| **MAE** | {stats.get('mae', 'N/A')} | - |
| **RMSE** | {stats.get('rmse', 'N/A')} | - |
| **Flash Usage (.text)** | {build_info.get('elf_size_bytes', 'N/A')} | Bytes (approx ELF) |
| **RAM Usage (.data+.bss)** | {build_info.get('data_bytes', 0) + build_info.get('bss_bytes', 0)} | Bytes |

*Note: RAM usage is static allocation estimated from map file. Dynamic heap usage is not included.*
"""
    md_path = os.path.join(outdir, 'paper_summary.md')
    with open(md_path, 'w') as f:
        f.write(md_table)
    print('Paper summary saved to', md_path)

    print('Summary:')
    for k,v in stats.items():
        print(f"  {k}: {v}")
    print('CSV saved to', csv_path)
    print('Summary JSON saved to', summary_path)
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM7', help='Serial port (default: COM7)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--samples', type=int, default=5000)
    ap.add_argument('--rate', type=float, default=0.0, help='Inter-sample delay in seconds (0 -> as fast as possible)')
    ap.add_argument('--timeout', type=float, default=1.0, help='Serial read timeout (s)')
    ap.add_argument('--outdir', default=None, help='Output directory (defaults to DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOC/quantized/results_TIMESTAMP)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for synthetic inputs')
    ap.add_argument('--data', default=None, help='Path to test data file (CSV/Parquet/NPZ) with features and SOC target')
    args = ap.parse_args()
    
    # Default data path if not provided
    if args.data is None:
        # Updated default path to MGFarm C07 parquet
        default_data = r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet"
        if os.path.exists(default_data):
            print(f"Using default data file: {default_data}")
            args.data = default_data
        else:
            # Fallback to old path just in case, or warn
            old_default = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DL_Models', 'LFP_LSTM_MLP', '5_benchmark', 'bench_v_soc_full', 'soc_streaming_base_quant_pruned_data.npz')
            if os.path.exists(old_default):
                 print(f"Warning: Primary default data not found. Trying fallback: {old_default}")
                 args.data = old_default

    if args.outdir is None:
        # Default output path as requested
        base_out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DL_Models', 'LFP_LSTM_MLP', '5_benchmark', 'STM32', 'SOC', 'quantized')
        # Ensure absolute path if running from root
        if not os.path.isabs(base_out):
             base_out = os.path.abspath(base_out)
        
        args.outdir = os.path.join(base_out, 'results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        
    run_benchmark(args.port, args.baud, args.samples, args.rate, args.timeout, args.outdir, seed=(None if args.seed==0 else args.seed), data_path=args.data)
