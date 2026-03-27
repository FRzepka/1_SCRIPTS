import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def find_cell_parquet(data_root: str, cell: str) -> str:
    """Locate Parquet file for a given cell under a data root.
    Tries several naming conventions used in this repo.
    """
    # Common patterns
    candidates = [
        os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet"),  # C07
        os.path.join(data_root, f"df_FE_{cell}.parquet"),                 # full name
    ]
    # Try numbered suffix (Cxx)
    cid = cell[-3:]
    candidates.append(os.path.join(data_root, f"df_FE_C{cid}.parquet"))

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Parquet for {cell} not found under {data_root}. Tried: {candidates}")


def infer_dt_hours(df: pd.DataFrame) -> np.ndarray:
    """Return an array of dt in hours between samples (same length as df, first elt = 0).
    Prefers existing 'delta_time_h', else diffs of 'Testtime[s]' or timestamp.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    if 'delta_time_h' in df.columns:
        dth = pd.to_numeric(df['delta_time_h'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        # Ensure non-negative and start at 0 for the first sample
        dth = np.maximum(dth, 0.0)
        if dth.size:
            dth[0] = 0.0
        return dth

    # Try Testtime[s]
    for col in ['Testtime[s]', 'Time[s]']:
        if col in df.columns:
            t = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
            dt = np.diff(t, prepend=t[0]) / 3600.0
            dt = np.where(np.isfinite(dt), dt, 0.0)
            dt = np.maximum(dt, 0.0)
            return dt

    # Try absolute timestamp
    ts_col = 'Absolute_Time[yyyy-mm-dd hh:mm:ss]'
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors='coerce')
        t = ts.view('int64') / 1e9  # seconds
        dt = np.diff(t, prepend=t.iloc[0]) / 3600.0
        dt = np.where(np.isfinite(dt), dt, 0.0)
        dt = np.maximum(dt, 0.0)
        return dt

    # Fallback: try per-step time if present
    if 'Step_Time[s]' in df.columns:
        st = pd.to_numeric(df['Step_Time[s]'], errors='coerce').to_numpy(dtype=float)
        dt = np.diff(st, prepend=st[0]) / 3600.0
        dt = np.where(np.isfinite(dt), dt, 0.0)
        dt = np.maximum(dt, 0.0)
        return dt

    raise ValueError("Could not infer time delta. Expected one of: 'delta_time_h', 'Testtime[s]', 'Time[s]', absolute timestamp, or 'Step_Time[s]'.")


def compute_qsum(df: pd.DataFrame, vmax_tol: float = 0.01) -> pd.Series:
    """Compute cumulative charge (Ah) from signed current and dt with resets.
    - Signed accumulation: inc = I[A] * dt[h]
    - Reset behavior: whenever Voltage[V] is near its max (>= max - tol), set Q_sum to 0
      and continue accumulating from 0 afterwards.
    Returns a Series aligned with df index named 'Q_sum_recalc'.
    """
    cur_col = 'Current[A]'
    if cur_col not in df.columns:
        # fallback commonly seen in this repo
        if 'DC_Current[A]' in df.columns:
            cur_col = 'DC_Current[A]'
        else:
            raise KeyError("Neither 'Current[A]' nor 'DC_Current[A]' found in dataframe")
    I = pd.to_numeric(df[cur_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    if 'Voltage[V]' not in df.columns:
        raise KeyError("Column 'Voltage[V]' not found in dataframe for reset detection")
    V = pd.to_numeric(df['Voltage[V]'], errors='coerce').to_numpy(dtype=float)
    dt_h = infer_dt_hours(df)
    if len(I) != len(dt_h):
        raise ValueError("Length mismatch between current and dt")
    # Signed accumulation (no absolute), so positive/negative contributions cancel
    inc = I * dt_h  # Ah per step (signed)
    cum = np.cumsum(inc)

    # Reset when Voltage is near its maximum
    vmax = np.nanmax(V)
    if not np.isfinite(vmax):
        vmax = np.nan
    is_high = np.isfinite(V) & (V >= (vmax - float(vmax_tol)))
    # Mark only the leading edge of high-voltage region as a reset event
    prev = np.zeros_like(is_high, dtype=bool)
    if is_high.size:
        prev[1:] = is_high[:-1]
    reset_at = is_high & (~prev)

    base = np.full_like(cum, np.nan, dtype=float)
    base[reset_at] = cum[reset_at]
    # Forward-fill base; NaN -> 0 before first reset
    base_ffill = pd.Series(base).ffill().fillna(0.0).to_numpy(dtype=float)
    qsum = cum - base_ffill
    # Hold exactly zero during the high-voltage plateau
    qsum[is_high] = 0.0
    # Prevent positive values: clamp to 0 (ceiling)
    qsum = np.minimum(qsum, 0.0)
    return pd.Series(qsum, index=df.index, name='Q_sum_recalc')


def main():
    ap = argparse.ArgumentParser(description="Recalculate Q_sum from current and compare with Q_c")
    ap.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'train_soc.yaml'), help='Path to SOC training YAML to get data_root (optional)')
    ap.add_argument('--data-root', type=str, default=None, help='Override data root containing df_FE_*.parquet')
    ap.add_argument('--cell', type=str, default='MGFarm_18650_C07', help='Cell identifier (e.g., MGFarm_18650_C07)')
    ap.add_argument('--out', type=str, default=None, help='Output plot path (PNG). Default: under 1.5.0.0/outputs')
    args = ap.parse_args()

    data_root = args.data_root
    if data_root is None:
        # try YAML
        try:
            import yaml  # lazy import to avoid hard dependency if not needed
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            data_root = cfg['paths']['data_root']
        except Exception:
            raise ValueError("Provide --data-root or a valid --config with paths.data_root")

    pq_path = find_cell_parquet(data_root, args.cell)
    print(f"Loading: {pq_path}")
    df = pd.read_parquet(pq_path)

    # Compute Q_sum from |I| and dt
    qsum = compute_qsum(df)
    df = df.copy()
    df['Q_sum_recalc'] = qsum

    # Align and normalize for comparison
    if 'Q_c' not in df.columns:
        raise KeyError("Column 'Q_c' not found in parquet to compare against")

    qc = pd.to_numeric(df['Q_c'], errors='coerce').to_numpy(dtype=float)
    qsumv = df['Q_sum_recalc'].to_numpy(dtype=float)

    # Normalize both to start value (first finite) for shape comparison
    def norm_to_start(arr: np.ndarray) -> np.ndarray:
        arr0 = arr.copy()
        m = np.isfinite(arr0)
        if m.any():
            first_idx = np.argmax(m)  # first True
            start_val = arr0[first_idx]
            arr0 = arr0 - start_val
        return arr0

    qc0 = norm_to_start(qc)
    qsum0 = norm_to_start(qsumv)

    # Compute simple metrics on overlapping finite region
    m = np.isfinite(qc0) & np.isfinite(qsum0)
    if m.sum() > 10:
        diff = qc0[m] - qsum0[m]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        scale = float(np.nan)
        # optional: linear fit qc0 ~ a * qsum0
        try:
            a = np.polyfit(qsum0[m], qc0[m], 1)[0]
            scale = float(a)
        except Exception:
            pass
        print(f"Compare on {m.sum()} points -> MAE={mae:.6f} Ah, RMSE={rmse:.6f} Ah, scale~{scale:.6f}")
    else:
        print("Insufficient finite overlap for metrics.")

    # Build x-axis (hours) using cumulative time for clarity
    try:
        dt_h = infer_dt_hours(df)
        t_h = np.cumsum(dt_h)
    except Exception:
        # fallback to index
        t_h = np.arange(len(df), dtype=float)

    # Plot 1: normalized-to-start comparison
    plt.figure(figsize=(12, 5))
    plt.plot(t_h, qc0, label='Q_c (normalized)', linewidth=1.5)
    plt.plot(t_h, qsum0, label='Q_sum signed I·dt (normalized)', linewidth=1.2, alpha=0.85)
    plt.xlabel('Time [h] (cumulative)')
    plt.ylabel('Charge [Ah] (normalized to start)')
    plt.title(f"{args.cell}: Q_c vs. recalculated Q_sum (signed)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    out_path = args.out
    if out_path is None:
        # default under 1.5.0.0/outputs
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.cell}_Qc_vs_Qsum.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")

    # Plot 2: raw Q_c only (no normalization)
    plt.figure(figsize=(12, 4.5))
    plt.plot(t_h, qc, linewidth=1.0, color='tab:blue')
    plt.xlabel('Time [h] (cumulative)')
    plt.ylabel('Q_c [as in data]')
    plt.title(f"{args.cell}: Q_c (raw)")
    plt.grid(True, alpha=0.3)
    out_path_qc = out_path.replace('_Qc_vs_Qsum.png', '_Qc_raw.png') if out_path.endswith('_Qc_vs_Qsum.png') else out_path.replace('.png', '_Qc_raw.png')
    plt.tight_layout()
    plt.savefig(out_path_qc, dpi=150)
    print(f"Saved plot -> {out_path_qc}")


if __name__ == '__main__':
    main()
