import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = "/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"


def find_fe_path(data_root: str, cell: str) -> Path:
    c = cell.split('_')[-1]
    candidates = [
        Path(data_root) / f"df_FE_{c}.parquet",
        Path(data_root) / f"df_FE_C{c[-2:]}.parquet",
        Path(data_root) / f"df_FE_{cell}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No FE parquet found for {cell} in {data_root}")


def load_model_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / 'summary.json').read_text())

    csv_map = {
        'CC_1.0.0': f"soc_cc_fullcell_{summary['cell']}.csv",
        'CC_SOH_1.0.0': f"soc_cc_soh_fullcell_{summary['cell']}.csv",
        'ECM_0.0.1': f"ecm_soc_fullcell_{summary['cell']}.csv",
        'SOC_SOH_1.6.0.0_GRU_0.3.1.2': f"soc_pred_fullcell_{summary['cell']}.csv",
    }
    model = summary['model']
    if model not in csv_map:
        raise ValueError(f"Unknown model in summary: {model} ({run_dir})")

    df = pd.read_csv(run_dir / csv_map[model])

    if model in ('CC_1.0.0', 'CC_SOH_1.0.0'):
        pred_col = 'soc_cc'
    elif model == 'ECM_0.0.1':
        pred_col = 'soc_ecm'
    else:
        pred_col = 'soc_pred'

    out = {
        'model': model,
        'run_dir': str(run_dir),
        'summary': summary,
        'time_s': df['time_s'].to_numpy(dtype=np.float64),
        'soc_true': df['soc_true'].to_numpy(dtype=np.float64),
        'soc_pred': df[pred_col].to_numpy(dtype=np.float64),
    }
    out['abs_err'] = np.abs(out['soc_pred'] - out['soc_true'])
    return out


def compute_gap_interval(t: np.ndarray, gap_seconds: float) -> tuple:
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if gap_seconds <= 0 or span <= gap_seconds:
        return np.nan, np.nan
    start = t0 + (span - gap_seconds) * 0.5
    end = start + gap_seconds
    return start, end


def freeze_series_by_gap(t: np.ndarray, x: np.ndarray, g0: float, g1: float) -> np.ndarray:
    y = x.astype(np.float64).copy()
    if not np.isfinite(g0):
        return y
    mask = (t >= g0) & (t <= g1)
    if not np.any(mask):
        return y
    first = int(np.where(mask)[0][0])
    hold = y[first - 1] if first > 0 else y[first]
    y[mask] = hold
    return y


def subset_window(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return (t >= start_s) & (t <= end_s)


def plot_full(models: list, out_dir: Path):
    plt.figure(figsize=(14, 5.2))
    ref = models[0]
    plt.plot(ref['time_s'] / 3600.0, ref['soc_true'], label='SOC true', color='black', linewidth=1.0)
    for m in models:
        s = m['summary']
        plt.plot(m['time_s'] / 3600.0, m['soc_pred'], linewidth=0.9,
                 label=f"{m['model']} (MAE={s['mae']:.4f}, RMSE={s['rmse']:.4f})")
    plt.xlabel('Time [h]')
    plt.ylabel('SOC')
    plt.title('Full run: SOC predictions (all models)')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / '01_full_soc_all_models.png', dpi=160)
    plt.close()


def plot_gap_overview(models: list, i_sig: np.ndarray, u_sig: np.ndarray, t_sig: np.ndarray,
                      t_ref: np.ndarray, g0: float, g1: float, out_dir: Path,
                      pre_min: float = 20.0, post_min: float = 20.0):
    w0 = g0 - pre_min * 60.0
    w1 = g1 + post_min * 60.0

    m_ref = subset_window(t_ref, w0, w1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # SOC panel
    axes[0].plot(t_ref[m_ref] / 3600.0, models[0]['soc_true'][m_ref], color='black', linewidth=1.0, label='SOC true')
    for m in models:
        mk = subset_window(m['time_s'], w0, w1)
        axes[0].plot(m['time_s'][mk] / 3600.0, m['soc_pred'][mk], linewidth=0.9, label=m['model'])
    axes[0].set_ylabel('SOC')
    axes[0].set_title('Gap overview: SOC + measurements (gap and transitions)')
    axes[0].legend(loc='best', fontsize=8)

    axes[1].plot(t_ref[m_ref] / 3600.0, i_sig[m_ref], color='tab:blue', linewidth=0.9)
    axes[1].set_ylabel('I [A]')

    axes[2].plot(t_ref[m_ref] / 3600.0, u_sig[m_ref], color='tab:orange', linewidth=0.9)
    axes[2].set_ylabel('U [V]')

    axes[3].plot(t_ref[m_ref] / 3600.0, t_sig[m_ref], color='tab:green', linewidth=0.9)
    axes[3].set_ylabel('T [C]')
    axes[3].set_xlabel('Time [h]')

    for ax in axes:
        ax.axvspan(g0 / 3600.0, g1 / 3600.0, color='red', alpha=0.15, label='Missing gap')

    plt.tight_layout()
    plt.savefig(out_dir / '02_gap_overview_soc_i_u_t.png', dpi=160)
    plt.close(fig)


def plot_transition(models: list, i_sig: np.ndarray, u_sig: np.ndarray, t_sig: np.ndarray,
                    t_ref: np.ndarray, center_s: float, out_png: Path,
                    before_min: float, after_min: float, title: str):
    w0 = center_s - before_min * 60.0
    w1 = center_s + after_min * 60.0
    m_ref = subset_window(t_ref, w0, w1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t_ref[m_ref] / 3600.0, models[0]['soc_true'][m_ref], color='black', linewidth=1.0, label='SOC true')
    for m in models:
        mk = subset_window(m['time_s'], w0, w1)
        axes[0].plot(m['time_s'][mk] / 3600.0, m['soc_pred'][mk], linewidth=0.9, label=m['model'])
    axes[0].set_ylabel('SOC')
    axes[0].set_title(title)
    axes[0].legend(loc='best', fontsize=8)

    axes[1].plot(t_ref[m_ref] / 3600.0, i_sig[m_ref], color='tab:blue', linewidth=0.9)
    axes[1].set_ylabel('I [A]')

    axes[2].plot(t_ref[m_ref] / 3600.0, u_sig[m_ref], color='tab:orange', linewidth=0.9)
    axes[2].set_ylabel('U [V]')

    axes[3].plot(t_ref[m_ref] / 3600.0, t_sig[m_ref], color='tab:green', linewidth=0.9)
    axes[3].set_ylabel('T [C]')
    axes[3].set_xlabel('Time [h]')

    for ax in axes:
        ax.axvline(center_s / 3600.0, color='red', linestyle='--', linewidth=1.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_gap_only(models: list, g0: float, g1: float, out_dir: Path,
                  pre_min: float = 10.0, post_min: float = 30.0):
    w0 = g0 - pre_min * 60.0
    w1 = g1 + post_min * 60.0

    plt.figure(figsize=(14, 5.2))
    ref = models[0]
    mr = subset_window(ref['time_s'], w0, w1)
    plt.plot(ref['time_s'][mr] / 3600.0, ref['soc_true'][mr], color='black', linewidth=1.1, label='SOC true')
    for m in models:
        mk = subset_window(m['time_s'], w0, w1)
        plt.plot(m['time_s'][mk] / 3600.0, m['soc_pred'][mk], linewidth=1.0, label=m['model'])

    plt.axvspan(g0 / 3600.0, g1 / 3600.0, color='red', alpha=0.15, label='Missing gap')
    plt.xlabel('Time [h]')
    plt.ylabel('SOC')
    plt.title('All model predictions around missing gap')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / '05_gap_only_all_predictions.png', dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Create missing-gap comparison plots across all SOC models.')
    ap.add_argument('--cc_run', required=True)
    ap.add_argument('--cc_soh_run', required=True)
    ap.add_argument('--ecm_run', required=True)
    ap.add_argument('--soc_soh_run', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        load_model_run(Path(args.cc_run)),
        load_model_run(Path(args.cc_soh_run)),
        load_model_run(Path(args.ecm_run)),
        load_model_run(Path(args.soc_soh_run)),
    ]

    # Keep display order fixed
    order = ['CC_1.0.0', 'CC_SOH_1.0.0', 'ECM_0.0.1', 'SOC_SOH_1.6.0.0_GRU_0.3.1.2']
    by_name = {m['model']: m for m in models}
    models = [by_name[k] for k in order]

    ref = models[0]
    gap_seconds = float(ref['summary'].get('missing_gap_seconds', 0.0))
    g0, g1 = compute_gap_interval(ref['time_s'], gap_seconds)

    fe = pd.read_parquet(find_fe_path(args.data_root, args.cell))
    fe = fe[['Testtime[s]', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']].dropna().reset_index(drop=True)

    # Align FE signals to CC timeline for a clean overlay.
    t_ref = ref['time_s']
    i_raw = np.interp(t_ref, fe['Testtime[s]'].to_numpy(dtype=np.float64), fe['Current[A]'].to_numpy(dtype=np.float64))
    u_raw = np.interp(t_ref, fe['Testtime[s]'].to_numpy(dtype=np.float64), fe['Voltage[V]'].to_numpy(dtype=np.float64))
    temp_raw = np.interp(t_ref, fe['Testtime[s]'].to_numpy(dtype=np.float64), fe['Temperature[°C]'].to_numpy(dtype=np.float64))

    i_sig = freeze_series_by_gap(t_ref, i_raw, g0, g1)
    u_sig = freeze_series_by_gap(t_ref, u_raw, g0, g1)
    temp_sig = freeze_series_by_gap(t_ref, temp_raw, g0, g1)

    plot_full(models, out_dir)
    plot_gap_overview(models, i_sig, u_sig, temp_sig, t_ref, g0, g1, out_dir,
                      pre_min=20.0, post_min=20.0)
    plot_transition(models, i_sig, u_sig, temp_sig, t_ref, g0,
                    out_dir / '03_transition_gap_start.png',
                    before_min=8.0, after_min=25.0,
                    title='Transition around gap start (zoom)')
    plot_transition(models, i_sig, u_sig, temp_sig, t_ref, g1,
                    out_dir / '04_transition_gap_end.png',
                    before_min=8.0, after_min=45.0,
                    title='Transition around gap end (zoom + recovery)')
    plot_gap_only(models, g0, g1, out_dir, pre_min=10.0, post_min=30.0)

    overview = {
        'cell': args.cell,
        'gap_seconds': gap_seconds,
        'gap_start_s': float(g0),
        'gap_end_s': float(g1),
        'runs': {m['model']: m['run_dir'] for m in models},
        'metrics': {
            m['model']: {
                'mae': float(m['summary']['mae']),
                'rmse': float(m['summary']['rmse']),
                'mae_after_warmup': float(m['summary']['mae_after_warmup']),
                'rmse_after_warmup': float(m['summary']['rmse_after_warmup']),
            }
            for m in models
        },
    }
    (out_dir / 'missing_gap_comparison_summary.json').write_text(json.dumps(overview, indent=2))


if __name__ == '__main__':
    main()
