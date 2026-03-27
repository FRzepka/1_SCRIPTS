import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gap_bounds(t: np.ndarray, gap_seconds: float) -> tuple:
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if gap_seconds <= 0 or span <= gap_seconds:
        return np.nan, np.nan
    g0 = t0 + (span - gap_seconds) * 0.5
    g1 = g0 + gap_seconds
    return g0, g1


def find_first_sustained(t: np.ndarray, err: np.ndarray, start_t: float, thr: float, sustain_s: float = 600.0):
    post = np.where(t >= start_t)[0]
    if len(post) == 0:
        return None

    dt = np.diff(t[post])
    dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    max_gap = max(1.5 * dt_med, 1.0)

    ok_idx = post[err[post] <= thr]
    if len(ok_idx) == 0:
        return None

    seg_start = ok_idx[0]
    prev = ok_idx[0]
    for idx in ok_idx[1:]:
        if idx == prev + 1 and (t[idx] - t[prev]) <= max_gap:
            prev = idx
        else:
            if (t[prev] - t[seg_start]) >= sustain_s:
                return seg_start
            seg_start = idx
            prev = idx

    if (t[prev] - t[seg_start]) >= sustain_s:
        return seg_start
    return None


def detect_cc_reset_indices(t: np.ndarray, soc: np.ndarray, q_m_new: np.ndarray | None = None):
    reset = np.zeros(len(t), dtype=bool)
    ds = np.diff(soc, prepend=soc[0])
    reset |= ds > 0.08
    if q_m_new is not None:
        dq = np.diff(q_m_new, prepend=q_m_new[0])
        reset |= (np.abs(q_m_new) < 1e-4) & (np.abs(np.roll(q_m_new, 1)) > 0.05)
        reset |= dq > 0.08
    idx = np.where(reset)[0]
    return idx


def nearest_before(arr: np.ndarray, ref: float):
    if len(arr) == 0:
        return None
    arr = arr[arr <= ref]
    if len(arr) == 0:
        return None
    return arr[-1]


def plot_recovery_timeline(models: list[dict], out_png: Path, hours: float = 40.0):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for m in models:
        t_rel_h = (m['t'] - m['g1']) / 3600.0
        sel = (t_rel_h >= 0) & (t_rel_h <= hours)
        # thin for plotting speed
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        step = max(1, len(idx) // 12000)
        pick = idx[::step]
        ax.plot(t_rel_h[pick], m['err'][pick], linewidth=0.9, label=m['model'])
        ax.hlines(m['thr'], xmin=0, xmax=hours, colors='gray', linestyles=':', linewidth=0.5)
        if m['rec_idx'] is not None:
            x = (m['t'][m['rec_idx']] - m['g1']) / 3600.0
            y = m['err'][m['rec_idx']]
            ax.scatter([x], [y], s=22)
            ax.annotate(f"{m['model']}\n{x:.2f}h", (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlim(0, hours)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel('Hours after gap end')
    ax.set_ylabel('Absolute SOC error')
    ax.set_title('Recovery timeline after missing-gap (all models)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_zoom_model(m: dict, i_sig: np.ndarray, u_sig: np.ndarray, temp_sig: np.ndarray, out_png: Path):
    if m['rec_idx'] is None:
        return

    t = m['t']
    rec_t = t[m['rec_idx']]
    w0 = rec_t - 2.0 * 3600.0
    w1 = rec_t + 2.0 * 3600.0
    mask = (t >= w0) & (t <= w1)
    if not np.any(mask):
        return

    tt = t[mask] / 3600.0
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(tt, m['soc_true'][mask], color='black', linewidth=1.0, label='SOC true')
    axes[0].plot(tt, m['soc_pred'][mask], linewidth=1.0, label=m['model'])
    axes[0].axvspan(m['g0'] / 3600.0, m['g1'] / 3600.0, color='red', alpha=0.15)
    axes[0].axvline(rec_t / 3600.0, color='tab:green', linestyle='--', linewidth=1.2, label='recovery point')
    axes[0].set_ylabel('SOC')
    axes[0].set_title(f"{m['model']} recovery zoom")
    axes[0].legend(loc='best', fontsize=8)

    axes[1].plot(tt, m['err'][mask], color='tab:red', linewidth=1.0, label='abs error')
    axes[1].axhline(m['thr'], color='gray', linestyle=':', linewidth=1.0, label='recovery threshold')
    axes[1].axvspan(m['g0'] / 3600.0, m['g1'] / 3600.0, color='red', alpha=0.15)
    axes[1].axvline(rec_t / 3600.0, color='tab:green', linestyle='--', linewidth=1.2)
    axes[1].set_ylabel('Abs err')
    axes[1].set_ylim(0, 0.42)
    axes[1].legend(loc='best', fontsize=8)

    axes[2].plot(tt, i_sig[mask], color='tab:blue', linewidth=0.9)
    axes[2].axvspan(m['g0'] / 3600.0, m['g1'] / 3600.0, color='red', alpha=0.15)
    axes[2].axvline(rec_t / 3600.0, color='tab:green', linestyle='--', linewidth=1.2)
    axes[2].set_ylabel('I [A]')

    ax_u = axes[3]
    ax_u.plot(tt, u_sig[mask], color='tab:orange', linewidth=0.9, label='U')
    ax_t = ax_u.twinx()
    ax_t.plot(tt, temp_sig[mask], color='tab:green', linewidth=0.9, alpha=0.7, label='T')
    ax_u.axvspan(m['g0'] / 3600.0, m['g1'] / 3600.0, color='red', alpha=0.15)
    ax_u.axvline(rec_t / 3600.0, color='tab:green', linestyle='--', linewidth=1.2)
    ax_u.set_ylabel('U [V]')
    ax_t.set_ylabel('T [C]')
    ax_u.set_xlabel('Time [h]')

    if m.get('reset_t') is not None:
        for ax in axes:
            ax.axvline(m['reset_t'] / 3600.0, color='tab:purple', linestyle='-.', linewidth=1.0)
        axes[0].text(m['reset_t'] / 3600.0, np.nanmax(m['soc_true'][mask]), 'reset', color='tab:purple', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fe_data', default='/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE/df_FE_C07.parquet')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_map = {
        'CC_1.0.0': {
            'csv': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_1.0.0/runs/missing_gap/2026-02-18_1543_missing_gap_all_sensors_3600/soc_cc_fullcell_MGFarm_18650_C07.csv'),
            'summary': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_1.0.0/runs/missing_gap/2026-02-18_1543_missing_gap_all_sensors_3600/summary.json'),
            'baseline': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_1.0.0/runs/baseline/2026-02-18_1043_baseline/summary.json'),
            'pred_col': 'soc_cc',
        },
        'CC_SOH_1.0.0': {
            'csv': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_SOH_1.0.0/runs/missing_gap/2026-02-18_1543_missing_gap_all_sensors_3600/soc_cc_soh_fullcell_MGFarm_18650_C07.csv'),
            'summary': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_SOH_1.0.0/runs/missing_gap/2026-02-18_1543_missing_gap_all_sensors_3600/summary.json'),
            'baseline': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/CC_SOH_1.0.0/runs/baseline/2026-02-18_1043_baseline/summary.json'),
            'pred_col': 'soc_cc',
        },
        'ECM_0.0.1': {
            'csv': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/ECM_0.0.1/runs/missing_gap/2026-02-18_1628_missing_gap_all_sensors_3600_1s/ecm_soc_fullcell_MGFarm_18650_C07.csv'),
            'summary': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/ECM_0.0.1/runs/missing_gap/2026-02-18_1628_missing_gap_all_sensors_3600_1s/summary.json'),
            'baseline': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/ECM_0.0.1/runs/baseline/2026-02-18_1043_baseline/summary.json'),
            'pred_col': 'soc_ecm',
        },
        'SOC_SOH_1.6.0.0_GRU_0.3.1.2': {
            'csv': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/SOC_SOH_1.6.0.0_0.1.2.3/runs/missing_gap/2026-02-18_1601_missing_gap_all_sensors_3600/soc_pred_fullcell_MGFarm_18650_C07.csv'),
            'summary': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/SOC_SOH_1.6.0.0_0.1.2.3/runs/missing_gap/2026-02-18_1601_missing_gap_all_sensors_3600/summary.json'),
            'baseline': Path('DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/SOC_SOH_1.6.0.0_0.1.2.3/runs/baseline/2026-02-18_1043_baseline/summary.json'),
            'pred_col': 'soc_pred',
        },
    }

    models = []
    for model, cfg in run_map.items():
        cols = ['time_s', 'soc_true', cfg['pred_col'], 'abs_err']
        extra_cols = []
        if model == 'CC_1.0.0':
            extra_cols = ['q_m_new']
        df = pd.read_csv(cfg['csv'], usecols=cols + extra_cols)
        t = df['time_s'].to_numpy(dtype=np.float64)
        soc_true = df['soc_true'].to_numpy(dtype=np.float64)
        soc_pred = df[cfg['pred_col']].to_numpy(dtype=np.float64)
        err = np.abs(soc_pred - soc_true)

        summary = json.loads(cfg['summary'].read_text())
        base = json.loads(cfg['baseline'].read_text())
        base_mae = float(base.get('mae', base.get('soc_mae')))
        thr = 1.2 * base_mae

        g0, g1 = gap_bounds(t, float(summary.get('missing_gap_seconds', 0.0)))
        rec_idx = find_first_sustained(t, err, g1, thr, sustain_s=600.0)

        m = {
            'model': model,
            't': t,
            'soc_true': soc_true,
            'soc_pred': soc_pred,
            'err': err,
            'thr': thr,
            'g0': g0,
            'g1': g1,
            'rec_idx': rec_idx,
            'run_csv': str(cfg['csv']),
            'summary': summary,
            'baseline_mae': base_mae,
        }

        if model == 'CC_1.0.0':
            q = df['q_m_new'].to_numpy(dtype=np.float64)
            m['q_m_new'] = q
            reset_idx = detect_cc_reset_indices(t, soc_pred, q)
            m['reset_idx'] = reset_idx
            if rec_idx is not None:
                rt = nearest_before(t[reset_idx], t[rec_idx])
                m['reset_t'] = rt
        elif model == 'CC_SOH_1.0.0':
            reset_idx = detect_cc_reset_indices(t, soc_pred, None)
            m['reset_idx'] = reset_idx
            if rec_idx is not None:
                rt = nearest_before(t[reset_idx], t[rec_idx])
                m['reset_t'] = rt
        models.append(m)

    # Use ECM run as common I/U because it already contains the scenario signal (incl. freeze in gap)
    df_ecm = pd.read_csv(run_map['ECM_0.0.1']['csv'], usecols=['time_s', 'I', 'U'])
    t_sig = df_ecm['time_s'].to_numpy(dtype=np.float64)
    i_sig = df_ecm['I'].to_numpy(dtype=np.float64)
    u_sig = df_ecm['U'].to_numpy(dtype=np.float64)

    # Temperature from FE data, interpolated to ECM timeline
    fe = pd.read_parquet(args.fe_data, columns=['Testtime[s]', 'Temperature[°C]'])
    fe = fe.dropna().reset_index(drop=True)
    t_fe = fe['Testtime[s]'].to_numpy(dtype=np.float64)
    temp_fe = fe['Temperature[°C]'].to_numpy(dtype=np.float64)
    temp_sig = np.interp(t_sig, t_fe, temp_fe)

    # Freeze temperature over gap to mirror scenario
    g0, g1 = models[0]['g0'], models[0]['g1']
    mask_gap = (t_sig >= g0) & (t_sig <= g1)
    if np.any(mask_gap):
        first = int(np.where(mask_gap)[0][0])
        hold = temp_sig[first - 1] if first > 0 else temp_sig[first]
        temp_sig[mask_gap] = hold

    # Recovery ranking
    rows = []
    for m in models:
        if m['rec_idx'] is None:
            rec_h = np.nan
        else:
            rec_h = (m['t'][m['rec_idx']] - m['g1']) / 3600.0
        rows.append((m['model'], rec_h))
    rows_sorted = sorted(rows, key=lambda x: (np.inf if np.isnan(x[1]) else x[1]))

    # Plots
    plot_recovery_timeline(models, out_dir / '07_recovery_timeline_abs_error.png', hours=40)

    for m in models:
        # align signals to this model timeline
        i_m = np.interp(m['t'], t_sig, i_sig)
        u_m = np.interp(m['t'], t_sig, u_sig)
        t_m = np.interp(m['t'], t_sig, temp_sig)
        fname = f"08_recovery_zoom_{m['model'].replace('.', '_')}.png"
        plot_zoom_model(m, i_m, u_m, t_m, out_dir / fname)

    # Summary markdown
    lines = []
    lines.append('|rank|model|recovery_h_after_gap|baseline_mae|threshold_1_2x|notes|')
    lines.append('|---|---|---:|---:|---:|---|')
    rank = 1
    model_by_name = {m['model']: m for m in models}
    for model, rec_h in rows_sorted:
        mm = model_by_name[model]
        note = ''
        if model.startswith('CC') and mm.get('reset_t') is not None and mm['rec_idx'] is not None:
            dt_min = (mm['t'][mm['rec_idx']] - mm['reset_t']) / 60.0
            note = f"nearest reset before recovery: {dt_min:.1f} min"
        elif model == 'ECM_0.0.1':
            note = 'no hard reset; recovery via voltage correction (EKF residual)'
        elif model.startswith('SOC_SOH'):
            note = 'no hard reset; recovery via learned sequence dynamics'

        rec_txt = '-' if np.isnan(rec_h) else f"{rec_h:.3f}"
        lines.append(
            f"|{rank}|{model}|{rec_txt}|{mm['baseline_mae']:.6f}|{mm['thr']:.6f}|{note}|"
        )
        rank += 1

    lines.append('')
    lines.append('Interpretation hint: if CC/CC_SOH recovery aligns with a reset marker, the reset is the main reason for re-locking.')
    (out_dir / 'RECOVERY_RANKING.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
