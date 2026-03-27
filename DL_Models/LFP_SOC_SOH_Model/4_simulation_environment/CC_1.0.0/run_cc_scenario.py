import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIM_ENV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SIM_ENV_DIR)
CC_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '2_models', 'CC_1.0.0'
)
sys.path.append(os.path.abspath(CC_MODEL_DIR))

from cc_model import CCModel, CCModelConfig
from robustness_common import (
    add_common_scenario_args,
    apply_measurement_scenario,
    compute_robustness_metrics,
    load_cell_dataframe,
)


def main():
    ap = argparse.ArgumentParser(description="Coulomb counting scenario runner (online CC model).")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--out_dir", required=True)
    add_common_scenario_args(ap)

    ap.add_argument("--soc_init", type=float, default=1.0)
    ap.add_argument("--capacity_ah", type=float, default=1.8)
    ap.add_argument("--capacity_source", choices=["const", "data"], default="const",
                    help="const=use capacity_ah parameter, data=use Capacity[Ah] column")
    ap.add_argument("--current_sign", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=3.65)
    ap.add_argument("--v_tol", type=float, default=0.02)
    ap.add_argument("--cv_seconds", type=float, default=300.0)
    ap.add_argument("--warmup_seconds", type=float, default=600.0,
                    help="Ignore first N seconds for error plot/metrics")

    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")

    args = ap.parse_args()
    np.random.seed(int(args.seed))

    df = load_cell_dataframe(args.data_root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan)
    df, scenario_info = apply_measurement_scenario(df, args.scenario, args)
    df = df.dropna(subset=['Testtime[s]', 'Current[A]', 'Voltage[V]', 'SOC']).reset_index(drop=True)

    soc_init = float(np.clip(float(args.soc_init) + float(scenario_info.get('soc_init_delta', 0.0)), 0.0, 1.0))

    cfg = CCModelConfig(
        capacity_ah=float(args.capacity_ah),
        soc_init=soc_init,
        current_sign=float(args.current_sign),
        v_max=float(args.v_max),
        v_tol=float(args.v_tol),
        cv_seconds=float(args.cv_seconds),
    )
    model = CCModel(cfg)

    t = df['Testtime[s]'].to_numpy(dtype=np.float64)
    dt_s = np.diff(t, prepend=t[0])
    dt_s[dt_s < 0] = 0.0
    freeze_mask = np.asarray(scenario_info.get('freeze_mask', np.zeros(len(df), dtype=bool)), dtype=bool)
    has_gap = bool(np.any(freeze_mask))
    if has_gap:
        nominal_dt = np.median(dt_s[(~freeze_mask) & (dt_s > 0)])
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            nominal_dt = 1.0
        dt_s[freeze_mask] = 0.0
        for k in range(1, len(dt_s)):
            if freeze_mask[k - 1] and not freeze_mask[k]:
                dt_s[k] = nominal_dt
    i = df['Current[A]'].to_numpy(dtype=np.float64)
    v = df['Voltage[V]'].to_numpy(dtype=np.float64)

    cap = None
    if args.capacity_source == "data":
        if 'Capacity[Ah]' not in df.columns:
            raise ValueError("Capacity[Ah] column missing but capacity_source=data was requested")
        cap = df['Capacity[Ah]'].to_numpy(dtype=np.float64)

    soc_true = df['SOC'].to_numpy(dtype=np.float32)
    soc_cc = np.zeros(len(df), dtype=np.float32)
    q_m_new = np.zeros(len(df), dtype=np.float32)

    for k in range(len(df)):
        if has_gap and freeze_mask[k]:
            if k == 0:
                soc_cc[k] = soc_init
                q_m_new[k] = model.q_m_new
            else:
                soc_cc[k] = soc_cc[k - 1]
                q_m_new[k] = q_m_new[k - 1]
            continue
        cap_k = cap[k] if cap is not None else None
        soc_cc[k] = model.step(i[k], v[k], capacity_ah=cap_k, dt_s=dt_s[k])
        q_m_new[k] = model.q_m_new

    abs_err = np.abs(soc_true - soc_cc)
    metrics = compute_robustness_metrics(
        time_s=t,
        y_true=soc_true,
        y_pred=soc_cc,
        warmup_seconds=float(args.warmup_seconds),
        disturbance_mask=np.asarray(scenario_info.get('disturbance_mask', freeze_mask), dtype=bool),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    out_df = pd.DataFrame({
        'index': np.arange(len(df)),
        'time_s': t,
        'soc_true': soc_true,
        'soc_cc': soc_cc,
        'q_m_new': q_m_new,
        'abs_err': abs_err,
    })
    out_csv = os.path.join(args.out_dir, f"soc_cc_fullcell_{args.cell}.csv")
    out_df.to_csv(out_csv, index=False)

    summary = {
        'model': 'CC_1.0.0',
        'cell': args.cell,
        'scenario': args.scenario,
        'soc_init': soc_init,
        'capacity_ah': float(args.capacity_ah),
        'capacity_source': args.capacity_source,
        'current_sign': float(args.current_sign),
        'v_max': float(args.v_max),
        'v_tol': float(args.v_tol),
        'cv_seconds': float(args.cv_seconds),
        'warmup_seconds': float(args.warmup_seconds),
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'bias': metrics['bias'],
        'missing_gap_seconds': float(args.missing_gap_seconds),
        'data_root': args.data_root,
        'scenario_meta': {k: v for k, v in scenario_info.items() if k not in ('freeze_mask', 'disturbance_mask')},
    }
    summary.update(metrics)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    try:
        mask = out_df['time_s'] >= float(args.warmup_seconds)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(out_df['time_s'] / 3600.0, out_df['soc_true'], label='SOC true', linewidth=1.0)
        ax1.plot(out_df['time_s'] / 3600.0, out_df['soc_cc'], label='SOC CC', linewidth=1.0, alpha=0.8)
        ax1.set_title(f"CC Model – Full Cell ({args.cell}) [{args.scenario}]")
        ax1.set_ylabel('SOC')
        ax1.legend(loc='best')
        fig.text(0.12, 0.93, f"MAE: {summary['mae']:.5f} | RMSE: {summary['rmse']:.5f} | P95: {summary['p95_error']:.5f}", fontsize=13,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        mask = out_df['time_s'] >= float(args.warmup_seconds)
        t_plot = out_df.loc[mask, 'time_s'] / 3600.0
        err_plot = out_df.loc[mask, 'abs_err']
        ax2.plot(t_plot, err_plot, label='Absolute Error', linewidth=1.0, color='tab:red')
        ax2.set_xlabel('Time [h]')
        ax2.set_ylabel('Abs Error')
        ax2.set_ylim(0.0, 0.4)
        ax2.legend(loc='best')

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"soc_cc_fullcell_{args.cell}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
