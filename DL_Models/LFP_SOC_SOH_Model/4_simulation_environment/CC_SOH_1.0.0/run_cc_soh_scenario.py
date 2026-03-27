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
# Model path
CC_SOH_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '2_models', 'CC_SOH_1.0.0'
)
sys.path.append(os.path.abspath(CC_SOH_DIR))
from cc_soh_model import CCSOHModel, CCSOHConfig
from robustness_common import (
    add_common_scenario_args,
    apply_measurement_scenario,
    build_online_aux_features,
    compute_robustness_metrics,
    load_cell_dataframe,
)


def main():
    ap = argparse.ArgumentParser(description="CC + shared SOH scenario runner.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require_gpu", action="store_true",
                    help="Fail if CUDA is not available.")
    ap.add_argument("--warmup_seconds", type=float, default=600.0,
                    help="Ignore first N seconds for error plot/metrics")
    add_common_scenario_args(ap)

    # model paths
    ap.add_argument("--soh_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml")
    ap.add_argument("--soh_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt")
    ap.add_argument("--soh_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib")

    ap.add_argument("--soh_init", type=float, default=1.0)
    ap.add_argument("--nominal_capacity_ah", type=float, default=1.8)

    # CC config
    ap.add_argument("--soc_init", type=float, default=1.0)
    ap.add_argument("--current_sign", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=3.65)
    ap.add_argument("--v_tol", type=float, default=0.02)
    ap.add_argument("--cv_seconds", type=float, default=300.0)

    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")

    args = ap.parse_args()
    np.random.seed(int(args.seed))

    df = load_cell_dataframe(args.data_root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan)
    df, scenario_info = apply_measurement_scenario(df, args.scenario, args)
    df = df.dropna(subset=['Testtime[s]', 'Current[A]', 'Voltage[V]', 'SOC']).reset_index(drop=True)

    t = df['Testtime[s]'].to_numpy(dtype=np.float64)
    freeze_mask = np.asarray(scenario_info.get('freeze_mask', np.zeros(len(df), dtype=bool)), dtype=bool)
    soc_init = float(np.clip(float(args.soc_init) + float(scenario_info.get('soc_init_delta', 0.0)), 0.0, 1.0))

    cfg = CCSOHConfig(
        soh_config=args.soh_config,
        soh_checkpoint=args.soh_ckpt,
        soh_scaler=args.soh_scaler,
        nominal_capacity_ah=float(args.nominal_capacity_ah),
        soh_init=float(args.soh_init),
        device=args.device,
        soc_init=soc_init,
        current_sign=float(args.current_sign),
        v_max=float(args.v_max),
        v_tol=float(args.v_tol),
        cv_seconds=float(args.cv_seconds),
    )

    model = CCSOHModel(cfg)
    if args.require_gpu and model.device.type != 'cuda':
        raise RuntimeError("GPU required (--require_gpu), but CUDA is not available.")
    print(f"Using device: {model.device}")
    df = build_online_aux_features(
        df=df,
        freeze_mask=freeze_mask,
        current_sign=float(args.current_sign),
        v_max=float(args.v_max),
        v_tol=float(args.v_tol),
        cv_seconds=float(args.cv_seconds),
        nominal_capacity_ah=float(args.nominal_capacity_ah),
    )
    req_cols = sorted(set(model.soh_base_features + ['Testtime[s]', 'Current[A]', 'Voltage[V]', 'SOC']))
    miss = [c for c in req_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required online features for CC+SOH: {miss}")
    for c in req_cols:
        if c == 'SOC':
            continue
        df[c] = df[c].ffill().bfill()
    df = df.dropna(subset=req_cols).reset_index(drop=True)
    if len(df) != len(freeze_mask):
        freeze_mask = np.asarray(scenario_info.get('freeze_mask', np.zeros(len(df), dtype=bool)), dtype=bool)
        freeze_mask = freeze_mask[:len(df)]
        if len(freeze_mask) < len(df):
            freeze_mask = np.pad(freeze_mask, (0, len(df) - len(freeze_mask)), constant_values=False)

    soc_cc, soh_pred = model.process_dataframe(df, gap_mask=freeze_mask)

    soc_true = df['SOC'].to_numpy(dtype=np.float32)
    t = df['Testtime[s]'].to_numpy(dtype=np.float64)
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
        'soh_pred': soh_pred,
        'abs_err': abs_err,
    })
    out_csv = os.path.join(args.out_dir, f"soc_cc_soh_fullcell_{args.cell}.csv")
    out_df.to_csv(out_csv, index=False)

    mask = out_df['time_s'] >= float(args.warmup_seconds)

    summary = {
        'model': 'CC_SOH_1.0.0',
        'cell': args.cell,
        'scenario': args.scenario,
        'soc_init': soc_init,
        'nominal_capacity_ah': float(args.nominal_capacity_ah),
        'soh_init': float(args.soh_init),
        'v_max': float(args.v_max),
        'v_tol': float(args.v_tol),
        'cv_seconds': float(args.cv_seconds),
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'soh_config': args.soh_config,
        'soh_ckpt': args.soh_ckpt,
        'soh_scaler': args.soh_scaler,
        'missing_gap_seconds': float(args.missing_gap_seconds),
        'device': str(model.device),
        'scenario_meta': {k: v for k, v in scenario_info.items() if k not in ('freeze_mask', 'disturbance_mask')},
    }
    summary.update(metrics)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(out_df['time_s'] / 3600.0, out_df['soc_true'], label='SOC true', linewidth=1.0)
    ax1.plot(out_df['time_s'] / 3600.0, out_df['soc_cc'], label='SOC CC+SOH', linewidth=1.0, alpha=0.8)
    ax1.set_title(f"CC+SOH – Full Cell ({args.cell}) [{args.scenario}]")
    ax1.set_ylabel('SOC')
    ax1.legend(loc='best')
    fig.text(0.12, 0.93, f"MAE: {summary['mae']:.5f} | RMSE: {summary['rmse']:.5f} | P95: {summary['p95_error']:.5f}", fontsize=13,
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    t_plot = out_df.loc[mask, 'time_s'] / 3600.0
    err_plot = out_df.loc[mask, 'abs_err']
    ax2.plot(t_plot, err_plot, label='Absolute Error', linewidth=1.0, color='tab:red')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Abs Error')
    ax2.set_ylim(0.0, 0.4)
    ax2.legend(loc='best')

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"soc_cc_soh_fullcell_{args.cell}.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
