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
from cc_model import CCModel, CCModelConfig
from robustness_common import (
    add_common_scenario_args,
    apply_measurement_scenario,
    build_online_aux_features,
    compute_protocol_event_metrics,
    compute_robustness_metrics,
    compute_stratified_error_metrics,
    load_cell_dataframe,
    write_temporal_error_metrics,
)
from shared_soh_trace import load_soh_trace


def run_cc_with_external_soh(df, soh, gap_mask, args, soc_init):
    model = CCModel(
        CCModelConfig(
            capacity_ah=float(args.nominal_capacity_ah),
            soc_init=float(soc_init),
            current_sign=float(args.current_sign),
            v_max=float(args.v_max),
            v_tol=float(args.v_tol),
            cv_seconds=float(args.cv_seconds),
        )
    )
    time_s = df['Testtime[s]'].to_numpy(dtype=np.float64)
    dt_s = np.diff(time_s, prepend=time_s[0])
    dt_s[dt_s < 0] = 0.0
    has_gap = bool(np.any(gap_mask))
    if has_gap:
        nominal_dt = np.median(dt_s[(~gap_mask) & (dt_s > 0)])
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            nominal_dt = 1.0
        dt_s[gap_mask] = 0.0
        for idx in range(1, len(dt_s)):
            if gap_mask[idx - 1] and not gap_mask[idx]:
                dt_s[idx] = nominal_dt

    current = df['Current[A]'].to_numpy(dtype=np.float64)
    voltage = df['Voltage[V]'].to_numpy(dtype=np.float64)
    soc = np.zeros(len(df), dtype=np.float32)
    for idx in range(len(df)):
        if has_gap and gap_mask[idx]:
            soc[idx] = float(soc_init) if idx == 0 else soc[idx - 1]
            continue
        capacity = float(args.nominal_capacity_ah) * float(soh[idx])
        soc[idx] = model.step(current[idx], voltage[idx], capacity_ah=capacity, dt_s=dt_s[idx])
    return soc


def main():
    ap = argparse.ArgumentParser(description="CC + shared SOH scenario runner.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require_gpu", action="store_true",
                    help="Fail if CUDA is not available.")
    ap.add_argument("--warmup_seconds", type=float, default=600.0,
                    help="Ignore first N seconds for error plot/metrics")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    add_common_scenario_args(ap)

    # model paths
    ap.add_argument("--soh_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml")
    ap.add_argument("--soh_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt")
    ap.add_argument("--soh_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib")
    ap.add_argument("--soh_trace", default=None,
                    help="Shared causal SOH .npz trace; skips local SOH inference when set.")

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

    df = load_cell_dataframe(args.data_root, args.cell, args.start_row, args.max_rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df, scenario_info = apply_measurement_scenario(df, args.scenario, args)
    df = df.dropna(subset=['Testtime[s]', 'Current[A]', 'Voltage[V]', 'SOC']).reset_index(drop=True)

    t = df['Testtime[s]'].to_numpy(dtype=np.float64)
    freeze_mask = np.asarray(scenario_info.get('freeze_mask', np.zeros(len(df), dtype=bool)), dtype=bool)
    soc_init = float(np.clip(float(args.soc_init) + float(scenario_info.get('soc_init_delta', 0.0)), 0.0, 1.0))

    model = None
    if not args.soh_trace:
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
        q_c_reset_voltage_v=float(args.q_c_reset_voltage_v),
        q_c_reset_current_a=float(args.q_c_reset_current_a),
        q_c_capacity_ah=args.q_c_capacity_ah,
    )
    soh_features = [] if model is None else model.soh_base_features
    req_cols = sorted(set(soh_features + ['Testtime[s]', 'Current[A]', 'Voltage[V]', 'SOC']))
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

    trace_metadata = None
    if args.soh_trace:
        soh_pred, trace_metadata = load_soh_trace(args.soh_trace, df['Testtime[s]'].to_numpy(dtype=np.float64))
        soc_cc = run_cc_with_external_soh(df, soh_pred, freeze_mask, args, soc_init)
        device_label = "external_soh_trace"
    else:
        soc_cc, soh_pred = model.process_dataframe(df, gap_mask=freeze_mask)
        device_label = str(model.device)

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
    metrics.update(compute_protocol_event_metrics(
        scenario=args.scenario,
        time_s=t,
        y_true=soc_true,
        y_pred=soc_cc,
        current_a=df['_protocol_current_a'].to_numpy(dtype=np.float64),
        freeze_mask=freeze_mask,
        threshold=args.recovery_abs_error_threshold,
        sustain_seconds=args.recovery_sustain_seconds,
        horizon_seconds=args.recovery_horizon_seconds,
    ))
    stratified_mask = t >= float(args.warmup_seconds)
    if not np.any(stratified_mask):
        stratified_mask = np.ones(len(t), dtype=bool)
    stratified_metrics = compute_stratified_error_metrics(
        y_true=soc_true[stratified_mask],
        y_pred=soc_cc[stratified_mask],
        reference_soh=(
            df['_reference_soh'].to_numpy(dtype=np.float64)[stratified_mask]
            if '_reference_soh' in df else None
        ),
        reference_temperature_c=(
            df['_reference_temperature_c'].to_numpy(dtype=np.float64)[stratified_mask]
            if '_reference_temperature_c' in df else None
        ),
        reference_c_rate=(
            df['_reference_c_rate'].to_numpy(dtype=np.float64)[stratified_mask]
            if '_reference_c_rate' in df else None
        ),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    temporal_paths = None
    if float(args.temporal_metrics_seconds) > 0:
        temporal_paths = write_temporal_error_metrics(
            out_dir=args.out_dir, cell=args.cell, time_s=t, y_true=soc_true, y_pred=soc_cc,
            voltage_v=df['Voltage[V]'].to_numpy(dtype=np.float64),
            interval_seconds=args.temporal_metrics_seconds,
            reference_soh=(df['_reference_soh'].to_numpy(dtype=np.float64) if '_reference_soh' in df else None),
            reset_voltage_v=float(args.v_max - args.v_tol), reset_sustain_seconds=float(args.cv_seconds),
        )

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
        'soh_source': 'external_trace' if args.soh_trace else 'local_lstm',
        'soh_trace': args.soh_trace,
        'soh_trace_metadata': trace_metadata,
        'missing_gap_seconds': float(args.missing_gap_seconds),
        'device': device_label,
        'start_row': int(args.start_row),
        'max_rows': int(args.max_rows),
        'output_policy': 'summary_only' if args.summary_only else 'full_run_artifacts',
        'temporal_metrics': temporal_paths,
        'scenario_meta': {k: v for k, v in scenario_info.items() if k not in ('freeze_mask', 'disturbance_mask')},
        'stratified_metrics': stratified_metrics,
    }
    summary.update(metrics)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if args.summary_only:
        print(json.dumps(summary, indent=2))
        return

    out_df = pd.DataFrame({
        'index': np.arange(len(df)),
        'time_s': t,
        'soc_true': soc_true,
        'soc_cc': soc_cc,
        'soh_pred': soh_pred,
        'current_a_observed': df['Current[A]'].to_numpy(dtype=np.float64),
        'voltage_v_observed': df['Voltage[V]'].to_numpy(dtype=np.float64),
        'q_c_online': df['Q_c'].to_numpy(dtype=np.float64),
        'efc_online': df['EFC'].to_numpy(dtype=np.float64),
        'dt_s_online': df['_dt_s_online'].to_numpy(dtype=np.float64),
        'input_missing': freeze_mask,
        'abs_err': abs_err,
    })
    out_csv = os.path.join(args.out_dir, f"soc_cc_soh_fullcell_{args.cell}.csv")
    out_df.to_csv(out_csv, index=False)
    mask = out_df['time_s'] >= float(args.warmup_seconds)

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
