import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import load

BASE = Path('/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/SOC_SOH_1.7.0.0_0.1.2.3/2026-03-24_04_freeze_gap_10h_full_hold')
RUN_SCRIPT = BASE / 'run_freeze_gap_compare.py'
CELL = 'MGFarm_18650_C07'
MAX_ROWS = 100000
GAP_SECONDS = 36000

spec = importlib.util.spec_from_file_location('freeze_gap_compare', RUN_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_root = '/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE'
soh_cfg_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml'
soh_ckpt = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt'
soh_scaler_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib'
with open(soh_cfg_path) as f:
    soh_cfg = yaml.safe_load(f)

df = mod.load_df(data_root, CELL)
keep_cols = ['Testtime[s]', 'SOC', 'Voltage[V]', 'Current[A]', 'Temperature[°C]'] + [c for c in soh_cfg['model']['features'] if c in df.columns]
keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=keep_cols).reset_index(drop=True)
df = df.iloc[:MAX_ROWS].copy()

gap_start = len(df) // 2
gap_end = min(gap_start + GAP_SECONDS, len(df))
freeze_mask = np.zeros(len(df), dtype=bool)
freeze_mask[gap_start:gap_end] = True

df_clean = mod.build_online_aux_features(
    df=df,
    freeze_mask=np.zeros(len(df), dtype=bool),
    current_sign=1.0,
    v_max=3.65,
    v_tol=0.02,
    cv_seconds=300.0,
    nominal_capacity_ah=1.8,
    initial_soc_delta=0.0,
)
df_hold = mod.build_online_aux_features(
    df=df,
    freeze_mask=freeze_mask,
    current_sign=1.0,
    v_max=3.65,
    v_tol=0.02,
    cv_seconds=300.0,
    nominal_capacity_ah=1.8,
    initial_soc_delta=0.0,
)

soh_scaler = load(soh_scaler_path)
base_features = soh_cfg['model']['features']
sampling_cfg = soh_cfg.get('sampling', {})
interval_seconds = int(sampling_cfg.get('interval_seconds', 3600))
feature_aggs = sampling_cfg.get('feature_aggs', ['mean', 'std', 'min', 'max'])
soh_model = mod.SOH_LSTM_Seq2Seq(
    in_features=len(mod.expand_features_for_sampling(base_features, feature_aggs)),
    embed_size=int(soh_cfg['model']['embed_size']),
    hidden_size=int(soh_cfg['model']['hidden_size']),
    mlp_hidden=int(soh_cfg['model']['mlp_hidden']),
    num_layers=int(soh_cfg['model'].get('num_layers', 2)),
    res_blocks=int(soh_cfg['model'].get('res_blocks', 2)),
    bidirectional=bool(soh_cfg['model'].get('bidirectional', False)),
    dropout=float(soh_cfg['model'].get('dropout', 0.15)),
).to(device)
soh_state = torch.load(soh_ckpt, map_location=device)
soh_model.load_state_dict(soh_state['model_state_dict'])

clean_soh_hourly, clean_bins = mod.predict_soh_hourly(df_clean, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device)
gap_soh_hourly, gap_bins = mod.predict_soh_hourly(df_hold, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device)

df_clean['SOH'] = mod.expand_soh_to_rows(df_clean, clean_bins, clean_soh_hourly, interval_seconds)
df_hold['SOH'] = mod.expand_soh_to_rows(df_hold, gap_bins, gap_soh_hourly, interval_seconds)
if freeze_mask.any():
    first_gap = int(np.argmax(freeze_mask))
    hold = float(df_hold['SOH'].iloc[first_gap - 1]) if first_gap > 0 else 1.0
    df_hold.loc[freeze_mask, 'SOH'] = hold

relevant = ['Current[A]', 'Voltage[V]', 'Temperature[°C]', 'Q_c', 'EFC', 'SOH']
time_h = df_clean['Testtime[s]'].to_numpy() / 3600.0
gap_start_h = time_h[gap_start]
gap_end_h = time_h[gap_end - 1]

fig, axes = plt.subplots(len(relevant), 1, figsize=(14, 13), sharex=True)
for ax, col in zip(axes, relevant):
    ax.plot(time_h, df_clean[col].to_numpy(), color='black', linewidth=1.0, label='clean')
    ax.plot(time_h, df_hold[col].to_numpy(), color='#c01c28', linewidth=1.0, label='freeze-hold')
    ax.axvspan(gap_start_h, gap_end_h, color='grey', alpha=0.18)
    ax.set_ylabel(col)
    ax.grid(alpha=0.2)
    ax.legend(loc='best')
axes[-1].set_xlabel('Time [h]')
fig.tight_layout()
fig.savefig(BASE / 'relevant_input_features_freeze_gap_10h_full_hold_MGFarm_18650_C07.png', dpi=170)
