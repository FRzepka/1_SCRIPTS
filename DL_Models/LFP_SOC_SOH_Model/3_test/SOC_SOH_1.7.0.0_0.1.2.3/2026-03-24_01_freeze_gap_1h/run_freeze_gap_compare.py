import json, math, argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import load
from tqdm import tqdm

import sys
ROOT = Path('/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/4_simulation_environment')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from robustness_common import build_online_aux_features


class LSTMMLP(nn.Module):
    def __init__(self, in_features, hidden_size, mlp_hidden, num_layers=1, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :]).squeeze(-1)


class GRUMLP(nn.Module):
    def __init__(self, in_features, hidden_size, mlp_hidden, num_layers=1, dropout=0.05):
        super().__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.gru(x)
        return self.mlp(out[:, -1, :]).squeeze(-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        out = self.drop(out)
        return self.norm(x + out)


class SOH_LSTM_Seq2Seq(nn.Module):
    def __init__(self, in_features: int, embed_size: int, hidden_size: int, mlp_hidden: int, num_layers: int = 2, res_blocks: int = 2, bidirectional: bool = False, dropout: float = 0.15):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size), nn.LayerNorm(embed_size), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.post_norm = nn.LayerNorm(out_dim)
        self.res_blocks = nn.ModuleList([ResidualMLPBlock(out_dim, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))])
        self.head = nn.Sequential(
            nn.Linear(out_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
    def forward(self, x, state=None, return_state=False):
        x = self.feature_proj(x)
        out, new_state = self.lstm(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y = self.head(out).squeeze(-1)
        return (y, new_state) if return_state else y


def load_df(root, cell):
    for name in [f'df_FE_{cell.split("_")[-1]}.parquet', f'df_FE_{cell}.parquet', f'df_FE_C{cell[-3:]}.parquet']:
        p = Path(root) / name
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(cell)


def expand_features_for_sampling(base_features: List[str], feature_aggs: List[str]) -> List[str]:
    return [f'{feat}_{agg}' for feat in base_features for agg in feature_aggs]


def aggregate_hourly(df: pd.DataFrame, base_features: List[str], interval_seconds: int, feature_aggs: List[str]) -> pd.DataFrame:
    work = df[base_features + ['Testtime[s]']].replace([np.inf, -np.inf], np.nan).dropna().copy().sort_values('Testtime[s]')
    work['_bin'] = (work['Testtime[s]'] // interval_seconds).astype(np.int64)
    agg_spec = {feat: feature_aggs for feat in base_features}
    out = work.groupby('_bin', sort=True).agg(agg_spec)
    out.columns = [f'{c[0]}_{c[1]}' for c in out.columns]
    out['bin'] = out.index
    return out.reset_index(drop=True)


def predict_soh_hourly(df_raw, base_features, interval_seconds, feature_aggs, scaler, model, device, soh_init=1.0):
    hourly = aggregate_hourly(df_raw, base_features, interval_seconds, feature_aggs)
    feat_cols = expand_features_for_sampling(base_features, feature_aggs)
    Xs = scaler.transform(hourly[feat_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    preds = []
    state = None
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(hourly)), desc='SOH hourly', leave=False):
            x_step = torch.from_numpy(Xs[i:i+1]).unsqueeze(1).to(device)
            y_seq, state = model(x_step, state=state, return_state=True)
            preds.append(float(soh_init) if i == 0 else float(y_seq.squeeze().detach().cpu().numpy()))
    return np.array(preds, dtype=np.float32), hourly['bin'].to_numpy(dtype=np.int64)


def expand_soh_to_rows(df_raw, bins, soh_preds, interval_seconds, soh_init=1.0):
    bin_to_soh = {int(b): float(s) for b, s in zip(bins, soh_preds)}
    if len(bins):
        last = soh_init
        for b in range(int(bins.min()), int(bins.max()) + 1):
            if b in bin_to_soh:
                last = bin_to_soh[b]
            else:
                bin_to_soh[b] = last
    return np.array([bin_to_soh.get(int(t // interval_seconds), soh_init) for t in df_raw['Testtime[s]'].to_numpy(dtype=np.float64)], dtype=np.float32)


def prepare_features_standard(df, features):
    work = df.copy()
    if 'dt_s' in features and 'dt_s' not in work.columns:
        t = work['Testtime[s]'].to_numpy(dtype=np.float32)
        dt = np.empty_like(t)
        if len(t) > 1:
            dt[0] = max(float(t[1] - t[0]), 1e-6)
            dt[1:] = np.diff(t)
        else:
            dt[0] = 1.0
        work['dt_s'] = dt
    return work[features].to_numpy(dtype=np.float32)


def run_soc_variant(name, cfg_path, ckpt_path, scaler_path, df_soc, device, batch_size):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    features = cfg['model']['features']
    chunk = int(cfg['training']['seq_chunk_size'])
    Xs = load(scaler_path).transform(prepare_features_standard(df_soc, features)).astype(np.float32)
    y = df_soc['SOC'].to_numpy(dtype=np.float32)
    model_cls = GRUMLP if cfg['model'].get('type', 'LSTM_MLP') == 'GRU_MLP' else LSTMMLP
    model = model_cls(len(features), int(cfg['model']['hidden_size']), int(cfg['model']['mlp_hidden']), int(cfg['model'].get('num_layers', 1)), float(cfg['model'].get('dropout', 0.05))).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    n_pred = len(df_soc) - chunk + 1
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, n_pred, batch_size), desc=name, leave=False):
            end = min(i + batch_size, n_pred)
            xb = np.stack([Xs[j:j+chunk] for j in range(i, end)], axis=0)
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy()
            preds.append(pred)
    pred = np.concatenate(preds)
    idx = np.arange(chunk - 1, len(df_soc))
    y_ref = y[chunk - 1:]
    err = np.abs(y_ref - pred)
    rmse = float(math.sqrt(np.mean((y_ref - pred) ** 2)))
    mae = float(np.mean(err))
    return {'name': name, 'features': features, 'chunk': chunk, 'rmse': rmse, 'mae': mae, 'pred': pred, 'y_ref': y_ref, 'index': idx, 'abs_err': err}


def find_recovery_mark(idx_actual, abs_err, gap_start_idx, gap_end_idx):
    pre_mask = (idx_actual >= max(idx_actual[0], gap_start_idx - 3 * 3600)) & (idx_actual < max(idx_actual[0], gap_start_idx - 300))
    pre_vals = abs_err[pre_mask]
    if len(pre_vals) == 0:
        pre_vals = abs_err[max(0, len(abs_err) // 3 - 500):len(abs_err) // 3]

    stable_level = float(np.nanmedian(pre_vals))
    stable_p90 = float(np.nanquantile(pre_vals, 0.90))
    threshold = max(stable_level * 1.25, stable_level + 0.0015)

    def first_window_stable(start_actual, horizon_s, max_p90):
        horizon_s = int(horizon_s)
        for start in range(int(start_actual), int(idx_actual[-1] - horizon_s) + 1, 60):
            mask = (idx_actual >= start) & (idx_actual < start + horizon_s)
            vals = abs_err[mask]
            if len(vals) < max(300, int(0.8 * horizon_s)):
                continue
            if float(np.nanmedian(vals)) <= threshold and float(np.nanquantile(vals, 0.90)) <= max_p90:
                return start
        return None

    # Initial settling should mean "from here on the model behaves like the later stable regime",
    # not just "briefly dips below a threshold". Use a 2 h forward window for that.
    init_idx = first_window_stable(
        start_actual=int(idx_actual[0]),
        horizon_s=2 * 3600,
        max_p90=max(stable_p90 * 1.5, stable_level + 0.005),
    )

    # Post-gap recovery is a shorter local event. A 10 min forward window is enough here.
    recover_idx = first_window_stable(
        start_actual=int(gap_end_idx),
        horizon_s=10 * 60,
        max_p90=max(stable_p90 * 1.5, stable_level + 0.005),
    )
    return {
        'stable_level': stable_level,
        'stable_p90': stable_p90,
        'threshold': threshold,
        'initial_settle_idx': init_idx,
        'recovery_idx': recover_idx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    ap.add_argument('--max_rows', type=int, default=100000)
    ap.add_argument('--soc_batch', type=int, default=256)
    ap.add_argument('--gap_seconds', type=int, default=300)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    data_root = '/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE'
    soh_cfg_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml'
    soh_ckpt = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt'
    soh_scaler_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib'
    with open(soh_cfg_path) as f:
        soh_cfg = yaml.safe_load(f)

    df = load_df(data_root, args.cell)
    keep_cols = ['Testtime[s]','SOC','Voltage[V]','Current[A]','Temperature[°C]'] + [c for c in soh_cfg['model']['features'] if c in df.columns]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols).reset_index(drop=True)
    df = df.iloc[:args.max_rows].copy()

    gap_start = len(df) // 2
    gap_end = min(gap_start + int(args.gap_seconds), len(df))
    freeze_mask = np.zeros(len(df), dtype=bool)
    freeze_mask[gap_start:gap_end] = True

    df_clean = build_online_aux_features(
        df=df,
        freeze_mask=np.zeros(len(df), dtype=bool),
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=1.8,
        initial_soc_delta=0.0,
    )
    df_gap = build_online_aux_features(
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
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean','std','min','max'])
    soh_model = SOH_LSTM_Seq2Seq(
        in_features=len(expand_features_for_sampling(base_features, feature_aggs)),
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

    clean_soh_hourly, clean_bins = predict_soh_hourly(df_clean, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device)
    gap_soh_hourly, gap_bins = predict_soh_hourly(df_gap, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device)

    df_clean_soc = df_clean.copy(); df_clean_soc['SOH'] = expand_soh_to_rows(df_clean, clean_bins, clean_soh_hourly, interval_seconds)
    df_gap_soc = df_gap.copy(); df_gap_soc['SOH'] = expand_soh_to_rows(df_gap, gap_bins, gap_soh_hourly, interval_seconds)
    if freeze_mask.any():
        first_gap = int(np.argmax(freeze_mask))
        hold = float(df_gap_soc['SOH'].iloc[first_gap - 1]) if first_gap > 0 else 1.0
        df_gap_soc.loc[freeze_mask, 'SOH'] = hold

    variants = [
        ('SOC_1.6.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib'),
        ('SOC_1.7.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib'),
    ]

    results = {}
    for name, cfg_path, ckpt, scaler in variants:
        clean = run_soc_variant(name + '_clean', cfg_path, ckpt, scaler, df_clean_soc, device, args.soc_batch)
        gap = run_soc_variant(name + '_gap', cfg_path, ckpt, scaler, df_gap_soc, device, args.soc_batch)
        marks = find_recovery_mark(gap['index'], gap['abs_err'], gap_start, gap_end)
        results[name] = {'clean': clean, 'gap': gap, 'marks': marks}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {
        'cell': args.cell,
        'max_rows': int(len(df)),
        'device': str(device),
        'gap_start_idx': int(gap_start),
        'gap_end_idx': int(gap_end),
        'gap_seconds': int(gap_end - gap_start),
        'models': {},
    }
    for name, payload in results.items():
        m = payload['marks']
        init_min = None if m['initial_settle_idx'] is None else float(m['initial_settle_idx'] / 60.0)
        rec_min_after_gap = None if m['recovery_idx'] is None else float((m['recovery_idx'] - gap_end) / 60.0)
        summary['models'][name] = {
            'clean_mae': float(payload['clean']['mae']),
            'clean_rmse': float(payload['clean']['rmse']),
            'gap_mae': float(payload['gap']['mae']),
            'gap_rmse': float(payload['gap']['rmse']),
            'initial_settle_min_from_start': init_min,
            'recovery_min_after_gap': rec_min_after_gap,
            'threshold': float(m['threshold']),
            'stable_level': float(m['stable_level']),
            'stable_p90': float(m['stable_p90']),
        }
        pd.DataFrame({
            'index': payload['gap']['index'],
            'soc_true': payload['gap']['y_ref'],
            'soc_pred_clean': payload['clean']['pred'],
            'soc_pred_gap': payload['gap']['pred'],
            'abs_err_gap': payload['gap']['abs_err'],
        }).to_csv(out / f'{name}_freeze_gap_compare_{args.cell}.csv', index=False)

    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = {'SOC_1.6.0.0': '#6e2fc4', 'SOC_1.7.0.0': '#08bdba'}
    x_hours = (results['SOC_1.6.0.0']['gap']['index']) / 3600.0
    gap_start_h = gap_start / 3600.0
    gap_end_h = gap_end / 3600.0

    for ax, name in zip(axes, ['SOC_1.6.0.0', 'SOC_1.7.0.0']):
        payload = results[name]
        color = colors[name]
        ax.plot(x_hours, payload['gap']['y_ref'], color='black', linewidth=1.0, label='SOC true')
        ax.plot(x_hours, payload['clean']['pred'], color=color, linestyle='--', linewidth=1.0, alpha=0.8, label='clean run')
        ax.plot(x_hours, payload['gap']['pred'], color=color, linewidth=1.2, label='freeze-gap run')
        ax.axvspan(gap_start_h, gap_end_h, color='grey', alpha=0.18, label='5 min freeze')
        marks = payload['marks']
        if marks['initial_settle_idx'] is not None:
            x0 = marks['initial_settle_idx'] / 3600.0
            ax.axvline(x0, color='red', linestyle='--', linewidth=1.0)
            ax.text(x0, 0.96, f"settled ~{marks['initial_settle_idx']/60.0:.1f} min", color='red', rotation=90, va='top', ha='left', transform=ax.get_xaxis_transform())
        if marks['recovery_idx'] is not None:
            xr = marks['recovery_idx'] / 3600.0
            rec_min = (marks['recovery_idx'] - gap_end) / 60.0
            ax.axvline(xr, color='red', linestyle='--', linewidth=1.0)
            ax.text(xr, 0.62, f"recovered ~{rec_min:.1f} min after gap", color='red', rotation=90, va='top', ha='left', transform=ax.get_xaxis_transform())
        ax.set_ylabel('SOC [-]')
        ax.set_title(name)
        ax.legend(loc='best')
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('Time [h]')
    fig.tight_layout()
    fig.savefig(out / f'freeze_gap_5min_compare_{args.cell}.png', dpi=170)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
