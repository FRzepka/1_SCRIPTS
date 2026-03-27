import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from joblib import load
from tqdm import tqdm


class LSTMMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :]).squeeze(-1)


class GRUMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, dropout=gru_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

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
        self.num_directions = 2 if bidirectional else 1
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        lstm_out = hidden_size * self.num_directions
        self.post_norm = nn.LayerNorm(lstm_out)
        self.res_blocks = nn.ModuleList([ResidualMLPBlock(lstm_out, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))])
        self.head = nn.Sequential(
            nn.Linear(lstm_out, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x, state=None, return_state: bool = False):
        x = self.feature_proj(x)
        out, new_state = self.lstm(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y = self.head(out).squeeze(-1)
        if return_state:
            return y, new_state
        return y


def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
    path = Path(data_root) / f"df_FE_{cell.split('_')[-1]}.parquet"
    if not path.exists():
        path = Path(data_root) / f"df_FE_{cell}.parquet"
    if not path.exists():
        path = Path(data_root) / f"df_FE_C{cell[-3:]}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def expand_features_for_sampling(base_features: List[str], feature_aggs: List[str]) -> List[str]:
    return [f'{feat}_{agg}' for feat in base_features for agg in feature_aggs]


def aggregate_hourly(df: pd.DataFrame, base_features: List[str], interval_seconds: int, feature_aggs: List[str]) -> pd.DataFrame:
    work = df[base_features + ['Testtime[s]']].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work.sort_values('Testtime[s]')
    work['_bin'] = (work['Testtime[s]'] // interval_seconds).astype(np.int64)
    agg_spec = {feat: feature_aggs for feat in base_features}
    out = work.groupby('_bin', sort=True).agg(agg_spec)
    out.columns = [f'{c[0]}_{c[1]}' for c in out.columns]
    out['bin'] = out.index
    return out.reset_index(drop=True)


def predict_soh_hourly(df_raw: pd.DataFrame, base_features: List[str], interval_seconds: int, feature_aggs: List[str], scaler: RobustScaler, model: SOH_LSTM_Seq2Seq, device: torch.device, soh_init: float = 1.0):
    hourly = aggregate_hourly(df_raw, base_features, interval_seconds, feature_aggs)
    feat_cols = expand_features_for_sampling(base_features, feature_aggs)
    X = hourly[feat_cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    preds = []
    state = None
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(hourly)), desc='SOH hourly', leave=False):
            x_step = torch.from_numpy(Xs[i:i+1]).unsqueeze(1).to(device)
            y_seq, state = model(x_step, state=state, return_state=True)
            y_val = float(y_seq.squeeze().detach().cpu().numpy())
            preds.append(float(soh_init) if i == 0 else y_val)
    return np.array(preds, dtype=np.float32), hourly['bin'].to_numpy(dtype=np.int64)


def expand_soh_to_rows(df_raw: pd.DataFrame, bins: np.ndarray, soh_preds: np.ndarray, interval_seconds: int, soh_init: float = 1.0) -> np.ndarray:
    bin_to_soh = {int(b): float(s) for b, s in zip(bins, soh_preds)}
    if len(bins):
        last = soh_init
        for b in range(int(bins.min()), int(bins.max()) + 1):
            if b in bin_to_soh:
                last = bin_to_soh[b]
            else:
                bin_to_soh[b] = last
    out = []
    for t in df_raw['Testtime[s]'].to_numpy(dtype=np.float64):
        out.append(bin_to_soh.get(int(t // interval_seconds), soh_init))
    return np.array(out, dtype=np.float32)


def engineer_soc_features(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    time_s = df['Testtime[s]'].to_numpy(dtype=np.float32)
    voltage = df['Voltage[V]'].to_numpy(dtype=np.float32)
    current = df['Current[A]'].to_numpy(dtype=np.float32)
    temperature = df['Temperature[°C]'].to_numpy(dtype=np.float32)
    soh = df['SOH'].to_numpy(dtype=np.float32)
    q_c = df['Q_c'].to_numpy(dtype=np.float32)

    if len(time_s) > 1:
        dt_s = np.empty_like(time_s)
        dt_s[0] = max(float(time_s[1] - time_s[0]), 1e-6)
        dt_s[1:] = np.diff(time_s)
    else:
        dt_s = np.ones_like(time_s, dtype=np.float32)
    dt_safe = np.maximum(dt_s, 1e-6)

    d_u_dt = np.zeros_like(voltage, dtype=np.float32)
    d_i_dt = np.zeros_like(current, dtype=np.float32)
    if len(voltage) > 1:
        d_u_dt[1:] = np.diff(voltage) / dt_safe[1:]
        d_i_dt[1:] = np.diff(current) / dt_safe[1:]

    feat_map = {
        'Voltage[V]': voltage,
        'Current[A]': current,
        'Temperature[°C]': temperature,
        'SOH': soh,
        'Q_c': q_c,
        'dU_dt[V/s]': d_u_dt,
        'dI_dt[A/s]': d_i_dt,
        'dt_s': dt_s,
    }
    return np.column_stack([feat_map[f] for f in features]).astype(np.float32)


def build_soc_model(cfg: dict, device: torch.device):
    mcfg = cfg['model']
    kind = mcfg.get('type', 'LSTM_MLP')
    kwargs = dict(
        in_features=len(mcfg['features']),
        hidden_size=int(mcfg['hidden_size']),
        mlp_hidden=int(mcfg['mlp_hidden']),
        num_layers=int(mcfg.get('num_layers', 1)),
        dropout=float(mcfg.get('dropout', 0.05)),
    )
    if kind == 'GRU_MLP':
        return GRUMLP(**kwargs).to(device)
    return LSTMMLP(**kwargs).to(device)


def rolling_predict_soc(Xs: np.ndarray, y_true: np.ndarray, chunk: int, model: nn.Module, device: torch.device, batch_size: int = 256):
    n_pred = len(Xs) - chunk + 1
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_pred, batch_size), desc='SOC rolling', leave=False):
            end = min(i + batch_size, n_pred)
            xb = np.stack([Xs[j:j+chunk] for j in range(i, end)], axis=0)
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_ref = y_true[chunk - 1:]
    rmse = float(math.sqrt(np.mean((y_ref - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_ref - y_pred)))
    return y_pred, rmse, mae


def run_soc_variant(name: str, soc_cfg_path: str, soc_ckpt: str, soc_scaler_path: str, df_soc: pd.DataFrame, device: torch.device, batch_size: int):
    import yaml
    with open(soc_cfg_path) as f:
        soc_cfg = yaml.safe_load(f)
    features = soc_cfg['model']['features']
    chunk = int(soc_cfg['training']['seq_chunk_size'])
    scaler = load(soc_scaler_path)
    X = engineer_soc_features(df_soc, features)
    Xs = scaler.transform(X).astype(np.float32)
    y_true = df_soc['SOC'].to_numpy(dtype=np.float32)
    model = build_soc_model(soc_cfg, device)
    state = torch.load(soc_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    pred, rmse, mae = rolling_predict_soc(Xs, y_true, chunk, model, device, batch_size=batch_size)
    return {
        'name': name,
        'config': soc_cfg_path,
        'checkpoint': soc_ckpt,
        'scaler': soc_scaler_path,
        'features': features,
        'chunk': chunk,
        'rmse': rmse,
        'mae': mae,
        'pred': pred,
        'y_true': y_true[chunk - 1:],
        'index': np.arange(chunk - 1, len(df_soc)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    ap.add_argument('--max_rows', type=int, default=100000)
    ap.add_argument('--device', default=None)
    ap.add_argument('--soc_batch', type=int, default=256)
    ap.add_argument('--soh_init', type=float, default=1.0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    data_root = '/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE'
    soh_cfg_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml'
    soh_ckpt = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt'
    soh_scaler_path = '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib'

    import yaml
    with open(soh_cfg_path) as f:
        soh_cfg = yaml.safe_load(f)

    df = load_cell_dataframe(data_root, args.cell)
    df = df.head(args.max_rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOC', 'Q_c']).reset_index(drop=True)

    base_features = soh_cfg['model']['features']
    sampling_cfg = soh_cfg.get('sampling', {})
    interval_seconds = int(sampling_cfg.get('interval_seconds', 3600))
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean', 'std', 'min', 'max'])

    soh_scaler = load(soh_scaler_path)
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

    soh_hourly, bins = predict_soh_hourly(df, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device, args.soh_init)
    soh_per_row = expand_soh_to_rows(df, bins, soh_hourly, interval_seconds, args.soh_init)
    df_soc = df.copy()
    df_soc['SOH'] = soh_per_row

    baseline = run_soc_variant(
        name='SOC_1.6.0.0',
        soc_cfg_path='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml',
        soc_ckpt='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt',
        soc_scaler_path='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib',
        df_soc=df_soc,
        device=device,
        batch_size=args.soc_batch,
    )
    new = run_soc_variant(
        name='SOC_1.7.0.0',
        soc_cfg_path='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml',
        soc_ckpt='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt',
        soc_scaler_path='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib',
        df_soc=df_soc,
        device=device,
        batch_size=args.soc_batch,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'bin': bins, 'time_s': bins * interval_seconds, 'soh_pred': soh_hourly}).to_csv(out_dir / f'soh_hourly_{args.cell}.csv', index=False)
    pd.DataFrame({'index': baseline['index'], 'soc_true': baseline['y_true'], 'soc_pred_160': baseline['pred'], 'soc_pred_170': new['pred']}).to_csv(out_dir / f'soc_compare_{args.cell}.csv', index=False)

    summary = {
        'cell': args.cell,
        'max_rows': int(len(df)),
        'device': str(device),
        'soh_hours': int(len(soh_hourly)),
        'baseline': {k: baseline[k] for k in ['name', 'config', 'checkpoint', 'scaler', 'chunk', 'rmse', 'mae']},
        'new_model': {k: new[k] for k in ['name', 'config', 'checkpoint', 'scaler', 'chunk', 'rmse', 'mae']},
        'delta_mae_new_minus_old': float(new['mae'] - baseline['mae']),
        'delta_rmse_new_minus_old': float(new['rmse'] - baseline['rmse']),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.plot(baseline['index'], baseline['y_true'], label='SOC true', linewidth=1.0, color='black')
    ax.plot(baseline['index'], baseline['pred'], label=f"1.6.0.0 (MAE={baseline['mae']:.4f})", linewidth=1.0, alpha=0.9)
    ax.plot(new['index'], new['pred'], label=f"1.7.0.0 (MAE={new['mae']:.4f})", linewidth=1.0, alpha=0.9)
    ax.set_xlabel('Index')
    ax.set_ylabel('SOC [-]')
    ax.set_title(f'SOC+SOH quick compare ({args.cell}, first {len(df):,} rows)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / f'soc_compare_{args.cell}.png', dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.plot(bins * interval_seconds / 3600.0, soh_hourly, linewidth=1.0)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('SOH [-]')
    ax.set_title(f'SOH hourly ({args.cell})')
    fig.tight_layout()
    fig.savefig(out_dir / f'soh_hourly_{args.cell}.png', dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
