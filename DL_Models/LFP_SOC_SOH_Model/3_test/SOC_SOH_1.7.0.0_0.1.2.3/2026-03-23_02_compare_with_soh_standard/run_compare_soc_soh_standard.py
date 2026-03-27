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
    y_ref = y[chunk - 1:]
    rmse = float(math.sqrt(np.mean((y_ref - pred) ** 2)))
    mae = float(np.mean(np.abs(y_ref - pred)))
    return {'name': name, 'features': features, 'chunk': chunk, 'rmse': rmse, 'mae': mae, 'pred': pred, 'y_ref': y_ref, 'index': np.arange(chunk - 1, len(df_soc))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    ap.add_argument('--max_rows', type=int, default=100000)
    ap.add_argument('--soc_batch', type=int, default=256)
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Testtime[s]','SOC','Voltage[V]','Current[A]','Temperature[°C]','Q_c','dU_dt[V/s]','dI_dt[A/s]']).reset_index(drop=True)
    df = df.iloc[:args.max_rows].copy()
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
    soh_hourly, bins = predict_soh_hourly(df, base_features, interval_seconds, feature_aggs, soh_scaler, soh_model, device)
    df_soc = df.copy()
    df_soc['SOH'] = expand_soh_to_rows(df, bins, soh_hourly, interval_seconds)
    old = run_soc_variant('SOC_1.6.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib', df_soc, device, args.soc_batch)
    new = run_soc_variant('SOC_1.7.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib', df_soc, device, args.soc_batch)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'bin': bins, 'time_s': bins * interval_seconds, 'soh_pred': soh_hourly}).to_csv(out / f'soh_hourly_{args.cell}.csv', index=False)
    pd.DataFrame({'index': old['index'], 'soc_true': old['y_ref'], 'soc_pred_160': old['pred'], 'soc_pred_170': new['pred']}).to_csv(out / f'soc_compare_{args.cell}.csv', index=False)
    summary = {
        'cell': args.cell,
        'max_rows': int(len(df)),
        'device': str(device),
        'soh_hours': int(len(soh_hourly)),
        'baseline': {k: old[k] for k in ['name','features','chunk','rmse','mae']},
        'new_model': {k: new[k] for k in ['name','features','chunk','rmse','mae']},
        'delta_mae_new_minus_old': float(new['mae'] - old['mae']),
        'delta_rmse_new_minus_old': float(new['rmse'] - old['rmse']),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.plot(old['index'], old['y_ref'], label='SOC true', linewidth=1.0, color='black')
    ax.plot(old['index'], old['pred'], label=f"1.6.0.0 (MAE={old['mae']:.4f})", linewidth=1.0, alpha=0.9)
    ax.plot(new['index'], new['pred'], label=f"1.7.0.0 (MAE={new['mae']:.4f})", linewidth=1.0, alpha=0.9)
    ax.set_title(f'SOC+SOH standard compare ({args.cell}, first {len(df):,} rows)')
    ax.set_xlabel('Index'); ax.set_ylabel('SOC [-]'); ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out / f'soc_compare_{args.cell}.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
