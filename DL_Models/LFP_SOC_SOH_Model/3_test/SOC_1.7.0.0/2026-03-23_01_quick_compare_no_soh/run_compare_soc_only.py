import json, math, argparse
from pathlib import Path
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

def load_df(root, cell):
    for name in [f'df_FE_{cell.split("_")[-1]}.parquet', f'df_FE_{cell}.parquet', f'df_FE_C{cell[-3:]}.parquet']:
        p = Path(root) / name
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(cell)

def prepare_features(df, features):
    work = df.copy()
    if 'dt_s' in features and 'dt_s' not in work.columns:
        t = work['Testtime[s]'].to_numpy(dtype=np.float32)
        dt = np.empty_like(t)
        if len(t) > 1:
            dt[0] = max(float(t[1]-t[0]), 1e-6)
            dt[1:] = np.diff(t)
        else:
            dt[0] = 1.0
        work['dt_s'] = dt
    return work[features].to_numpy(dtype=np.float32)

def run_variant(name, cfg_path, ckpt_path, scaler_path, df, device, batch_size):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    features = cfg['model']['features']
    chunk = int(cfg['training']['seq_chunk_size'])
    X = prepare_features(df, features)
    Xs = load(scaler_path).transform(X).astype(np.float32)
    y = df['SOC'].to_numpy(dtype=np.float32)
    kind = cfg['model'].get('type', 'LSTM_MLP')
    model_cls = GRUMLP if kind == 'GRU_MLP' else LSTMMLP
    model = model_cls(len(features), int(cfg['model']['hidden_size']), int(cfg['model']['mlp_hidden']), int(cfg['model'].get('num_layers', 1)), float(cfg['model'].get('dropout', 0.05))).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    n_pred = len(df) - chunk + 1
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
    return {'name': name, 'features': features, 'chunk': chunk, 'rmse': rmse, 'mae': mae, 'pred': pred, 'y_ref': y_ref, 'index': np.arange(chunk - 1, len(df))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    ap.add_argument('--max_rows', type=int, default=100000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    root = '/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE'
    df = load_df(root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Testtime[s]', 'SOC', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH', 'Q_c', 'dU_dt[V/s]', 'dI_dt[A/s]']).reset_index(drop=True)
    df = df.iloc[:args.max_rows].copy()
    old = run_variant('SOC_1.6.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib', df, device, args.batch_size)
    new = run_variant('SOC_1.7.0.0', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt', '/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib', df, device, args.batch_size)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': old['index'], 'soc_true': old['y_ref'], 'soc_pred_160': old['pred'], 'soc_pred_170': new['pred']}).to_csv(out / f'soc_compare_{args.cell}.csv', index=False)
    summary = {
        'cell': args.cell,
        'max_rows': int(len(df)),
        'device': str(device),
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
    ax.set_title(f'SOC-only compare ({args.cell}, first {len(df):,} rows)')
    ax.set_xlabel('Index'); ax.set_ylabel('SOC [-]'); ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out / f'soc_compare_{args.cell}.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
