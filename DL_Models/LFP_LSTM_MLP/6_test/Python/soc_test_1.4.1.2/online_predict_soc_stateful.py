#!/usr/bin/env python
"""
Pure stateful SOC online prediction for 1.5.0.0.
- Loads the exact checkpoint
- Uses the scaler from checkpoint/config
- Primes once with chunk-1 samples, then predicts 1 sample per step with carried hidden state
- No rolling windows, no seq2one tricks, no batching beyond 1 step, no smoothing.
"""
import os, re, json, math, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load as joblib_load
from tqdm.auto import tqdm

class LSTMMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1), nn.Sigmoid()
        )
    def forward(self, x, state=None, return_state: bool=False):
        out, new_state = self.lstm(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


def load_checkpoint(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device)
    if not isinstance(raw, dict):
        raise ValueError("Unsupported checkpoint format")
    state = raw
    cfg = state.get('config') or state.get('cfg')
    if cfg is None:
        raise KeyError("'config' not found in checkpoint")
    features = state.get('features') or state.get('feature_list')
    if features is None:
        raise KeyError("'features' not found in checkpoint")
    chunk = state.get('chunk') or state.get('window') or state.get('seq_len')
    if chunk is None:
        raise KeyError("'chunk/window/seq_len' not in checkpoint")
    model = LSTMMLP(
        in_features=len(features),
        hidden_size=int(cfg['model']['hidden_size']),
        mlp_hidden=int(cfg['model']['mlp_hidden']),
        num_layers=int(cfg['model'].get('num_layers', 1)),
        dropout=float(cfg['model'].get('dropout', 0.05))
    ).to(device)
    msd = state.get('model_state_dict') or state.get('state_dict') or state.get('model')
    if msd is None: raise KeyError("No weights in checkpoint")
    model.load_state_dict(msd, strict=False)
    model.eval()
    scaler_path = state.get('scaler_path') or os.path.join(cfg.get('paths',{}).get('out_root','.'),'scaler_robust.joblib')
    return model, features, int(chunk), scaler_path


def locate_cell_parquet(data_root: str, cell: str):
    m = re.search(r'_C(\d{2})$', cell)
    if not m: raise ValueError(f"Cell format invalid: {cell}")
    c2 = m.group(1)
    cands = [
        os.path.join(data_root, f'df_FE_C{c2}.parquet'),
        os.path.join(data_root, f'df_FE_{c2}.parquet'),
        os.path.join(data_root, f'MGFarm_18650_C{c2}.parquet'),
    ]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"None found: {cands}")


def run_stateful(model, features, scaler, df: pd.DataFrame, chunk: int, device: torch.device, max_preds: int=-1):
    clean = df.replace([np.inf,-np.inf], np.nan).dropna(subset=features+['SOC'])
    X = clean[features].to_numpy(dtype=np.float32)
    y = clean['SOC'].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    total = len(Xs)
    preds = []
    state = None
    with torch.no_grad():
        start = 0
        # prime if possible
        if total >= chunk:
            prime = torch.from_numpy(Xs[:chunk-1]).unsqueeze(0).to(device)
            _, state = model(prime, state=None, return_state=True)
            start = chunk - 1
        # step by step
        target = (total - start) if max_preds <= 0 else min(total - start, max_preds)
        pbar = tqdm(total=target, unit="step", desc="stateful", dynamic_ncols=True)
        for i in range(start, total):
            xi = torch.from_numpy(Xs[i:i+1]).unsqueeze(0).to(device)
            out, state = model(xi, state=state, return_state=True)
            preds.append(float(out.item()))
            pbar.update(1)
            if max_preds>0 and len(preds)>=max_preds: break
        pbar.close()
    y_valid = y[start:start+len(preds)]
    return np.array(preds), y_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1.5.0.0/outputs/checkpoints/soc_epoch0001_rmse0.02897.pt')
    ap.add_argument('--data_root', type=str, default='/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE')
    ap.add_argument('--cell', type=str, default='MGFarm_18650_C07')
    ap.add_argument('--out_dir', type=str, default='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/Tests/soc_test_1.4.1.2/soct_test_1.5.0.0_first_50000_statefull1')
    ap.add_argument('--max_preds', type=int, default=50000)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, features, chunk, scaler_path = load_checkpoint(args.checkpoint, device)
    scaler = joblib_load(scaler_path)
    cell_parquet = locate_cell_parquet(args.data_root, args.cell)
    df = pd.read_parquet(cell_parquet)

    preds, y_true = run_stateful(model, features, scaler, df, chunk, device, max_preds=args.max_preds)

    # metrics & save
    mets = {
        'rmse': math.sqrt(mean_squared_error(y_true, preds)),
        'mae': mean_absolute_error(y_true, preds),
        'r2': r2_score(y_true, preds),
        'n': int(len(preds))
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(mets, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir,'preds_true_first.npz'), y_true=y_true, y_pred=preds)
    # plots
    try:
        x = np.arange(len(preds))
        plt.figure(figsize=(12,5))
        plt.plot(x, y_true, label='SOC true', lw=1.5)
        plt.plot(x, preds, label='SOC pred', lw=1)
        plt.xlabel('step')
        plt.ylabel('SOC')
        plt.title(f"SOC stateful online 1.5.0.0 – {args.cell} – n={len(preds)}, rmse={mets['rmse']:.4f}, mae={mets['mae']:.4f}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'soc_pred_vs_true.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(12,4))
        resid = preds - y_true
        plt.plot(x, resid, label='pred - true', lw=1)
        plt.axhline(0, color='k', lw=0.8)
        plt.xlabel('step'); plt.ylabel('residual')
        plt.title('Residuals')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'soc_residuals.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")
    print(json.dumps({'cell': args.cell, 'chunk': chunk, **mets}, indent=2))

if __name__=='__main__':
    main()
