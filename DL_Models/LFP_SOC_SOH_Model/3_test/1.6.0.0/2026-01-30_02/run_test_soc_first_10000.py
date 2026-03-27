import os
import re
import json
import yaml
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Model (same as training) ---
class LSTMMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        return pred


def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
    path = os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet")
    if not os.path.exists(path):
        path = os.path.join(data_root, f"df_FE_{cell}.parquet")
    if not os.path.exists(path):
        cid = cell[-3:]
        alt = os.path.join(data_root, f"df_FE_C{cid}.parquet")
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate parquet for cell {cell} in {data_root}")
    return pd.read_parquet(path)


def find_best_checkpoint(ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    best = None
    best_rmse = None
    any_pt = []
    for name in os.listdir(ckpt_dir):
        if not name.endswith('.pt'):
            continue
        any_pt.append(name)
        m = re.search(r"rmse([0-9]+\\.[0-9]+)", name)
        if m:
            rmse = float(m.group(1))
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best = os.path.join(ckpt_dir, name)
    if best is None and any_pt:
        # fallback to first .pt if no rmse pattern is found
        best = os.path.join(ckpt_dir, sorted(any_pt)[-1])
    if best is None:
        raise FileNotFoundError("No checkpoint .pt found in checkpoint directory.")
    return best


def main():
    ap = argparse.ArgumentParser(description="Predict SOC over the first N rows and save results.")
    ap.add_argument("--config", required=True, help="Path to train_soc.yaml")
    ap.add_argument("--out_dir", required=True, help="Output directory for test results")
    ap.add_argument("--cell", default=None, help="Cell id to use (e.g., MGFarm_18650_C07). Defaults to first val cell.")
    ap.add_argument("--n_rows", type=int, default=10000, help="Number of initial rows to evaluate")
    ap.add_argument("--full_cell", action="store_true", help="Use full cell length (ignores --n_rows)")
    ap.add_argument("--device", default=None, help="cuda or cpu. Default: auto")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['paths']['data_root']
    features: List[str] = cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    dropout = float(cfg['model'].get('dropout', 0.05))
    chunk = int(cfg['training']['seq_chunk_size'])

    cell = args.cell or cfg['cells']['val'][0]

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # load scaler
    scaler_path = os.path.join(cfg['paths']['out_root'], 'scaler_robust.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    from joblib import load
    scaler: RobustScaler = load(scaler_path)

    # load data
    df = load_cell_dataframe(data_root, cell)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['SOC'])
    df = df.reset_index(drop=True)
    if args.full_cell:
        n_rows_used = len(df)
    else:
        df = df.iloc[:args.n_rows].copy()
        n_rows_used = len(df)

    if len(df) < chunk:
        raise ValueError(f"Not enough rows ({len(df)}) for chunk size {chunk}.")

    # scale features
    X = df[features].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    y_true = df['SOC'].to_numpy(dtype=np.float32)

    # build sequences for rolling predictions
    n_pred = len(df) - chunk + 1

    # model
    ckpt_dir = os.path.join(cfg['paths']['out_root'], 'checkpoints')
    ckpt_path = find_best_checkpoint(ckpt_dir)

    model = LSTMMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden,
                    num_layers=num_layers, dropout=dropout).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # predict in batches (streaming to avoid huge RAM use)
    batch_size = int(args.batch_size)
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, n_pred, batch_size), desc="Predict", unit="batch"):
            end = min(i + batch_size, n_pred)
            xb_np = np.stack([Xs[j:j+chunk] for j in range(i, end)], axis=0)
            xb = torch.from_numpy(xb_np).to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds) if preds else np.array([])

    # metrics
    rmse = math.sqrt(np.mean((y_true[chunk-1:] - y_pred) ** 2))
    mae = np.mean(np.abs(y_true[chunk-1:] - y_pred))

    # save outputs
    if args.full_cell:
        result_csv = os.path.join(out_dir, f"soc_pred_fullcell_{cell}.csv")
    else:
        result_csv = os.path.join(out_dir, f"soc_pred_first_{n_rows_used}_rows_{cell}.csv")
    out_df = pd.DataFrame({
        'index': np.arange(chunk-1, len(df)),
        'soc_true': y_true[chunk-1:],
        'soc_pred': y_pred,
    })
    out_df.to_csv(result_csv, index=False)

    summary = {
        'cell': cell,
        'n_rows': int(n_rows_used),
        'chunk': int(chunk),
        'n_pred': int(n_pred),
        'rmse': float(rmse),
        'mae': float(mae),
        'config': args.config,
        'scaler': scaler_path,
        'checkpoint': ckpt_path,
        'features': features,
        'device': str(device),
    }
    with open(os.path.join(out_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # plot
    try:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(out_df['index'], out_df['soc_true'], label='SOC true', linewidth=1.0)
        ax.plot(out_df['index'], out_df['soc_pred'], label='SOC pred', linewidth=1.0, alpha=0.8)
        if args.full_cell:
            ax.set_title(f"SOC Prediction – Full Cell ({cell})")
        else:
            ax.set_title(f"SOC Prediction – First {n_rows_used} Rows ({cell})")
        ax.set_xlabel('Index')
        ax.set_ylabel('SOC')
        ax.legend(loc='best')
        fig.tight_layout()
        if args.full_cell:
            plot_path = os.path.join(out_dir, f"soc_pred_fullcell_{cell}.png")
        else:
            plot_path = os.path.join(out_dir, f"soc_pred_first_{n_rows_used}_rows_{cell}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
