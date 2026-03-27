import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler


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


def load_parquet(data_root: str, cell: str) -> pd.DataFrame:
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


def main():
    ap = argparse.ArgumentParser(description="Standalone SOC predictor (1.6.0.0).")
    ap.add_argument("--config", required=True, help="Path to train_soc.yaml (for features/chunk sizes)")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--scaler", required=True, help="Path to scaler_robust.joblib")
    ap.add_argument("--data_root", required=True, help="Root of parquet FE data")
    ap.add_argument("--cell", required=True, help="Cell id, e.g. MGFarm_18650_C07")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--n_rows", type=int, default=10000, help="First N rows to use")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    features = cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    dropout = float(cfg['model'].get('dropout', 0.05))
    chunk = int(cfg['training']['seq_chunk_size'])

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    from joblib import load
    scaler: RobustScaler = load(args.scaler)

    df = load_parquet(args.data_root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    df = df.reset_index(drop=True)
    df = df.iloc[:args.n_rows].copy()

    if len(df) < chunk:
        raise ValueError(f"Not enough rows ({len(df)}) for chunk size {chunk}.")

    X = df[features].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    n_pred = len(df) - chunk + 1
    xs = np.stack([Xs[i:i+chunk] for i in range(n_pred)], axis=0)

    model = LSTMMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden,
                    num_layers=num_layers, dropout=dropout).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    preds = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, n_pred, batch_size):
            xb = torch.from_numpy(xs[i:i+batch_size]).to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds) if preds else np.array([])

    out_df = pd.DataFrame({
        'index': np.arange(chunk-1, len(df)),
        'soc_pred': y_pred,
    })
    out_df.to_csv(args.out_csv, index=False)

    meta = {
        'cell': args.cell,
        'n_rows': int(args.n_rows),
        'chunk': int(chunk),
        'n_pred': int(n_pred),
        'checkpoint': args.checkpoint,
        'scaler': args.scaler,
        'features': features,
        'device': str(device),
    }
    meta_path = os.path.splitext(args.out_csv)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
