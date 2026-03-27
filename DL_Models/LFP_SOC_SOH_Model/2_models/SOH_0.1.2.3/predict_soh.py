import os
import argparse
import json
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler


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
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        if bidirectional:
            print('Warning: bidirectional=True breaks true stateful inference.')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out = hidden_size * self.num_directions
        self.post_norm = nn.LayerNorm(lstm_out)
        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(lstm_out, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_out, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, state=None, return_state: bool = False):
        x = self.feature_proj(x)
        out, new_state = self.lstm(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y_seq = self.head(out).squeeze(-1)
        if return_state:
            return y_seq, new_state
        return y_seq


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


def expand_features_for_sampling(base_features: List[str], feature_aggs: List[str]) -> List[str]:
    return [f'{feat}_{agg}' for feat in base_features for agg in feature_aggs]


def aggregate_hourly(df: pd.DataFrame, base_features: List[str], interval_seconds: int, feature_aggs: List[str]) -> pd.DataFrame:
    if 'Testtime[s]' not in df.columns:
        raise ValueError("Testtime[s] column required for hourly aggregation")
    work = df[base_features + ['Testtime[s]']].replace([np.inf, -np.inf], np.nan).dropna(subset=base_features + ['Testtime[s]']).copy()
    work = work.sort_values('Testtime[s]')
    if work.empty:
        return work
    bins = (work['Testtime[s]'] // interval_seconds).astype(np.int64)
    work['_bin'] = bins
    agg_spec = {feat: feature_aggs for feat in base_features}
    out = work.groupby('_bin', sort=True).agg(agg_spec)
    out.columns = [f'{col[0]}_{col[1]}' for col in out.columns]
    out['bin'] = out.index
    out = out.reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Predict hourly SOH using model 0.1.2.3.")
    ap.add_argument("--config", required=True, help="Path to train_soh.yaml")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--scaler", required=True, help="Path to scaler_robust.joblib")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    base_features = cfg['model']['features']
    embed_size = int(cfg['model']['embed_size'])
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 2))
    res_blocks = int(cfg['model'].get('res_blocks', 2))
    bidirectional = bool(cfg['model'].get('bidirectional', False))
    dropout = float(cfg['model'].get('dropout', 0.15))

    sampling_cfg = cfg.get('sampling', {})
    interval_seconds = int(sampling_cfg.get('interval_seconds', 3600))
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean', 'std', 'min', 'max'])

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    df = load_cell_dataframe(args.data_root, args.cell)
    hourly = aggregate_hourly(df, base_features, interval_seconds, feature_aggs)
    if hourly.empty:
        raise ValueError("No data after hourly aggregation.")

    feature_cols = expand_features_for_sampling(base_features, feature_aggs)
    X = hourly[feature_cols].to_numpy(dtype=np.float32)

    from joblib import load
    scaler: RobustScaler = load(args.scaler)
    Xs = scaler.transform(X).astype(np.float32)

    model = SOH_LSTM_Seq2Seq(
        in_features=len(feature_cols),
        embed_size=embed_size,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        res_blocks=res_blocks,
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    preds = []
    stateful = None
    with torch.no_grad():
        for i in range(len(Xs)):
            x_step = torch.from_numpy(Xs[i:i+1]).unsqueeze(1).to(device)
            y_seq, stateful = model(x_step, state=stateful, return_state=True)
            preds.append(float(y_seq.squeeze().detach().cpu().numpy()))

    out = pd.DataFrame({
        'bin': hourly['bin'],
        'time_s': hourly['bin'] * interval_seconds,
        'soh_pred': preds,
    })
    out.to_csv(args.out_csv, index=False)

    meta = {
        'cell': args.cell,
        'device': str(device),
        'interval_seconds': int(interval_seconds),
        'feature_aggs': feature_aggs,
        'checkpoint': args.checkpoint,
        'scaler': args.scaler,
        'config': args.config,
    }
    meta_path = os.path.splitext(args.out_csv)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
