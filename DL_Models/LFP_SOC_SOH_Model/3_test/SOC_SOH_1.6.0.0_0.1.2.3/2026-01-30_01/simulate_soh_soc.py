import os
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# SOC model (1.6.0.0)
# -------------------------
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

# -------------------------
# SOH model (0.1.2.3)
# -------------------------
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

# -------------------------
# Helpers
# -------------------------

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


def predict_soh_hourly(
    df_raw: pd.DataFrame,
    base_features: List[str],
    interval_seconds: int,
    feature_aggs: List[str],
    scaler: RobustScaler,
    model: SOH_LSTM_Seq2Seq,
    device: torch.device,
    soh_init: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    hourly = aggregate_hourly(df_raw, base_features, interval_seconds, feature_aggs)
    if hourly.empty:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    feature_cols = expand_features_for_sampling(base_features, feature_aggs)
    X = hourly[feature_cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    soh_preds = []
    state = None
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(hourly)), desc="SOH hourly", unit="hour"):
            x_step = torch.from_numpy(Xs[i:i+1, :]).unsqueeze(1).to(device)  # [1,1,F]
            y_seq, state = model(x_step, state=state, return_state=True)
            y_val = float(y_seq.squeeze().detach().cpu().numpy())
            if i == 0:
                soh_preds.append(float(soh_init))
            else:
                soh_preds.append(y_val)

    bins = hourly['bin'].to_numpy(dtype=np.int64)
    return np.array(soh_preds, dtype=np.float32), bins


def expand_soh_to_rows(df_raw: pd.DataFrame, bins: np.ndarray, soh_preds: np.ndarray, interval_seconds: int, soh_init: float = 1.0) -> np.ndarray:
    if 'Testtime[s]' not in df_raw.columns:
        raise ValueError("Testtime[s] column required")
    bin_to_soh: Dict[int, float] = {int(b): float(s) for b, s in zip(bins, soh_preds)}
    # forward fill missing bins
    if len(bins) > 0:
        max_bin = int(np.max(bins))
        last = soh_init
        for b in range(0, max_bin + 1):
            if b in bin_to_soh:
                last = bin_to_soh[b]
            else:
                bin_to_soh[b] = last
    out = []
    for t in df_raw['Testtime[s]'].to_numpy(dtype=np.float64):
        b = int(t // interval_seconds)
        out.append(bin_to_soh.get(b, soh_init))
    return np.array(out, dtype=np.float32)


def rolling_predict_soc(
    Xs: np.ndarray,
    y_true: np.ndarray,
    chunk: int,
    model: LSTMMLP,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, float, float]:
    n_pred = len(Xs) - chunk + 1
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_pred, batch_size), desc="SOC rolling", unit="batch"):
            end = min(i + batch_size, n_pred)
            xb_np = np.stack([Xs[j:j+chunk] for j in range(i, end)], axis=0)
            xb = torch.from_numpy(xb_np).to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_ref = y_true[chunk-1:]
    rmse = math.sqrt(np.mean((y_ref - y_pred) ** 2)) if len(y_pred) else float('nan')
    mae = np.mean(np.abs(y_ref - y_pred)) if len(y_pred) else float('nan')
    return y_pred, rmse, mae

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate SOC with hourly SOH updates (SOH model 0.1.2.3 -> SOC model 1.6.0.0)")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    ap.add_argument("--soc_batch", type=int, default=64)
    ap.add_argument("--soh_init", type=float, default=1.0)

    ap.add_argument("--soc_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/config/train_soc.yaml")
    ap.add_argument("--soc_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/outputs/checkpoints/soc_epoch0005_rmse0.01393.pt")
    ap.add_argument("--soc_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/outputs/scaler_robust.joblib")

    ap.add_argument("--soh_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.3/config/train_soh.yaml")
    ap.add_argument("--soh_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.3/outputs/soh/run_20260109_163030/checkpoints/best_epoch0093_rmse0.02165.pt")
    ap.add_argument("--soh_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.3/outputs/soh/run_20260109_163030/scaler_robust.joblib")

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load SOC config
    import yaml
    with open(args.soc_config, 'r') as f:
        soc_cfg = yaml.safe_load(f)
    soc_features = soc_cfg['model']['features']
    soc_chunk = int(soc_cfg['training']['seq_chunk_size'])
    soc_hidden = int(soc_cfg['model']['hidden_size'])
    soc_mlp_hidden = int(soc_cfg['model']['mlp_hidden'])
    soc_num_layers = int(soc_cfg['model'].get('num_layers', 1))
    soc_dropout = float(soc_cfg['model'].get('dropout', 0.05))
    data_root = soc_cfg['paths']['data_root']

    # Load SOH config
    with open(args.soh_config, 'r') as f:
        soh_cfg = yaml.safe_load(f)
    soh_base_features = soh_cfg['model']['features']
    soh_embed = int(soh_cfg['model']['embed_size'])
    soh_hidden = int(soh_cfg['model']['hidden_size'])
    soh_mlp_hidden = int(soh_cfg['model']['mlp_hidden'])
    soh_num_layers = int(soh_cfg['model'].get('num_layers', 2))
    soh_res_blocks = int(soh_cfg['model'].get('res_blocks', 2))
    soh_bidirectional = bool(soh_cfg['model'].get('bidirectional', False))
    soh_dropout = float(soh_cfg['model'].get('dropout', 0.15))
    sampling_cfg = soh_cfg.get('sampling', {})
    interval_seconds = int(sampling_cfg.get('interval_seconds', 3600))
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean', 'std', 'min', 'max'])

    # Load raw data
    df = load_cell_dataframe(data_root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=list(set(soc_features + ['SOC', 'Testtime[s]'])))
    df = df.reset_index(drop=True)

    # Load scalers
    from joblib import load
    soc_scaler: RobustScaler = load(args.soc_scaler)
    soh_scaler: RobustScaler = load(args.soh_scaler)

    # Build SOH model
    soh_model = SOH_LSTM_Seq2Seq(
        in_features=len(expand_features_for_sampling(soh_base_features, feature_aggs)),
        embed_size=soh_embed,
        hidden_size=soh_hidden,
        mlp_hidden=soh_mlp_hidden,
        num_layers=soh_num_layers,
        res_blocks=soh_res_blocks,
        bidirectional=soh_bidirectional,
        dropout=soh_dropout,
    ).to(device)
    soh_state = torch.load(args.soh_ckpt, map_location=device)
    soh_model.load_state_dict(soh_state['model_state_dict'])

    # Predict SOH hourly
    soh_hourly, bins = predict_soh_hourly(
        df_raw=df,
        base_features=soh_base_features,
        interval_seconds=interval_seconds,
        feature_aggs=feature_aggs,
        scaler=soh_scaler,
        model=soh_model,
        device=device,
        soh_init=args.soh_init,
    )

    soh_per_row = expand_soh_to_rows(df, bins, soh_hourly, interval_seconds, soh_init=args.soh_init)

    # Build SOC input features with simulated SOH
    df_soc = df.copy()
    df_soc['SOH'] = soh_per_row
    X_soc = df_soc[soc_features].to_numpy(dtype=np.float32)
    X_soc_scaled = soc_scaler.transform(X_soc).astype(np.float32)
    y_soc_true = df_soc['SOC'].to_numpy(dtype=np.float32)

    # SOC model
    soc_model = LSTMMLP(
        in_features=len(soc_features),
        hidden_size=soc_hidden,
        mlp_hidden=soc_mlp_hidden,
        num_layers=soc_num_layers,
        dropout=soc_dropout,
    ).to(device)
    soc_state = torch.load(args.soc_ckpt, map_location=device)
    soc_model.load_state_dict(soc_state['model_state_dict'])

    # Rolling SOC prediction
    soc_pred, soc_rmse, soc_mae = rolling_predict_soc(
        Xs=X_soc_scaled,
        y_true=y_soc_true,
        chunk=soc_chunk,
        model=soc_model,
        device=device,
        batch_size=int(args.soc_batch),
    )

    # Outputs
    os.makedirs(args.out_dir, exist_ok=True)

    # Save hourly SOH
    soh_df = pd.DataFrame({
        'bin': bins,
        'time_s': bins * interval_seconds,
        'soh_pred': soh_hourly,
    })
    # optional true SOH hourly (last)
    if 'SOH' in df.columns:
        df_tmp = df[['SOH', 'Testtime[s]']].dropna().copy()
        df_tmp['_bin'] = (df_tmp['Testtime[s]'] // interval_seconds).astype(np.int64)
        soh_true_hourly = df_tmp.groupby('_bin', sort=True)['SOH'].last().reset_index()
        soh_df = soh_df.merge(soh_true_hourly, left_on='bin', right_on='_bin', how='left')
        soh_df = soh_df.drop(columns=['_bin'])
        soh_df = soh_df.rename(columns={'SOH': 'soh_true_last'})

    soh_csv = os.path.join(args.out_dir, f"soh_hourly_{args.cell}.csv")
    soh_df.to_csv(soh_csv, index=False)

    # Save SOC predictions
    soc_idx = np.arange(soc_chunk - 1, len(df_soc))
    soc_df = pd.DataFrame({
        'index': soc_idx,
        'soc_true': y_soc_true[soc_chunk - 1:],
        'soc_pred': soc_pred,
    })
    soc_csv = os.path.join(args.out_dir, f"soc_pred_fullcell_{args.cell}.csv")
    soc_df.to_csv(soc_csv, index=False)

    # Summary
    summary = {
        'cell': args.cell,
        'device': str(device),
        'soc_rmse': float(soc_rmse),
        'soc_mae': float(soc_mae),
        'soc_chunk': int(soc_chunk),
        'soc_features': soc_features,
        'soh_interval_seconds': int(interval_seconds),
        'soh_feature_aggs': feature_aggs,
        'soh_init': float(args.soh_init),
        'soh_hours': int(len(soh_hourly)),
        'paths': {
            'soc_config': args.soc_config,
            'soc_ckpt': args.soc_ckpt,
            'soc_scaler': args.soc_scaler,
            'soh_config': args.soh_config,
            'soh_ckpt': args.soh_ckpt,
            'soh_scaler': args.soh_scaler,
        }
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plots
    try:
        # SOC plot
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(soc_df['index'], soc_df['soc_true'], label='SOC true', linewidth=1.0)
        ax.plot(soc_df['index'], soc_df['soc_pred'], label='SOC pred', linewidth=1.0, alpha=0.8)
        ax.set_title(f"SOC Prediction – Full Cell ({args.cell})")
        ax.set_xlabel('Index')
        ax.set_ylabel('SOC')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"soc_pred_fullcell_{args.cell}.png"), dpi=150)
        plt.close(fig)

        # SOH plot
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.plot(soh_df['time_s'] / 3600.0, soh_df['soh_pred'], label='SOH pred (hourly)', linewidth=1.0)
        if 'soh_true_last' in soh_df.columns:
            ax.plot(soh_df['time_s'] / 3600.0, soh_df['soh_true_last'], label='SOH true (hourly last)', linewidth=1.0, alpha=0.8)
        ax.set_title(f"SOH Prediction – Hourly ({args.cell})")
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('SOH')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"soh_hourly_{args.cell}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
