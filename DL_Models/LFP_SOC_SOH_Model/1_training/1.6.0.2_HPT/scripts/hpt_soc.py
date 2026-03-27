import os
import json
import math
import time
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import yaml
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Model
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
# Dataset
# -------------------------

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str], target: str, chunk: int, stride: int, scaler: RobustScaler):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target
        self.chunk = chunk
        self.stride = max(1, int(stride))

        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.features + [self.target])
        X = self.df[self.features].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X).astype(np.float32)
        y = self.df[self.target].to_numpy(dtype=np.float32)

        self.X = torch.from_numpy(X_scaled)
        self.y = torch.from_numpy(y)

        total = len(self.df)
        if total < self.chunk:
            self.nseq = 0
        else:
            self.nseq = 1 + (total - self.chunk) // self.stride

    def __len__(self):
        return self.nseq

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.chunk
        x_seq = self.X[start:end]
        y_last = self.y[end - 1]
        return x_seq, y_last

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def make_dataloaders(train_dfs, val_dfs, features, chunk, stride, scaler, batch_size, dl_cfg):
    train_ds = [SeqDataset(d, features, 'SOC', chunk, stride, scaler) for d in train_dfs]
    val_ds = [SeqDataset(d, features, 'SOC', chunk, stride, scaler) for d in val_dfs]

    cpu_cnt = os.cpu_count() or 8
    num_workers = int(dl_cfg.get('num_workers', max(4, min(16, cpu_cnt - 2))))
    prefetch_factor = int(dl_cfg.get('prefetch_factor', 6))
    pin_memory = bool(dl_cfg.get('pin_memory', True))
    persistent_workers = bool(dl_cfg.get('persistent_workers', True)) if num_workers > 0 else False

    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        common_kwargs.update(dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers))

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_ds),
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b])),
        **common_kwargs,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_ds),
        shuffle=False,
        collate_fn=lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b])),
        **common_kwargs,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, device, optimizer, max_grad_norm, accum_steps, max_batches=None):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        (loss / max(1, accum_steps)).backward()
        do_step = ((batch_idx + 1) % max(1, accum_steps) == 0)
        if do_step:
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(count, 1)


def eval_model(model, loader, device, max_batches=None):
    model.eval()
    preds_all = []
    ys_all = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds_all.append(pred.detach().cpu().numpy())
            ys_all.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    y_true = np.concatenate(ys_all) if ys_all else np.array([])
    rmse, mae = compute_metrics(y_true, y_pred)
    return rmse, mae


def plot_progress(out_path, history):
    try:
        plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(history['epoch'], history['val_rmse'], label='Val RMSE', color='tab:red')
        ax2.plot(history['epoch'], history['val_mae'], label='Val MAE', color='tab:orange')
        ax2.set_ylabel('Validation Metrics', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.title('Training Progress')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Optuna HPT for SOC model')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get('seed', 42)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = cfg['paths']['out_root']
    os.makedirs(out_root, exist_ok=True)

    features = cfg['model']['features']
    chunk = int(cfg['training']['seq_chunk_size'])
    stride = int(cfg['training'].get('window_stride', 1))
    epochs = int(cfg['training'].get('epochs', 30))
    val_interval = int(cfg['training'].get('val_interval', 1))
    early_stopping = int(cfg['training'].get('early_stopping', 5))
    max_grad_norm = float(cfg['training'].get('max_grad_norm', 1.0))

    hpt = cfg.get('hpt', {})
    n_trials = int(hpt.get('n_trials', 20))
    timeout_min = int(hpt.get('timeout_min', 0))
    metric = hpt.get('metric', 'val_rmse')
    max_train_batches = int(hpt.get('max_train_batches', 0)) or None
    max_val_batches = int(hpt.get('max_val_batches', 0)) or None

    ss = cfg.get('search_space', {})

    # Load data once
    train_cells = cfg['cells']['train']
    val_cells = cfg['cells']['val']
    train_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in train_cells]
    val_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in val_cells]

    # Fit scaler on train
    scaler = RobustScaler()
    x_train = pd.concat([d[features] for d in train_dfs], axis=0)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))

    def objective(trial: optuna.Trial):
        hidden_size = trial.suggest_categorical('hidden_size', ss.get('hidden_size', [64]))
        mlp_hidden = trial.suggest_categorical('mlp_hidden', ss.get('mlp_hidden', [64]))
        num_layers = trial.suggest_categorical('num_layers', ss.get('num_layers', [1]))
        dropout = trial.suggest_categorical('dropout', ss.get('dropout', [0.05]))
        lr = trial.suggest_categorical('lr', ss.get('lr', [1e-4]))
        weight_decay = trial.suggest_categorical('weight_decay', ss.get('weight_decay', [0.0]))
        batch_size = trial.suggest_categorical('batch_size', ss.get('batch_size', [512]))
        accum_steps = trial.suggest_categorical('accum_steps', ss.get('accum_steps', [1]))

        dl_cfg = cfg.get('dataloader', {})
        train_loader, val_loader = make_dataloaders(train_dfs, val_dfs, features, chunk, stride, scaler, batch_size, dl_cfg)

        model = LSTMMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        history = {'epoch': [], 'train_loss': [], 'val_rmse': [], 'val_mae': []}
        best = float('inf')
        patience = 0

        trial_dir = os.path.join(out_root, 'trials', f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, device, optimizer, max_grad_norm, accum_steps, max_batches=max_train_batches)
            if epoch % val_interval == 0:
                val_rmse, val_mae = eval_model(model, val_loader, device, max_batches=max_val_batches)
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss)
                history['val_rmse'].append(val_rmse)
                history['val_mae'].append(val_mae)

                if val_rmse < best:
                    best = val_rmse
                    patience = 0
                else:
                    patience += 1

                trial.report(val_rmse if metric == 'val_rmse' else val_mae, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if patience >= early_stopping:
                    break

        # save trial artifacts
        with open(os.path.join(trial_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        plot_progress(os.path.join(trial_dir, 'training_progress.png'), history)

        return best

    study = optuna.create_study(direction=cfg['hpt'].get('direction', 'minimize'))
    study.optimize(objective, n_trials=n_trials, timeout=None if timeout_min <= 0 else timeout_min * 60)

    # Save results
    best = study.best_trial
    result = {
        'best_value': best.value,
        'best_params': best.params,
        'metric': metric,
        'n_trials': len(study.trials),
    }
    with open(os.path.join(out_root, 'hpt_best.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Save all trials summary
    rows = []
    for t in study.trials:
        rows.append({
            'number': t.number,
            'state': str(t.state),
            'value': t.value,
            **t.params,
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_root, 'hpt_trials.csv'), index=False)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
