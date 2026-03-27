#!/usr/bin/env python3
"""
0.4.1.1 SOH Training – Hourly Aggregation + Stats (Seq2Seq, Causal CNN)
Key updates:
- Predicts SOH at every timestep (seq2seq) for smooth stateful inference
- Hourly aggregation with mean/std/min/max per feature
- Dilated causal CNN blocks for larger receptive field
- Optional warmup_steps to ignore early unstable states
"""
import os
import json
import math
import time
import yaml
import random
import typing as T
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================
# Reproducibility
# ========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================
# Model
# ========================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            return out[:, :, :-self.pad]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class SOH_CNN_Seq2Seq(nn.Module):
    """Causal CNN that outputs SOH at every timestep."""
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mlp_hidden: int,
        kernel_size: int = 5,
        dilations: T.Optional[T.List[int]] = None,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_size, kernel_size=1)
        if dilations is None:
            dilations = [1] * max(1, int(num_blocks))
        self.dilations = [int(d) for d in dilations]
        self.blocks = nn.Sequential(
            *[ConvBlock(hidden_size, hidden_size, kernel_size, dropout, dilation=d) for d in self.dilations]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mlp_hidden, 1, kernel_size=1),
        )
        self.kernel_size = kernel_size
        self.num_blocks = len(self.dilations)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.input_proj(x)
        out = self.blocks(out)
        y = self.head(out).squeeze(1)
        return y

# ========================
# Dataset
# ========================

@dataclass
class SeqConfig:
    features: T.List[str]
    target: str = 'SOH'
    chunk: int = 512
    stride: int = 1

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_cfg: SeqConfig, scaler: RobustScaler):
        self.df = df.reset_index(drop=True)
        self.seq_cfg = seq_cfg
        self.scaler = scaler
        self.features = seq_cfg.features
        self.target = seq_cfg.target
        self.chunk = seq_cfg.chunk
        self.stride = max(1, int(seq_cfg.stride))

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
        y_seq = self.y[start:end]
        return x_seq, y_seq

# ========================
# Helpers
# ========================

def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, cfg):
    sched_cfg = cfg.get('scheduler', {})
    stype = sched_cfg.get('type', 'cosine_warm_restarts')
    if stype == 'cosine_warm_restarts':
        T_0 = int(sched_cfg.get('T_0', 100))
        T_mult = int(sched_cfg.get('T_mult', 1))
        eta_min_factor = float(sched_cfg.get('eta_min_factor', 0.01))
        base_lr = optimizer.param_groups[0]['lr']
        eta_min = base_lr * eta_min_factor
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    return None


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return {"rmse": None, "mae": None, "r2": None, "max_error": None}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')
    max_error = np.max(np.abs(y_true - y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "max_error": max_error}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_csv_row(path: str, row: dict, header: T.Optional[T.List[str]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    df = pd.DataFrame([row])
    if header and not file_exists:
        df = df[header]
    mode = 'a' if file_exists else 'w'
    df.to_csv(path, mode=mode, index=False, header=not file_exists)

# -------------------------
# Env expansion
# -------------------------

_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

def expand_env_with_defaults(val: str) -> str:
    if not isinstance(val, str):
        return val
    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, '') else default
    val2 = _ENV_DEFAULT_PATTERN.sub(repl, val)
    return os.path.expandvars(val2)

# -------------------------
# Sampling helpers
# -------------------------

def expand_features_for_sampling(base_features: T.List[str], sampling_cfg: dict) -> T.List[str]:
    if not sampling_cfg or not sampling_cfg.get('enabled', False):
        return list(base_features)
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean'])
    if not feature_aggs:
        feature_aggs = ['mean']
    return [f'{feat}_{agg}' for feat in base_features for agg in feature_aggs]

# ========================
# Data loading
# ========================

def load_cell_parquet(data_root: str, cell: str) -> pd.DataFrame:
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


def maybe_aggregate_hourly(df: pd.DataFrame, base_features: T.List[str], target: str, sampling_cfg: dict) -> pd.DataFrame:
    if not sampling_cfg or not sampling_cfg.get('enabled', False):
        return df
    if 'Testtime[s]' not in df.columns:
        return df
    interval = int(sampling_cfg.get('interval_seconds', 3600))
    if interval <= 0:
        return df

    feature_aggs = sampling_cfg.get('feature_aggs', ['mean'])
    if not feature_aggs:
        feature_aggs = ['mean']
    target_agg = sampling_cfg.get('target_agg', 'mean')

    cols = list(dict.fromkeys(base_features + [target, 'Testtime[s]']))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=base_features + [target, 'Testtime[s]']).copy()
    work = work.sort_values('Testtime[s]')
    if work.empty:
        return work
    bins = (work['Testtime[s]'] // interval).astype(np.int64)
    work['_bin'] = bins
    agg_spec = {feat: feature_aggs for feat in base_features}
    agg_spec[target] = [target_agg]
    out = work.groupby('_bin', sort=True).agg(agg_spec)
    out.columns = [
        target if col[0] == target else f'{col[0]}_{col[1]}'
        for col in out.columns
    ]
    return out.reset_index(drop=True)


def create_dataloaders(
    cfg: dict,
    base_features: T.List[str],
    features: T.List[str],
    target: str,
    chunk: int,
    scaler: RobustScaler,
    batch_size: int = 64,
):
    data_root = cfg['paths']['data_root']
    train_cells = cfg['cells']['train']
    val_cells = cfg['cells']['val']
    sampling_cfg = cfg.get('sampling', {})

    print('Loading training cells...')
    train_dfs = [load_cell_parquet(data_root, c) for c in train_cells]
    train_dfs = [maybe_aggregate_hourly(d, base_features, target, sampling_cfg) for d in train_dfs]

    print('Fitting scaler...')
    train_all = pd.concat(train_dfs, axis=0)
    X_train_all = train_all[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler.fit(X_train_all.to_numpy(dtype=np.float32))
    print(f'  Scaler fit on {len(X_train_all):,} samples')

    seq_cfg = SeqConfig(features=features, target=target, chunk=chunk, stride=int(cfg['training'].get('window_stride', 1)))

    train_ds = [SeqDataset(df, seq_cfg, scaler) for df in train_dfs]

    print('Loading validation cells...')
    val_dfs = [load_cell_parquet(data_root, c) for c in val_cells]
    val_dfs = [maybe_aggregate_hourly(d, base_features, target, sampling_cfg) for d in val_dfs]
    val_ds = [SeqDataset(df, seq_cfg, scaler) for df in val_dfs]

    dl_cfg = cfg.get('dataloader', {})
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=int(dl_cfg.get('num_workers', 8)),
        prefetch_factor=int(dl_cfg.get('prefetch_factor', 4)),
        persistent_workers=bool(dl_cfg.get('persistent_workers', True)),
        pin_memory=bool(dl_cfg.get('pin_memory', True)),
    )

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_ds),
        shuffle=True,
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

# ========================
# Training / Eval
# ========================

def _select_loss_targets(pred_seq, target_seq, loss_mode: str, warmup_steps: int):
    if loss_mode == 'last':
        return pred_seq[:, -1], target_seq[:, -1]
    if warmup_steps > 0:
        return pred_seq[:, warmup_steps:], target_seq[:, warmup_steps:]
    return pred_seq, target_seq


def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler_amp,
    max_grad_norm,
    epoch_idx: int,
    total_epochs: int,
    accum_steps: int = 1,
    loss_mode: str = 'seq2seq',
    warmup_steps: int = 0,
    smooth_loss_weight: float = 0.0,
    smooth_loss_type: str = 'l1',
    max_batches: T.Optional[int] = None,
):
    model.train()
    loss_fn = nn.MSELoss()
    pbar = tqdm(loader, desc=f'Epoch {epoch_idx:03d}/{total_epochs} [TRAIN]', leave=True, ncols=120, mininterval=5.0)

    total_loss = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (xb, yb) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=scaler_amp is not None):
            pred_seq = model(xb)
            pred_sel, target_sel = _select_loss_targets(pred_seq, yb, loss_mode, warmup_steps)
            loss = loss_fn(pred_sel, target_sel)
            if smooth_loss_weight > 0.0 and pred_sel.ndim == 2 and pred_sel.size(1) > 1:
                diffs = pred_sel[:, 1:] - pred_sel[:, :-1]
                if smooth_loss_type.lower() == 'l2':
                    smooth = (diffs ** 2).mean()
                else:
                    smooth = diffs.abs().mean()
                loss = loss + smooth_loss_weight * smooth

        if scaler_amp is not None:
            scaler_amp.scale(loss / max(1, accum_steps)).backward()
            do_step = ((batch_idx + 1) % max(1, accum_steps) == 0) or (batch_idx + 1 == len(loader))
            if do_step:
                if max_grad_norm:
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            (loss / max(1, accum_steps)).backward()
            do_step = ((batch_idx + 1) % max(1, accum_steps) == 0) or (batch_idx + 1 == len(loader))
            if do_step:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        count += 1
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'avg': f'{total_loss/max(1,count):.6f}'})

    return total_loss / max(1, count)


def eval_model(model, loader, device, loss_mode: str, warmup_steps: int, max_batches: T.Optional[int] = None):
    model.eval()
    pbar = tqdm(loader, desc='[VALIDATION]', leave=True, ncols=120, mininterval=5.0)

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred_seq = model(xb)
            pred_sel, target_sel = _select_loss_targets(pred_seq, yb, loss_mode, warmup_steps)

            preds_all.append(pred_sel.detach().cpu().reshape(-1))
            targets_all.append(target_sel.detach().cpu().reshape(-1))

    if not preds_all:
        return {"rmse": None, "mae": None, "r2": None, "max_error": None}, None, None

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(targets_all).numpy()

    metrics = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred

# ========================
# Plotting
# ========================

def plot_predictions(y_true, y_pred, out_path: str, epoch: int, metrics: dict):
    if y_true is None or y_pred is None:
        return
    max_points = 20000
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_true, y_pred, alpha=0.3, s=5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    ax1.set_xlabel('True SOH')
    ax1.set_ylabel('Predicted SOH')
    ax1.set_title(f'Epoch {epoch} | RMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}')
    ax1.grid(True, alpha=0.3)

    ax2.hist(y_true - y_pred, bins=50, alpha=0.7)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, out_path: str):
    if not history['epoch']:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = history['epoch']
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_rmse'], 'r-', label='Val RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_mae'], 'g-', label='Val MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Validation MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['val_r2'], 'm-', label='Val R2')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].set_title('Validation R2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ========================
# Main
# ========================

def main():
    import argparse
    ap = argparse.ArgumentParser(description='0.4.1.1 SOH Training – Hourly Aggregation + Stats (Seq2Seq, Causal CNN)')
    ap.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'train_soh.yaml'))
    ap.add_argument('--data-root', type=str, default=None)
    ap.add_argument('--out-root', type=str, default=None)
    ap.add_argument('--run-id', type=str, default=None)
    ap.add_argument('--smoke-test', action='store_true', help='Quick test run (1 epoch, 2 batches)')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['paths']['data_root'] = expand_env_with_defaults(cfg['paths']['data_root'])
    cfg['paths']['out_root'] = expand_env_with_defaults(cfg['paths']['out_root'])
    if args.data_root:
        cfg['paths']['data_root'] = args.data_root
    if args.out_root:
        cfg['paths']['out_root'] = args.out_root

    run_id = args.run_id or time.strftime('run_%Y%m%d_%H%M%S')
    out_root = os.path.join(cfg['paths']['out_root'], run_id)
    cfg['paths']['out_root'] = out_root
    os.makedirs(out_root, exist_ok=True)

    if cfg.get('tracking', {}).get('csv_file'):
        cfg['tracking']['csv_file'] = os.path.join(out_root, 'training_log.csv')

    set_seed(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_features = cfg['model']['features']
    sampling_cfg = cfg.get('sampling', {})
    features = expand_features_for_sampling(base_features, sampling_cfg)
    target = cfg.get('training', {}).get('target', 'SOH')
    hidden_size = int(cfg['model'].get('hidden_size', 128))
    mlp_hidden = int(cfg['model'].get('mlp_hidden', 96))
    kernel_size = int(cfg['model'].get('kernel_size', 5))
    num_blocks = int(cfg['model'].get('num_blocks', 4))
    dilations = cfg['model'].get('dilations')
    if dilations is not None:
        dilations = [int(d) for d in dilations]
    dropout = float(cfg['model'].get('dropout', 0.15))

    chunk = int(cfg['training']['seq_chunk_size'])
    batch_size = int(cfg['training'].get('batch_size', 96))
    accum_steps = int(cfg['training'].get('accum_steps', 1))
    lr = float(cfg['training']['lr'])
    weight_decay = float(cfg['training'].get('weight_decay', 0.0))
    max_grad_norm = float(cfg['training'].get('max_grad_norm', 1.0))
    epochs = int(cfg['training'].get('epochs', 200))
    val_interval = int(cfg['training'].get('val_interval', 5))
    early_stopping = int(cfg['training'].get('early_stopping', 30))
    warmup_epochs = int(cfg['training'].get('warmup_epochs', 5))
    loss_mode = str(cfg['training'].get('loss_mode', 'seq2seq')).lower()
    warmup_steps = int(cfg['training'].get('warmup_steps', 0))
    smooth_loss_weight = float(cfg['training'].get('smooth_loss_weight', 0.0))
    smooth_loss_type = str(cfg['training'].get('smooth_loss_type', 'l1')).lower()

    if args.smoke_test:
        epochs = 1
        val_interval = 1

    print('\n' + '=' * 80)
    print('0.4.1.1 SOH Training – Hourly Aggregation + Stats (Seq2Seq, Causal CNN)')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'Output: {out_root}')
    print(f'Device: {device}')
    if device.type == 'cuda':
        try:
            print(f'GPU: {torch.cuda.get_device_name(0)}')
        except Exception:
            pass
    print(f'Features ({len(features)}): {features}')
    if dilations is not None:
        print(f'Model: hidden={hidden_size}, dilations={dilations}, kernel={kernel_size}, mlp={mlp_hidden}')
    else:
        print(f'Model: hidden={hidden_size}, blocks={num_blocks}, kernel={kernel_size}, mlp={mlp_hidden}')
    print(f'Training: epochs={epochs}, batch={batch_size}, accum={accum_steps} (eff_batch={batch_size*accum_steps})')
    if sampling_cfg.get('enabled', False):
        feature_aggs = sampling_cfg.get('feature_aggs', ['mean'])
        target_agg = sampling_cfg.get('target_agg', 'mean')
        print(f"Sampling: interval={sampling_cfg.get('interval_seconds', 3600)}s, feature_aggs={feature_aggs}, target_agg={target_agg}")
    print(f'Loss: mode={loss_mode}, warmup_steps={warmup_steps}, smooth_w={smooth_loss_weight} ({smooth_loss_type})')
    print(f'Learning rate: {lr}, weight_decay: {weight_decay}, warmup_epochs: {warmup_epochs}')
    print('=' * 80 + '\n')

    scaler = RobustScaler()
    train_loader, val_loader = create_dataloaders(cfg, base_features, features, target, chunk, scaler, batch_size=batch_size)

    from joblib import dump
    scaler_path = os.path.join(out_root, 'scaler_robust.joblib')
    dump(scaler, scaler_path)
    print(f'\nScaler saved: {scaler_path}')

    model = SOH_CNN_Seq2Seq(
        in_features=len(features),
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        kernel_size=kernel_size,
        dilations=dilations,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,} (trainable: {trainable_params:,})\n')

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, cfg['training'])
    amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    csv_path = cfg.get('tracking', {}).get('csv_file') if cfg.get('tracking', {}).get('csv_log', True) else None
    csv_header = ['epoch', 'train_loss', 'val_rmse', 'val_mae', 'val_r2', 'val_max_error', 'lr', 'best_val_rmse']

    best = {"val_rmse": float('inf'), "epoch": -1, "path": None}
    patience = 0
    save_every_n = int(cfg.get('export', {}).get('save_every_n', 20))
    plot_interval = int(cfg.get('tracking', {}).get('plot_interval', 10))
    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []}

    max_batches = 2 if args.smoke_test else None

    print('Starting training...\n')

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        train_loss = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            amp_scaler,
            max_grad_norm,
            epoch,
            epochs,
            accum_steps,
            loss_mode=loss_mode,
            warmup_steps=warmup_steps,
            smooth_loss_weight=smooth_loss_weight,
            smooth_loss_type=smooth_loss_type,
            max_batches=max_batches,
        )

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step(epoch)

        current_lr = get_lr(optimizer)

        metrics = {"rmse": None, "mae": None, "r2": None, "max_error": None}
        y_true, y_pred = None, None

        if (epoch % val_interval == 0) or (epoch == 1) or (epoch == epochs):
            metrics, y_true, y_pred = eval_model(
                model,
                val_loader,
                device,
                loss_mode=loss_mode,
                warmup_steps=warmup_steps,
                max_batches=max_batches,
            )

            print(
                f'\n[Epoch {epoch:03d}] Train Loss: {train_loss:.6f} | '
                f'Val RMSE: {metrics["rmse"]:.6f} | MAE: {metrics["mae"]:.6f} | '
                f'R2: {metrics["r2"]:.4f} | LR: {current_lr:.2e}\n'
            )

            if metrics['rmse'] is not None and metrics['rmse'] < best['val_rmse']:
                best.update({"val_rmse": metrics['rmse'], "epoch": epoch})
                patience = 0

                ckpt_dir = os.path.join(out_root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch:04d}_rmse{metrics['rmse']:.5f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': scaler_path,
                    'config': cfg,
                    'metrics': metrics,
                }, ckpt_path)
                best['path'] = ckpt_path
                print(f'✓ New best model saved: {ckpt_path}\n')
            else:
                patience += val_interval

            if y_true is not None and (epoch % plot_interval == 0 or epoch == 1):
                plot_dir = os.path.join(out_root, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f'predictions_epoch{epoch:04d}.png')
                plot_predictions(y_true, y_pred, plot_path, epoch, metrics)

            if save_every_n > 0 and epoch % save_every_n == 0:
                ckpt_dir = os.path.join(out_root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'metrics': metrics,
                }, ckpt_path)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(metrics['rmse'] if metrics['rmse'] is not None else np.nan)
        history['val_mae'].append(metrics['mae'] if metrics['mae'] is not None else np.nan)
        history['val_r2'].append(metrics['r2'] if metrics['r2'] is not None else np.nan)

        if csv_path:
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_rmse': metrics['rmse'],
                'val_mae': metrics['mae'],
                'val_r2': metrics['r2'],
                'val_max_error': metrics.get('max_error'),
                'lr': current_lr,
                'best_val_rmse': best['val_rmse'],
            }
            save_csv_row(csv_path, row, header=csv_header)

        if patience >= early_stopping:
            print(f'\n[Early Stopping] No improvement for {early_stopping} epochs. Stopping.\n')
            break

    print('\nGenerating final plots...')
    plot_dir = os.path.join(out_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_curves(history, os.path.join(plot_dir, 'training_curves.png'))

    final_path = os.path.join(out_root, 'checkpoints', 'final_model.pt')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler_path,
        'config': cfg,
        'history': history,
    }, final_path)

    print('\n' + '=' * 80)
    print('Training Complete!')
    print('=' * 80)
    print(f'Best epoch: {best["epoch"]} | Best RMSE: {best["val_rmse"]:.6f}')
    print(f'Best model: {best["path"]}')
    print(f'Output directory: {out_root}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
