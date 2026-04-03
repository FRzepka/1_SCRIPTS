import os
import json
import math
import time
import yaml
import random
import shutil
import sys
import typing as T
from dataclasses import dataclass
from pathlib import Path

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GRUMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, state: T.Optional[torch.Tensor] = None, return_state: bool = False):
        out, new_state = self.gru(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


@dataclass
class SeqConfig:
    features: T.List[str]
    target: str = 'SOC'
    chunk: int = 4096
    stride: int = 1
    training: bool = False
    seed: int = 42
    augmentation: T.Optional[dict] = None


class SeqDataset(Dataset):
    RAW_REQUIRED = [
        'Testtime[s]',
        'Voltage[V]',
        'Current[A]',
        'Temperature[°C]',
        'SOH',
        'SOC',
        'Q_c',
    ]

    def __init__(self, df: pd.DataFrame, seq_cfg: SeqConfig, scaler: RobustScaler):
        self.seq_cfg = seq_cfg
        self.scaler = scaler
        self.features = seq_cfg.features
        self.target = seq_cfg.target
        self.chunk = seq_cfg.chunk
        self.stride = max(1, int(seq_cfg.stride))
        self.training = bool(seq_cfg.training)
        self.seed = int(seq_cfg.seed)
        self.aug = dict(seq_cfg.augmentation or {})

        work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.RAW_REQUIRED).reset_index(drop=True)
        self.time_s = work['Testtime[s]'].to_numpy(dtype=np.float32)
        self.voltage = work['Voltage[V]'].to_numpy(dtype=np.float32)
        self.current = work['Current[A]'].to_numpy(dtype=np.float32)
        self.temperature = work['Temperature[°C]'].to_numpy(dtype=np.float32)
        self.soh = work['SOH'].to_numpy(dtype=np.float32)
        self.soc = work['SOC'].to_numpy(dtype=np.float32)
        self.q_c = work['Q_c'].to_numpy(dtype=np.float32)

        total = len(work)
        self.nseq = 0 if total < self.chunk else 1 + (total - self.chunk) // self.stride

    def __len__(self):
        return self.nseq

    def _rng_for_idx(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx)

    def _warp_relative_time(self, rel_t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if len(rel_t) <= 1:
            return rel_t.copy()
        prob = float(self.aug.get('time_warp_prob', 0.0))
        if (not self.training) or prob <= 0.0 or rng.random() > prob:
            return rel_t.copy()

        intervals = np.diff(rel_t)
        if len(intervals) == 0:
            return rel_t.copy()

        std_s = float(self.aug.get('dt_jitter_std_s', 0.0))
        clip_s = float(self.aug.get('dt_jitter_clip_s', std_s * 3.0 if std_s > 0 else 0.0))
        min_dt = float(self.aug.get('min_dt_s', 0.85))
        max_dt = float(self.aug.get('max_dt_s', 1.15))
        if std_s <= 0.0:
            return rel_t.copy()

        noise = rng.normal(0.0, std_s, size=len(intervals)).astype(np.float32)
        if clip_s > 0.0:
            noise = np.clip(noise, -clip_s, clip_s)
        dt_new = np.clip(intervals + noise, min_dt, max_dt)
        total_orig = float(rel_t[-1] - rel_t[0])
        total_new = float(dt_new.sum())
        if total_new <= 1e-8:
            return rel_t.copy()
        dt_new *= total_orig / total_new
        warped = np.concatenate(([0.0], np.cumsum(dt_new, dtype=np.float64))).astype(np.float32)
        warped[-1] = rel_t[-1]
        return warped

    def _engineer_window(self, start: int, end: int, idx: int) -> T.Tuple[np.ndarray, np.float32]:
        rel_t = self.time_s[start:end] - self.time_s[start]
        rng = self._rng_for_idx(idx)
        warped_t = self._warp_relative_time(rel_t, rng)

        voltage = np.interp(warped_t, rel_t, self.voltage[start:end]).astype(np.float32)
        current = np.interp(warped_t, rel_t, self.current[start:end]).astype(np.float32)
        temperature = np.interp(warped_t, rel_t, self.temperature[start:end]).astype(np.float32)
        soh = np.interp(warped_t, rel_t, self.soh[start:end]).astype(np.float32)
        soc = np.interp(warped_t, rel_t, self.soc[start:end]).astype(np.float32)

        dt_s = np.empty_like(warped_t, dtype=np.float32)
        if len(warped_t) > 1:
            dt_s[0] = float(max(warped_t[1] - warped_t[0], 1e-6))
            dt_s[1:] = np.diff(warped_t).astype(np.float32)
        else:
            dt_s[0] = 1.0
        dt_safe = np.maximum(dt_s, 1e-6)

        q_c = np.empty_like(current, dtype=np.float32)
        q_c[0] = float(self.q_c[start])
        if len(q_c) > 1:
            q_c[1:] = q_c[0] + np.cumsum(current[1:] * dt_safe[1:] / 3600.0, dtype=np.float64).astype(np.float32)

        d_u_dt = np.zeros_like(voltage, dtype=np.float32)
        d_i_dt = np.zeros_like(current, dtype=np.float32)
        if len(voltage) > 1:
            d_u_dt[1:] = np.diff(voltage) / dt_safe[1:]
            d_i_dt[1:] = np.diff(current) / dt_safe[1:]

        channel_drop_prob = float(self.aug.get('current_channel_dropout_prob', 0.0))
        if self.training and channel_drop_prob > 0.0 and rng.random() < channel_drop_prob:
            current = current * 0.0
            q_c = np.full_like(q_c, q_c[0])
            d_i_dt = d_i_dt * 0.0

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
        X = np.column_stack([feat_map[name] for name in self.features]).astype(np.float32)
        return X, np.float32(soc[-1])

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.chunk
        X, y_last = self._engineer_window(start, end, idx)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        return torch.from_numpy(X_scaled), torch.tensor(y_last, dtype=torch.float32)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, cfg):
    sch = cfg.get('scheduler', {'type': None})
    if sch.get('type') == 'cosine_warm_restarts':
        t_0 = int(sch.get('T_0', 150))
        eta_min_factor = float(sch.get('eta_min_factor', 0.05))
        base_lrs = [group['lr'] for group in optimizer.param_groups]
        eta_min = min(base_lrs) * eta_min_factor
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=eta_min)
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=2)


def save_csv_row(path: str, row: dict, header: T.Optional[T.List[str]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    frame = pd.DataFrame([row])
    if header and not file_exists:
        frame = frame[header]
    frame.to_csv(path, mode='a' if file_exists else 'w', index=False, header=not file_exists)


def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
    path = os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet")
    if not os.path.exists(path):
        path = os.path.join(data_root, f'df_FE_{cell}.parquet')
    if not os.path.exists(path):
        cid = cell[-3:]
        alt = os.path.join(data_root, f'df_FE_C{cid}.parquet')
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f'Could not locate parquet for cell {cell} in {data_root}')
    return pd.read_parquet(path)


def engineer_frame_for_scaler(df: pd.DataFrame, features: T.List[str]) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH', 'SOC', 'Q_c']).copy()
    time_s = work['Testtime[s]'].to_numpy(dtype=np.float32)
    voltage = work['Voltage[V]'].to_numpy(dtype=np.float32)
    current = work['Current[A]'].to_numpy(dtype=np.float32)
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
        'Temperature[°C]': work['Temperature[°C]'].to_numpy(dtype=np.float32),
        'SOH': work['SOH'].to_numpy(dtype=np.float32),
        'Q_c': work['Q_c'].to_numpy(dtype=np.float32),
        'dU_dt[V/s]': d_u_dt,
        'dI_dt[A/s]': d_i_dt,
        'dt_s': dt_s,
    }
    return pd.DataFrame({name: feat_map[name] for name in features})


def train_one_epoch(model, loader, device, optimizer, scaler_amp, max_grad_norm, epoch_idx: int, total_epochs: int, accum_steps: int = 1, max_batches: T.Optional[int] = None):
    model.train()
    loss_fn = nn.MSELoss()
    show_progress = sys.stdout.isatty()
    pbar = tqdm(loader, desc=f'Epoch {epoch_idx}/{total_epochs} • train', leave=False, disable=not show_progress)
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (xb, yb) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=scaler_amp is not None):
            preds = model(xb)
            loss = loss_fn(preds, yb)
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
        total += loss.item() * xb.size(0)
        count += xb.size(0)
        if show_progress:
            pbar.set_postfix({'loss': total / max(count, 1)})
    return total / max(count, 1)


def eval_model(model, loader, device, max_batches: T.Optional[int] = None):
    model.eval()
    preds_all = []
    ys_all = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds_all.append(pred.detach().cpu().numpy())
            ys_all.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    y_true = np.concatenate(ys_all) if ys_all else np.array([])
    return compute_metrics(y_true, y_pred), y_true, y_pred


def create_dataloaders(cfg: dict, features: T.List[str], chunk: int, scaler: RobustScaler, batch_size: int = 64):
    train_dfs = [load_cell_dataframe(cfg['paths']['data_root'], cell) for cell in cfg['cells']['train']]
    val_dfs = [load_cell_dataframe(cfg['paths']['data_root'], cell) for cell in cfg['cells']['val']]

    dl_cfg = cfg.get('dataloader', {})
    scaler_fit_stride = max(1, int(dl_cfg.get('scaler_fit_stride', 1)))
    scaler_fit_max_rows_per_cell = max(0, int(dl_cfg.get('scaler_fit_max_rows_per_cell', 0)))

    scaler_frames = []
    for df in train_dfs:
        frame = engineer_frame_for_scaler(df, features)
        if scaler_fit_stride > 1:
            frame = frame.iloc[::scaler_fit_stride].reset_index(drop=True)
        if scaler_fit_max_rows_per_cell and len(frame) > scaler_fit_max_rows_per_cell:
            idx = np.linspace(0, len(frame) - 1, scaler_fit_max_rows_per_cell, dtype=np.int64)
            frame = frame.iloc[idx].reset_index(drop=True)
        scaler_frames.append(frame)
    x_train = pd.concat(scaler_frames, axis=0)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))

    train_cfg = cfg.get('training', {})
    train_stride = int(train_cfg.get('train_window_stride', train_cfg.get('window_stride', 1)))
    val_stride = int(train_cfg.get('val_window_stride', train_cfg.get('window_stride', 1)))
    aug_cfg = cfg.get('augmentation', {})
    train_ds = [
        SeqDataset(df, SeqConfig(features=features, target='SOC', chunk=chunk, stride=train_stride, training=True, seed=int(cfg.get('seed', 42)) + i * 100000, augmentation=aug_cfg), scaler)
        for i, df in enumerate(train_dfs)
    ]
    val_ds = [
        SeqDataset(df, SeqConfig(features=features, target='SOC', chunk=chunk, stride=val_stride, training=False, seed=int(cfg.get('seed', 42)) + 900000 + i * 100000, augmentation=aug_cfg), scaler)
        for i, df in enumerate(val_dfs)
    ]

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
        collate_fn=lambda batch: (torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])),
        **common_kwargs,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_ds),
        shuffle=False,
        collate_fn=lambda batch: (torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])),
        **common_kwargs,
    )
    return train_loader, val_loader


def _save_progress_plot(path: str, history: dict):
    try:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        if any(v is not None for v in history['val_rmse']):
            ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_rmse']], label='Val RMSE', color='tab:red')
        if any(v is not None for v in history['val_mae']):
            ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_mae']], label='Val MAE', color='tab:orange')
        ax2.set_ylabel('Validation Metrics', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.title('Training Progress')
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
    except Exception as exc:
        print(f'Progress-plot warn: {exc}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'train_soc.yaml'))
    args = ap.parse_args()

    with open(args.config, 'r') as handle:
        cfg = yaml.safe_load(handle)

    set_seed(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = cfg['paths']['out_root']
    os.makedirs(out_root, exist_ok=True)

    features = cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    dropout = float(cfg['model'].get('dropout', 0.05))

    chunk = int(cfg['training']['seq_chunk_size'])
    batch_size = int(cfg['training'].get('batch_size', 64))
    accum_steps = int(cfg['training'].get('accum_steps', 1))
    lr = float(cfg['training']['lr'])
    weight_decay = float(cfg['training'].get('weight_decay', 0.0))
    max_grad_norm = float(cfg['training'].get('max_grad_norm', 1.0))
    epochs = int(cfg['training'].get('epochs', 100))
    val_interval = int(cfg['training'].get('val_interval', 5))
    early_stopping = int(cfg['training'].get('early_stopping', 20))

    scaler = RobustScaler()
    train_loader, val_loader = create_dataloaders(cfg, features, chunk, scaler, batch_size=batch_size)

    from joblib import dump
    scaler_path = os.path.join(out_root, 'scaler_robust.joblib')
    dump(scaler, scaler_path)

    model = GRUMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
    if cfg['training'].get('compile', False):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('torch.compile: enabled')
        except Exception as exc:
            print(f'torch.compile failed: {exc}')

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, cfg['training'])
    amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    csv_path = cfg['tracking'].get('csv_file') if cfg.get('tracking', {}).get('csv_log', True) else None
    csv_header = ['epoch', 'train_loss', 'val_rmse', 'val_mae', 'val_r2', 'lr', 'scaler_path', 'checkpoint', 'best_val_rmse']
    best = {'val_rmse': float('inf'), 'epoch': -1, 'path': None}
    patience = 0
    save_every_n = int(cfg.get('export', {}).get('save_every_n', 0) or 0)
    history = {'epoch': [], 'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}
    dummy = torch.zeros((1, chunk, len(features)), dtype=torch.float32).to(device)

    print(f'Using device: {device} | CUDA available: {torch.cuda.is_available()}')
    if device.type == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")
        except Exception:
            pass
    print(f'Data: train steps/epoch={len(train_loader)}, val steps={len(val_loader)}, batch_size={getattr(train_loader, "batch_size", "N/A")}, seq_chunk={chunk}')
    print(f"Features: {features}")
    print(f"Augmentation: {cfg.get('augmentation', {})}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch, epochs, accum_steps=accum_steps)
        if scheduler is not None:
            scheduler.step(epoch + train_loss)

        metrics = {'rmse': None, 'mae': None, 'r2': None}
        if (epoch % val_interval == 0) or (epoch == 1):
            val_metrics, _, _ = eval_model(model, val_loader, device)
            metrics = val_metrics
            if val_metrics['rmse'] < best['val_rmse']:
                best.update({'val_rmse': val_metrics['rmse'], 'epoch': epoch})
                patience = 0
                ckpt_dir = os.path.join(out_root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'soc_epoch{epoch:04d}_rmse{val_metrics["rmse"]:.5f}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'scaler_path': scaler_path,
                    'features': features,
                    'chunk': chunk,
                    'model_type': 'GRU_MLP',
                }, ckpt_path)
                best['path'] = ckpt_path
            else:
                patience += 1

        if save_every_n and (epoch % save_every_n == 0):
            ckpt_dir = os.path.join(out_root, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'soc_epoch{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'scaler_path': scaler_path,
                'features': features,
                'chunk': chunk,
                'model_type': 'GRU_MLP',
            }, ckpt_path)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(metrics['rmse'] if metrics['rmse'] is not None else None)
        history['val_mae'].append(metrics['mae'] if metrics['mae'] is not None else None)
        history['val_r2'].append(metrics['r2'] if metrics['r2'] is not None else None)

        if csv_path:
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_rmse': metrics['rmse'] if metrics['rmse'] is not None else '',
                'val_mae': metrics['mae'] if metrics['mae'] is not None else '',
                'val_r2': metrics['r2'] if metrics['r2'] is not None else '',
                'lr': optimizer.param_groups[0]['lr'],
                'scaler_path': scaler_path,
                'checkpoint': best['path'] or '',
                'best_val_rmse': best['val_rmse'] if best['val_rmse'] != float('inf') else '',
            }
            save_csv_row(csv_path, row, header=csv_header)
            if metrics.get('rmse') is not None:
                _save_progress_plot(os.path.join(os.path.dirname(csv_path) or out_root, 'training_progress.png'), history)

        if metrics['rmse'] is not None:
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_rmse={metrics['rmse']:.6f} | best_rmse={best['val_rmse']:.6f}")
        else:
            print(f'Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}')

        if patience >= early_stopping:
            break

    if cfg.get('export', {}).get('to_onnx', True) and best['path']:
        state = torch.load(best['path'], map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        onnx_path = os.path.join(out_root, f"soc_best_epoch{best['epoch']:04d}.onnx")
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['input_seq'],
            output_names=['soc'],
            dynamic_axes={'input_seq': {0: 'batch'}, 'soc': {0: 'batch'}},
            opset_version=int(cfg['export'].get('onnx_opset', 17)),
        )
        if cfg.get('export', {}).get('stateful_onnx', False):
            feat_dim = len(features)
            hid_dim = hidden_size
            x1 = torch.zeros((1, 1, feat_dim), dtype=torch.float32).to(device)
            h0 = torch.zeros((num_layers, 1, hid_dim), dtype=torch.float32).to(device)

            class StatefulWrapper(torch.nn.Module):
                def __init__(self, core):
                    super().__init__()
                    self.core = core

                def forward(self, x_step, h):
                    pred, h1 = self.core(x_step, state=h, return_state=True)
                    return pred, h1

            wrapper = StatefulWrapper(model).to(device)
            onnx_stateful = os.path.join(out_root, f"soc_best_epoch{best['epoch']:04d}_stateful.onnx")
            torch.onnx.export(
                wrapper,
                (x1, h0),
                onnx_stateful,
                input_names=['x_step', 'h0'],
                output_names=['y_step', 'h1'],
                dynamic_axes={'x_step': {0: 'batch'}, 'h0': {1: 'batch'}, 'y_step': {0: 'batch'}, 'h1': {1: 'batch'}},
                opset_version=int(cfg['export'].get('onnx_opset', 17)),
            )
        save_json({
            'best_epoch': best['epoch'],
            'best_val_rmse': best['val_rmse'],
            'checkpoint': best['path'],
            'onnx': onnx_path,
            'scaler': scaler_path,
            'features': features,
            'chunk': chunk,
            'model_type': 'GRU_MLP',
        }, os.path.join(out_root, 'export_manifest.json'))

    save_json(history, os.path.join(out_root, 'training_progress.json'))
    _save_progress_plot(os.path.join(out_root, 'training_progress.png'), history)

    model_out_root = cfg['paths'].get('model_out_root')
    if model_out_root and best['path']:
        os.makedirs(model_out_root, exist_ok=True)
        shutil.copy2(best['path'], os.path.join(model_out_root, os.path.basename(best['path'])))
        shutil.copy2(scaler_path, os.path.join(model_out_root, 'scaler_robust.joblib'))
        shutil.copy2(args.config, os.path.join(model_out_root, 'train_soc.yaml'))
        manifest = {
            'best_epoch': best['epoch'],
            'best_val_rmse': best['val_rmse'],
            'checkpoint': os.path.join(model_out_root, os.path.basename(best['path'])),
            'scaler': os.path.join(model_out_root, 'scaler_robust.joblib'),
            'config': os.path.join(model_out_root, 'train_soc.yaml'),
            'features': features,
            'chunk': chunk,
            'model_type': 'GRU_MLP',
        }
        save_json(manifest, os.path.join(model_out_root, 'export_manifest.json'))


if __name__ == '__main__':
    main()
