import os
import json
import math
import time
import yaml
import random
import shutil
import typing as T
from dataclasses import dataclass, asdict

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

# -------------------------
# Reproducibility helpers
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
            nn.Sigmoid()  # SOC in [0,1]
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

    def forward(self, x, state: T.Optional[T.Tuple[torch.Tensor, torch.Tensor]] = None, return_state: bool = False):
        """Forward pass.
        Args:
            x: Tensor [B, T, F]
            state: Optional LSTM (h, c) tuple for stateful inference
            return_state: If True, also return the new (h, c)
        Returns:
            If return_state: (pred [B], (h, c)) else pred [B]
        """
        out, new_state = self.lstm(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


# -------------------------
# Dataset utilities
# -------------------------

@dataclass
class SeqConfig:
    features: T.List[str]
    target: str = 'SOC'
    chunk: int = 4096
    stride: int = 1  # step between starting indices


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_cfg: SeqConfig, scaler: RobustScaler):
        self.df = df.reset_index(drop=True)
        self.seq_cfg = seq_cfg
        self.scaler = scaler
        self.features = seq_cfg.features
        self.target = seq_cfg.target
        self.chunk = seq_cfg.chunk
        self.stride = max(1, int(seq_cfg.stride))

        # guard against NaNs
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.features + [self.target])

        X = self.df[self.features].to_numpy(dtype=np.float32)
        # scale features (ensure float32)
        X_scaled = scaler.transform(X).astype(np.float32)
        y = self.df[self.target].to_numpy(dtype=np.float32)

        self.X = torch.from_numpy(X_scaled)
        self.y = torch.from_numpy(y)

        # number of sequential samples with stride
        total = len(self.df)
        if total < self.chunk:
            self.nseq = 0
        else:
            # start positions are 0, stride, 2*stride, ...
            # last valid start is total - chunk
            self.nseq = 1 + (total - self.chunk) // self.stride

    def __len__(self):
        return self.nseq

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.chunk
        x_seq = self.X[start: end]
        y_last = self.y[end - 1]
        return x_seq, y_last


# -------------------------
# Training/Eval helpers
# -------------------------

def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, cfg):
    sch = cfg.get('scheduler', {"type": None})
    if sch.get('type') == 'cosine_warm_restarts':
        T_0 = int(sch.get('T_0', 150))
        eta_min_factor = float(sch.get('eta_min_factor', 0.05))
        base_lrs = [g['lr'] for g in optimizer.param_groups]
        eta_min = min(base_lrs) * eta_min_factor
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


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
# Data loading
# -------------------------

def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
    # Expect df_FE_*.parquet under data_root
    path = os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet")
    if not os.path.exists(path):
        # fallback: try exact name
        path = os.path.join(data_root, f"df_FE_{cell}.parquet")
    if not os.path.exists(path):
        # try within 0_Data/MGFarm_18650_FE structure if provided differently
        # We'll also try numbered Cxx from file name
        cid = cell[-3:]
        alt = os.path.join(data_root, f"df_FE_C{cid}.parquet")
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate parquet for cell {cell} in {data_root}")
    return pd.read_parquet(path)


# -------------------------
# Main training
# -------------------------

def train_one_epoch(model, loader, device, optimizer, scaler_amp, max_grad_norm, epoch_idx: int, total_epochs: int, accum_steps: int = 1):
    model.train()
    loss_fn = nn.MSELoss()
    pbar = tqdm(loader, desc=f'Epoch {epoch_idx}/{total_epochs} • train', leave=False)
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (xb, yb) in enumerate(pbar):
        # non_blocking speeds up host->device when pin_memory=True on the loader
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        # use new torch.amp API
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
        pbar.set_postfix({"loss": total / max(count, 1)})
    return total / max(count, 1)


def eval_model(model, loader, device):
    model.eval()
    preds_all = []
    ys_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds_all.append(pred.detach().cpu().numpy())
            ys_all.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    y_true = np.concatenate(ys_all) if ys_all else np.array([])
    return compute_metrics(y_true, y_pred), y_true, y_pred


def create_dataloaders(cfg: dict, features: T.List[str], chunk: int, scaler: RobustScaler, batch_size: int = 64):
    train_dfs = []
    for c in cfg['cells']['train']:
        df = load_cell_dataframe(cfg['paths']['data_root'], c)
        train_dfs.append(df)
    val_dfs = []
    for c in cfg['cells']['val']:
        df = load_cell_dataframe(cfg['paths']['data_root'], c)
        val_dfs.append(df)

    # Fit scaler on train only
    x_train = pd.concat([d[features] for d in train_dfs], axis=0)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))

    stride = int(cfg.get('training', {}).get('window_stride', 1))
    train_ds = [SeqDataset(d, SeqConfig(features=features, target='SOC', chunk=chunk, stride=stride), scaler) for d in train_dfs]
    val_ds = [SeqDataset(d, SeqConfig(features=features, target='SOC', chunk=chunk, stride=stride), scaler) for d in val_dfs]

    # Concat via lists inside a single dataset by torch.utils.data.ConcatDataset
    dl_cfg = cfg.get('dataloader', {})
    # Hefty but safe defaults; YAML overrides still apply
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'train_soc.yaml'))
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

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
    # Require at least 100 epochs before early stopping can trigger
    early_stopping = max(int(cfg['training'].get('early_stopping', 20)), 100)

    scaler = RobustScaler()

    # Prepare data
    train_loader, val_loader = create_dataloaders(cfg, features, chunk, scaler, batch_size=batch_size)

    # Save scaler for reproducibility
    from joblib import dump
    scaler_path = os.path.join(out_root, 'scaler_robust.joblib')
    dump(scaler, scaler_path)

    model = LSTMMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
    # Optional compile for potential speedups
    if cfg['training'].get('compile', False):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, cfg['training'])

    # Mixed precision scaler (new API)
    amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # CSV tracking header
    csv_path = cfg['tracking'].get('csv_file') if cfg.get('tracking', {}).get('csv_log', True) else None
    csv_header = [
        'epoch', 'train_loss', 'val_rmse', 'val_mae', 'val_r2', 'lr', 'scaler_path', 'checkpoint', 'best_val_rmse'
    ]

    best = {"val_rmse": float('inf'), "epoch": -1, "path": None}
    patience = 0
    save_every_n = int(cfg.get('export', {}).get('save_every_n', 0) or 0)
    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []}

    # Create data-dependent dummy input for ONNX later
    dummy = torch.zeros((1, chunk, len(features)), dtype=torch.float32).to(device)

    # Info prints
    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    if device.type == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")
        except Exception:
            pass
    print(f"Data: train steps/epoch={len(train_loader)}, val steps={len(val_loader)}, batch_size={getattr(train_loader, 'batch_size', 'N/A')}, seq_chunk={chunk}")
    if cfg.get('training', {}).get('window_stride', 1) != 1:
        print(f"Window stride: {cfg['training']['window_stride']}")
    if accum_steps and accum_steps > 1:
        effective_bs = batch_size * accum_steps
        print(f"Grad accumulation: {accum_steps} (effective batch_size={effective_bs})")
    print(f"Mixed precision: {'ON' if amp_scaler is not None else 'OFF'}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch_idx=epoch, total_epochs=epochs, accum_steps=accum_steps)

        if scheduler is not None:
            scheduler.step(epoch + train_loss)

        metrics = {"rmse": None, "mae": None, "r2": None}
        if (epoch % val_interval == 0) or (epoch == 1):
            val_metrics, y_true, y_pred = eval_model(model, val_loader, device)
            metrics = val_metrics

            # Early stopping tracking
            if val_metrics['rmse'] < best['val_rmse']:
                best.update({"val_rmse": val_metrics['rmse'], "epoch": epoch})
                patience = 0
                # checkpoint
                ckpt_dir = os.path.join(out_root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"soc_epoch{epoch:04d}_rmse{val_metrics['rmse']:.5f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'scaler_path': scaler_path,
                    'features': features,
                    'chunk': chunk,
                }, ckpt_path)
                best['path'] = ckpt_path
            else:
                patience += 1

        # Periodic checkpointing
        if save_every_n and (epoch % save_every_n == 0):
            ckpt_dir = os.path.join(out_root, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"soc_epoch{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'scaler_path': scaler_path,
                'features': features,
                'chunk': chunk,
            }, ckpt_path)

        # Track history for plots
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(metrics['rmse'] if metrics['rmse'] is not None else None)
        history["val_mae"].append(metrics['mae'] if metrics['mae'] is not None else None)
        history["val_r2"].append(metrics['r2'] if metrics['r2'] is not None else None)

        # CSV log
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
                'best_val_rmse': best['val_rmse'] if best['val_rmse'] != float('inf') else ''
            }
            save_csv_row(csv_path, row, header=csv_header)

        # After each evaluation (when val metrics exist) also write an updated PNG next to the CSV
        try:
            if csv_path and metrics.get('rmse') is not None:
                png_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else out_root
                os.makedirs(png_dir, exist_ok=True)
                png_path = os.path.join(png_dir, 'training_progress.png')
                plt.figure(figsize=(10,6))
                ax1 = plt.gca()
                ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='tab:blue')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax2 = ax1.twinx()
                if any(v is not None for v in history['val_rmse']):
                    xs = [e for e, v in zip(history['epoch'], history['val_rmse']) if v is not None]
                    ys = [v for v in history['val_rmse'] if v is not None]
                    if xs:
                        ax2.plot(xs, ys, label='Val RMSE', color='tab:red')
                if any(v is not None for v in history['val_mae']):
                    xs = [e for e, v in zip(history['epoch'], history['val_mae']) if v is not None]
                    ys = [v for v in history['val_mae'] if v is not None]
                    if xs:
                        ax2.plot(xs, ys, label='Val MAE', color='tab:orange')
                ax2.set_ylabel('Validation Metrics', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                plt.title('Training Progress')
                plt.tight_layout()
                plt.savefig(png_path, dpi=150)
                plt.close()
        except Exception as e:
            print(f"Progress-plot warn: {e}")

        # Concise epoch print
        if metrics['rmse'] is not None:
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_rmse={metrics['rmse']:.6f} | best_rmse={best['val_rmse']:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}")

        # Early stopping
        if patience >= early_stopping:
            break

    # Export best to ONNX if requested
    if cfg.get('export', {}).get('to_onnx', True) and best['path']:
        # Load best
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
            opset_version=int(cfg['export'].get('onnx_opset', 17))
        )

        # Optional: export a stateful ONNX for step-by-step streaming
        if cfg.get('export', {}).get('stateful_onnx', False):
            # dummy single step input and hidden/cell states
            f = len(features)
            hdim = hidden_size
            x1 = torch.zeros((1, 1, f), dtype=torch.float32).to(device)
            h0 = torch.zeros((model.lstm.num_layers, 1, hdim), dtype=torch.float32).to(device)
            c0 = torch.zeros((model.lstm.num_layers, 1, hdim), dtype=torch.float32).to(device)

            class StatefulWrapper(torch.nn.Module):
                def __init__(self, core):
                    super().__init__()
                    self.core = core

                def forward(self, x_step, h, c):
                    # x_step: [B=1, T=1, F]
                    pred, (h1, c1) = self.core(x_step, state=(h, c), return_state=True)
                    return pred, h1, c1

            wrapper = StatefulWrapper(model).to(device)
            onnx_path_stateful = os.path.join(out_root, f"soc_best_epoch{best['epoch']:04d}_stateful.onnx")
            torch.onnx.export(
                wrapper,
                (x1, h0, c0),
                onnx_path_stateful,
                input_names=['x_step', 'h0', 'c0'],
                output_names=['y_step', 'h1', 'c1'],
                dynamic_axes={
                    'x_step': {0: 'batch'},
                    'h0': {1: 'batch'},
                    'c0': {1: 'batch'},
                    'y_step': {0: 'batch'},
                    'h1': {1: 'batch'},
                    'c1': {1: 'batch'},
                },
                opset_version=int(cfg['export'].get('onnx_opset', 17))
            )
        # Save a small manifest
        save_json({
            'best_epoch': best['epoch'],
            'best_val_rmse': best['val_rmse'],
            'checkpoint': best['path'],
            'onnx': onnx_path,
            'scaler': scaler_path,
            'features': features,
            'chunk': chunk
        }, os.path.join(out_root, 'export_manifest.json'))

    # Save progress artifacts (history JSON + training_progress.png)
    progress_json = os.path.join(out_root, 'training_progress.json')
    save_json(history, progress_json)
    try:
        plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        if any(v is not None for v in history['val_rmse']):
            xs = [e for e, v in zip(history['epoch'], history['val_rmse']) if v is not None]
            ys = [v for v in history['val_rmse'] if v is not None]
            if xs:
                ax2.plot(xs, ys, label='Val RMSE', color='tab:red')
        if any(v is not None for v in history['val_mae']):
            xs = [e for e, v in zip(history['epoch'], history['val_mae']) if v is not None]
            ys = [v for v in history['val_mae'] if v is not None]
            if xs:
                ax2.plot(xs, ys, label='Val MAE', color='tab:orange')
        ax2.set_ylabel('Validation Metrics', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.title('Training Progress')
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, 'training_progress.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
