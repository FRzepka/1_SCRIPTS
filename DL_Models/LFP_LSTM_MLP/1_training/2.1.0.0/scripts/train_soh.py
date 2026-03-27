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
# Model (SOH: linear head)
# -------------------------

class LSTMMLP_SOHTarget(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)  # linear output (SOH not constrained to [0,1] strictly; dataset likely in 0-1)
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
    target: str = 'SOH'
    chunk: int = 4096
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
        y_last = self.y[end - 1]
        return x_seq, y_last

# -------------------------
# Helpers
# -------------------------

def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, sch_cfg):
    sch = sch_cfg.get('scheduler', sch_cfg.get('type')) if isinstance(sch_cfg, dict) else None
    if isinstance(sch_cfg, dict) and sch_cfg.get('type') == 'cosine_warm_restarts':
        T_0 = int(sch_cfg.get('T_0', 150))
        eta_min_factor = float(sch_cfg.get('eta_min_factor', 0.05))
        base_lrs = [g['lr'] for g in optimizer.param_groups]
        eta_min = min(base_lrs) * eta_min_factor
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2}


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
# Env expansion with default support (${VAR:-default})
# -------------------------

_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

def expand_env_with_defaults(val: str) -> str:
    """Expand ${VAR:-default} patterns plus normal $VAR using environment.

    Python's os.path.expandvars doesn't understand the ':-' bash default syntax,
    so we manually resolve those first, then let expandvars handle plain $VAR.
    """
    if not isinstance(val, str):
        return val
    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, '') else default
    val2 = _ENV_DEFAULT_PATTERN.sub(repl, val)
    # second pass for any remaining ${VAR} or $VAR patterns
    return os.path.expandvars(val2)

# -------------------------
# Data loading
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

# -------------------------
# Training / Eval loops
# -------------------------

def train_one_epoch(model, loader, device, optimizer, scaler_amp, max_grad_norm, epoch_idx: int, total_epochs: int, accum_steps: int = 1):
    model.train()
    loss_fn = nn.MSELoss()  # use MSE for smoother gradients; MAE primary metric
    pbar = tqdm(loader, desc=f'Epoch {epoch_idx}/{total_epochs} • train', leave=False)
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (xb, yb) in enumerate(pbar):
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

# -------------------------
# Dataloaders
# -------------------------

def create_dataloaders(cfg: dict, features: T.List[str], chunk: int, scaler: RobustScaler, batch_size: int = 64):
    train_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['train']]
    val_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['val']]

    x_train = pd.concat([d[features] for d in train_dfs], axis=0)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))

    stride = int(cfg.get('training', {}).get('window_stride', 1))
    train_ds = [SeqDataset(d, SeqConfig(features=features, target='SOH', chunk=chunk, stride=stride), scaler) for d in train_dfs]
    val_ds = [SeqDataset(d, SeqConfig(features=features, target='SOH', chunk=chunk, stride=stride), scaler) for d in val_dfs]

    dl_cfg = cfg.get('dataloader', {})
    cpu_cnt = os.cpu_count() or 8
    num_workers = int(dl_cfg.get('num_workers', max(4, min(16, cpu_cnt - 2))))
    prefetch_factor = int(dl_cfg.get('prefetch_factor', 6))
    pin_memory = bool(dl_cfg.get('pin_memory', True))
    persistent_workers = bool(dl_cfg.get('persistent_workers', True)) if num_workers > 0 else False

    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        common_kwargs.update(dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers))

    collate = lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b]))

    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_ds), shuffle=True, drop_last=True, collate_fn=collate, **common_kwargs)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_ds), shuffle=False, collate_fn=collate, **common_kwargs)
    return train_loader, val_loader

# -------------------------
# Main
# -------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'train_soh.yaml'))
    # New flexible overrides for multi-cluster usage
    ap.add_argument('--data-root', type=str, default=None, help='Override data root (else from config / env)')
    ap.add_argument('--out-root', type=str, default=None, help='Base output root (without site/run-id). If given combined with site + run-id.')
    ap.add_argument('--site', type=str, default=None, help='Logical site label (e.g. TU, EET, local). Auto-detected if omitted.')
    ap.add_argument('--run-id', type=str, default=None, help='Optional run identifier (e.g. timestamp). Auto-created if needed.')
    ap.add_argument('--output-layout', type=str, default=None, choices=['flat','site_run'],
                    help='"flat" = use out_root exactly as given in config/override. "site_run" = out_root/<site>/<run-id>. Default: site_run inside Slurm, flat otherwise.')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Expand env variables (supporting ${VAR:-default}) in config paths
    for k in list(cfg.get('paths', {}).keys()):
        if isinstance(cfg['paths'][k], str):
            cfg['paths'][k] = expand_env_with_defaults(cfg['paths'][k])

    # Determine execution context (Slurm vs direct)
    in_slurm = any(os.getenv(v) for v in ['SLURM_JOB_ID','SLURM_JOB_NAME'])

    # Determine site label priority: CLI > ENV > heuristic > fallback
    if args.site:
        site = args.site
    elif os.getenv('HPC_SITE'):
        site = os.getenv('HPC_SITE')
    else:
        # simple heuristic: if running under slurm assume TU unless overridden
        site = 'TU' if in_slurm else 'EET'

    # Decide output layout mode
    layout = args.output_layout or ('site_run' if in_slurm else 'flat')

    # Resolve data root
    data_root_override = args.data_root or os.getenv('DATA_ROOT')
    if data_root_override:
        cfg['paths']['data_root'] = data_root_override

    # Resolve output root strategy
    # Priority: --out-root > env OUTPUT_BASE/OUT_ROOT > config out_root
    out_root_base = args.out_root or os.getenv('OUT_ROOT') or os.getenv('OUTPUT_BASE') or cfg['paths'].get('out_root')

    # Normalise base (expand again in case of env placeholders)
    out_root_base = os.path.expandvars(out_root_base)

    if layout == 'site_run':
        run_id = args.run_id or os.getenv('RUN_ID') or time.strftime('%Y%m%d_%H%M%S')
        out_root = os.path.join(out_root_base, site, run_id)
    else:
        # flat layout ignores run-id unless explicitly supplied (then make subfolder)
        if args.run_id or os.getenv('RUN_ID'):
            rid = args.run_id or os.getenv('RUN_ID')
            out_root = os.path.join(out_root_base, rid)
        else:
            out_root = out_root_base
    cfg['paths']['out_root'] = out_root

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

    primary_metric = cfg.get('metrics', {}).get('primary', 'mae')

    scaler = RobustScaler()
    train_loader, val_loader = create_dataloaders(cfg, features, chunk, scaler, batch_size=batch_size)

    from joblib import dump
    scaler_path = os.path.join(out_root, 'scaler_robust.joblib')
    dump(scaler, scaler_path)

    model = LSTMMLP_SOHTarget(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
    if cfg['training'].get('compile', False):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('torch.compile: enabled')
        except Exception as e:
            print(f'torch.compile failed: {e}')

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, cfg['training'].get('scheduler', {}))
    amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Adjust tracking csv path: if config path contains original absolute prefix or env placeholder, rewrite into new out_root
    # Expand possible env syntax in tracking sub-config
    if cfg.get('tracking'):
        for tk in list(cfg['tracking'].keys()):
            if isinstance(cfg['tracking'][tk], str):
                cfg['tracking'][tk] = expand_env_with_defaults(cfg['tracking'][tk])
    csv_path_cfg = cfg.get('tracking', {}).get('csv_file')
    csv_path = None
    if cfg.get('tracking', {}).get('csv_log', True):
        if not csv_path_cfg or '${' in csv_path_cfg or '$' in csv_path_cfg:
            csv_path = os.path.join(out_root, 'training_log.csv')
        else:
            # if original path points elsewhere, mirror into new out_root for isolation
            csv_path = os.path.join(out_root, 'training_log.csv')
        cfg['tracking']['csv_file'] = csv_path
    csv_header = ['epoch', 'train_loss', 'val_mae', 'val_rmse', 'val_r2', 'lr', 'scaler_path', 'checkpoint', 'best_val_metric']

    best = {"metric": float('inf'), "epoch": -1, "path": None}
    patience = 0
    save_every_n = int(cfg.get('export', {}).get('save_every_n', 0) or 0)
    history = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}

    dummy = torch.zeros((1, chunk, len(features)), dtype=torch.float32).to(device)

    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    print(f"Site: {site} | Layout: {layout} | Output root: {out_root}")
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
    print(f"Primary metric for selection: {primary_metric}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch_idx=epoch, total_epochs=epochs, accum_steps=accum_steps)
        if scheduler is not None:
            scheduler.step(epoch + train_loss)

        metrics = {"mae": None, "rmse": None, "r2": None}
        if (epoch % val_interval == 0) or (epoch == 1):
            val_metrics, y_true, y_pred = eval_model(model, val_loader, device)
            metrics = val_metrics
            current = metrics.get(primary_metric, float('inf')) if metrics.get(primary_metric) is not None else float('inf')
            if current < best['metric']:
                best.update({"metric": current, "epoch": epoch})
                patience = 0
                ckpt_dir = os.path.join(out_root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"soh_epoch{epoch:04d}_{primary_metric}{current:.5f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'scaler_path': scaler_path,
                    'features': features,
                    'chunk': chunk,
                    'primary_metric': primary_metric,
                }, ckpt_path)
                best['path'] = ckpt_path
            else:
                patience += 1

        if save_every_n and (epoch % save_every_n == 0):
            ckpt_dir = os.path.join(out_root, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"soh_epoch{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'scaler_path': scaler_path,
                'features': features,
                'chunk': chunk,
                'primary_metric': primary_metric,
            }, ckpt_path)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_mae'].append(metrics['mae'] if metrics['mae'] is not None else None)
        history['val_rmse'].append(metrics['rmse'] if metrics['rmse'] is not None else None)
        history['val_r2'].append(metrics['r2'] if metrics['r2'] is not None else None)

        if csv_path:
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mae': metrics['mae'] if metrics['mae'] is not None else '',
                'val_rmse': metrics['rmse'] if metrics['rmse'] is not None else '',
                'val_r2': metrics['r2'] if metrics['r2'] is not None else '',
                'lr': optimizer.param_groups[0]['lr'],
                'scaler_path': scaler_path,
                'checkpoint': best['path'] or '',
                'best_val_metric': best['metric'] if best['metric'] != float('inf') else ''
            }
            save_csv_row(csv_path, row, header=csv_header)

        # Also emit an updated training_progress.png next to the CSV after each evaluation
        try:
            if csv_path and metrics.get('mae') is not None:
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
                if any(v is not None for v in history['val_mae']):
                    ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_mae']], label='Val MAE', color='tab:orange')
                if any(v is not None for v in history['val_rmse']):
                    ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_rmse']], label='Val RMSE', color='tab:red')
                ax2.set_ylabel('Validation Metrics', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                plt.title('SOH Training Progress')
                plt.tight_layout()
                plt.savefig(png_path, dpi=150)
                plt.close()
        except Exception as e:
            print(f'Progress-plot warn: {e}')

        if metrics['mae'] is not None:
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_mae={metrics['mae']:.6f} | best_{primary_metric}={best['metric']:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}")

        if patience >= early_stopping:
            print('Early stopping triggered.')
            break

    # Export ONNX
    if cfg.get('export', {}).get('to_onnx', True) and best['path']:
        state = torch.load(best['path'], map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        onnx_path = os.path.join(out_root, f"soh_best_epoch{best['epoch']:04d}.onnx")
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['input_seq'],
            output_names=['soh'],
            dynamic_axes={'input_seq': {0: 'batch'}, 'soh': {0: 'batch'}},
            opset_version=int(cfg['export'].get('onnx_opset', 17))
        )
        if cfg.get('export', {}).get('stateful_onnx', False):
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
                    pred, (h1, c1) = self.core(x_step, state=(h, c), return_state=True)
                    return pred, h1, c1
            wrapper = StatefulWrapper(model).to(device)
            onnx_path_stateful = os.path.join(out_root, f"soh_best_epoch{best['epoch']:04d}_stateful.onnx")
            torch.onnx.export(
                wrapper,
                (x1, h0, c0),
                onnx_path_stateful,
                input_names=['x_step', 'h0', 'c0'],
                output_names=['y_step', 'h1', 'c1'],
                dynamic_axes={'x_step': {0: 'batch'}, 'h0': {1: 'batch'}, 'c0': {1: 'batch'}, 'y_step': {0: 'batch'}, 'h1': {1: 'batch'}, 'c1': {1: 'batch'}},
                opset_version=int(cfg['export'].get('onnx_opset', 17))
            )
        save_json({
            'best_epoch': best['epoch'],
            'best_val_metric': best['metric'],
            'primary_metric': primary_metric,
            'checkpoint': best['path'],
            'onnx': onnx_path,
            'scaler': scaler_path,
            'features': features,
            'chunk': chunk
        }, os.path.join(out_root, 'export_manifest.json'))

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
        if any(v is not None for v in history['val_mae']):
            ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_mae']], label='Val MAE', color='tab:orange')
        if any(v is not None for v in history['val_rmse']):
            ax2.plot(history['epoch'], [v if v is not None else float('nan') for v in history['val_rmse']], label='Val RMSE', color='tab:red')
        ax2.set_ylabel('Validation Metrics', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.title('SOH Training Progress')
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, 'training_progress.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f'Plotting failed: {e}')

if __name__ == '__main__':
    main()
