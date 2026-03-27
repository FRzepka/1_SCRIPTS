import os
import json
import time
import math
import argparse
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


# -------------------------
# Model definition (SOH)
# -------------------------

class LSTMMLP_SOH(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),  # linear head for SOH
        )

    def forward(self, x, state=None, return_state: bool = False):
        out, new_state = self.lstm(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


# -------------------------
# Dataset utilities (reuse from training)
# -------------------------

class SeqDataset(Dataset):
    def __init__(self, df, features: List[str], target: str, chunk: int, stride: int, scaler: RobustScaler):
        import pandas as pd
        df = df.reset_index(drop=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
        X = df[features].to_numpy(dtype=np.float32)
        y = df[target].to_numpy(dtype=np.float32)
        Xs = scaler.transform(X).astype(np.float32)
        self.X = torch.from_numpy(Xs)
        self.y = torch.from_numpy(y)
        self.features = features
        self.target = target
        self.chunk = int(chunk)
        self.stride = max(1, int(stride))
        n = len(df)
        self.nseq = 0 if n < self.chunk else 1 + (n - self.chunk) // self.stride

    def __len__(self):
        return self.nseq

    def __getitem__(self, idx):
        s = idx * self.stride
        e = s + self.chunk
        return self.X[s:e], self.y[e - 1]


# Top-level collate (picklable on Windows)
def collate_batch_soh(batch):
    import torch as _torch
    return (_torch.stack([x for x, _ in batch]), _torch.stack([y for _, y in batch]))


def load_cell_dataframe(data_root: str, cell: str):
    import os, pandas as pd
    print(f"[data] Loading parquet for {cell}...", flush=True)
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
    df = pd.read_parquet(path)
    print(f"[data] Loaded {cell}: {len(df)} rows", flush=True)
    return df


def create_dataloaders(cfg: dict, features: List[str], chunk: int, train_batch_size: int = 256, val_batch_size: int = 128, pin_memory: bool = True, num_workers: int = None):
    # Fit scaler on train features
    import pandas as pd
    print("[data] Loading training cells...", flush=True)
    scaler = RobustScaler()
    train_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['train']]
    print("[data] Loading validation cells...", flush=True)
    val_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['val']]
    
    print("[data] Concatenating training data for scaler fit...", flush=True)
    x_train = pd.concat([d[features] for d in train_dfs], axis=0)
    print(f"[data] Fitting RobustScaler on {len(x_train)} rows...", flush=True)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))
    print("[data] Scaler fitted.", flush=True)

    stride = int(cfg.get('training', {}).get('window_stride', 1))
    train_ds = [SeqDataset(d, features, 'SOH', chunk, stride, scaler) for d in train_dfs]
    val_ds = [SeqDataset(d, features, 'SOH', chunk, stride, scaler) for d in val_dfs]

    # Determine num_workers
    if num_workers is not None:
        nw = num_workers
    else:
        cpu_cnt = os.cpu_count() or 8
        # On Windows, avoid multiprocessing issues (pickling) — use single-worker by default
        import os as _os
        if _os.name == 'nt':
            nw = 0
        else:
            nw = min(8, max(2, cpu_cnt - 2))
    
    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_ds),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin_memory,
        collate_fn=collate_batch_soh,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_ds),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
        collate_fn=collate_batch_soh,
    )
    return scaler, train_loader, val_loader


# -------------------------
# Metrics and helpers
# -------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"rmse": None, "mae": None, "r2": None}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _to_serializable(obj):
    """Recursively convert numpy containers into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


@torch.no_grad()
def eval_model(model, loader, device, microbatch: int = None, progress: bool = False, desc: str = "eval"):
    model.eval()
    preds, ys = [], []
    iterator = tqdm(loader, total=len(loader), desc=desc, leave=False) if progress else loader
    for xb, yb in iterator:
        xb = xb.to(device, non_blocking=True)
        if microbatch and xb.size(0) > microbatch:
            batch_preds = []
            for xb_mb in torch.split(xb, microbatch, dim=0):
                pred_mb = model(xb_mb)
                batch_preds.append(pred_mb.detach().cpu())
            pred = torch.cat(batch_preds, dim=0)
        else:
            pred = model(xb).detach().cpu()
        preds.append(pred.numpy())
        ys.append(yb.numpy())
    if not preds:
        return {"rmse": None, "mae": None, "r2": None}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    return compute_metrics(y_true, y_pred)


# -------------------------
# LSTM unit saliency and pruning
# -------------------------

def lstm_unit_saliency(lstm: nn.LSTM) -> np.ndarray:
    """Compute unit-wise magnitude saliency across gates and both ih/hh for layer 0.
    Assumes num_layers == 1 for now.
    Returns: scores shape [hidden_size]
    """
    assert lstm.num_layers == 1, "This pruning script currently supports num_layers==1"
    H = lstm.hidden_size
    Wih = lstm.weight_ih_l0.detach().cpu().numpy()  # [4H, in]
    Whh = lstm.weight_hh_l0.detach().cpu().numpy()  # [4H, H]
    Wih_g = Wih.reshape(4, H, -1)
    Whh_g = Whh.reshape(4, H, H)
    s = np.zeros(H, dtype=np.float64)
    for g in range(4):
        s += np.linalg.norm(Wih_g[g], axis=1)
        s += np.linalg.norm(Whh_g[g], axis=1)
    return s.astype(np.float64)


def prune_lstm_mlp(model: LSTMMLP_SOH, keep_idx: np.ndarray) -> LSTMMLP_SOH:
    old = model
    lstm = old.lstm
    assert lstm.num_layers == 1, "This pruning script currently supports num_layers==1"
    H = lstm.hidden_size
    Hn = int(len(keep_idx))
    in_f = lstm.input_size

    new_model = LSTMMLP_SOH(in_features=in_f, hidden_size=Hn, mlp_hidden=old.mlp[0].out_features,
                            num_layers=1, dropout=0.05)

    with torch.no_grad():
        Wih = lstm.weight_ih_l0.data.clone()  # [4H, in]
        Whh = lstm.weight_hh_l0.data.clone()  # [4H, H]
        bih = lstm.bias_ih_l0.data.clone() if hasattr(lstm, 'bias_ih_l0') else None
        bhh = lstm.bias_hh_l0.data.clone() if hasattr(lstm, 'bias_hh_l0') else None

        def gate_slice(mat_rows, gate, idx):
            Hloc = H
            start = gate * Hloc
            end = (gate + 1) * Hloc
            return mat_rows[start:end][idx]

        Wih_new = []
        Whh_new = []
        bih_new = [] if bih is not None else None
        bhh_new = [] if bhh is not None else None
        for g in range(4):
            Wih_g = gate_slice(Wih, g, keep_idx)
            Whh_g_rows = gate_slice(Whh, g, keep_idx)
            Whh_g_rows_cols = Whh_g_rows[:, keep_idx]
            Wih_new.append(Wih_g)
            Whh_new.append(Whh_g_rows_cols)
            if bih is not None:
                bih_g = gate_slice(bih.unsqueeze(1), g, keep_idx).squeeze(1)
                bih_new.append(bih_g)
            if bhh is not None:
                bhh_g = gate_slice(bhh.unsqueeze(1), g, keep_idx).squeeze(1)
                bhh_new.append(bhh_g)

        Wih_new = torch.cat(Wih_new, dim=0)
        Whh_new = torch.cat(Whh_new, dim=0)
        if bih is not None:
            bih_new = torch.cat(bih_new, dim=0)
        if bhh is not None:
            bhh_new = torch.cat(bhh_new, dim=0)

        new_model.lstm.weight_ih_l0.copy_(Wih_new)
        new_model.lstm.weight_hh_l0.copy_(Whh_new)
        if hasattr(new_model.lstm, 'bias_ih_l0') and bih is not None:
            new_model.lstm.bias_ih_l0.copy_(bih_new)
        if hasattr(new_model.lstm, 'bias_hh_l0') and bhh is not None:
            new_model.lstm.bias_hh_l0.copy_(bhh_new)

        # Adjust MLP first layer: Linear(H, mlp_hidden) -> slice columns
        lin0_old: nn.Linear = old.mlp[0]
        lin0_new: nn.Linear = new_model.mlp[0]
        lin0_new.bias.copy_(lin0_old.bias.data)
        lin0_new.weight.copy_(lin0_old.weight.data[:, keep_idx])

        # Copy rest layers as-is
        for i in range(2, len(old.mlp)):
            if isinstance(old.mlp[i], nn.Linear) and isinstance(new_model.mlp[i], nn.Linear):
                new_model.mlp[i].weight.copy_(old.mlp[i].weight.data)
                new_model.mlp[i].bias.copy_(old.mlp[i].bias.data)

    return new_model


def train_finetune(model, train_loader, val_loader, device, epochs=5, lr=5e-5, max_grad_norm=1.0, microbatch: int = None, eval_microbatch: int = None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    loss_fn = nn.MSELoss()
    best = {"rmse": float('inf')}
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        tbar = tqdm(train_loader, total=len(train_loader), desc=f"train[{ep}/{epochs}]", leave=False)
        for xb, yb in tbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if microbatch and xb.size(0) > microbatch:
                mb_losses = []
                for xb_mb, yb_mb in zip(torch.split(xb, microbatch, dim=0), torch.split(yb, microbatch, dim=0)):
                    pred_mb = model(xb_mb)
                    loss_mb = loss_fn(pred_mb, yb_mb)
                    loss_mb.backward()
                    mb_losses.append(loss_mb.detach())
                loss = torch.stack(mb_losses).mean()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {ep}, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()

            # Check for NaNs/Infs in gradients and sanitize instead of bailing out repeatedly
            has_bad_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_bad_grad = True
                    # replace non-finite entries with 0 to keep training stable
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if has_bad_grad:
                print(f"Warning: non-finite gradient detected at epoch {ep}. Sanitized grads and continuing.", flush=True)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            opt.step()
            total += float(loss.detach().cpu()) * xb.size(0)
            count += xb.size(0)
            tbar.set_postfix({"loss": total / max(1, count)})
        # Eval each epoch
        val = eval_model(model, val_loader, device, microbatch=eval_microbatch or microbatch, progress=False, desc="val")
        if val['rmse'] is not None and val['rmse'] < best.get('rmse', float('inf')):
            best = val
    return best


def main():
    ap = argparse.ArgumentParser(description='Prune SOH LSTM+MLP model (unit-wise hidden pruning + brief finetune)')
    ap.add_argument('--yaml', required=True, help='Path to train_soh.yaml')
    ap.add_argument('--ckpt', required=True, help='Path to baseline SOH checkpoint (.pt)')
    ap.add_argument('--out-dir', required=True, help='Output directory for pruned artifacts')
    ap.add_argument('--prune-ratio', type=float, default=0.3)
    ap.add_argument('--finetune-epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--train-batch', type=int, default=None)
    ap.add_argument('--val-batch', type=int, default=None)
    ap.add_argument('--microbatch', type=int, default=None)
    ap.add_argument('--eval-microbatch', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=None, help='DataLoader workers (default: auto, 0 = no multiprocessing)')
    ap.add_argument('--export-stateful', action='store_true', help='Also export stateful ONNX wrapper')
    ap.add_argument('--skip-onnx', action='store_true', help='Skip ONNX exports (dense/stateful)')
    ap.add_argument('--data-root', default='', help='Override data_root (expands env like ${VAR:-default})')
    ap.add_argument('--no-data', action='store_true', help='Skip dataloaders/eval/finetune (prune weights only)')
    args = ap.parse_args()

    print("[init] Starting pruning run...", flush=True)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[config] device={device.type} | prune_ratio={args.prune_ratio} | finetune_epochs={args.finetune_epochs}", flush=True)

    # Load baseline checkpoint and config
    print(f"[load] Loading checkpoint: {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location='cpu')
    print("[load] Checkpoint loaded.", flush=True)
    with open(args.yaml, 'r', encoding='utf-8') as f:
        import yaml
        cfg = yaml.safe_load(f)

    features: List[str] = state.get('features') or cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    assert num_layers == 1, 'Current pruner supports num_layers==1 only'
    chunk = int(state.get('chunk') or cfg['training']['seq_chunk_size'])

    # Build and load baseline
    print(f"[model] Preparing baseline with {len(features)} feature(s)", flush=True)
    base = LSTMMLP_SOH(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers)
    base.load_state_dict(state['model_state_dict'])
    base.to(device)
    print(f"[model] features={len(features)} | hidden_size={hidden_size} | mlp_hidden={mlp_hidden} | chunk={chunk}", flush=True)

    # Resolve data_root from YAML + env or --data-root
    def _resolve_env_default(s: str) -> str:
        import os, re
        if not isinstance(s, str):
            return s
        # Pattern: ${VAR:-default}
        m = re.match(r'^\$\{([^}:]+)(?::-(.+))?\}$', s)
        if m:
            var = m.group(1)
            default = m.group(2) if m.group(2) is not None else ''
            return os.environ.get(var, default)
        # Fallback simple expansion (${VAR})
        return os.path.expandvars(s)

    if args.data_root:
        cfg['paths']['data_root'] = args.data_root
    else:
        cfg['paths']['data_root'] = _resolve_env_default(cfg['paths']['data_root'])

    print(f"[data_root] {cfg['paths']['data_root']}", flush=True)

    # Data
    default_train_bs = int(cfg['training'].get('batch_size', 256))
    default_val_bs = min(128, default_train_bs)
    train_bs = args.train_batch or default_train_bs
    val_bs = args.val_batch or default_val_bs
    pin_memory = (device.type == 'cuda')
    if not args.no_data:
        print("[data] Building dataloaders (this may take a moment)...", flush=True)
        scaler, train_loader, val_loader = create_dataloaders(cfg, features, chunk, train_batch_size=train_bs, val_batch_size=val_bs, pin_memory=pin_memory, num_workers=args.num_workers)
        try:
            print(f"[data] train_bs={train_bs} | val_bs={val_bs} | pin_memory={pin_memory}", flush=True)
            print(f"[data] train_batches={len(train_loader)} | val_batches={len(val_loader)}", flush=True)
        except Exception:
            pass
    else:
        scaler = None
        train_loader = None
        val_loader = None
        print("[data] Skipping dataloaders (no-data mode)")

    # Baseline metrics
    if val_loader is not None:
        print("[eval] Baseline evaluation...", flush=True)
        base_val = eval_model(base, val_loader, device, microbatch=args.eval_microbatch or args.microbatch, progress=True, desc="baseline-val")
        print(f"Baseline val: rmse={base_val['rmse']:.6f} mae={base_val['mae']:.6f} r2={base_val['r2']:.6f}")
    else:
        base_val = {'rmse': None, 'mae': None, 'r2': None}

    # Saliency and keep indices
    scores = lstm_unit_saliency(base.lstm)
    H = base.lstm.hidden_size
    remove = int(round(H * args.prune_ratio))
    keep = max(1, H - remove)
    keep_idx = np.argsort(scores)[-keep:]
    keep_idx = np.sort(keep_idx)
    print(f"[prune] hidden_size {H} -> {keep} (remove {remove})", flush=True)

    # Prune
    pruned = prune_lstm_mlp(base, keep_idx)
    pruned.to(device)
    print("[prune] Model pruned and moved to device", flush=True)

    # Finetune
    if args.finetune_epochs > 0 and train_loader is not None and val_loader is not None:
        print(f"[finetune] Starting fine-tune for {args.finetune_epochs} epoch(s), lr={args.lr}, microbatch={args.microbatch}, eval_microbatch={args.eval_microbatch}", flush=True)
        best_after = train_finetune(pruned, train_loader, val_loader, device, epochs=args.finetune_epochs, lr=args.lr, microbatch=args.microbatch, eval_microbatch=args.eval_microbatch)
    else:
        if args.finetune_epochs > 0 and (train_loader is None or val_loader is None):
            print("[finetune] Skipped because dataloaders unavailable", flush=True)
        else:
            print("[finetune] Skipped (epochs=0)", flush=True)
        best_after = {'rmse': None, 'mae': None, 'r2': None}

    # Prepare out dir
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, f"prune_{int(args.prune_ratio*100)}pct_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Save pruned checkpoint
    ckpt_path = os.path.join(out_dir, f"soh_pruned_hidden{pruned.lstm.hidden_size}.pt")
    torch.save({
        'epoch': state.get('epoch', -1),
        'model_state_dict': pruned.state_dict(),
        'config': cfg,
        'features': features,
        'chunk': chunk,
        'pruned': True,
        'keep_indices': keep_idx.tolist(),
        'baseline_ckpt': args.ckpt,
    }, ckpt_path)

    # Export ONNX (dense)
    dummy = torch.zeros((1, chunk, len(features)), dtype=torch.float32).to(device)
    onnx_path = os.path.join(out_dir, f"soh_pruned_hidden{pruned.lstm.hidden_size}.onnx")
    if not args.skip_onnx:
        print("[export] Exporting dense ONNX...")
        torch.onnx.export(
            pruned.eval(),
            dummy,
            onnx_path,
            input_names=['input_seq'],
            output_names=['soh'],
            dynamic_axes={'input_seq': {0: 'batch'}, 'soh': {0: 'batch'}},
            opset_version=17,
        )
        try:
            onnx_sz = os.path.getsize(onnx_path)
            print(f"[export] ONNX saved: {onnx_path} ({onnx_sz/1024:.1f} KB)")
        except Exception:
            pass
    else:
        onnx_path = os.path.join(out_dir, f"soh_pruned_hidden{pruned.lstm.hidden_size}.onnx.skipped")
        open(onnx_path, 'w').write('skipped')

    if args.export_stateful and not args.skip_onnx:
        f = len(features)
        hdim = pruned.lstm.hidden_size
        x1 = torch.zeros((1, 1, f), dtype=torch.float32).to(device)
        h0 = torch.zeros((pruned.lstm.num_layers, 1, hdim), dtype=torch.float32).to(device)
        c0 = torch.zeros((pruned.lstm.num_layers, 1, hdim), dtype=torch.float32).to(device)

        class StatefulWrapper(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x_step, h, c):
                pred, (h1, c1) = self.core(x_step, state=(h, c), return_state=True)
                return pred, h1, c1

        wrapper = StatefulWrapper(pruned).to(device)
        onnx_path_stateful = os.path.join(out_dir, f"soh_pruned_hidden{hdim}_stateful.onnx")
        print("[export] Exporting stateful ONNX...")
        torch.onnx.export(
            wrapper,
            (x1, h0, c0),
            onnx_path_stateful,
            input_names=['x_step', 'h0', 'c0'],
            output_names=['y_step', 'h1', 'c1'],
            dynamic_axes={'x_step': {0: 'batch'}, 'h0': {1: 'batch'}, 'c0': {1: 'batch'}, 'y_step': {0: 'batch'}, 'h1': {1: 'batch'}, 'c1': {1: 'batch'}},
            opset_version=17,
        )
        try:
            onnx_sz2 = os.path.getsize(onnx_path_stateful)
            print(f"[export] Stateful ONNX saved: {onnx_path_stateful} ({onnx_sz2/1024:.1f} KB)")
        except Exception:
            pass

    # Save manifest
    manifest = {
        'baseline': {
            'hidden_size': int(H),
            'val': base_val,
        },
        'pruned': {
            'hidden_size': int(pruned.lstm.hidden_size),
            'val_after_finetune': best_after,
            'keep_indices': keep_idx.tolist(),
            'checkpoint': ckpt_path,
            'onnx': onnx_path,
        },
        'prune_ratio': args.prune_ratio,
    }
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(_to_serializable(manifest), f, indent=2)

    print(f"[done] Pruned model saved to: {out_dir}")


if __name__ == '__main__':
    # Local imports used above
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    main()
