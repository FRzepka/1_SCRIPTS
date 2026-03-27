#!/usr/bin/env python
"""
Advanced structured pruning for SOC LSTM+MLP models.

Adds options on top of prune_lstm_soc.py:
 - --saliency {unit,gate-aware} to choose saliency type for LSTM unit pruning
 - --combine {sum,mean,max,min} to combine gate-aware saliency across gates
 - --mlp-prune-ratio to prune neurons in the MLP hidden layer structurally
 - --export-stateful to also export a stateful streaming ONNX (x_step, h0, c0)

Default behavior matches unit-wise LSTM pruning without MLP pruning.
"""
import os
import json
import time
import math
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


class LSTMMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, state=None, return_state: bool = False):
        out, new_state = self.lstm(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


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


def load_cell_dataframe(data_root: str, cell: str):
    import os, pandas as pd
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


def create_dataloaders(cfg: dict, features: List[str], chunk: int, train_batch_size: int = 256, val_batch_size: int = 128, pin_memory: bool = True):
    import pandas as pd
    scaler = RobustScaler()
    train_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['train']]
    val_dfs = [load_cell_dataframe(cfg['paths']['data_root'], c) for c in cfg['cells']['val']]
    x_train = pd.concat([d[features] for d in train_dfs], axis=0)
    scaler.fit(x_train.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))
    stride = int(cfg.get('training', {}).get('window_stride', 1))
    train_ds = [SeqDataset(d, features, 'SOC', chunk, stride, scaler) for d in train_dfs]
    val_ds = [SeqDataset(d, features, 'SOC', chunk, stride, scaler) for d in val_dfs]

    collate = lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b]))
    cpu_cnt = os.cpu_count() or 8
    num_workers = min(8, max(2, cpu_cnt - 2))
    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_ds),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_ds),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return scaler, train_loader, val_loader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


@torch.no_grad()
def eval_model(model, loader, device, microbatch: int = None, progress: bool = False, desc: str = "eval"):
    model.eval()
    preds, ys = [], []
    iterator = tqdm(loader, total=len(loader), desc=desc, leave=False) if progress else loader
    for xb, yb in iterator:
        if device.type == 'cuda':
            xb = xb.to(device, non_blocking=True)
        if microbatch and xb.size(0) > microbatch:
            outs = []
            for xb_mb in torch.split(xb, microbatch, dim=0):
                out_mb = model(xb_mb).detach().cpu()
                outs.append(out_mb)
            pred = torch.cat(outs, dim=0)
        else:
            pred = model(xb).detach().cpu()
        preds.append(pred.numpy())
        ys.append(yb.numpy())
    if not preds:
        return {"rmse": None, "mae": None, "r2": None}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    return compute_metrics(y_true, y_pred)


def lstm_unit_saliency(lstm: nn.LSTM, mode: str = 'unit', combine: str = 'sum') -> np.ndarray:
    """Compute unit saliency.
    mode='unit': magnitude across Wih/Whh aggregated over all gates (unit-wise)
    mode='gate-aware': per-gate unit magnitude then combine across gates by {sum,mean,max,min}
    """
    assert lstm.num_layers == 1, "Supports num_layers==1"
    H = lstm.hidden_size
    Wih = lstm.weight_ih_l0.detach().cpu().numpy()  # [4H, in]
    Whh = lstm.weight_hh_l0.detach().cpu().numpy()  # [4H, H]
    Wih_g = Wih.reshape(4, H, -1)
    Whh_g = Whh.reshape(4, H, H)
    if mode == 'unit':
        s = np.zeros(H, dtype=np.float64)
        for g in range(4):
            s += np.linalg.norm(Wih_g[g], axis=1)
            s += np.linalg.norm(Whh_g[g], axis=1)
        return s
    # gate-aware
    gate_scores = []
    for g in range(4):
        sg = np.linalg.norm(Wih_g[g], axis=1) + np.linalg.norm(Whh_g[g], axis=1)  # [H]
        gate_scores.append(sg)
    GS = np.stack(gate_scores, axis=0)  # [4, H]
    if combine == 'sum':
        return GS.sum(axis=0)
    if combine == 'mean':
        return GS.mean(axis=0)
    if combine == 'max':
        return GS.max(axis=0)
    if combine == 'min':
        return GS.min(axis=0)
    return GS.sum(axis=0)


def prune_lstm_units(model: LSTMMLP, keep_idx: np.ndarray) -> LSTMMLP:
    old = model
    lstm = old.lstm
    assert lstm.num_layers == 1, "Supports num_layers==1"
    H = lstm.hidden_size
    Hn = int(len(keep_idx))
    in_f = lstm.input_size

    new_model = LSTMMLP(in_features=in_f, hidden_size=Hn, mlp_hidden=old.mlp[0].out_features, num_layers=1, dropout=0.05)

    with torch.no_grad():
        Wih = lstm.weight_ih_l0.data.clone()  # [4H, in]
        Whh = lstm.weight_hh_l0.data.clone()  # [4H, H]
        bih = lstm.bias_ih_l0.data.clone() if hasattr(lstm, 'bias_ih_l0') else None
        bhh = lstm.bias_hh_l0.data.clone() if hasattr(lstm, 'bias_hh_l0') else None

        def gate_slice(mat_rows, gate, idx):
            start = gate * H
            end = (gate + 1) * H
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

        # Adjust MLP input (slice columns of first Linear)
        lin0_old: nn.Linear = old.mlp[0]
        lin0_new: nn.Linear = new_model.mlp[0]
        lin0_new.bias.copy_(lin0_old.bias.data)
        lin0_new.weight.copy_(lin0_old.weight.data[:, keep_idx])

        # Copy remaining MLP layers
        for i in range(2, len(old.mlp)):
            if isinstance(old.mlp[i], nn.Linear) and isinstance(new_model.mlp[i], nn.Linear):
                new_model.mlp[i].weight.copy_(old.mlp[i].weight.data)
                new_model.mlp[i].bias.copy_(old.mlp[i].bias.data)

    return new_model


def prune_mlp_hidden(model: LSTMMLP, keep_idx: np.ndarray) -> LSTMMLP:
    """Prune neurons in the MLP hidden layer (first Linear).
    keep_idx selects which hidden neurons to keep (row indices of first Linear, and column indices of second Linear).
    """
    old = model
    lstm = old.lstm
    in_f = lstm.input_size
    Hn = lstm.hidden_size
    old_lin0: nn.Linear = old.mlp[0]
    old_lin1: nn.Linear = old.mlp[3]  # second Linear (to 1)
    keep_m = int(len(keep_idx))

    # Build new with reduced mlp_hidden
    new_model = LSTMMLP(in_features=in_f, hidden_size=Hn, mlp_hidden=keep_m, num_layers=lstm.num_layers, dropout=0.05)

    with torch.no_grad():
        # LSTM weights identical
        new_model.lstm.load_state_dict(old.lstm.state_dict())

        # First Linear: rows select neurons, columns unchanged (hidden features)
        lin0_new: nn.Linear = new_model.mlp[0]
        lin0_new.bias.copy_(old_lin0.bias.data[keep_idx])
        lin0_new.weight.copy_(old_lin0.weight.data[keep_idx, :])

        # Copy ReLU, Dropout are stateless

        # Second Linear: select matching columns
        lin1_new: nn.Linear = new_model.mlp[3]
        lin1_new.bias.copy_(old_lin1.bias.data)
        lin1_new.weight.copy_(old_lin1.weight.data[:, keep_idx])

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
                for xb_mb, yb_mb in zip(torch.split(xb, microbatch, dim=0), torch.split(yb, microbatch, dim=0)):
                    with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=scaler is not None):
                        pred_mb = model(xb_mb)
                        loss_mb = loss_fn(pred_mb, yb_mb)
                    if scaler is not None:
                        scaler.scale(loss_mb).backward()
                    else:
                        loss_mb.backward()
                if scaler is not None:
                    if max_grad_norm:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                total += loss_mb.item() * xb.size(0)
                count += xb.size(0)
            else:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=scaler is not None):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if max_grad_norm:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                total += loss.item() * xb.size(0)
                count += xb.size(0)
        val = eval_model(model, val_loader, device, microbatch=eval_microbatch or microbatch, progress=True, desc=f"val[{ep}/{epochs}]")
        if val['rmse'] is not None and val['rmse'] < best['rmse']:
            best = val
        print(f"[finetune] epoch {ep}/{epochs} train_loss={total/max(count,1):.6f} | val_rmse={val['rmse']:.6f} val_mae={val['mae']:.6f}")
    return best


def main():
    ap = argparse.ArgumentParser(description='Advanced structured pruning for SOC LSTM (1.5.0.0)')
    ap.add_argument('--ckpt', type=str, required=True, help='Path to baseline checkpoint .pt from SOC training')
    ap.add_argument('--yaml', type=str, required=True, help='Path to training YAML (for data paths, features, etc.)')
    ap.add_argument('--prune-ratio', type=float, default=0.3, help='Fraction of LSTM hidden units to remove (0..1)')
    ap.add_argument('--saliency', type=str, default='unit', choices=['unit','gate-aware'], help='Saliency mode for LSTM unit pruning')
    ap.add_argument('--combine', type=str, default='sum', choices=['sum','mean','max','min'], help='Combine gate scores in gate-aware mode')
    ap.add_argument('--mlp-prune-ratio', type=float, default=0.0, help='Fraction of MLP hidden neurons to remove (0..1). 0 disables MLP pruning.')
    ap.add_argument('--finetune-epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--out-dir', type=str, default='/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0')
    ap.add_argument('--export-stateful', action='store_true', help='Also export stateful ONNX')
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--train-batch', type=int, default=None)
    ap.add_argument('--val-batch', type=int, default=None)
    ap.add_argument('--microbatch', type=int, default=None)
    ap.add_argument('--eval-microbatch', type=int, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu'))
    print(f"[config] device={device.type} | prune_ratio={args.prune_ratio} | finetune_epochs={args.finetune_epochs} | saliency={args.saliency} | mlp_prune_ratio={args.mlp_prune_ratio}")

    # Load baseline
    state = torch.load(args.ckpt, map_location='cpu')
    with open(args.yaml, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)

    features: List[str] = state.get('features') or cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    assert num_layers == 1, 'Current pruner supports num_layers==1 only'
    chunk = int(state.get('chunk') or cfg['training']['seq_chunk_size'])

    # Build baseline model
    base = LSTMMLP(in_features=len(features), hidden_size=hidden_size, mlp_hidden=mlp_hidden, num_layers=num_layers)
    base.load_state_dict(state['model_state_dict'])
    base.to(device)
    print(f"[model] features={len(features)} | hidden_size={hidden_size} | mlp_hidden={mlp_hidden} | chunk={chunk}")

    # Data
    default_train_bs = int(cfg['training'].get('batch_size', 256))
    default_val_bs = min(128, default_train_bs)
    train_bs = args.train_batch or default_train_bs
    val_bs = args.val_batch or default_val_bs
    pin_memory = (device.type == 'cuda')
    scaler, train_loader, val_loader = create_dataloaders(cfg, features, chunk, train_batch_size=train_bs, val_batch_size=val_bs, pin_memory=pin_memory)
    try:
        print(f"[data] train_bs={train_bs} | val_bs={val_bs} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")
    except Exception:
        pass

    # Baseline eval
    base_val = eval_model(base, val_loader, device, microbatch=args.eval_microbatch or args.microbatch, progress=True, desc="baseline-val")
    print(f"Baseline val: rmse={base_val['rmse']:.6f} mae={base_val['mae']:.6f} r2={base_val['r2']:.6f}")

    # LSTM saliency and prune
    scores = lstm_unit_saliency(base.lstm, mode=args.saliency, combine=args.combine)
    H = base.lstm.hidden_size
    remove = int(round(H * args.prune_ratio))
    keep = max(1, H - remove)
    keep_idx = np.argsort(scores)[-keep:]
    keep_idx = np.sort(keep_idx)
    print(f"[prune-LSTM] hidden_size {H} -> {keep} (remove {remove}) | mode={args.saliency}")

    pruned = prune_lstm_units(base, keep_idx)
    pruned.to(device)

    # Optional MLP head pruning
    mlp_info = None
    if args.mlp_prune_ratio and args.mlp_prune_ratio > 0:
        old_m = pruned.mlp[0].out_features
        remove_m = int(round(old_m * args.mlp_prune_ratio))
        keep_m = max(1, old_m - remove_m)
        # Saliency per neuron in first Linear: L2 of row weights
        with torch.no_grad():
            w0 = pruned.mlp[0].weight.detach().cpu().numpy()
            s_m = np.linalg.norm(w0, axis=1)  # [mlp_hidden]
            keep_idx_m = np.argsort(s_m)[-keep_m:]
            keep_idx_m = np.sort(keep_idx_m)
        print(f"[prune-MLP] mlp_hidden {old_m} -> {keep_m} (remove {remove_m})")
        pruned = prune_mlp_hidden(pruned, keep_idx_m)
        pruned.to(device)
        mlp_info = {"old": old_m, "new": keep_m}

    # Finetune
    print(f"[finetune] epochs={args.finetune_epochs} lr={args.lr} microbatch={args.microbatch} eval_microbatch={args.eval_microbatch}")
    best_after = train_finetune(pruned, train_loader, val_loader, device, epochs=args.finetune_epochs, lr=args.lr, microbatch=args.microbatch, eval_microbatch=args.eval_microbatch)

    # Out dir
    ts = time.strftime('%Y%m%d_%H%M%S')
    tag = f"prune_{int(args.prune_ratio*100)}pct_{args.saliency}"
    if mlp_info:
        tag += f"_mlp{mlp_info['new']}"
    out_dir = os.path.join(args.out-dir if hasattr(args,'out-dir') else args.out_dir, f"{tag}_{ts}")
    # Fix typo if any
    if not isinstance(out_dir, str):
        out_dir = os.path.join(args.out_dir, f"{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, f"soc_pruned_hidden{pruned.lstm.hidden_size}.pt")
    torch.save({
        'epoch': state.get('epoch', -1),
        'model_state_dict': pruned.state_dict(),
        'config': cfg,
        'features': features,
        'chunk': chunk,
        'pruned': True,
        'keep_indices_lstm': keep_idx.tolist(),
        'mlp_prune': mlp_info,
        'baseline_ckpt': args.ckpt,
        'saliency_mode': args.saliency,
        'combine': args.combine,
    }, ckpt_path)

    # Export ONNX (dense)
    dummy = torch.zeros((1, chunk, len(features)), dtype=torch.float32).to(device)
    onnx_path = os.path.join(out_dir, f"soc_pruned_hidden{pruned.lstm.hidden_size}.onnx")
    print("[export] Exporting dense ONNX...")
    torch.onnx.export(
        pruned.eval(),
        dummy,
        onnx_path,
        input_names=['input_seq'],
        output_names=['soc'],
        dynamic_axes={'input_seq': {0: 'batch'}, 'soc': {0: 'batch'}},
        opset_version=17,
    )
    try:
        print(f"[export] ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    except Exception:
        pass

    if args.export_stateful:
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
        onnx_path_stateful = os.path.join(out_dir, f"soc_pruned_hidden{hdim}_stateful.onnx")
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
            print(f"[export] Stateful ONNX saved: {onnx_path_stateful} ({os.path.getsize(onnx_path_stateful)/1024:.1f} KB)")
        except Exception:
            pass

    # Manifest
    manifest = {
        'baseline': {
            'hidden_size': hidden_size,
            'val': base_val,
        },
        'pruned': {
            'hidden_size': int(pruned.lstm.hidden_size),
            'val_after_finetune': best_after,
            'keep_indices_lstm': keep_idx.tolist(),
            'mlp_prune': mlp_info,
            'checkpoint': ckpt_path,
            'onnx': onnx_path,
        },
        'prune_ratio_lstm': args.prune_ratio,
        'saliency_mode': args.saliency,
        'combine': args.combine,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Pruned model saved to: {out_dir}")


if __name__ == '__main__':
    main()
