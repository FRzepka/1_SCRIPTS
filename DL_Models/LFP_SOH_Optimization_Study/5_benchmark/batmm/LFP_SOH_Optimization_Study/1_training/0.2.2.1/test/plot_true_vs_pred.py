#!/usr/bin/env python3
"""TCN inference and plots for SOH (train/val/test cells)."""
import os
import re
import time
import math
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import load as joblib_load


def expand_env_with_defaults(val: str) -> str:
    pattern = re.compile(r"\$\{([^:}]+):-([^}]+)\}")
    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, '') else default
    if not isinstance(val, str):
        return val
    return os.path.expandvars(pattern.sub(repl, val))


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location('train_soh_module', str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def find_latest_run(out_root: Path) -> Path:
    runs = [p for p in out_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f'No run folders under {out_root}')
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def find_best_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'Missing checkpoints dir: {ckpt_dir}')
    best = list(ckpt_dir.glob('best_epoch*_rmse*.pt'))
    if best:
        def _rmse(path: Path) -> float:
            match = re.search(r'rmse([0-9]+(?:\.[0-9]+)?)', path.name)
            return float(match.group(1)) if match else float('inf')
        best.sort(key=_rmse)
        return best[0]
    final = ckpt_dir / 'final_model.pt'
    if final.exists():
        return final
    any_ckpt = sorted(ckpt_dir.glob('*.pt'))
    if not any_ckpt:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    return any_ckpt[0]


def load_cell_dataframe(data_root: Path, cell: str) -> pd.DataFrame:
    path = data_root / f"df_FE_{cell.split('_')[-1]}.parquet"
    if not path.exists():
        path = data_root / f"df_FE_{cell}.parquet"
    if not path.exists():
        cid = cell[-3:]
        alt = data_root / f"df_FE_C{cid}.parquet"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f'Could not locate parquet for cell {cell} in {data_root}')
    return pd.read_parquet(path)


def plot_timeseries(y_true, y_pred, out_path: Path, title: str, max_points: int = 50000):
    if len(y_true) > max_points:
        step = max(1, len(y_true) // max_points)
        y_true = y_true[::step]
        y_pred = y_pred[::step]
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='SOH true', linewidth=0.8, alpha=0.7)
    plt.plot(y_pred, label='SOH pred', linewidth=0.8, alpha=0.7)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('SOH')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(y_true, y_pred, out_path: Path, title: str, max_points: int = 50000):
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=4, alpha=0.3)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.title(title)
    plt.xlabel('SOH true')
    plt.ylabel('SOH pred')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=str(Path(__file__).resolve().parents[1] / 'config' / 'train_soh.yaml'))
    ap.add_argument('--run-dir', type=str, default=None)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--group', type=str, default='both', choices=['train', 'val', 'test', 'both', 'all'])
    ap.add_argument('--cells', type=str, default=None, help='Comma list of cells (overrides group)')
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out-dir', type=str, default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_root = Path(expand_env_with_defaults(cfg['paths']['data_root']))
    out_root = Path(expand_env_with_defaults(cfg['paths']['out_root']))

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(out_root)
    ckpt_path = Path(args.ckpt) if args.ckpt else find_best_ckpt(run_dir)
    scaler_path = run_dir / 'scaler_robust.joblib'
    if not scaler_path.exists():
        raise FileNotFoundError(f'Missing scaler: {scaler_path}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_py = Path(__file__).resolve().parents[1] / 'scripts' / 'train_soh.py'
    train_mod = load_train_module(train_py)

    base_features = cfg['model']['features']
    target = cfg.get('training', {}).get('target', 'SOH')
    sampling_cfg = cfg.get('sampling', {})
    if hasattr(train_mod, 'expand_features_for_sampling'):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    kernel_size = int(cfg['model'].get('kernel_size', 3))
    dilations = cfg['model'].get('dilations')
    if not dilations:
        num_layers = int(cfg['model'].get('num_layers', 4))
        dilations = [2 ** i for i in range(num_layers)]
    dropout = float(cfg['model'].get('dropout', 0.05))

    model = train_mod.CausalTCN_SOH(
        in_features=len(features),
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    model.load_state_dict(state_dict)

    scaler = joblib_load(scaler_path)

    if args.cells:
        cells = [c.strip() for c in args.cells.split(',') if c.strip()]
        groups = {}
        train_cells = set(cfg['cells'].get('train', []))
        val_cells = set(cfg['cells'].get('val', []))
        test_cells = set(cfg['cells'].get('test', []))
        for cell in cells:
            if cell in train_cells:
                group = 'train'
            elif cell in val_cells:
                group = 'val'
            elif cell in test_cells:
                group = 'test'
            else:
                group = 'custom'
            groups.setdefault(group, []).append(cell)
    else:
        groups = {}
        if args.group in ('train', 'all'):
            groups['train'] = cfg['cells'].get('train', [])
        if args.group in ('val', 'both', 'all'):
            groups['val'] = cfg['cells'].get('val', [])
        if args.group in ('test', 'both', 'all'):
            groups['test'] = cfg['cells'].get('test', [])

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / f'PRED_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    model.eval()
    with torch.inference_mode():
        for group_name, cell_list in groups.items():
            for cell in cell_list:
                print(f'Running {group_name} cell {cell}...')
                df = load_cell_dataframe(data_root, cell)
                if hasattr(train_mod, 'maybe_aggregate_hourly'):
                    df = train_mod.maybe_aggregate_hourly(df, base_features, target, sampling_cfg)

                keep_cols = features + [target]
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols)
                if args.max_samples:
                    df = df.iloc[:args.max_samples]

                X = df[features].to_numpy(dtype=np.float32)
                X_scaled = scaler.transform(X).astype(np.float32)
                y_true = df[target].to_numpy(dtype=np.float32)

                xb = torch.from_numpy(X_scaled).unsqueeze(0).to(device)
                y_pred = model(xb).squeeze(0).detach().cpu().numpy()

                if len(y_pred) != len(y_true):
                    n = min(len(y_pred), len(y_true))
                    y_pred = y_pred[:n]
                    y_true = y_true[:n]

                rmse = math.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')

                metrics_rows.append({
                    'group': group_name,
                    'cell': cell,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'samples': len(y_true),
                    'ckpt': str(ckpt_path),
                })

                title = f'{cell} ({group_name}) | RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}'
                plot_timeseries(y_true, y_pred, out_dir / f'{cell}_{group_name}_timeseries.png', title)
                plot_scatter(y_true, y_pred, out_dir / f'{cell}_{group_name}_scatter.png', title)

    metrics_path = out_dir / 'metrics.csv'
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f'Saved metrics: {metrics_path}')
    print(f'Plots: {out_dir}')


if __name__ == '__main__':
    main()
