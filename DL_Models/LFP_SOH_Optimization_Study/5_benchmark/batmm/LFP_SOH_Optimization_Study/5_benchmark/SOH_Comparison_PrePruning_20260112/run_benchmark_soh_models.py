#!/usr/bin/env python3
"""
Pre-pruning SOH model benchmark: LSTM vs TCN vs GRU vs CNN.
Generates per-cell comparison plots and summary tables.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import typing as T

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODELS = [
    {
        "key": "LSTM_0.1.2.3",
        "label": "LSTM 0.1.2.3",
        "family": "LSTM",
        "config": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.3/config/train_soh.yaml",
        "train_py": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.3/scripts/train_soh.py",
    },
    {
        "key": "TCN_0.2.2.1",
        "label": "TCN 0.2.2.1",
        "family": "TCN",
        "config": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.2.2.1/config/train_soh.yaml",
        "train_py": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.2.2.1/scripts/train_soh.py",
    },
    {
        "key": "GRU_0.3.1.1",
        "label": "GRU 0.3.1.1",
        "family": "GRU",
        "config": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.3.1.1/config/train_soh.yaml",
        "train_py": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.3.1.1/scripts/train_soh.py",
    },
    {
        "key": "CNN_0.4.1.1",
        "label": "CNN 0.4.1.1",
        "family": "CNN",
        "config": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.4.1.1/config/train_soh.yaml",
        "train_py": "DL_Models/LFP_SOH_Optimization_Study/1_training/0.4.1.1/scripts/train_soh.py",
    },
]


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location('train_soh_module', str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def expand_env_with_defaults(val: str) -> str:
    if not isinstance(val, str):
        return val
    out = val
    while '${' in out and ':-' in out:
        start = out.find('${')
        end = out.find('}', start)
        if start == -1 or end == -1:
            break
        expr = out[start + 2:end]
        if ':-' in expr:
            var, default = expr.split(':-', 1)
            env_val = os.getenv(var)
            repl = env_val if env_val not in (None, '') else default
        else:
            repl = os.getenv(expr, '')
        out = out[:start] + repl + out[end + 1:]
    return os.path.expandvars(out)


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
            name = path.name
            if 'rmse' not in name:
                return float('inf')
            try:
                return float(name.split('rmse')[-1].replace('.pt', ''))
            except ValueError:
                return float('inf')
        best.sort(key=_rmse)
        return best[0]
    final = ckpt_dir / 'final_model.pt'
    if final.exists():
        return final
    any_ckpt = sorted(ckpt_dir.glob('*.pt'))
    if not any_ckpt:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    return any_ckpt[0]


def expand_features_for_sampling(base_features: T.List[str], sampling_cfg: dict) -> T.List[str]:
    if not sampling_cfg or not sampling_cfg.get('enabled', False):
        return list(base_features)
    feature_aggs = sampling_cfg.get('feature_aggs', ['mean'])
    if not feature_aggs:
        feature_aggs = ['mean']
    return [f'{feat}_{agg}' for feat in base_features for agg in feature_aggs]


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


def aggregate_hourly(df: pd.DataFrame, base_features: T.List[str], target: str, sampling_cfg: dict) -> pd.DataFrame:
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
    target_agg = sampling_cfg.get('target_agg', 'last')

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')
    return {"rmse": rmse, "mae": mae, "r2": r2}


def predict_stateful_rnn(model, X: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    state = None
    with torch.inference_mode():
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            xb = torch.from_numpy(X[start:end]).unsqueeze(0).to(device)
            if state is None:
                y_seq, state = model(xb, state=None, return_state=True)
            else:
                y_seq, state = model(xb, state=state, return_state=True)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_causal_buffer(model, X: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    rf = int(getattr(model, 'receptive_field', 1))
    context = None
    with torch.inference_mode():
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            xb = X[start:end]
            if rf > 1:
                if context is not None:
                    xb_in = np.concatenate([context, xb], axis=0)
                else:
                    xb_in = xb
                context = xb_in[-(rf - 1):].copy()
            else:
                xb_in = xb
            xb_t = torch.from_numpy(xb_in).unsqueeze(0).to(device)
            y_seq = model(xb_t)
            y_np = y_seq.squeeze(0).detach().cpu().numpy()
            if rf > 1:
                y_np = y_np[-len(xb):]
            preds.append(y_np)
    return np.concatenate(preds, axis=0)


def build_model(cfg: dict, train_mod, in_features: int) -> torch.nn.Module:
    mtype = str(cfg['model'].get('type', '')).lower()
    if 'lstm' in mtype:
        return train_mod.SOH_LSTM_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg['model'].get('embed_size', 96)),
            hidden_size=int(cfg['model']['hidden_size']),
            mlp_hidden=int(cfg['model']['mlp_hidden']),
            num_layers=int(cfg['model'].get('num_layers', 2)),
            res_blocks=int(cfg['model'].get('res_blocks', 2)),
            bidirectional=bool(cfg['model'].get('bidirectional', False)),
            dropout=float(cfg['model'].get('dropout', 0.15)),
        )
    if 'gru' in mtype:
        return train_mod.SOH_GRU_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg['model'].get('embed_size', 96)),
            hidden_size=int(cfg['model']['hidden_size']),
            mlp_hidden=int(cfg['model']['mlp_hidden']),
            num_layers=int(cfg['model'].get('num_layers', 2)),
            res_blocks=int(cfg['model'].get('res_blocks', 2)),
            bidirectional=bool(cfg['model'].get('bidirectional', False)),
            dropout=float(cfg['model'].get('dropout', 0.15)),
        )
    if 'tcn' in mtype:
        kernel_size = int(cfg['model'].get('kernel_size', 3))
        dilations = cfg['model'].get('dilations')
        if not dilations:
            num_layers = int(cfg['model'].get('num_layers', 4))
            dilations = [2 ** i for i in range(num_layers)]
        return train_mod.CausalTCN_SOH(
            in_features=in_features,
            hidden_size=int(cfg['model']['hidden_size']),
            mlp_hidden=int(cfg['model']['mlp_hidden']),
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=float(cfg['model'].get('dropout', 0.05)),
        )
    if 'cnn' in mtype:
        return train_mod.SOH_CNN_Seq2Seq(
            in_features=in_features,
            hidden_size=int(cfg['model'].get('hidden_size', 128)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 96)),
            kernel_size=int(cfg['model'].get('kernel_size', 5)),
            dilations=cfg['model'].get('dilations'),
            num_blocks=int(cfg['model'].get('num_blocks', 4)),
            dropout=float(cfg['model'].get('dropout', 0.15)),
        )
    raise ValueError(f'Unsupported model type: {cfg["model"].get("type")}')


def plot_cell_comparison(cell: str, y_true: np.ndarray, preds: dict, out_path: Path, max_points: int = 20000):
    if len(y_true) > max_points:
        step = max(1, len(y_true) // max_points)
        y_true = y_true[::step]
        preds = {k: v[::step] for k, v in preds.items()}
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='SOH true', color='black', linewidth=1.2, alpha=0.8)
    for label, y_pred in preds.items():
        plt.plot(y_pred, label=label, linewidth=0.9, alpha=0.75)
    plt.title(f'SOH Prediction Comparison - {cell}')
    plt.xlabel('Sample index (hourly)')
    plt.ylabel('SOH')
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_bars(summary_df: pd.DataFrame, out_path: Path, metric: str, ylabel: str):
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df['model'], summary_df[metric], color='#3b82f6', alpha=0.85)
    plt.xticks(rotation=25, ha='right')
    plt.ylabel(ylabel)
    plt.title(f'Test {ylabel} by Model')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_param_bars(summary_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df['model'], summary_df['params'], color='#10b981', alpha=0.85)
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Parameters')
    plt.title('Parameter Count by Model')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--cells', type=str, default=None, help='Comma list of test cells')
    args = ap.parse_args()

    ts = time.strftime('%Y%m%d_%H%M%S')
    default_root = Path(__file__).resolve().parent / 'results_base'
    out_dir = Path(args.out_dir) if args.out_dir else default_root / f'RESULTS_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_root = Path(__file__).resolve()
    for _ in range(8):
        if (base_root / 'DL_Models').exists():
            break
        base_root = base_root.parent

    model_cfgs = []
    for spec in MODELS:
        cfg_path = (base_root / spec['config']).resolve()
        train_py = (base_root / spec['train_py']).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        if not train_py.exists():
            raise FileNotFoundError(train_py)
        import yaml
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        model_cfgs.append((spec, cfg, train_py))

    ref_spec, ref_cfg, _ = model_cfgs[0]
    base_features = ref_cfg['model']['features']
    target = ref_cfg.get('training', {}).get('target', 'SOH')
    sampling_cfg = dict(ref_cfg.get('sampling', {}))
    sampling_cfg['target_agg'] = 'last'
    features = expand_features_for_sampling(base_features, sampling_cfg)

    if args.cells:
        test_cells = [c.strip() for c in args.cells.split(',') if c.strip()]
    else:
        test_cells = ref_cfg['cells'].get('test', [])

    data_root = expand_env_with_defaults(ref_cfg['paths']['data_root'])

    cell_data = {}
    for cell in test_cells:
        df = load_cell_parquet(data_root, cell)
        df = aggregate_hourly(df, base_features, target, sampling_cfg)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)
        X = df[features].to_numpy(dtype=np.float32)
        y = df[target].to_numpy(dtype=np.float32)
        cell_data[cell] = (X, y)

    metrics_rows = []
    summary_rows = []
    per_cell_preds = {cell: {} for cell in test_cells}

    for spec, cfg, train_py in model_cfgs:
        train_mod = load_train_module(train_py)
        out_root = Path(expand_env_with_defaults(cfg['paths']['out_root'])).resolve()
        run_dir = find_latest_run(out_root)
        ckpt_path = find_best_ckpt(run_dir)
        scaler_path = run_dir / 'scaler_robust.joblib'
        if not scaler_path.exists():
            raise FileNotFoundError(scaler_path)

        model = build_model(cfg, train_mod, in_features=len(features)).to(device)
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get('model_state_dict', state)
        model.load_state_dict(state_dict)

        params = sum(p.numel() for p in model.parameters())
        param_size_kb = (params * 4) / 1024
        ckpt_size_kb = ckpt_path.stat().st_size / 1024

        scaler = joblib_load(scaler_path)
        chunk = int(cfg['training'].get('seq_chunk_size', 168))

        model_metrics = []
        for cell, (X, y_true) in cell_data.items():
            X_scaled = scaler.transform(X).astype(np.float32)
            if spec['family'] in ('LSTM', 'GRU'):
                y_pred = predict_stateful_rnn(model, X_scaled, chunk, device)
            else:
                y_pred = predict_causal_buffer(model, X_scaled, chunk, device)
            per_cell_preds[cell][spec['label']] = y_pred
            metrics = compute_metrics(y_true, y_pred)
            metrics_rows.append({
                "model": spec['label'],
                "family": spec['family'],
                "cell": cell,
                "group": "test",
                "samples": len(y_true),
                "rmse": metrics['rmse'],
                "mae": metrics['mae'],
                "r2": metrics['r2'],
                "params": params,
                "param_size_kb": param_size_kb,
                "ckpt_size_kb": ckpt_size_kb,
                "ckpt": str(ckpt_path),
            })
            model_metrics.append(metrics)

        mean_rmse = float(np.mean([m['rmse'] for m in model_metrics]))
        mean_mae = float(np.mean([m['mae'] for m in model_metrics]))
        mean_r2 = float(np.mean([m['r2'] for m in model_metrics]))
        summary_rows.append({
            "model": spec['label'],
            "family": spec['family'],
            "params": params,
            "param_size_kb": param_size_kb,
            "ckpt_size_kb": ckpt_size_kb,
            "test_rmse": mean_rmse,
            "test_mae": mean_mae,
            "test_r2": mean_r2,
            "ckpt": str(ckpt_path),
        })

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values('test_mae')

    metrics_df.to_csv(out_dir / 'metrics_by_cell.csv', index=False)
    summary_df.to_csv(out_dir / 'metrics_summary.csv', index=False)

    for cell, preds in per_cell_preds.items():
        _, y_true = cell_data[cell]
        plot_cell_comparison(cell, y_true, preds, out_dir / 'plots' / f'{cell}_comparison.png')

    plot_metric_bars(summary_df, out_dir / 'plots' / 'test_mae_bar.png', 'test_mae', 'MAE')
    plot_metric_bars(summary_df, out_dir / 'plots' / 'test_rmse_bar.png', 'test_rmse', 'RMSE')
    plot_metric_bars(summary_df, out_dir / 'plots' / 'test_r2_bar.png', 'test_r2', 'R²')
    plot_param_bars(summary_df, out_dir / 'plots' / 'params_bar.png')

    report = {
        "generated_at": ts,
        "device": str(device),
        "models": [m[0]["label"] for m in model_cfgs],
        "cells": test_cells,
        "notes": "All plots/metrics use hourly aggregation with target=last-of-hour for comparability.",
    }
    with open(out_dir / 'benchmark_meta.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Outputs saved to: {out_dir}')


if __name__ == '__main__':
    main()
