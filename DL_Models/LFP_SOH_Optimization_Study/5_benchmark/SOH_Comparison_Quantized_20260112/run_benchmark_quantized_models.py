#!/usr/bin/env python3
"""
Benchmark base vs quantized SOH models (LSTM, TCN, GRU, CNN).
Creates per-model multi-panel plots and summary metrics.
"""
import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import time
import typing as T
import inspect

import numpy as np
import pandas as pd
import torch
from torch.ao.quantization import QConfig, get_default_qconfig, QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
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
        "base_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Base/0.1.2.3",
        "quant_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Quantized/0.1.2.3",
    },
    {
        "key": "TCN_0.2.2.1",
        "label": "TCN 0.2.2.1",
        "family": "TCN",
        "base_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/TCN/Base/0.2.2.1",
        "quant_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/TCN/Quantized/0.2.2.1",
    },
    {
        "key": "GRU_0.3.1.1",
        "label": "GRU 0.3.1.1",
        "family": "GRU",
        "base_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/GRU/Base/0.3.1.1",
        "quant_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/GRU/Quantized/0.3.1.1",
    },
    {
        "key": "CNN_0.4.1.1",
        "label": "CNN 0.4.1.1",
        "family": "CNN",
        "base_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/CNN/Base/0.4.1.1",
        "quant_dir": "DL_Models/LFP_SOH_Optimization_Study/2_models/CNN/Quantized/0.4.1.1",
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


def find_best_ckpt(ckpt_dir: Path) -> Path:
    best = None
    best_rmse = None
    for path in ckpt_dir.glob('*.pt'):
        match = re.search(r'rmse([0-9]+(?:\\.[0-9]+)?)', path.name)
        if not match:
            continue
        rmse = float(match.group(1))
        if best is None or rmse < best_rmse:
            best = path
            best_rmse = rmse
    if best is None:
        any_ckpt = sorted(ckpt_dir.glob('*.pt'))
        if not any_ckpt:
            raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
        return any_ckpt[0]
    return best


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
        cls = train_mod.SOH_CNN_Seq2Seq
        kwargs = dict(
            in_features=in_features,
            hidden_size=int(cfg['model'].get('hidden_size', 128)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 96)),
            kernel_size=int(cfg['model'].get('kernel_size', 5)),
            dilations=cfg['model'].get('dilations'),
            num_blocks=int(cfg['model'].get('num_blocks', 4)),
            dropout=float(cfg['model'].get('dropout', 0.15)),
        )
        if 'output_kernel_size' in inspect.signature(cls).parameters:
            kwargs['output_kernel_size'] = int(cfg['model'].get('output_kernel_size', 1))
        return cls(**kwargs)
    raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")


def get_quant_types(mtype: str) -> set:
    mtype = mtype.lower()
    if 'lstm' in mtype or 'gru' in mtype:
        return {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    return {torch.nn.Linear}


def make_static_qconfig():
    return QConfig(
        activation=HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )


def build_quantized_model(cfg: dict, train_mod, state_dict: dict, in_features: int, chunk: int, qstate_path: Path):
    mtype = str(cfg['model'].get('type', '')).lower()
    if 'tcn' in mtype or 'cnn' in mtype:
        torch.backends.quantized.engine = 'fbgemm'
        qconfig = make_static_qconfig()
        if 'tcn' in mtype:
            qconfig_mapping = QConfigMapping().set_global(None).set_module_name_regex(r'^head\\.', qconfig)
        else:
            qconfig_mapping = QConfigMapping().set_global(qconfig)
        float_model = build_model(cfg, train_mod, in_features=in_features).cpu()
        float_model.load_state_dict(state_dict)
        example_inputs = (torch.zeros(1, chunk, in_features),)
        prepared = prepare_fx(float_model, qconfig_mapping, example_inputs)
        quant_model = convert_fx(prepared)
        quant_model.load_state_dict(torch.load(qstate_path, map_location='cpu'))
        if hasattr(float_model, 'receptive_field'):
            quant_model.receptive_field = float_model.receptive_field
        return quant_model

    qtypes = get_quant_types(mtype)
    quant_float = build_model(cfg, train_mod, in_features=in_features).cpu()
    quant_float.load_state_dict(state_dict)
    quant_model = torch.ao.quantization.quantize_dynamic(quant_float, qtypes, dtype=torch.qint8)
    quant_model.load_state_dict(torch.load(qstate_path, map_location='cpu'))
    return quant_model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_size_kb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0


def compute_macs(model: torch.nn.Module, input_shape: T.Tuple[int, int, int]) -> int:
    total_macs = 0
    handles = []

    def add_macs(macs: int):
        nonlocal total_macs
        total_macs += int(macs)

    def conv1d_hook(module, inp, out):
        x = inp[0]
        if x.dim() != 3:
            return
        batch, in_ch, _ = x.shape
        out_ch = out.shape[1]
        out_len = out.shape[2]
        kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        groups = module.groups
        macs = batch * out_ch * out_len * (in_ch // groups) * kernel
        add_macs(macs)

    def linear_hook(module, inp, out):
        in_features = module.in_features
        macs = out.numel() * in_features
        add_macs(macs)

    def rnn_hook(module, inp, out):
        x = inp[0]
        if x.dim() != 3:
            return
        if module.batch_first:
            batch, steps, _ = x.shape
        else:
            steps, batch, _ = x.shape
        hidden = module.hidden_size
        macs = 0
        for weights in module.all_weights:
            w_ih = weights[0]
            gates = w_ih.shape[0] // hidden
            in_size = w_ih.shape[1]
            macs += batch * steps * gates * (in_size * hidden + hidden * hidden)
        add_macs(macs)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            handles.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
            handles.append(m.register_forward_hook(rnn_hook))

    dummy = torch.zeros(input_shape, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        _ = model(dummy)

    for h in handles:
        h.remove()
    return total_macs


def plot_model_panels(label: str, cells: T.List[str], y_true_map: dict, base_map: dict, quant_map: dict,
                      metrics_base: dict, metrics_quant: dict, out_path: Path):
    n = len(cells)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for idx, cell in enumerate(cells):
        ax = axes[idx]
        y_true = y_true_map[cell]
        base_pred = base_map[cell]
        quant_pred = quant_map[cell]
        max_points = 20000
        if len(y_true) > max_points:
            step = max(1, len(y_true) // max_points)
            y_true = y_true[::step]
            base_pred = base_pred[::step]
            quant_pred = quant_pred[::step]
        ax.plot(y_true, label='SOH true', color='black', linewidth=1.0, alpha=0.8)
        ax.plot(base_pred, label='Base', linewidth=0.9, alpha=0.75)
        ax.plot(quant_pred, label='Quantized', linewidth=0.9, alpha=0.75)
        ax.set_ylabel('SOH')
        ax.set_ylim(0.0, 1.0)
        mb = metrics_base[cell]
        mq = metrics_quant[cell]
        ax.set_title(
            f'{label} - {cell} | Base MAE {mb["mae"]:.4f}, RMSE {mb["rmse"]:.4f}, R2 {mb["r2"]:.3f} '
            f'| Quant MAE {mq["mae"]:.4f}, RMSE {mq["rmse"]:.4f}, R2 {mq["r2"]:.3f}'
        )
        ax.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel('Sample index (hourly)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(summary_df: pd.DataFrame, out_path: Path, metric: str, ylabel: str):
    plt.figure(figsize=(9, 4))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width / 2, summary_df[f'base_{metric}'], width, label='Base', color='#2563eb', alpha=0.85)
    plt.bar(x + width / 2, summary_df[f'quant_{metric}'], width, label='Quantized', color='#10b981', alpha=0.85)
    plt.xticks(x, summary_df['model'], rotation=20, ha='right')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} (Base vs Quantized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_single_bars(summary_df: pd.DataFrame, out_path: Path, metric: str, ylabel: str):
    plt.figure(figsize=(9, 4))
    plt.bar(summary_df['model'], summary_df[metric], color='#f59e0b', alpha=0.85)
    plt.xticks(rotation=20, ha='right')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_size_compare_bars(summary_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9, 4))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width / 2, summary_df['param_size_kb'], width, label='Base params', color='#ef4444', alpha=0.85)
    plt.bar(x + width / 2, summary_df['quant_weight_kb'], width, label='Quant weights', color='#14b8a6', alpha=0.85)
    plt.xticks(x, summary_df['model'], rotation=20, ha='right')
    plt.ylabel('Size (KB)')
    plt.title('Base params vs quantized weights')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--cells', type=str, default=None, help='Comma list of test cells (override config)')
    args = ap.parse_args()

    ts = time.strftime('%Y%m%d_%H%M%S')
    default_root = Path(__file__).resolve().parent / 'results_quantized'
    out_dir = Path(args.out_dir) if args.out_dir else default_root / f'RESULTS_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)

    base_root = Path(__file__).resolve()
    for _ in range(8):
        if (base_root / 'DL_Models').exists():
            break
        base_root = base_root.parent

    device = torch.device(args.device)

    metrics_rows = []
    summary_rows = []

    for spec in MODELS:
        base_dir = (base_root / spec['base_dir']).resolve()
        quant_dir = (base_root / spec['quant_dir']).resolve()
        cfg_path = base_dir / 'config' / 'train_soh.yaml'
        train_py = base_dir / 'scripts' / 'train_soh.py'
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        if not train_py.exists():
            raise FileNotFoundError(train_py)

        import yaml
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        train_mod = load_train_module(train_py)
        base_features = cfg['model']['features']
        target = cfg.get('training', {}).get('target', 'SOH')
        sampling_cfg = dict(cfg.get('sampling', {}))
        features = expand_features_for_sampling(base_features, sampling_cfg)

        if args.cells:
            test_cells = [c.strip() for c in args.cells.split(',') if c.strip()]
        else:
            test_cells = cfg['cells'].get('test', [])

        data_root = expand_env_with_defaults(cfg['paths']['data_root'])

        cell_data = {}
        for cell in test_cells:
            df = load_cell_parquet(data_root, cell)
            df = aggregate_hourly(df, base_features, target, sampling_cfg)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)
            X = df[features].to_numpy(dtype=np.float32)
            y = df[target].to_numpy(dtype=np.float32)
            cell_data[cell] = (X, y)

        scaler_path = base_dir / 'scaler_robust.joblib'
        scaler = joblib_load(scaler_path)
        chunk = int(cfg['training'].get('seq_chunk_size', 168))

        ckpt_dir = base_dir / 'checkpoints'
        ckpt_path = find_best_ckpt(ckpt_dir)

        base_model = build_model(cfg, train_mod, in_features=len(features)).to(device)
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get('model_state_dict', state)
        base_model.load_state_dict(state_dict)

        qstate_path = quant_dir / 'quantized_state_dict.pt'
        quant_model = build_quantized_model(
            cfg,
            train_mod,
            state_dict=state_dict,
            in_features=len(features),
            chunk=chunk,
            qstate_path=qstate_path,
        )

        cost_model = build_model(cfg, train_mod, in_features=len(features)).cpu()
        macs_per_seq = compute_macs(cost_model, (1, chunk, len(features)))
        macs_per_step = macs_per_seq / max(1, chunk)

        base_params = count_parameters(base_model)
        base_param_kb = param_size_kb(base_model)
        quant_weight_kb = qstate_path.stat().st_size / 1024.0 if qstate_path.exists() else float('nan')

        y_true_map = {}
        base_map = {}
        quant_map = {}
        metrics_base = {}
        metrics_quant = {}

        for cell, (X, y_true) in cell_data.items():
            X_scaled = scaler.transform(X).astype(np.float32)
            if spec['family'] in ('LSTM', 'GRU'):
                y_base = predict_stateful_rnn(base_model, X_scaled, chunk, device)
                y_quant = predict_stateful_rnn(quant_model, X_scaled, chunk, torch.device('cpu'))
            else:
                y_base = predict_causal_buffer(base_model, X_scaled, chunk, device)
                y_quant = predict_causal_buffer(quant_model, X_scaled, chunk, torch.device('cpu'))

            y_true_map[cell] = y_true
            base_map[cell] = y_base
            quant_map[cell] = y_quant

            mb = compute_metrics(y_true, y_base)
            mq = compute_metrics(y_true, y_quant)
            metrics_base[cell] = mb
            metrics_quant[cell] = mq

            metrics_rows.append({
                "model": spec['label'],
                "family": spec['family'],
                "cell": cell,
                "group": "test",
                "variant": "base",
                "rmse": mb['rmse'],
                "mae": mb['mae'],
                "r2": mb['r2'],
            })
            metrics_rows.append({
                "model": spec['label'],
                "family": spec['family'],
                "cell": cell,
                "group": "test",
                "variant": "quant",
                "rmse": mq['rmse'],
                "mae": mq['mae'],
                "r2": mq['r2'],
            })

        mean_base_mae = float(np.mean([m['mae'] for m in metrics_base.values()]))
        mean_base_rmse = float(np.mean([m['rmse'] for m in metrics_base.values()]))
        mean_base_r2 = float(np.mean([m['r2'] for m in metrics_base.values()]))
        mean_quant_mae = float(np.mean([m['mae'] for m in metrics_quant.values()]))
        mean_quant_rmse = float(np.mean([m['rmse'] for m in metrics_quant.values()]))
        mean_quant_r2 = float(np.mean([m['r2'] for m in metrics_quant.values()]))
        summary_rows.append({
            "model": spec['label'],
            "family": spec['family'],
            "base_mae": mean_base_mae,
            "base_rmse": mean_base_rmse,
            "base_r2": mean_base_r2,
            "quant_mae": mean_quant_mae,
            "quant_rmse": mean_quant_rmse,
            "quant_r2": mean_quant_r2,
            "params": base_params,
            "param_size_kb": base_param_kb,
            "quant_weight_kb": quant_weight_kb,
            "macs_per_seq": macs_per_seq,
            "macs_per_step": macs_per_step,
        })

        plot_model_panels(
            spec['label'],
            test_cells,
            y_true_map,
            base_map,
            quant_map,
            metrics_base,
            metrics_quant,
            out_dir / 'plots' / f'{spec["key"]}_base_vs_quant.png'
        )

        del base_model
        del quant_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(out_dir / 'metrics_by_cell.csv', index=False)
    summary_df.to_csv(out_dir / 'metrics_summary.csv', index=False)

    plot_metric_bars(summary_df, out_dir / 'plots' / 'mae_base_vs_quant.png', 'mae', 'MAE')
    plot_metric_bars(summary_df, out_dir / 'plots' / 'rmse_base_vs_quant.png', 'rmse', 'RMSE')
    plot_metric_bars(summary_df, out_dir / 'plots' / 'r2_base_vs_quant.png', 'r2', 'R2')
    plot_single_bars(summary_df, out_dir / 'plots' / 'macs_per_step.png', 'macs_per_step', 'MACs per step')
    plot_single_bars(summary_df, out_dir / 'plots' / 'param_size_kb.png', 'param_size_kb', 'Base param size (KB)')
    plot_single_bars(summary_df, out_dir / 'plots' / 'quant_weight_kb.png', 'quant_weight_kb', 'Quantized weight size (KB)')
    plot_size_compare_bars(summary_df, out_dir / 'plots' / 'size_base_vs_quant.png')

    report = {
        "generated_at": ts,
        "device": str(device),
        "models": [m["label"] for m in MODELS],
        "notes": "Base vs quantized comparison; LSTM/GRU use dynamic int8; CNN uses static FX int8; TCN uses head-only static FX for stability. MACs are model-only (hardware independent).",
    }
    with open(out_dir / 'benchmark_meta.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Outputs saved to: {out_dir}')


if __name__ == '__main__':
    main()
