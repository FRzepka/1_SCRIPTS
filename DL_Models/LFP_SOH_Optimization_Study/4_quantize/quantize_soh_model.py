#!/usr/bin/env python3
"""Quantize a trained SOH model (dynamic or static FX quantization)."""
import argparse
import json
import os
import re
from pathlib import Path
import importlib.util

import torch
import yaml
import inspect
import numpy as np
import pandas as pd
from joblib import load as joblib_load
from torch.ao.quantization import QConfig, get_default_qconfig, QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from typing import Optional


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


def find_best_checkpoint(ckpt_dir: Path) -> Path:
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
        raise FileNotFoundError(f'No checkpoint with rmse found in {ckpt_dir}')
    return best


def build_model(cfg: dict, train_mod):
    base_features = cfg['model']['features']
    sampling_cfg = cfg.get('sampling', {})
    if hasattr(train_mod, 'expand_features_for_sampling'):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    mtype = str(cfg['model'].get('type', '')).lower()
    if 'lstm' in mtype:
        cls = train_mod.SOH_LSTM_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg['model'].get('embed_size', 96)),
            hidden_size=int(cfg['model'].get('hidden_size', 160)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 128)),
            num_layers=int(cfg['model'].get('num_layers', 2)),
            res_blocks=int(cfg['model'].get('res_blocks', 0)),
            bidirectional=bool(cfg['model'].get('bidirectional', False)),
            dropout=float(cfg['model'].get('dropout', 0.1)),
        )
        qtypes = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    elif 'gru' in mtype:
        cls = train_mod.SOH_GRU_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg['model'].get('embed_size', 96)),
            hidden_size=int(cfg['model'].get('hidden_size', 160)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 128)),
            num_layers=int(cfg['model'].get('num_layers', 2)),
            res_blocks=int(cfg['model'].get('res_blocks', 0)),
            bidirectional=bool(cfg['model'].get('bidirectional', False)),
            dropout=float(cfg['model'].get('dropout', 0.1)),
        )
        qtypes = {torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM}
    elif 'tcn' in mtype:
        cls = train_mod.CausalTCN_SOH
        dilations = cfg['model'].get('dilations') or [1, 2, 4, 8]
        model = cls(
            in_features=len(features),
            hidden_size=int(cfg['model'].get('hidden_size', 128)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 96)),
            kernel_size=int(cfg['model'].get('kernel_size', 3)),
            dilations=[int(d) for d in dilations],
            dropout=float(cfg['model'].get('dropout', 0.1)),
        )
        qtypes = {torch.nn.Linear}
    elif 'cnn' in mtype:
        cls = train_mod.SOH_CNN_Seq2Seq
        dilations = cfg['model'].get('dilations')
        kwargs = dict(
            in_features=len(features),
            hidden_size=int(cfg['model'].get('hidden_size', 128)),
            mlp_hidden=int(cfg['model'].get('mlp_hidden', 96)),
            kernel_size=int(cfg['model'].get('kernel_size', 5)),
            dilations=[int(d) for d in dilations] if dilations is not None else None,
            num_blocks=int(cfg['model'].get('num_blocks', 4)),
            dropout=float(cfg['model'].get('dropout', 0.1)),
        )
        if 'output_kernel_size' in inspect.signature(cls).parameters:
            kwargs['output_kernel_size'] = int(cfg['model'].get('output_kernel_size', 1))
        model = cls(**kwargs)
        qtypes = {torch.nn.Linear}
    else:
        raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")

    return model, qtypes


def expand_features_for_sampling(base_features, sampling_cfg):
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


def aggregate_hourly(df: pd.DataFrame, base_features, target: str, sampling_cfg: dict) -> pd.DataFrame:
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


def make_static_qconfig():
    return QConfig(
        activation=HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )


def calibration_batches(cfg: dict, base_features, features, target: str, scaler, max_batches: int = 64):
    data_root = expand_env_with_defaults(cfg['paths']['data_root'])
    seq_len = int(cfg['training'].get('seq_chunk_size', 168))
    sampling_cfg = cfg.get('sampling', {})
    cells = list(cfg.get('cells', {}).get('train', []))
    if not cells:
        raise ValueError('No training cells for calibration')

    batches = 0
    for cell in cells:
        df = load_cell_parquet(data_root, cell)
        df = aggregate_hourly(df, base_features, target, sampling_cfg)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)
        if df.empty:
            continue
        X = df[features].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X).astype(np.float32)
        total = len(X_scaled)
        if total < seq_len:
            continue
        for start in range(0, total - seq_len + 1, seq_len):
            batch = torch.from_numpy(X_scaled[start:start + seq_len]).unsqueeze(0)
            yield batch
            batches += 1
            if batches >= max_batches:
                return


def run_quantize(model_dir: Path, out_dir: Path, ckpt_path: Optional[Path]):
    config_path = model_dir / 'config' / 'train_soh.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config: {config_path}')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['paths']['data_root'] = expand_env_with_defaults(cfg['paths']['data_root'])
    cfg['paths']['out_root'] = expand_env_with_defaults(cfg['paths']['out_root'])

    train_py = model_dir / 'scripts' / 'train_soh.py'
    train_mod = load_train_module(train_py)

    model, qtypes = build_model(cfg, train_mod)
    model.eval()

    if ckpt_path is None:
        ckpt_dir = model_dir / 'checkpoints'
        if not ckpt_dir.exists():
            raise FileNotFoundError(f'Missing checkpoints dir: {ckpt_dir}')
        ckpt_path = find_best_checkpoint(ckpt_dir)

    state = torch.load(ckpt_path, map_location='cpu')
    state_dict = state.get('model_state_dict', state)
    model.load_state_dict(state_dict)

    base_features = cfg['model']['features']
    sampling_cfg = cfg.get('sampling', {})
    features = expand_features_for_sampling(base_features, sampling_cfg)
    target = cfg.get('training', {}).get('target', 'SOH')
    scaler_path = model_dir / 'scaler_robust.joblib'
    scaler = joblib_load(scaler_path) if scaler_path.exists() else None

    mtype = str(cfg['model'].get('type', '')).lower()
    quant_scope = 'full'
    if 'tcn' in mtype or 'cnn' in mtype:
        torch.backends.quantized.engine = 'fbgemm'
        qconfig = make_static_qconfig()
        if 'tcn' in mtype:
            quant_scope = 'head_only'
            qconfig_mapping = QConfigMapping().set_global(None).set_module_name_regex(r'^head\\.', qconfig)
        else:
            qconfig_mapping = QConfigMapping().set_global(qconfig)
        example_inputs = (torch.zeros(1, int(cfg['training'].get('seq_chunk_size', 168)), len(features)),)
        prepared = prepare_fx(model, qconfig_mapping, example_inputs)
        if scaler is None:
            raise FileNotFoundError(f'Missing scaler for calibration: {scaler_path}')
        for batch in calibration_batches(cfg, base_features, features, target, scaler, max_batches=64):
            prepared(batch)
        qmodel = convert_fx(prepared)
        quant_mode = 'static_fx'
    else:
        qmodel = torch.ao.quantization.quantize_dynamic(model, qtypes, dtype=torch.qint8)
        quant_mode = 'dynamic'

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out_dir / 'quantized_state_dict.pt')

    for sub in ['config', 'scripts', 'test']:
        src = model_dir / sub
        dst = out_dir / sub
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.is_file():
                    (dst / item.name).write_bytes(item.read_bytes())

    scaler_src = model_dir / 'scaler_robust.joblib'
    if scaler_src.exists():
        scaler_dst = out_dir / 'scaler_robust.joblib'
        if scaler_dst.resolve() != scaler_src.resolve():
            scaler_dst.write_bytes(scaler_src.read_bytes())

    meta = {
        'model_dir': str(model_dir),
        'checkpoint': str(ckpt_path),
        'model_type': cfg['model'].get('type'),
        'quantized_modules': [t.__name__ for t in sorted(qtypes, key=lambda x: x.__name__)],
        'quant_mode': quant_mode,
        'quant_scope': quant_scope,
    }
    with open(out_dir / 'quantize_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description='Quantize a SOH model (dynamic or static FX).')
    ap.add_argument('--model-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--ckpt', type=str, default=None)
    args = ap.parse_args()

    run_quantize(
        model_dir=Path(args.model_dir),
        out_dir=Path(args.out_dir),
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
    )


if __name__ == '__main__':
    main()
