#!/usr/bin/env python3
"""
Compare SOH on the same timeframe:
  - SOH true (from parquet)
  - SOH STM32 (from hardware CSV)
  - SOH PC prediction (model 0.1.2.3, as on PC)

Writes into the given HW run folder:
  - stm32_hw_c11_with_pc.csv   (adds soh_pc)
  - stm32_hw_c11_compare.png   (true vs stm32 vs pc)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
from joblib import load as joblib_load


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "DL_Models").exists() and (p / "STM32DC").exists():
            return p
    return here.parent


def _find_latest_hw_run(root: Path, prefix: str = "HW_C11_") -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No HW run folders under {root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(train_py))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _stateful_predict(model, x: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    h = None
    c = None
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            end = min(start + chunk, len(x))
            xb = torch.from_numpy(x[start:end]).unsqueeze(0).to(device)
            if h is None:
                y_seq, (h, c) = model(xb, state=None, return_state=True)
            else:
                y_seq, (h, c) = model(xb, state=(h, c), return_state=True)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _read_parquet_head(parquet: Path, columns: list[str], n_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet)
    chunks = []
    read_rows = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=columns)
        df_rg = table.to_pandas()
        chunks.append(df_rg)
        read_rows += len(df_rg)
        if read_rows >= n_rows:
            break
    df = pd.concat(chunks, ignore_index=True)
    return df.iloc[:n_rows].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-dir", type=str, default="", help="HW run dir (default: latest HW_C11_* under LFP_SOH_Optimization_Study/6_test)")
    ap.add_argument("--cell", type=str, default="C11")
    ap.add_argument("--config", type=str, default="", help="train_soh.yaml (default: 1_training/0.1.2.3/config/train_soh.yaml)")
    ap.add_argument("--ckpt", type=str, default="", help="Checkpoint .pt (default: 2_models/LSTM/Base/0.1.2.3/checkpoints/best*.pt)")
    ap.add_argument("--scaler", type=str, default="", help="Scaler joblib (default: 2_models/LSTM/Base/0.1.2.3/scaler_robust.joblib)")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    repo = _repo_root()
    out_root = repo / "DL_Models" / "LFP_SOH_Optimization_Study" / "6_test" / "STM32DC" / "LSTM_0.1.2.3"
    hw_dir = Path(args.hw_dir) if args.hw_dir else _find_latest_hw_run(out_root)

    csv_in = hw_dir / "stm32_hw_c11.csv"
    meta_in = hw_dir / "metadata.json"
    if not csv_in.exists():
        raise FileNotFoundError(f"Missing: {csv_in}")

    df_hw = pd.read_csv(csv_in)

    meta = {}
    if meta_in.exists():
        meta = json.loads(meta_in.read_text(encoding="utf-8"))

    data_root = Path(meta.get("data_root", r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE"))
    parquet = data_root / f"df_FE_{args.cell}.parquet"
    if not parquet.exists():
        parquet = data_root / f"df_FE_C{args.cell[-2:]}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet}")

    cfg_path = Path(args.config) if args.config else (repo / "DL_Models" / "LFP_SOH_Optimization_Study" / "1_training" / "0.1.2.3" / "config" / "train_soh.yaml")
    train_py = repo / "DL_Models" / "LFP_SOH_Optimization_Study" / "1_training" / "0.1.2.3" / "scripts" / "train_soh.py"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not train_py.exists():
        raise FileNotFoundError(f"Missing train_soh.py: {train_py}")

    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base_features = list(cfg["model"]["features"])
    target = cfg.get("training", {}).get("target", "SOH")
    sampling_cfg = cfg.get("sampling", {})
    interval_seconds = int(sampling_cfg.get("interval_seconds", 3600))
    seq_len = int(cfg["training"]["seq_chunk_size"])

    train_mod = _load_train_module(train_py)
    if hasattr(train_mod, "expand_features_for_sampling"):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    model_dir = repo / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.3"
    ckpt_path = Path(args.ckpt) if args.ckpt else next(iter(sorted((model_dir / "checkpoints").glob("best_epoch*_rmse*.pt"))), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found under: {model_dir / 'checkpoints'}")

    scaler_path = Path(args.scaler) if args.scaler else (model_dir / "scaler_robust.joblib")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    # Load only the raw rows covering the HW playback window.
    max_points = int(meta.get("max_points", len(df_hw)))
    row_stride = int(meta.get("row_stride", 1))
    want_raw = max_points * max(1, row_stride)

    raw_cols = list(dict.fromkeys(["Testtime[s]"] + base_features + [target]))
    df_raw = _read_parquet_head(parquet, raw_cols, want_raw)
    if row_stride > 1:
        df_raw = df_raw.iloc[::row_stride].reset_index(drop=True)
    df_raw = df_raw.iloc[: len(df_hw)].reset_index(drop=True)

    # Apply the same hourly aggregation + feature expansion as the PC pipeline.
    if sampling_cfg.get("enabled", False) and hasattr(train_mod, "maybe_aggregate_hourly"):
        df_agg = train_mod.maybe_aggregate_hourly(df_raw, base_features, target, sampling_cfg)
    else:
        df_agg = df_raw.copy()

    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])

    device = torch.device(args.device)
    model = train_mod.SOH_LSTM_Seq2Seq(
        in_features=len(features),
        embed_size=int(cfg["model"].get("embed_size", 96)),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 2)),
        res_blocks=int(cfg["model"].get("res_blocks", 2)),
        bidirectional=bool(cfg["model"].get("bidirectional", False)),
        dropout=float(cfg["model"].get("dropout", 0.15)),
    ).to(device)

    # Local checkpoint from this repo: allow full load (PyTorch >=2.6 defaults weights_only=True).
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)

    scaler = joblib_load(scaler_path)
    x = df_agg[features].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)
    y_pred_hourly = _stateful_predict(model, x_scaled, seq_len, device).astype(np.float64)

    # Map hourly predictions to HW timeline.
    #
    # Important: the STM32 implementation is causal (1h latency): it can only output hour-bin k
    # after finishing that hour. Therefore the firmware also returns a `ts:` that points to the
    # last sample timestamp of the hour-bin used for the prediction.
    #
    # New HW CSVs contain `soh_ts_ms` (the firmware's returned ts). Use that for alignment.
    # Old CSVs don't; fall back to `ts_ms` which is less precise for causal alignment.
    if "Testtime[s]" in df_raw.columns and len(df_raw) > 0:
        t0_s = float(df_raw["Testtime[s]"].iloc[0])
        t0_ms = int(round(t0_s * 1000.0))
    else:
        t0_ms = int(df_hw["ts_ms"].iloc[0]) if "ts_ms" in df_hw.columns and len(df_hw) > 0 else 0

    if "soh_ts_ms" in df_hw.columns and df_hw["soh_ts_ms"].notna().any():
        ts_ref_ms = pd.to_numeric(df_hw["soh_ts_ms"], errors="coerce").fillna(df_hw.get("ts_ms", 0)).to_numpy(dtype=np.int64)
    else:
        ts_ref_ms = pd.to_numeric(df_hw.get("ts_ms", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    bin_ms = int(interval_seconds) * 1000
    hour_bins = np.floor((ts_ref_ms.astype(np.float64) - float(t0_ms)) / float(bin_ms)).astype(int)

    soh_pc = np.full(len(df_hw), np.nan, dtype=np.float64)
    for i, b in enumerate(hour_bins.tolist()):
        if 0 <= b < len(y_pred_hourly):
            soh_pc[i] = float(y_pred_hourly[b])

    df_out = df_hw.copy()
    df_out["soh_pc"] = soh_pc

    csv_out = hw_dir / "stm32_hw_c11_with_pc.csv"
    df_out.to_csv(csv_out, index=False)

    summary = {
        "hw_dir": str(hw_dir),
        "csv_in": str(csv_in),
        "csv_out": str(csv_out),
        "ckpt": str(ckpt_path),
        "scaler": str(scaler_path),
        "interval_seconds": interval_seconds,
        "agg_points": int(len(df_agg)),
    }
    (hw_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_out}")
    print("Next: run plot_hw_soh_compare.py to generate PNG.")


if __name__ == "__main__":
    main()
