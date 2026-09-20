import csv
import io
import os
import time
import typing as T
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from config import TRAIN_CELLS, VAL_CELLS, TEST_CELLS, LIMIT_DATA_PER_DATASET


# ----- CONFIG -----

BASE_FEATURES = ["Voltage[V]", "Current[A]", "Temperature[°C]", "EFC", "Q_c"]
FEATURE_AGGS = ["mean", "std", "min", "max"]
FEATURES = [f"{f}_{a}" for f in BASE_FEATURES for a in FEATURE_AGGS]  # 20 cols
TARGET = "SOH"
TARGET_AGG = "last"
INTERVAL_S = 3600


# ----- LOAD HELPERS -----

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
        NON_CONSTRUCTOR_KEYS = {"type", "features"}
        raw = cfg.get("model", {})
    return {k: v for k, v in raw.items() if k not in NON_CONSTRUCTOR_KEYS}


# Read hyperparameters directly out of yaml
def load_finetune_hyperparameters(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    t_cfg = config.get('training', {})

    finetune_kwargs = {
        "weight_decay":      float(t_cfg.get("weight_decay", 1e-4)),
        "max_grad_norm":     float(t_cfg.get("max_grad_norm", 1.0)),
        "smooth_loss_weight": float(t_cfg.get("smooth_loss_weight", 0.05)),
        "smooth_loss_type":   t_cfg.get("smooth_loss_type", "l1"),
        "warmup_steps":      int(t_cfg.get("warmup_steps", 8))
    }
    return finetune_kwargs


# ----- DATA HELPERS -----

def aggregate_hourly(
    df:         pd.DataFrame,
    interval_s: int       = INTERVAL_S,
    feat_aggs:  list[str] = FEATURE_AGGS,
    target_agg: str       = TARGET_AGG,
) -> pd.DataFrame:

    required = BASE_FEATURES + [TARGET, "Testtime[s]"]

    # Filter and Downcast to float32
    work = df[required].copy()
    for col in work.select_dtypes(include=['float64']).columns:
        work[col] = work[col].astype(np.float32)

    # Basic cleanup
    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Testtime[s]")
    )

    # Create the bin
    work["_bin"] = (work["Testtime[s]"] // interval_s).astype(np.int32)

    # GroupBy
    agg_spec = {f: feat_aggs for f in BASE_FEATURES}
    agg_spec[TARGET] = [target_agg]
    out = work.groupby("_bin", sort=False).agg(agg_spec)

    # Flatten columns
    out.columns = [
        TARGET if c[0] == TARGET else f"{c[0]}_{c[1]}"
        for c in out.columns
    ]

    # Free the intermediate dataframe
    del work

    return out.reset_index(drop=True).astype(np.float32)


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler, chunk: int, stride: int = None):
        self.stride = stride if stride is not None else chunk

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
        X = scaler.transform(df[FEATURES].to_numpy(dtype=np.float32))
        y = df[TARGET].to_numpy(dtype=np.float32)

        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        self.chunk = chunk
        n = len(df)
        self.nseq = max(0, 1 + (n - chunk) // self.stride) if n >= chunk else 0

    def __len__(self) -> int:
        return self.nseq

    def __getitem__(self, idx: int):
        s = idx * self.stride
        return self.X[s : s + self.chunk], self.y[s : s + self.chunk]


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ----- DATALOADER -----

def build_dataloaders(
    data_root: str,
    scaler,
    chunk: int = 1,
    batch_size: int = 128,
    stride: int = 1,
    num_workers: int = 2,
    load_train: bool = True,
    load_val: bool = True,
    load_test: bool = True,
) -> tuple[DataLoader | None, DataLoader | None, list[DataLoader] | None]:

    def _load(cells: list[str]) -> list[pd.DataFrame]:
        dfs = []
        required_cols = BASE_FEATURES + [TARGET, "Testtime[s]"]

        for c in cells:
            try:
                path_to_cell = os.path.join(data_root, f"{c}.parquet")
                print(f"%%%%%   -> Loading {c}...")

                cell_data = pd.read_parquet(path_to_cell, columns=required_cols)
                agg_df = aggregate_hourly(cell_data)

                if LIMIT_DATA_PER_DATASET >= 1 and len(agg_df) > LIMIT_DATA_PER_DATASET:
                    agg_df = agg_df.head(LIMIT_DATA_PER_DATASET).reset_index(drop=True)

                dfs.append(agg_df)

                del cell_data       # to save RAM
                gc.collect()

            except FileNotFoundError as exc:
                print(f"##### {exc}")
        return dfs

    loaders = {"train": None, "val": None, "test": None}


    def _make_loader(cells, shuffle):
        ds = ConcatDataset([SeqDataset(df, scaler, chunk, stride) for df in _load(cells)])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=_collate, num_workers=num_workers, pin_memory=True)

    if load_train:
        print("%%%%% Loading train dataset...")
        loaders["train"] = _make_loader(TRAIN_CELLS, shuffle=True)

    if load_val:
        print("%%%%% Loading val dataset...")
        loaders["val"] = _make_loader(VAL_CELLS, shuffle=False)

    if load_test:
        print("%%%%% Loading test dataset...")
        loaders["test"] = [
            DataLoader(SeqDataset(df, scaler, chunk, stride=chunk), batch_size=1, shuffle=False)
            for df in _load(TEST_CELLS)
        ]

    return loaders["train"], loaders["val"], loaders["test"]
