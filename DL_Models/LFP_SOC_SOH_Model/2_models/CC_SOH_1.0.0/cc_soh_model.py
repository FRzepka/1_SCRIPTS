"""
CC + SOH coupling model.
- SOH predicted hourly by a config-driven recurrent sequence model
- SOH held constant within each hour
- Capacity scaled by SOH -> CC SOC
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

# CC model
CC_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "CC_1.0.0")
sys.path.append(os.path.abspath(CC_MODEL_DIR))
from cc_model import CCModel, CCModelConfig


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        out = self.drop(out)
        return self.norm(x + out)


class SOH_GRU_Seq2Seq(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        if bidirectional:
            print("Warning: bidirectional=True breaks true stateful inference.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out = hidden_size * self.num_directions
        self.post_norm = nn.LayerNorm(gru_out)
        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(gru_out, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(gru_out, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, state: Optional[torch.Tensor] = None, return_state: bool = False):
        x = self.feature_proj(x)
        out, new_state = self.gru(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y_seq = self.head(out).squeeze(-1)
        if return_state:
            return y_seq, new_state
        return y_seq


class SOH_LSTM_Seq2Seq(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        if bidirectional:
            print("Warning: bidirectional=True breaks true stateful inference.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out = hidden_size * self.num_directions
        self.post_norm = nn.LayerNorm(lstm_out)
        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(lstm_out, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_out, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, state=None, return_state: bool = False):
        x = self.feature_proj(x)
        out, new_state = self.lstm(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y_seq = self.head(out).squeeze(-1)
        if return_state:
            return y_seq, new_state
        return y_seq


def expand_features_for_sampling(base_features: List[str], feature_aggs: List[str]) -> List[str]:
    return [f"{feat}_{agg}" for feat in base_features for agg in feature_aggs]


def aggregate_hourly(df: pd.DataFrame, base_features: List[str], interval_seconds: int, feature_aggs: List[str]) -> pd.DataFrame:
    if "Testtime[s]" not in df.columns:
        raise ValueError("Testtime[s] column required for hourly aggregation")
    work = df[base_features + ["Testtime[s]"]].replace([np.inf, -np.inf], np.nan).dropna(
        subset=base_features + ["Testtime[s]"]
    ).copy()
    work = work.sort_values("Testtime[s]")
    if work.empty:
        return work
    bins = (work["Testtime[s]"] // interval_seconds).astype(np.int64)
    work["_bin"] = bins
    agg_spec = {feat: feature_aggs for feat in base_features}
    out = work.groupby("_bin", sort=True).agg(agg_spec)
    out.columns = [f"{col[0]}_{col[1]}" for col in out.columns]
    out["bin"] = out.index
    return out.reset_index(drop=True)


@dataclass
class CCSOHConfig:
    soh_config: str
    soh_checkpoint: str
    soh_scaler: str
    nominal_capacity_ah: float = 1.8
    soh_interval_seconds: int = 3600
    soh_init: float = 1.0
    device: Optional[str] = None

    # CC config
    soc_init: float = 1.0
    current_sign: float = 1.0
    v_max: float = 3.65
    v_tol: float = 0.02
    cv_seconds: float = 300.0


class CCSOHModel:
    def __init__(self, config: CCSOHConfig):
        self.config = config
        self.device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_soh_model()
        self._cc = CCModel(
            CCModelConfig(
                capacity_ah=float(config.nominal_capacity_ah),
                soc_init=float(config.soc_init),
                current_sign=float(config.current_sign),
                v_max=float(config.v_max),
                v_tol=float(config.v_tol),
                cv_seconds=float(config.cv_seconds),
            )
        )

    def _load_soh_model(self):
        import yaml
        from joblib import load

        with open(self.config.soh_config, "r") as f:
            soh_cfg = yaml.safe_load(f)

        self.soh_base_features = soh_cfg["model"]["features"]
        self.soh_feature_aggs = soh_cfg.get("sampling", {}).get("feature_aggs", ["mean", "std", "min", "max"])
        self.soh_interval_seconds = int(
            soh_cfg.get("sampling", {}).get("interval_seconds", self.config.soh_interval_seconds)
        )

        in_features = len(expand_features_for_sampling(self.soh_base_features, self.soh_feature_aggs))
        embed_size = int(soh_cfg["model"]["embed_size"])
        hidden_size = int(soh_cfg["model"]["hidden_size"])
        mlp_hidden = int(soh_cfg["model"]["mlp_hidden"])
        num_layers = int(soh_cfg["model"].get("num_layers", 2))
        res_blocks = int(soh_cfg["model"].get("res_blocks", 2))
        bidirectional = bool(soh_cfg["model"].get("bidirectional", False))
        dropout = float(soh_cfg["model"].get("dropout", 0.15))

        model_type = str(soh_cfg["model"].get("type", "GRU_Hybrid_Seq2Seq"))
        self.soh_model_type = model_type
        model_cls = SOH_LSTM_Seq2Seq if "LSTM" in model_type.upper() else SOH_GRU_Seq2Seq

        self.soh_scaler: RobustScaler = load(self.config.soh_scaler)
        self.soh_model = model_cls(
            in_features=in_features,
            embed_size=embed_size,
            hidden_size=hidden_size,
            mlp_hidden=mlp_hidden,
            num_layers=num_layers,
            res_blocks=res_blocks,
            bidirectional=bidirectional,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(self.config.soh_checkpoint, map_location=self.device)
        self.soh_model.load_state_dict(state["model_state_dict"])
        self.soh_model.eval()

    def predict_soh_hourly(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        hourly = aggregate_hourly(
            df_raw,
            self.soh_base_features,
            self.soh_interval_seconds,
            self.soh_feature_aggs,
        )
        if hourly.empty:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        feature_cols = expand_features_for_sampling(self.soh_base_features, self.soh_feature_aggs)
        X = hourly[feature_cols].to_numpy(dtype=np.float32)
        Xs = self.soh_scaler.transform(X).astype(np.float32)

        preds = []
        state = None
        with torch.no_grad():
            for i in range(len(Xs)):
                x_step = torch.from_numpy(Xs[i : i + 1]).unsqueeze(1).to(self.device)
                y_seq, state = self.soh_model(x_step, state=state, return_state=True)
                y_val = float(y_seq.squeeze().detach().cpu().numpy())
                if i == 0:
                    preds.append(float(self.config.soh_init))
                else:
                    preds.append(y_val)

        bins = hourly["bin"].to_numpy(dtype=np.int64)
        return np.array(preds, dtype=np.float32), bins

    def expand_soh_to_rows(self, df_raw: pd.DataFrame, bins: np.ndarray, soh_preds: np.ndarray) -> np.ndarray:
        if "Testtime[s]" not in df_raw.columns:
            raise ValueError("Testtime[s] column required")
        bin_to_soh = {int(b): float(s) for b, s in zip(bins, soh_preds)}
        if len(bins) > 0:
            max_bin = int(np.max(bins))
            last = float(self.config.soh_init)
            for b in range(0, max_bin + 1):
                if b in bin_to_soh:
                    last = bin_to_soh[b]
                else:
                    bin_to_soh[b] = last
        out = []
        for t in df_raw["Testtime[s]"].to_numpy(dtype=np.float64):
            b = int(t // self.soh_interval_seconds)
            out.append(bin_to_soh.get(b, float(self.config.soh_init)))
        return np.array(out, dtype=np.float32)

    def process_dataframe(self, df_raw: pd.DataFrame, gap_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        df = df_raw.replace([np.inf, -np.inf], np.nan).copy()
        df = df.dropna(subset=self.soh_base_features + ["Testtime[s]", "Current[A]", "Voltage[V]"])
        df = df.reset_index(drop=True)

        soh_hourly, bins = self.predict_soh_hourly(df)
        soh_per_row = self.expand_soh_to_rows(df, bins, soh_hourly)

        t = df["Testtime[s]"].to_numpy(dtype=np.float64)
        dt_s = np.diff(t, prepend=t[0])
        dt_s[dt_s < 0] = 0.0
        has_gap = False
        if gap_mask is not None:
            if len(gap_mask) != len(df):
                raise ValueError(f"gap_mask length mismatch: got {len(gap_mask)} for dataframe length {len(df)}")
            has_gap = bool(np.any(gap_mask))
        if has_gap:
            nominal_dt = np.median(dt_s[(~gap_mask) & (dt_s > 0)])
            if not np.isfinite(nominal_dt) or nominal_dt <= 0:
                nominal_dt = 1.0
            dt_s[gap_mask] = 0.0
            for k in range(1, len(dt_s)):
                if gap_mask[k - 1] and not gap_mask[k]:
                    dt_s[k] = nominal_dt

        i = df["Current[A]"].to_numpy(dtype=np.float64)
        v = df["Voltage[V]"].to_numpy(dtype=np.float64)

        soc = np.zeros(len(df), dtype=np.float32)
        for k in range(len(df)):
            if has_gap and gap_mask[k]:
                soc[k] = float(self.config.soh_init) if k == 0 else soc[k - 1]
                continue
            cap_k = float(self.config.nominal_capacity_ah) * float(soh_per_row[k])
            soc[k] = self._cc.step(i[k], v[k], capacity_ah=cap_k, dt_s=dt_s[k])

        return soc, soh_per_row
