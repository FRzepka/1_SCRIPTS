#!/usr/bin/env python3
"""Apply global magnitude pruning to a trained SOH model checkpoint."""
import argparse
import copy
import importlib.util
import inspect
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.utils.prune as prune
import yaml


def expand_env_with_defaults(val: str) -> str:
    pattern = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, "") else default

    if not isinstance(val, str):
        return val
    return os.path.expandvars(pattern.sub(repl, val))


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def build_model(cfg: dict, train_mod):
    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    if hasattr(train_mod, "expand_features_for_sampling"):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    mtype = str(cfg["model"].get("type", "")).lower()
    if "lstm" in mtype:
        cls = train_mod.SOH_LSTM_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 0)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "gru" in mtype:
        cls = train_mod.SOH_GRU_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 0)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "tcn" in mtype:
        cls = train_mod.CausalTCN_SOH
        dilations = cfg["model"].get("dilations") or [1, 2, 4, 8]
        model = cls(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 3)),
            dilations=[int(d) for d in dilations],
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "cnn" in mtype:
        cls = train_mod.SOH_CNN_Seq2Seq
        dilations = cfg["model"].get("dilations")
        kwargs = dict(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dilations=[int(d) for d in dilations] if dilations is not None else None,
            num_blocks=int(cfg["model"].get("num_blocks", 4)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
        if "output_kernel_size" in inspect.signature(cls).parameters:
            kwargs["output_kernel_size"] = int(cfg["model"].get("output_kernel_size", 1))
        model = cls(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")

    return model


def pick_checkpoint(ckpt_dir: Path, ckpt_arg: Optional[Path]) -> Path:
    if ckpt_arg is not None:
        return ckpt_arg

    preferred = ckpt_dir / "best_model.pt"
    if preferred.exists():
        return preferred

    for name in ("final_model.pt",):
        p = ckpt_dir / name
        if p.exists():
            return p

    best = list(ckpt_dir.glob("best_epoch*_rmse*.pt"))
    if best:
        def _rmse(path: Path) -> float:
            m = re.search(r"rmse([0-9]+(?:\.[0-9]+)?)", path.name)
            return float(m.group(1)) if m else float("inf")
        best.sort(key=_rmse)
        return best[0]

    all_pts = sorted(ckpt_dir.glob("*.pt"))
    if not all_pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return all_pts[0]


def collect_prune_targets(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, str]]:
    targets: List[Tuple[torch.nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                targets.append((module, "weight"))
        elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            for name, param in module.named_parameters(recurse=False):
                if name.startswith("weight_") and isinstance(param, torch.Tensor):
                    targets.append((module, name))
    if not targets:
        raise RuntimeError("No prunable weight tensors found.")
    return targets


def _pick_tcn_keep_indices(model: torch.nn.Module, keep_channels: int) -> torch.Tensor:
    hidden = int(model.tcn[0].conv1.conv.weight.shape[0])
    scores = torch.zeros(hidden, dtype=torch.float32)

    for block in model.tcn:
        w1 = block.conv1.conv.weight.detach().abs().mean(dim=(1, 2))
        w2 = block.conv2.conv.weight.detach().abs().mean(dim=(1, 2))
        scores += w1.to(scores.dtype)
        scores += w2.to(scores.dtype)

    ds0 = model.tcn[0].downsample
    if ds0 is not None and isinstance(ds0, torch.nn.Conv1d):
        scores += ds0.weight.detach().abs().mean(dim=(1, 2)).to(scores.dtype)

    head0 = model.head[0]
    if isinstance(head0, torch.nn.Conv1d):
        # Contribution of each hidden channel as input to the head.
        scores += head0.weight.detach().abs().mean(dim=(0, 2)).to(scores.dtype)

    keep = torch.topk(scores, k=keep_channels, largest=True).indices
    keep, _ = torch.sort(keep)
    return keep.long()


def _copy_tcn_weights_structured(old_model: torch.nn.Module, new_model: torch.nn.Module, keep_idx: torch.Tensor) -> None:
    keep = keep_idx.tolist()

    old_blocks = list(old_model.tcn)
    new_blocks = list(new_model.tcn)
    for bi, (old_b, new_b) in enumerate(zip(old_blocks, new_blocks)):
        old_w = old_b.conv1.conv.weight.data
        new_w = new_b.conv1.conv.weight.data
        out_old = keep
        in_old = list(range(old_w.shape[1])) if bi == 0 else keep
        new_w.copy_(old_w[out_old][:, in_old, :])
        if old_b.conv1.conv.bias is not None and new_b.conv1.conv.bias is not None:
            new_b.conv1.conv.bias.data.copy_(old_b.conv1.conv.bias.data[out_old])

        old_w = old_b.conv2.conv.weight.data
        new_w = new_b.conv2.conv.weight.data
        new_w.copy_(old_w[keep][:, keep, :])
        if old_b.conv2.conv.bias is not None and new_b.conv2.conv.bias is not None:
            new_b.conv2.conv.bias.data.copy_(old_b.conv2.conv.bias.data[keep])

        if old_b.downsample is not None and new_b.downsample is not None:
            old_w = old_b.downsample.weight.data
            new_b.downsample.weight.data.copy_(old_w[keep][:, list(range(old_w.shape[1])), :])
            if old_b.downsample.bias is not None and new_b.downsample.bias is not None:
                new_b.downsample.bias.data.copy_(old_b.downsample.bias.data[keep])

    if isinstance(old_model.head[0], torch.nn.Conv1d) and isinstance(new_model.head[0], torch.nn.Conv1d):
        old_w = old_model.head[0].weight.data
        new_model.head[0].weight.data.copy_(old_w[:, keep, :])
        if old_model.head[0].bias is not None and new_model.head[0].bias is not None:
            new_model.head[0].bias.data.copy_(old_model.head[0].bias.data)

    if isinstance(old_model.head[3], torch.nn.Conv1d) and isinstance(new_model.head[3], torch.nn.Conv1d):
        new_model.head[3].weight.data.copy_(old_model.head[3].weight.data)
        if old_model.head[3].bias is not None and new_model.head[3].bias is not None:
            new_model.head[3].bias.data.copy_(old_model.head[3].bias.data)


def _structured_prune_tcn(
    cfg: dict,
    train_mod,
    loaded_model: torch.nn.Module,
    amount: float,
    min_hidden_channels: int,
    round_to: int,
) -> Tuple[dict, torch.nn.Module, dict]:
    mtype = str(cfg["model"].get("type", "")).lower()
    if "tcn" not in mtype:
        raise ValueError("Structured mode currently supports TCN only.")

    old_hidden = int(cfg["model"].get("hidden_size", 128))
    keep_raw = int(round(old_hidden * (1.0 - amount)))
    keep_raw = max(1, keep_raw)

    if round_to > 1:
        keep_rounded = int(round(keep_raw / float(round_to))) * int(round_to)
    else:
        keep_rounded = keep_raw
    keep_rounded = max(min_hidden_channels, keep_rounded)
    keep_rounded = min(old_hidden, keep_rounded)

    if keep_rounded >= old_hidden:
        raise ValueError(
            f"Structured prune would keep all channels (old_hidden={old_hidden}, keep={keep_rounded}). "
            "Increase --amount or lower --min-hidden-channels."
        )

    keep_idx = _pick_tcn_keep_indices(loaded_model, keep_rounded)

    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["hidden_size"] = int(keep_rounded)
    new_model = build_model(new_cfg, train_mod).cpu().eval()
    _copy_tcn_weights_structured(loaded_model.cpu().eval(), new_model, keep_idx)

    meta = {
        "mode": "structured",
        "structured_kind": "tcn_channel_shrink",
        "old_hidden_size": int(old_hidden),
        "new_hidden_size": int(keep_rounded),
        "keep_indices": [int(x) for x in keep_idx.tolist()],
        "amount": float(amount),
        "min_hidden_channels": int(min_hidden_channels),
        "round_to": int(round_to),
    }
    return new_cfg, new_model, meta


def _gate_row_indices(hidden_old: int, keep_idx: torch.Tensor, gates: int) -> List[int]:
    keep = [int(i) for i in keep_idx.tolist()]
    out: List[int] = []
    for g in range(gates):
        off = g * hidden_old
        out.extend([off + i for i in keep])
    return out


def _pick_rnn_keep_indices(model: torch.nn.Module, rnn_attr: str, keep_hidden: int, gates: int) -> torch.Tensor:
    rnn = getattr(model, rnn_attr)
    hidden_old = int(rnn.hidden_size)
    num_layers = int(rnn.num_layers)
    if bool(getattr(rnn, "bidirectional", False)):
        raise ValueError("Structured pruning currently supports only unidirectional LSTM/GRU.")

    scores = torch.zeros(hidden_old, dtype=torch.float32)

    for l in range(num_layers):
        w_hh = getattr(rnn, f"weight_hh_l{l}").detach()
        for g in range(gates):
            rows = w_hh[g * hidden_old:(g + 1) * hidden_old, :]
            scores += rows.abs().mean(dim=1).to(scores.dtype)
            scores += rows.abs().mean(dim=0).to(scores.dtype)

        if l + 1 < num_layers:
            w_ih_next = getattr(rnn, f"weight_ih_l{l+1}").detach()
            for g in range(gates):
                rows = w_ih_next[g * hidden_old:(g + 1) * hidden_old, :]
                scores += rows.abs().mean(dim=0).to(scores.dtype)

    if isinstance(model.head[0], torch.nn.Linear):
        scores += model.head[0].weight.detach().abs().mean(dim=0).to(scores.dtype)
    for blk in model.res_blocks:
        if hasattr(blk, "fc1") and isinstance(blk.fc1, torch.nn.Linear):
            scores += blk.fc1.weight.detach().abs().mean(dim=0).to(scores.dtype)
    if hasattr(model, "post_norm") and isinstance(model.post_norm, torch.nn.LayerNorm):
        scores += model.post_norm.weight.detach().abs().to(scores.dtype)

    keep = torch.topk(scores, k=keep_hidden, largest=True).indices
    keep, _ = torch.sort(keep)
    return keep.long()


def _copy_rnn_hidden_shrink_weights(
    old_model: torch.nn.Module,
    new_model: torch.nn.Module,
    rnn_attr: str,
    keep_idx: torch.Tensor,
    gates: int,
) -> None:
    old_rnn = getattr(old_model, rnn_attr)
    new_rnn = getattr(new_model, rnn_attr)
    hidden_old = int(old_rnn.hidden_size)
    num_layers = int(old_rnn.num_layers)
    keep = [int(i) for i in keep_idx.tolist()]
    rows = _gate_row_indices(hidden_old, keep_idx, gates)

    new_model.feature_proj.load_state_dict(old_model.feature_proj.state_dict())

    for l in range(num_layers):
        old_w_ih = getattr(old_rnn, f"weight_ih_l{l}").data
        new_w_ih = getattr(new_rnn, f"weight_ih_l{l}").data
        if l == 0:
            in_cols = list(range(old_w_ih.shape[1]))
        else:
            in_cols = keep
        new_w_ih.copy_(old_w_ih[rows][:, in_cols])

        old_w_hh = getattr(old_rnn, f"weight_hh_l{l}").data
        new_w_hh = getattr(new_rnn, f"weight_hh_l{l}").data
        new_w_hh.copy_(old_w_hh[rows][:, keep])

        old_b_ih = getattr(old_rnn, f"bias_ih_l{l}").data
        new_b_ih = getattr(new_rnn, f"bias_ih_l{l}").data
        new_b_ih.copy_(old_b_ih[rows])

        old_b_hh = getattr(old_rnn, f"bias_hh_l{l}").data
        new_b_hh = getattr(new_rnn, f"bias_hh_l{l}").data
        new_b_hh.copy_(old_b_hh[rows])

    new_model.post_norm.weight.data.copy_(old_model.post_norm.weight.data[keep])
    new_model.post_norm.bias.data.copy_(old_model.post_norm.bias.data[keep])

    for old_blk, new_blk in zip(old_model.res_blocks, new_model.res_blocks):
        new_blk.fc1.weight.data.copy_(old_blk.fc1.weight.data[:, keep])
        new_blk.fc1.bias.data.copy_(old_blk.fc1.bias.data)
        new_blk.fc2.weight.data.copy_(old_blk.fc2.weight.data[keep, :])
        new_blk.fc2.bias.data.copy_(old_blk.fc2.bias.data[keep])
        new_blk.norm.weight.data.copy_(old_blk.norm.weight.data[keep])
        new_blk.norm.bias.data.copy_(old_blk.norm.bias.data[keep])

    if isinstance(old_model.head[0], torch.nn.Linear) and isinstance(new_model.head[0], torch.nn.Linear):
        new_model.head[0].weight.data.copy_(old_model.head[0].weight.data[:, keep])
        new_model.head[0].bias.data.copy_(old_model.head[0].bias.data)
    for idx in (3, 6):
        if idx < len(old_model.head) and isinstance(old_model.head[idx], torch.nn.Linear):
            new_model.head[idx].weight.data.copy_(old_model.head[idx].weight.data)
            new_model.head[idx].bias.data.copy_(old_model.head[idx].bias.data)


def _structured_prune_lstm_or_gru(
    cfg: dict,
    train_mod,
    loaded_model: torch.nn.Module,
    amount: float,
    min_hidden_channels: int,
    round_to: int,
) -> Tuple[dict, torch.nn.Module, dict]:
    mtype = str(cfg["model"].get("type", "")).lower()
    if "lstm" in mtype:
        rnn_attr = "lstm"
        gates = 4
    elif "gru" in mtype:
        rnn_attr = "gru"
        gates = 3
    else:
        raise ValueError("Expected LSTM or GRU model.")

    old_hidden = int(cfg["model"].get("hidden_size", 128))
    keep_raw = max(1, int(round(old_hidden * (1.0 - amount))))
    if round_to > 1:
        keep_rounded = int(round(keep_raw / float(round_to))) * int(round_to)
    else:
        keep_rounded = keep_raw
    keep_rounded = max(min_hidden_channels, keep_rounded)
    keep_rounded = min(old_hidden, keep_rounded)
    if keep_rounded >= old_hidden:
        raise ValueError(
            f"Structured prune would keep all hidden units (old_hidden={old_hidden}, keep={keep_rounded})."
        )

    keep_idx = _pick_rnn_keep_indices(loaded_model, rnn_attr=rnn_attr, keep_hidden=keep_rounded, gates=gates)
    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["hidden_size"] = int(keep_rounded)
    new_model = build_model(new_cfg, train_mod).cpu().eval()
    _copy_rnn_hidden_shrink_weights(
        old_model=loaded_model.cpu().eval(),
        new_model=new_model,
        rnn_attr=rnn_attr,
        keep_idx=keep_idx,
        gates=gates,
    )
    meta = {
        "mode": "structured",
        "structured_kind": f"{rnn_attr}_hidden_shrink",
        "old_hidden_size": int(old_hidden),
        "new_hidden_size": int(keep_rounded),
        "keep_indices": [int(x) for x in keep_idx.tolist()],
        "amount": float(amount),
        "min_hidden_channels": int(min_hidden_channels),
        "round_to": int(round_to),
    }
    return new_cfg, new_model, meta


def _pick_cnn_keep_indices(model: torch.nn.Module, keep_channels: int) -> torch.Tensor:
    hidden = int(model.input_proj.weight.shape[0])
    scores = torch.zeros(hidden, dtype=torch.float32)
    scores += model.input_proj.weight.detach().abs().mean(dim=(1, 2)).to(scores.dtype)
    for block in model.blocks:
        scores += block.conv1.conv.weight.detach().abs().mean(dim=(1, 2)).to(scores.dtype)
        scores += block.conv2.conv.weight.detach().abs().mean(dim=(1, 2)).to(scores.dtype)
        if block.downsample is not None:
            scores += block.downsample.weight.detach().abs().mean(dim=(1, 2)).to(scores.dtype)
    if isinstance(model.head[0], torch.nn.Conv1d):
        scores += model.head[0].weight.detach().abs().mean(dim=(0, 2)).to(scores.dtype)
    keep = torch.topk(scores, k=keep_channels, largest=True).indices
    keep, _ = torch.sort(keep)
    return keep.long()


def _copy_cnn_structured_weights(old_model: torch.nn.Module, new_model: torch.nn.Module, keep_idx: torch.Tensor) -> None:
    keep = keep_idx.tolist()
    new_model.input_proj.weight.data.copy_(old_model.input_proj.weight.data[keep, :, :])
    if old_model.input_proj.bias is not None and new_model.input_proj.bias is not None:
        new_model.input_proj.bias.data.copy_(old_model.input_proj.bias.data[keep])

    for old_b, new_b in zip(old_model.blocks, new_model.blocks):
        new_b.conv1.conv.weight.data.copy_(old_b.conv1.conv.weight.data[keep][:, keep, :])
        if old_b.conv1.conv.bias is not None and new_b.conv1.conv.bias is not None:
            new_b.conv1.conv.bias.data.copy_(old_b.conv1.conv.bias.data[keep])
        new_b.conv2.conv.weight.data.copy_(old_b.conv2.conv.weight.data[keep][:, keep, :])
        if old_b.conv2.conv.bias is not None and new_b.conv2.conv.bias is not None:
            new_b.conv2.conv.bias.data.copy_(old_b.conv2.conv.bias.data[keep])
        if old_b.downsample is not None and new_b.downsample is not None:
            new_b.downsample.weight.data.copy_(old_b.downsample.weight.data[keep][:, keep, :])
            if old_b.downsample.bias is not None and new_b.downsample.bias is not None:
                new_b.downsample.bias.data.copy_(old_b.downsample.bias.data[keep])

    if isinstance(old_model.head[0], torch.nn.Conv1d) and isinstance(new_model.head[0], torch.nn.Conv1d):
        new_model.head[0].weight.data.copy_(old_model.head[0].weight.data[:, keep, :])
        new_model.head[0].bias.data.copy_(old_model.head[0].bias.data)
    if isinstance(old_model.head[3], torch.nn.Conv1d) and isinstance(new_model.head[3], torch.nn.Conv1d):
        new_model.head[3].weight.data.copy_(old_model.head[3].weight.data)
        new_model.head[3].bias.data.copy_(old_model.head[3].bias.data)

    if hasattr(old_model, "output_smoother") and old_model.output_smoother is not None:
        if new_model.output_smoother is not None:
            new_model.output_smoother.conv.weight.data.copy_(old_model.output_smoother.conv.weight.data)
            if old_model.output_smoother.conv.bias is not None and new_model.output_smoother.conv.bias is not None:
                new_model.output_smoother.conv.bias.data.copy_(old_model.output_smoother.conv.bias.data)


def _structured_prune_cnn(
    cfg: dict,
    train_mod,
    loaded_model: torch.nn.Module,
    amount: float,
    min_hidden_channels: int,
    round_to: int,
) -> Tuple[dict, torch.nn.Module, dict]:
    old_hidden = int(cfg["model"].get("hidden_size", 128))
    keep_raw = max(1, int(round(old_hidden * (1.0 - amount))))
    if round_to > 1:
        keep_rounded = int(round(keep_raw / float(round_to))) * int(round_to)
    else:
        keep_rounded = keep_raw
    keep_rounded = max(min_hidden_channels, keep_rounded)
    keep_rounded = min(old_hidden, keep_rounded)
    if keep_rounded >= old_hidden:
        raise ValueError(
            f"Structured prune would keep all channels (old_hidden={old_hidden}, keep={keep_rounded})."
        )
    keep_idx = _pick_cnn_keep_indices(loaded_model, keep_channels=keep_rounded)
    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["hidden_size"] = int(keep_rounded)
    new_model = build_model(new_cfg, train_mod).cpu().eval()
    _copy_cnn_structured_weights(loaded_model.cpu().eval(), new_model, keep_idx)
    meta = {
        "mode": "structured",
        "structured_kind": "cnn_channel_shrink",
        "old_hidden_size": int(old_hidden),
        "new_hidden_size": int(keep_rounded),
        "keep_indices": [int(x) for x in keep_idx.tolist()],
        "amount": float(amount),
        "min_hidden_channels": int(min_hidden_channels),
        "round_to": int(round_to),
    }
    return new_cfg, new_model, meta


def _structured_prune_model(
    cfg: dict,
    train_mod,
    loaded_model: torch.nn.Module,
    amount: float,
    min_hidden_channels: int,
    round_to: int,
) -> Tuple[dict, torch.nn.Module, dict]:
    mtype = str(cfg["model"].get("type", "")).lower()
    if "tcn" in mtype:
        return _structured_prune_tcn(cfg, train_mod, loaded_model, amount, min_hidden_channels, round_to)
    if "lstm" in mtype or "gru" in mtype:
        return _structured_prune_lstm_or_gru(cfg, train_mod, loaded_model, amount, min_hidden_channels, round_to)
    if "cnn" in mtype:
        return _structured_prune_cnn(cfg, train_mod, loaded_model, amount, min_hidden_channels, round_to)
    raise ValueError(f"Structured pruning is unsupported for model type: {cfg['model'].get('type')}")


def tensor_sparsity(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    zeros = int((t == 0).sum().item())
    return zeros / float(t.numel())


def prune_checkpoint(
    model_dir: Path,
    out_dir: Path,
    amount: float,
    ckpt_path: Optional[Path],
    mode: str,
    min_hidden_channels: int,
    round_to: int,
) -> None:
    cfg_path = model_dir / "config" / "train_soh.yaml"
    train_py = model_dir / "scripts" / "train_soh.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not train_py.exists():
        raise FileNotFoundError(f"Missing training script: {train_py}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["data_root"] = expand_env_with_defaults(cfg["paths"]["data_root"])
    cfg["paths"]["out_root"] = expand_env_with_defaults(cfg["paths"]["out_root"])

    train_mod = load_train_module(train_py)
    model = build_model(cfg, train_mod).cpu().eval()

    ckpt_dir = model_dir / "checkpoints"
    chosen_ckpt = pick_checkpoint(ckpt_dir, ckpt_path)
    state = torch.load(chosen_ckpt, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)

    if mode == "unstructured":
        targets = collect_prune_targets(model)
        total_target_elems = sum(getattr(mod, name).numel() for mod, name in targets)

        pre_zero = sum(int((getattr(mod, name) == 0).sum().item()) for mod, name in targets)
        prune.global_unstructured(
            targets,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        for mod, name in targets:
            prune.remove(mod, name)
        post_zero = sum(int((getattr(mod, name) == 0).sum().item()) for mod, name in targets)
        global_sparsity = post_zero / float(total_target_elems) if total_target_elems else 0.0
        cfg_to_save = cfg
        model_to_save = model
        extra_meta = {
            "mode": "unstructured",
            "target_tensors": len(targets),
            "target_params": int(total_target_elems),
            "pre_zero": int(pre_zero),
            "post_zero": int(post_zero),
            "global_sparsity": float(global_sparsity),
        }
    elif mode == "structured":
        cfg_to_save, model_to_save, extra_meta = _structured_prune_model(
            cfg=cfg,
            train_mod=train_mod,
            loaded_model=model,
            amount=amount,
            min_hidden_channels=min_hidden_channels,
            round_to=round_to,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    pruned_ckpt_path = out_dir / "checkpoints" / "best_model_pruned.pt"

    if isinstance(state, dict) and "model_state_dict" in state:
        state_to_save = dict(state)
        state_to_save["model_state_dict"] = model_to_save.state_dict()
    else:
        state_to_save = model_to_save.state_dict()
    torch.save(state_to_save, pruned_ckpt_path)

    for sub in ("config", "scripts", "test"):
        src = model_dir / sub
        dst = out_dir / sub
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    for fname in ("scaler_robust.joblib", "hpo_summary.json", "trials.json"):
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    out_cfg_path = out_dir / "config" / "train_soh.yaml"
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_to_save, f, sort_keys=False)

    meta = {
        "model_dir": str(model_dir),
        "source_checkpoint": str(chosen_ckpt),
        "pruned_checkpoint": str(pruned_ckpt_path),
        "amount": amount,
        **extra_meta,
    }
    with open(out_dir / "prune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if mode == "unstructured":
        print(
            f"Pruned {out_dir.name}: mode=unstructured amount={amount:.3f}, "
            f"targets={extra_meta['target_tensors']}, target_params={extra_meta['target_params']}, "
            f"sparsity={extra_meta['global_sparsity']:.4f}"
        )
    else:
        print(
            f"Pruned {out_dir.name}: mode=structured amount={amount:.3f}, "
            f"hidden={extra_meta['old_hidden_size']}->{extra_meta['new_hidden_size']}"
        )
    print(f"Saved: {pruned_ckpt_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pruning for SOH model checkpoints (unstructured or structured).")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--amount", type=float, default=0.5, help="Global pruning amount in [0,1).")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mode", type=str, default="unstructured", choices=["unstructured", "structured"])
    ap.add_argument("--min-hidden-channels", type=int, default=32, help="Only used for --mode structured.")
    ap.add_argument("--round-to", type=int, default=8, help="Only used for --mode structured.")
    args = ap.parse_args()

    if not (0.0 <= args.amount < 1.0):
        raise ValueError("--amount must be in [0, 1).")
    if args.min_hidden_channels < 1:
        raise ValueError("--min-hidden-channels must be >= 1")
    if args.round_to < 1:
        raise ValueError("--round-to must be >= 1")

    prune_checkpoint(
        model_dir=Path(args.model_dir),
        out_dir=Path(args.out_dir),
        amount=float(args.amount),
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        mode=str(args.mode),
        min_hidden_channels=int(args.min_hidden_channels),
        round_to=int(args.round_to),
    )


if __name__ == "__main__":
    main()
