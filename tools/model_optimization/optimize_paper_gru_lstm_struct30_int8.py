#!/usr/bin/env python3
"""Create structured-pruned + int8-quantized variants of the paper GRU/LSTM models.

Outputs:
- SOC GRU 1.7.0.0:
  DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/Pruned_Quantized_1.7.0.0_s30_struct_int8
- SOH LSTM 0.1.2.5_base_h160:
  DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Base/0.1.2.5_base_h160/Pruned_Quantized_0.1.2.5_base_h160_s30_struct_int8
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")

SOC_MODEL_DIR = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0"
SOC_TRAIN_PY = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "1_training" / "1.7.0.0" / "scripts" / "train_soc.py"
SOC_OUT_ROOT = SOC_MODEL_DIR / "Pruned_Quantized_1.7.0.0_s30_struct_int8"

SOH_MODEL_DIR = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.5_base_h160"
SOH_PRUNE_PY = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "3_pruning" / "prune_soh_model.py"
SOH_QUANT_PY = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "4_quantize" / "quantize_soh_model.py"
SOH_OUT_ROOT = SOH_MODEL_DIR / "Pruned_Quantized_0.1.2.5_base_h160_s30_struct_int8"

PRUNE_AMOUNT = 0.30


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_best_rmse_checkpoint(ckpt_dir: Path) -> Path:
    best: Path | None = None
    best_rmse: float | None = None
    for path in ckpt_dir.glob("*.pt"):
        match = re.search(r"rmse([0-9]+(?:\.[0-9]+)?)", path.name)
        if not match:
            continue
        rmse = float(match.group(1))
        if best is None or rmse < best_rmse:
            best = path
            best_rmse = rmse
    if best is None:
        pts = sorted(ckpt_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        best = pts[0]
    return best


def _gate_row_indices(hidden_old: int, keep_idx: torch.Tensor, gates: int) -> list[int]:
    keep = [int(i) for i in keep_idx.tolist()]
    out: list[int] = []
    for g in range(gates):
        off = g * hidden_old
        out.extend([off + i for i in keep])
    return out


def _pick_gru_keep_indices(model: torch.nn.Module, keep_hidden: int) -> torch.Tensor:
    gru = model.gru
    hidden_old = int(gru.hidden_size)
    num_layers = int(gru.num_layers)
    scores = torch.zeros(hidden_old, dtype=torch.float32)

    for layer in range(num_layers):
        w_hh = getattr(gru, f"weight_hh_l{layer}").detach()
        for g in range(3):
            rows = w_hh[g * hidden_old:(g + 1) * hidden_old, :]
            scores += rows.abs().mean(dim=1).to(scores.dtype)
            scores += rows.abs().mean(dim=0).to(scores.dtype)
        if layer + 1 < num_layers:
            w_ih_next = getattr(gru, f"weight_ih_l{layer+1}").detach()
            for g in range(3):
                rows = w_ih_next[g * hidden_old:(g + 1) * hidden_old, :]
                scores += rows.abs().mean(dim=0).to(scores.dtype)

    first_linear = model.mlp[0]
    if isinstance(first_linear, torch.nn.Linear):
        scores += first_linear.weight.detach().abs().mean(dim=0).to(scores.dtype)

    keep = torch.topk(scores, k=keep_hidden, largest=True).indices
    keep, _ = torch.sort(keep)
    return keep.long()


def _copy_gru_hidden_shrink_weights(old_model: torch.nn.Module, new_model: torch.nn.Module, keep_idx: torch.Tensor) -> None:
    old_gru = old_model.gru
    new_gru = new_model.gru
    hidden_old = int(old_gru.hidden_size)
    num_layers = int(old_gru.num_layers)
    keep = [int(i) for i in keep_idx.tolist()]
    rows = _gate_row_indices(hidden_old, keep_idx, 3)

    for layer in range(num_layers):
        old_w_ih = getattr(old_gru, f"weight_ih_l{layer}").data
        new_w_ih = getattr(new_gru, f"weight_ih_l{layer}").data
        if layer == 0:
            in_cols = list(range(old_w_ih.shape[1]))
        else:
            in_cols = keep
        new_w_ih.copy_(old_w_ih[rows][:, in_cols])

        old_w_hh = getattr(old_gru, f"weight_hh_l{layer}").data
        new_w_hh = getattr(new_gru, f"weight_hh_l{layer}").data
        new_w_hh.copy_(old_w_hh[rows][:, keep])

        old_b_ih = getattr(old_gru, f"bias_ih_l{layer}").data
        new_b_ih = getattr(new_gru, f"bias_ih_l{layer}").data
        new_b_ih.copy_(old_b_ih[rows])

        old_b_hh = getattr(old_gru, f"bias_hh_l{layer}").data
        new_b_hh = getattr(new_gru, f"bias_hh_l{layer}").data
        new_b_hh.copy_(old_b_hh[rows])

    old_fc1 = old_model.mlp[0]
    new_fc1 = new_model.mlp[0]
    new_fc1.weight.data.copy_(old_fc1.weight.data[:, keep])
    new_fc1.bias.data.copy_(old_fc1.bias.data)

    for idx in (3,):
        old_fc = old_model.mlp[idx]
        new_fc = new_model.mlp[idx]
        if isinstance(old_fc, torch.nn.Linear) and isinstance(new_fc, torch.nn.Linear):
            new_fc.weight.data.copy_(old_fc.weight.data)
            new_fc.bias.data.copy_(old_fc.bias.data)


def optimize_soc() -> dict[str, Any]:
    SOC_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pruned_dir = SOC_OUT_ROOT / "pruned_model"
    quant_dir = SOC_OUT_ROOT / "quantized_model"

    train_mod = load_module("soc_train_mod_170", SOC_TRAIN_PY)
    cfg = yaml.safe_load((SOC_MODEL_DIR / "train_soc.yaml").read_text(encoding="utf-8"))
    ckpt_path = find_best_rmse_checkpoint(SOC_MODEL_DIR)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)

    old_hidden = int(cfg["model"]["hidden_size"])
    new_hidden = max(1, int(round(old_hidden * (1.0 - PRUNE_AMOUNT))))
    model = train_mod.GRUMLP(
        in_features=len(cfg["model"]["features"]),
        hidden_size=old_hidden,
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).cpu().eval()
    model.load_state_dict(state_dict)

    keep_idx = _pick_gru_keep_indices(model, keep_hidden=new_hidden)
    pruned_cfg = copy.deepcopy(cfg)
    pruned_cfg["model"]["hidden_size"] = int(new_hidden)
    pruned_model = train_mod.GRUMLP(
        in_features=len(pruned_cfg["model"]["features"]),
        hidden_size=new_hidden,
        mlp_hidden=int(pruned_cfg["model"]["mlp_hidden"]),
        num_layers=int(pruned_cfg["model"].get("num_layers", 1)),
        dropout=float(pruned_cfg["model"].get("dropout", 0.05)),
    ).cpu().eval()
    _copy_gru_hidden_shrink_weights(model, pruned_model, keep_idx)

    (pruned_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (pruned_dir / "config").mkdir(parents=True, exist_ok=True)
    (pruned_dir / "scripts").mkdir(parents=True, exist_ok=True)
    pruned_ckpt = pruned_dir / "checkpoints" / "soc_best_model_pruned.pt"
    torch.save({"model_state_dict": pruned_model.state_dict(), "config": pruned_cfg}, pruned_ckpt)
    with open(pruned_dir / "config" / "train_soc.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(pruned_cfg, f, sort_keys=False)
    shutil.copy2(SOC_TRAIN_PY, pruned_dir / "scripts" / "train_soc.py")
    shutil.copy2(SOC_MODEL_DIR / "scaler_robust.joblib", pruned_dir / "scaler_robust.joblib")
    if (SOC_MODEL_DIR / "export_manifest.json").exists():
        shutil.copy2(SOC_MODEL_DIR / "export_manifest.json", pruned_dir / "export_manifest.json")
    prune_meta = {
        "model_dir": str(SOC_MODEL_DIR),
        "source_checkpoint": str(ckpt_path),
        "pruned_checkpoint": str(pruned_ckpt),
        "amount": PRUNE_AMOUNT,
        "mode": "structured",
        "structured_kind": "gru_hidden_shrink",
        "old_hidden_size": old_hidden,
        "new_hidden_size": new_hidden,
        "keep_indices": [int(x) for x in keep_idx.tolist()],
    }
    write_json(pruned_dir / "prune_meta.json", prune_meta)

    qmodel = torch.ao.quantization.quantize_dynamic(
        pruned_model,
        {torch.nn.GRU, torch.nn.Linear},
        dtype=torch.qint8,
    )
    (quant_dir / "config").mkdir(parents=True, exist_ok=True)
    (quant_dir / "scripts").mkdir(parents=True, exist_ok=True)
    qstate_path = quant_dir / "quantized_state_dict.pt"
    torch.save(qmodel.state_dict(), qstate_path)
    with open(quant_dir / "config" / "train_soc.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(pruned_cfg, f, sort_keys=False)
    shutil.copy2(SOC_TRAIN_PY, quant_dir / "scripts" / "train_soc.py")
    shutil.copy2(SOC_MODEL_DIR / "scaler_robust.joblib", quant_dir / "scaler_robust.joblib")
    quant_meta = {
        "model_dir": str(pruned_dir),
        "checkpoint": str(pruned_ckpt),
        "model_type": pruned_cfg["model"].get("type"),
        "quantized_modules": ["GRU", "Linear"],
        "quant_mode": "dynamic",
        "quant_scope": "full",
    }
    write_json(quant_dir / "quantize_meta.json", quant_meta)

    summary = {
        "base_checkpoint": str(ckpt_path),
        "old_hidden_size": old_hidden,
        "new_hidden_size": new_hidden,
        "base_checkpoint_bytes": int(ckpt_path.stat().st_size),
        "pruned_checkpoint_bytes": int(pruned_ckpt.stat().st_size),
        "quantized_state_dict_bytes": int(qstate_path.stat().st_size),
    }
    write_json(SOC_OUT_ROOT / "summary.json", summary)
    return summary


def optimize_soh() -> dict[str, Any]:
    SOH_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pruned_dir = SOH_OUT_ROOT / "pruned_model"
    quant_dir = SOH_OUT_ROOT / "quantized_model"

    prune_mod = load_module("soh_prune_mod", SOH_PRUNE_PY)
    quant_mod = load_module("soh_quant_mod", SOH_QUANT_PY)

    prune_mod.prune_checkpoint(
        model_dir=SOH_MODEL_DIR,
        out_dir=pruned_dir,
        amount=PRUNE_AMOUNT,
        ckpt_path=None,
        mode="structured",
        min_hidden_channels=32,
        round_to=1,
    )
    pruned_ckpt = pruned_dir / "checkpoints" / "best_model_pruned.pt"
    quant_mod.run_quantize(
        model_dir=pruned_dir,
        out_dir=quant_dir,
        ckpt_path=pruned_ckpt,
    )

    base_ckpt = find_best_rmse_checkpoint(SOH_MODEL_DIR / "checkpoints")
    qstate_path = quant_dir / "quantized_state_dict.pt"
    prune_meta = json.loads((pruned_dir / "prune_meta.json").read_text(encoding="utf-8"))
    summary = {
        "base_checkpoint": str(base_ckpt),
        "old_hidden_size": int(prune_meta["old_hidden_size"]),
        "new_hidden_size": int(prune_meta["new_hidden_size"]),
        "base_checkpoint_bytes": int(base_ckpt.stat().st_size),
        "pruned_checkpoint_bytes": int(pruned_ckpt.stat().st_size),
        "quantized_state_dict_bytes": int(qstate_path.stat().st_size),
    }
    write_json(SOH_OUT_ROOT / "summary.json", summary)
    return summary


def main() -> None:
    soc = optimize_soc()
    soh = optimize_soh()
    combined = {"soc_1.7.0.0": soc, "soh_0.1.2.5_base_h160": soh}
    write_json(ROOT / "tools" / "model_optimization" / "latest_struct30_int8_summary.json", combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
