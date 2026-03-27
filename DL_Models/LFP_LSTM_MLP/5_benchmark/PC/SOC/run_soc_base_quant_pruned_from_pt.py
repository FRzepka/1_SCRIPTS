#!/usr/bin/env python
"""
SOC Streaming-Simulation (PC) – Base .pt, Quantized, Pruned
===========================================================

Verwendet GENAU die .pt-Modelle aus den Ordnern:

Base:
  DL_Models/LFP_LSTM_MLP/2_models/base/soc_1.5.0.0_base/1.5.0.0_soc_epoch0001_rmse0.02897.pt

Pruned:
  DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/soc_pruned_hidden45.pt

Quantized:
  Manuelle INT8-Gewichte wie im STM32-Paket, via manual_lstm_int8_from_pt.py
  (gleicher Quantisierungsweg wie für 2_models/quantized/soc_1.5.0.0_quantized)

Was das Script macht:
----------------------
- Lädt SOC-Daten (CFG + RobustScaler + df_FE_C07.parquet) via manual_lstm_int8_from_pt.load_data
- Berechnet gestreamte Vorhersagen:
    * BASE:   FP32 LSTM+MLP aus Base-Checkpoint
    * QUANT:  INT8-gewichtetes LSTM (manuelle Quantisierung aus Base-Checkpoint)
    * PRUNED: FP32 LSTM+MLP aus Pruned-Checkpoint (Hidden=45)
- Vergleicht alle drei mit Groundtruth SOC:
    * Overlay-Plot (GT, BASE, QUANT, PRUNED)
    * Error-Plots (BASE-GT, QUANT-GT, PRUNED-GT)
    * Error-Histogramme, Parity-Plot, Balkenplots für Fehler-Metriken
    * Balkenplot für Modellgrößen / Parameteranzahl
- Speichert alle Arrays in einer NPZ-Datei, um später für Paper-Plots genutzt zu werden.

Beispielaufruf (von Repo-Root):
-------------------------------
python DL_Models\\LFP_LSTM_MLP\\5_benchmark\\PC\\run_soc_base_quant_pruned_from_pt.py ^
  --out-dir DL_Models\\LFP_LSTM_MLP\\5_benchmark\\bench_v_soc_pt ^
  --max-steps -1
"""

import argparse
from pathlib import Path
import math

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def import_manual_quant_module():
    """manual_lstm_int8_from_pt.py aus 4_quantize importieren."""
    import importlib.util
    import sys

    here = Path(__file__).resolve()
    # .../DL_Models/LFP_LSTM_MLP/5_benchmark/PC -> parents[2] = LFP_LSTM_MLP
    lfp_root = here.parents[2]
    quant_dir = lfp_root / "4_quantize"
    sys.path.append(str(quant_dir))
    spec = importlib.util.spec_from_file_location(
        "manual_lstm_int8_from_pt", quant_dir / "manual_lstm_int8_from_pt.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def truncate_common(max_steps: int, *arrays):
    """Trunkiert alle Arrays auf dieselbe Länge <= max_steps."""
    valid = [a for a in arrays if a is not None]
    if not valid:
        return (), 0
    n = min(len(a) for a in valid)
    n = min(n, max_steps)
    return [a[:n] if a is not None else None for a in arrays], n


def plot_overlay(y, base, quant, pruned, out_png: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(y, label="GT SOC", linewidth=1.0, alpha=0.9, color="black")
    plt.plot(base, label="BASE (fp32)", linewidth=0.9, alpha=0.9, color="#1f77b4")
    if quant is not None:
        plt.plot(quant, label="QUANT (int8)", linewidth=0.9, alpha=0.9, color="#2ca02c")
    if pruned is not None:
        plt.plot(pruned, label="PRUNED (fp32)", linewidth=0.9, alpha=0.9, color="#9467bd")
    plt.xlabel("step")
    plt.ylabel("SOC")
    plt.title("SOC – GT vs BASE / QUANT / PRUNED (Streaming)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_error(y, pred, label: str, out_png: Path):
    if pred is None:
        return
    err = pred - y
    plt.figure(figsize=(12, 3))
    plt.plot(err, linewidth=0.7)
    plt.axhline(0.0, color="r", linestyle="--", alpha=0.4)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.title(f"{label} - GT (Streaming, first N steps)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def compute_error_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(float(np.mean(err ** 2))))
    max_abs = float(np.max(np.abs(err)))
    return {"MAE": mae, "RMSE": rmse, "MAX": max_abs}


def plot_metrics_bars(metrics: dict, out_png: Path):
    names = list(metrics.keys())
    mae = [metrics[n]["MAE"] for n in names]
    rmse = [metrics[n]["RMSE"] for n in names]
    mxe = [metrics[n]["MAX"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, mae, width, label="MAE")
    ax.bar(x, rmse, width, label="RMSE")
    ax.bar(x + width, mxe, width, label="MAX |err|")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Error")
    ax.set_title("SOC Error-Metriken vs. Groundtruth")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_error_histograms(y: np.ndarray, preds: dict, out_png: Path, bins: int = 100):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, pred in preds.items():
        err = pred - y
        ax.hist(
            err,
            bins=bins,
            alpha=0.4,
            label=name,
            histtype="stepfilled",
        )
    ax.set_xlabel("Error (pred - GT)")
    ax.set_ylabel("Count")
    ax.set_title("SOC error distribution (streaming, full run)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_parity(y: np.ndarray, preds: dict, out_png: Path, max_points: int = 20000):
    n = len(y)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        y_s = y[idx]
        preds_s = {k: v[idx] for k, v in preds.items()}
    else:
        y_s = y
        preds_s = preds

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
    for name, pred in preds_s.items():
        ax.scatter(
            y_s,
            pred,
            s=2,
            alpha=0.5,
            label=name,
        )
    ax.set_xlabel("GT SOC")
    ax.set_ylabel("Pred SOC")
    ax.set_title("Parity-Plot SOC (Streaming, full run)")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_model_size_bars(param_counts: dict, size_bytes: dict, out_png: Path):
    names = list(param_counts.keys())
    params_m = [param_counts[n] / 1e6 for n in names]
    size_kb = [size_bytes[n] / 1024.0 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, params_m, width)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Parameter [Mio.]")
    ax1.set_title("Parameteranzahl pro Modell")
    ax1.grid(axis="y", alpha=0.2)

    ax2.bar(x, size_kb, width, color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Größe [KB]")
    ax2.set_title("Gewichtsspeicher (geschätzt)")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-ckpt",
        default="DL_Models/LFP_LSTM_MLP/2_models/base/soc_1.5.0.0_base/1.5.0.0_soc_epoch0001_rmse0.02897.pt",
        help="Pfad zum Base-SOC-Checkpoint (.pt)",
    )
    ap.add_argument(
        "--pruned-ckpt",
        default="DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/soc_pruned_hidden45.pt",
        help="Pfad zum pruned SOC-Checkpoint (.pt)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Anzahl Samples (Streaming-Schritte) für den Vergleich; <=0 = voller Datensatz",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Ausgabeordner für Plots und Daten",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Daten + SCALER/Features via manual_lstm_int8_from_pt laden
    qmod = import_manual_quant_module()

    num_samples = args.max_steps if args.max_steps and args.max_steps > 0 else 10**9
    print(f"[info] Lade Daten via manual_lstm_int8_from_pt.load_data (num_samples={num_samples})")
    Xs, y, cfg = qmod.load_data(num_samples=num_samples)
    Xs = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    print(f"[info] Daten geladen: Xs.shape={Xs.shape}, y.shape={y.shape}")

    # 2) BASE: FP32-Streaming + QUANT: INT8-Streaming (manuelle Quantisierung)
    base_ckpt = Path(args.base_ckpt)
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base-Checkpoint nicht gefunden: {base_ckpt}")
    state_base = torch.load(base_ckpt, map_location="cpu")
    sd_base = state_base.get("model_state_dict", state_base)
    H_base = sd_base["lstm.weight_hh_l0"].shape[1]

    print(f"[info] BASE ckpt: {base_ckpt}, hidden_size={H_base}")
    preds_base, preds_quant, metrics_bq, pack_base = qmod.run_streaming_compare(
        Xs, sd_base, H_base, show_progress=True, desc="BASE+QUANT streaming"
    )
    preds_base = np.asarray(preds_base, dtype=np.float32)
    preds_quant = np.asarray(preds_quant, dtype=np.float32)
    print(f"[info] BASE/QUANT streaming fertig, metrics (BASE vs QUANT): {metrics_bq}")

    # 3) PRUNED: FP32-Streaming aus pruned-Checkpoint
    pruned_ckpt = Path(args.pruned_ckpt)
    if not pruned_ckpt.exists():
        raise FileNotFoundError(f"Pruned-Checkpoint nicht gefunden: {pruned_ckpt}")
    state_pruned = torch.load(pruned_ckpt, map_location="cpu")
    sd_pruned = state_pruned.get("model_state_dict", state_pruned)
    H_pruned = sd_pruned["lstm.weight_hh_l0"].shape[1]
    print(f"[info] PRUNED ckpt: {pruned_ckpt}, hidden_size={H_pruned}")
    preds_pruned, preds_pruned_i8, metrics_pp, pack_pruned = qmod.run_streaming_compare(
        Xs, sd_pruned, H_pruned, show_progress=True, desc="PRUNED streaming"
    )
    preds_pruned = np.asarray(preds_pruned, dtype=np.float32)
    print(f"[info] PRUNED streaming fertig (intern vs INT8): {metrics_pp}")

    # 4) Gemeinsame Trunkierung auf max_steps
    effective_max = args.max_steps if args.max_steps and args.max_steps > 0 else len(y)
    (y_t, base_t, quant_t, pruned_t), n = truncate_common(
        effective_max, y, preds_base, preds_quant, preds_pruned
    )
    if n == 0:
        raise RuntimeError("Keine gemeinsamen Schritte nach Trunkierung (n=0).")

    print(f"[info] Verwende n={n} Schritte für Overlay, Fehlerplots und Metriken")

    # 5) Plots (Overlay + Fehler)
    plot_overlay(
        y_t,
        base_t,
        quant_t,
        pruned_t,
        out_dir / "soc_streaming_overlay_firstN.png",
    )
    plot_error(
        y_t,
        base_t,
        "BASE (fp32)",
        out_dir / "soc_streaming_error_base_firstN.png",
    )
    plot_error(
        y_t,
        quant_t,
        "QUANT (int8)",
        out_dir / "soc_streaming_error_quant_firstN.png",
    )
    plot_error(
        y_t,
        pruned_t,
        "PRUNED (fp32)",
        out_dir / "soc_streaming_error_pruned_firstN.png",
    )

    # 6) Fehler-Metriken gegen Groundtruth
    metrics_gt = {
        "BASE": compute_error_metrics(y_t, base_t),
        "QUANT": compute_error_metrics(y_t, quant_t),
        "PRUNED": compute_error_metrics(y_t, pruned_t),
    }
    print("[metrics] Fehler vs. Groundtruth (full run / n-Schritte):")
    for name, m in metrics_gt.items():
        print(
            f"  {name}: MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, MAX|err|={m['MAX']:.6f}"
        )

    plot_metrics_bars(metrics_gt, out_dir / "soc_metrics_bar.png")
    plot_error_histograms(
        y_t,
        {"BASE": base_t, "QUANT": quant_t, "PRUNED": pruned_t},
        out_dir / "soc_error_hist.png",
    )
    plot_parity(
        y_t,
        {"BASE": base_t, "QUANT": quant_t, "PRUNED": pruned_t},
        out_dir / "soc_parity_plot.png",
    )

    # 7) Modellgrößen / Parameteranzahl
    def count_params(sd: dict) -> int:
        return int(
            sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        )

    def bytes_of_tensor(t: torch.Tensor) -> int:
        return int(t.numel() * 4)  # float32

    base_params = count_params(sd_base)
    pruned_params = count_params(sd_pruned)

    base_bytes = base_params * 4

    lstm_keys = [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "lstm.bias_ih_l0",
        "lstm.bias_hh_l0",
    ]
    base_lstm_bytes = sum(bytes_of_tensor(sd_base[k]) for k in lstm_keys)
    base_rest_bytes = base_bytes - base_lstm_bytes

    W_ih_q, S_ih, W_hh_q, S_hh, B = pack_base
    int8_bytes = int(W_ih_q.size + W_hh_q.size)  # 1 Byte pro Gewicht
    float_bytes = int(
        (S_ih.size + S_hh.size + B.size) * 4
    )  # Skalen + Bias als float32
    quant_lstm_bytes = int8_bytes + float_bytes
    quant_total_bytes = base_rest_bytes + quant_lstm_bytes

    pruned_bytes = pruned_params * 4

    size_bytes = {
        "BASE": base_bytes,
        "QUANT": quant_total_bytes,
        "PRUNED": pruned_bytes,
    }
    param_counts = {
        "BASE": base_params,
        "QUANT": base_params,  # gleiche Anzahl Parameter, aber andere Kodierung im LSTM
        "PRUNED": pruned_params,
    }

    print("[model] Geschätzte Modellgrößen:")
    for name in ["BASE", "QUANT", "PRUNED"]:
        print(
            f"  {name}: params={param_counts[name]:,d}, size≈{size_bytes[name]/1024.0:.1f} KB"
        )

    plot_model_size_bars(param_counts, size_bytes, out_dir / "soc_model_sizes.png")

    # 8) Daten speichern
    data_npz = out_dir / "soc_streaming_base_quant_pruned_data.npz"
    np.savez_compressed(
        data_npz,
        y=y_t,
        base=base_t,
        quant=quant_t,
        pruned=pruned_t,
    )

    print("[done] Streaming-SOC Base/Quant/Pruned Simulation (full run)")
    print(f"       Daten + Plots in: {out_dir}")
    print(f"       Daten-NPZ:        {data_npz}")


if __name__ == "__main__":
    main()

