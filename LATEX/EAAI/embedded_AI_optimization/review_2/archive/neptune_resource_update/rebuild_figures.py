"""Rebuild Neptune Figures 14/19 from audited flash and archived device records."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[2]
MODELS = ("Base", "Pruned", "Quantized")
TASKS = ("SOC", "SOH")
COLORS = {"Base": "#2ca02c", "Pruned": "#d62728", "Quantized": "#1f77b4"}
AUDIT_PATH = ROOT / "LATEX/DISS/Florian_Rzepka_Dissertation/tools/figure_7_12_flash_build_audit.json"
REFERENCE_SCRIPT = PAPER / "review_1/review_analysis/tools/generate_review_analyses.ps1"

# Keep the published full-stream accuracy inputs; do not substitute the downsampled
# cumulative-error trace or the separate Windows filter re-execution.
MAE_PP = {
    "SOC": (2.6845146, 2.3379620, 2.7911590),
    "SOH": (0.8523505, 1.4573121, 1.4103794),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flash_segments(blob: bytes) -> list[dict]:
    if blob[:6] != b"\x7fELF\x01\x01":
        raise ValueError("Expected a little-endian ELF32 firmware")
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", blob)
    result = []
    for i in range(header[10]):
        kind, _, _, physical, size, _, _, _ = struct.unpack_from(
            "<IIIIIIII", blob, header[5] + i * header[9]
        )
        if kind == 1 and size and 0x08000000 <= physical < 0x08200000:
            if physical + size > 0x08200000:
                raise ValueError("Flash load segment exceeds the device range")
            result.append({"address": hex(physical), "bytes": size})
    if not result:
        raise ValueError("No flash load segments found")
    return result


def load_records() -> list[dict]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    reference_text = REFERENCE_SCRIPT.read_text(encoding="utf-8-sig")
    records = []
    for task in TASKS:
        for i, model in enumerate(MODELS):
            entry = audit[task][model]
            elf = ROOT / entry["elf"].replace("\\", "/")
            blob = elf.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == entry["sha256"], elf
            segments = flash_segments(blob)
            flash = sum(s["bytes"] for s in segments)
            assert flash == entry["flash_bytes"]
            log = ROOT / f"DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/{task}/result_{model}.json"
            data = json.loads(log.read_text(encoding="utf-8"))
            ram = data["static_ram_bytes"] + data["max_stack_bytes"]
            assert ram == data["total_ram_bytes"]
            metrics = data["raw_metrics"]
            count = len(metrics["time_us"])
            assert count == 10000
            assert all(len(metrics[k]) == count for k in
                       ("stack_ram", "cycles", "host_latency_ms", "energy_uj", "predictions"))
            assert max(metrics["stack_ram"]) == data["max_stack_bytes"]
            assert np.isclose(np.mean(metrics["time_us"]), data["avg_time_us"])
            assert f"{MAE_PP[task][i]:.7f}" in reference_text
            count_variant = "Base" if model == "Quantized" else model
            count_elf = ROOT / audit[task][count_variant]["elf"].replace("\\", "/")
            header_name = "model_weights.h" if task == "SOC" else "model_weights_soh.h"
            header = count_elf.parents[1] / "Core/Inc" / header_name
            arrays = {name: int(size) for name, size in re.findall(
                r"^const float (LSTM_\w+|MLP_\w+)\[(\d+)\] =", header.read_text(), re.MULTILINE)}
            assert set(arrays) == {"LSTM_WEIGHT_IH", "LSTM_WEIGHT_HH", "LSTM_BIAS",
                                   "MLP_FC1_WEIGHT", "MLP_FC1_BIAS", "MLP_FC2_WEIGHT", "MLP_FC2_BIAS"}
            parameters = sum(arrays.values())
            records.append({
                "task": task, "model": model, "parameters": parameters,
                "mae_pp": MAE_PP[task][i], "flash_bytes": flash,
                "flash_KiB": flash / 1024, "static_ram_bytes": data["static_ram_bytes"],
                "max_stack_bytes": data["max_stack_bytes"], "ram_bytes": ram,
                "ram_KiB": ram / 1024, "inference_ms": data["avg_time_us"] / 1000,
                "host_latency_ms": data["avg_host_latency_ms"],
                "energy_proxy_mJ": data["avg_time_us"] * 0.5 / 1000,
                "inference_count": count,
                "elf": elf.relative_to(ROOT).as_posix(), "elf_sha256": entry["sha256"],
                "runtime_log": log.relative_to(ROOT).as_posix(), "runtime_log_sha256": digest(log),
                "parameter_header": header.relative_to(ROOT).as_posix(),
                "parameter_header_sha256": digest(header),
            })
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_utility(records: list[dict]) -> tuple[list, list, list]:
    grid = np.array([
        (a, f, r, 20 - a - f - r)
        for a in range(21) for f in range(21 - a) for r in range(21 - a - f)
    ], dtype=float) / 20
    assert grid.shape == (1771, 4)
    assert np.allclose(grid.sum(axis=1), 1)
    grid_rows, sweep_rows, summaries = [], [], []
    for task in TASKS:
        selected = [r for r in records if r["task"] == task]
        values = np.array([[r[k] for k in ("mae_pp", "flash_KiB", "ram_KiB", "energy_proxy_mJ")]
                           for r in selected])
        ratios = values / values[0]
        scores = grid @ ratios.T
        winners = scores.argmin(axis=1)
        counts = np.bincount(winners, minlength=3)
        assert counts.sum() == 1771
        for weights, scores_row, winner in zip(grid, scores, winners):
            grid_rows.append({"task": task,
                              **dict(zip(("w_accuracy", "w_flash", "w_ram", "w_energy"), weights)),
                              **dict(zip(("U_Base", "U_Pruned", "U_Quantized"), scores_row)),
                              "winner": MODELS[winner]})
        for j, metric in enumerate(("Accuracy", "Flash", "RAM", "Energy")):
            for percent in range(25, 86, 5):
                weight = percent / 100
                weights = np.full(4, (1 - weight) / 3)
                weights[j] = weight
                scores_row = ratios @ weights
                sweep_rows.append({"task": task, "metric": metric, "weight_percent": percent,
                                   "winner": MODELS[int(scores_row.argmin())]})
        for i, row in enumerate(selected):
            summaries.append({
                "task": task, "model": row["model"], "equal_weight_U": ratios[i].mean(),
                "winning_count": int(counts[i]), "winning_share_percent": counts[i] / 1771 * 100,
                "flash_saving_percent": 100 * (1 - ratios[i, 1]),
                "ram_saving_percent": 100 * (1 - ratios[i, 2]),
                "inference_change_percent": 100 * (ratios[i, 3] - 1),
                "host_latency_change_percent": 100 * (row["host_latency_ms"] / selected[0]["host_latency_ms"] - 1),
                "mae_change_pp": row["mae_pp"] - selected[0]["mae_pp"],
            })
    return grid_rows, sweep_rows, summaries


def tint(color: str, strength: float) -> tuple:
    return tuple(1 - strength * (1 - c) for c in to_rgb(color))


def save_figure(fig, name: str) -> None:
    fig.savefig(PAPER / f"{name}.pdf", metadata={"Title": f"Neptune {name}: audited resource comparison"})
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_sizes(records: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    titles = ("Exported parameter count", "Idealized parameter storage\n(all FP32 / all INT8)",
              "Firmware flash\n(available ELF builds)", "RAM usage\n(static + recorded peak stack)")
    handles = [Patch(facecolor=tint("#000000", 0.4), edgecolor="black", label="SOC (left, darker)"),
               Patch(facecolor=tint("#000000", 0.2), edgecolor="gray", label="SOH (right, lighter)")]
    for panel, (ax, title) in enumerate(zip(axes.flat, titles)):
        maximum = 0
        for ti, task in enumerate(TASKS):
            for i, record in enumerate(r for r in records if r["task"] == task):
                value = (record["parameters"], record["parameters"] * (1 if i == 2 else 4) / 1024,
                         record["flash_KiB"], record["ram_KiB"])[panel]
                maximum = max(maximum, value)
                color = COLORS[record["model"]]
                bar = ax.bar(i + (ti - 0.5) * 0.35, value, 0.35,
                             facecolor=tint(color, 0.4 if ti == 0 else 0.22),
                             edgecolor=color if ti == 0 else tint(color, 0.65), linewidth=1.4, zorder=3)
                label = f"{value:,}" if panel == 0 else f"{value:.2f}"
                ax.bar_label(bar, labels=[label], padding=3, fontsize=10)
        ax.set(title=title, ylabel="Count" if panel == 0 else "Size [KiB]",
               xticks=range(3), xticklabels=MODELS, ylim=(0, maximum * 1.25))
        ax.grid(axis="y", color="#dfe3e6", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=1)
    save_figure(fig, "gr14")


def plot_utility(sweep: list[dict], summaries: list[dict]) -> None:
    fig = plt.figure(figsize=(14.67, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.45, wspace=0.27)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14)
    for ti, task in enumerate(TASKS):
        ax = fig.add_subplot(gs[0, ti])
        for row in (r for r in sweep if r["task"] == task):
            y = ("Energy", "RAM", "Flash", "Accuracy").index(row["metric"])
            color = COLORS[row["winner"]]
            ax.scatter(row["weight_percent"], y, marker="s", s=210,
                       facecolor=tint(color, 0.4), edgecolor=color, linewidth=1.2, zorder=3)
        ax.set(title=f"({chr(97 + ti)}) {task}", yticks=range(4),
               yticklabels=("Energy", "RAM", "Flash", "Accuracy"),
               xticks=range(25, 86, 10), xlim=(22, 88), ylim=(-0.5, 3.5),
               xlabel="Weight of highlighted objective [%]")
        ax.grid(axis="x", color="#e5e8eb")
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    for i, model in enumerate(MODELS):
        values = [next(r["winning_share_percent"] for r in summaries
                       if r["task"] == task and r["model"] == model) for task in TASKS]
        color = COLORS[model]
        bars = ax.bar(np.arange(2) + (i - 1) * 0.24, values, 0.24,
                      facecolor=tint(color, 0.4), edgecolor=color, linewidth=1.3, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=12)
    ax.set(title="(c) Ranking across all 1771 weight combinations", xticks=range(2),
           xticklabels=TASKS, ylabel="Winning combinations [%]", ylim=(0, 105))
    ax.grid(axis="y", color="#e5e8eb", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(facecolor=tint(COLORS[m], 0.4), edgecolor=COLORS[m], label=m)
                        for m in MODELS], loc="lower center", ncol=3, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.025))
    save_figure(fig, "gr19")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "axes.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    records = load_records()
    grid, sweep, summaries = compute_utility(records)
    write_csv(HERE / "resource_kpis.csv", records)
    write_csv(HERE / "utility_weight_grid.csv", grid)
    write_csv(HERE / "utility_priority_sweep.csv", sweep)
    write_csv(HERE / "utility_scores.csv", summaries)
    plot_sizes(records)
    plot_utility(sweep, summaries)
    provenance = {
        "flash_audit": AUDIT_PATH.relative_to(ROOT).as_posix(), "flash_audit_sha256": digest(AUDIT_PATH),
        "accuracy_reference": REFERENCE_SCRIPT.relative_to(ROOT).as_posix(),
        "accuracy_reference_sha256": digest(REFERENCE_SCRIPT),
        "ram_definition": "static_ram_bytes + max_stack_bytes from the archived device JSON",
        "utility_precision": "Unrounded flash bytes, RAM bytes, and recorded kernel times",
        "build_identity": "Archived runtime records do not identify the executed ELF hashes",
        "parameter_count_note": "Weights and merged biases in the C exports, excluding row scales. "
                                "Quantized retains the Base topology. No approximate plot constants used.",
        "records": records, "scores": summaries,
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with ZipFile(PAPER / "Neptune_Korrekturen_20261001.zip", "w", ZIP_DEFLATED) as package:
        for path in (PAPER / "Neptune.tex", PAPER / "gr14.pdf", PAPER / "gr19.pdf", HERE / "UPLOAD.txt"):
            package.write(path, path.name)
        for name in ("gr14.png", "gr19.png"):
            package.write(HERE / name, f"PNG_alternative/{name}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
