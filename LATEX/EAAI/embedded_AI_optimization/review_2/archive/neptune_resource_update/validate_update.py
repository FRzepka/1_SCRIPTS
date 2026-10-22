"""Check replacement assets, numerical values, and preserved proof structure."""

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pymupdf as fitz

from rebuild_figures import HERE, PAPER, ROOT, load_records


def main():
    current = (PAPER / "Neptune.tex").read_text(encoding="utf-8")
    original_path = PAPER / "Neptune.before_resource_update_20261001.tex.bak"
    original = original_path.read_text(encoding="utf-8")
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == (
        "e9c8d0d61fafE40eb3ef31ee7d24217231f5ede806d8641a7f38254d7a47683a".lower())
    report = []
    for name, pattern in {
        "figure inclusions": r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}",
        "labels": r"\\xlabel\{[^}]+\}",
        "citations": r"\\cite\w*\{[^}]+\}",
        "environment boundaries": r"\\(?:begin|end)\{[^}]+\}",
        "headings": r"^\\(?:sub)*section[^\n]*",
    }.items():
        before = re.findall(pattern, original, re.MULTILINE)
        after = re.findall(pattern, current, re.MULTILINE)
        assert before == after, name
        report.append(f"Unchanged {name}: {len(after)}")

    def brace_depth(text):
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        text = re.sub(r"\\[{}]", "", text)
        depth = 0
        for character in text:
            depth += (character == "{") - (character == "}")
            assert depth >= 0
        return depth

    assert brace_depth(current) == brace_depth(original) == 0
    records = load_records()
    for record in records:
        start = current.index(r"\xlabel{tab:kpi}" if record["task"] == "SOC"
                              else r"\xlabel{tab:kpi_soh}")
        table = current[start:current.index(r"\end{table*}", start)]
        prefix = {"Base": "Base FP32", "Pruned": "Pruned FP32", "Quantized": "Quant INT8"}[record["model"]]
        line = next(line for line in table.splitlines() if line.strip().startswith(prefix + " &"))
        cells = line.split("&")
        for i, key in enumerate(("host_latency_ms", "inference_ms", "flash_KiB", "ram_KiB", "energy_proxy_mJ"), 1):
            assert cells[i].strip().startswith(f"{record[key]:.2f}"), (record["task"], prefix, key)
    report.append("All six KPI rows match source data before display rounding")

    with (HERE / "utility_scores.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    for row in scores:
        if row["model"] != "Base":
            for value, precision in (("equal_weight_U", 2), ("winning_share_percent", 2),
                                     ("flash_saving_percent", 1), ("mae_change_pp", 2)):
                number = f"{abs(float(row[value])):.{precision}f}"
                assert number in current, (row["task"], row["model"], value)
    report.append("Utility scores, weight-grid shares, flash savings, and MAE deltas appear in text")

    nm_paths = sorted(Path("C:/ST").glob("**/arm-none-eabi-nm.exe"))
    assert nm_paths, "STM32 GNU nm is needed to check exported coefficients against ELF symbols"
    parameter_audit = []
    for record in records:
        if record["model"] == "Quantized":
            continue
        symbols = subprocess.check_output([str(nm_paths[0]), "--print-size", "--defined-only", "--radix=d",
                                           str(ROOT / record["elf"])], text=True)
        coefficients = re.findall(r"^\d+\s+(\d+)\s+\w\s+((?:SOH_)?(?:LSTM_(?:WEIGHT_\w+|BIAS)|MLP_FC\w+))$",
                                  symbols, re.MULTILINE)
        assert len(coefficients) == 7
        assert sum(int(size) // 4 for size, _ in coefficients) == record["parameters"]
        parameter_audit.extend({"task": record["task"], "model": record["model"], "symbol": name,
                                "bytes": int(size), "fp32_coefficients": int(size) // 4}
                               for size, name in coefficients)
    with (HERE / "parameter_symbol_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_audit[0]))
        writer.writeheader()
        writer.writerows(parameter_audit)
    report.append("Base/Pruned parameter counts agree with all 28 array symbols in the four ELF builds")

    for name in ("gr14", "gr19"):
        with fitz.open(PAPER / (name + ".pdf")) as document:
            assert len(document) == 1
            assert document[0].get_fonts()
            text = document[0].get_text()
            if name == "gr14":
                for record in records:
                    for value in (f"{record['parameters']:,}", f"{record['flash_KiB']:.2f}", f"{record['ram_KiB']:.2f}"):
                        assert value in text, value
            else:
                for score in scores:
                    assert f"{float(score['winning_share_percent']):.2f}" in text
        report.append(f"{name}.pdf: single-page vector PDF, embedded font, numerical labels verified")

    with ZipFile(PAPER / "Neptune_Korrekturen_20261001.zip") as package:
        assert package.testzip() is None
        assert set(package.namelist()) == {"Neptune.tex", "gr14.pdf", "gr19.pdf", "UPLOAD.txt",
                                          "PNG_alternative/gr14.png", "PNG_alternative/gr19.png"}
        for name in package.namelist():
            source = HERE / Path(name).name if name == "UPLOAD.txt" or name.startswith("PNG_alternative/") else PAPER / name
            assert package.read(name) == source.read_bytes(), name
    report.append("ZIP: all six entries intact and identical to the current source files")
    report.append("Full Neptune typesetting remains unverified: proprietary neptune.cls unavailable locally")
    (HERE / "validation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
