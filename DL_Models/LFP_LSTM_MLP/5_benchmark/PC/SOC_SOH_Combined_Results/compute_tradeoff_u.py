from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kpis:
    mae_pct: float
    flash_kb: float
    ram_kb: float
    energy_uj: float


def compute_u(base: Kpis, variant: Kpis) -> float:
    return 0.25 * (
        (variant.mae_pct / base.mae_pct)
        + (variant.flash_kb / base.flash_kb)
        + (variant.ram_kb / base.ram_kb)
        + (variant.energy_uj / base.energy_uj)
    )


def _parse_float(s: str) -> float:
    s = s.replace(",", "").strip()
    return float(s)


def parse_benchmark_summary_md(path: Path) -> dict[str, dict[str, Kpis]]:
    """
    Parses the markdown tables in BENCHMARK_SUMMARY.md and returns:
      {"SOC": {"Base": Kpis(...), "Pruned": ..., "Quantized": ...},
       "SOH": {...}}
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    # Tolerate weird encoding artifacts by allowing non-ascii around numbers.
    row_re = re.compile(
        r"^\|\s*(?P<model>Base FP32|Pruned FP32|Quant INT8)\s*\|"
        r"\s*(?P<latency>[-+0-9.,]+)\s*\|"
        r"\s*(?P<infer>[-+0-9.,]+)\s*\|"
        r"\s*(?P<flash>[-+0-9.,]+)\s*\|"
        r"\s*(?P<static>[-+0-9.,]+)\s*\|"
        r"\s*(?P<stack>[-+0-9.,]+)\s*\|"
        r"\s*(?P<total>[-+0-9.,]+)\s*\|"
        r"\s*(?P<energy>[-+0-9.,]+)\s*\|",
        re.MULTILINE,
    )

    sections: dict[str, dict[str, Kpis]] = {"SOC": {}, "SOH": {}}
    current_task: str | None = None
    for line in text.splitlines():
        if line.strip().startswith("## SOC models"):
            current_task = "SOC"
            continue
        if line.strip().startswith("## SOH models"):
            current_task = "SOH"
            continue
        if current_task is None:
            continue

        # BENCHMARK_SUMMARY.md may contain encoding artefacts (e.g. "3┐?Н494.66").
        # Strip non-alphanumeric noise while keeping table structure and decimals.
        clean_line = re.sub(r"[^0-9A-Za-z|.,+ \\-]", "", line)
        m = row_re.match(clean_line)
        if not m:
            continue

        model = m.group("model")
        variant_name = {"Base FP32": "Base", "Pruned FP32": "Pruned", "Quant INT8": "Quantized"}[model]
        flash_kb = _parse_float(m.group("flash"))
        total_ram_kb = _parse_float(m.group("total"))
        energy_uj = _parse_float(m.group("energy"))
        # MAE is not in the benchmark summary tables (it comes from the streaming accuracy eval).
        sections[current_task][variant_name] = Kpis(
            mae_pct=float("nan"),
            flash_kb=flash_kb,
            ram_kb=total_ram_kb,
            energy_uj=energy_uj,
        )

    return sections


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path(__file__).with_name("BENCHMARK_SUMMARY.md"),
        help="Path to BENCHMARK_SUMMARY.md (STM32 KPIs).",
    )
    args = parser.parse_args()

    # MAE values used in the paper (streaming, in % full scale).
    mae = {
        "SOC": {"Base": 2.68, "Pruned": 2.33, "Quantized": 2.78},
        "SOH": {"Base": 0.85, "Pruned": 1.46, "Quantized": 1.41},
    }

    data = parse_benchmark_summary_md(args.summary_md)
    for task in ("SOC", "SOH"):
        for variant in ("Base", "Pruned", "Quantized"):
            if variant not in data[task]:
                raise SystemExit(f"Missing {task} {variant} row in {args.summary_md}")
            k = data[task][variant]
            data[task][variant] = Kpis(
                mae_pct=mae[task][variant],
                flash_kb=k.flash_kb,
                ram_kb=k.ram_kb,
                energy_uj=k.energy_uj,
            )

    for task in ("SOC", "SOH"):
        base = data[task]["Base"]
        print(f"\n{task}")
        print(
            f"  Base:      MAE={base.mae_pct:.2f}%  Flash={base.flash_kb:.2f}kB  "
            f"RAM={base.ram_kb:.2f}kB  Energy={base.energy_uj:.2f}uJ  U=1.00"
        )
        for variant in ("Pruned", "Quantized"):
            v = data[task][variant]
            u = compute_u(base, v)
            d_mae_pp = v.mae_pct - base.mae_pct
            d_flash_pct = 100.0 * (v.flash_kb / base.flash_kb - 1.0)
            d_ram_pct = 100.0 * (v.ram_kb / base.ram_kb - 1.0)
            d_energy_pct = 100.0 * (v.energy_uj / base.energy_uj - 1.0)
            print(
                f"  {variant:<9} MAE={v.mae_pct:.2f}% (dMAE={d_mae_pp:+.2f} pp)  "
                f"Flash={v.flash_kb:.2f}kB (dFlash={d_flash_pct:+.1f}%)  "
                f"RAM={v.ram_kb:.2f}kB (dRAM={d_ram_pct:+.1f}%)  "
                f"Energy={v.energy_uj:.2f}uJ (dEnergy={d_energy_pct:+.1f}%)  "
                f"U={u:.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
