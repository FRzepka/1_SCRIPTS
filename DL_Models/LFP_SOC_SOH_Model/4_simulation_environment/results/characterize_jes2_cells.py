from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CELLS = [
    "MGFarm_18650_C09",
    "MGFarm_18650_C13",
    "MGFarm_18650_C15",
    "MGFarm_18650_C25",
    "MGFarm_18650_C27",
    "MGFarm_18650_C29",
]

CELL_ROLES = {
    "MGFarm_18650_C09": "long-duration cycling; broad aging range",
    "MGFarm_18650_C13": "shorter trajectory; low end-of-life SOH",
    "MGFarm_18650_C15": "elevated thermal exposure; deep aging",
    "MGFarm_18650_C25": "longest-duration trajectory; moderate peak current",
    "MGFarm_18650_C27": "mild-aging control; narrow temperature range",
    "MGFarm_18650_C29": "highest thermal/current stress; deepest aging",
}

# Descriptive groups frozen from the visible gaps in holdout-cell p95 |C-rate|.
# High is intentionally a single-cell exploratory case rather than a forced tertile.
CELL_LOAD_CLASSES = {
    "C25": "low",
    "C27": "low",
    "C09": "middle",
    "C13": "middle",
    "C15": "middle",
    "C29": "high",
}


def parquet_path(data_root: Path, cell: str) -> Path:
    return data_root / f"df_FE_{cell.split('_')[-1]}.parquet"


def characterize_cell(data_root: Path, cell: str) -> dict[str, float | int | str]:
    columns = [
        "Testtime[s]",
        "Voltage[V]",
        "Current[A]",
        "Temperature[°C]",
        "SOH",
        "SOC",
        "EFC",
        "C_Rate",
    ]
    frame = pd.read_parquet(parquet_path(data_root, cell), columns=columns)
    current = frame["Current[A]"].to_numpy(dtype=np.float64)
    temperature = frame["Temperature[°C]"].to_numpy(dtype=np.float64)
    soh = frame["SOH"].to_numpy(dtype=np.float64)
    c_rate = np.abs(frame["C_Rate"].to_numpy(dtype=np.float64))
    time_s = frame["Testtime[s]"].to_numpy(dtype=np.float64)
    canonical = cell.split("_")[-1]
    finite_soh = np.isfinite(soh)
    finite_temperature = np.isfinite(temperature)
    return {
        "cell": cell,
        "cell_load_class": CELL_LOAD_CLASSES.get(canonical, "unassigned"),
        "role": CELL_ROLES.get(f"MGFarm_18650_{canonical}", "holdout cell"),
        "samples": int(len(frame)),
        "duration_h": float((time_s[-1] - time_s[0]) / 3600.0),
        "soh_min": float(np.nanmin(soh)),
        "soh_mean": float(np.nanmean(soh)),
        "soh_range": float(np.nanmax(soh) - np.nanmin(soh)),
        "soh_fresh_fraction": float(np.mean(soh[finite_soh] >= 0.90)),
        "soh_mid_life_fraction": float(np.mean((soh[finite_soh] >= 0.80) & (soh[finite_soh] < 0.90))),
        "soh_aged_fraction": float(np.mean(soh[finite_soh] < 0.80)),
        "temperature_min_c": float(np.nanmin(temperature)),
        "temperature_mean_c": float(np.nanmean(temperature)),
        "temperature_max_c": float(np.nanmax(temperature)),
        "temperature_hot_fraction": float(np.mean(temperature[finite_temperature] > 35.0)),
        "current_min_a": float(np.nanmin(current)),
        "current_max_a": float(np.nanmax(current)),
        "current_rms_a": float(np.sqrt(np.nanmean(current ** 2))),
        "abs_current_p95_a": float(np.nanpercentile(np.abs(current), 95.0)),
        "abs_c_rate_p95": float(np.nanpercentile(c_rate, 95.0)),
        "efc_max": float(np.nanmax(frame["EFC"].to_numpy(dtype=np.float64))),
    }


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(3)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize the independent JES2 holdout cells.")
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--data_root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [characterize_cell(args.data_root, cell) for cell in args.cells]
    result = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "jes2_cell_characteristics.csv", index=False)
    write_markdown(result, args.out_dir / "jes2_cell_characteristics.md")
    (args.out_dir / "jes2_cell_characteristics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
