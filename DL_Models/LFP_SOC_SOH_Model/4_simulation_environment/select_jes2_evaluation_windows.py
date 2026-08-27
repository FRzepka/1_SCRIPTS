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

SOH_STATES = {
    "aged": (-np.inf, 0.80),
    "mid_life": (0.80, 0.90),
    "fresh": (0.90, np.inf),
}

CELL_LOAD_CLASSES = {
    "C25": "low",
    "C27": "low",
    "C09": "middle",
    "C13": "middle",
    "C15": "middle",
    "C29": "high",
}

MEDOID_FEATURES = [
    "soh_median",
    "temperature_mean_c",
    "temperature_p95_c",
    "abs_c_rate_p95",
    "throughput_ah",
    "soc_low_fraction",
]


def parquet_path(data_root: Path, cell: str) -> Path:
    return data_root / f"df_FE_{cell.split('_')[-1]}.parquet"


def full_charge_anchors(frame: pd.DataFrame, min_separation_rows: int) -> np.ndarray:
    full = (
        frame["SOC"].to_numpy(dtype=np.float64) >= 0.98
    ) & (
        frame["Voltage[V]"].to_numpy(dtype=np.float64) >= 3.58
    )
    ends = np.flatnonzero(full & ~np.r_[full[1:], False])
    selected: list[int] = []
    for row in ends:
        if not selected or row - selected[-1] >= min_separation_rows:
            selected.append(int(row))
    return np.asarray(selected, dtype=np.int64)


def summarize_candidate(
    frame: pd.DataFrame,
    cell: str,
    start_row: int,
    primary_rows: int,
    event_rows: int,
) -> dict[str, float | int | str]:
    window = frame.iloc[start_row:start_row + primary_rows]
    current = window["Current[A]"].to_numpy(dtype=np.float64)
    temperature = window["Temperature[°C]"].to_numpy(dtype=np.float64)
    soh = window["SOH"].to_numpy(dtype=np.float64)
    soc = window["SOC"].to_numpy(dtype=np.float64)
    c_rate = np.abs(window["C_Rate"].to_numpy(dtype=np.float64))
    time_s = window["Testtime[s]"].to_numpy(dtype=np.float64)
    dt_s = np.diff(time_s, prepend=time_s[0])
    valid_dt = dt_s[(dt_s > 0.0) & np.isfinite(dt_s)]
    fallback_dt = float(np.median(valid_dt)) if len(valid_dt) else 1.0
    dt_s = np.where((dt_s > 0.0) & np.isfinite(dt_s), dt_s, fallback_dt)
    throughput_ah = float(np.sum(np.abs(current) * dt_s) / 3600.0)
    soh_median = float(np.nanmedian(soh))
    soh_state = next(
        name for name, (low, high) in SOH_STATES.items() if low <= soh_median < high
    )
    canonical = cell.split("_")[-1]
    return {
        "cell": cell,
        "canonical_cell": canonical,
        "cell_load_class": CELL_LOAD_CLASSES.get(canonical, "unassigned"),
        "soh_state": soh_state,
        "start_row": int(start_row),
        "primary_rows": int(primary_rows),
        "event_rows": int(event_rows),
        "anchor_soc": float(frame.iloc[start_row]["SOC"]),
        "anchor_voltage_v": float(frame.iloc[start_row]["Voltage[V]"]),
        "anchor_soh": float(frame.iloc[start_row]["SOH"]),
        "duration_h": float((time_s[-1] - time_s[0]) / 3600.0),
        "soh_median": soh_median,
        "soh_min": float(np.nanmin(soh)),
        "soh_max": float(np.nanmax(soh)),
        "temperature_mean_c": float(np.nanmean(temperature)),
        "temperature_p95_c": float(np.nanpercentile(temperature, 95.0)),
        "abs_c_rate_p95": float(np.nanpercentile(c_rate, 95.0)),
        "throughput_ah": throughput_ah,
        "soc_low_fraction": float(np.mean(soc < 0.20)),
        "soc_middle_fraction": float(np.mean((soc >= 0.20) & (soc <= 0.80))),
        "soc_high_fraction": float(np.mean(soc > 0.80)),
    }


def medoid_index(candidates: pd.DataFrame) -> int:
    values = candidates[MEDOID_FEATURES].to_numpy(dtype=np.float64)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    distance = np.sqrt(np.sum(((values - center) / scale) ** 2, axis=1))
    return int(np.nanargmin(distance))


def select_cell_windows(
    data_root: Path,
    cell: str,
    primary_rows: int,
    event_rows: int,
    min_separation_rows: int,
) -> tuple[list[dict], list[dict]]:
    columns = ["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "SOH", "SOC", "C_Rate"]
    frame = pd.read_parquet(parquet_path(data_root, cell), columns=columns)
    anchors = full_charge_anchors(frame, min_separation_rows)
    anchors = anchors[anchors + event_rows <= len(frame)]
    candidates = [
        summarize_candidate(frame, cell, int(anchor), primary_rows, event_rows)
        for anchor in anchors
    ]
    if not candidates:
        raise ValueError(f"No eligible full-charge window found for {cell}")
    candidate_frame = pd.DataFrame(candidates)
    selected: list[dict] = []
    for state in ["fresh", "mid_life", "aged"]:
        state_candidates = candidate_frame[candidate_frame["soh_state"] == state].reset_index(drop=True)
        if state_candidates.empty:
            continue
        row = state_candidates.iloc[medoid_index(state_candidates)].to_dict()
        row["window_id"] = f"{row['canonical_cell']}_{state}"
        row["selection_rule"] = "measured_feature_medoid_at_full_charge"
        row["candidate_count_in_state"] = int(len(state_candidates))
        selected.append(row)
    return selected, candidates


def write_markdown(selected: pd.DataFrame, path: Path) -> None:
    display_columns = [
        "window_id", "cell_load_class", "soh_state", "start_row", "primary_rows",
        "event_rows", "soh_median", "temperature_mean_c", "abs_c_rate_p95",
        "throughput_ah", "candidate_count_in_state",
    ]
    display = selected[display_columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].round(4)
    lines = [
        "# JES2 evaluation windows",
        "",
        "Windows start at a measured full-charge anchor (SOC >= 0.98 and voltage >= 3.58 V).",
        "Selection is the multivariate medoid of measured operating features within each available SOH state; estimator outputs are not used.",
        "Primary scenarios use 24 h (86400 rows); the 1 h missing-gap scenario uses 48 h (172800 rows).",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze representative, model-independent JES2 evaluation windows.")
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--data_root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path; JSON and Markdown are written alongside it.")
    parser.add_argument("--primary_rows", type=int, default=86400)
    parser.add_argument("--event_rows", type=int, default=172800)
    parser.add_argument("--min_separation_rows", type=int, default=43200)
    args = parser.parse_args()
    if args.primary_rows < 2024:
        parser.error("primary_rows must cover at least one trained DD sequence (2024 samples)")
    if args.event_rows < args.primary_rows:
        parser.error("event_rows must be at least primary_rows")

    selected: list[dict] = []
    candidates: list[dict] = []
    for cell in args.cells:
        cell_selected, cell_candidates = select_cell_windows(
            args.data_root, cell, args.primary_rows, args.event_rows, args.min_separation_rows
        )
        selected.extend(cell_selected)
        candidates.extend(cell_candidates)

    selected_frame = pd.DataFrame(selected).sort_values(["cell", "soh_median"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected_frame.to_csv(args.out, index=False)
    payload = {
        "protocol": {
            "anchor": "end of contiguous SOC >= 0.98 and Voltage >= 3.58 V segment",
            "selection": "multivariate measured-feature medoid within cell and SOH state",
            "medoid_features": MEDOID_FEATURES,
            "soh_states": {name: [float(low), float(high)] for name, (low, high) in SOH_STATES.items()},
            "primary_rows": int(args.primary_rows),
            "event_rows": int(args.event_rows),
            "min_separation_rows": int(args.min_separation_rows),
            "uses_model_outputs": False,
        },
        "windows": selected,
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(selected_frame, args.out.with_suffix(".md"))
    pd.DataFrame(candidates).to_csv(args.out.with_name(f"{args.out.stem}_candidates.csv"), index=False)
    print(selected_frame[["window_id", "cell_load_class", "soh_state", "start_row", "soh_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
