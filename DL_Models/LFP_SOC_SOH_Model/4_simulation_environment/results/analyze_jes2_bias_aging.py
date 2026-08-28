from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))
from robustness_common import load_cell_dataframe


MODELS = ("DM", "HDM", "HECM", "DD")
STATES = ("fresh", "mid_life", "aged")


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        ((frame.model == "DM") & frame.soh_condition.fillna("none").eq("none"))
        | ((frame.model != "DM") & frame.soh_condition.eq("lstm_h1"))
    ].copy()


def load_negative_runs(path: Path) -> pd.DataFrame:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["runs"]:
        if not np.isclose(abs(float(record["current_offset_pct"])), 0.03):
            continue
        summary = json.loads((Path(record["out_dir"]) / "summary.json").read_text(encoding="utf-8"))
        rows.append({
            "cell": record["cell"],
            "window_id": record["window_id"],
            "window_soh_state": record["soh_state"],
            "model": record["model"],
            "direction": "negative_3pct",
            "mae": float(summary["mae"]),
        })
    return pd.DataFrame(rows)


def charge_history(base: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows = []
    windows = base[(base.alias == "baseline") & (base.model == "DM")].drop_duplicates("window_id")
    for row in windows.itertuples(index=False):
        frame = load_cell_dataframe(str(data_root), row.cell, int(row.start_row), int(row.max_rows))
        frame = frame.dropna(subset=["Testtime[s]", "Current[A]"]).reset_index(drop=True)
        time_s = frame["Testtime[s]"].to_numpy(float)
        current_a = frame["Current[A]"].to_numpy(float)
        dt_s = np.diff(time_s, prepend=time_s[0])
        dt_s[dt_s < 0] = 0.0
        increments_ah = current_a * dt_s / 3600.0
        cumulative_ah = np.cumsum(increments_ah)
        rows.append({
            "cell": row.cell,
            "window_id": row.window_id,
            "window_soh_state": row.window_soh_state,
            "net_charge_ah": float(cumulative_ah[-1]),
            "absolute_net_charge_ah": float(abs(cumulative_ah[-1])),
            "throughput_ah": float(np.abs(increments_ah).sum()),
            "mean_abs_cumulative_charge_ah": float(np.abs(cumulative_ah).mean()),
            "cumulative_charge_range_ah": float(cumulative_ah.max() - cumulative_ah.min()),
            "mean_dataset_soh": float(frame.SOH.mean()) if "SOH" in frame else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate SOH and charge-history effects in the JES2 bias sweep.")
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument(
        "--negative-manifest", type=Path,
        default=SIMULATION / "campaigns/jes2_signed_bias_20260826/jes2_manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = primary(pd.read_csv(args.base_results))
    selected = base[base.alias.isin(["baseline", "current_bias_3p0pct"])].copy()
    selected["direction"] = selected.alias.map({
        "baseline": "baseline", "current_bias_3p0pct": "positive_3pct"
    })
    selected = selected[
        ["cell", "window_id", "window_soh_state", "model", "direction", "mae",
         "start_row", "max_rows"]
    ]
    negative = load_negative_runs(args.negative_manifest)
    runs = pd.concat([selected, negative], ignore_index=True)
    baseline = runs[runs.direction == "baseline"].set_index(["cell", "window_id", "model"])["mae"]
    disturbed = runs[runs.direction != "baseline"].copy()
    disturbed["delta_mae"] = disturbed.apply(
        lambda row: row.mae - baseline.loc[(row.cell, row.window_id, row.model)], axis=1
    )
    adverse = disturbed.loc[
        disturbed.groupby(["cell", "window_id", "model"]).delta_mae.idxmax()
    ].copy()

    complete_cells = sorted(
        cell for cell, group in adverse.groupby("cell")
        if set(group.window_soh_state) == set(STATES)
    )
    adverse["complete_three_state_cell"] = adverse.cell.isin(complete_cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adverse.to_csv(args.out_dir / "jes2_bias_aging_natural_cells.csv", index=False)

    complete = adverse[adverse.complete_three_state_cell]
    summary = complete.groupby(["window_soh_state", "model"], as_index=False).agg(
        mean_delta_mae=("delta_mae", "mean"),
        min_delta_mae=("delta_mae", "min"),
        max_delta_mae=("delta_mae", "max"),
        n_cells=("cell", "nunique"),
    )
    summary.to_csv(args.out_dir / "jes2_bias_aging_natural_summary.csv", index=False)

    histories = charge_history(base, args.data_root)
    merged = adverse.merge(histories, on=["cell", "window_id", "window_soh_state"], how="left")
    merged.to_csv(args.out_dir / "jes2_bias_charge_history.csv", index=False)
    correlations = []
    descriptors = (
        "absolute_net_charge_ah", "throughput_ah", "mean_abs_cumulative_charge_ah",
        "cumulative_charge_range_ah", "mean_dataset_soh",
    )
    for model, group in merged[merged.complete_three_state_cell].groupby("model"):
        for descriptor in descriptors:
            correlations.append({
                "model": model,
                "descriptor": descriptor,
                "pearson_r": float(group.delta_mae.corr(group[descriptor])),
                "n_windows": len(group),
            })
    pd.DataFrame(correlations).to_csv(args.out_dir / "jes2_bias_charge_correlations.csv", index=False)

    print(f"Complete three-state cells: {', '.join(complete_cells)}")
    print(summary.to_string(index=False))
    print("\nCorrelation with mean |cumulative signed charge|:")
    print(pd.DataFrame(correlations).query(
        "descriptor == 'mean_abs_cumulative_charge_ah'"
    )[["model", "pearson_r", "n_windows"]].to_string(index=False))


if __name__ == "__main__":
    main()
