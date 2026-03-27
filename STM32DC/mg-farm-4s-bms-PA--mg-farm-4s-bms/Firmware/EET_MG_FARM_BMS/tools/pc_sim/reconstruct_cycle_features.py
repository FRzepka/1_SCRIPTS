#!/usr/bin/env python3
"""
Reconstruct cycle features (EFC, Q_c) from raw signals in the FE parquet.

Goal
----
We want to compute the same feature signals on the STM32 later, i.e. without
reading precomputed EFC/Q_c from the parquet.

This script reconstructs:
  - Q_c [Ah]: Coulomb-count since "full" (0 at full, negative when discharged)
    - q_c += I[A] * dt[s] / 3600
    - clamp to <= 0  (never positive)
    - reset to 0 when U >= U_max (drift protection, "full reached")
    - optionally clamp lower bound to -capacity_ref_ah
  - EFC [-]: Equivalent full cycles as cumulative absolute throughput:
    - efc += |I| * dt / 3600 / capacity_ref_ah

It can also compare the reconstructed features against the parquet columns
(`EFC` and `Q_c`) to validate correctness.

This is designed to be easy to port to C later (simple state update per sample).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyarrow") from exc


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE"
)


def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "DL_Models").exists() and (p / "STM32DC").exists():
            return p
    return start


@dataclass
class FeatureState:
    cap_ref_qc_ah: float
    cap_ref_efc_ah: float
    u_max_v: float
    q_c_ah: float = 0.0
    efc: float = 0.0
    t_prev_s: float | None = None

    def update(self, t_s: float, v_v: float, i_a: float) -> tuple[float, float]:
        if self.t_prev_s is None:
            dt_s = 0.0
        else:
            dt_s = t_s - self.t_prev_s
            if dt_s < 0.0:
                dt_s = 0.0
        self.t_prev_s = t_s

        # Q_c: coulomb counter (Ah), clamped at <= 0.
        self.q_c_ah += (i_a * dt_s) / 3600.0

        # Drift protection: once we hit U_max, force "full" (q_c = 0).
        if v_v >= self.u_max_v:
            self.q_c_ah = 0.0

        if self.q_c_ah > 0.0:
            self.q_c_ah = 0.0

        # Optional lower clamp: in the dataset Q_c is bounded by approx -capacity_ref.
        if self.cap_ref_qc_ah > 0.0 and self.q_c_ah < (-self.cap_ref_qc_ah):
            self.q_c_ah = -self.cap_ref_qc_ah

        # EFC: cumulative absolute throughput / capacity_ref.
        if self.cap_ref_efc_ah > 0.0:
            self.efc += (abs(i_a) * dt_s) / 3600.0 / self.cap_ref_efc_ah

        return self.q_c_ah, self.efc


def _infer_u_max_from_parquet(pf: pq.ParquetFile, *, max_row_groups: int = 200) -> float | None:
    """
    Infer U_max from the dataset's own Q_c resets:
    find rows where Q_c == 0, current > 0, previous Q_c < -0.05 and take median voltage.
    """
    cols = set(pf.schema.names)
    if not {"Voltage[V]", "Current[A]", "Q_c"}.issubset(cols):
        return None

    vs: list[float] = []
    prev_q: float | None = None
    for rg in range(min(pf.num_row_groups, max_row_groups)):
        df = pf.read_row_group(rg, columns=["Voltage[V]", "Current[A]", "Q_c"]).to_pandas()
        v = df["Voltage[V]"].to_numpy(dtype=np.float64, copy=False)
        i = df["Current[A]"].to_numpy(dtype=np.float64, copy=False)
        q = df["Q_c"].to_numpy(dtype=np.float64, copy=False)
        if len(q) == 0:
            continue
        if prev_q is None:
            prev_q = float(q[0])

        idxs = np.where((q == 0.0) & (i > 0.1))[0]
        for idx in idxs.tolist():
            pqv = prev_q if idx == 0 else float(q[idx - 1])
            if math.isfinite(pqv) and pqv < -0.05:
                vv = float(v[idx])
                if math.isfinite(vv):
                    vs.append(vv)
        prev_q = float(q[-1])
        if len(vs) >= 50:
            break

    if not vs:
        return None
    # The dataset uses a hard threshold: Q_c is forced to 0 as soon as U crosses that value.
    # Use the minimum observed transition voltage (robust against later plateau values).
    return float(np.min(np.asarray(vs, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="C11")
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--max-rows", type=int, default=200000, help="0 = all rows")
    ap.add_argument("--row-stride", type=int, default=1, help="Downsample by taking every Nth row")
    ap.add_argument("--u-max-v", type=float, default=float("nan"), help="Force U_max in volts (default: infer from parquet)")
    ap.add_argument(
        "--cap-ref-mode",
        type=str,
        default="capacity_first",
        choices=["capacity_first", "capacity_median", "efc_fit"],
        help=(
            "How to choose capacity_ref_ah used for EFC/Q_c scaling. "
            "capacity_first=first Capacity[Ah] value (simple), "
            "capacity_median=median of first 50k rows (stable), "
            "efc_fit=fit capacity_ref_ah from the parquet's EFC column (best match; for validation)."
        ),
    )
    ap.add_argument("--cap-ref-ah", type=float, default=float("nan"), help="Override reference capacity in Ah")
    ap.add_argument(
        "--qc-cap-ref-ah",
        type=float,
        default=float("nan"),
        help="Override Q_c clamp capacity in Ah (default: infer from Q_c min if present, else use cap-ref-ah/mode)",
    )
    ap.add_argument("--out-dir", type=str, default="", help="Output folder (default: DL_Models/.../6_test/STM32DC/LSTM_0.1.2.3/FEATURE_RECON_*)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    parquet = data_root / f"df_FE_{args.cell}.parquet"
    if not parquet.exists():
        parquet = data_root / f"df_FE_C{args.cell[-2:]}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet}")

    pf = pq.ParquetFile(parquet)
    total_rows = int(pf.metadata.num_rows)
    stride = max(1, int(args.row_stride))
    max_rows = int(args.max_rows)
    if max_rows <= 0:
        want_rows = total_rows
    else:
        want_rows = min(total_rows, max_rows)

    cols = set(pf.schema.names)
    need = {"Testtime[s]", "Voltage[V]", "Current[A]"}
    if not need.issubset(cols):
        raise KeyError(f"Missing required columns in parquet. Need {sorted(need)}. Have {sorted(cols)}")

    # Capacity reference for EFC (scaling).
    cap_ref_efc_ah = float(args.cap_ref_ah)
    if not math.isfinite(cap_ref_efc_ah):
        if args.cap_ref_mode in ("capacity_first", "capacity_median") and "Capacity[Ah]" in cols:
            # Read a prefix of Capacity[Ah] and use either the first value or a robust median.
            prefix_n = min(total_rows, 50000)
            df_cap = pq.read_table(parquet, columns=["Capacity[Ah]"]).to_pandas().head(prefix_n)
            cap_vals = df_cap["Capacity[Ah]"].to_numpy(dtype=np.float64, copy=False)
            cap_vals = cap_vals[np.isfinite(cap_vals) & (cap_vals > 0)]
            if len(cap_vals) == 0:
                cap_ref_efc_ah = float("nan")
            elif args.cap_ref_mode == "capacity_median":
                cap_ref_efc_ah = float(np.median(cap_vals))
            else:
                cap_ref_efc_ah = float(cap_vals[0])
        elif args.cap_ref_mode == "efc_fit" and {"EFC", "Capacity[Ah]"}.issubset(cols):
            # Best-match mode: fit capacity_ref_ah so absAh/cap_ref matches parquet EFC.
            # This is purely to validate our STM32-intended algorithm against the dataset.
            prefix_n = min(total_rows, 200000)
            df_fit = pq.read_table(parquet, columns=["Testtime[s]", "Current[A]", "EFC"]).to_pandas().head(prefix_n)
            t = df_fit["Testtime[s]"].to_numpy(dtype=np.float64, copy=False)
            i = df_fit["Current[A]"].to_numpy(dtype=np.float64, copy=False)
            efc_ref = df_fit["EFC"].to_numpy(dtype=np.float64, copy=False)
            dt = np.diff(t, prepend=t[0])
            dt = np.where(dt < 0, 0, dt)
            cum_ah = np.cumsum(np.abs(i) * dt / 3600.0)
            mask = np.isfinite(efc_ref) & (efc_ref > 0.1) & np.isfinite(cum_ah) & (cum_ah > 0)
            if int(np.sum(mask)) >= 100:
                cap_est = np.median(cum_ah[mask] / efc_ref[mask])
                cap_ref_efc_ah = float(cap_est)
            else:
                cap_ref_efc_ah = float("nan")
        else:
            cap_ref_efc_ah = float("nan")
    if not math.isfinite(cap_ref_efc_ah) or cap_ref_efc_ah <= 0.0:
        raise ValueError("Could not infer cap_ref_efc_ah. Provide --cap-ref-ah explicitly.")

    # Capacity reference for Q_c clamp: default to min(Q_c) if available (matches dataset behavior).
    cap_ref_qc_ah = float(args.qc_cap_ref_ah)
    if not math.isfinite(cap_ref_qc_ah):
        if "Q_c" in cols:
            df0 = pq.read_table(parquet, columns=["Q_c"]).to_pandas().head(200000)
            cap_ref_qc_ah = float(-np.nanmin(df0["Q_c"].to_numpy(dtype=np.float64)))
        else:
            cap_ref_qc_ah = cap_ref_efc_ah
    if not math.isfinite(cap_ref_qc_ah) or cap_ref_qc_ah <= 0.0:
        raise ValueError("Could not infer cap_ref_qc_ah. Provide --qc-cap-ref-ah explicitly.")

    u_max_v = float(args.u_max_v)
    if not math.isfinite(u_max_v):
        inferred = _infer_u_max_from_parquet(pf)
        if inferred is None:
            # Reasonable fallback for LFP: 3.60V.
            u_max_v = 3.60
        else:
            u_max_v = inferred

    ts = time.strftime("%Y%m%d_%H%M%S")
    repo_root = _find_repo_root(Path(__file__).resolve())
    default_out_root = repo_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "6_test" / "STM32DC" / "LSTM_0.1.2.3"
    out_dir = Path(args.out_dir) if args.out_dir else (default_out_root / f"FEATURE_RECON_{args.cell}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid matplotlib writing config into temp folders (ACL issues).
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))

    meta = {
        "cell": args.cell,
        "parquet": str(parquet),
        "total_rows": total_rows,
        "used_rows": want_rows,
        "row_stride": stride,
        "cap_ref_qc_ah": cap_ref_qc_ah,
        "cap_ref_efc_ah": cap_ref_efc_ah,
        "u_max_v": u_max_v,
    }

    state = FeatureState(cap_ref_qc_ah=cap_ref_qc_ah, cap_ref_efc_ah=cap_ref_efc_ah, u_max_v=u_max_v)

    # Stats vs parquet columns (if present).
    has_ref = {"EFC", "Q_c"}.issubset(cols)
    abs_err_q: list[float] = []
    abs_err_e: list[float] = []

    # Write a lightweight CSV (downsampled by row_stride).
    out_csv = out_dir / "reconstructed_cycle_features.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("idx,Testtime_s,Voltage_V,Current_A,Q_c_rec_Ah,EFC_rec")
        if has_ref:
            f.write(",Q_c_ref_Ah,EFC_ref,Q_c_abs_err,EFC_abs_err")
        f.write("\n")

        written = 0
        idx_global = 0
        remaining = want_rows

        for rg in range(pf.num_row_groups):
            if remaining <= 0:
                break

            df = pf.read_row_group(
                rg,
                columns=[c for c in ["Testtime[s]", "Voltage[V]", "Current[A]", "Q_c", "EFC"] if c in cols],
            ).to_pandas()

            if len(df) == 0:
                continue

            # Trim to overall requested window.
            if remaining < len(df):
                df = df.iloc[:remaining].reset_index(drop=True)

            t = df["Testtime[s]"].to_numpy(dtype=np.float64, copy=False)
            v = df["Voltage[V]"].to_numpy(dtype=np.float64, copy=False)
            i = df["Current[A]"].to_numpy(dtype=np.float64, copy=False)

            q_ref = df["Q_c"].to_numpy(dtype=np.float64, copy=False) if "Q_c" in df.columns else None
            e_ref = df["EFC"].to_numpy(dtype=np.float64, copy=False) if "EFC" in df.columns else None

            for k in range(0, len(df), stride):
                qc, efc = state.update(float(t[k]), float(v[k]), float(i[k]))

                if has_ref and q_ref is not None and e_ref is not None:
                    qerr = abs(float(q_ref[k]) - float(qc))
                    eerr = abs(float(e_ref[k]) - float(efc))
                    abs_err_q.append(qerr)
                    abs_err_e.append(eerr)
                    f.write(
                        f"{idx_global},{t[k]},{v[k]},{i[k]},{qc},{efc},{q_ref[k]},{e_ref[k]},{qerr},{eerr}\n"
                    )
                else:
                    f.write(f"{idx_global},{t[k]},{v[k]},{i[k]},{qc},{efc}\n")

                idx_global += stride
                written += 1

            remaining -= len(df)

    if abs_err_q:
        meta["q_c_mae"] = float(np.mean(abs_err_q))
        meta["q_c_max_abs_err"] = float(np.max(abs_err_q))
    if abs_err_e:
        meta["efc_mae"] = float(np.mean(abs_err_e))
        meta["efc_max_abs_err"] = float(np.max(abs_err_e))
    (out_dir / "reconstruct_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Optional plot (only if matplotlib is available).
    if (not args.no_plot) and has_ref:
        try:
            import pandas as pd
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            dfp = pd.read_csv(out_csv)
            t_s = dfp["Testtime_s"]
            fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

            ax[0].plot(t_s, dfp["Q_c_ref_Ah"], label="Q_c ref", linewidth=0.9, alpha=0.75)
            ax[0].plot(t_s, dfp["Q_c_rec_Ah"], label="Q_c reconstructed", linewidth=0.9, alpha=0.85)
            ax[0].set_ylabel("Q_c [Ah]")
            ax[0].grid(True, alpha=0.2)
            ax[0].legend(loc="best")

            ax[1].plot(t_s, dfp["EFC_ref"], label="EFC ref", linewidth=0.9, alpha=0.75)
            ax[1].plot(t_s, dfp["EFC_rec"], label="EFC reconstructed", linewidth=0.9, alpha=0.85)
            ax[1].set_ylabel("EFC [-]")
            ax[1].set_xlabel("Testtime [s]")
            ax[1].grid(True, alpha=0.2)
            ax[1].legend(loc="best")

            out_png = out_dir / "reconstructed_cycle_features.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            meta["plot"] = str(out_png)
            (out_dir / "reconstruct_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    print(f"Out: {out_dir}")
    print(f"CSV: {out_csv}")
    if abs_err_q or abs_err_e:
        print(f"Summary: {out_dir / 'reconstruct_summary.json'}")


if __name__ == "__main__":
    main()
