#!/usr/bin/env python3
"""
Hardware test: replay MGFarm_18650_C11 timeseries to STM32DC devboard and log STM32 SOH.

Reads df_FE_C11.parquet (feature-engineered) and streams a subset of:
  - Voltage[V]  (scaled to pack voltage = cell_voltage * pack_cells)
  - Current[A]
  - Temperature[*C]

Over USART3 (ST-LINK VCP) using the existing protocol (cmd 7 + cmd 6).

Outputs:
  - CSV with inputs + STM32 outputs
  - Plot PNG
  - metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing/invalid dependencies (pyarrow/matplotlib). "
        "Use the provided venv runner: run_hwtest_c11.ps1"
    ) from exc

try:
    import serial  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyserial. Install with: pip install pyserial") from exc

from send_sim_measurements import (
    build_get_state_estimations_packet,
    build_packet,
    read_frame_ascii,
)


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE"
)


def _find_repo_root(start: Path) -> Path:
    # Find repo root by locating a parent directory that contains `DL_Models`.
    for p in [start] + list(start.parents):
        if (p / "DL_Models").exists():
            return p
    return start


def _find_col(columns: list[str], candidates: list[str], regexes: list[str]) -> str:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    for pattern in regexes:
        rx = re.compile(pattern, re.IGNORECASE)
        for c in columns:
            if rx.fullmatch(c) or rx.search(c):
                return c
    raise KeyError(f"Could not find required column. Candidates={candidates} regexes={regexes}")


@dataclass
class InputRow:
    pack_v: float
    current_a: float
    temp_c: float
    efc: float | None
    q_c: float | None
    soh_true: float | None


def load_c11_inputs(
    data_root: Path,
    cell_id: str,
    *,
    pack_cells: int,
    max_points: int,
    row_stride: int,
) -> tuple[int, Iterator[InputRow]]:
    parquet = data_root / f"df_FE_{cell_id}.parquet"
    if not parquet.exists():
        # Support alternate naming like df_FE_C11.parquet
        parquet = data_root / f"df_FE_C{cell_id[-2:]}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet}")

    pf = pq.ParquetFile(parquet)
    cols = list(pf.schema.names)

    col_v = _find_col(cols, ["Voltage[V]"], [r".*Voltage.*\[V\].*"])
    col_i = _find_col(cols, ["Current[A]"], [r".*Current.*\[A\].*"])
    col_t = _find_col(
        cols,
        ["Temperature[°C]", "Temperature[Â°C]", "Temperature[¶øC]", "Temperature[Ç'¶øC]"],
        [r".*Temperature.*C.*"],
    )

    col_efc = None
    for c in ["EFC", "efc"]:
        if c in cols:
            col_efc = c
            break

    col_qc = None
    for c in ["Q_c", "q_c", "Qc", "Q_C"]:
        if c in cols:
            col_qc = c
            break

    col_soh = None
    for c in ["SOH", "soh", "SoH"]:
        if c in cols:
            col_soh = c
            break

    needed_cols = [col_v, col_i, col_t]
    if col_efc:
        needed_cols.append(col_efc)
    if col_qc:
        needed_cols.append(col_qc)
    if col_soh:
        needed_cols.append(col_soh)

    stride = max(1, int(row_stride))
    max_points_i = int(max_points)

    if max_points_i > 0:
        n_emit = max_points_i
    else:
        n_emit = int((pf.metadata.num_rows + (stride - 1)) // stride)

    def _iter() -> Iterator[InputRow]:
        emitted = 0
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=needed_cols)

            def _col(name: str) -> np.ndarray:
                arr = table[name].combine_chunks().to_numpy(zero_copy_only=False)
                return np.asarray(arr)

            v = _col(col_v).astype(np.float64, copy=False)
            i = _col(col_i).astype(np.float64, copy=False)
            t = _col(col_t).astype(np.float64, copy=False)
            efc = _col(col_efc).astype(np.float64, copy=False) if col_efc else None
            qc = _col(col_qc).astype(np.float64, copy=False) if col_qc else None
            s = _col(col_soh).astype(np.float64, copy=False) if col_soh else None

            for k in range(0, len(v), stride):
                if max_points_i > 0 and emitted >= max_points_i:
                    return
                yield InputRow(
                    pack_v=float(v[k]) * float(pack_cells),
                    current_a=float(i[k]),
                    temp_c=float(t[k]),
                    efc=float(efc[k]) if efc is not None else None,
                    q_c=float(qc[k]) if qc is not None else None,
                    soh_true=float(s[k]) if s is not None else None,
                )
                emitted += 1

    return n_emit, _iter()


def _parse_soh(line: str) -> float | None:
    m = re.search(r"SoH:([0-9eE+\\-\\.]+)", line)
    return float(m.group(1)) if m else None


def _parse_ts(line: str) -> int | None:
    m = re.search(r"ts:([0-9]+)", line)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COM port (ST-LINK VCP), e.g. COM8")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cell", default="C11", help="Cell id suffix, e.g. C11")
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--pack-cells", type=int, default=4, help="How many series cells in the BMS pack")
    ap.add_argument("--max-points", type=int, default=5000, help="Max points to stream (0=all)")
    ap.add_argument("--row-stride", type=int, default=1, help="Downsample by taking every Nth row from parquet")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="Send rate [Hz] (set 0 for fastest possible)")
    ap.add_argument("--ack-timeout-s", type=float, default=0.0, help="Timeout to wait for cmd7 ACK [s] (per sample). Set >0 only if your firmware sends ACKs.")
    ap.add_argument("--settle-ms", type=float, default=50.0, help="Wait time before polling cmd6 [ms] (needed for CM7 compute). Only used on polling steps.")
    ap.add_argument("--poll-every", type=int, default=1, help="Poll cmd6 every N samples (3600 for hourly model)")
    ap.add_argument("--expected-ts-offset-ms", type=int, default=0, help="When waiting for ts, expect ts_ms + offset (-1000 for hourly model)")
    ap.add_argument("--wait-for-ts", action="store_true", help="Wait until returned ts matches injected timestamp (robust)")
    ap.add_argument("--max-wait-ms", type=float, default=200.0, help="Max wait for a matching ts [ms]")
    ap.add_argument("--poll-interval-ms", type=float, default=5.0, help="Polling interval while waiting [ms]")
    ap.add_argument("--no-plot", action="store_true", help="Skip PNG generation (recommended for very long runs)")
    ap.add_argument("--plot-max-points", type=int, default=50000, help="Max points plotted (downsample if needed)")
    ap.add_argument("--flush-every", type=int, default=200, help="Flush CSV to disk every N rows (0=never)")
    ap.add_argument("--progress-every", type=int, default=5000, help="Print progress every N rows (0=never)")
    ap.add_argument(
        "--progress-bar",
        type=str,
        default="auto",
        choices=["auto", "tqdm", "simple", "off"],
        help=(
            "Progress display. 'tqdm' uses tqdm if installed, otherwise falls back to 'simple'. "
            "'auto' enables tqdm when available and stdout/stderr is a TTY."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: DL_Models/LFP_SOH_Optimization_Study/6_test/STM32DC/LSTM_0.1.2.3/<timestamp>)",
    )
    ap.add_argument("--poll-timeout-s", type=float, default=0.7)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    n_points, inputs = load_c11_inputs(
        data_root,
        args.cell,
        pack_cells=int(args.pack_cells),
        max_points=int(args.max_points),
        row_stride=int(args.row_stride),
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    repo_root = _find_repo_root(Path(__file__).resolve())
    default_out_root = repo_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "6_test" / "STM32DC" / "LSTM_0.1.2.3"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (default_out_root / f"HW_C11_{ts}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid matplotlib writing config into a temp folder (problematic ACLs in this environment).
    # Set early so it applies before any matplotlib import.
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))

    csv_path = out_dir / "stm32_hw_c11.csv"
    run_log = default_out_root / "RUN_LOG.txt"
    progress_path = out_dir / "progress.json"

    # Write metadata early so partial runs are still attributable.
    meta = {
        "cell": args.cell,
        "data_root": str(data_root),
        "port": args.port,
        "baud": args.baud,
        "pack_cells": args.pack_cells,
        "max_points": args.max_points,
        "row_stride": args.row_stride,
        "rate_hz": args.rate_hz,
        "rows": 0,
        "csv": str(csv_path),
        "plot": "",
        "settle_ms": float(args.settle_ms),
        "wait_for_ts": bool(args.wait_for_ts),
        "max_wait_ms": float(args.max_wait_ms),
        "poll_interval_ms": float(args.poll_interval_ms),
        "flush_every": int(args.flush_every),
        "poll_every": int(args.poll_every),
        "expected_ts_offset_ms": int(args.expected_ts_offset_ms),
        "progress_every": int(args.progress_every),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    plot_stride = 1
    if not args.no_plot and int(args.plot_max_points) > 0:
        plot_stride = max(1, int(n_points) // int(args.plot_max_points))

    plot_t: list[float] = [] if not args.no_plot else []
    plot_soh_true: list[float] = [] if not args.no_plot else []
    plot_soh_stm32: list[float] = [] if not args.no_plot else []
    rows_written = 0

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_csv:
        period_s = 0.0 if args.rate_hz <= 0 else (1.0 / max(0.1, float(args.rate_hz)))

        # Optional tqdm progress bar (nice ETA). Fallback to a lightweight single-line progress.
        progress_mode = str(args.progress_bar).lower()
        use_tqdm = False
        tqdm = None
        if progress_mode in ("auto", "tqdm"):
            want_tqdm = (progress_mode == "tqdm") or (sys.stderr.isatty() or sys.stdout.isatty())
            if want_tqdm:
                try:
                    from tqdm import tqdm as _tqdm  # type: ignore

                    tqdm = _tqdm
                    use_tqdm = True
                except Exception:
                    if progress_mode == "tqdm":
                        print("Warning: tqdm not available; falling back to simple progress.", flush=True)
                    use_tqdm = False
        if progress_mode == "off":
            use_tqdm = False

        start_wall = time.time()
        last_simple_update = 0.0
        pbar = None
        if use_tqdm and tqdm is not None:
            pbar = tqdm(total=int(n_points), unit="row", mininterval=0.5, smoothing=0.1)

        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "idx",
                "t_s",
                "ts_ms",
                "pack_v",
                "current_a",
                "temp_c",
                "efc",
                "q_c",
                "soh_stm32",
                "soh_ts_ms",
                "soh_true",
                "fw_line",
            ],
        )
        writer.writeheader()

        last_soh = float("nan")
        last_soh_ts_ms: int | None = None
        last_line = ""

        try:
            for n, sample in enumerate(inputs):
                ts_ms = int(n * 1000)
                ser.write(build_packet(sample.pack_v, sample.current_a, sample.temp_c, sample.efc, sample.q_c, ts_ms))
                ser.flush()

                # Optional cmd7 ACK (some firmwares don't send it).
                if float(args.ack_timeout_s) > 0:
                    read_frame_ascii(ser, timeout_s=float(args.ack_timeout_s))

                poll_every = max(1, int(args.poll_every))
                do_poll = (poll_every == 1) or (n > 0 and (n % poll_every) == 0)

                if do_poll:
                    # Give CM7 time to compute + publish mailbox, then poll cmd6.
                    if args.settle_ms > 0:
                        time.sleep(float(args.settle_ms) / 1000.0)
                    # Drop any remaining bytes so cmd6 response parsing stays aligned.
                    try:
                        pending = ser.in_waiting
                        if pending:
                            ser.read(pending)
                    except Exception:
                        pass

                    expected_ts = ts_ms + int(args.expected_ts_offset_ms)
                    deadline = time.time() + (float(args.max_wait_ms) / 1000.0)
                    line = ""
                    soh = float("nan")
                    while True:
                        ser.write(build_get_state_estimations_packet())
                        ser.flush()
                        _cmd, line = read_frame_ascii(ser, timeout_s=float(args.poll_timeout_s))
                        if line:
                            soh_val = _parse_soh(line)
                            if soh_val is not None:
                                soh = float(soh_val)
                            if args.wait_for_ts:
                                ts_rx = _parse_ts(line)
                                if ts_rx is None or ts_rx == expected_ts:
                                    last_soh_ts_ms = ts_rx
                                    break
                            else:
                                last_soh_ts_ms = _parse_ts(line)
                                break

                        if (not args.wait_for_ts) or time.time() >= deadline:
                            last_soh_ts_ms = _parse_ts(line) if line else last_soh_ts_ms
                            break
                        if args.poll_interval_ms > 0:
                            time.sleep(float(args.poll_interval_ms) / 1000.0)

                    last_soh = soh
                    if line:
                        last_line = line

                writer.writerow(
                    {
                        "idx": n,
                        "t_s": float(n),
                        "ts_ms": ts_ms,
                        "pack_v": sample.pack_v,
                        "current_a": sample.current_a,
                        "temp_c": sample.temp_c,
                        "efc": sample.efc if sample.efc is not None else "",
                        "q_c": sample.q_c if sample.q_c is not None else "",
                        "soh_stm32": last_soh,
                        "soh_ts_ms": last_soh_ts_ms if last_soh_ts_ms is not None else "",
                        "soh_true": sample.soh_true if sample.soh_true is not None else "",
                        "fw_line": last_line or "",
                    }
                )
                rows_written += 1

                if pbar is not None:
                    pbar.update(1)
                    if rows_written % 250 == 0:
                        pbar.set_postfix_str(f"ts_ms={ts_ms} soh={last_soh} soh_ts_ms={last_soh_ts_ms}")
                elif progress_mode in ("auto", "simple"):
                    now = time.time()
                    if (now - last_simple_update) >= 1.0:
                        last_simple_update = now
                        elapsed = max(1e-6, now - start_wall)
                        rate = rows_written / elapsed
                        eta_s = (n_points - rows_written) / rate if rate > 1e-9 else float("inf")
                        pct = 100.0 * (rows_written / max(1, n_points))
                        msg = (
                            f"\r{rows_written}/{n_points} ({pct:5.1f}%) "
                            f"elapsed={elapsed:6.1f}s eta={eta_s:6.1f}s "
                            f"ts_ms={ts_ms} soh={last_soh} soh_ts_ms={last_soh_ts_ms}"
                        )
                        print(msg, end="", flush=True)

                flush_every = int(args.flush_every)
                if flush_every > 0 and (rows_written % flush_every == 0):
                    f_csv.flush()

                progress_every = int(args.progress_every)
                if progress_every > 0 and (rows_written % progress_every == 0):
                    msg = (
                        f"progress rows={rows_written}/{n_points} ts_ms={ts_ms} "
                        f"soh={last_soh} soh_ts_ms={last_soh_ts_ms}\n"
                    )
                    print(msg, end="", flush=True)
                    progress_path.write_text(
                        json.dumps(
                            {
                                "rows_written": int(rows_written),
                                "total_rows": int(n_points),
                                "ts_ms": int(ts_ms),
                                "soh": float(last_soh) if last_soh == last_soh else None,  # NaN -> None
                                "soh_ts_ms": int(last_soh_ts_ms) if last_soh_ts_ms is not None else None,
                                "last_line": last_line,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                if not args.no_plot and (n % plot_stride == 0):
                    plot_t.append(float(n))
                    plot_soh_stm32.append(float(last_soh))
                    plot_soh_true.append(float(sample.soh_true) if sample.soh_true is not None else float("nan"))

                if period_s > 0:
                    time.sleep(period_s)
        except KeyboardInterrupt:
            print("Interrupted by user; keeping partial CSV.")
        finally:
            if pbar is not None:
                pbar.close()
            elif progress_mode in ("auto", "simple"):
                # Ensure we end the '\r' progress line cleanly.
                print("", flush=True)
            f_csv.flush()

    plot_path = out_dir / "stm32_hw_c11.png"
    if args.no_plot:
        plot_path = Path("")
    else:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        if any(np.isfinite(np.asarray(plot_soh_true, dtype=np.float64))):
            ax.plot(plot_t, plot_soh_true, label="SOH true", linewidth=0.9, alpha=0.7)
        ax.plot(plot_t, plot_soh_stm32, label="SOH stm32", linewidth=0.9, alpha=0.85)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("SOH")
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
        plot_path = out_dir / "stm32_hw_c11.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    meta["rows"] = int(rows_written)
    meta["plot"] = str(plot_path) if plot_path else ""
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Append a short run log (so we can always see what worked).
    msg = (
        f"{ts} C11 port={args.port} baud={args.baud} rows={int(rows_written)} "
        f"rate_hz={args.rate_hz} row_stride={args.row_stride} pack_cells={args.pack_cells} "
        f"wait_for_ts={bool(args.wait_for_ts)} settle_ms={args.settle_ms} out={out_dir}\n"
    )
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as f:
        f.write(msg)

    print(f"Saved: {csv_path}")
    if plot_path:
        print(f"Plot : {plot_path}")


if __name__ == "__main__":
    main()
