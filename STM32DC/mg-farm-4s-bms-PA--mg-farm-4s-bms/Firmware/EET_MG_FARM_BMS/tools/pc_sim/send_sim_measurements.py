#!/usr/bin/env python3
"""
Send simulated BMS measurements to the devboard over the ST-LINK VCP (USART3).

Firmware expects RS485-style frames:
  [id=1][cmd=7][len=16|24][payload...]

Payload (little-endian):
  float pack_voltage_v
  float current_a
  float temperature_c
  float efc              (optional; include to match parquet features)
  float q_c              (optional; include to match parquet features)
  uint32 timestamp_ms
"""

from __future__ import annotations

import argparse
import struct
import time

try:
    import serial  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyserial. Install with: pip install pyserial"
    ) from exc


RS485_ID = 1
CMD_SET_SIM_MEASUREMENTS = 7
CMD_GET_STATE_ESTIMATIONS_AS_STRING = 6


def build_packet(
    pack_voltage_v: float,
    current_a: float,
    temperature_c: float,
    efc: float | None,
    q_c: float | None,
    timestamp_ms: int,
) -> bytes:
    if efc is None or q_c is None:
        payload = struct.pack("<fffI", pack_voltage_v, current_a, temperature_c, timestamp_ms)
    else:
        payload = struct.pack("<fffffI", pack_voltage_v, current_a, temperature_c, efc, q_c, timestamp_ms)
    if len(payload) > 255:
        raise ValueError("payload too large")
    header = bytes([RS485_ID, CMD_SET_SIM_MEASUREMENTS, len(payload)])
    return header + payload


def build_get_state_estimations_packet() -> bytes:
    return bytes([RS485_ID, CMD_GET_STATE_ESTIMATIONS_AS_STRING, 0])


def read_frame_ascii(ser, timeout_s: float = 0.5):
    """
    Read one firmware response frame:
      [0x80][cmd][len][ASCII...\\n]

    Note: some firmware replies have an inconsistent length byte; we therefore read the payload
    as a newline-terminated ASCII line.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b != b"\x80":
            continue
        hdr = ser.read(2)
        if len(hdr) != 2:
            continue
        cmd = hdr[0]
        _length = hdr[1]
        line = ser.readline()
        return cmd, line.decode("utf-8", errors="ignore").strip()
    return None, None


def _parse_soh(line: str) -> float | None:
    import re

    m = re.search(r"SoH:([0-9eE+\\-\\.]+)", line)
    return float(m.group(1)) if m else None


def _parse_ts(line: str) -> int | None:
    import re

    m = re.search(r"ts:([0-9]+)", line)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COM port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--voltage", type=float, default=14.8, help="Pack voltage [V]")
    ap.add_argument("--current", type=float, default=0.0, help="Current [A] (+charge, -discharge)")
    ap.add_argument("--temp", type=float, default=25.0, help="Temperature [°C]")
    ap.add_argument("--efc", type=float, default=None, help="EFC feature (optional)")
    ap.add_argument("--q-c", type=float, default=None, help="Q_c feature (optional)")
    ap.add_argument("--rate", type=float, default=2.0, help="Send rate [Hz]")
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = forever)")
    ap.add_argument("--read-ack", action="store_true", help="Read and print ACK bytes")
    ap.add_argument("--poll-est", action="store_true", help="Poll and print SoC/SoH after each send")
    ap.add_argument("--soh-only", action="store_true", help="When polling, print only SoH (no SoC)")
    ap.add_argument("--settle-ms", type=float, default=50.0, help="Wait time between cmd7 and cmd6 [ms] (needed for CM7 compute)")
    ap.add_argument("--wait-ts", action="store_true", help="Wait until returned ts matches injected timestamp (robust)")
    args = ap.parse_args()

    period_s = 1.0 / max(0.1, args.rate)

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        t0 = time.time()
        while True:
            now = time.time()
            ts_ms = int((now - t0) * 1000.0)
            pkt = build_packet(args.voltage, args.current, args.temp, args.efc, args.q_c, ts_ms)
            ser.write(pkt)
            ser.flush()

            # Optional: read cmd7 ACK (some firmwares don't send it).
            if args.read_ack:
                cmd, line = read_frame_ascii(ser, timeout_s=0.2)
                if cmd is not None:
                    print(f"RX cmd={cmd}: {line!r}")

            if args.poll_est:
                if args.settle_ms > 0:
                    time.sleep(float(args.settle_ms) / 1000.0)

                expected_ts = ts_ms
                line = ""
                if args.wait_ts:
                    deadline = time.time() + 1.0
                    while time.time() < deadline:
                        ser.write(build_get_state_estimations_packet())
                        ser.flush()
                        cmd, line = read_frame_ascii(ser, timeout_s=0.5)
                        if cmd is None or not line:
                            continue
                        ts_rx = _parse_ts(line)
                        if ts_rx is None or ts_rx == expected_ts:
                            break
                        time.sleep(0.01)
                else:
                    ser.write(build_get_state_estimations_packet())
                    ser.flush()
                    _cmd, line = read_frame_ascii(ser, timeout_s=0.5)

                if line:
                    if args.soh_only:
                        soh = _parse_soh(line)
                        print(f"{soh:.6f}" if soh is not None else line)
                    else:
                        print(line)

            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break
            time.sleep(period_s)


if __name__ == "__main__":
    main()
