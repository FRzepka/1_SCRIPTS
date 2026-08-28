#!/usr/bin/env python3
"""Remove sparse NUL runs from completed measurement CSV files atomically."""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        raw = path.read_bytes()
        cleaned = raw.replace(b"\x00", b"")
        if cleaned == raw:
            print(f"unchanged: {path}")
            continue
        text = cleaned.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        if not rows or any(not row.get("round", "").isdigit() for row in rows):
            raise ValueError(f"Refusing to replace malformed CSV: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(cleaned)
        os.replace(temporary, path)
        print(f"cleaned: {path} ({raw.count(bytes([0]))} NUL bytes, {len(rows)} rows)")


if __name__ == "__main__":
    main()
