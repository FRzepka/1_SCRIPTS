#!/usr/bin/env python3
"""Extract comparable static Flash/RAM section sizes from STM32 ELF images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FLASH_PREFIXES = (".isr_vector", ".text", ".rodata", ".ARM", ".init_array", ".fini_array", ".data")
RAM_PREFIXES = (".data", ".bss", ".noinit", "._user_heap_stack")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_size_output(text: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\.[A-Za-z0-9_.$-]+)\s+(\d+)\s+", line)
        if match:
            sections[match.group(1)] = sections.get(match.group(1), 0) + int(match.group(2))
    if not sections:
        raise ValueError("No ELF sections found in size-tool output")
    return sections


def inspect_image(model: str, elf: Path, size_tool: str) -> dict[str, object]:
    if not elf.is_file():
        raise FileNotFoundError(elf)
    completed = subprocess.run(
        [size_tool, "-A", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    )
    sections = parse_size_output(completed.stdout)
    flash = sum(size for name, size in sections.items() if name.startswith(FLASH_PREFIXES))
    ram = sum(size for name, size in sections.items() if name.startswith(RAM_PREFIXES))
    return {
        "model": model,
        "elf": str(elf.resolve()),
        "elf_sha256": sha256(elf),
        "elf_file_bytes": elf.stat().st_size,
        "flash_load_bytes": flash,
        "static_ram_bytes": ram,
        "data_bytes": sections.get(".data", 0),
        "bss_bytes": sections.get(".bss", 0),
        "heap_stack_reserved_bytes": sections.get("._user_heap_stack", 0),
        "sections": sections,
    }


def parse_image(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use MODEL=path/to/image.elf")
    model, path = value.split("=", 1)
    model = model.upper()
    if model not in {"DM", "HDM", "HECM", "DD", "DDS", "DDP"}:
        raise argparse.ArgumentTypeError(f"Unknown model: {model}")
    return model, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, type=parse_image)
    parser.add_argument("--size-tool", default="arm-none-eabi-size")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    models = [inspect_image(model, elf, args.size_tool) for model, elf in args.image]
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "size_tool": args.size_tool,
        "notes": "Static section sizes only; add measured peak stack and activation buffers separately.",
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "elf", "elf_sha256", "flash_load_bytes", "static_ram_bytes", "data_bytes", "bss_bytes", "heap_stack_reserved_bytes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in models:
            writer.writerow({name: row[name] for name in fields})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
