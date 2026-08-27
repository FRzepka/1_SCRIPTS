from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ALIASES = ("baseline", "current_bias_3p0pct", "initial_soc_error", "missing_gap_1h", "voltage_spikes")
MODELS = ("DM", "HDM", "HECM", "DD")


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay selected JES2 runs with full trajectory output.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--cell", default="C29")
    parser.add_argument("--window_id", default="C29_mid_life")
    parser.add_argument("--aliases", nargs="+", choices=ALIASES, default=list(ALIASES))
    args = parser.parse_args()

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected: dict[tuple[str, str], dict] = {}
    for record in source["runs"]:
        condition_matches = (
            record["model"] == "DM" or record.get("soh_condition") == "lstm_h1"
        )
        key = (record["alias"], record["model"])
        if (
            record["cell"] == args.cell
            and record.get("window_id") == args.window_id
            and record["alias"] in args.aliases
            and record["model"] in MODELS
            and condition_matches
            and int(record["seed"]) == 42
        ):
            selected[key] = record

    expected = {(alias, model) for alias in args.aliases for model in MODELS}
    missing = expected - set(selected)
    if missing:
        raise ValueError(f"Missing representative records: {sorted(missing)}")

    output_records = []
    for index, (alias, model) in enumerate(sorted(expected)):
        record = selected[(alias, model)]
        target = args.out_dir / alias / model
        target.mkdir(parents=True, exist_ok=True)
        command = [item for item in record["command"] if item != "--summary_only"]
        replace_option(command, "--out_dir", str(target.resolve()))
        subprocess.run(command, check=True)
        output_records.append({
            "alias": alias,
            "model": model,
            "cell": args.cell,
            "window_id": args.window_id,
            "seed": 42,
            "out_dir": str(target.resolve()),
            "source_out_dir": record["out_dir"],
            "command": command,
        })
        print(f"REPRESENTATIVE_PROGRESS={index + 1}/{len(expected)}", flush=True)

    (args.out_dir / "manifest.json").write_text(
        json.dumps({"runs": output_records}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
