from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from jes2_protocol import PRIMARY_STOCHASTIC_ALIASES, STOCHASTIC_ALIASES


def repeat_count(manifest: dict, alias: str) -> int:
    if alias in PRIMARY_STOCHASTIC_ALIASES:
        return int(manifest.get("stochastic_repeats", 1))
    if alias in STOCHASTIC_ALIASES:
        return int(manifest.get("secondary_stochastic_repeats", manifest.get("stochastic_repeats", 1)))
    return 1


def expected_runs(manifest: dict) -> int:
    models = set(manifest.get("models", ["DM", "HDM", "HECM", "DD"]))
    soh_models = models - {"DM"}
    modes = manifest.get("soh_modes", [])
    reference_aliases = set(manifest.get("reference_aliases", []))
    cadence_aliases = set(manifest.get("cadence_aliases", ["baseline"]))
    lstm_intervals = manifest.get("lstm_publish_intervals", [1])
    reference_intervals = manifest.get("reference_publish_intervals", [1])
    scenarios = manifest.get("protocol", {}).get("scenarios", [])
    definitions = manifest.get("window", {}).get("definitions", [])
    window_count = len(definitions) if definitions else len(manifest.get("cells", []))
    per_window = 0
    for row in scenarios:
        alias = row["alias"]
        repeats = repeat_count(manifest, alias)
        if "DM" in models:
            per_window += repeats
        for mode in modes:
            if mode == "reference" and alias not in reference_aliases:
                continue
            intervals = reference_intervals if mode == "reference" else lstm_intervals
            if mode == "lstm" and alias not in cadence_aliases:
                intervals = [1]
            per_window += repeats * len(intervals) * len(soh_models)
    return window_count * per_window


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show JES2 manifest progress without external search tools.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = expected_runs(manifest)
    completed = sum(
        row.get("status") in {"completed", "skipped_existing"}
        for row in manifest.get("runs", [])
    )
    started = datetime.fromisoformat(manifest["started_utc"])
    now = datetime.now(timezone.utc)
    elapsed = max(0.0, (now - started).total_seconds())
    fraction = completed / expected if expected else 0.0
    eta_seconds = elapsed * (expected - completed) / completed if completed else float("nan")
    last = manifest.get("runs", [])[-1] if manifest.get("runs") else {}
    print(f"Campaign: {manifest.get('tag', args.manifest.parent.name)}")
    print(f"Progress: {completed}/{expected} ({100.0 * fraction:.1f}%)")
    print(f"Elapsed:  {format_duration(elapsed)}")
    print(f"ETA:      {format_duration(eta_seconds) if completed else 'calculating'}")
    if last:
        print(
            "Last:     "
            f"{last.get('cell')} / {last.get('window_id', 'single_window')} / "
            f"{last.get('alias')} / seed {last.get('seed')} / {last.get('model')}"
        )


if __name__ == "__main__":
    main()
