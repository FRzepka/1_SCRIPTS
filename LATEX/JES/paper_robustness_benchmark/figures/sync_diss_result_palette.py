from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = (
    SCRIPT_ROOT
    / "LATEX/DISS/Florian_Rzepka_Dissertation_260723_rev/pictures/red_muted"
)
DEFAULT_TARGET = Path(__file__).resolve().parent / "Results"

MAPPING = {
    "robustness_baseline.png": "Figure_04_Baseline_Performance.png",
    "robustness_current_bias.png": "Figure_05_Current_Bias.png",
    "robustness_noise.png": "Figure_06_Noise_Robustness.png",
    "robustness_init_recovery.png": "Figure_07_Initial_State_Recovery.png",
    "robustness_signal_integrity.png": "Figure_08_Signal_Integrity.png",
    "robustness_dropout_transition.png": "Figure_09_Burst_Dropout_Transition.png",
    "robustness_dropout_recovery.png": "Figure_10_Burst_Dropout_Recovery.png",
    "robustness_spike_response.png": "Figure_11_Voltage_Spike_Response.png",
    "robustness_cross_scenario.png": "Figure_12_Cross_Scenario_Heatmap.png",
    "robustness_decision.png": "Figure_13_Decision_Synthesis.png",
    "robustness_adc_quantization.png": "Figure_14_ADC_Quantization.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the dissertation red-muted palette to legacy JES V4 figures.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    records = []
    for source_name, target_name in MAPPING.items():
        source = args.source / source_name
        target = args.target / target_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source.resolve()),
                "target": str(target.resolve()),
                "sha256": sha256(target),
            }
        )
        print(f"{source.name} -> {target.name}")

    (args.target / "legacy_diss_palette_manifest.json").write_text(
        json.dumps(
            {
                "status": "legacy_single_cell_v4_recolored_only",
                "replacement": "build_jes2_paper_results.py output after the complete multi-cell campaign",
                "palette": "DISS 01_tu_red_muted",
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
