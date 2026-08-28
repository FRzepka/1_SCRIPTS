#!/usr/bin/env python3
"""Build the curated, consecutively numbered All Cells review collection."""

from __future__ import annotations

import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / "figures"
RESULTS = FIGURES / "Results"
OUT = RESULTS / "All Cells"
EAAI = Path(
    "/home/florianr/MG_Farm/1_Scripts/LATEX/DISS/"
    "Florian_Rzepka_Dissertation/pictures/eaai_palette"
)


def copy_file(source: Path, name: str | None = None) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = OUT / (name or source.name)
    shutil.copy2(source, target)
    return target


def copy_png(source_stem: Path, target_stem: str) -> None:
    copy_file(source_stem.with_suffix(".png"), f"{target_stem}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Conceptual figures use the final dissertation EAAI palette.
    for source, target in (
        ("bms_requirements.png", "Figure_01_Requirements_Overview.png"),
        ("robustness_methodology.png", "Figure_02_Methodology_Overview.png"),
        ("robustness_disturbance_taxonomy.png", "Figure_03_Disturbance_Taxonomy.png"),
    ):
        copy_file(EAAI / source, target)

    # Figures 05, 06, 14--20, and Appendix Figure 24 are generated directly in
    # All Cells by dedicated scripts and are intentionally not overwritten here.
    copy_png(RESULTS / "Figure_04_Baseline_Performance", "Figure_04_Baseline_Performance")

    # Selected dissertation figures in the current green/purple/blue/red palette.
    copy_file(EAAI / "robustness_noise.png", "Figure_07_Noise_Robustness.png")
    copy_png(
        RESULTS / "Figure_09_Burst_Dropout_Transition_CORR",
        "Figure_11_Burst_Dropout_Transition",
    )

    # Current six-cell analyses, numbered in their review order.
    supplements = (
        ("Figure_06_Noise_Robustness", "Figure_08_Noise_Robustness_Six_Cell_Overview"),
        ("Figure_07_Initial_State_Recovery_CORR", "Figure_09_Initial_State_Recovery_Six_Cell"),
        ("Figure_08_Signal_Integrity", "Figure_10_Signal_Integrity_Six_Cell_Overview"),
        ("Figure_11_Voltage_Spike_Response_REVISED", "Figure_12_Voltage_Spike_Response_Six_Cell"),
        ("Figure_11c_Voltage_Spike_Response_JES2", "Figure_13_Voltage_Spike_Response_JES2"),
        ("Figure_16_Holdout_Cell_Coverage", "Figure_21_APPENDIX_Holdout_Cell_Coverage"),
        ("Figure_27_JES2_Test_Matrix", "Figure_22_APPENDIX_JES2_Test_Matrix"),
        ("Figure_29_Evaluation_Window_Protocol", "Figure_23_APPENDIX_Evaluation_Window_Protocol"),
    )
    for source, target in supplements:
        copy_png(RESULTS / source, target)

    print(f"Built review collection in {OUT}")


if __name__ == "__main__":
    main()
