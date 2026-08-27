#!/usr/bin/env python3
"""Build the non-destructive All Cells review collection.

The dissertation/JES figures remain the visual baseline.  Existing EAAI-palette
renders supply the current DM/HDM/HECM/DD colors, while explicitly selected
six-cell result panels are copied as supplements.  Figure 05 is the only raster
composition: its original panels (a) and (c) remain untouched and panel (b) is
replaced by the six-cell current-gain sensitivity panel.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / "figures"
RESULTS = FIGURES / "Results"
OUT = RESULTS / "All Cells"
UPLOAD = PAPER / "JES_Upload" / "Figures"
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


def fit_inside(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def build_figure_05() -> None:
    """Keep legacy panels (a)/(c), replacing only lower-left panel (b)."""
    base = Image.open(EAAI / "robustness_current_bias.png").convert("RGB")
    current = Image.open(RESULTS / "Figure_05_Current_Bias.png").convert("RGB")

    # The source figure is a fixed three-column matplotlib export.  This crop is
    # limited to the central six-cell panel, including its labels and title.
    panel_b = current.crop((1470, 0, 2910, current.height))

    # Clear only the legacy lower-left panel.  The upper current trace and the
    # lower-right trajectory panel are byte-for-byte inherited from the base.
    target_box = (0, 915, 1475, base.height)
    base.paste("white", target_box)
    fitted = fit_inside(panel_b, target_box[2] - target_box[0], target_box[3] - target_box[1])
    x = target_box[0] + (target_box[2] - target_box[0] - fitted.width) // 2
    y = target_box[1] + (target_box[3] - target_box[1] - fitted.height) // 2
    base.paste(fitted, (x, y))

    png = OUT / "Figure_05_Current_Bias.png"
    base.save(png, dpi=(300, 300), optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Original conceptual figures: no model-color remapping is needed.
    for number, filename in (
        (1, "Figure_01_Requirements_Overview.png"),
        (2, "Figure_02_Methodology_Overview.png"),
        (3, "Figure_03_Disturbance_Taxonomy.png"),
    ):
        del number
        copy_file(UPLOAD / filename)

    # Explicitly selected current replacements.
    copy_png(RESULTS / "Figure_04_Baseline_Performance", "Figure_04_Baseline_Performance")
    build_figure_05()

    # Dissertation figures in the current green/purple/blue/red palette.
    copy_file(EAAI / "robustness_noise.png", "Figure_06_Noise_Robustness.png")
    copy_file(EAAI / "robustness_init_recovery.png", "Figure_07_Initial_State_Recovery.png")
    copy_file(EAAI / "robustness_signal_integrity.png", "Figure_08_Signal_Integrity.png")
    copy_png(
        RESULTS / "Figure_09_Burst_Dropout_Transition_CORR",
        "Figure_09_Burst_Dropout_Transition",
    )
    copy_file(EAAI / "robustness_spike_response.png", "Figure_11_Voltage_Spike_Response.png")
    copy_file(EAAI / "robustness_cross_scenario.png", "Figure_12_Cross_Scenario_Heatmap.png")
    copy_file(EAAI / "robustness_decision.png", "Figure_13_Decision_Synthesis.png")
    copy_file(EAAI / "robustness_adc_quantization.png", "Figure_14_ADC_Quantization.png")

    # Six-cell supplements that preserve rather than replace the detailed plots.
    supplements = (
        ("Figure_06_Noise_Robustness", "Figure_06b_Noise_Robustness_Six_Cell_Overview"),
        ("Figure_07_Initial_State_Recovery_CORR", "Figure_07b_Initial_State_Recovery_Six_Cell"),
        ("Figure_08_Signal_Integrity", "Figure_08b_Signal_Integrity_Six_Cell_Overview"),
        ("Figure_11_Voltage_Spike_Response_REVISED", "Figure_11b_Voltage_Spike_Response_Six_Cell"),
        ("Figure_12_Cross_Scenario_Heatmap_REVISED", "Figure_12b_Cross_Scenario_Heatmap_Six_Cell"),
        ("Figure_13_Decision_Synthesis_REVISED", "Figure_13b_Decision_Synthesis_Six_Cell"),
        ("Figure_14_ADC_Quantization", "Figure_14b_ADC_Quantization_Six_Cell"),
        ("Figure_16_Holdout_Cell_Coverage", "Figure_16_Holdout_Cell_Coverage"),
        ("Figure_17_Statistical_Robustness", "Figure_17_Statistical_Robustness"),
        ("Figure_27_JES2_Test_Matrix", "Figure_27_JES2_Test_Matrix"),
        ("Figure_28_Statistical_Analysis_Workflow", "Figure_28_Statistical_Analysis_Workflow"),
        ("Figure_29_Evaluation_Window_Protocol", "Figure_29_Evaluation_Window_Protocol"),
    )
    for source, target in supplements:
        copy_png(RESULTS / source, target)

    print(f"Built review collection in {OUT}")


if __name__ == "__main__":
    main()
