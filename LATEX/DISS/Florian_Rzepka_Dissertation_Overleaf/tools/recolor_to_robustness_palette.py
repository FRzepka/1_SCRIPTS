from __future__ import annotations

from pathlib import Path
from typing import Iterable
import colorsys

from PIL import Image


PICTURES_DIR = Path(__file__).resolve().parents[1] / "pictures"

# Robustness-paper palette sampled from the existing dissertation figures.
PALETTE = {
    "teal": (21, 192, 189),
    "blue": (79, 143, 255),
    "purple": (117, 58, 199),
}

TARGET_FILES = [
    "bms_requirements.png",
    "embedded_doe_cube.png",
    "embedded_doe_cube_original.png",
    "embedded_latency_hist.png",
    "embedded_lstm_schematic.png",
    "embedded_model_sizes.png",
    "embedded_model_sizes_original.png",
    "embedded_pipeline.png",
    "embedded_pipeline_original.png",
    "embedded_pruning_schematic.png",
    "embedded_quantization.png",
    "embedded_quantization_schematic.png",
    "embedded_soc_dashboard.png",
    "embedded_soc_error.png",
    "embedded_soc_test_trajectory.png",
    "embedded_soc_zoom_checkup.png",
    "embedded_soc_zoom_pulse.png",
    "embedded_soh_all_days.png",
    "embedded_soh_dashboard.png",
    "embedded_soh_error.png",
    "paper1_architecture.png",
    "paper1_correlation.png",
    "paper1_lag_sequence.png",
    "paper1_mae_matrix.png",
    "paper1_mlp_architecture.png",
    "paper1_process.png",
    "paper1_results.png",
    "paper1_soh_cycles.png",
    "paper1_soh_pred.png",
    "paper1_soh_scatter.png",
    "paper1_soh_time.png",
]


def rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (channel / 255.0 for channel in rgb)
    return colorsys.rgb_to_hsv(r, g, b)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


PALETTE_HUES = {
    name: rgb_to_hsv(rgb)[0] for name, rgb in PALETTE.items()
}


def circular_hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 1.0 - diff)


def nearest_palette_hue(source_hue: float) -> float:
    return min(PALETTE_HUES.values(), key=lambda target_hue: circular_hue_distance(source_hue, target_hue))


def recolor_pixel(rgba: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r, g, b, a = rgba

    if a == 0:
        return rgba

    # Preserve white backgrounds, near-black text/axes, and neutral grays.
    if r > 245 and g > 245 and b > 245:
        return rgba
    if r < 30 and g < 30 and b < 30:
        return rgba
    if max(r, g, b) - min(r, g, b) < 18:
        return rgba

    h, s, v = rgb_to_hsv((r, g, b))
    if s < 0.12:
        return rgba

    target_h = nearest_palette_hue(h)

    # Keep luminance structure; slightly stabilize saturation for plot elements.
    new_s = min(1.0, max(0.18, s * 0.95))
    new_v = v

    nr, ng, nb = hsv_to_rgb(target_h, new_s, new_v)
    return nr, ng, nb, a


def recolor_image(path: Path) -> Path:
    out_path = path.with_stem(f"{path.stem}_rb")
    image = Image.open(path).convert("RGBA")
    pixels = [recolor_pixel(px) for px in image.getdata()]
    result = Image.new("RGBA", image.size)
    result.putdata(pixels)
    result.save(out_path)
    return out_path


def main(files: Iterable[str]) -> None:
    for name in files:
        src = PICTURES_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = recolor_image(src)
        print(f"{src.name} -> {dst.name}")


if __name__ == "__main__":
    main(TARGET_FILES)
