"""Create the red-blue heatmaps used as Figures 4.1 and 4.4.

Figure 4.1 is reconstructed from the coefficients printed in the published
source figure. Figure 4.4 retains the original MAE matrix and its numerical
color-bar scales, but replaces the former white-ended blue map with a
blue-to-red map whose extrema remain visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "pictures" / "eaai_palette"
MDPI_FIGURES = (
    SCRIPTS_ROOT
    / "LATEX"
    / "MDPI"
    / "MPDI_23__Neural_Network_Architecture_for_Determining_the_Aging_of_Stationary_Storage_Systems_in_Smart_Grids"
    / "Figures"
)

RED = np.array(mcolors.to_rgb("#d62728"))
RED_LIGHT = np.array(mcolors.to_rgb("#f2b2b3"))
BLUE = np.array(mcolors.to_rgb("#1f77b4"))
BLUE_LIGHT = np.array(mcolors.to_rgb("#d5e7f3"))


def _blend(start: np.ndarray, end: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    return start + (end - start) * fraction[..., None]


def red_blue_colors(values: np.ndarray) -> np.ndarray:
    """Map normalized values to blue shades below 0.5 and red shades above."""

    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    colors = np.empty(values.shape + (3,), dtype=float)
    lower = values <= 0.5
    colors[lower] = _blend(BLUE, BLUE_LIGHT, values[lower] / 0.5)
    colors[~lower] = _blend(
        RED_LIGHT,
        RED,
        (values[~lower] - 0.5) / 0.5,
    )
    return colors


def correlation_colormap() -> mcolors.LinearSegmentedColormap:
    """Negative correlations are red, positive correlations are blue.

    Zero is intentionally light blue instead of white so that a measured
    coefficient of zero is distinct from the masked upper triangle.
    """

    return mcolors.LinearSegmentedColormap.from_list(
        "correlation_red_blue",
        [
            (0.00, "#d62728"),
            (0.495, "#f2b2b3"),
            (0.500, "#d5e7f3"),
            (1.00, "#1f77b4"),
        ],
    )


def make_correlation_heatmap() -> None:
    values = np.array(
        [
            [0.00, np.nan, np.nan, np.nan],
            [-0.58, -0.56, np.nan, np.nan],
            [-0.99, 0.00, 0.60, np.nan],
            [-0.99, 0.00, 0.60, 1.00],
        ]
    )
    x_labels = [
        "SOH",
        "Voltage [V]",
        r"Temperature [$^\circ$C]",
        r"$Q_{\mathrm{pos}}$ [Ah]",
    ]
    y_labels = [
        "Voltage [V]",
        r"Temperature [$^\circ$C]",
        r"$Q_{\mathrm{pos}}$ [Ah]",
        r"$Q_{\mathrm{neg}}$ [Ah]",
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(
        masked,
        cmap=correlation_colormap(),
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(4), labels=x_labels)
    ax.set_yticks(np.arange(4), labels=y_labels)
    ax.tick_params(axis="x", labelrotation=0, pad=7)
    ax.tick_params(axis="y", pad=7)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if np.isfinite(values[row, col]):
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#202020",
                )

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.035)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.outline.set_linewidth(0.8)
    fig.subplots_adjust(left=0.22, right=0.90, bottom=0.14, top=0.98)
    fig.savefig(
        OUTPUT_DIR / "paper1_correlation.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def _nearest_blues_r_position(rgb: np.ndarray) -> np.ndarray:
    source = plt.colormaps["Blues_r"](np.linspace(0.0, 1.0, 1024))[:, :3]
    flat = rgb.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    mapped = np.empty(unique.shape[0], dtype=float)
    chunk_size = 20000
    for start in range(0, unique.shape[0], chunk_size):
        chunk = unique[start : start + chunk_size]
        distance = np.sum((chunk[:, None, :] - source[None, :, :]) ** 2, axis=2)
        mapped[start : start + len(chunk)] = np.argmin(distance, axis=1) / 1023.0
    return mapped[inverse].reshape(rgb.shape[:2])


def _recolor_region(array: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    region = array[y0:y1, x0:x1]
    rgb = region[..., :3].astype(float) / 255.0
    dark = np.max(rgb, axis=2) < 0.22
    normalized = _nearest_blues_r_position(rgb)
    converted = red_blue_colors(normalized)
    rgb[~dark] = converted[~dark]
    region[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)


def make_mae_matrix() -> None:
    source = MDPI_FIGURES / "MAE_matrix.png"
    array = np.asarray(Image.open(source).convert("RGBA")).copy()

    # Plot and color-bar interiors in the 4500 x 1500 source image.
    regions = [
        (223, 280, 1165, 1221),
        (1702, 280, 2644, 1221),
        (3181, 280, 4123, 1221),
        (1227, 64, 1293, 1437),
        (2706, 64, 2772, 1437),
        (4185, 64, 4251, 1437),
    ]
    for region in regions:
        _recolor_region(array, region)

    Image.fromarray(array, mode="RGBA").save(
        OUTPUT_DIR / "paper1_mae_matrix.png",
        dpi=(300, 300),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_correlation_heatmap()
    make_mae_matrix()
    print("Updated Figure 4.1 and Figure 4.4.")


if __name__ == "__main__":
    main()
