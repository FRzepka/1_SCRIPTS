"""Create the heatmaps used as Figures 4.1 and 4.4.

Figure 4.1 is reconstructed from the coefficients printed in the published
source figure. Figure 4.4 retains the original MAE matrix and its numerical
color-bar scales, but replaces the former white-ended blue map with a smooth
blue-to-red map whose extrema remain visible.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
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

RED = np.array(mcolors.to_rgb("#eea4a5"))
BLUE = np.array(mcolors.to_rgb("#a1c6e0"))


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    r, g, b = np.moveaxis(linear, -1, 0)
    l_root = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_root = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_root = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ],
        axis=-1,
    )


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=float)
    lightness, axis_a, axis_b = np.moveaxis(lab, -1, 0)
    l_root = lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b
    m_root = lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b
    s_root = lightness - 0.0894841775 * axis_a - 1.2914855480 * axis_b
    l_value = l_root**3
    m_value = m_root**3
    s_value = s_root**3
    linear = np.stack(
        [
            4.0767416621 * l_value - 3.3077115913 * m_value + 0.2309699292 * s_value,
            -1.2684380046 * l_value + 2.6097574011 * m_value - 0.3413193965 * s_value,
            -0.0041960863 * l_value - 0.7034186147 * m_value + 1.7076147010 * s_value,
        ],
        axis=-1,
    )
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def _oklab_blue_red_colors(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    blue_lab = _srgb_to_oklab(BLUE)
    red_lab = _srgb_to_oklab(RED)
    lab = blue_lab + (red_lab - blue_lab) * values[..., None]
    return _oklab_to_srgb(lab)


def direct_blue_red_colormap() -> mcolors.LinearSegmentedColormap:
    """Interpolate perceptually between the EAAI blue and red in OKLAB."""

    samples = _oklab_blue_red_colors(np.linspace(0.0, 1.0, 257))
    return mcolors.LinearSegmentedColormap.from_list(
        "direct_oklab_blue_red",
        samples,
        N=256,
    )


def red_blue_colors(values: np.ndarray) -> np.ndarray:
    """Map normalized values perceptually from blue to red in OKLAB."""

    return _oklab_blue_red_colors(values)


def correlation_colormap() -> mcolors.LinearSegmentedColormap:
    """Map zero to blue and both correlation extrema to the same red."""

    positions = np.linspace(0.0, 1.0, 257)
    magnitude = np.abs(2.0 * positions - 1.0)
    samples = _oklab_blue_red_colors(magnitude)
    return mcolors.LinearSegmentedColormap.from_list(
        "symmetric_oklab_blue_red",
        samples,
        N=256,
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


def _source_positions(
    array: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    region = array[y0:y1, x0:x1]
    rgb = region[..., :3].astype(float) / 255.0
    dark = np.max(rgb, axis=2) < 0.22
    return _nearest_blues_r_position(rgb), dark


def _recolor_region(
    array: np.ndarray,
    box: tuple[int, int, int, int],
    source_min: float,
    source_max: float,
) -> None:
    x0, y0, x1, y1 = box
    region = array[y0:y1, x0:x1]
    rgb = region[..., :3].astype(float) / 255.0
    normalized, dark = _source_positions(array, box)
    normalized = np.clip(
        (normalized - source_min) / (source_max - source_min),
        0.0,
        1.0,
    )
    converted = red_blue_colors(normalized)
    rgb[~dark] = converted[~dark]
    region[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)


def make_mae_matrix() -> None:
    source = MDPI_FIGURES / "MAE_matrix.png"
    array = np.asarray(Image.open(source).convert("RGBA")).copy()

    # Plot and color-bar interiors in the 4500 x 1500 source image.
    panel_regions = [
        ((223, 280, 1165, 1221), (1227, 64, 1293, 1437)),
        ((1702, 280, 2644, 1221), (2706, 64, 2772, 1437)),
        ((3181, 280, 4123, 1221), (4185, 64, 4251, 1437)),
    ]
    for plot_region, colorbar_region in panel_regions:
        colorbar_positions, colorbar_dark = _source_positions(array, colorbar_region)
        valid_positions = colorbar_positions[~colorbar_dark]
        source_min, source_max = np.quantile(valid_positions, [0.01, 0.99])
        _recolor_region(array, plot_region, source_min, source_max)
        _recolor_region(array, colorbar_region, source_min, source_max)

    Image.fromarray(array, mode="RGBA").save(
        OUTPUT_DIR / "paper1_mae_matrix.png",
        dpi=(300, 300),
    )


def _read_cross_scenario_values() -> tuple[np.ndarray, list[str], list[str]]:
    results = (
        SCRIPTS_ROOT
        / "DL_Models"
        / "LFP_SOC_SOH_Model"
        / "4_simulation_environment"
        / "results"
    )
    class_order = [
        "Direct measurement",
        "Hybrid direct measurement",
        "Hybrid ECM",
        "Data-driven",
    ]
    class_short = ["DM", "HDM", "HECM", "DD"]
    scenarios = [
        "ADC quantization",
        "Voltage spikes",
        "Current noise (high)",
        "Voltage noise",
        "Temperature noise",
        "Current bias",
        "Missing samples",
        "Irregular sampling",
        "Burst dropout",
    ]
    scenario_labels = [
        "ADC\nquantization",
        "Voltage\nspikes",
        "Current\nnoise",
        "Voltage\nnoise",
        "Temperature\nnoise",
        "Current\nbias",
        "Missing\nsamples",
        "Irregular\nsampling",
        "Burst\ndropout",
    ]

    values: dict[tuple[str, str], float] = {}
    key_results = results / "paper_tables_v4" / "table_key_results.md"
    with key_results.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["class"], row["scenario_label"])] = float(row["delta_mae"])

    adc_table = (
        results
        / "paper_tables_v4_adc_extension"
        / "table_adc_quantization_v4.md"
    )
    lines = adc_table.read_text(encoding="utf-8").splitlines()
    headers = [part.strip() for part in lines[0].strip("|").split("|")]
    class_index = headers.index("class")
    delta_index = headers.index("delta_mae")
    for line in lines[2:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        values[(parts[class_index], "ADC quantization")] = float(parts[delta_index])

    matrix = np.array(
        [[values[(class_name, scenario)] for scenario in scenarios] for class_name in class_order],
        dtype=float,
    )
    return matrix, class_short, scenario_labels


def make_cross_scenario_heatmap() -> None:
    values, y_labels, x_labels = _read_cross_scenario_values()
    vmax = float(np.nanmax(np.abs(values)))
    norm = SymLogNorm(
        linthresh=0.005,
        linscale=1.0,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )
    colormap = correlation_colormap()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    image = ax.imshow(values, aspect="auto", cmap=colormap, norm=norm)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            rgba = colormap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                col,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if luminance < 0.55 else "#202020",
            )

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\Delta$MAE")
    colorbar.set_ticks(
        [-0.30, -0.10, -0.03, -0.01, -0.003, 0.0, 0.003, 0.01, 0.03, 0.10, 0.30]
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "robustness_cross_scenario.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_correlation_heatmap()
    make_mae_matrix()
    make_cross_scenario_heatmap()
    print("Updated Figures 4.1, 4.4, and 5.12.")


if __name__ == "__main__":
    main()
