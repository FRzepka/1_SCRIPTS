import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def make_model_cmap(lighten: float = 0.35) -> ListedColormap:
    """
    Colormap built from the three Modellfarben der ursprünglichen Plots:
    grün (Base), rot (Pruned), blau (Quantized). Anschließend wird in
    Richtung Weiß aufgehellt, um einen weichen Pastell-Look zu erhalten.
    """
    base = LinearSegmentedColormap.from_list(
        "model_cmap",
        [
            "#d62728",  # red    (Pruned, ersetzt das frühere Lila)
            "#2ca02c",  # green  (Base)
            "#1f77b4",  # blue   (Quantized)
        ],
    )
    colors = base(np.linspace(0.0, 1.0, 256))
    white = np.ones_like(colors)
    mixed = (1.0 - lighten) * colors + lighten * white
    mixed[:, -1] = 1.0  # volle Deckkraft
    return ListedColormap(mixed)


def save_matrix_image(mat, fname, cmap, figsize=(3, 3)):
    plt.figure(figsize=figsize)
    plt.imshow(mat, cmap=cmap, interpolation="nearest", aspect="equal")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    out_dir = Path(__file__).parent
    os.makedirs(out_dir, exist_ok=True)

    # Global style
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    # Colormaps auf Basis des Grün/Rot/Blau‑Schemas.
    cmap_fp32 = make_model_cmap(lighten=0.25)     # kräftig für FP32‑Gewichte
    cmap_int8 = make_model_cmap(lighten=0.45)     # etwas heller für INT8
    cmap_scales = make_model_cmap(lighten=0.15)   # am kräftigsten für Skalen

    rng = np.random.default_rng(42)

    # 1) FP32 weight matrix: moderates Raster mit geordneter 2D‑Gradientenstruktur.
    h, w = 14, 14
    x = np.linspace(0.0, 1.0, w)
    y = np.linspace(0.0, 1.0, h)
    xv, yv = np.meshgrid(x, y)
    fp32_weights = 0.6 * xv + 0.4 * yv
    save_matrix_image(fp32_weights, out_dir / "fp32_weights.png", cmap_fp32, figsize=(3.0, 3.0))

    # 2) FP32 biases + activations: breites, niedriges Rechteck mit gleichem Farbschema.
    h_b, w_b = 5, 14
    xb = np.linspace(0.0, 1.0, w_b)
    yb = np.linspace(0.0, 1.0, h_b)
    xbv, ybv = np.meshgrid(xb, yb)
    fp32_bias_act = 0.7 * xbv + 0.3 * ybv
    save_matrix_image(fp32_bias_act, out_dir / "fp32_biases_activations.png", cmap_fp32, figsize=(3.0, 1.0))

    # 3) INT8 weights: gröberes Raster (weniger "Pixel").
    int8_weights = rng.integers(low=-128, high=127, size=(8, 8))
    save_matrix_image(int8_weights, out_dir / "int8_weights.png", cmap_int8, figsize=(1.6, 1.6))

    # 4) Per‑row scales: vertikale Farbskala.
    scales = np.linspace(0.0, 1.0, 32).reshape(-1, 1)
    save_matrix_image(scales, out_dir / "per_row_scales.png", cmap_scales, figsize=(0.7, 2.5))


if __name__ == "__main__":
    main()

