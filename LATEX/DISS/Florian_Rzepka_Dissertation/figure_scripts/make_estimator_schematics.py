# -*- coding: utf-8 -*-
"""Block-level architecture schematics for the dissertation.

Generates:
  pictures/robustness_dd_architecture_rb.png   (Ch6: DD SOC-GRU + shared SOH-LSTM)
  pictures/embedded_architecture_rb.png         (Ch7: embedded SOC-LSTM + SOH-LSTM)

Color palette matches the existing PowerPoint-style dissertation figures
(embedded_pipeline_original_rb.png etc.).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

# Palette sampled from existing dissertation figures
BLUE_FILL = "#C9DCEA"      # light blue (dataset / input blocks)
TEAL_FILL = "#9CCEC8"      # teal (recurrent cores)
PURPLE_FILL = "#C7A2E6"    # purple (MLP / optimization blocks)
LAVENDER_FILL = "#E2DAF2"  # light lavender (embedding / aux blocks)
GRAY_EDGE = "#404040"      # dark gray strokes and text
ARROW_BLUE = "#4472C4"     # office blue arrows
NOTE_PURPLE = "#7030A0"    # accent for pruning notes

FONT = {"family": "sans-serif", "color": GRAY_EDGE}


def rounded_box(ax, x, y, w, h, fill, title, lines, title_size=11.5, line_size=10):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=GRAY_EDGE, facecolor=fill)
    ax.add_patch(box)
    n = len(lines)
    ax.text(x + w / 2, y + h - 0.085, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", fontdict=FONT)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.175 - i * 0.072, ln, ha="center", va="center",
                fontsize=line_size, fontdict=FONT)
    return box


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_BLUE,
                                linewidth=1.8, mutation_scale=18))


def state_loop(ax, cx, top_y, label):
    """Recurrent state feedback loop drawn above a block."""
    r = 0.055
    verts = [(cx + 0.05, top_y), (cx + 0.05, top_y + r), (cx - 0.05, top_y + r), (cx - 0.05, top_y + 0.012)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    patch = mpatches.PathPatch(Path(verts, codes), facecolor="none",
                               edgecolor=ARROW_BLUE, linewidth=1.6)
    ax.add_patch(patch)
    ax.annotate("", xy=(cx - 0.05, top_y + 0.005), xytext=(cx - 0.05, top_y + 0.03),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_BLUE,
                                linewidth=1.6, mutation_scale=14))
    ax.text(cx, top_y + r + 0.022, label, ha="center", va="bottom",
            fontsize=9.5, style="italic", fontdict=FONT)


def main():
    from pathlib import Path as FilePath

    red = "#d62728"
    red_fill = "#eea4a5"
    blue = "#1f77b4"
    blue_fill = "#a1c6e0"
    gray = "#434343"
    gray_fill = "#eeeeee"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.2, 7.4))

    def box(ax, x, y, w, h, face, edge, title, lines,
            title_size=14.5, line_size=12.5):
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h - 0.105,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=gray,
        )
        if lines:
            line_top = y + h - 0.225
            spacing = 0.085 if len(lines) <= 3 else 0.070
            for index, line in enumerate(lines):
                ax.text(
                    x + w / 2,
                    line_top - index * spacing,
                    line,
                    ha="center",
                    va="center",
                    fontsize=line_size,
                    color=gray,
                )

    def connector(ax, x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=gray,
                linewidth=1.7,
                mutation_scale=16,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def recurrent_loop(ax, x0, x1, y, label):
        top = y + 0.15
        ax.plot(
            [x1 - 0.025, x1 - 0.025, x0 + 0.025, x0 + 0.025],
            [y, top, top, y + 0.025],
            color=red,
            linewidth=1.7,
            solid_capstyle="round",
        )
        ax.annotate(
            "",
            xy=(x0 + 0.025, y),
            xytext=(x0 + 0.025, y + 0.055),
            arrowprops=dict(
                arrowstyle="-|>",
                color=red,
                linewidth=1.7,
                mutation_scale=14,
            ),
        )
        ax.text(
            (x0 + x1) / 2,
            top + 0.028,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=gray,
        )

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # (a) Stateful SOC branch
    ax = ax1
    y, h = 0.19, 0.56
    box(
        ax, 0.035, y, 0.205, h, gray_fill, gray, "Online inputs",
        [r"$U,\ I,\ T,\ \widehat{\mathrm{SOH}}$",
         r"$Q_c,\ dU/dt,\ dI/dt,\ \Delta t$",
         "8 channels at 1 Hz"],
    )
    box(
        ax, 0.310, y, 0.205, h, red_fill, red, "GRU",
        ["1 recurrent layer",
         "hidden size 96",
         r"state $h_t$"],
    )
    box(
        ax, 0.585, y, 0.205, h, blue_fill, blue, "MLP head",
        [r"Linear $96 \rightarrow 96$ + ReLU",
         r"Linear $96 \rightarrow 1$",
         "sigmoid output"],
    )
    box(
        ax, 0.860, y + 0.07, 0.115, h - 0.14, gray_fill, gray, "Output",
        [r"$\widehat{\mathrm{SOC}}_t$",
         "each second"],
        title_size=13.0,
        line_size=11.5,
    )
    connector(ax, 0.240, y + h / 2, 0.310, y + h / 2)
    connector(ax, 0.515, y + h / 2, 0.585, y + h / 2)
    connector(ax, 0.790, y + h / 2, 0.860, y + h / 2)
    recurrent_loop(ax, 0.310, 0.515, y + h, r"$h_{t-1}$")
    ax.text(0.003, 0.96, "(a)", fontsize=14.5, fontweight="bold", color=gray,
            ha="left", va="top")

    # (b) Hourly SOH branch
    ax = ax2
    y, h = 0.16, 0.58
    specs = [
        (0.015, 0.155, gray_fill, gray, "Hourly aggregates",
         [r"$U,\ I,\ T,\ \mathrm{EFC},\ Q_c$",
          "mean, std, min, max",
          "20 features"]),
        (0.205, 0.155, blue_fill, blue, "Projection",
         [r"Linear $20 \rightarrow 128$",
          r"Linear $128 \rightarrow 128$",
          "GELU + LayerNorm"]),
        (0.395, 0.155, red_fill, red, "LSTM",
         ["2 recurrent layers",
          "hidden size 160",
          r"states $(h_t,c_t)$"]),
        (0.585, 0.155, blue_fill, blue, "Residual MLP",
         ["3 residual blocks",
          "width 160 + GELU",
          "skip + LayerNorm"]),
        (0.775, 0.115, blue_fill, blue, "Head",
         ["width 160",
          r"Linear $160 \rightarrow 1$"],
         13.0, 11.2),
        (0.925, 0.065, gray_fill, gray, "Output",
         [r"$\widehat{\mathrm{SOH}}_k$",
          "hourly"],
         11.5, 10.4),
    ]
    for spec in specs:
        x, width, face, edge, title, lines, *sizes = spec
        if sizes:
            box(ax, x, y, width, h, face, edge, title, lines, *sizes)
        else:
            box(ax, x, y, width, h, face, edge, title, lines)

    for left, right in [
        (0.170, 0.205),
        (0.360, 0.395),
        (0.550, 0.585),
        (0.740, 0.775),
        (0.890, 0.925),
    ]:
        connector(ax, left, y + h / 2, right, y + h / 2)
    recurrent_loop(ax, 0.395, 0.550, y + h, r"$(h_{t-1},c_{t-1})$")
    ax.text(0.003, 0.96, "(b)", fontsize=14.5, fontweight="bold", color=gray,
            ha="left", va="top")

    fig.subplots_adjust(left=0.01, right=0.995, top=0.985, bottom=0.02, hspace=0.14)
    project_root = FilePath(__file__).resolve().parents[1]
    png_out = project_root / "pictures" / "eaai_palette" / "robustness_dd_architecture.png"
    svg_out = project_root / "pictures" / "schematics" / "robustness_dd_architecture.svg"
    png_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=300, facecolor="white")
    fig.savefig(svg_out, facecolor="white")
    print("saved:", png_out)
    print("saved:", svg_out)
    plt.close(fig)


def embedded():
    """Ch7: the two stand-alone embedded estimators (LSTM + MLP).

    These differ from the Ch6 models: SOC uses 6 channels without SOH/dt,
    SOH runs sample-by-sample (no hourly aggregation), both are single-layer.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.4))

    def panel(ax, tag, title, in_lines, lstm_lines, head_lines, out_lines,
              prune_note):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        bh, by = 0.52, 0.18
        rounded_box(ax, 0.025, by, 0.215, bh, BLUE_FILL, "Online inputs (1 Hz)", in_lines)
        rounded_box(ax, 0.310, by, 0.205, bh, TEAL_FILL, "LSTM layer", lstm_lines)
        rounded_box(ax, 0.585, by, 0.205, bh, PURPLE_FILL, "MLP head", head_lines)
        rounded_box(ax, 0.860, by + 0.07, 0.125, bh - 0.14, "#FFFFFF", "Output", out_lines)
        arrow(ax, 0.240, by + bh / 2, 0.308, by + bh / 2)
        arrow(ax, 0.515, by + bh / 2, 0.583, by + bh / 2)
        arrow(ax, 0.790, by + bh / 2, 0.858, by + bh / 2)
        state_loop(ax, 0.4125, by + bh + 0.012, r"$(h_{t-1},\,c_{t-1})$")
        ax.text(0.4125, by - 0.10, prune_note, ha="center", va="center",
                fontsize=9.5, style="italic", color=NOTE_PURPLE)
        ax.text(0.0, 0.985, tag + "  " + title, ha="left", va="top",
                fontsize=12.5, fontweight="bold", fontdict=FONT)

    panel(
        ax1, "(a)", "Embedded SOC estimator: stateful LSTM + MLP",
        [r"$U,\;I,\;T$",
         r"$Q_c,\;dU/dt,\;dI/dt$",
         "6 channels, robust-scaled"],
        ["1 layer, hidden size 64",
         r"stateful: $(h_t, c_t)$ carried",
         "forward sample-by-sample"],
        [r"Linear 64$\,\rightarrow\,$64 + ReLU",
         r"Linear 64$\,\rightarrow\,$1 + Sigmoid"],
        [r"$\widehat{\mathrm{SOC}}_t \in [0,1]$", "every sample"],
        "deployment-prepared variant: structured pruning to hidden size 45 + INT8 weights")

    panel(
        ax2, "(b)", "Embedded SOH estimator: stateful LSTM + MLP",
        [r"$t,\;U,\;I,\;T$",
         r"$\mathrm{EFC},\;Q_c$",
         "6 channels, robust-scaled"],
        ["1 layer, hidden size 128",
         r"stateful: $(h_t, c_t)$ carried",
         "forward sample-by-sample"],
        [r"Linear 128$\,\rightarrow\,$128 + ReLU",
         r"Linear 128$\,\rightarrow\,$1 (linear)"],
        [r"$\widehat{\mathrm{SOH}}_t$", "every sample"],
        "deployment-prepared variant: structured pruning to hidden size 90 + INT8 weights")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.16)
    out = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\LATEX\DISS\Florian_Rzepka_Dissertation\pictures\embedded_architecture_rb.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print("saved:", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
    embedded()
