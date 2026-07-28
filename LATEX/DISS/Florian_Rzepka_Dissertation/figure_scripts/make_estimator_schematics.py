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
    red_fill = "#f8dfe0"
    blue = "#1f77b4"
    blue_fill = "#dcebf5"
    gray = "#434343"
    gray_fill = "#f0f0f0"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.2, 7.7))

    def block(
        ax,
        x,
        y,
        w,
        h,
        body_color,
        accent,
        title,
        lines,
        title_size=13.5,
        line_size=11.5,
    ):
        header_h = 0.145
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.7,
                edgecolor=accent,
                facecolor=body_color,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (x, y + h - header_h),
                w,
                header_h,
                linewidth=0,
                facecolor=accent,
            )
        )
        ax.text(
            x + w / 2,
            y + h - header_h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color="white",
        )
        body_center = y + (h - header_h) / 2
        spacing = 0.080 if len(lines) <= 3 else 0.066
        first_y = body_center + spacing * (len(lines) - 1) / 2
        for index, line in enumerate(lines):
            ax.text(
                x + w / 2,
                first_y - index * spacing,
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
        top = y + 0.125
        ax.plot(
            [x1 - 0.020, x1 - 0.020, x0 + 0.020],
            [y, top, top],
            color=red,
            linewidth=1.7,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        ax.annotate(
            "",
            xy=(x0 + 0.020, y),
            xytext=(x0 + 0.020, top),
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
    y, h = 0.20, 0.53
    block(
        ax, 0.035, y, 0.235, h, gray_fill, gray, "Online inputs",
        [r"$U,\ I,\ T,\ \widehat{\mathrm{SOH}}$",
         r"$Q_c,\ dU/dt,\ dI/dt,\ \Delta t$",
         "8 channels at 1 Hz"],
    )
    block(
        ax, 0.350, y, 0.235, h, red_fill, red, "GRU core",
        ["1 recurrent layer",
         "hidden size 96",
         r"state $h_t$"],
    )
    block(
        ax, 0.665, y, 0.235, h, blue_fill, blue, "MLP head",
        [r"Linear $96 \rightarrow 96$ + ReLU",
         r"Linear $96 \rightarrow 1$",
         "sigmoid output"],
    )
    connector(ax, 0.270, y + h / 2, 0.350, y + h / 2)
    connector(ax, 0.585, y + h / 2, 0.665, y + h / 2)
    connector(ax, 0.900, y + h / 2, 0.955, y + h / 2)
    ax.text(
        0.967,
        y + h / 2 + 0.035,
        r"$\widehat{\mathrm{SOC}}_t$",
        ha="center",
        va="center",
        fontsize=13.5,
        color=gray,
    )
    ax.text(
        0.967,
        y + h / 2 - 0.050,
        "1 Hz",
        ha="center",
        va="center",
        fontsize=10.5,
        color=gray,
    )
    recurrent_loop(ax, 0.350, 0.585, y + h, r"$h_{t-1}$")
    ax.text(0.003, 0.96, "(a)", fontsize=14.5, fontweight="bold", color=gray,
            ha="left", va="top")

    # (b) Hourly SOH branch
    ax = ax2
    y, h = 0.17, 0.54
    specs = [
        (0.015, 0.165, gray_fill, gray, "Hourly inputs",
         [r"$U,\ I,\ T,\ \mathrm{EFC},\ Q_c$",
          "mean, std, min, max",
          "20 features"]),
        (0.215, 0.165, blue_fill, blue, "Projection",
         [r"Linear $20 \rightarrow 128$",
          r"Linear $128 \rightarrow 128$",
          "GELU + LayerNorm"]),
        (0.415, 0.165, red_fill, red, "LSTM core",
         ["2 recurrent layers",
          "hidden size 160",
          r"states $(h_t,c_t)$"]),
        (0.615, 0.175, blue_fill, blue, "Residual MLP",
         ["3 residual blocks",
          "width 160 + GELU",
          "skip + LayerNorm"]),
        (0.825, 0.125, blue_fill, blue, "Head",
         ["width 160",
          r"Linear $160 \rightarrow 1$"],
         12.5, 10.5),
    ]
    for spec in specs:
        x, width, face, edge, title, lines, *sizes = spec
        if sizes:
            block(ax, x, y, width, h, face, edge, title, lines, *sizes)
        else:
            block(ax, x, y, width, h, face, edge, title, lines)

    for left, right in [
        (0.180, 0.215),
        (0.380, 0.415),
        (0.580, 0.615),
        (0.790, 0.825),
    ]:
        connector(ax, left, y + h / 2, right, y + h / 2)
    connector(ax, 0.950, y + h / 2, 0.978, y + h / 2)
    ax.text(
        0.982,
        y + h / 2 + 0.035,
        r"$\widehat{\mathrm{SOH}}_k$",
        ha="center",
        va="center",
        fontsize=12.5,
        color=gray,
    )
    ax.text(
        0.982,
        y + h / 2 - 0.050,
        "hourly",
        ha="center",
        va="center",
        fontsize=10.0,
        color=gray,
    )
    recurrent_loop(ax, 0.415, 0.580, y + h, r"$(h_{t-1},c_{t-1})$")
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
