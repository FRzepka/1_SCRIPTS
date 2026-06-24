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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.6))

    # ------------------------------------------------------------------
    # Panel (a): DD SOC estimator (stateful GRU + MLP head)
    # ------------------------------------------------------------------
    ax = ax1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bh, by = 0.52, 0.18
    rounded_box(ax, 0.025, by, 0.21, bh, BLUE_FILL, "Online inputs (1 Hz)",
                [r"$U,\;I,\;T,\;\widehat{\mathrm{SOH}}$",
                 r"$Q_c,\;dU/dt,\;dI/dt,\;\Delta t$",
                 "8 channels, robust-scaled"])
    rounded_box(ax, 0.305, by, 0.20, bh, TEAL_FILL, "GRU layer",
                ["1 layer, hidden size 96",
                 "stateful: $h_t$ carried",
                 "forward sample-by-sample"])
    rounded_box(ax, 0.575, by, 0.20, bh, PURPLE_FILL, "MLP head",
                [r"Linear 96$\,\rightarrow\,$96 + ReLU",
                 r"Linear 96$\,\rightarrow\,$1 + Sigmoid"])
    rounded_box(ax, 0.845, by + 0.07, 0.13, bh - 0.14, "#FFFFFF", "Output",
                [r"$\widehat{\mathrm{SOC}}_t \in [0,1]$",
                 "every sample"])

    arrow(ax, 0.235, by + bh / 2, 0.303, by + bh / 2)
    arrow(ax, 0.505, by + bh / 2, 0.573, by + bh / 2)
    arrow(ax, 0.775, by + bh / 2, 0.843, by + bh / 2)
    state_loop(ax, 0.405, by + bh + 0.012, r"$h_{t-1}$")

    ax.text(0.405, by - 0.10,
            "deployment-prepared variant: structured pruning to hidden size 67 + INT8 weights",
            ha="center", va="center", fontsize=9.5, style="italic", color=NOTE_PURPLE)
    ax.text(0.0, 0.97, "(a)  Data-driven SOC estimator (DD): stateful GRU + MLP",
            ha="left", va="top", fontsize=12.5, fontweight="bold", fontdict=FONT)

    # ------------------------------------------------------------------
    # Panel (b): shared SOH estimator (hourly LSTM seq2seq)
    # ------------------------------------------------------------------
    ax = ax2
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bh, by = 0.56, 0.16
    rounded_box(ax, 0.012, by, 0.165, bh, BLUE_FILL, "Hourly aggregates",
                [r"$\{U, I, T, \mathrm{EFC}, Q_c\}$",
                 r"$\times$ {mean, std, min, max}",
                 "20 channels, 1/h"], title_size=11, line_size=9.3)
    rounded_box(ax, 0.222, by, 0.16, bh, LAVENDER_FILL, "Embedding block",
                [r"Linear 20$\,\rightarrow\,$128 + GELU",
                 r"Linear 128$\,\rightarrow\,$128 + GELU",
                 "LayerNorm"], title_size=11, line_size=9.3)
    rounded_box(ax, 0.427, by, 0.155, bh, TEAL_FILL, "LSTM core",
                ["3 layers, hidden 192",
                 r"stateful: $(h_t, c_t)$",
                 "carried hour-to-hour"], title_size=11, line_size=9.3)
    rounded_box(ax, 0.627, by, 0.16, bh, PURPLE_FILL, "2 residual blocks",
                [r"Linear 192$\,\rightarrow\,$160 + GELU",
                 r"Linear 160$\,\rightarrow\,$192",
                 "skip + LayerNorm"], title_size=11, line_size=9.3)
    rounded_box(ax, 0.832, by + 0.30, 0.155, 0.26, PURPLE_FILL, "Prediction head",
                [r"192$\,\rightarrow\,$160$\,\rightarrow\,$160 + GELU",
                 r"Linear 160$\,\rightarrow\,$1"], title_size=10.5, line_size=9.0)
    rounded_box(ax, 0.842, by - 0.06, 0.135, 0.24, "#FFFFFF", "Output",
                [r"$\widehat{\mathrm{SOH}}_k$ hourly,",
                 "held between updates"], title_size=10.5, line_size=8.8)

    arrow(ax, 0.177, by + bh / 2, 0.220, by + bh / 2)
    arrow(ax, 0.382, by + bh / 2, 0.425, by + bh / 2)
    arrow(ax, 0.582, by + bh / 2, 0.625, by + bh / 2)
    arrow(ax, 0.787, by + bh / 2, 0.830, by + bh / 2 + 0.05)
    arrow(ax, 0.909, by + 0.295, 0.909, by + 0.19)
    state_loop(ax, 0.5045, by + bh + 0.012, r"$(h_{t-1},\,c_{t-1})$")

    ax.text(0.5045, by - 0.115,
            "deployed in base form (pruning not viable) + INT8 weight quantization",
            ha="center", va="center", fontsize=9.5, style="italic", color=NOTE_PURPLE)
    ax.text(0.0, 1.0, "(b)  Shared SOH estimator: hourly LSTM sequence-to-sequence model",
            ha="left", va="top", fontsize=12.5, fontweight="bold", fontdict=FONT)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.12)
    out = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\LATEX\DISS\Florian_Rzepka_Dissertation\pictures\robustness_dd_architecture_rb.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print("saved:", out)
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
