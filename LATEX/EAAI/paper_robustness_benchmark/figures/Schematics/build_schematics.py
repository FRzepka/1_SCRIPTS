from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("/home/florianr/MG_Farm/1_Scripts/LATEX/EAAI/paper_robustness_benchmark/figures/Schematics")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#ffffff"
PANEL = "#f7f7f7"
EDGE = "#4d4d4d"
TEXT = "#1f1f1f"
PURPLE = "#6e2fc4"
TEAL = "#08bdba"
LILAC = "#d4bbff"
BLUE = "#4589ff"
GREY = "#d9d9d9"
GREY2 = "#ececec"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "font.size": 11,
            "axes.titlesize": 18,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def box(ax, x, y, w, h, text, fc=PANEL, ec=EDGE, lw=1.4, fs=12, weight="bold"):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TEXT, fontweight=weight)
    return rect


def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def clean(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def performance_requirements():
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    clean(ax)
    fig.suptitle("Deployment-oriented requirement space for embedded BMS estimation", y=0.98, fontsize=20, fontweight="bold")

    box(ax, 0.04, 0.54, 0.25, 0.28, "Implementation", fc=GREY2)
    box(ax, 0.375, 0.54, 0.25, 0.28, "Hardware", fc=GREY2)
    box(ax, 0.71, 0.54, 0.25, 0.28, "Performance\n(paper focus)", fc="#e8fbfa", ec=TEAL, lw=2.0)

    box(ax, 0.06, 0.24, 0.21, 0.18, "Calibration effort", fc="#fafafa", fs=11, weight="normal")
    box(ax, 0.39, 0.24, 0.21, 0.18, "Flash / RAM /\nreal-time budget", fc="#fafafa", fs=11, weight="normal")

    box(ax, 0.69, 0.27, 0.12, 0.13, "Accuracy", fc="#eef5ff", ec=BLUE, fs=11)
    box(ax, 0.82, 0.27, 0.12, 0.13, "Recovery", fc="#f2ecff", ec=PURPLE, fs=11)
    box(ax, 0.755, 0.10, 0.12, 0.13, "Robustness", fc="#e8fbfa", ec=TEAL, fs=11)

    arrow(ax, 0.835, 0.54, 0.75, 0.41, color=TEAL)
    arrow(ax, 0.835, 0.54, 0.88, 0.41, color=TEAL)
    arrow(ax, 0.835, 0.54, 0.815, 0.23, color=TEAL)

    ax.text(
        0.71,
        0.88,
        "Controlled cross-class benchmark under a matched embedded envelope",
        fontsize=12,
        color=TEXT,
        fontweight="bold",
    )
    ax.text(0.05, 0.05, "The manuscript isolates the performance branch while keeping the estimator set embedded-feasible.", fontsize=11, color="#444444")
    fig.savefig(OUT / "performance_requirements_overview.pdf")
    plt.close(fig)


def methodology_overview():
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    clean(ax)
    fig.suptitle("Methodology overview of the measurement-only benchmark", y=0.98, fontsize=20, fontweight="bold")

    box(ax, 0.03, 0.40, 0.16, 0.22, "Measured signals\nU, I, T, t", fc="#fafafa")
    box(ax, 0.24, 0.40, 0.18, 0.22, "Scenario injection\n(measurement-only)", fc="#fff5f2", ec="#d96b3b")
    box(ax, 0.47, 0.40, 0.18, 0.22, "Online feature rebuild\nQc, dU/dt, dI/dt,\nSOH aggregates", fc="#f7fbff", ec=BLUE)
    box(ax, 0.70, 0.57, 0.11, 0.12, "DM", fc="#f2ecff", ec=PURPLE)
    box(ax, 0.83, 0.57, 0.11, 0.12, "HDM", fc="#e8fbfa", ec=TEAL)
    box(ax, 0.70, 0.40, 0.11, 0.12, "HECM", fc="#f6f0ff", ec=LILAC)
    box(ax, 0.83, 0.40, 0.11, 0.12, "DD", fc="#eef5ff", ec=BLUE)
    box(ax, 0.74, 0.17, 0.16, 0.12, "Global + local\nperformance metrics", fc="#fafafa")

    arrow(ax, 0.19, 0.51, 0.24, 0.51)
    arrow(ax, 0.42, 0.51, 0.47, 0.51)
    arrow(ax, 0.65, 0.51, 0.70, 0.63)
    arrow(ax, 0.65, 0.51, 0.83, 0.63)
    arrow(ax, 0.65, 0.51, 0.70, 0.46)
    arrow(ax, 0.65, 0.51, 0.83, 0.46)
    arrow(ax, 0.755, 0.40, 0.82, 0.29)
    arrow(ax, 0.885, 0.40, 0.82, 0.29)
    arrow(ax, 0.755, 0.57, 0.82, 0.29)
    arrow(ax, 0.885, 0.57, 0.82, 0.29)

    ax.text(0.03, 0.08, "Same trajectory, same disturbances, same online reconstruction rules, same metrics.", fontsize=11, color="#444444")
    fig.savefig(OUT / "methodology_overview.pdf")
    plt.close(fig)


def benchmark_pipeline():
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    clean(ax)
    fig.suptitle("Benchmark pipeline from raw cell data to paper-facing results", y=0.98, fontsize=20, fontweight="bold")

    xs = [0.03, 0.23, 0.43, 0.63, 0.83]
    labels = [
        "Full-cell dataset\n(MGFarm LFP 18650)",
        "Scenario runner\nbaseline + disturbances",
        "Causal online\nestimator execution",
        "Global / local\nmetric extraction",
        "Figures, tables,\nand decision guidance",
    ]
    colors = ["#fafafa", "#fff5f2", "#f7fbff", "#fafafa", "#f2f8ff"]
    ecs = [EDGE, "#d96b3b", BLUE, EDGE, BLUE]
    for x, label, fc, ec in zip(xs, labels, colors, ecs):
        box(ax, x, 0.34, 0.14, 0.22, label, fc=fc, ec=ec)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 0.14, 0.45, xs[i + 1], 0.45)

    ax.text(0.23, 0.18, "Measurement corruption only:\nno hidden-state corruption,\nno synthetic parameter mismatch", ha="center", fontsize=10.5, color="#444444")
    ax.text(0.63, 0.18, "Trajectory-wide error metrics plus local\nrecovery and disturbance-window behavior", ha="center", fontsize=10.5, color="#444444")
    fig.savefig(OUT / "benchmark_pipeline_overview.pdf")
    plt.close(fig)


def scenario_taxonomy():
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    clean(ax)
    fig.suptitle("Taxonomy of the disturbance scenarios used in the benchmark", y=0.98, fontsize=20, fontweight="bold")

    box(ax, 0.05, 0.18, 0.27, 0.58, "Input disturbances", fc="#fff5f2", ec="#d96b3b", lw=1.8)
    box(ax, 0.365, 0.18, 0.27, 0.58, "Initialization errors", fc="#f2ecff", ec=PURPLE, lw=1.8)
    box(ax, 0.68, 0.18, 0.27, 0.58, "Signal-integrity faults", fc="#e8fbfa", ec=TEAL, lw=1.8)

    for y, txt in [(0.60, "Current bias"), (0.49, "Current noise"), (0.38, "Voltage noise"), (0.27, "Temperature noise"), (0.16, "Voltage spikes")]:
        box(ax, 0.09, y, 0.19, 0.08, txt, fc="#fafafa", ec="#d96b3b", lw=1.2, fs=11, weight="normal")

    box(ax, 0.41, 0.44, 0.18, 0.12, "Initial SOC mismatch", fc="#fafafa", ec=PURPLE, lw=1.2, fs=11, weight="normal")
    box(ax, 0.72, 0.58, 0.19, 0.08, "Missing samples", fc="#fafafa", ec=TEAL, lw=1.2, fs=11, weight="normal")
    box(ax, 0.72, 0.46, 0.19, 0.08, "Irregular sampling", fc="#fafafa", ec=TEAL, lw=1.2, fs=11, weight="normal")
    box(ax, 0.72, 0.34, 0.19, 0.08, "Burst dropout", fc="#fafafa", ec=TEAL, lw=1.2, fs=11, weight="normal")

    ax.text(0.06, 0.07, "The scenario set probes bias, noise, restart inconsistency, timing faults, and local signal loss under matched online execution.", fontsize=11, color="#444444")
    fig.savefig(OUT / "scenario_taxonomy.pdf")
    plt.close(fig)


def main():
    style()
    performance_requirements()
    methodology_overview()
    benchmark_pipeline()
    scenario_taxonomy()


if __name__ == "__main__":
    main()
