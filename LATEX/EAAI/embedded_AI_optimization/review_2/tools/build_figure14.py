"""Render the single-column Figure 14 without changing any benchmark values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pymupdf as fitz


REVIEW = Path(__file__).resolve().parents[1]
MODELS = ("Base", "Pruned", "Quantized")
TASKS = ("SOC", "SOH")
COLORS = ("#2ca02c", "#d62728", "#1f77b4")
WIDTH_MM = 85
HEIGHT_MM = 67
PT_PER_MM = 72 / 25.4
# Figure 10 uses a 12-inch canvas and 20/18/15/13 pt type before scaling.
REFERENCE_SCALE = WIDTH_MM / (12 * 25.4)
FONT_TITLE = 20 * REFERENCE_SCALE
FONT_AXIS = 18 * REFERENCE_SCALE
FONT_TICK = 15 * REFERENCE_SCALE
FONT_LEGEND = 13 * REFERENCE_SCALE
FONT_VALUE = 14 * REFERENCE_SCALE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tint(color: str, strength: float) -> tuple:
    return tuple(1 - strength * (1 - channel) for channel in to_rgb(color))


def load_records() -> dict:
    source = REVIEW / "sources/resource_kpis.csv"
    assert source.read_bytes() == (REVIEW / "archive/neptune_resource_update/resource_kpis.csv").read_bytes()
    with source.open(encoding="utf-8", newline="") as handle:
        records = {(row["task"], row["model"]): row for row in csv.DictReader(handle)}
    assert set(records) == {(task, model) for task in TASKS for model in MODELS}
    for row in records.values():
        assert int(row["static_ram_bytes"]) + int(row["max_stack_bytes"]) == int(row["ram_bytes"])
        assert float(row["ram_KiB"]) == int(row["ram_bytes"]) / 1024
        assert float(row["flash_KiB"]) == int(row["flash_bytes"]) / 1024
    return records


def plot_figure(records: dict) -> list[str]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_TICK,
        "axes.titlesize": FONT_TITLE, "axes.titleweight": "normal",
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK,
        "axes.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.88, wspace=0.32, hspace=0.40)
    legend_handles = [
        Patch(facecolor="gray", edgecolor="black", linewidth=0.35, label="SOC (left, darker)"),
        Patch(facecolor="lightgray", edgecolor="black", linewidth=0.35, label="SOH (right, lighter)"),
    ]
    legend = fig.legend(handles=legend_handles, loc="upper center", ncol=2,
                        bbox_to_anchor=(0.55, 0.995), fontsize=FONT_LEGEND,
                        frameon=False, handlelength=1.7, handletextpad=0.6,
                        columnspacing=2.0, borderaxespad=0.4)
    titles = ("Exported parameter count", "Idealized parameter storage",
              "Firmware flash (ELF builds)", "RAM usage (static + stack)")
    labels = []
    annotations = []
    bar_patches = []
    for panel, (ax, title) in enumerate(zip(axes.flat, titles)):
        maximum = 0
        for task_index, task in enumerate(TASKS):
            for model_index, model in enumerate(MODELS):
                row = records[task, model]
                parameters = int(row["parameters"])
                values = (parameters, parameters * (1 if model == "Quantized" else 4) / 1024,
                          float(row["flash_KiB"]), float(row["ram_KiB"]))
                value = values[panel]
                maximum = max(maximum, value)
                color = COLORS[model_index]
                bars = ax.bar(model_index + (task_index - 0.5) * 0.35, value, 0.35,
                              facecolor=tint(color, 0.40 if task_index == 0 else 0.24),
                              edgecolor=color if task_index == 0 else tint(color, 0.60),
                              linewidth=0.5, zorder=3)
                bar_patches.extend(bars)
                label = f"{value:,}" if panel == 0 else f"{value:.2f}"
                labels.append(label)
                # Align each label away from the shared bar edge to avoid the taller neighbour.
                annotations.append(ax.annotate(
                    label, xy=(model_index + (-0.015 if task_index == 0 else 0.015), value),
                    xytext=(0, 1.3), textcoords="offset points",
                    ha="right" if task_index == 0 else "left", va="bottom",
                    fontsize=FONT_VALUE, zorder=4,
                ))
        ax.set(xticks=range(3), xticklabels=MODELS,
               xlim=(-0.65, 2.65), ylim=(0, maximum * 1.24))
        ax.set_title(title, pad=3)
        ax.set_ylabel("Count" if panel == 0 else "Size [KiB]", labelpad=2)
        ax.tick_params(axis="both", length=1.7, pad=1.7, width=0.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4)
        ax.set_axisbelow(True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer) for item in annotations]
    overlaps = [(labels[i], labels[j]) for i, a in enumerate(boxes)
                for j, b in enumerate(boxes) if j > i and a.overlaps(b)]
    assert not overlaps, f"Value labels overlap: {overlaps}"
    bar_boxes = [patch.get_window_extent(renderer) for patch in bar_patches]
    occluded = [labels[i] for i, box in enumerate(boxes) if any(box.overlaps(bar) for bar in bar_boxes)]
    assert not occluded, f"Value labels intersect bars: {occluded}"
    legend_boxes = [legend.get_window_extent(renderer)]
    assert not any(a.overlaps(b) for a in boxes + bar_boxes for b in legend_boxes), "Legend overlaps data"
    visible_text = annotations + [ax.title for ax in axes.flat] + [ax.yaxis.label for ax in axes.flat]
    visible_text += [label for ax in axes.flat for label in ax.get_xticklabels()]
    assert not any(item.get_window_extent(renderer).overlaps(legend_boxes[0])
                   for item in visible_text), "Legend overlaps figure text"
    for item in visible_text:
        box = item.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), "Clipped figure text"
    assert fig.bbox.contains(legend_boxes[0].x0, legend_boxes[0].y0)
    assert fig.bbox.contains(legend_boxes[0].x1, legend_boxes[0].y1)
    fig.savefig(REVIEW / "gr14.pdf", metadata={
        "Title": "Figure 14 - single-column resource comparison",
        "Subject": "2x2 layout; shared top legend; larger plot areas; unchanged type sizes and values",
    })
    fig.savefig(REVIEW / "gr14.png", dpi=400)
    plt.close(fig)
    return labels


def paper_preview() -> None:
    document = fitz.open()
    page = document.new_page(width=210 * PT_PER_MM, height=297 * PT_PER_MM)
    page.insert_font(fontname="PreviewSans", fontfile=findfont(FontProperties(family="DejaVu Sans")))
    page.insert_font(fontname="PreviewSerif", fontfile=findfont(FontProperties(family="DejaVu Serif")))

    def text(x_mm: float, y_mm: float, value: str, size: float = 9, font: str = "PreviewSans") -> None:
        page.insert_text((x_mm * PT_PER_MM, y_mm * PT_PER_MM), value, fontsize=size, fontname=font)

    def rect(x: float, y: float, width: float, height: float) -> fitz.Rect:
        return fitz.Rect(x * PT_PER_MM, y * PT_PER_MM, (x + width) * PT_PER_MM, (y + height) * PT_PER_MM)

    text(15, 14, "Figure 14: single-column layout check", 14)
    text(15, 21, "A4 preview at 85 mm column width. Not the publisher's typeset proof.", 9)
    text(15, 31, "New: shared legend, larger plots", 10)
    text(110, 31, "Previous version: same type size", 10)
    text(110, 114, "Figure 10: typography reference", 10)
    caption = ("Fig. 14. Resource landscape for SOC and SOH models. The panels compare exported "
               "parameter count, idealized parameter storage, flash occupancy of the available firmware "
               "builds, and recorded static-plus-peak-stack RAM. Parameter counts include weights and "
               "merged biases in the C exports. The storage estimate assumes four bytes per parameter "
               "for Base and Pruned and one byte for Quantized. The all-INT8 estimate is idealized because "
               "the implemented quantized model retains FP32 parameters and additionally stores row "
               "scales. Dark bars denote SOC, light bars denote SOH. Memory is expressed in KiB.")
    caption_y = 35 + HEIGHT_MM + 4
    remaining = page.insert_textbox(rect(15, caption_y, WIDTH_MM, 280 - caption_y), caption, fontsize=8,
                                    fontname="PreviewSerif", lineheight=1.15)
    assert remaining >= 0, "Preview caption does not fit"
    text(15, 268, "All figures shown at the same 85 mm width. Print at 100% to compare type sizes.", 8)
    text(15, 275, "Type sizes and all 24 values are unchanged. The new figure is slightly taller (67 mm).", 8)
    # Place external resources after text insertion to retain the page's XObjects.
    with fitz.open(REVIEW / "gr14.pdf") as source:
        page.show_pdf_page(rect(15, 35, WIDTH_MM, HEIGHT_MM), source, 0)
    with fitz.open(REVIEW / "archive/figure14_v3_original_style/gr14.pdf") as source:
        old_height = WIDTH_MM * source[0].rect.height / source[0].rect.width
        page.show_pdf_page(rect(110, 35, WIDTH_MM, old_height), source, 0)
    page.insert_image(rect(110, 119, WIDTH_MM, WIDTH_MM * 5 / 12),
                      filename=str(REVIEW / "sources/figure10_style_reference.png"))
    document.save(REVIEW / "preview/figure14_paper_preview.pdf", garbage=4, deflate=True)
    document.close()


def render_previews() -> None:
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("Poppler pdftoppm is required for visual QA")
    for source, target, resolution in (
        (REVIEW / "gr14.pdf", REVIEW / "preview/gr14_render_v4", 1700),
        (REVIEW / "preview/figure14_paper_preview.pdf", REVIEW / "preview/paper_render_v4", 2000),
    ):
        subprocess.run([renderer, "-scale-to", str(resolution), "-singlefile", "-png",
                        str(source), str(target)], check=True)


def verify(labels: list[str]) -> dict:
    with fitz.open(REVIEW / "gr14.pdf") as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        assert abs(page.rect.width / PT_PER_MM - WIDTH_MM) < 0.01
        assert abs(page.rect.height / PT_PER_MM - HEIGHT_MM) < 0.01
        assert not page.get_images(), "Figure should be vector-based"
        text = page.get_text()
        assert all(label in text for label in labels)
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        minimum = min(span["size"] for span in spans)
        assert minimum >= FONT_LEGEND - 0.01
        assert max(span["size"] for span in spans) <= FONT_TITLE + 0.01
        assert all(pdf.extract_font(font[0])[3] for font in page.get_fonts())
        assert text.count("SOC (left, darker)") == 1
        assert text.count("SOH (right, lighter)") == 1
        with fitz.open(REVIEW / "archive/figure14_v3_original_style/gr14.pdf") as previous:
            previous_spans = [span for block in previous[0].get_text("dict")["blocks"] if "lines" in block
                              for line in block["lines"] for span in line["spans"]]
            previous_fonts = {}
            for span in previous_spans:
                previous_fonts.setdefault(span["text"], set()).add(round(span["size"], 3))
            for span in spans:
                if span["text"] in previous_fonts:
                    assert round(span["size"], 3) in previous_fonts[span["text"]], "Type size changed"
    with fitz.open(REVIEW / "preview/figure14_paper_preview.pdf") as preview:
        assert len(preview) == 1
        assert "Not the publisher's typeset proof" in preview[0].get_text()
        assert len(preview[0].get_drawings()) > 50, "Preview vector figures missing"
        assert len(preview[0].get_images()) == 1, "Preview typography reference missing"
        assert all(label in preview[0].get_text() for label in labels)
        assert all(preview.extract_font(font[0])[3] for font in preview[0].get_fonts())
    baseline = (REVIEW / "archive/Neptune.tex").read_text(encoding="utf-8")
    updated = (REVIEW / "Neptune.tex").read_text(encoding="utf-8")
    start = baseline.rindex(r"\begin{figure*}", 0, baseline.index(r"\xlabel{fig:resources_sizes}"))
    end = baseline.index(r"\end{figure*}", start) + len(r"\end{figure*}")
    block = baseline[start:end].replace("{figure*}", "{figure}")
    block = block.replace(r"\includegraphics{gr14}", r"\includegraphics[width=\columnwidth]{gr14}")
    expected = baseline[:start] + block + baseline[end:]
    expected = expected.replace(r"\includegraphics{gr19}", r"\includegraphics[width=\textwidth]{gr19}")
    assert updated == expected, "Unexpected manuscript changes"
    assert (REVIEW / "gr19.pdf").read_bytes() == (REVIEW / "archive/gr19.pdf").read_bytes()
    for name in ("Neptune.tex", "gr14.pdf", "gr19.pdf", "Neptune_Korrekturen_20261001.zip"):
        assert sha256(REVIEW.parent / name) == sha256(REVIEW / "archive" / name), "Original changed"
    return {"column_width_mm": WIDTH_MM, "height_mm": HEIGHT_MM,
            "minimum_figure_font_pt": round(minimum, 2), "value_labels_verified": len(labels),
            "vector_graphics": True, "embedded_fonts": True, "no_value_label_overlap": True,
            "no_value_labels_intersect_bars": True,
            "no_legends_intersect_data": True,
            "resource_data_unchanged": True, "original_files_unchanged": True,
            "panel_layout": "2x2 matrix; adjacent SOC/SOH bars; horizontal labels; one shared top legend",
            "type_sizes_unchanged_vs_v3": True,
            "number_format_unchanged": True,
            "plot_width_change_percent": 100 * ((0.87 / 2.32) / (0.86 / 2.43) - 1),
            "plot_height_change_percent": 100 * ((67 * 0.80 / 2.40) / ((85 * 10 / 14) * 0.83 / 2.44) - 1),
            "figure10_equivalent_font_pt": {"title": FONT_TITLE, "axis": FONT_AXIS,
                                            "tick": FONT_TICK, "legend": FONT_LEGEND},
            "value_label_font_pt": FONT_VALUE,
            "manuscript_changes": "Figure 14 environment/width; Figure 19 width; original panel reference restored",
            "publisher_proof_compiled": False,
            "proof_limitation": "neptune.cls and complete publisher figure package unavailable"}


def main() -> None:
    records = load_records()
    labels = plot_figure(records)
    paper_preview()
    render_previews()
    results = verify(labels)
    (REVIEW / "validation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with ZipFile(REVIEW / "Neptune_review_2_replacements.zip", "w", ZIP_DEFLATED) as package:
        for name in ("Neptune.tex", "gr14.pdf", "gr19.pdf", "UPLOAD.txt"):
            package.write(REVIEW / name, name)
        package.write(REVIEW / "gr14.png", "PNG_alternative/gr14.png")
    with ZipFile(REVIEW / "Neptune_review_2_replacements.zip") as package:
        assert package.testzip() is None
        for name in ("Neptune.tex", "gr14.pdf", "gr19.pdf", "UPLOAD.txt"):
            assert package.read(name) == (REVIEW / name).read_bytes()
    files = [path for path in REVIEW.rglob("*") if path.is_file()
             and path.name != "manifest.json" and "__pycache__" not in path.parts]
    manifest = {"revision_date": "2026-10-02", "scope": "Figure 14 v4: shared top legend, larger plots, unchanged typography",
                "files": {path.relative_to(REVIEW).as_posix(): sha256(path) for path in sorted(files)}}
    (REVIEW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
