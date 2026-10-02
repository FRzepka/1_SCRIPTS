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
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pymupdf as fitz


REVIEW = Path(__file__).resolve().parents[1]
MODELS = ("Base", "Pruned", "Quantized")
TASKS = ("SOC", "SOH")
COLORS = ("#2ca02c", "#d62728", "#1f77b4")
WIDTH_MM = 85
HEIGHT_MM = 110
PT_PER_MM = 72 / 25.4


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
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.titleweight": "normal",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.84, wspace=0.53, hspace=0.66)
    fig.legend(handles=[
        Patch(facecolor=tint("#000000", 0.40), edgecolor="#555555", label="SOC (dark)"),
        Patch(facecolor=tint("#000000", 0.22), edgecolor="#999999", label="SOH (light)"),
    ], loc="upper center", bbox_to_anchor=(0.51, 0.99), ncol=2, frameon=False,
        fontsize=8, handlelength=1.3, columnspacing=1.1, handletextpad=0.5)
    titles = ("(a) Exported\nparameters", "(b) Idealized\nstorage [KiB]",
              "(c) Firmware\nflash [KiB]", "(d) RAM [KiB]\nstatic + peak stack")
    labels = []
    annotations = []
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
                bars = ax.bar(model_index + (task_index - 0.5) * 0.36, value, 0.36,
                              facecolor=tint(color, 0.40 if task_index == 0 else 0.22),
                              edgecolor=color if task_index == 0 else tint(color, 0.65),
                              linewidth=0.65, zorder=3)
                label = f"{value:,}" if panel == 0 else f"{value:.2f}"
                labels.append(label)
                annotations.extend(ax.bar_label(bars, labels=[label], padding=2.5,
                                                fontsize=7, rotation=90))
        ax.set(title=title, xticks=range(3), xticklabels=MODELS,
               xlim=(-0.65, 2.65), ylim=(0, maximum * 1.65))
        ax.set_title(title, pad=6)
        ax.tick_params(axis="both", length=2.5, pad=2, width=0.6)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        if panel == 0:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer) for item in annotations]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), "Value labels overlap"
    for item in annotations + [ax.title for ax in axes.flat]:
        box = item.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), "Clipped figure text"
    fig.savefig(REVIEW / "gr14.pdf", metadata={
        "Title": "Figure 14 - single-column resource comparison",
        "Subject": "Unchanged audited resource data; 85 mm single-column layout",
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
    text(15, 31, "New single-column version", 10)
    text(110, 31, "Previous version at the same width", 10)
    text(110, 114, "Latency figure: style reference", 10)
    caption = ("Fig. 14. Resource landscape for SOC and SOH models. The panels compare exported "
               "parameter count, idealized parameter storage, flash occupancy of the available firmware "
               "builds, and recorded static-plus-peak-stack RAM. Parameter counts include weights and "
               "merged biases in the C exports. The storage estimate assumes four bytes per parameter "
               "for Base and Pruned and one byte for Quantized. The all-INT8 estimate is idealized because "
               "the implemented quantized model retains FP32 parameters and additionally stores row "
               "scales. Dark bars denote SOC, light bars denote SOH. Memory is expressed in KiB.")
    remaining = page.insert_textbox(rect(15, 149, WIDTH_MM, 90), caption, fontsize=8,
                                    fontname="PreviewSerif", lineheight=1.15)
    assert remaining >= 0, "Preview caption does not fit"
    text(15, 258, "Print at 100% to judge physical type size; the final column width is set by the journal.", 8)
    text(15, 264, "Data and 2 x 2 panel order are unchanged. PDF remains vector-based with embedded fonts.", 8)
    # Place external resources after text insertion to retain the page's XObjects.
    with fitz.open(REVIEW / "gr14.pdf") as source:
        page.show_pdf_page(rect(15, 35, WIDTH_MM, HEIGHT_MM), source, 0)
    with fitz.open(REVIEW / "archive/gr14.pdf") as source:
        old_height = WIDTH_MM * source[0].rect.height / source[0].rect.width
        page.show_pdf_page(rect(110, 35, WIDTH_MM, old_height), source, 0)
    page.insert_image(rect(110, 119, WIDTH_MM, WIDTH_MM * 750 / 1800),
                      filename=str(REVIEW / "sources/latency_style_reference.png"))
    document.save(REVIEW / "preview/figure14_paper_preview.pdf", garbage=4, deflate=True)
    document.close()


def render_previews() -> None:
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("Poppler pdftoppm is required for visual QA")
    for source, target, resolution in (
        (REVIEW / "gr14.pdf", REVIEW / "preview/gr14_render", 1700),
        (REVIEW / "preview/figure14_paper_preview.pdf", REVIEW / "preview/paper_render", 1800),
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
        assert minimum >= 6.99
        assert all(pdf.extract_font(font[0])[3] for font in page.get_fonts())
    with fitz.open(REVIEW / "preview/figure14_paper_preview.pdf") as preview:
        assert len(preview) == 1
        assert "Not the publisher's typeset proof" in preview[0].get_text()
        assert len(preview[0].get_drawings()) > 50, "Preview vector figures missing"
        assert len(preview[0].get_images()) == 1, "Preview latency reference missing"
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
            "resource_data_unchanged": True, "original_files_unchanged": True,
            "manuscript_changes": "Only Figure 14 environment/width and explicit Figure 19 width",
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
    manifest = {"revision_date": "2026-10-01", "scope": "Figure 14 single-column typography",
                "files": {path.relative_to(REVIEW).as_posix(): sha256(path) for path in sorted(files)}}
    (REVIEW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
