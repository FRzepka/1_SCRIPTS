from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pictures"

PALETTES = {
    "red_monochrome": {
        "categorical": [
            (0x7C, 0x1F, 0x1D),
            (0xB6, 0x30, 0x2D),
            (0xD6, 0x76, 0x6D),
            (0xED, 0xBD, 0xB8),
        ],
        "heatmap": [
            (0x72, 0x17, 0x16),
            (0xA4, 0x28, 0x26),
            (0xCB, 0x5B, 0x53),
            (0xE7, 0xA6, 0x9F),
        ],
    },
    "red_muted": {
        "categorical": [
            (0xB6, 0x30, 0x2D),
            (0xD1, 0x88, 0x7E),
            (0x8B, 0x67, 0x63),
            (0x56, 0x6B, 0x78),
        ],
        "heatmap": [
            (0x72, 0x17, 0x16),
            (0xB6, 0x30, 0x2D),
            (0xD1, 0x88, 0x7E),
            (0xEA, 0xC0, 0xBA),
        ],
    },
}

HEATMAP_FILES = {
    "paper1_correlation.png",
    "paper1_correlation_rb.png",
    "paper1_mae_matrix.png",
    "paper1_mae_matrix_rb.png",
    "robustness_cross_scenario.png",
}

ROBUSTNESS_MODEL_FILES = {
    "robustness_adc_quantization.png",
    "robustness_baseline.png",
    "robustness_current_bias.png",
    "robustness_dropout_recovery.png",
    "robustness_dropout_transition.png",
    "robustness_init_recovery.png",
    "robustness_noise.png",
    "robustness_signal_integrity.png",
    "robustness_spike_response.png",
}

SOLID_FILL_FILES = {
    "embedded_latency_hist.png",
    "embedded_model_sizes.png",
    "embedded_model_sizes_original.png",
    "robustness_baseline.png",
}

RIGHT_HALF_SOLID_FILL_FILES = {
    "robustness_adc_quantization.png",
    "robustness_signal_integrity.png",
}

RED_MUTED_ALIASES = {
    "embedded_architecture_rb.png": "embedded_architecture.png",
    "robustness_dd_architecture_rb.png": "robustness_dd_architecture.png",
    "robustness_decision_synthesis.png": "robustness_decision.png",
}

SOURCE_RENDERED_RED_MUTED_FILES = {
    # These figures are regenerated from the EAAI benchmark arrays by
    # tools/render_eaai_red_muted_figures.py. Do not overwrite them with
    # pixel-level recoloring during a palette refresh.
    "embedded_soc_error.png",
    "embedded_soh_error.png",
    "embedded_soc_zoom_checkup.png",
    "embedded_soc_zoom_pulse.png",
    # Regenerated from the JES benchmark tables by
    # tools/render_jes_red_muted_figures.py.
    "robustness_decision.png",
}

SKIP_FILES = {
    # Keep the real PCB render in original colors. The LaTeX graphicspath falls
    # back to pictures/ when this file is absent from a palette folder.
    "bms_board_render.png",
}

MIXED_MONOCHROME_FILES = {
    "paper1_soh_cycles.png",
    "paper1_soh_cycles_rb.png",
    "paper1_soh_time.png",
    "paper1_soh_time_rb.png",
    "embedded_soh_all_days.png",
    "embedded_soh_all_days_rb.png",
    "robustness_disturbance_taxonomy.png",
}

MIXED_BMS_REQUIREMENT_FILES = {
    "bms_requirements.png",
    "bms_requirements_rb.png",
    "bms_requirements_pdfsafe.png",
}

MIXED_SIX_COLOR_FILES = {
    "embedded_soc_test_trajectory.png",
    "embedded_soc_test_trajectory_rb.png",
}

SIX_COLOR_PALETTE = [
    (0xB6, 0x30, 0x2D),  # TU red
    (0x6B, 0x7F, 0x8F),  # blue-gray
    (0xB8, 0x8A, 0x5A),  # ochre
    (0x6F, 0x85, 0x7D),  # green-gray
    (0x1F, 0x24, 0x2A),  # near black
    (0xE0, 0xA3, 0x9B),  # soft rose
]

SCRIPTS_ROOT = ROOT.parents[2]

CLASS_ORDER = [
    "Direct measurement",
    "Hybrid direct measurement",
    "Hybrid ECM",
    "Data-driven",
]

CLASS_SHORT = {
    "Direct measurement": "DM",
    "Hybrid direct measurement": "HDM",
    "Hybrid ECM": "HECM",
    "Data-driven": "DD",
}

SHORT_TO_COLOR = {
    "DM": PALETTES["red_muted"]["categorical"][0],
    "HDM": PALETTES["red_muted"]["categorical"][1],
    "HECM": PALETTES["red_muted"]["categorical"][2],
    "DD": PALETTES["red_muted"]["categorical"][3],
}

EMBEDDED_COLORS = {
    "Base": PALETTES["red_muted"]["categorical"][2],
    "Pruned": PALETTES["red_muted"]["categorical"][0],
    "Quantized": PALETTES["red_muted"]["categorical"][3],
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf", "calibri.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    w, h = _text_size(draw, text, font)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def _rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    angle: int = 90,
) -> None:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    w, h = _text_size(d, text, font)
    layer = Image.new("RGBA", (w + 12, h + 12), (255, 255, 255, 0))
    ld = ImageDraw.Draw(layer)
    ld.text((6, 6), text, font=font, fill=fill + (255,))
    layer = layer.rotate(angle, expand=True)
    image.alpha_composite(layer, (int(xy[0] - layer.width / 2), int(xy[1] - layer.height / 2)))


def _read_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8").splitlines()
    lines = [line.rstrip() for line in text if line.strip()]
    if not lines:
        return []
    if lines[0].lstrip().startswith("|"):
        rows = []
        headers: list[str] | None = None
        for line in lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not cells:
                continue
            if all(set(cell) <= {":", "-"} for cell in cells):
                continue
            if headers is None:
                headers = cells
                continue
            rows.append(dict(zip(headers, cells)))
        return rows
    return list(csv.DictReader(lines))


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    value = value.strip()
    if not value or value.lower() == "nan":
        return float("nan")
    return float(value)


def _lower_better(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    vmin = min(finite)
    vmax = max(finite)
    if math.isclose(vmax, vmin):
        return [1.0 for _ in values]
    return [(vmax - v) / (vmax - vmin) if math.isfinite(v) else 0.0 for v in values]


def _penalized_lower_better(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    vmin = min(finite)
    vmax = max(finite)
    penalty = max(vmax * 1.25, vmin + 1e-6)
    if math.isclose(penalty, vmin):
        return [1.0 for _ in values]
    return [(penalty - (v if math.isfinite(v) else penalty)) / (penalty - vmin) for v in values]


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    ticks: list[float],
    y_label: str,
    image: Image.Image,
    tick_formatter=None,
    label_offset: int = 150,
) -> None:
    x0, y0, x1, y1 = plot
    axis_color = (70, 70, 70)
    grid_color = (220, 220, 220)
    tick_font = _font(28)
    label_font = _font(34)
    draw.line((x0, y1, x1, y1), fill=axis_color, width=3)
    draw.line((x0, y0, x0, y1), fill=axis_color, width=3)
    for tick in ticks:
        y = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=grid_color, width=2)
        draw.line((x0 - 8, y, x0, y), fill=axis_color, width=2)
        if tick_formatter is not None:
            label = tick_formatter(tick)
        else:
            label = f"{tick:g}" if abs(tick) >= 1 else f"{tick:.3f}".rstrip("0").rstrip(".")
        w, h = _text_size(draw, label, tick_font)
        draw.text((x0 - 18 - w, y - h / 2), label, font=tick_font, fill=(40, 40, 40))
    _rotated_text(image, (x0 - label_offset, (y0 + y1) // 2), y_label, label_font, (20, 20, 20), 90)


def _draw_hatched_bar(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    hatch: str,
    fill: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 5,
) -> None:
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return
    patch = Image.new("RGBA", (x1 - x0, y1 - y0), fill + (255,))
    pd = ImageDraw.Draw(patch)
    if hatch == "//":
        spacing = 22
        for offset in range(-(y1 - y0), x1 - x0 + spacing, spacing):
            pd.line((offset, y1 - y0, offset + (y1 - y0), 0), fill=color + (255,), width=line_width)
    elif hatch == "..":
        spacing = 18
        radius = 4
        for yy in range(8, y1 - y0, spacing):
            for xx in range(8, x1 - x0, spacing):
                pd.ellipse((xx - radius, yy - radius, xx + radius, yy + radius), fill=color + (255,))
    pd.rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), outline=color + (255,), width=line_width)
    image.alpha_composite(patch, (x0, y0))


def _draw_solid_bar(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 4,
) -> None:
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return
    draw.rectangle(rect, fill=fill, outline=outline or fill, width=width)


def _render_signal_integrity(out_path: Path) -> None:
    table_dir = SCRIPTS_ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_tables_v4"
    key = _read_rows(table_dir / "table_key_results.md")
    local = _read_rows(table_dir / "table_local_behaviour.md")

    scenarios = [
        ("Missing samples", "//"),
        ("Irregular sampling", ".."),
        ("Burst dropout", ""),
    ]
    delta: dict[tuple[str, str], float] = {}
    for row in key:
        if row.get("scenario_label") in {s[0] for s in scenarios}:
            delta[(row["class"], row["scenario_label"])] = _to_float(row.get("delta_mae"))

    recovery: dict[str, float] = {}
    for row in local:
        if row.get("focus_scenario") == "missing_gap" and row.get("local_metric") == "recovery_time_h":
            recovery[row["class"]] = _to_float(row.get("value"))

    width, height = 2798, 1068
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(40, bold=True)
    tick_font = _font(30)
    label_font = _font(34)
    legend_font = _font(27)

    left_plot = (190, 170, 1285, 900)
    right_plot = (1640, 170, 2685, 900)
    _draw_axes(draw, left_plot, -0.0015, 0.009, [0.0, 0.002, 0.004, 0.006, 0.008], "\N{GREEK CAPITAL LETTER DELTA}MAE", image)
    _draw_axes(draw, right_plot, 0.0, 30.0, [0, 5, 10, 15, 20, 25, 30], "Recovery time [h]", image)
    draw.text((left_plot[0], 80), "(a)", font=title_font, fill=(20, 20, 20))
    draw.text((right_plot[0], 80), "(b)", font=title_font, fill=(20, 20, 20))

    def y_left(value: float) -> int:
        return round(left_plot[3] - (value + 0.0015) / (0.009 + 0.0015) * (left_plot[3] - left_plot[1]))

    def y_right(value: float) -> int:
        return round(right_plot[3] - value / 30.0 * (right_plot[3] - right_plot[1]))

    group_step = (left_plot[2] - left_plot[0]) / len(CLASS_ORDER)
    centers = [left_plot[0] + group_step * (i + 0.5) for i in range(len(CLASS_ORDER))]
    bar_w = 64
    offsets = [-76, 0, 76]
    zero_y = y_left(0.0)
    for gi, cls in enumerate(CLASS_ORDER):
        short = CLASS_SHORT[cls]
        color = SHORT_TO_COLOR[short]
        for si, (scenario, hatch) in enumerate(scenarios):
            value = delta.get((cls, scenario), float("nan"))
            if not math.isfinite(value):
                continue
            x = round(centers[gi] + offsets[si])
            y = y_left(value)
            rect = (x - bar_w // 2, min(y, zero_y), x + bar_w // 2, max(y, zero_y))
            if rect[3] - rect[1] < 6:
                rect = (rect[0], zero_y - 3 if value >= 0 else zero_y, rect[2], zero_y + 3 if value < 0 else zero_y)
            if hatch:
                _draw_hatched_bar(image, rect, color, hatch)
            else:
                _draw_solid_bar(draw, rect, color, color, width=4)
        _center_text(draw, (centers[gi], left_plot[3] + 52), short, tick_font)

    group_step = (right_plot[2] - right_plot[0]) / len(CLASS_ORDER)
    centers = [right_plot[0] + group_step * (i + 0.5) for i in range(len(CLASS_ORDER))]
    bar_w = 120
    for gi, cls in enumerate(CLASS_ORDER):
        short = CLASS_SHORT[cls]
        color = SHORT_TO_COLOR[short]
        value = recovery.get(cls, float("nan"))
        if math.isfinite(value):
            rect = (round(centers[gi] - bar_w / 2), y_right(value), round(centers[gi] + bar_w / 2), right_plot[3])
            _draw_solid_bar(draw, rect, color, color, width=4)
            txt = f"{value:.1f}"
            text_fill = (255, 255, 255) if short in {"DM", "HECM", "DD"} else (35, 35, 35)
            _center_text(draw, (centers[gi], y_right(value) + 32), txt, tick_font, text_fill)
        _center_text(draw, (centers[gi], right_plot[3] + 52), short, tick_font)

    legend_x, legend_y = left_plot[2] - 410, left_plot[1] + 28
    legend_items = scenarios
    draw.rounded_rectangle((legend_x - 18, legend_y - 18, legend_x + 390, legend_y + 158), radius=8, fill=(255, 255, 255), outline=(170, 170, 170), width=2)
    for i, (label, hatch) in enumerate(legend_items):
        y = legend_y + i * 52
        box = (legend_x, y, legend_x + 44, y + 32)
        if hatch:
            _draw_hatched_bar(image, box, (90, 90, 90), hatch, line_width=3)
        else:
            _draw_solid_bar(draw, box, (150, 150, 150), (90, 90, 90), width=3)
        draw.text((legend_x + 62, y - 1), label, font=legend_font, fill=(30, 30, 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def _decision_scores() -> tuple[list[dict[str, float | str]], list[str]]:
    table_dir = SCRIPTS_ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/paper_tables_v4"
    baseline = _read_rows(table_dir / "table_baseline.md")
    key = _read_rows(table_dir / "table_key_results.md")
    local = _read_rows(table_dir / "table_local_behaviour.md")
    robustness_scenarios = [
        "Current noise (high)",
        "Current bias",
        "Irregular sampling",
        "Burst dropout",
        "Missing samples",
        "Voltage spikes",
        "Temperature noise",
        "Voltage noise",
    ]
    by_class = {row["class"]: row for row in baseline}
    accuracy_parts = []
    for metric in ["mae", "rmse", "p95_error"]:
        values = [_to_float(by_class[cls].get(metric)) for cls in CLASS_ORDER]
        accuracy_parts.append(_lower_better(values))
    accuracy = [sum(parts[i] for parts in accuracy_parts) / len(accuracy_parts) for i in range(len(CLASS_ORDER))]

    key_lookup = {(row["class"], row["scenario_label"]): _to_float(row.get("delta_mae")) for row in key}
    robustness_parts = []
    for scenario in robustness_scenarios:
        values = [key_lookup.get((cls, scenario), float("nan")) for cls in CLASS_ORDER]
        robustness_parts.append(_lower_better(values))
    robustness = [sum(parts[i] for parts in robustness_parts) / len(robustness_parts) for i in range(len(CLASS_ORDER))]

    local_lookup = {}
    for row in local:
        if row.get("local_metric") == "recovery_time_to_baseline_band_strict_h":
            local_lookup[row["class"]] = _to_float(row.get("value"))
    recovery = _penalized_lower_better([local_lookup.get(cls, float("nan")) for cls in CLASS_ORDER])

    rows: list[dict[str, float | str]] = []
    for idx, cls in enumerate(CLASS_ORDER):
        rows.append(
            {
                "Model": CLASS_SHORT[cls],
                "Class": cls,
                "Accuracy": accuracy[idx],
                "Robustness": robustness[idx],
                "Recovery": recovery[idx],
            }
        )
    profiles = ["Accuracy-weighted", "Robustness-weighted", "Recovery-weighted"]
    weights = {
        "Accuracy-weighted": {"Accuracy": 0.60, "Robustness": 0.20, "Recovery": 0.20},
        "Robustness-weighted": {"Accuracy": 0.20, "Robustness": 0.60, "Recovery": 0.20},
        "Recovery-weighted": {"Accuracy": 0.20, "Robustness": 0.20, "Recovery": 0.60},
    }
    for row in rows:
        for profile in profiles:
            row[profile] = sum(float(row[key]) * value for key, value in weights[profile].items())
    return rows, profiles


def _render_decision(out_path: Path) -> None:
    rows, profiles = _decision_scores()
    width, height = 3172, 1400
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(42, bold=True)
    tick_font = _font(28)
    label_font = _font(34)
    legend_font = _font(30)

    draw.text((185, 70), "(a) Relative decision dimensions", font=title_font, fill=(20, 20, 20))
    draw.text((1390, 70), "(b) Priority-weighted composite scores", font=title_font, fill=(20, 20, 20))

    center = (670, 730)
    radius = 410
    labels = ["Accuracy", "Robustness", "Recovery"]
    angles = [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]
    grid_color = (215, 215, 215)
    axis_color = (90, 90, 90)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = [
            (center[0] + math.cos(a) * radius * frac, center[1] + math.sin(a) * radius * frac)
            for a in angles
        ]
        draw.line(pts + [pts[0]], fill=grid_color, width=3)
        _center_text(draw, (center[0] + 14, center[1] - radius * frac), f"{frac:.2f}", tick_font, (80, 80, 80))
    for angle, label in zip(angles, labels):
        end = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
        draw.line((center[0], center[1], end[0], end[1]), fill=axis_color, width=3)
        lab = (center[0] + math.cos(angle) * (radius + 90), center[1] + math.sin(angle) * (radius + 75))
        _center_text(draw, lab, label, label_font)

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    polygons: list[tuple[str, list[tuple[float, float]]]] = []
    for row in rows:
        short = str(row["Model"])
        values = [float(row[label]) for label in labels]
        pts = [
            (center[0] + math.cos(angle) * radius * value, center[1] + math.sin(angle) * radius * value)
            for angle, value in zip(angles, values)
        ]
        polygons.append((short, pts))
        color = SHORT_TO_COLOR[short]
        odraw.polygon(pts, fill=color + (38,))
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    for short, pts in polygons:
        color = SHORT_TO_COLOR[short]
        draw.line(pts + [pts[0]], fill=color, width=8, joint="curve")
        for pt in pts:
            draw.ellipse((pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8), fill=color, outline=(255, 255, 255), width=3)

    legend_x, legend_y = 1045, 205
    for i, row in enumerate(rows):
        short = str(row["Model"])
        color = SHORT_TO_COLOR[short]
        y = legend_y + i * 55
        draw.rectangle((legend_x, y, legend_x + 42, y + 28), fill=color, outline=color)
        draw.text((legend_x + 58, y - 4), short, font=legend_font, fill=(30, 30, 30))

    plot = (1395, 210, 3000, 1120)
    _draw_axes(draw, plot, 0.0, 1.02, [0.0, 0.25, 0.5, 0.75, 1.0], "Composite score", image)
    group_step = (plot[2] - plot[0]) / len(profiles)
    group_centers = [plot[0] + group_step * (i + 0.5) for i in range(len(profiles))]
    bar_w = 70
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]

    def y_bar(value: float) -> int:
        return round(plot[3] - value / 1.02 * (plot[3] - plot[1]))

    for gi, profile in enumerate(profiles):
        for mi, row in enumerate(rows):
            short = str(row["Model"])
            color = SHORT_TO_COLOR[short]
            value = float(row[profile])
            x = round(group_centers[gi] + offsets[mi])
            rect = (x - bar_w // 2, y_bar(value), x + bar_w // 2, plot[3])
            _draw_solid_bar(draw, rect, color, color, width=3)
        label = profile.replace("-weighted", "-weighted")
        _center_text(draw, (group_centers[gi], plot[3] + 58), label, tick_font)

    legend_x, legend_y = 1850, 130
    for i, row in enumerate(rows):
        short = str(row["Model"])
        color = SHORT_TO_COLOR[short]
        x = legend_x + i * 185
        draw.rectangle((x, legend_y, x + 42, legend_y + 26), fill=color, outline=color)
        draw.text((x + 55, legend_y - 5), short, font=legend_font, fill=(30, 30, 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def interpolate_palette(t: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    palette = np.array(colors, dtype=np.float32)
    x = np.clip(t, 0.0, 1.0) * (len(colors) - 1)
    i0 = np.floor(x).astype(np.int16)
    i1 = np.clip(i0 + 1, 0, len(colors) - 1)
    frac = (x - i0)[..., None]
    return palette[i0] * (1.0 - frac) + palette[i1] * frac


def lighten(color: tuple[int, int, int], amount: float) -> np.ndarray:
    c = np.array(color, dtype=np.float32)
    return c * (1.0 - amount) + 255.0 * amount


def convert_heatmap_array(arr: np.ndarray, heatmap_colors: list[tuple[int, int, int]]) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    out = rgb.copy()

    pure_white = (maxc > 252) & (chroma < 6)
    black_text = (maxc < 55) | ((minc < 35) & (chroma < 45))
    dark_antialias = (chroma < 15) & (lum < 175)
    mask = ~(pure_white | black_text | dark_antialias)

    if np.any(mask):
        values = lum[mask]
        low = np.percentile(values, 2)
        high = np.percentile(values, 98)
        if high <= low:
            high = low + 1.0
        t = (lum - low) / (high - low)
        # Keep the lightest values visibly tinted and slightly darker than white.
        t = np.clip(0.04 + 0.84 * t, 0.0, 0.88)
        out[mask] = interpolate_palette(t, heatmap_colors)[mask]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.concatenate([out, alpha], axis=-1)


def convert_robustness_model_array(arr: np.ndarray, categorical: list[tuple[int, int, int]]) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4]

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    out = rgb.copy()
    white = (maxc > 245) & (chroma < 12)
    black = (maxc < 55) | ((minc < 35) & (chroma < 45))
    gray = (chroma < 18) & ~white & ~black
    colored = ~(white | black | gray)

    out[gray] = np.repeat(lum[..., None], 3, axis=-1)[gray]

    delta = np.maximum(chroma, 1.0)
    hue = np.zeros_like(maxc)
    rmax = (maxc == r) & colored
    gmax = (maxc == g) & colored
    bmax = (maxc == b) & colored
    hue[rmax] = (((g[rmax] - b[rmax]) / delta[rmax]) % 6.0) * 60.0
    hue[gmax] = (((b[gmax] - r[gmax]) / delta[gmax]) + 2.0) * 60.0
    hue[bmax] = (((r[bmax] - g[bmax]) / delta[bmax]) + 4.0) * 60.0

    colors = np.array(categorical, dtype=np.float32)
    base = np.zeros_like(rgb)

    light_purple = (hue >= 245) & (hue < 285) & ((maxc > 210) | (chroma < 110))
    dark_purple = (hue >= 245) & (hue < 285) & ~light_purple
    cyan = (hue >= 165) & (hue < 195)
    blue = (hue >= 195) & (hue < 245)
    red_orange = (hue < 80) | (hue >= 330)
    green_yellow = (hue >= 80) & (hue < 165)
    other_purple = (hue >= 285) & (hue < 330)

    base[dark_purple | red_orange | other_purple] = colors[0]
    base[cyan | green_yellow] = colors[1]
    base[light_purple] = colors[2]
    base[blue] = colors[3]
    base[colored & (base.sum(axis=-1) == 0)] = colors[1]

    out[colored] = base[colored]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.concatenate([out, alpha], axis=-1)


def convert_decision_array(arr: np.ndarray, categorical: list[tuple[int, int, int]]) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4]

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    out = rgb.copy()
    white = (maxc > 245) & (chroma < 12)
    black = (maxc < 55) | ((minc < 35) & (chroma < 45))
    gray = (chroma < 18) & ~white & ~black
    colored = ~(white | black | gray)
    out[gray] = np.repeat(lum[..., None], 3, axis=-1)[gray]

    delta = np.maximum(chroma, 1.0)
    hue = np.zeros_like(maxc)
    rmax = (maxc == r) & colored
    gmax = (maxc == g) & colored
    bmax = (maxc == b) & colored
    hue[rmax] = (((g[rmax] - b[rmax]) / delta[rmax]) % 6.0) * 60.0
    hue[gmax] = (((b[gmax] - r[gmax]) / delta[gmax]) + 2.0) * 60.0
    hue[bmax] = (((r[bmax] - g[bmax]) / delta[bmax]) + 4.0) * 60.0

    colors = np.array(categorical, dtype=np.float32)
    base = np.zeros_like(rgb)
    light_purple = (hue >= 245) & (hue < 285) & ((maxc > 210) | (chroma < 110))
    dark_purple = (hue >= 245) & (hue < 285) & ~light_purple
    cyan = (hue >= 165) & (hue < 195)
    blue = (hue >= 195) & (hue < 245)
    red_orange = (hue < 80) | (hue >= 330)
    green_yellow = (hue >= 80) & (hue < 165)
    other_purple = (hue >= 285) & (hue < 330)

    base[dark_purple | red_orange | other_purple] = colors[0]
    base[cyan | green_yellow] = colors[1]
    base[light_purple] = colors[2]
    base[blue] = colors[3]
    base[colored & (base.sum(axis=-1) == 0)] = colors[1]

    height, width = hue.shape
    xx = np.broadcast_to(np.linspace(0.0, 1.0, width, dtype=np.float32), hue.shape)
    yy = np.broadcast_to(np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], hue.shape)
    radar_fill = colored & (xx < 0.48) & (yy > 0.08) & ((lum > 145) | (chroma < 130))
    color_out = base.copy()
    color_out[radar_fill] = base[radar_fill] * 0.38 + 255.0 * 0.62
    out[colored] = color_out[colored]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.concatenate([out, alpha], axis=-1)


def fill_grid_gaps_inside_colored_regions(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32).copy()
    alpha = arr[..., 3:4]

    for _ in range(9):
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        chroma = maxc - minc
        black = (maxc < 65) | ((minc < 40) & (chroma < 55))
        colored = (chroma > 20) & (maxc < 245) & ~black
        candidate = ~black

        updates = np.zeros_like(candidate, dtype=bool)
        replacement = np.zeros_like(rgb)

        left = rgb[:, :-2]
        right = rgb[:, 2:]
        center_lr = rgb[:, 1:-1]
        avg_lr = (left + right) * 0.5
        same_lr = np.linalg.norm(left - right, axis=-1) < 52
        differs_lr = np.linalg.norm(center_lr - avg_lr, axis=-1) > 8
        fill_lr = candidate[:, 1:-1] & colored[:, :-2] & colored[:, 2:] & same_lr & differs_lr
        updates[:, 1:-1] |= fill_lr
        replacement[:, 1:-1][fill_lr] = avg_lr[fill_lr]

        up = rgb[:-2, :]
        down = rgb[2:, :]
        center_ud = rgb[1:-1, :]
        avg_ud = (up + down) * 0.5
        same_ud = np.linalg.norm(up - down, axis=-1) < 52
        differs_ud = np.linalg.norm(center_ud - avg_ud, axis=-1) > 8
        fill_ud = candidate[1:-1, :] & colored[:-2, :] & colored[2:, :] & same_ud & differs_ud
        new_ud = fill_ud & ~updates[1:-1, :]
        replacement[1:-1, :][new_ud] = avg_ud[new_ud]
        updates[1:-1, :] |= fill_ud

        if not np.any(updates):
            break
        rgb[updates] = replacement[updates]

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    black = (maxc < 65) | ((minc < 40) & (chroma < 55))
    colored = (chroma > 20) & (maxc < 245) & ~black
    flat = np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 3)
    flat_mask = colored.reshape(-1)
    if np.any(flat_mask):
        colors, counts = np.unique(flat[flat_mask], axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        for color, count in zip(colors[order[:24]], counts[order[:24]]):
            if count < 450:
                continue
            mask = np.all(flat.reshape(rgb.shape) == color, axis=-1)
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            closed = mask_img.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
            add = (np.array(closed) > 0) & ~mask & ~black
            rgb[add] = color

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def fill_grid_gaps_in_right_half(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    x0 = out.shape[1] // 2
    out[:, x0:] = fill_grid_gaps_inside_colored_regions(out[:, x0:])
    return out


def adjust_embedded_model_sizes_array(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.uint8).copy()
    alpha = arr[..., 3:4]

    base_fill = np.array([0xBF, 0xAB, 0xA9], dtype=np.int16)
    base_edge = np.array([0x9C, 0x7D, 0x7A], dtype=np.int16)
    dark_fill = np.array([0x8B, 0x67, 0x63], dtype=np.uint8)
    dark_edge = np.array([0x73, 0x52, 0x4E], dtype=np.uint8)

    base_mask = np.linalg.norm(rgb.astype(np.int16) - base_fill, axis=-1) < 10
    col_counts = base_mask.sum(axis=0)
    active = np.where(col_counts > 200)[0]
    if active.size:
        intervals: list[tuple[int, int]] = []
        start = int(active[0])
        prev = int(active[0])
        for idx in active[1:]:
            idx = int(idx)
            if idx > prev + 1:
                intervals.append((start, prev))
                start = idx
            prev = idx
        intervals.append((start, prev))

        for left, right in zip(intervals[0::2], intervals[1::2]):
            if right[0] - left[1] > 24:
                continue
            x0, x1 = max(0, left[0] - 3), min(rgb.shape[1], left[1] + 4)
            region = rgb[:, x0:x1].astype(np.int16)
            fill_region = np.linalg.norm(region - base_fill, axis=-1) < 16
            edge_region = np.linalg.norm(region - base_edge, axis=-1) < 20
            rgb[:, x0:x1][fill_region] = dark_fill
            rgb[:, x0:x1][edge_region] = dark_edge

    return np.concatenate([rgb, alpha], axis=-1)


def convert_array(arr: np.ndarray, categorical: list[tuple[int, int, int]]) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4]

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    out = rgb.copy()

    white = (maxc > 245) & (chroma < 12)
    black = (minc < 35) & (chroma < 30)
    gray = (chroma < 18) & ~white & ~black
    colored = ~(white | black | gray)

    gray_value = lum[..., None]
    out[gray] = np.repeat(gray_value, 3, axis=-1)[gray]

    delta = np.maximum(chroma, 1.0)
    hue = np.zeros_like(maxc)
    rmax = (maxc == r) & colored
    gmax = (maxc == g) & colored
    bmax = (maxc == b) & colored
    hue[rmax] = (((g[rmax] - b[rmax]) / delta[rmax]) % 6.0) * 60.0
    hue[gmax] = (((b[gmax] - r[gmax]) / delta[gmax]) + 2.0) * 60.0
    hue[bmax] = (((r[bmax] - g[bmax]) / delta[bmax]) + 4.0) * 60.0

    colors = np.array(categorical, dtype=np.float32)
    base = np.zeros_like(rgb)
    base[(hue < 35) | (hue >= 330)] = colors[0]
    base[(hue >= 35) & (hue < 95)] = colors[1]
    base[(hue >= 95) & (hue < 180)] = colors[2]
    base[(hue >= 180) & (hue < 260)] = colors[3]
    base[(hue >= 260) & (hue < 330)] = colors[1]

    light = colored & (lum > 199)
    dark = colored & (lum < 56)
    low_sat = colored & (chroma < 90)
    color_out = base.copy()
    color_out[light] = base[light] * 0.55 + 255.0 * 0.45
    color_out[dark] = base[dark] * 0.65 + np.array([55.0, 20.0, 20.0]) * 0.35
    color_out[low_sat & ~light & ~dark] = base[low_sat & ~light & ~dark] * 0.85 + 255.0 * 0.15
    out[colored] = color_out[colored]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.concatenate([out, alpha], axis=-1)


def convert_six_color_array(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4]
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    out = rgb.copy()
    white = (maxc > 245) & (chroma < 12)
    black = (minc < 35) & (chroma < 30)
    gray = (chroma < 18) & ~white & ~black
    colored = ~(white | black | gray)
    out[gray] = np.repeat(lum[..., None], 3, axis=-1)[gray]

    delta = np.maximum(chroma, 1.0)
    hue = np.zeros_like(maxc)
    rmax = (maxc == r) & colored
    gmax = (maxc == g) & colored
    bmax = (maxc == b) & colored
    hue[rmax] = (((g[rmax] - b[rmax]) / delta[rmax]) % 6.0) * 60.0
    hue[gmax] = (((b[gmax] - r[gmax]) / delta[gmax]) + 2.0) * 60.0
    hue[bmax] = (((r[bmax] - g[bmax]) / delta[bmax]) + 4.0) * 60.0

    colors = np.array(SIX_COLOR_PALETTE, dtype=np.float32)
    base = np.zeros_like(rgb)
    h, w = hue.shape
    xx = np.broadcast_to(np.linspace(0.0, 1.0, w, dtype=np.float32), hue.shape)
    yy = np.broadcast_to(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], hue.shape)

    capacity = (hue >= 170) & (hue < 195)
    base[capacity & (xx < 0.25)] = colors[3]
    base[capacity & (xx >= 0.25)] = colors[2]
    base[(hue >= 195) & (hue < 245)] = colors[1]
    purple = (hue >= 245) & (hue < 330)
    base[purple & (xx < 0.56)] = colors[0]
    base[purple & (xx >= 0.56) & (xx < 0.68)] = colors[4]
    base[purple & (xx >= 0.68)] = colors[5]
    base[(hue < 35) | (hue >= 330)] = colors[0]
    base[(hue >= 35) & (hue < 170)] = colors[2]

    legend = colored & (xx > 0.76)
    base[legend & (yy < 0.13)] = colors[3]
    base[legend & (yy >= 0.13) & (yy < 0.19)] = colors[1]
    base[legend & (yy >= 0.19) & (yy < 0.25)] = colors[2]
    base[legend & (yy >= 0.25) & (yy < 0.29)] = colors[0]
    base[legend & (yy >= 0.29) & (yy < 0.35)] = colors[4]
    base[legend & (yy >= 0.35)] = colors[5]

    light = colored & (lum > 205)
    low_sat = colored & (chroma < 80)
    color_out = base.copy()
    color_out[light] = base[light] * 0.70 + 255.0 * 0.30
    color_out[low_sat & ~light] = base[low_sat & ~light] * 0.88 + 255.0 * 0.12
    out[colored] = color_out[colored]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.concatenate([out, alpha], axis=-1)


def convert_bms_requirements_array(arr: np.ndarray) -> np.ndarray:
    out_array = convert_array(arr, PALETTES["red_monochrome"]["categorical"])
    rgb = out_array[..., :3].astype(np.float32)
    alpha = out_array[..., 3:4]
    h, w = rgb.shape[:2]

    polygons = [
        ([(0.018, 0.685), (0.209, 0.384), (0.399, 0.685), (0.210, 0.997)], lighten((0xD1, 0x88, 0x7E), 0.55)),
        ([(0.500, 0.002), (0.309, 0.307), (0.500, 0.613), (0.689, 0.307)], lighten((0xB6, 0x30, 0x2D), 0.76)),
        ([(0.807, 0.381), (0.617, 0.685), (0.806, 0.994), (0.996, 0.685)], lighten((0x56, 0x6B, 0x78), 0.66)),
    ]

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Only recolor pale background pixels inside the large diamonds. Text,
    # icons, connector lines, and strong borders remain untouched.
    background = (lum > 170) & ((maxc - minc) < 90)

    for points, color in polygons:
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        xy = [(round(px * w), round(py * h)) for px, py in points]
        draw.polygon(xy, fill=255)
        mask = (np.array(mask_img) > 0) & background
        rgb[mask] = color

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def convert_image(
    path: Path,
    dst: Path,
    palette: dict[str, list[tuple[int, int, int]]],
    out_name: str | None = None,
) -> None:
    rel = path.relative_to(SRC)
    if out_name is not None:
        rel = rel.with_name(out_name)
    out_path = dst / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dst.name == "red_muted":
        target_name = rel.name
        if target_name == "robustness_signal_integrity.png":
            _render_signal_integrity(out_path)
            return
    with Image.open(path) as image:
        converted = image.convert("RGBA")
        arr = np.array(converted)
        if path.name in MIXED_BMS_REQUIREMENT_FILES and dst.name == "mixed_palette":
            out_array = convert_bms_requirements_array(arr)
        elif path.name in MIXED_SIX_COLOR_FILES and dst.name == "mixed_palette":
            out_array = convert_six_color_array(arr)
        elif path.name in HEATMAP_FILES:
            out_array = convert_heatmap_array(arr, palette["heatmap"])
        elif path.name.startswith("robustness_decision") and dst.name == "red_muted":
            out_array = convert_decision_array(arr, palette["categorical"])
        elif path.name in ROBUSTNESS_MODEL_FILES and dst.name == "red_muted":
            out_array = convert_robustness_model_array(arr, palette["categorical"])
        else:
            out_array = convert_array(arr, palette["categorical"])
        if dst.name == "red_muted" and (path.name in SOLID_FILL_FILES or (out_name in SOLID_FILL_FILES if out_name else False)):
            out_array = fill_grid_gaps_inside_colored_regions(out_array)
        if dst.name == "red_muted" and (path.name in RIGHT_HALF_SOLID_FILL_FILES or (out_name in RIGHT_HALF_SOLID_FILL_FILES if out_name else False)):
            out_array = fill_grid_gaps_in_right_half(out_array)
        if dst.name == "red_muted" and (path.name == "embedded_model_sizes.png" or out_name == "embedded_model_sizes.png"):
            out_array = adjust_embedded_model_sizes_array(out_array)
        if np.all(out_array[..., 3] == 255):
            out = Image.fromarray(out_array[..., :3], mode="RGB")
        else:
            out = Image.fromarray(out_array, mode="RGBA")
        out.save(out_path)


def main() -> None:
    for name, palette in PALETTES.items():
        dst = SRC / name
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in sorted(SRC.glob("*.png")):
            if path.name in SKIP_FILES:
                skipped = dst / path.name
                if skipped.exists():
                    skipped.unlink()
                continue
            if name == "red_muted" and (
                "_rb" in path.stem or "_original" in path.stem or "_synthesis" in path.stem
            ):
                skipped = dst / path.name
                if skipped.exists():
                    skipped.unlink()
                continue
            if name == "red_muted" and path.name in SOURCE_RENDERED_RED_MUTED_FILES:
                continue
            convert_image(path, dst, palette)
            count += 1
        if name == "red_muted":
            for source_name, alias_name in RED_MUTED_ALIASES.items():
                if alias_name in SOURCE_RENDERED_RED_MUTED_FILES:
                    continue
                source_path = SRC / source_name
                if source_path.exists():
                    convert_image(source_path, dst, palette, out_name=alias_name)
                    count += 1
        print(f"Converted {count} PNG files into {dst}")

    mixed = SRC / "mixed_palette"
    mixed.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(SRC.glob("*.png")):
        if path.name in SKIP_FILES:
            skipped = mixed / path.name
            if skipped.exists():
                skipped.unlink()
            continue
        if path.name in MIXED_MONOCHROME_FILES:
            palette = PALETTES["red_monochrome"]
        else:
            palette = PALETTES["red_muted"]
        convert_image(path, mixed, palette)
        count += 1
    print(f"Converted {count} PNG files into {mixed}")


if __name__ == "__main__":
    main()
