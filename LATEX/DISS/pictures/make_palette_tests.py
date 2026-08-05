from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parent

PALETTES = {
    "01_tu_red_muted": {
        "label": "TU red muted categorical",
        "comment": "Main recommendation for four estimator classes: TU red, muted rose, desaturated brown-gray, and blue-gray.",
        "colors": ["#b6302d", "#d1887e", "#8b6763", "#566b78"],
    },
    "02_tu_red_scientific": {
        "label": "TU red scientific contrast",
        "comment": "Still red-led, but with cool and ochre accents for stronger distinction in line and scatter plots.",
        "colors": ["#b6302d", "#6b7f8f", "#b88a5a", "#6f857d"],
    },
    "03_tu_red_pastel": {
        "label": "TU red pastel",
        "comment": "Softest palette. Good for filled bars and boxes, slightly weaker for thin lines.",
        "colors": ["#b6302d", "#e0a39b", "#9f7d7a", "#7f9caf"],
    },
    "04_tu_red_earth": {
        "label": "TU red earth tones",
        "comment": "Warm, calm print palette with red, clay, olive-gray, and slate.",
        "colors": ["#b6302d", "#c27a54", "#8b806b", "#596b76"],
    },
    "05_tu_red_monochrome": {
        "label": "TU red monochrome sequence",
        "comment": "Useful when the four categories are ordered. Not ideal when categories must be equally distinct.",
        "colors": ["#7c1f1d", "#b6302d", "#d6766d", "#edbdb8"],
    },
    "06_tu_red_print_safe": {
        "label": "TU red print-safe",
        "comment": "More conservative and robust in grayscale or print. Less pastel, but still not poppy.",
        "colors": ["#9e2a2b", "#6c757d", "#b06d47", "#4f6d7a"],
    },
    "07_tu_red_muted_reordered": {
        "label": "TU red muted reordered",
        "comment": "Same family as the current red-muted palette, but M2 is moved to blue-gray. This is the quickest fix when M1 and M2 must be separated clearly.",
        "colors": ["#b6302d", "#566b78", "#d1887e", "#8b6763"],
    },
    "08_tu_red_cool_muted": {
        "label": "TU red with cool muted accents",
        "comment": "Red-led but with blue, olive, and brown-gray accents. Good for nominal model classes because the four hues are not just red shades.",
        "colors": ["#b6302d", "#2f6f88", "#7f8f6b", "#8b6763"],
    },
    "09_tu_red_neutral_highcontrast": {
        "label": "TU red and neutral contrast",
        "comment": "Very restrained palette for print-heavy figures. Red marks the main model, while the other classes use graphite, blue-gray, and light gray.",
        "colors": ["#b6302d", "#434343", "#6f8290", "#b2b2b2"],
    },
    "10_user_bright_direct": {
        "label": "Requested bright palette",
        "comment": "Uses the requested saturated red, purple, blue, and green exactly. Strong separation, but noticeably more colorful than the dissertation style.",
        "colors": ["#c40d1e", "#9013fe", "#1f90cc", "#49cb40"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "11_user_bright_tempered": {
        "label": "Requested palette, dissertation-tempered",
        "comment": "A calmer version inspired by the requested colors. It keeps the red-purple-blue-green structure but lowers saturation for a less poppy scientific look.",
        "colors": ["#c40d1e", "#6f4a8e", "#2f7898", "#6f9a63"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "12_tu_red_colorblind_safe": {
        "label": "TU red plus colorblind-safe accents",
        "comment": "Scientific/high-contrast option: TU red combined with blue, green, and ochre accents. Best candidate when lines overlap heavily.",
        "colors": ["#b6302d", "#0072b2", "#009e73", "#e69f00"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "13_tu_red_muted_purple_green": {
        "label": "TU red muted with purple and soft green",
        "comment": "Close to the current red-muted style, but the rose and brown-gray slots are replaced by muted purple and a softer light green for clearer M1/M2/M3 separation.",
        "colors": ["#b6302d", "#7f5f9f", "#8fbf8a", "#566b78"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "14_tu_red_muted_purple_m2": {
        "label": "TU red muted with purple M2",
        "comment": "Minimal variant of the current red-muted palette: M1, M3, and M4 stay unchanged, while the rose M2 is replaced by muted purple.",
        "colors": ["#b6302d", "#7f5f9f", "#8b6763", "#566b78"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "15_default_cycler_red_green_orange_cyan": {
        "label": "Default cycler red, green, orange, cyan",
        "comment": "Direct test of the proposed Matplotlib cycler colors. Strongly separated and readable, but more saturated than the muted dissertation palettes.",
        "colors": ["#cc0000", "#22a15c", "#ff8000", "#00a6b3"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
    "16_eaai_extracted_light": {
        "label": "EAAI extracted light fill palette",
        "comment": "Palette extracted from the EAAI figures: green, red, and blue as the main colours plus matching purple for a fourth class. Use the main colours for lines and outlines; use the fill colours for bars, boxes, and histograms.",
        "colors": ["#2ca02c", "#d62728", "#1f77b4", "#9467bd"],
        "fills": ["#a6d7a6", "#eea4a5", "#a1c6e0", "#d2bfe3"],
        "neutrals": ["#000000", "#434343", "#b2b2b2"],
    },
}


def shade(hex_color: str, factor: float = 0.35) -> str:
    """Blend a hex color with white by factor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = round(r + (255 - r) * factor)
    g = round(g + (255 - g) * factor)
    b = round(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }',
        '.small { font-size: 12px; }',
        '.label { font-size: 13px; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.panel { font-size: 14px; font-weight: 700; }',
        '.axis { stroke: #404850; stroke-width: 1.2; }',
        '.grid { stroke: #e4e8ec; stroke-width: 1; }',
        '</style>',
    ]


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(value)}</text>'


def line(points: list[tuple[float, float]], color: str, width: float = 2.4) -> str:
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def panel_frame(x: int, y: int, w: int, h: int) -> list[str]:
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#d7dde3" stroke-width="1"/>']
    for i in range(1, 4):
        gy = y + h - i * h / 4
        parts.append(f'<line class="grid" x1="{x}" y1="{gy:.1f}" x2="{x+w}" y2="{gy:.1f}"/>')
    parts.append(f'<line class="axis" x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}"/>')
    parts.append(f'<line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y+h}"/>')
    return parts


def draw_lines(parts: list[str], x: int, y: int, w: int, h: int, colors: list[str]) -> None:
    parts += panel_frame(x, y, w, h)
    parts.append(text(x, y - 10, "time-series / estimator traces", "panel"))
    for j, color in enumerate(colors):
        pts = []
        for i in range(48):
            xx = x + i * w / 47
            val = 0.52 + 0.18 * math.sin((i / 7.0) + j * 0.55) + 0.035 * j + 0.02 * math.cos(i / 3.7)
            yy = y + h - max(0.05, min(0.95, val)) * h
            pts.append((xx, yy))
        parts.append(line(pts, color, 2.5))
    for j, color in enumerate(colors):
        lx = x + 16 + j * 85
        ly = y + h + 28
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(text(lx + 28, ly + 4, f"M{j+1}", "small"))


def draw_bars(parts: list[str], x: int, y: int, w: int, h: int, colors: list[str]) -> None:
    parts += panel_frame(x, y, w, h)
    parts.append(text(x, y - 10, "bar chart / MAE-style comparison", "panel"))
    values = [0.72, 0.47, 0.58, 0.36]
    bw = 42
    gap = (w - 4 * bw) / 5
    for i, (val, color) in enumerate(zip(values, colors)):
        bx = x + gap + i * (bw + gap)
        bh = val * h
        by = y + h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{shade(color, 0.20)}" stroke="{color}" stroke-width="2"/>')
        parts.append(text(bx + bw / 2, y + h + 20, f"M{i+1}", "small", "middle"))


def draw_boxplots(parts: list[str], x: int, y: int, w: int, h: int, colors: list[str]) -> None:
    parts += panel_frame(x, y, w, h)
    parts.append(text(x, y - 10, "boxplot / distribution view", "panel"))
    stats = [
        (0.20, 0.34, 0.46, 0.61, 0.78, 0.50),
        (0.14, 0.25, 0.39, 0.55, 0.69, 0.43),
        (0.18, 0.30, 0.42, 0.52, 0.72, 0.45),
        (0.10, 0.21, 0.32, 0.45, 0.60, 0.36),
    ]
    box_w = 38
    gap = w / 5
    for i, (low, q1, med, q3, high, mean) in enumerate(stats):
        color = colors[i]
        cx = x + gap * (i + 1)
        def yy(v: float) -> float:
            return y + h - v * h
        parts.append(f'<line x1="{cx:.1f}" y1="{yy(low):.1f}" x2="{cx:.1f}" y2="{yy(high):.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-12:.1f}" y1="{yy(low):.1f}" x2="{cx+12:.1f}" y2="{yy(low):.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-12:.1f}" y1="{yy(high):.1f}" x2="{cx+12:.1f}" y2="{yy(high):.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{cx-box_w/2:.1f}" y="{yy(q3):.1f}" width="{box_w}" height="{(q3-q1)*h:.1f}" fill="{shade(color, 0.28)}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-box_w/2:.1f}" y1="{yy(med):.1f}" x2="{cx+box_w/2:.1f}" y2="{yy(med):.1f}" stroke="#1f2933" stroke-width="2"/>')
        diamond = [(cx, yy(mean)-6), (cx+6, yy(mean)), (cx, yy(mean)+6), (cx-6, yy(mean))]
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in diamond)
        parts.append(f'<polygon points="{d}" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        parts.append(text(cx, y + h + 20, f"M{i+1}", "small", "middle"))
    parts.append(text(x + w - 4, y + h + 42, "line = median, diamond = mean/MAE marker", "small", "end"))


def draw_scatter(parts: list[str], x: int, y: int, w: int, h: int, colors: list[str]) -> None:
    parts += panel_frame(x, y, w, h)
    parts.append(text(x, y - 10, "scatter / measured vs. predicted", "panel"))
    for j, color in enumerate(colors):
        for i in range(22):
            px = x + 18 + ((i * 13 + j * 31) % (w - 36))
            base = 0.15 + ((i * 17 + j * 9) % 70) / 100
            py = y + h - base * h + 9 * math.sin(i * 0.8 + j)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.7" fill="{shade(color, 0.12)}" stroke="{color}" stroke-width="1.2" opacity="0.88"/>')


def write_palette_preview() -> None:
    width, height = 1420, max(820, 150 + len(PALETTES) * 105)
    parts = svg_header(width, height)
    parts.append(text(40, 42, "Color-palette candidates for dissertation figures", "title"))
    parts.append(text(40, 68, "Base TU-oriented red: #b6302d. Each row contains four colors for estimator-class comparisons.", "label"))
    y = 115
    for name, spec in PALETTES.items():
        colors = spec["colors"]
        parts.append(text(40, y + 24, f"{name}: {spec['label']}", "panel"))
        parts.append(text(40, y + 54, spec["comment"], "small"))
        for i, color in enumerate(colors):
            x = 730 + i * 155
            parts.append(f'<rect x="{x}" y="{y}" width="92" height="46" rx="4" fill="{color}" stroke="#222" stroke-width="0.5"/>')
            parts.append(text(x + 46, y + 67, color, "small", "middle"))
        fills = spec.get("fills", [])
        for i, color in enumerate(fills):
            x = 730 + i * 155
            edge = colors[i] if i < len(colors) else "#222222"
            parts.append(f'<rect x="{x}" y="{y+77}" width="62" height="22" rx="3" fill="{color}" stroke="{edge}" stroke-width="1.5"/>')
            parts.append(text(x + 76, y + 94, color, "small"))
        if fills:
            y += 105
            continue
        for i, color in enumerate(spec.get("neutrals", [])):
            x = 730 + i * 105
            parts.append(f'<rect x="{x}" y="{y+77}" width="62" height="22" rx="3" fill="{color}" stroke="#222" stroke-width="0.5"/>')
            parts.append(text(x + 76, y + 94, color, "small"))
        y += 105
    parts.append("</svg>")
    (OUT / "palette_preview_all.svg").write_text("\n".join(parts), encoding="utf-8")


def write_palette_test(name: str, spec: dict[str, object]) -> None:
    if name == "16_eaai_extracted_light":
        parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1200" viewBox="0 0 1600 1200">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            "<style>",
            "text { font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }",
            ".title { font-size: 28px; font-weight: 700; }",
            ".label { font-size: 18px; }",
            ".small { font-size: 15px; }",
            ".panel { font-size: 20px; font-weight: 700; }",
            "</style>",
            '<text class="title" x="50" y="55">16_eaai_extracted_light: real-figure palette check</text>',
            '<text class="label" x="50" y="90">Main colours are used for lines and outlines; light fills are used where the real plots contain bars, boxes, or filled distributions.</text>',
            '<text class="panel" x="50" y="145">Main / outline colours</text>',
            '<rect x="50" y="165" width="70" height="38" rx="3" fill="#2ca02c" stroke="#222" stroke-width="0.5"/>',
            '<text class="small" x="132" y="181">DM / Base</text>',
            '<text class="small" x="132" y="201">#2ca02c</text>',
            '<rect x="410" y="165" width="70" height="38" rx="3" fill="#d62728" stroke="#222" stroke-width="0.5"/>',
            '<text class="small" x="492" y="181">HDM / Pruned</text>',
            '<text class="small" x="492" y="201">#d62728</text>',
            '<rect x="770" y="165" width="70" height="38" rx="3" fill="#1f77b4" stroke="#222" stroke-width="0.5"/>',
            '<text class="small" x="852" y="181">HECM / Quantized</text>',
            '<text class="small" x="852" y="201">#1f77b4</text>',
            '<rect x="1130" y="165" width="70" height="38" rx="3" fill="#9467bd" stroke="#222" stroke-width="0.5"/>',
            '<text class="small" x="1212" y="181">DD</text>',
            '<text class="small" x="1212" y="201">#9467bd</text>',
            '<text class="panel" x="50" y="245">Light fill colours</text>',
            '<rect x="50" y="265" width="70" height="38" rx="3" fill="#a6d7a6" stroke="#2ca02c" stroke-width="3"/>',
            '<text class="small" x="132" y="290">#a6d7a6</text>',
            '<rect x="410" y="265" width="70" height="38" rx="3" fill="#eea4a5" stroke="#d62728" stroke-width="3"/>',
            '<text class="small" x="492" y="290">#eea4a5</text>',
            '<rect x="770" y="265" width="70" height="38" rx="3" fill="#a1c6e0" stroke="#1f77b4" stroke-width="3"/>',
            '<text class="small" x="852" y="290">#a1c6e0</text>',
            '<rect x="1130" y="265" width="70" height="38" rx="3" fill="#d2bfe3" stroke="#9467bd" stroke-width="3"/>',
            '<text class="small" x="1212" y="290">#d2bfe3</text>',
            '<text class="panel" x="50" y="355">Embedded benchmark: real EAAI error-distribution plot</text>',
            '<image x="50" y="375" width="700" height="350" preserveAspectRatio="xMidYMid meet" href="../Florian_Rzepka_Dissertation/pictures/eaai_palette/embedded_soc_error.png"/>',
            '<text class="panel" x="850" y="355">Robustness baseline: real JES result plot</text>',
            '<image x="850" y="375" width="700" height="350" preserveAspectRatio="xMidYMid meet" href="../Florian_Rzepka_Dissertation/pictures/eaai_palette/robustness_baseline.png"/>',
            '<text class="panel" x="50" y="790">Current-bias sensitivity: real JES line plot</text>',
            '<image x="50" y="810" width="700" height="340" preserveAspectRatio="xMidYMid meet" href="../Florian_Rzepka_Dissertation/pictures/eaai_palette/robustness_current_bias.png"/>',
            '<text class="panel" x="850" y="790">Signal integrity: real JES bar/hatch plot</text>',
            '<image x="850" y="810" width="700" height="340" preserveAspectRatio="xMidYMid meet" href="../Florian_Rzepka_Dissertation/pictures/eaai_palette/robustness_signal_integrity.png"/>',
            "</svg>",
        ]
        (OUT / f"testplot_{name}.svg").write_text("\n".join(parts), encoding="utf-8")
        return

    colors = list(spec["colors"])[:4]
    width, height = 1200, 820
    parts = svg_header(width, height)
    parts.append(text(42, 42, f"{name}: {spec['label']}", "title"))
    parts.append(text(42, 68, str(spec["comment"]), "label"))
    for i, color in enumerate(colors):
        x = 42 + i * 170
        parts.append(f'<rect x="{x}" y="92" width="44" height="26" rx="3" fill="{color}" stroke="#222" stroke-width="0.5"/>')
        parts.append(text(x + 56, 111, f"M{i+1} {color}", "small"))
    for i, color in enumerate(spec.get("neutrals", [])):
        x = 42 + i * 128
        parts.append(f'<rect x="{x}" y="128" width="34" height="18" rx="2" fill="{color}" stroke="#222" stroke-width="0.5"/>')
        parts.append(text(x + 44, 143, f"N{i+1} {color}", "small"))
    draw_lines(parts, 70, 170, 470, 210, colors)
    draw_bars(parts, 675, 170, 390, 210, colors)
    draw_boxplots(parts, 70, 510, 470, 210, colors)
    draw_scatter(parts, 675, 510, 390, 210, colors)
    parts.append("</svg>")
    (OUT / f"testplot_{name}.svg").write_text("\n".join(parts), encoding="utf-8")


def write_catalog() -> None:
    rows = []
    rows.append("# Dissertation color palette tests")
    rows.append("")
    rows.append("Base color: `#b6302d`.")
    rows.append("")
    rows.append("Recommended starting point: `07_tu_red_muted_reordered` if the current red-muted look should be preserved but M1/M2 need stronger separation. Use `12_tu_red_colorblind_safe` when many lines overlap or when maximum distinguishability is more important than a red-dominant appearance. `05_tu_red_monochrome` should only be used when the four series are ordered, not when they are nominal model classes.")
    rows.append("")
    rows.append("| Palette | Colors | Use case |")
    rows.append("|---|---|---|")
    for name, spec in PALETTES.items():
        neutrals = spec.get("neutrals", [])
        neutral_text = f"<br>Neutral: {' '.join('`' + c + '`' for c in neutrals)}" if neutrals else ""
        fills = spec.get("fills", [])
        fill_text = f"<br>Fill: {' '.join('`' + c + '`' for c in fills)}" if fills else ""
        rows.append(f"| `{name}` | {' '.join('`' + c + '`' for c in spec['colors'])}{fill_text}{neutral_text} | {spec['comment']} |")
    rows.append("")
    rows.append("Generated files:")
    rows.append("")
    rows.append("- `palette_preview_all.svg` / `palette_preview_all.png`: compact overview of all palettes")
    for name in PALETTES:
        if name == "16_eaai_extracted_light":
            rows.append(f"- `testplot_{name}.svg` / `testplot_{name}.png`: real-figure contact sheet using EAAI/JES dissertation plots")
        else:
            rows.append(f"- `testplot_{name}.svg`: line, bar, boxplot, and scatter examples")
    (OUT / "palette_catalog.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "palette_settings.json").write_text(json.dumps(PALETTES, indent=2), encoding="utf-8")


def export_pngs() -> None:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        print("Inkscape not found; SVG files were written but PNG export was skipped.")
        return
    svg_files = [OUT / "palette_preview_all.svg"] + [OUT / f"testplot_{name}.svg" for name in PALETTES]
    for svg in svg_files:
        png = svg.with_suffix(".png")
        subprocess.run(
            [inkscape, str(svg), "--export-type=png", f"--export-filename={png}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    write_palette_preview()
    for name, spec in PALETTES.items():
        write_palette_test(name, spec)
    write_catalog()
    export_pngs()
    print(f"Wrote palette tests to {OUT}")


if __name__ == "__main__":
    main()
