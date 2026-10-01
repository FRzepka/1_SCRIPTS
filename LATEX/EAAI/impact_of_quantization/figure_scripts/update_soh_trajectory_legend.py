"""Rebuild only the legend of the archived SOH figure, preserving its plot.

The available original PDF contains one raster image, not editable plot data.
Its plot is reused unchanged as a clipped PDF object. Legend entries are
reversed together with their parameter values, never recolored independently.
Run with Python and PyMuPDF. The source archive is never overwritten.
"""
from collections import Counter
from pathlib import Path
import argparse

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = [
    (0.7, 3.0, 55), (0.9, 2.5, 65), (0.9, 2.5, 45),
    (1.0, 1.8, 55), (0.5, 2.5, 65), (0.5, 2.5, 45),
    (0.7, 1.8, 55), (0.7, 1.8, 71), (0.4, 1.8, 55),
    (0.7, 1.8, 38), (0.9, 1.0, 65), (0.9, 1.0, 45),
    (0.5, 1.0, 65), (0.5, 1.0, 45), (0.7, 0.5, 55),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path,
                        default=Path('/home/florianr/anaconda3/fonts/DejaVuSans.ttf'))
    args = parser.parse_args()
    source = ROOT / 'figure_scripts/source/lfp_soh_trajectories_original.pdf'
    output = ROOT / 'figures/lfp_soh_trajectories.pdf'
    src = fitz.open(source)
    pix = fitz.Pixmap(src, src[0].get_images()[0][0])
    assert (pix.width, pix.height, pix.n) == (2991, 1693, 3)
    colors = []
    pixels = pix.samples
    for row in range(15):
        y = int(201 + row * 88.5)
        counts = Counter(
            tuple(pixels[(yy * pix.width + xx) * 3:(yy * pix.width + xx) * 3 + 3])
            for yy in range(y - 6, y + 7) for xx in range(2320, 2450)
        )
        rgb = next(c for c, _ in counts.most_common() if max(c) - min(c) > 10)
        colors.append(tuple(v / 255 for v in rgb))

    doc = fitz.open()
    page = doc.new_page(width=2380, height=src[0].rect.height)
    # The cut is to the right of the axes and to the left of the old legend.
    plot = fitz.Rect(0, 0, 1590, src[0].rect.height)
    page.show_pdf_page(plot, src, 0, clip=plot)
    font = fitz.Font(fontfile=str(args.font))
    page.insert_font(fontname='legend', fontfile=str(args.font))

    def centered(text, x, baseline, size=44):
        left = x - font.text_length(text, fontsize=size) / 2
        page.insert_text((left, baseline), text, fontname='legend', fontsize=size)

    def subscript_header(subscript, x, baseline):
        main_width = font.text_length('C', fontsize=44)
        sub_width = font.text_length(subscript, fontsize=31)
        left = x - (main_width + sub_width) / 2
        page.insert_text((left, baseline), 'C', fontname='legend', fontsize=44)
        page.insert_text((left + main_width, baseline + 8), subscript,
                         fontname='legend', fontsize=31)

    page.draw_rect(fitz.Rect(1608, 42, 2364, 1077), color=(.83, .83, .83),
                   width=2, radius=.018)
    centers = (1830, 2073, 2290)
    separators = (1944, 2214)
    subscript_header('Charge', centers[0], 92)
    subscript_header('Discharge', centers[1], 92)
    centered('DOD', centers[2], 92)
    for x in separators:
        centered('|', x, 92)
    for row, (color, values) in enumerate(reversed(list(zip(colors, PARAMETERS)))):
        baseline = 160 + row * 63.5
        page.draw_rect(fitz.Rect(1640, baseline - 17.1, 1726, baseline - 12.9),
                       color=None, fill=color)
        for x, value in zip(centers, (f'{values[0]:.1f}', f'{values[1]:.1f}', str(values[2]))):
            centered(value, x, baseline)
        for x in separators:
            centered('|', x, baseline)
    doc.set_metadata({'title': 'LFP SOH trajectories: aligned reversed legend',
                      'subject': 'Original plot unchanged; legend ordered green to red'})
    doc.save(output, garbage=4, deflate=True)
    print(output)


if __name__ == '__main__':
    main()
