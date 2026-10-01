"""Add top/right axis space to the saved vector PDF without changing curves.

The original CSV referenced by generate_vector_figures.py is absent locally.
This version-specific fallback extends the original vector canvas and clip
rectangle at unchanged scale. It does not infer or replace any data points.
The normal generator separately uses ylim=(0.62, 1.02) and 2% right padding.
"""
from pathlib import Path
import re

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'figure_scripts/source/selected_baseline_soh_trajectory_original.pdf'
    target = ROOT / 'figures/selected_baseline_soh_trajectory.pdf'
    doc = fitz.open(source)
    page = doc[0]
    stream = page.read_contents().decode('latin1')
    old_clip = '91.613125 77.37875 1303.2 586.8 re W n'
    assert old_clip in stream
    # Original y range is 0.62..1.00. Keep the existing physical axis scale.
    top_padding = 586.8 * .02 / .38
    right_padding = 1303.2 * .02
    new_top = 664.17875 + top_padding
    new_right = 1394.813125 + right_padding
    curve_start = stream.index('Q q ' + old_clip + ' /A2 gs 2 J 1 j 2.2 w')
    head, curves_and_legend = stream[:curve_start], stream[curve_start:]
    head = head.replace('1396.973125', f'{1396.973125 + right_padding:.9f}')
    head = head.replace('675.643437', f'{675.643437 + top_padding:.9f}')
    head = head.replace('1394.813125', f'{new_right:.9f}')
    background = (f'{new_right:.9f} 664.17875 l\n'
                  '91.613125 664.17875 l\nh')
    assert background in head
    head = head.replace(background,
                        f'{new_right:.9f} {new_top:.9f} l\n'
                        f'91.613125 {new_top:.9f} l\nh')
    # Extend vertical grid lines and the left spine, not the grid at SOH=1.
    head, count = re.subn(
        r'(\d+\.\d+) 77\.37875 m\n\1 664\.17875 l',
        lambda m: f'{m[1]} 77.37875 m\n{m[1]} {new_top:.9f} l', head)
    assert count == 7
    head = head.replace('689.5803125 8.12875 cm',
                        f'{689.5803125 + right_padding / 2:.9f} 8.12875 cm')
    head = head.replace('24.019375 304.0678125 cm',
                        f'24.019375 {304.0678125 + top_padding / 2:.9f} cm')
    new_clip = (f'91.613125 77.37875 {1303.2 + right_padding:.9f} '
                f'{586.8 + top_padding:.9f} re W n')
    output = (head + curves_and_legend).replace(old_clip, new_clip)
    # The five scientific curves and legend are byte-identical except clipping.
    assert output[len(head.replace(old_clip, new_clip)):].replace(new_clip, old_clip) == curves_and_legend
    page.set_mediabox(fitz.Rect(0, 0, page.rect.width + right_padding,
                                page.rect.height + top_padding))
    page.set_cropbox(page.mediabox)
    assert len(page.get_contents()) == 1
    doc.update_stream(page.get_contents()[0], output.encode('latin1'))
    doc.save(target, garbage=4, deflate=True)
    print(f'Saved {target}; top: +0.02 SOH, right: +2% time range')


if __name__ == '__main__':
    main()
