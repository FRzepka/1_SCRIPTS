"""Calibri copies of the approved presentation figures, without size changes."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Bilder_Vortrag/Times_New_Roman"
OUT = ROOT / "Bilder_Vortrag/Calibri"
BUILD = ROOT / "archive/calibri_build"
QA = ROOT / "archive/calibri_qa"
TNR_BUILD = ROOT / "archive/tnr_build"
INKSCAPE = "C:/Program Files/Inkscape/bin/inkscape.com"
SVG = "http://www.w3.org/2000/svg"
RASTER = ("02a_", "02b_", "06_", "07_", "08_", "09_", "10_", "23_", "24_")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(svg, name):
    with Image.open(SOURCE / name) as image:
        w, h = image.size
    subprocess.run([INKSCAPE, str(svg), "--export-type=png", "--export-area-page",
                    f"--export-width={w}", f"--export-height={h}",
                    f"--export-filename={OUT / name}", "--export-background=white",
                    "--export-background-opacity=1"], check=True, capture_output=True)


def clone_svg(path):
    name = path.stem + ".png"
    original = path.read_text(encoding="utf-8")
    changed = original.replace("Times New Roman", "Calibri")
    assert re.findall(r"font-size[^;\"<>}]*", original) == re.findall(
        r"font-size[^;\"<>}]*", changed)
    tree = ET.ElementTree(ET.fromstring(changed))
    root = tree.getroot()
    if name.startswith("13_"):
        # Freeze the approved drawing bounds before changing font metrics.
        result = subprocess.run([INKSCAPE, str(path), "--query-all"],
                                check=True, capture_output=True, text=True)
        row = next(line.split(",") for line in result.stdout.splitlines()
                   if line.startswith("g33,"))
        root.set("viewBox", " ".join(row[1:5]))
        with Image.open(SOURCE / name) as image:
            root.set("width", str(image.width))
            root.set("height", str(image.height))
    target = BUILD / path.name
    tree.write(target, encoding="utf-8", xml_declaration=True)
    render(target, name)
    print(name, flush=True)


def inventory():
    return {p.name: {"sha256": digest(p), "size": list(Image.open(p).size)}
            for p in sorted(SOURCE.glob("*.png"))}


def verify():
    before = json.loads((QA / "approved_source_inventory.json").read_text())
    assert inventory() == before, "Approved Times New Roman images changed"
    files = sorted(OUT.glob("*.png"))
    assert {p.name for p in files} == set(before), "Incomplete Calibri image set"
    for path in files:
        with Image.open(path) as image:
            assert list(image.size) == before[path.name]["size"], path.name
            image.verify()
    for path in BUILD.glob("*.svg"):
        source = path.read_text(encoding="utf-8")
        assert "Times New Roman" not in source, path.name
        assert not re.search(r"font-weight\s*[:=]\s*[\"']?(?:bold|[6-9]00)", source), path.name
    for path in TNR_BUILD.glob("*.svg"):
        target = BUILD / path.name
        if path.name.startswith(RASTER) or not target.exists():
            continue
        a = ET.parse(path).getroot()
        b = ET.parse(target).getroot()
        old = list(a.iter())
        new = list(b.iter())
        assert len(old) == len(new), path.name
        for index, (u, v) in enumerate(zip(old, new)):
            expected = {k: value.replace("Times New Roman", "Calibri") for k, value in u.attrib.items()}
            if index == 0 and path.name.startswith("13_"):
                expected.update({k: v.attrib[k] for k in ("width", "height", "viewBox")})
            assert expected == v.attrib, (path.name, index)
            assert (u.text or "").replace("Times New Roman", "Calibri") == (v.text or ""), path.name
    font = ImageFont.truetype("C:/Windows/Fonts/calibri.ttf", 19)
    for start in range(0, len(files), 6):
        sheet = Image.new("RGB", (1600, 1350), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(files[start:start + 6]):
            with Image.open(path) as image:
                white = Image.new("RGBA", image.size, "white")
                white.alpha_composite(image.convert("RGBA"))
                thumb = ImageOps.contain(white.convert("RGB"), (780, 395))
            x, y = (i % 2) * 800, (i // 2) * 450
            draw.text((x + 10, y + 8), path.name, fill="black", font=font)
            sheet.paste(thumb, (x + (800 - thumb.width) // 2, y + 42))
        sheet.save(QA / f"contact_{start // 6 + 1}.jpg", quality=93)
    (QA / "output_inventory.json").write_text(json.dumps(
        {p.name: {"sha256": digest(p), "size": before[p.name]["size"]} for p in files},
        indent=2), encoding="utf-8")
    print(f"Verified {len(files)} PNGs. Approved sources unchanged.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prefix")
    args = parser.parse_args()
    for folder in (OUT, BUILD, QA):
        folder.mkdir(parents=True, exist_ok=True)
    saved = QA / "approved_source_inventory.json"
    if not saved.exists():
        saved.write_text(json.dumps(inventory(), indent=2), encoding="utf-8")
    if args.verify_only:
        verify()
        return
    for path in sorted(TNR_BUILD.glob("*.svg")):
        if path.name.startswith(RASTER) or not (SOURCE / (path.stem + ".png")).exists():
            continue
        if args.prefix and not path.name.startswith(args.prefix):
            continue
        clone_svg(path)
    for path in SOURCE.glob("04_*.png"):
        shutil.copy2(path, OUT / path.name)
    if not args.svg_only:
        from calibri_raster_labels import build_all
        build_all(args.prefix)
        verify()


if __name__ == "__main__":
    main()
