"""Read-only figure/source inventory and contact sheets for visual checking."""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "Bilder_Vortrag"
QA = ROOT / "archive/tnr_qa"


def main():
    QA.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGES.glob("*.png"))
    data = []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    for start in range(0, len(files), 6):
        sheet = Image.new("RGB", (1500, 1260), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(files[start:start + 6]):
            with Image.open(path) as source:
                data.append({"file": path.name, "size": source.size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                white.alpha_composite(rgba)
                thumb = ImageOps.contain(white.convert("RGB"), (735, 365))
                x, y = (i % 2) * 750, (i // 2) * 420
                draw.text((x + 8, y + 6), path.name, fill="black", font=font)
                sheet.paste(thumb, (x + (750 - thumb.width) // 2, y + 40))
        sheet.save(QA / f"originals_{start // 6 + 1}.jpg")
    (QA / "original_inventory.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
