from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIGURE = (
    Path(__file__).resolve().parent
    / "Results/All Cells/Figure_03_Disturbance_Taxonomy.png"
)

FONT_CANDIDATES = (
    Path("C:/texlive/2024/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def load_font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(path, 24)
    raise FileNotFoundError("DejaVu Sans was not found on this system")


def main() -> None:
    image = Image.open(FIGURE).convert("RGBA")
    draw = ImageDraw.Draw(image)
    # The source is a flattened dissertation asset. Replace only its obsolete
    # label and retain the surrounding geometry and palette byte-for-byte.
    draw.rectangle((15, 258, 299, 315), fill=(255, 255, 255, 255))
    font = load_font()
    text = "Current-gain and bias"
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((292 - width, 274), text, font=font, fill=(0, 0, 0, 255))
    image.save(FIGURE)
    print(FIGURE)


if __name__ == "__main__":
    main()
