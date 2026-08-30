from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIGURE = (
    Path(__file__).resolve().parent
    / "Results/All Cells/Figure_03_Disturbance_Taxonomy.png"
)


def main() -> None:
    image = Image.open(FIGURE).convert("RGBA")
    draw = ImageDraw.Draw(image)
    # The source is a flattened dissertation asset. Replace only its obsolete
    # label and retain the surrounding geometry and palette byte-for-byte.
    draw.rectangle((55, 258, 299, 315), fill=(255, 255, 255, 255))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    text = "Current-gain error"
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((292 - width, 274), text, font=font, fill=(0, 0, 0, 255))
    image.save(FIGURE)
    print(FIGURE)


if __name__ == "__main__":
    main()
