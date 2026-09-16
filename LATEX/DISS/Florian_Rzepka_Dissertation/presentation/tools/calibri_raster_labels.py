"""Calibri labels for approved figures whose original text is rasterized."""
import base64
import copy
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFont

from build_calibri import SOURCE, OUT, BUILD, QA, INKSCAPE, SVG, TNR_BUILD

XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


class Figure:
    def __init__(self, prefix, virtual_width=None):
        self.path = next(SOURCE.glob(prefix + "*.png"))
        self.original = Image.open(self.path).convert("RGBA")
        self.bg = self.original.copy()
        self.w, self.h = self.bg.size
        self.scale = self.w / (virtual_width or self.w)
        self.vw, self.vh = self.w / self.scale, self.h / self.scale
        self.root = ET.Element(f"{{{SVG}}}svg", {
            "width": str(self.w), "height": str(self.h),
            "viewBox": f"0 0 {self.vw} {self.vh}"})
        self.labels = []
        self.masks = []

    def crop(self, box):
        return self.original.crop(tuple(round(v * self.scale) for v in box))

    def cover(self, box, color=None):
        rect = tuple(round(v * self.scale) for v in box)
        if color is None:
            pixels = np.asarray(self.original.crop(rect))
            samples = np.concatenate((pixels[:2].reshape(-1, 4), pixels[-2:].reshape(-1, 4),
                                      pixels[:, :2].reshape(-1, 4), pixels[:, -2:].reshape(-1, 4)))
            color = tuple(np.median(samples, axis=0).astype(int))
        self.bg.paste(color, rect)
        self.masks.append(list(box))

    def text(self, text, x, y, size, anchor="middle", rotation=0, color="#222222", italic=False):
        node = ET.SubElement(self.root, f"{{{SVG}}}text", {
            "x": str(x), "y": str(y), "font-family": "Calibri", "font-size": str(size),
            "font-weight": "normal", "font-style": "italic" if italic else "normal",
            "fill": color, "text-anchor": anchor})
        if rotation:
            node.set("transform", f"rotate({rotation} {x} {y})")
        parts = re.split(r"(\{[^}]+\})", text)
        node.text = parts[0]
        for index in range(1, len(parts), 2):
            sub = ET.SubElement(node, f"{{{SVG}}}tspan", {
                "font-size": str(size * 0.7), "baseline-shift": "sub"})
            sub.text = parts[index][1:-1]
            sub.tail = parts[index + 1] if index + 1 < len(parts) else ""
        self.labels.append({"text": text, "x": x, "y": y, "size": size, "rotation": rotation})

    def replace(self, box, text, size=None, anchor="middle", rotation=0,
                color="#222222", background=None, italic=False):
        sample = np.asarray(self.crop(box))
        if np.median(sample[:, :, 3]) < 64:
            ink = sample[:, :, 3] > 128
        elif color == "white":
            ink = sample[:, :, :3].min(axis=2) > 230
        else:
            ink = sample[:, :, :3].max(axis=2) < 130
        yy, xx = np.where(ink)
        assert len(xx), (self.path.name, text, box)
        x0, x1 = box[0] + xx.min() / self.scale, box[0] + (xx.max() + 1) / self.scale
        y0, y1 = box[1] + yy.min() / self.scale, box[1] + (yy.max() + 1) / self.scale
        plain = re.sub(r"\{[^}]+\}", "", text)
        if size is None:
            target_h = (x1 - x0) if rotation else (y1 - y0)
            target_w = (y1 - y0) if rotation else (x1 - x0)
            scores = []
            for candidate in range(8, 130):
                font = ImageFont.truetype("C:/Windows/Fonts/timesi.ttf" if italic else
                                         "C:/Windows/Fonts/times.ttf", candidate)
                a, b, c, d = font.getbbox(plain, anchor="ls")
                error = abs((d - b) - target_h) + 0.18 * abs((c - a) - target_w)
                scores.append((error, candidate))
            size = min(scores)[1]
        font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", round(size))
        a, b, c, d = font.getbbox(plain, anchor="ls")
        self.cover(box, background)
        if rotation:
            self.text(text, (x0 + x1) / 2 - (b + d) / 2, (y0 + y1) / 2,
                      size, rotation=rotation, color=color, italic=italic)
        else:
            x = x0 if anchor == "start" else x1 if anchor == "end" else (x0 + x1) / 2
            self.text(text, x, (y0 + y1) / 2 - (b + d) / 2,
                      size, anchor=anchor, color=color, italic=italic)

    def ocr(self):
        path = QA.parent / "tnr_qa/ocr" / (self.path.stem + ".json")
        return json.loads(path.read_text(encoding="utf-8-sig"))["lines"]

    def numeric_ocr(self, predicate=lambda x, y: True, size=None):
        for line in self.ocr():
            for word in line["words"]:
                text = word["text"]
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
                    continue
                x, y, w, h = (word[k] for k in ("x", "y", "w", "h"))
                if predicate(x, y):
                    self.replace((x - 3, y - 3, x + w + 3, y + h + 3), text, size=size)

    def approved_heading(self, text, box):
        tree = ET.parse(TNR_BUILD / (self.path.stem + ".svg"))
        node = next(n for n in tree.getroot().iter(f"{{{SVG}}}text")
                    if "".join(n.itertext()) == text)
        self.cover(box)
        changed = copy.deepcopy(node)
        for child in changed.iter():
            for key, value in list(child.attrib.items()):
                child.set(key, value.replace("Times New Roman", "Calibri"))
        self.root.append(changed)
        self.labels.append({"text": text, "source": "approved SVG", "size": node.get("font-size")})

    def save(self):
        stream = io.BytesIO()
        self.bg.save(stream, format="PNG")
        node = ET.Element(f"{{{SVG}}}image", {"x": "0", "y": "0",
            "width": str(self.vw), "height": str(self.vh),
            f"{{{XLINK}}}href": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()})
        self.root.insert(0, node)
        target = BUILD / (self.path.stem + ".svg")
        ET.ElementTree(self.root).write(target, encoding="utf-8", xml_declaration=True)
        subprocess.run([INKSCAPE, str(target), "--export-type=png", "--export-area-page",
                        f"--export-width={self.w}", f"--export-height={self.h}",
                        f"--export-filename={OUT / self.path.name}"], check=True, capture_output=True)
        (QA / (self.path.stem + "_labels.json")).write_text(json.dumps({
            "source": str(self.path), "font": "Calibri", "labels": self.labels,
            "masks": self.masks, "size_policy": "Approved native sizes or measured raster type size",
            "data_operation": "none"}, indent=2), encoding="utf-8")
        print(self.path.name, len(self.labels), "labels", flush=True)


def nmc(prefix):
    f = Figure(prefix)
    f.numeric_ocr(lambda x, y: x < 190 or y > 1200, size=58.3333)
    f.replace((15, 440, 88, 760), "SOH in %", rotation=-90, size=58.3333)
    if prefix == "02a_":
        f.replace((1260, 1290, 1760, 1360), "Equivalent full cycle", size=58.3333)
        f.replace((217, 1225, 247, 1280), "0", size=58.3333)
    else:
        f.replace((1350, 1290, 1670, 1360), "Time in days", size=58.3333)
        f.replace((217, 1225, 247, 1280), "0", size=58.3333)
    settings = [(50, 100, 1, 35), (20, 30, 1, 35), (50, 30, 1, 35),
                (80, 30, 1, 35), (50, 100, 2, 35), (50, 100, 1, 45), (50, 100, 2, 45)]
    for i, (soc, dod, rate, temp) in enumerate(settings):
        top = 64 + i * 65
        f.cover((2026, top, 2756, top + 57), (255, 255, 255, 255))
        f.text(f"{soc}% SOC{{mean}}, {dod}% DOD, {rate}C, {temp}\u00b0C",
               2034, top + 41, 50, anchor="start")
    f.save()


def correlation():
    f = Figure("06_")
    f.numeric_ocr(lambda x, y: y < 1850, size=56)
    for box, text in [
        ((110, 705, 577, 770), "Temperature [\u00b0C]"),
        ((285, 265, 578, 335), "Voltage [V]"),
        ((313, 1137, 578, 1220), "Q{pos}(t) [Ah]"),
        ((313, 1576, 578, 1660), "Q{neg}(t) [Ah]"),
        ((775, 1860, 905, 1925), "SOH"),
        ((1125, 1860, 1420, 1930), "Voltage [V]"),
        ((1475, 1860, 1950, 1930), "Temperature [\u00b0C]"),
        ((2010, 1858, 2285, 1945), "Q{pos}(t) [Ah]")]:
        f.replace(box, text, size=56)
    f.save()


def lag():
    f = Figure("07_", virtual_width=2048)
    for box, text, size in [
        ((710, 0, 947, 85), "Input", 69 / f.scale),
        ((1375, 0, 1630, 85), "Output", 69 / f.scale),
        ((370, 122, 573, 191), "Voltage", 64),
        ((615, 122, 942, 191), "Temperature", 64),
        ((1005, 120, 1285, 198), "Current{sum}", 64),
        ((1440, 124, 1577, 188), "SOH", 64),
        ((430, 201, 528, 269), "[V]", 64),
        ((742, 201, 856, 269), "[\u00b0C]", 64),
        ((1088, 201, 1189, 269), "[A]", 64),
        ((1450, 201, 1560, 269), "[%]", 64),
        ((0, 328, 199, 379), "Data set", 58),
        ((0, 701, 180, 758), "Sample", 58)]:
        f.replace(box, text, size=size)
    for col, symbol, left, right in [(0, "U", 417, 519), (1, "T", 738, 836),
                                    (2, "Q", 1093, 1195), (3, "SOH", 1418, 1597)]:
        for i, (top, bottom) in enumerate([(301, 369), (380, 447), (456, 521), (531, 595)]):
            f.cover((left, top, right, bottom))
            f.text(f"{symbol}{{t-{3 - i}}}", (left + right) / 2, top + 49, 64)
    for left, right, text in [(370, 451, "t{0}"), (491, 594, "U{t-0}"),
                              (647, 750, "U{t-1}"), (812, 918, "U{t-2}"),
                              (976, 1080, "T{t-0}"), (1115, 1219, "T{t-1}"),
                              (1246, 1350, "T{t-2}"), (1383, 1493, "Q{t-0}"),
                              (1536, 1648, "Q{t-1}"), (1693, 1805, "Q{t-2}"),
                              (1842, 2030, "SOH{t-0}")]:
        f.cover((left, 826, right, 898))
        f.text(text, (left + right) / 2, 880, 64)
    f.save()


def heatmaps():
    f = Figure("09_")
    f.numeric_ocr(lambda x, y: y > 280 or (x % 1479 > 1200 and y < 300), size=58.3333)
    for i, (cell, soc, dod, rate) in enumerate([(1, 50, 100, 1), (7, 80, 30, 1), (9, 50, 100, 2)]):
        dx = i * 1479
        f.cover((210 + dx, 205, 1174 + dx, 274), (255, 255, 255, 255))
        f.text(f"Cell {cell} - {soc}% SOC{{mean}}, {dod}% DOD, {rate}C, 35\u00b0C",
               693 + dx, 254, 50)
        f.replace((428 + dx, 1313, 949 + dx, 1385), "Number of sequences", size=58.3333)
        f.replace((50 + dx, 385, 126 + dx, 1080), "Time resolution in minutes", rotation=-90, size=58.3333)
        f.replace((1398 + dx, 600, 1478 + dx, 870), "MAE", rotation=-90, size=58.3333)
        f.replace((147 + dx, 373, 191 + dx, 423), "1", size=58.3333)
        f.replace((312 + dx, 1245, 346 + dx, 1300), "8", size=58.3333)
    # The second colourbar's 1.4 tick was not returned by the source OCR.
    f.replace((2801, 439, 2877, 495), "1.4", size=58.3333)
    f.save()


def soh_results():
    f = Figure("10_")
    f.numeric_ocr(lambda x, y: (y > 130 and x % 1479 > 90) and not (1530 < x < 1590 and y < 1000), size=58.3333)
    for i, (cell, soc, dod, rate) in enumerate([(1, 50, 100, 1), (7, 80, 30, 1), (9, 50, 100, 2)]):
        dx = i * 1479
        f.cover((375 + dx, 60, 1345 + dx, 126), (255, 255, 255, 255))
        f.text(f"Cell {cell} - {soc}% SOC{{mean}}, {dod}% DOD, {rate}C, 35\u00b0C",
               865 + dx, 106, 50)
        f.replace((701 + dx, 1374, 1023 + dx, 1448), "Time in days", size=58.3333)
        f.replace((20 + dx, 555, 112 + dx, 930), "SOH in %", rotation=-90, size=58.3333)
        f.replace((236 + dx, 1305, 278 + dx, 1364), "0", size=58.3333)
        f.replace((1210 + dx, 171, 1340 + dx, 236), "SOH", size=58.3333)
        f.cover((1210 + dx, 253, 1435 + dx, 331), (255, 255, 255, 255))
        f.text("SOH{pred}", 1220 + dx, 307, 58.3333, anchor="start")
    f.save()


def optimization():
    f = Figure("24_")
    for line in f.ocr():
        if line["text"] in ("o onnoo", "a ooaao"):
            continue
        words = [w for w in line["words"] if w["text"] != "\u2022"]
        x0, y0 = min(w["x"] for w in words), min(w["y"] for w in words)
        x1 = max(w["x"] + w["w"] for w in words)
        y1 = max(w["y"] + w["h"] for w in words)
        text = line["text"].removeprefix("\u2022 ").replace("(IHz)", "(1Hz)")
        text = text.replace("UARTfeature", "UART feature")
        if text in ("Pruning", "Quantization"):
            f.approved_heading(text, (644, 80, 797, 112) if text == "Pruning" else (632, 251, 814, 287))
            continue
        if text == "MLP rows":
            text, x1 = "MLP rows)", 781
        f.replace((x0 - 3, y0 - 3, x1 + 3, y1 + 3), text,
                  anchor="start" if x0 > 1190 and y0 > 200 else "middle")
    f.save()


def embedded_lstm():
    f = Figure("23_")
    for box, text, size, italic in [
        ((110, 33, 183, 70), "Input", 30, False),
        ((18, 88, 290, 121), "Battery measurements", 30, False),
        ((18, 123, 280, 157), "over time (current,", 30, False),
        ((18, 157, 280, 193), "voltage, temperature)", 30, False),
        ((716, 10, 806, 43), "LSTM", 30, False),
        ((1280, 10, 1355, 44), "MLP", 30, False),
        ((1519, 118, 1611, 153), "Output", 30, False),
        ((24, 275, 59, 312), "x{0}", 30, True),
        ((97, 275, 138, 312), "x{1}", 30, True),
        ((210, 275, 250, 313), "x{t}", 30, True),
        ((339, 111, 400, 153), "c{t-1}", 30, True),
        ((339, 299, 400, 336), "h{t-1}", 30, True),
        ((1086, 110, 1125, 149), "c{t}", 30, True),
        ((1088, 306, 1126, 345), "h{t}", 30, True),
        ((548, 451, 587, 483), "x{t}", 30, True),
        ((1200, 88, 1271, 121), "Input", 30, False),
        ((1278, 88, 1373, 121), "Hidden", 30, False),
        ((1374, 88, 1463, 121), "Output", 30, False),
        ((1197, 125, 1274, 157), "Layer", 30, False),
        ((1287, 125, 1368, 157), "Layer", 30, False),
        ((1380, 125, 1459, 157), "Layer", 30, False),
        ((1519, 182, 1610, 215), "SOC{t},", 30, True),
        ((1519, 218, 1605, 251), "SOH{t}", 30, True),
        ((548, 267, 581, 300), "\u03c3", 32, True),
        ((641, 267, 674, 300), "\u03c3", 32, True),
        ((843, 267, 876, 300), "\u03c3", 32, True),
        ((722, 264, 791, 301), "tanh", 32, True),
        ((907, 182, 979, 219), "tanh", 32, True),
    ]:
        f.replace(box, text, size=size, italic=italic)
    # Operation symbols are text too, but the surrounding circles stay intact.
    for box, text in [((550, 121, 579, 145), "\u00d7"), ((742, 117, 773, 150), "+"),
                      ((744, 193, 770, 219), "\u00d7"), ((930, 266, 955, 293), "\u00d7")]:
        f.replace(box, text, size=30)
    f.save()


def training():
    f = Figure("08_")
    for line in f.ocr():
        words = line["words"]
        x0, y0 = min(w["x"] for w in words), min(w["y"] for w in words)
        x1 = max(w["x"] + w["w"] for w in words)
        y1 = max(w["y"] + w["h"] for w in words)
        if 800 < x0 < 1310 and 2140 < y0 < 2590 or line["text"] == "ooo":
            continue
        text = line["text"].replace("\u2014SOH, Is", "\u2192 SOH, I{s}").replace("-+ Lag Sequence", "\u2192 Lag Sequence")
        f.replace((x0 - 4, y0 - 4, x1 + 4, y1 + 4), text,
                  color="white" if text in ("PREPROCESSING", "TRAINING &", "VALIDATION", "EVALUATION") else "#222222")
    # Labels inside the miniature example plot retain their small original size.
    for box, text, rotation in [
        ((1157, 2168, 1290, 2189), "SOH measured", 0),
        ((1157, 2191, 1290, 2216), "SOH predicted", 0),
        ((819, 2161, 858, 2188), "100", 0),
        ((829, 2230, 858, 2257), "90", 0),
        ((829, 2297, 858, 2323), "80", 0),
        ((829, 2362, 858, 2389), "70", 0),
        ((829, 2431, 858, 2459), "60", 0),
        ((829, 2498, 858, 2525), "50", 0),
        ((802, 2300, 825, 2385), "SOH [%]", -90),
        ((937, 2538, 970, 2563), "50", 0),
        ((1023, 2538, 1062, 2563), "100", 0),
        ((1110, 2538, 1150, 2563), "150", 0),
        ((1197, 2538, 1240, 2563), "200", 0),
        ((1011, 2563, 1154, 2590), "Testtime [Days]", 0)]:
        f.replace(box, text, size=22, rotation=rotation)
    f.save()


def build_all(prefix=None):
    tasks = [("02a_", lambda: nmc("02a_")), ("02b_", lambda: nmc("02b_")),
             ("06_", correlation), ("07_", lag), ("08_", training), ("09_", heatmaps),
             ("10_", soh_results), ("23_", embedded_lstm), ("24_", optimization)]
    for name, task in tasks:
        if prefix is None or name.startswith(prefix):
            task()


if __name__ == "__main__":
    import sys
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
