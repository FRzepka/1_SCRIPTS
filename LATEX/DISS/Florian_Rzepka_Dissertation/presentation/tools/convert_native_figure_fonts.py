"""Render Times New Roman variants from the archived editable SVG sources."""
from pathlib import Path
import re
import copy
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/tnr_sources/LATEX/DISS/Florian_Rzepka_Dissertation_Overleaf_260814/pictures/schematics"
BUILD = ROOT / "archive/tnr_build"
OUT = ROOT / "Bilder_Vortrag/Times_New_Roman"
INKSCAPE = "C:/Program Files/Inkscape/bin/inkscape.com"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

FILES = {
    "01_Abb_2-1_BMS_Anforderungen.png": "bms_requirements_icon_black.svg",
    "05_Abb_3-2_MLP_Architektur.png": "mlp_general_architecture.svg",
    "11_Abb_3-3_LSTM_Zelle.png": "lstm_cell_eaai_style.svg",
    "12_Abb_3-4_GRU_Zelle.png": "gru_cell_eaai_style.svg",
    "13_Abb_6-1_Benchmark_Methodik.png": "group:g33",
    "14_Abb_6-2_Stoerungsfamilien.png": "group:g34",
    "25_Abb_3-5_Pruning.png": "pruning_granularity_comparison.svg",
    "26_Abb_3-7_Quantisierung.png": "quantization_scope_and_grids.svg",
}


def extract_group(group_id):
    giant = ROOT / "archive/tnr_sources/LATEX/JES/paper_robustness_benchmark/figures/Schematics/Schematics.svg"
    original = ET.parse(giant).getroot()
    root = ET.Element(f"{{{SVG}}}svg", {"version": "1.1"})
    defs = next(e for e in original if e.tag == f"{{{SVG}}}defs")
    group = next(e for e in original.iter() if e.get("id") == group_id)
    root.append(copy.deepcopy(defs))
    root.append(copy.deepcopy(group))
    return ET.ElementTree(root)


def bms_headings(root):
    remove = {"path5750-0-5-3", "path5752-8-2-1", "path5754-1-4-5", "central-bms-board-vector"}
    for parent in root.iter():
        for child in list(parent):
            if child.get("id") in remove or child.tag == f"{{{SVG}}}text":
                parent.remove(child)
    giant = ROOT / "archive/tnr_sources/LATEX/JES/paper_robustness_benchmark/figures/Schematics/Schematics.svg"
    source = ET.parse(giant).getroot()
    icon = next(e for e in source.iter() if e.get("id") == "image6475-2-0-7")
    icon = copy.deepcopy(icon)
    icon.set("x", "453.04")
    icon.set("y", "225.3")
    icon.set("width", "77.34")
    icon.set("height", "51.56")
    group = next(e for e in root.iter() if e.get("id") == "g35")
    group.append(icon)
    overlay = ET.SubElement(root, f"{{{SVG}}}g", {"transform": "translate(78.564 -36.099) scale(0.479378635 0.478062264)"})
    for text, x, y in [("DEPLOYMENT", 567, 224), ("HARDWARE", 82, 641), ("PERFORMANCE", 1260, 453)]:
        node = ET.SubElement(overlay, f"{{{SVG}}}text", {
            "x": str(x), "y": str(y), "transform": f"rotate(-47 {x} {y})",
            "font-size": "35", "text-anchor": "middle", "dominant-baseline": "central",
            "fill": "black",
        })
        node.text = text
    for text, x, y, size in [
        ("Model", 851, 115, 34), ("complexity", 851, 155, 34),
        ("Adaptability", 690, 289, 34), ("Scalability", 1010, 289, 34),
        ("RAM usage", 352, 501, 34), ("Flash usage", 352, 679, 34),
        ("Inference Time", 352, 864, 34), ("Robustness", 1364, 506, 34),
        ("Convergence behavior", 1365, 679, 34), ("Accuracy", 1364, 863, 34),
        ("BATTERY", 850, 933, 35), ("MANAGEMENT", 850, 979, 35),
        ("REQUIREMENTS", 850, 1026, 35),
    ]:
        node = ET.SubElement(overlay, f"{{{SVG}}}text", {
            "x": str(x), "y": str(y), "font-size": str(size),
            "text-anchor": "middle", "dominant-baseline": "central", "fill": "black",
        })
        node.text = text


def taxonomy_labels(root):
    # Keep the ring and pictograms at their original pixel positions.
    root.set("width", "1730")
    root.set("height", "756")
    root.set("viewBox", "118.008 887.415 830.439 362.719")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == f"{{{SVG}}}text" and "".join(child.itertext()).strip() != "x":
                parent.remove(child)
    overlay = ET.SubElement(root, f"{{{SVG}}}g", {
        "transform": "translate(118.008 887.415) scale(0.48002254 0.47978704)"})
    for text,x,y,size,anchor,color in [
        ("Signal-integrity faults",10,99,46,"start","#db9897"),
        ("Initialization errors",1050,259,46,"start","#e7b7b7"),
        ("Input Disturbances",997,590,46,"start","#c55957"),
        ("Missing samples",493,26,34,"middle","black"),
        ("Irregular sampling",752,26,34,"middle","black"),
        ("Burst dropout",961,140,36,"start","black"),
        ("Current-gain",167,273,35,"middle","black"),
        ("and bias",167,313,35,"middle","black"),
        ("Current, voltage,",163,482,35,"middle","black"),
        ("temperature noise",163,522,35,"middle","black"),
        ("Initial SOC mismatch",1072,421,36,"start","black"),
        ("Voltage spikes",402,718,36,"middle","black"),
        ("ADC quantization",845,735,36,"middle","black"),
    ]:
        node=ET.SubElement(overlay,f"{{{SVG}}}text",{"x":str(x),"y":str(y),
            "font-size":str(size),"text-anchor":anchor,"dominant-baseline":"central","fill":color})
        node.text=text


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in FILES.items():
        tree = extract_group(source.split(":")[1]) if source.startswith("group:") else ET.parse(SOURCE / source)
        root = tree.getroot()
        if name.startswith("01_"):
            bms_headings(root)
        if name.startswith("14_"):
            taxonomy_labels(root)
        count = 0
        for element in root.iter():
            if element.tag == f"{{{SVG}}}style":
                element.text = re.sub(r"font-family\s*:[^;}]+", "font-family:'Times New Roman'", element.text or "")
                element.text = re.sub(r"font-weight\s*:[^;}]+", "font-weight:normal", element.text)
            if element.tag in (f"{{{SVG}}}text", f"{{{SVG}}}tspan"):
                style = element.get("style", "")
                style = re.sub(r"font-family\s*:[^;]+;?", "", style)
                style = re.sub(r"font-weight\s*:[^;]+;?", "", style)
                element.set("style", style.rstrip(";") + ";font-family:'Times New Roman' !important;font-weight:normal !important")
                element.set("font-family", "Times New Roman")
                element.set("font-weight", "normal")
                if element.text:
                    element.text = re.sub(r"^\s*(?:\([a-z]\)|[a-z]\))\s*", "", element.text)
                if element.text and not element.text.strip():
                    element.text = ""
                if element.tail and not element.tail.strip():
                    element.tail = ""
                count += 1
        target = BUILD / (Path(name).stem + ".svg")
        tree.write(target, encoding="utf-8", xml_declaration=True)
        extra = []
        if source.startswith("group:") and not name.startswith("14_"):
            with Image.open(ROOT / "Bilder_Vortrag" / name) as image:
                extra = ["--export-area-drawing", f"--export-width={image.width}", f"--export-height={image.height}"]
        subprocess.run([INKSCAPE, str(target), "--export-type=png", *extra,
                        f"--export-filename={OUT / name}", "--export-background=white",
                        "--export-background-opacity=1"], check=True, capture_output=True)
        print(f"{name}: {count} native text elements", flush=True)


if __name__ == "__main__":
    main()
