"""Typeset reviewed chart labels in SVG without recomputing any plotted samples.

The archived SVG keeps the complete original PNG as its immutable background.
Only explicitly specified label areas are covered. Plot curves are not retraced.
"""
from pathlib import Path
import base64
import json
import re
import subprocess
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont
import numpy as np
from scipy import ndimage
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "Bilder_Vortrag"
OUT = IMAGES / "Times_New_Roman"
BUILD = ROOT / "archive/tnr_build"
QA = ROOT / "archive/tnr_qa"
FONT = "C:/Windows/Fonts/times.ttf"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


class Figure:
    def __init__(self, name, reference_width=None):
        self.path = next(IMAGES.glob(name + "*.png"))
        self.name = self.path.name
        with Image.open(self.path) as image:
            self.w, self.h = image.size
        self.vw = reference_width or self.w
        self.vh = self.h * self.vw / self.w
        self.root = ET.Element(f"{{{SVG}}}svg", {"width": str(self.w), "height": str(self.h),
                                                   "viewBox": f"0 0 {self.vw} {self.vh}"})
        ET.SubElement(self.root, f"{{{SVG}}}rect", {"width": "100%", "height": "100%", "fill": "white"})
        ET.SubElement(self.root, f"{{{SVG}}}image", {
            "width": str(self.vw), "height": str(self.vh),
            "{http://www.w3.org/1999/xlink}href": "data:image/png;base64," + base64.b64encode(self.path.read_bytes()).decode("ascii"),
        })
        self.labels = []
        self.masks = []

    def yticks(self, box, values, ys, x, size=28):
        self.cover(box)
        for value, y in zip(values, ys):
            self.text(str(value), x, y, size, anchor="end")

    def xticks(self, box, values, xs, y, size=28):
        self.cover(box)
        for value, x in zip(values, xs):
            self.text(str(value), x, y, size)

    def background(self, x, y):
        with Image.open(self.path) as image:
            pixel = image.convert("RGBA").getpixel((round(x*self.w/self.vw), round(y*self.h/self.vh)))
        a = pixel[3]/255
        return "#" + "".join(f"{round(c*a+255*(1-a)):02x}" for c in pixel[:3])

    def ocr_labels(self, keep, corrections=None):
        data = json.loads((QA / "ocr" / (self.path.stem + ".json")).read_text(encoding="utf-8-sig"))
        scale = self.vw/self.w
        for line in data["lines"]:
            words = sorted(line["words"], key=lambda w: w["x"])
            x = min(w["x"] for w in words)*scale
            y = min(w["y"] for w in words)*scale
            right = max(w["x"]+w["w"] for w in words)*scale
            bottom = max(w["y"]+w["h"] for w in words)*scale
            text = " ".join(w["text"] for w in words)
            if not keep(x,y,text):
                continue
            text = (corrections or {}).get(text, text)
            pad = 2.0
            self.replace((x-pad,y-pad,right-x+2*pad,bottom-y+2*pad), text,
                         size=(bottom-y)*1.52)

    def cover(self, box, color="white"):
        x, y, w, h = box
        ET.SubElement(self.root, f"{{{SVG}}}rect", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "fill": color})
        self.masks.append(list(box))

    def legend_line(self, x, y, label, color="black", dash=None, size=34):
        attrs={"x1":str(x),"x2":str(x+52),"y1":str(y),"y2":str(y),
               "stroke":color,"stroke-width":"4"}
        if dash:
            attrs["stroke-dasharray"]=dash
        ET.SubElement(self.root,f"{{{SVG}}}line",attrs)
        self.text(label,x+66,y,size,anchor="start")

    def clear_plot_legend(self, box, grid_x=(), grid_y=()):
        self.cover(box)
        x,y,w,h=box
        for gx in grid_x:
            if x <= gx <= x+w:
                ET.SubElement(self.root,f"{{{SVG}}}line",{"x1":str(gx),"x2":str(gx),
                    "y1":str(y),"y2":str(y+h),"stroke":"#e8e8e8","stroke-width":"1.3"})
        for gy in grid_y:
            if y <= gy <= y+h:
                ET.SubElement(self.root,f"{{{SVG}}}line",{"x1":str(x),"x2":str(x+w),
                    "y1":str(gy),"y2":str(gy),"stroke":"#e8e8e8","stroke-width":"1.3"})

    def retain_colored_curves(self, boxes, translucent_notes=()):
        """Retain existing curve pixels, excluding disconnected legend handles."""
        with Image.open(self.path) as original:
            rgb=np.asarray(original.convert("RGB"))
        scale=self.w/self.vw
        region=np.zeros(rgb.shape[:2],dtype=bool)
        for x,y,w,h in boxes:
            region[round(y*scale):round((y+h)*scale),round(x*scale):round((x+w)*scale)]=True
        chromatic=np.ptp(rgb.astype(int),axis=2)>7
        components,_=ndimage.label(chromatic)
        outside=np.bincount(components[~region].ravel(),minlength=int(components.max())+1)
        connected=outside>500
        connected[0]=False
        keep=region & connected[components]
        restored=rgb.copy()
        for box,opacity in translucent_notes:
            x,y,w,h=box
            note=np.zeros(region.shape,dtype=bool)
            note[round(y*scale):round((y+h)*scale),round(x*scale):round((x+w)*scale)]=True
            visible=note & chromatic
            keep |= visible
            # Undo the source script's known white annotation-box opacity.
            # Only existing coloured pixels are retained, never interpolated.
            clear=visible & (rgb.min(axis=2)>170)
            restored[clear]=np.clip((rgb[clear].astype(float)-255*opacity)/(1-opacity),0,255).astype(np.uint8)
        rgba=np.dstack((restored,np.where(keep,255,0).astype(np.uint8)))
        buffer=BytesIO()
        Image.fromarray(rgba).save(buffer,format="PNG")
        ET.SubElement(self.root,f"{{{SVG}}}image",{"width":str(self.vw),"height":str(self.vh),
            "{http://www.w3.org/1999/xlink}href":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")})

    def reframe_panels(self, panels):
        """Move complete panels into a layout with an internal legend footer."""
        children=list(self.root)
        for child in children:
            self.root.remove(child)
        ET.SubElement(self.root,f"{{{SVG}}}rect",{"width":str(self.vw),"height":str(self.vh),"fill":"white"})
        defs=ET.SubElement(self.root,f"{{{SVG}}}defs")
        content=ET.SubElement(defs,f"{{{SVG}}}g",{"id":"assembled-panels"})
        content.extend(children)
        for index,(source,target) in enumerate(panels):
            x,y,w,h=target
            sx,sy,sw,sh=source
            scale=w/sw
            assert abs(scale-h/sh)<1e-6
            clip_id=f"panel-crop-{index}"
            clip=ET.SubElement(defs,f"{{{SVG}}}clipPath",{"id":clip_id,"clipPathUnits":"userSpaceOnUse"})
            ET.SubElement(clip,f"{{{SVG}}}rect",{"x":str(x),"y":str(y),"width":str(w),"height":str(h)})
            group=ET.SubElement(self.root,f"{{{SVG}}}g",{"clip-path":f"url(#{clip_id})"})
            ET.SubElement(group,f"{{{SVG}}}use",{"{http://www.w3.org/1999/xlink}href":"#assembled-panels",
                "transform":f"translate({x} {y}) scale({scale}) translate({-sx} {-sy})"})
        self.panel_layout=panels

    def restore_red_curve_pixels(self, boxes):
        """Keep original connected curve strokes beneath relocated annotations."""
        with Image.open(self.path) as image:
            rgb=np.asarray(image.convert("RGB"))
        region=np.zeros(rgb.shape[:2],dtype=bool)
        scale=self.w/self.vw
        for x,y,w,h in boxes:
            region[round(y*scale):round((y+h)*scale),round(x*scale):round((x+w)*scale)]=True
        red=(rgb[:,:,0]>150)&(rgb[:,:,0]>rgb[:,:,1]*1.4)&(rgb[:,:,1]<180)
        components,_=ndimage.label(red)
        counts=np.bincount(components[~region].ravel(),minlength=int(components.max())+1)
        large=counts>400
        large[0]=False
        core=(ndimage.distance_transform_edt(red)>2.5)&large[components]
        protect=ndimage.binary_dilation(core,iterations=3)
        # Neutral grid/background pixels are independent of the red annotations.
        neutral=(np.max(rgb,axis=2).astype(int)-np.min(rgb,axis=2).astype(int))==0
        protect |= neutral
        protect &= region
        rgba=np.dstack((rgb,np.where(protect,255,0).astype(np.uint8)))
        buffer=BytesIO()
        Image.fromarray(rgba).save(buffer,format="PNG")
        ET.SubElement(self.root,f"{{{SVG}}}image",{"width":str(self.vw),"height":str(self.vh),
            "{http://www.w3.org/1999/xlink}href":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")})

    def text(self, text, x, y, size, anchor="middle", rotation=0, color="black", weight="normal"):
        text = re.sub(r"^\s*(?:\([a-z]\)|[a-z]\))\s*", "", text)
        if not text:
            return
        attrs = {"x": str(x), "y": str(y), "font-family": "Times New Roman",
                 "font-size": str(size), "text-anchor": anchor, "dominant-baseline": "central",
                 "fill": color, "font-weight": "normal"}
        if rotation:
            attrs["transform"] = f"rotate({rotation} {x} {y})"
        node = ET.SubElement(self.root, f"{{{SVG}}}text", attrs)
        node.text = text
        self.labels.append({"text": text, "x": x, "y": y, "size": size, "rotation": rotation})

    def replace(self, box, text, size=None, bg="white", rotation=0, color="black", weight="normal"):
        x, y, w, h = box
        self.cover(box, bg)
        if size is None:
            size = (w if rotation else h) * 1.1
        # Keep label width within its allocated area, without stretching glyphs.
        max_width = h if rotation else w
        while size > 8 and ImageFont.truetype(FONT, round(size)).getlength(text) > max_width:
            size -= .5
        self.text(text, x + w / 2, y + h / 2, size, rotation=rotation, color=color, weight=weight)

    def save(self):
        BUILD.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        path = BUILD / (self.path.stem + ".svg")
        for node in list(self.root.iter(f"{{{SVG}}}text")):
            value=node.text or ""
            if "\u03c3I" in value:
                before,after=value.split("\u03c3I",1)
                node.text=before+"\u03c3"
                sub=ET.SubElement(node,f"{{{SVG}}}tspan",{"baseline-shift":"sub","font-size":"70%"})
                sub.text="I"
                sub.tail=after
            elif "\u0394\u0177k" in value:
                node.text="p95 |\u0394\u0177"
                sub=ET.SubElement(node,f"{{{SVG}}}tspan",{"baseline-shift":"sub","font-size":"70%"})
                sub.text="k"
                sub.tail=" - \u0394\u0177"
                sub=ET.SubElement(node,f"{{{SVG}}}tspan",{"baseline-shift":"sub","font-size":"70%"})
                sub.text="k-1"
                sub.tail="| [SOC]"
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=True)
        subprocess.run(["C:/Program Files/Inkscape/bin/inkscape.com", str(path),
                        "--export-type=png", f"--export-filename={OUT / self.name}",
                        "--export-background=white", "--export-background-opacity=1"],
                       check=True, capture_output=True)
        (QA / (self.path.stem + "_labels.json")).write_text(json.dumps({
            "file": self.name, "font": "Times New Roman", "labels": self.labels,
            "viewbox": [self.vw, self.vh],
            "covered_label_areas": self.masks,
            "panel_layout": getattr(self,"panel_layout",None),
            "data_operation": "none; original curves retained in SVG image layer",
        }, indent=2), encoding="utf-8")
        print(f"{self.name}: {len(self.labels)} labels", flush=True)


def lfp_dataset():
    f = Figure("03_")
    f.cover((0, 35, 285, 1440))
    for value, y in zip([1, .95, .9, .85, .8, .75, .7, .65, .6], np.linspace(103, 1430, 9)):
        f.text(f"{value:.2f}", 268, y, 75, anchor="end")
    f.text("SOH [-]", 59, 870, 78, rotation=-90)
    f.cover((295, 1503, 1910, 179))
    for value, x in zip(range(0, 201, 25), np.linspace(369, 2080, 9)):
        f.text(str(value), x, 1537, 74)
    f.text("Test time [Days]", 1246, 1632, 85)
    f.cover((2240, 67, 706, 88))
    f.text("C", 2272, 105, 69)
    f.text("Charge", 2368, 126, 47)
    f.text("|", 2481, 113, 67)
    f.text("C", 2520, 105, 69)
    f.text("Discharge", 2660, 126, 47)
    f.text("| DOD", 2855, 111, 65)
    f.cover((2480, 170, 470, 1308))
    labels = [(0.7,3.,55),(.9,2.5,65),(.9,2.5,45),(1.,1.8,55),(.5,2.5,65),(.5,2.5,45),
              (.7,1.8,55),(.7,1.8,71),(.4,1.8,55),(.7,1.8,38),(.9,1.,65),(.9,1.,45),
              (.5,1.,65),(.5,1.,45),(.7,.5,55)]
    for i, (charge, discharge, dod) in enumerate(labels):
        f.text(f"{charge:.1f} | {discharge:.1f} | {dod}", 2680, 204 + i * 88.2, 70)
    f.save()


def current_gain():
    f = Figure("16_", 2048)
    f.replace((500, 4, 1160, 37), "Current-gain sensitivity", size=30)
    f.yticks((0, 46, 114, 503), [f"{v:.3f}" for v in np.arange(0, .013, .002)],
             np.linspace(529, 64, 7), 111, 28)
    f.text("Adverse-pair \u0394MAE [SOC]", 25, 297, 31, rotation=-90)
    f.xticks((130, 562, 1897, 65), ["0.0", "0.5", "1.5", "3.0"], [212,501,1079,1946], 575)
    f.text("Current-gain error magnitude [%]", 1080, 607, 31)
    f.clear_plot_legend((136,51,514,37),grid_x=[212,501],grid_y=[64])
    f.replace((350, 657, 407, 41), "(b) Applied gain error", 30)
    f.yticks((0, 700, 114, 451), [f"{v:.2f}" for v in np.arange(0, -1.8, -.25)],
             np.linspace(720, 1107, 8), 111, 27)
    f.text("Measured current [A]", 25, 934, 31, rotation=-90)
    f.xticks((130, 1161, 850, 67), range(0,31,5), np.linspace(164, 940, 7), 1177)
    f.text("Time [min]", 550, 1205, 31)
    f.clear_plot_legend((753,706,222,64),grid_x=[810,940],grid_y=[720])
    f.replace((1280,657,744,42), "SOC response (+3% gain)", 29)
    f.yticks((1093, 700, 80, 451), [f"{v:.1f}" for v in np.arange(.2,1.,.1)],
             np.linspace(1127,720,8), 1166, 27)
    f.text("SOC [-]", 1111, 930, 31, rotation=-90)
    f.xticks((1190,1161,850,67), [f"{v:.1f}" for v in np.arange(0,3.1,.5)],
             np.linspace(1219,1995,7), 1177)
    f.text("Time in excerpt [h]", 1605, 1205, 31)
    boxes=[(1194,706,150,30),(1194,737,110,30),(1194,766,110,30),
           (1364,706,150,30),(1364,737,130,30),(1189,1087,261,56)]
    for box in boxes:
        f.clear_plot_legend(box,grid_x=np.linspace(1219,1995,7),grid_y=[720,778,1069,1127])
    f.retain_colored_curves(boxes,translucent_notes=[((1189,1087,261,56),.82)])
    f.reframe_panels([((0,0,f.vw,f.vh),(102.4,0,f.vw*.90,f.vh*.90))])
    for x,label,color in [(350,"DM","#2ca02c"),(570,"HDM","#9467bd"),
                           (820,"HECM","#1f77b4"),(1100,"DD","#d62728")]:
        f.legend_line(x,1150,label,color)
    f.legend_line(1320,1150,"Reference SOC",size=34)
    f.legend_line(100,1196,"Current: baseline",color="#444444",size=32)
    f.legend_line(585,1196,"Current: +3% gain",color="#d62728",dash="12 7",size=32)
    f.legend_line(1110,1196,"SOC: baseline",color="#777777",dash="12 7",size=32)
    f.legend_line(1550,1196,"SOC: gain error",color="#444444",size=32)
    f.save()


def baseline():
    f = Figure("15_", 2048)
    f.cover((190,8,1680,54))
    f.replace((410,82,330,43), "Baseline MAE", 35, weight="bold")
    f.replace((1420,82,354,43), "Baseline RMSE", 35, weight="bold")
    f.yticks((0,141,124,572), [f"{v:.3f}" for v in np.arange(0,.176,.025)], np.linspace(700,197,8), 116, 29)
    f.text("MAE [SOC]", 28, 411, 34, rotation=-90)
    f.yticks((1029,131,114,581), [f"{v:.3f}" for v in np.arange(0,.201,.025)], np.linspace(700,153,9), 1137, 29)
    f.text("RMSE [SOC]", 1052, 411, 34, rotation=-90)
    for box,xs in [((160,714,826,36),[260,469,678,887]), ((1168,714,843,36),[1279,1488,1697,1906])]:
        f.xticks(box,["DM","HDM","HECM","DD"],xs,729,31)
    f.save()


def memory():
    f = Figure("21_",2048)
    f.replace((335,9,500,46), "(a) Flash footprint", 37)
    f.replace((1307,9,560,46), "(b) Peak runtime RAM", 37)
    f.yticks((0,80,128,674), range(0,501,100), np.linspace(742,122,6), 122, 35)
    f.text("Flash [KiB]", 36, 404, 39, rotation=-90)
    f.yticks((1044,100,84,653), range(5), np.linspace(742,168,5), 1120, 35)
    f.text("Peak RAM [KiB]", 1069, 404, 39, rotation=-90)
    for box,xs in [((169,759,849,76),[251,474,696,918]), ((1162,759,856,76),[1250,1472,1695,1916])]:
        f.xticks(box,["DM","HDM","HECM","DD"],xs,777,36)
        f.text("continuous",xs[-1],814,34)
    for x,y,text in [(251,685,"27.0"),(474,685,"27.1"),(696,136,"470.1"),(918,575,"115.9"),
                     (1250,379,"2.4"),(1472,379,"2.4"),(1695,365,"2.5"),(1916,136,"4.1")]:
        f.replace((x-50,y-18,100,36),text,32)
    f.save()


def inference_modes():
    f=Figure("22_",2048)
    for box,text in [((238,6,293,38),"(a) Maximum error"),((968,6,196,38),"(b) Runtime"),
                     ((1568,6,365,38),"(c) Memory footprint")]:
        f.replace(box,text,31)
    f.yticks((44,45,37,493),range(0,41,5),np.linspace(524,56,9),78,27)
    f.replace((6,61,32,451),"Maximum SOC error [percentage points]",29,rotation=-90)
    f.cover((674,42,91,443))
    for power,y in zip(range(4),[453,321,188,56]):
        f.text("10",738,y,29)
        f.text(str(power),758,y-10,21)
    f.text("Median inference time [ms]",700,288,29,rotation=-90)
    f.yticks((1364,76,87,462),range(0,121,20),np.linspace(524,109,7),1446,27)
    f.text("Memory [KiB]",1382,287,29,rotation=-90)
    for box,xs in [((107,538,544,56),[181,379,576]),((791,538,545,56),[865,1063,1260]),
                    ((1471,538,547,56),[1555,1747,1938])]:
        f.xticks(box,["Rolling","Continuous","Periodic"],xs,553,28)
        for text,x in zip(["window","state","reset"],xs):
            f.text(text,x,580,28)
    for x,y,text in [(1522,107,"116.1"),(1589,277,"67.3"),(1714,107,"115.9"),(1780,496,"4.1"),
                     (1905,107,"116.0"),(1971,496,"4.1")]:
        f.replace((x-31,y-11,62,23),text,24)
    for box,text in [((486,654,197,35),"Rolling window"),((762,654,220,35),"Continuous state"),
                    ((1070,654,176,35),"Periodic reset"),((1330,654,78,35),"Flash"),
                    ((1499,654,130,35),"Peak RAM")]:
        f.replace(box,text,28)
    f.save()


def heatmap():
    f=Figure("19_",2048)
    f.replace((416,6,1080,41),"Cross-scenario robustness",34)
    f.yticks((5,82,93,352),["DM","HDM","HECM","DD"],[104,207,310,414],96,26)
    source=ROOT / "archive/plot_sources/LATEX/JES/paper_robustness_benchmark/JES_2.0/results/jes2_revised_delta_mae_matrix.csv"
    with source.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    for row_index,row in enumerate(rows):
        for column_index,raw in enumerate(list(row.values())[1:]):
            value=float(raw)
            value=0.0 if abs(value)<.0005 else value
            x=157.7+column_index*93.36
            y=103+row_index*103.5
            color=f.background(x-25,y-30)
            f.replace((x-42,y-13,84,27),f"{value:+.3f}",25,bg=color,
                      color="white" if value>.02 else "#202020")
    labels=[("Current noise","(0.02 A)"),("Current noise","(0.10 A)"),("Voltage noise","(0.01 V)"),
            ("Temperature noise","(1.0 \u00b0C)"),("Gain error","(\u00b10.5%)"),("Gain error","(\u00b11.5%)"),
            ("Gain error","(\u00b13.0%)"),("Current offset","(+/-50 mA)"),("Voltage offset","(0.02 V)"),
            ("Temperature offset","(3 \u00b0C)"),("ADC","quantization"),("Periodic missing","(1/50)"),
            ("Random missing","(2%)"),("Timing jitter","(\u00b10.1 s)"),("Timing jitter","(\u00b10.5 s)"),
            ("Timing jitter","(\u00b10.9 s)"),("Burst dropout","(1 h)"),("Voltage spikes","(\u00b10.20 V)")]
    f.cover((0,478,1831,170))
    for i,(first,second) in enumerate(labels):
        x=158+i*93.36
        group=ET.SubElement(f.root,f"{{{SVG}}}g",{"transform":f"translate({x} 492) rotate(-37)"})
        for line,y in [(first,0),(second,24)]:
            node=ET.SubElement(group,f"{{{SVG}}}text",{"x":"0","y":str(y),"font-family":"Times New Roman",
                "font-size":"24","text-anchor":"end","fill":"black"})
            node.text=line
            f.labels.append({"text":line,"font":"Times New Roman","rotation":-37})
    f.cover((1924,85,117,347))
    for exponent,y,negative in [(-1,111,False),(-2,161,False),(-3,211,False),(-3,306,True),(-2,356,True),(-1,405,True)]:
        f.text("-10" if negative else "10",1951,y,28)
        f.text(str(exponent),1979,y-11,20)
    f.text("0",1939,259,28)
    f.text("Cell-macro \u0394MAE [SOC]",2017,259,31,rotation=-90)
    f.save()


def model_sizes():
    f=Figure("29_")
    # OCR is used only for the reviewed, isolated horizontal labels of this figure.
    f.ocr_labels(lambda x,y,t: x>280 and not (2110<x<2320), {
        "Left: SOC(Darker)":"Left: SOC (Darker)",
    })
    f.yticks((15,90,225,1350),[0,20000,40000,60000,80000],[1406,1118,829,540,251],232,44)
    f.text("Count",74,761,53,rotation=-90)
    f.yticks((2135,83,165,1357),range(0,351,50),[1406,1221,1037,852,667,482,297,112],2285,44)
    f.text("Size [KB]",2168,760,53,rotation=-90)
    f.yticks((15,1560,225,1355),range(0,351,50),[2884,2699,2514,2329,2144,1960,1775,1590],232,44)
    f.text("Size [KB]",74,2237,53,rotation=-90)
    f.yticks((2129,1600,169,1315),range(0,9,2),[2884,2599,2314,2029,1743],2285,44)
    f.text("Size [KB]",2168,2237,53,rotation=-90)
    f.save()


def embedded_errors():
    for prefix, state, maes in [("27_", "SOC", ["2.68","2.33","2.78"]),
                                ("28_", "SOH", ["0.85","1.46","1.41"])]:
        f = Figure(prefix)
        f.replace((105,0,766,66), f"Error Distribution (Boxplot) - {state}", 45)
        f.replace((1025,0,750,66), f"Error Histogram (Counts) - {state}", 45)
        values, ys = (range(0,9,2), np.linspace(613,118,5)) if state == "SOC" else (range(6), np.linspace(613,69,6))
        f.yticks((0,55,91,583), values, ys, 86, 37)
        f.text("Absolute Error [%]", 35, 354, 43, rotation=-90)
        xs = [229,485,741] if state == "SOC" else [228,480,733]
        f.xticks((111,649,756,42), ["Base","Pruned","Quantized"], xs, 668, 36)
        f.cover((117,86,231,208))
        for i, (name, mae) in enumerate(zip(["Base","Pruned","Quantized"], maes)):
            color=["#2ca02c","#ed1c24","#1f77b4"][i]
            fill=["#acd9a7","#eba2a5","#a0c4df"][i]
            ET.SubElement(f.root,f"{{{SVG}}}rect",{"x":"124","y":str(111+i*70),"width":"56",
                "height":"19","fill":fill,"stroke":color,"stroke-width":"3"})
            f.text(name, 201, 103+i*70, 31, anchor="start")
            f.text(f"MAE: {mae}", 201, 133+i*70, 30, anchor="start")
        values = [0,3,6,9,12,15] if state == "SOC" else [0,30,60,90,120,150]
        f.yticks((945,84,61 if state == "SOC" else 71,577), values, np.linspace(639,159,6),
                 997 if state == "SOC" else 1008, 37)
        f.replace((891,207,68,288), "Count [k]", 41, rotation=-90)
        xs = [1044,1219,1394,1568,1743] if state == "SOC" else [1054,1226,1399,1571,1743]
        f.xticks((1013,649,762,82), ["-10","-5","0","5","10"], xs, 668, 36)
        f.text("Error (pred - GT) [%]", 1396, 709, 43)
        for i, name in enumerate(["Base","Pruned","Quantized"]):
            f.replace((1605,85+i*40,153,32), name, 33)
        f.save()


def host_latency():
    f = Figure("30_")
    f.replace((177,0,738,69), "SOC Host Latency Distribution", 52)
    f.replace((1018,0,738,69), "SOH Host Latency Distribution", 52)
    f.yticks((0,94,166,552), [0,1000,2000,3000,4000,5000], np.linspace(628,138,6), 164, 41)
    f.text("Count", 42, 352, 47, rotation=-90)
    f.replace((948,272,62,158), "Count", 47, rotation=-90)
    for box, xs, labelx in [((151,640,792,103),np.linspace(177,915,10),546),
                             ((994,640,793,103),np.linspace(1018,1755,10),1386)]:
        f.xticks(box, range(0,46,5), xs, 658, 41)
        f.text("Host Latency [ms]", labelx, 706, 46)
    for left in [718,1124]:
        for i, name in enumerate(["Base","Pruned","Quantized"]):
            f.replace((left,89+i*46,176,38), name, 38)
    f.save()


def initialization():
    f=Figure("17_",1984)
    f.replace((562,7,854,38),"Initial SOC mismatch (10%)",46)
    f.replace((743,104,583,36),"SOC trajectories",40)
    f.yticks((5,211,94,408),["0.0","0.2","0.4","0.6","0.8"],np.linspace(602,231,5),100,38)
    f.text("SOC",23,388,42,rotation=-90)
    f.xticks((300,637,1613,58),range(1,7),np.linspace(331,1884,6),649,38)
    f.text("Time after initialization [h]",1040,687,42)
    f.cover((1200,547,766,68))
    for x1,y1,x2,y2 in [(1200,602,1966,602),
                        *[(x,547,x,615) for x in (1263,1574,1884)]]:
        ET.SubElement(f.root,f"{{{SVG}}}line",{"x1":str(x1),"x2":str(x2),
            "y1":str(y1),"y2":str(y2),"stroke":"#ededed","stroke-width":"1.3"})

    f.cover((224,737,690,34))
    f.text("Initialization-induced difference",525,751,40)
    f.cover((1243,737,700,34))
    f.text("First entry and persistent recovery",1553,751,40)
    f.yticks((1,761,99,444),[f"{v:.2f}" for v in np.arange(0,.141,.02)],np.linspace(1191,776,8),98,34)
    f.text("|Shifted - correct SOC|",18,990,32,rotation=-90)
    f.xticks((174,1205,755,59),range(1,7),np.linspace(210,903,6),1217,36)
    f.text("Time after initialization [h]",525,1251,38)
    f.clear_plot_legend((487,786,445,70),grid_x=[487,626,765,903],grid_y=[835])
    annotations=[(217,836,176,48),(287,942,122,53),(420,1055,151,52)]
    for box in annotations:
        f.cover(box)
    f.restore_red_curve_pixels(annotations)
    # The three source guides share their dot spacing and phase. This clean
    # section of the third guide restores the second without touching a curve.
    scale=f.w/f.vw
    guide_top,guide_bottom=round(836*scale),round(884*scale)
    with Image.open(f.path) as original:
        guide=original.crop((778,guide_top,786,guide_bottom))
    buffer=BytesIO()
    guide.save(buffer,format="PNG")
    ET.SubElement(f.root,f"{{{SVG}}}image",{"x":str(529/scale),"y":str(guide_top/scale),
        "width":str(8/scale),"height":str((guide_bottom-guide_top)/scale),
        "{http://www.w3.org/1999/xlink}href":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")})
    f.yticks((1049,765,84,440),range(0,26,5),np.linspace(1191,778,6),1127,38)
    f.text("Recovery / 24-h censoring [h]",1061,981,37,rotation=-90)
    f.xticks((1210,1205,702,59),["DM","HDM","HECM","DD"],[1243,1451,1658,1865],1217,36)
    f.cover((1778,785,188,45))
    ET.SubElement(f.root,f"{{{SVG}}}line",{"x1":"1865","x2":"1865","y1":"785","y2":"830",
        "stroke":"#ededed","stroke-width":"1.3"})
    # The annotation crosses constant-height bar interiors. Extend a clean
    # neighbouring row to preserve their exact fills, edges and vertical grid.
    scale=f.w/f.vw
    x1,x2=round(1393*scale),round(1767*scale)
    y1,y2=round(1069*scale),round(1105*scale)
    with Image.open(f.path) as original:
        row=original.crop((x1,round(1059*scale),x2,round(1059*scale)+1))
        patch=row.resize((x2-x1,y2-y1),Image.Resampling.NEAREST)
    buffer=BytesIO()
    patch.save(buffer,format="PNG")
    ET.SubElement(f.root,f"{{{SVG}}}image",{"x":str(x1/scale),"y":str(y1/scale),
        "width":str((x2-x1)/scale),"height":str((y2-y1)/scale),
        "{http://www.w3.org/1999/xlink}href":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")})
    f.masks.append([1393,1069,374,36])
    f.reframe_panels([((0,141,1984,567),(0,46,1984,567)),
                      ((0,730,1984,543),(0,619,1984,543))])
    f.text("Initial SOC mismatch (10%)",992,22,40)
    for x,label,color in [(160,"DM","#2ca02c"),(360,"HDM","#9467bd"),
                           (570,"HECM","#1f77b4"),(805,"DD","#d62728")]:
        f.legend_line(x,1200,label,color)
    f.legend_line(1000,1200,"Reference",dash="12 7")
    f.legend_line(1250,1200,"Correct initial SOC",dash="3 7",size=32)
    f.legend_line(1630,1200,"Shifted initial SOC",size=32)
    f.legend_line(160,1245,"Recovery threshold (2%)",dash="10 5",size=30)
    f.legend_line(645,1245,"Scoring starts",color="#888888",dash="8 5",size=30)
    for x,label,fill in [(1020,"First 5-min entry","#e8e8e8"),
                         (1420,"Persistent return","#b8b8b8")]:
        ET.SubElement(f.root,f"{{{SVG}}}rect",{"x":str(x),"y":"1235","width":"42", "height":"20",
            "fill":fill,"stroke":"#888888","stroke-width":"2"})
        f.text(label,x+57,1245,32,anchor="start")
    f.save()


def voltage_spike():
    f=Figure("18_",1978)
    for box,text in [((127,9,60,40),"(a)"),((126,688,61,39),"(b)"),((1149,688,62,39),"(c)")]:
        f.replace(box,text,39,weight="bold")
    f.yticks((2,95,116,486),[f"{v:.2f}" for v in np.arange(.67,.741,.01)],np.linspace(568,115,8),114,31)
    f.text("SOC",29,317,35,rotation=-90)
    f.xticks((244,585,1460,71),["-50","0","50","100","150"],[282,629,976,1323,1670],600,31)
    f.text("Time relative to voltage spike [s]",1044,636,34)
    for box,text in [((1494,75,146,26),"Ground truth"),((1494,108,130,26),"DM"),
        ((1740,75,74,26),"HDM"),((1740,108,87,26),"HECM"),((1909,75,39,26),"DD")]:
        f.replace(box,text,28)
    f.yticks((1,740,117,457),[f"{v:.2f}" for v in np.arange(0,.301,.05)],np.linspace(1183,757,7),114,31)
    f.text("p95 peak excess error",29,956,34,rotation=-90)
    f.xticks((196,1198,682,38),["DM","HDM","HECM","DD"],[237,435,633,831],1212,32)
    f.yticks((992,749,148,435),[f"{v:.3f}" for v in np.arange(-.005,.0251,.005)],np.linspace(1167,770,7),1133,31)
    f.text("Excess absolute error",1009,959,34,rotation=-90)
    f.xticks((1180,1198,688,68),["-50","0","50","100","150"],[1218,1372,1525,1678,1832],1212,31)
    f.text("Time relative to voltage spike [s]",1555,1248,34)
    for box,text in [((1721,751,74,26),"DM"),((1721,784,83,26),"HDM"),
        ((1878,751,66,26),"HECM"),((1878,784,65,26),"DD")]:
        f.replace(box,text,29)
    f.save()


def current_noise():
    f=Figure("R01_",1742)
    for x,y,letter in [(922,23,"a"),(922,463,"b")]:
        f.replace((x-28,y-17,56,34),f"({letter})",32,weight="bold")
    f.yticks((3,50,101,325),["-0.5","0.0","0.5","1.0","1.5"],[361,288,215,142,70],99,28)
    f.text("Current [A]",29,202,30,rotation=-90)
    f.xticks((162,372,1521,51),range(0,31,5),[190,434,678,922,1166,1410,1654],387,27)
    f.text("Time [min]",922,414,29)
    f.replace((1230,59,147,23),"Baseline current",23)
    f.replace((1450,59,261,23),"Current with noise (\u03c3I = 0.10 A)",23)
    f.yticks((2,520,103,363),["0.0","0.1","0.2","0.3","0.4"],[869,787,705,623,541],99,28)
    f.text("SOC [-]",28,687,30,rotation=-90)
    f.xticks((162,902,1521,48),range(0,31,5),[190,434,678,922,1166,1410,1654],913,27)
    f.text("Time [min]",922,940,29)
    for box,text in [((237,497,119,22),"Model colors"),((183,524,153,23),"Reference SOC"),
        ((183,550,74,23),"DM"),((183,575,77,23),"HDM"),((399,524,58,23),"HECM"),((399,550,55,23),"DD"),
        ((1554,798,160,23),"Baseline prediction"),((1554,824,160,23),"Noise prediction"),
        ((1554,850,160,23),"Reference SOC")]:
        f.replace(box,text,25)
    f.replace((113,972,719,37),"Global error increase (95% CI)",30)
    f.replace((1010,972,717,37),"Local output response",30)
    f.yticks((6,1038,98,268),["-0.005","0.000","0.005","0.010"],[1287,1213,1139,1064],101,26)
    f.text("\u0394MAE [SOC]",27,1180,30,rotation=-90)
    f.yticks((890,1030,109,332),["0.0000","0.0005","0.0010","0.0015","0.0020","0.0025"],
             [1347,1288,1230,1171,1113,1054],996,24)
    f.text("p95 |\u0394\u0177k - \u0394\u0177k-1| [SOC]",916,1180,27,rotation=-90)
    for box,xs,center in [((121,1375,708,59),[149,800],475),((1010,1375,718,59),[1044,1695],1372)]:
        f.xticks(box,["0.02","0.10"],xs,1386,27)
        f.text("Current-noise std \u03c3I [A]",center,1415,30)
    for dx in [0,895]:
        for box,text in [((183+dx,1023,64,21),"DM"),((183+dx,1050,67,21),"HDM"),
            ((313+dx,1023,60,21),"HECM"),((313+dx,1050,57,21),"DD")]:
            f.replace(box,text,24)
    f.save()


def retained_heading_cleanup():
    f=Figure("07_")
    f.replace((800,0,265,95),"Input",69)
    f.replace((1550,0,285,95),"Output",69)
    f.save()
    f=Figure("24_")
    f.replace((644,82,150,27),"Pruning",27,bg=f.background(590,95),color="#444444")
    f.replace((632,255,180,31),"Quantization",27,bg=f.background(590,270),color="#444444")
    f.save()


if __name__ == "__main__":
    lfp_dataset()
    baseline()
    current_gain()
    heatmap()
    memory()
    inference_modes()
    model_sizes()
    embedded_errors()
    host_latency()
    initialization()
    voltage_spike()
    current_noise()
    retained_heading_cleanup()
