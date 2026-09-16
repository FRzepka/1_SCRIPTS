"""Build a complete, filename-compatible Times New Roman figure directory."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.patches import Patch
import convert_native_figure_fonts as native
import typeset_saved_figure_labels as labels

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Bilder_Vortrag"
OUT = SOURCE / "Times_New_Roman"
QA = ROOT / "archive/tnr_qa"
DATA = ROOT / "archive/plot_sources/LATEX/JES/paper_robustness_benchmark/JES_2.0/results"
ALREADY_SERIF = ("02a_", "02b_", "06_", "08_", "09_", "10_", "23_")


def synthesis():
    for name in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
        font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / name))
    assert Path(font_manager.findfont("Times New Roman", fallback_to_default=False)).stem == "times"
    plt.rcParams.update({"font.family":"Times New Roman", "font.size":22,
        "axes.labelsize":24, "xtick.labelsize":21, "ytick.labelsize":21,
        "legend.fontsize":23, "axes.axisbelow":True, "svg.fonttype":"none"})
    dimensions=pd.read_csv(DATA / "jes2_revised_decision_dimensions.csv").set_index("Model")
    profiles=pd.read_csv(DATA / "jes2_revised_decision_profiles.csv").set_index("Model")
    path=next(SOURCE.glob("20_*.png"))
    with Image.open(path) as original:
        width,height=original.size
    dpi=240
    fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi)
    polar=fig.add_axes([.083,.2,.27,.592],projection="polar")
    bars=fig.add_axes([.525,.219,.455,.561])
    models=["DM","HDM","HECM","DD"]
    colors=["#2ca02c","#9467bd","#1f77b4","#d62728"]
    categories=["Accuracy","Robustness","Recovery"]
    theta=np.linspace(0,2*np.pi,3,endpoint=False)
    polar.set_theta_offset(np.pi/2)
    polar.set_theta_direction(-1)
    for model,color in zip(models,colors):
        values=dimensions.loc[model,categories].to_numpy(float)
        polar.plot(np.r_[theta,theta[0]],np.r_[values,values[0]],color=color,lw=2.3,marker="o",markersize=7)
        polar.fill(np.r_[theta,theta[0]],np.r_[values,values[0]],color=color,alpha=.12)
    polar.set_thetagrids(np.rad2deg(theta),categories)
    polar.tick_params(axis="x",pad=24,labelsize=21)
    for label,alignment in zip(polar.get_xticklabels(),["center","left","right"]):
        label.set_horizontalalignment(alignment)
    polar.set_ylim(0,1)
    polar.set_yticks([.25,.5,.75,1])
    polar.set_yticklabels(["0.25","0.50","0.75","1.00"],fontsize=17,color="#666666")
    polar.set_rlabel_position(0)
    polar.grid(color="#dddddd",lw=.8)
    polar.spines["polar"].set_color("#cccccc")
    for i,(model,color) in enumerate(zip(models,colors)):
        y=profiles.loc[model,[c+"-weighted" for c in categories]].to_numpy(float)
        bars.bar(np.arange(3)+(i-1.5)*.16,y,.12,color=color,alpha=.40,edgecolor=color,lw=2)
    bars.set_xticks(range(3),[c+"-weighted" for c in categories],fontsize=20)
    bars.set(ylim=(0,1.02),ylabel="Composite score")
    bars.set_yticks(np.arange(0,1.01,.2))
    bars.grid(axis="y",color="#dddddd",lw=.8)
    bars.spines[["top","right"]].set_visible(False)
    fig.text(.236,.92,"Relative dimensions",ha="center",fontsize=23,fontweight="normal")
    fig.text(.741,.92,"Priority profiles",ha="center",fontsize=23,fontweight="normal")
    fig.legend(handles=[Patch(facecolor=c,edgecolor=c,alpha=.4,label=m) for m,c in zip(models,colors)],
        loc="center",bbox_to_anchor=(.74,.85),ncol=4,frameon=False,columnspacing=1.1,handlelength=1.6)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
                overflow.append(text.get_text())
    assert not overflow, overflow
    fig.savefig(OUT / path.name,dpi=dpi,facecolor="white")
    fig.savefig(ROOT / "archive/tnr_build" / (path.stem+".svg"),facecolor="white")
    plt.close(fig)
    print(path.name,flush=True)


def retained_figures():
    OUT.mkdir(parents=True,exist_ok=True)
    for prefix in (*ALREADY_SERIF,"04_"):
        path=next(SOURCE.glob(prefix+"*.png"))
        shutil.copy2(path,OUT / path.name)


def verify():
    inventory=json.loads((QA / "original_inventory.json").read_text(encoding="utf-8"))
    expected={row["file"]:row for row in inventory}
    actual={p.name for p in OUT.glob("*.png")}
    assert set(expected) == actual, {"missing":sorted(set(expected)-actual),"extra":sorted(actual-set(expected))}
    report=[]
    for name,row in expected.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()==row["sha256"],name
        with Image.open(OUT / name) as image:
            image.verify()
        with Image.open(OUT / name) as image:
            assert list(image.size)==row["size"],(name,image.size,row["size"])
        method="existing serif artwork retained" if name.startswith(ALREADY_SERIF) else "Times New Roman typesetting"
        if name.startswith("04_"):
            method="hardware photo unchanged"
        report.append({"file":name,"size":row["size"],"original_unchanged":True,
            "method":method,"sha256":hashlib.sha256((OUT / name).read_bytes()).hexdigest()})
    (QA / "complete_inventory.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    text_count=0
    for name in expected:
        svg=ROOT / "archive/tnr_build" / (Path(name).stem+".svg")
        if not svg.exists():
            continue
        for node in ET.parse(svg).getroot().iter():
            if node.tag.rsplit("}",1)[-1] not in ("text","tspan"):
                continue
            value="".join(node.itertext()).strip()
            if not value:
                continue
            assert not re.match(r"^(?:\([a-z]\)|[a-z]\))\s*",value),(name,value)
            assert node.get("font-weight","normal") not in ("bold","bolder","600","700","800","900"),(name,value)
            assert not re.search(r"font-weight\s*:\s*(?:bold|bolder|[6-9]00)",node.get("style","")),(name,value)
            text_count+=1
    print(f"Typography check: {text_count} SVG text elements, no bold weights or panel letters.",flush=True)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18)
    files=sorted(OUT.glob("*.png"))
    for start in range(0,len(files),6):
        sheet=Image.new("RGB",(1500,1260),"white")
        draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(files[start:start+6]):
            with Image.open(path) as source:
                rgba=source.convert("RGBA")
                white=Image.new("RGBA",rgba.size,"white")
                white.alpha_composite(rgba)
                thumb=ImageOps.contain(white.convert("RGB"),(735,365))
            x,y=(i%2)*750,(i//2)*420
            draw.text((x+8,y+6),path.name,fill="black",font=font)
            sheet.paste(thumb,(x+(750-thumb.width)//2,y+40))
        sheet.save(QA / f"completed_{start//6+1}.jpg")
    print(f"Verified: {len(report)} complete PNGs, identical filenames/dimensions, unchanged originals.",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--finish-only",action="store_true")
    args=parser.parse_args()
    if not args.verify_only:
        if not args.finish_only:
            native.main()
            for function in [labels.lfp_dataset,labels.baseline,labels.current_gain,labels.heatmap,
                labels.memory,labels.inference_modes,labels.model_sizes,labels.embedded_errors,
                labels.host_latency,labels.initialization,labels.voltage_spike,labels.current_noise,
                labels.retained_heading_cleanup]:
                function()
        retained_figures()
        synthesis()
    verify()


if __name__=="__main__":
    main()
