"""Report original OCR words not fully covered by the reviewed label masks."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT / "archive/tnr_qa"
for meta in sorted(QA.glob("*_labels.json")):
    data=json.loads(meta.read_text(encoding="utf-8"))
    if "viewbox" not in data:
        continue
    vw,vh=data["viewbox"]
    mask=np.zeros((math.ceil(vh),math.ceil(vw)),dtype=bool)
    for x,y,w,h in data["covered_label_areas"]:
        mask[max(0,int(y)):math.ceil(y+h),max(0,int(x)):math.ceil(x+w)]=True
    with Image.open(ROOT / "Bilder_Vortrag" / data["file"]) as image:
        scale=vw/image.width
    ocr=json.loads((QA / "ocr" / (Path(data["file"]).stem+".json")).read_text(encoding="utf-8-sig"))
    misses=[]
    for line in ocr["lines"]:
        for word in line["words"]:
            x,y,w,h=[word[k]*scale for k in ["x","y","w","h"]]
            region=mask[int(y):math.ceil(y+h),int(x):math.ceil(x+w)]
            fraction=region.mean() if region.size else 1
            if fraction<.98:
                misses.append([word["text"],round(x,1),round(y,1),round(w,1),round(h,1),round(float(fraction),2)])
    print(data["file"],json.dumps(misses))
