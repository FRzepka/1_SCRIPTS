"""Build a separate study guide and auditable figure inventory from the thesis PDF."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
from html import escape

import pymupdf as fitz
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
doc = fitz.open(ROOT / 'main.pdf')
source_hash = hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest()
old_inventory = HERE/'figure_inventory.json'
old_hash = json.loads(old_inventory.read_text()).get('source_pdf_sha256') if old_inventory.exists() else None
figures = []
for page_no, page in enumerate(doc):
    txt = page.get_text()
    for match in re.finditer(r'Figure\s+([A0-9]+\.[0-9]+):', txt):
        fid = match.group(1)
        if any(x['id'] == fid for x in figures):
            continue
        tail = txt[match.end():]
        figures.append({'id': fid, 'pdf_page': page_no + 1,
                        'caption_start': ' '.join(tail.split()[:65])})
pages = sorted(set(x['pdf_page'] for x in figures))
for n in pages:
    out = ASSETS / f'page_{n:03d}.jpg'
    if not out.exists() or old_hash != source_hash:
        doc[n-1].get_pixmap(matrix=fitz.Matrix(1.15, 1.15)).save(out)
for start in range(0, len(pages), 12):
    subset = pages[start:start+12]
    sheet = Image.new('RGB', (1600, 1800), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, n in enumerate(subset):
        im = Image.open(ASSETS / f'page_{n:03d}.jpg')
        im.thumbnail((390, 555))
        x, y = (j % 4)*400, (j // 4)*600
        sheet.paste(im, (x, y+30))
        ids = ', '.join(f['id'] for f in figures if f['pdf_page'] == n)
        draw.text((x+8,y+5), f'PDF {n} / Abb. {ids}', fill='black')
    sheet.save(ASSETS / f'contact_{start//12+1}.jpg')
inventory = {'source_pdf_sha256': source_hash,
             'source_pages': len(doc), 'figures': figures}
(HERE/'figure_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+'\n')
if '--inventory' in sys.argv:
    print(f'{len(figures)} figures on {len(pages)} pages')
    raise SystemExit(0)

fontdir = Path('/usr/share/fonts/truetype/dejavu')
for name, file in [('Guide','DejaVuSans.ttf'),('GuideBold','DejaVuSans-Bold.ttf'),
                   ('GuideItalic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Guide',normal='Guide',bold='GuideBold',italic='GuideItalic')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyGuide',fontName='Guide',fontSize=9.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='HeadingGuide',fontName='GuideBold',fontSize=20,leading=26,
                          textColor=colors.HexColor('#a90e24'),spaceAfter=16))
styles.add(ParagraphStyle(name='Question',fontName='GuideBold',fontSize=11,leading=15,
                          spaceBefore=11,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='SmallGuide',fontName='Guide',fontSize=8,leading=11,spaceAfter=6))

story=[]
class GuideDoc(SimpleDocTemplate):
    def beforeDocument(self):
        self.heading_count = 0
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'HeadingGuide':
            title = flowable.getPlainText()
            key = f'section-{self.heading_count}'
            self.heading_count += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0)
            self.notify('TOCEntry', (0, title, self.page, key))

story.append(Paragraph('Navigation und Lernumfang',styles['HeadingGuide']))
story.append(Paragraph('114 Fachfragen mit Antworten, 12 Rechenuebungen, 10 kritische Pruefpunkte, '
                       'ein Lernplan, 67 Abbildungen und 28 Tabellen. Das Inhaltsverzeichnis und '
                       'die PDF-Lesezeichen fuehren zu den Themen und einzelnen Abbildungen.',styles['BodyGuide']))
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle(name='TOCGuide',fontName='Guide',fontSize=8.5,leading=12,
                                leftIndent=0,firstLineIndent=0,spaceBefore=3)]
story.append(toc)
def rich(text):
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',r'<a href="\2" color="#007a86">\1</a>',text)
    return text

for src in ['Fragen_und_Antworten.md','Abbildungsatlas.md','Tabellen_und_Vertiefung.md']:
    content=(HERE/src).read_text()
    buffer=[]
    def flush():
        if buffer:
            story.append(Paragraph(rich(' '.join(buffer)),styles['BodyGuide']))
            buffer.clear()
    for line in content.splitlines():
        if line.startswith('# '):
            flush()
            if story: story.append(PageBreak())
            story.append(Paragraph(rich(line[2:]),styles['HeadingGuide']))
        elif line.startswith('## '):
            flush()
            story.append(Paragraph(rich(line[3:]),styles['Question']))
        elif line.startswith('@figure '):
            flush()
            fid=line.split()[1]
            f=next(x for x in figures if x['id']==fid)
            p=ASSETS/f"page_{f['pdf_page']:03d}.jpg"
            # Keep the full source page so axes and captions remain auditable.
            story.append(RLImage(str(p),width=123*mm,height=174*mm,kind='proportional'))
            story.append(Paragraph(f"Originalseite der Dissertation: PDF-Seite {f['pdf_page']}. "
                                   'Die folgende Interpretation bezieht sich auf die bezeichnete Abbildung.',styles['SmallGuide']))
        elif not line.strip(): flush()
        elif line.startswith('- '):
            flush()
            story.append(Paragraph('• '+rich(line[2:]),styles['BodyGuide']))
        else: buffer.append(line)
    flush()

def footer(canvas,document):
    canvas.setFont('Guide',8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(18*mm,12*mm,'Verteidigungsvorbereitung | Arbeitsstand 16.09.2026')
    canvas.drawRightString(192*mm,12*mm,str(document.page))

out=HERE/'Dissertation_Fragenkatalog_und_Lernskript.pdf'
draft=ASSETS/'guide_layout.pdf'
pdf=GuideDoc(str(draft),pagesize=(210*mm,297*mm),rightMargin=18*mm,leftMargin=18*mm,
                     topMargin=18*mm,bottomMargin=20*mm,
                     title='Dissertation: Fragenkatalog, Antworten und Abbildungsatlas',author='Vorbereitung fuer Florian Rzepka')
pdf.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
# Import original vector pages in place of preview bitmaps to preserve zoom quality.
assembled=fitz.open(draft)
for page in assembled:
    match=re.search(r'Originalseite der Dissertation: PDF-Seite (\d+)',page.get_text())
    if match:
        images=page.get_images()
        assert len(images)==1, 'Expected exactly one source-page placeholder'
        xref=images[0][0]
        rect=page.get_image_rects(xref)[0]
        page.delete_image(xref)
        page.show_pdf_page(rect,doc,int(match.group(1))-1)
complete=ASSETS/'guide_complete.pdf'
assembled.save(complete,garbage=4,deflate=True)
assembled.close()
os.replace(complete,out)
result=fitz.open(out)
print(f'Created {out.name}: {len(result)} pages, {len(figures)} source figures')
