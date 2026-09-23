"""Structural, provenance, arithmetic and full-page rendering checks."""
from pathlib import Path
import hashlib
import json
import re
import pymupdf

here=Path(__file__).resolve().parent
root=here.parent
inventory=json.loads((here/'figure_inventory.json').read_text())
pdf=pymupdf.open(here/'Dissertation_Fragenkatalog_und_Lernskript.pdf')
md='\n'.join((here/f).read_text() for f in ['Fragen_und_Antworten.md','Abbildungsatlas.md','Tabellen_und_Vertiefung.md'])
questions=re.findall(r'^## (F\d{3})\.',md,re.M)
exercises=re.findall(r'^## (R\d{2})\.',md,re.M)
points=re.findall(r'^## (P\d{2})\.',md,re.M)
atlas=re.findall(r'^@figure (\S+)',md,re.M)
assert questions == [f'F{i:03d}' for i in range(1,115)]
assert len(exercises)==12 and len(points)==10
assert atlas == [x['id'] for x in inventory['figures']]
assert hashlib.sha256((root/'main.pdf').read_bytes()).hexdigest()==inventory['source_pdf_sha256']
assert sum(a*b+b for a,b in zip([48,128,256,256,256,128,512,128,256,256],
                              [128,256,256,256,128,512,128,256,256,1]))==434561
assert 4*64*(6+64)+64*64+64==22080
assert 4*45*(6+45)+45*64+64==12124
texts=[p.get_text() for p in pdf]
assert all(x in '\n'.join(texts) for x in questions)
assert len(pdf.get_toc()) >= 80
for page in pdf:
    page.get_pixmap(matrix=pymupdf.Matrix(0.4,0.4))
out=here/'assets'
samples=[0,4,5,len(pdf)-1]
for marker in ['F030.','F044.','Abbildung 4.4: LFP-SOH','Abbildung 6.9: Initialisierungs',
               'Abbildung A.12: Statische','F101.']:
    hits=[i for i,t in enumerate(texts) if marker in t and len(t)>1200]
    if hits: samples.append(hits[-1])
for i in sorted(set(samples)):
    pdf[i].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(out/f'guide_check_{i+1:03d}.png')
source=pymupdf.open(root/'main.pdf')
table_ids=sorted(set(re.findall(r'Table\s+([A0-9]+\.[0-9]+):','\n'.join(p.get_text() for p in source))))
result={'pdf_pages':len(pdf),'question_count':len(questions),'exercises':len(exercises),
        'open_points':len(points),'covered_figures':len(atlas),'source_table_count':len(table_ids),
        'all_pages_rendered':True,'original_pdf_hash_unchanged':True,
        'source_pdf_sha256':inventory['source_pdf_sha256'],
        'checked_source_files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [root/'main.tex',*sorted((root/'chapters').glob('*.tex')),
                    root/'tables/jes_final/jes2_hecm_lookup_sensitivity_compact.tex',
                    root/'tables/jes_final/jes2_dataset_cell_split_coverage.tex']},
        'render_preview_pages':[i+1 for i in sorted(set(samples))],
        'scope':'Structural and rendering checks, not a rerun of thesis experiments.'}
(here/'validation_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checked_source_files'},indent=2))
