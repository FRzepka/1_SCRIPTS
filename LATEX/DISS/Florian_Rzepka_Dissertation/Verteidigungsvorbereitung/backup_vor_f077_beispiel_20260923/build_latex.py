"""Build the editable LaTeX study guide from its maintained Markdown sources.
Run: python build_latex.py --compile
Requires XeLaTeX with Arial; no third-party Python packages.
"""
from pathlib import Path
import re,json,subprocess,argparse
HERE=Path(__file__).resolve().parent

def escape(s):
 replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','/':r'/\allowbreak{}'}
 s=s.replace('–','-').replace('—','-').replace('‑','-')
 return ''.join(replacements.get(c,c) for c in s)

def rich(s):
 # Preserve original Markdown emphasis, code spans and web references.
 pat=r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\(https?://[^\s)]+\))'
 result=[]
 for p in re.split(pat,s):
  if p.startswith('**'):result.append(r'\textbf{'+escape(p[2:-2])+'}')
  elif p.startswith('`'):result.append(escape(p[1:-1]))
  elif re.match(r'\[[^\]]+\]\(https?://',p):
   m=re.match(r'\[([^\]]+)\]\(([^)]+)\)',p);result.append(escape(m[1])+' ('+escape(m[2])+')')
  else:result.append(escape(p))
 return ''.join(result)

def build():
 inv=json.loads((HERE/'figure_inventory.json').read_text(encoding='utf-8'))
 figs={f['id']:f['pdf_page'] for f in inv['figures']}
 out=[(HERE/'preamble.tex').read_text(encoding='utf-8')]
 for fn in ['Fragen_und_Antworten.md','Abbildungsatlas.md','Tabellen_und_Vertiefung.md']:
  buffer=[]
  def flush():
   if buffer:out.append(rich(' '.join(buffer))+'\n\n');buffer.clear()
  for line in (HERE/fn).read_text(encoding='utf-8').splitlines():
   if line.startswith('# '):flush();out.append(r'\topic{'+rich(line[2:])+'}\n')
   elif line.startswith('## '):flush();out.append(r'\question{'+rich(line[3:])+'}\n')
   elif line.startswith('@figure '):
    flush();page=figs[line.split()[1]];asset=f'figures/thesis-page-{page:03}.pdf'
    if not (HERE/asset).exists():raise FileNotFoundError(asset)
    out.append(r'\begin{center}\includegraphics[width=\linewidth,height=0.60\textheight,keepaspectratio]{'+asset+r'}\end{center}'+'\n')
    out.append(rich(f'Originalseite der Dissertation: PDF-Seite {page}. Die folgende Interpretation bezieht sich auf die bezeichnete Abbildung.')+'\n\n')
   elif not line.strip():flush()
   elif line.startswith('- '):flush();out.append(r'\noindent\textbullet\enspace '+rich(line[2:])+'\n\n')
   else:buffer.append(line)
  flush()
 out.append((HERE/'Modelluebersicht_und_Begriffe.tex').read_text(encoding='utf-8'))
 dst=HERE/'Fragenkatalog.tex';dst.write_text(''.join(out),encoding='utf-8')
 return dst

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compile',action='store_true');args=p.parse_args();print(build())
 if args.compile:
  for _ in range(3):subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-jobname=Dissertation_Fragenkatalog_und_Lernskript','Fragenkatalog.tex'],cwd=HERE,check=True)
