from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1]
out=root/'tools'
vals={'SOC':np.array([[2.6845146,108924/1024,4.93,700.83],[2.3379620,68436/1024,4.03,400.38],[2.7911590,54820/1024,3.96,3494.66]]),'SOH':np.array([[.8523505,359572/1024,8.69,11366.93],[1.4573121,203924/1024,6.96,6362.23],[1.4103794,158532/1024,6.70,14604.27]])}
w=np.array([[a,f,r,20-a-f-r] for a in range(21) for f in range(21-a) for r in range(21-a-f)])/20
results={}
for task,v in vals.items():
 ratios=v/v[0]; counts=np.bincount(np.argmin(w@ratios.T,axis=1),minlength=3);results[task]={'equal':ratios.mean(axis=1).tolist(),'shares':(counts/len(w)*100).tolist(),'flash_savings':((1-v[:,1]/v[0,1])*100).tolist()}
(out/'scores.json').write_text(json.dumps(results,indent=2))
# Regenerate related scientific summary from unchanged accuracy/RAM/timing and audited flash.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
colors=['#2ca02c','#e62828','#1f77b4'];fig=plt.figure(figsize=(14.67,10));gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]);axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
for ax,(task,v),letter in zip(axes,vals.items(),'ab'):
 for metric in range(4):
  for fw in np.arange(.25,.851,.05):
   ww=np.full(4,(1-fw)/3);ww[metric]=fw;win=np.argmin((v/v[0])@ww)
   ax.scatter(fw*100,3-metric,s=320,marker='s',facecolors=matplotlib.colors.to_rgba(colors[win],.4),edgecolors=colors[win])
 ax.set(yticks=range(4),yticklabels=['Energy','RAM','Flash','Accuracy'],xticks=range(25,86,10),ylim=(-.5,3.5),xlabel='Weight of highlighted objective [%]',title=f'({letter}) {task}')
 ax.grid(axis='x',alpha=.15)
ax=fig.add_subplot(gs[1,:]);xx=np.arange(2)
for i,name in enumerate(['Base','Pruned','Quantized']):
 bs=ax.bar(xx+(i-1)*.23,[results[t]['shares'][i] for t in vals],.23,color=matplotlib.colors.to_rgba(colors[i],.4),edgecolor=colors[i],label=name);ax.bar_label(bs,fmt='%.1f',padding=4)
ax.set(xticks=xx,xticklabels=list(vals),ylim=(0,105),ylabel='Winning combinations [%]',title='(c) Ranking across all 1771 weight combinations');ax.grid(axis='y',alpha=.15);ax.legend(loc='upper right');fig.tight_layout(pad=3)
fig.savefig(root/'pictures/eaai_palette/embedded_utility_sensitivity.png',dpi=180);plt.close(fig)
