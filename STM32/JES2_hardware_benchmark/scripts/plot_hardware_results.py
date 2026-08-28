#!/usr/bin/env python3
"""Create reproducible tables and figures from raw STM32 measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; OUT=RESULTS/'figures'; OUT.mkdir(parents=True,exist_ok=True)
ORDER=['DM','HDM','HECM','DD']; COLORS=dict(zip(ORDER,['#2ca02c','#9467bd','#1f77b4','#d62728']))
paths={'DM':RESULTS/'DM/measurements.csv','HDM':RESULTS/'HDM/measurements.csv','HECM':RESULTS/'HECM/measurements.csv','DD':RESULTS/'DD/manual_c_smoke/measurements.csv'}
frames=[]
for model,path in paths.items():
    frame=pd.read_csv(path); frame=frame[frame.status=='OK'].copy(); frame['model']=model; frames.append(frame)
raw=pd.concat(frames,ignore_index=True); raw.to_csv(RESULTS/'hardware_measurements_all.csv',index=False)
memory={x['model']:x for x in json.loads((RESULTS/'memory.json').read_text())['models']}
rows=[]
for model in ORDER:
    d=raw[raw.model==model]; latency=d.device_time_us.to_numpy(); error=d.abs_error.dropna().to_numpy()
    rows.append(dict(model=model,n=len(d),rounds=int(d['round'].nunique()),latency_median_us=np.median(latency),latency_p95_us=np.quantile(latency,.95),latency_max_us=np.max(latency),reference_mae=np.mean(error),reference_max_abs_error=np.max(error),flash_bytes=memory[model]['flash_load_bytes'],static_ram_bytes=memory[model]['static_ram_bytes']))
summary=pd.DataFrame(rows); dm=float(summary.loc[summary.model=='DM','latency_median_us'].iloc[0]); summary['relative_compute_energy']=summary.latency_median_us/dm
summary.to_csv(RESULTS/'hardware_benchmark_summary.csv',index=False); (RESULTS/'hardware_benchmark_summary.json').write_text(json.dumps({'models':summary.to_dict(orient='records')},indent=2))

plt.rcParams.update({'figure.dpi':160,'savefig.dpi':300,'font.size':10,'axes.grid':True,'grid.alpha':.2})
def save(name):
    plt.tight_layout(); plt.savefig(OUT/name,bbox_inches='tight'); plt.savefig(OUT/(Path(name).stem+'.pdf'),bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(8.2,4.5))
for model in ORDER:
    d=raw[(raw.model==model)&(raw['round']==raw[raw.model==model]['round'].min())]
    ax.plot(d.sample_id,d.device_time_us,color=COLORS[model],lw=1,label=model)
ax.set_yscale('log'); ax.set_xlabel('Sample ID'); ax.set_ylabel('Inference time [us]'); ax.legend(ncol=4); save('hardware_latency_trace.png')

fig,ax=plt.subplots(figsize=(7.2,4.6))
for model in ORDER:
    x=np.sort(raw[raw.model==model].device_time_us.to_numpy()); y=np.arange(1,len(x)+1)/len(x); ax.plot(x,y,color=COLORS[model],lw=2,label=model)
ax.set_xscale('log'); ax.set_xlabel('Inference time [us]'); ax.set_ylabel('Empirical cumulative probability'); ax.legend(); save('hardware_latency_ecdf.png')

fig,ax=plt.subplots(figsize=(8.2,4.5))
for model in ORDER:
    d=raw[(raw.model==model)&(raw['round']==raw[raw.model==model]['round'].min())]; ax.plot(d.sample_id,d.soc_device-d.soc_reference,color=COLORS[model],lw=.9,label=model)
ax.axhline(0,color='black',lw=.8); ax.set_xlabel('Sample ID'); ax.set_ylabel('STM32 minus software SOC'); ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3)); ax.legend(ncol=4); save('hardware_numerical_difference_trace.png')

fig,ax=plt.subplots(figsize=(7.2,5.0))
offsets={'DM':(8,-18),'HDM':(8,10),'HECM':(7,6),'DD':(7,7)}
for _,r in summary.iterrows():
    size=90+350*r.static_ram_bytes/summary.static_ram_bytes.max(); ax.scatter(r.flash_bytes/1024,r.latency_median_us,s=size,color=COLORS[r.model],edgecolor='white',linewidth=1.2); ax.annotate(r.model,(r.flash_bytes/1024,r.latency_median_us),xytext=offsets[r.model],textcoords='offset points',weight='bold')
ax.set_yscale('log'); ax.set_xlabel('Flash footprint [KiB]'); ax.set_ylabel('Median inference time [us]'); save('hardware_flash_latency_tradeoff.png')

fig,axes=plt.subplots(1,2,figsize=(9,4.3))
for ax,column,label in [(axes[0],'flash_bytes','Flash [KiB]'),(axes[1],'static_ram_bytes','Static RAM [KiB]')]:
    values=summary[column]/1024; ax.bar(ORDER,values,color=[COLORS[x] for x in ORDER]); ax.set_ylabel(label)
    for i,v in enumerate(values): ax.text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
save('hardware_memory_footprints.png')

print(summary.to_string(index=False))
