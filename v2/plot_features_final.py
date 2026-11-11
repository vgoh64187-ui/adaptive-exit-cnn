"""Plot checkpoint-bound feature and residual scales for both auxiliary exits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v2.evaluate_v2 import sha

root=Path.cwd()
data=json.loads((root/'results/v2/final_feature_analysis.json').read_text(encoding='utf-8'))
if data['n']!=100 or len(data['runs'])!=12:
    raise RuntimeError('Unexpected feature-analysis population')
if data['protocol_sha256']!=sha(root/'runs_v2/v2_screening_lock.json'):
    raise RuntimeError('Feature analysis belongs to another V2 lock')
if data['selection_sha256']!=sha(root/'runs_v2/screening_selection.json'):
    raise RuntimeError('Feature analysis belongs to another validation selection')
selection=json.loads((root/'runs_v2/screening_selection.json').read_text(encoding='utf-8'))
lock=json.loads((root/'runs_v2/v2_screening_lock.json').read_text(encoding='utf-8'))
selected=selection['selected_kd']
if selected not in ('residual_kd_seed42','residual_delayed_kd_seed42'):
    raise RuntimeError('Unexpected selected model')
prefix='residual_kd' if selected=='residual_kd_seed42' else 'residual_delayed_kd'
paths={'V1 local-global':('runs','lg'), 'V1 local-global + KD':('runs','lg_kd'),
       'V2 residual':('runs_v2','residual'), 'V2 residual + selected KD':('runs_v2',prefix)}
for record in data['runs']:
    base,stem=paths[record['group']]
    run=root/base/(stem+'_seed'+str(record['seed']))
    if sha(run/'best.pt')!=record['weights_sha256']:
        raise RuntimeError('Feature analysis does not match checkpoint')
    if base=='runs_v2' and lock['runs'][run.name]['best.pt']!=record['weights_sha256']:
        raise RuntimeError('Feature analysis does not match frozen V2 checkpoint')
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
groups=[('V1 local-global','原融合'),
        ('V1 local-global + KD','原融合\n＋蒸馏'),
        ('V2 residual','V2 残差融合'),
        ('V2 residual + selected KD','V2 残差融合\n＋选定蒸馏')]
metrics=[('local_over_global','局部特征 / 全局特征','#d57a23'),
         ('correction_over_global','残差修正 / 全局特征','#238e72')]
markers={17:'o',42:'s',73:'^'}
fig,axes=plt.subplots(1,2,figsize=(10,4.0),constrained_layout=True,sharey=True)
for index,ax in enumerate(axes):
    for g,(key,label) in enumerate(groups):
        rows=sorted([r for r in data['runs'] if r['group']==key],key=lambda r:r['seed'])
        if [r['seed'] for r in rows]!=[17,42,73]:
            raise RuntimeError('Expected three distinct seeds')
        for r in rows:
            seed=r['seed']
            offset={17:-.12,42:0.,73:.12}[seed]
            for metric,_,color in metrics:
                xpos=g+offset+(-.025 if metric=='local_over_global' else .025)
                value=r['exits'][index][metric]
                if not np.isfinite(value) or value<0:
                    raise RuntimeError('Invalid nonnegative ratio')
                ax.scatter(xpos,value,marker=markers[seed],color=color,s=35,
                           edgecolors='white',linewidths=.4,zorder=3)
    ax.axhline(1,color='#8897a3',linestyle='--',linewidth=.8)
    ax.set_yscale('symlog',linthresh=.01)
    ax.set_xticks(range(len(groups)),[g[1] for g in groups])
    ax.set_title('出口 '+str(index+1))
    ax.grid(axis='y',color='#e3e9ee',linewidth=.7)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
axes[0].set_ylabel('平均 L2 范数之比（对称对数刻度）')
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color=color,linestyle='none',label=label) for _,label,color in metrics]
handles += [Line2D([0],[0],marker=mark,color='#555',linestyle='none',label='种子 '+str(seed))
            for seed,mark in markers.items()]
fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
out=root/'figures/v2'
out.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf','svg'):
    fig.savefig(out/f'v2_feature_correction.{ext}',dpi=300,bbox_inches='tight')
print(out)
