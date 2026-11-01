"""Regenerate evidence-linked figures with type sized for A4 paper width.

The original plotting code and numerical sidecars are reused unchanged.  Only
font sizes are adjusted; data, axes, scales, and plotted values are unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.text as mtext
import make_figures as figures
import plot_ablation,plot_deployment,make_examples
import torch

original_save=figures.save

def print_save(fig,name,evidence,out):
    for text in fig.findobj(match=mtext.Text):
        text.set_fontsize(text.get_fontsize()*1.35)
    if name.endswith('dynamic_examples'):
        fig.tight_layout(rect=[0,0,1,.92])
    evidence=dict(evidence,print_layout='All text enlarged by 35%; plotted data unchanged.')
    original_save(fig,name,evidence,out)

figures.save=print_save
plot_ablation.save=print_save
plot_deployment.save=print_save
make_examples.save=print_save
torch.set_num_threads(6)
out='figures/print'
summary='results/final/summary.json'
figures.result_figures(summary,out)
figures.exit_figures(summary,out)
figures.routing_figures(summary,out,'local_global',.5,False)
paired='results/final/paired_effects.json'
manifest='configs/main_queue.json'
content=plot_ablation.read_audited(paired,manifest)
src,_=figures.source(paired)
plot_ablation.plot(content['comparisons'],.75,out,src)
plot_deployment.plot(plot_deployment.read_campaign('configs/benchmark_matrix.json','docs/protocol_lock.json'),out)
make_examples.build('runs/lg_kd_seed42',.75,False,'data',out)
print('Print figures regenerated with original audit guards.')
