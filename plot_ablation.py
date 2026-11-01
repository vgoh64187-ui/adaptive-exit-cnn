"""Plot seed-paired accuracy and actual compute changes at a target budget."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from make_figures import plt,save,source
from inference import file_sha256
from summarise import collect
from paired_effects import compare

LABELS={'local_features_vs_gap':'融合头 − GAP 头',
        'fusion_vs_matched_capacity':'融合头 − 近等参数 MLP',
        'distillation':'融合＋蒸馏 − 融合',
        'consensus_without_distillation':'一致性 − 置信度（无蒸馏）',
        'consensus_with_distillation':'一致性 − 置信度（有蒸馏）'}


def read_audited(path,manifest):
    payload=json.loads(Path(path).read_text());rows,provenance=collect(manifest)
    if (payload['manifest_sha256']!=file_sha256(manifest) or payload['provenance']!=provenance
        or payload['comparisons']!=compare(rows)):
        raise RuntimeError('Paired comparisons differ from current audited test data')
    return payload


def plot(records,budget,output,source_record,development=False):
    wanted=[r for r in records if r['target_validation_budget']==budget]
    if len(wanted)!=len(LABELS) or {r['contrast'] for r in wanted}!=set(LABELS):
        raise RuntimeError('Missing or duplicate paired comparison')
    wanted.sort(key=lambda r:list(LABELS).index(r['contrast']))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained',sharey=True)
    offset={17:-.15,42:0.,73:.15};colors={17:'#397eb5',42:'#e48d32',73:'#39956d'}
    for i,record in enumerate(wanted):
        if [x['seed'] for x in record['pairs']]!=[17,42,73]:raise RuntimeError('Three predetermined seeds are required')
        for col,metric in enumerate(['accuracy','cost_ratio']):
            values=[pair[metric+'_delta_pp'] for pair in record['pairs']]
            mean=record['summary'][metric+'_delta_pp_mean'];std=record['summary'][metric+'_delta_pp_std']
            if not np.isclose(mean,np.mean(values)) or not np.isclose(std,np.std(values,ddof=1)):
                raise RuntimeError('Paired statistics do not match seed points')
            axes[col].errorbar(mean,i,xerr=std,fmt='D',color='#343b47',markersize=4,capsize=3,lw=1,zorder=3)
            for pair in record['pairs']:
                axes[col].scatter(pair[metric+'_delta_pp'],i+offset[pair['seed']],s=20,
                                  color=colors[pair['seed']],zorder=4)
    axes[0].set_yticks(range(len(wanted)),[LABELS[x['contrast']] for x in wanted]);axes[0].invert_yaxis()
    for ax in axes:ax.axvline(0,color='#888888',ls='--',lw=.8);ax.grid(axis='x',alpha=.2)
    axes[0].set(xlabel='测试准确率差（百分点）',title='(a) 识别变化')
    axes[1].set(xlabel='测试实际计算比差（百分点）',title='(b) 实际计算变化')
    handles=[plt.Line2D([],[],color=colors[s],marker='o',ls='',label=f'种子 {s}') for s in [17,42,73]]
    handles.append(plt.Line2D([],[],color='#343b47',marker='D',ls='',label='均值；横线为样本标准差'))
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle(f'验证目标预算 {budget:.0%} 下的同种子配对差值'+(' · 仅布局检查' if development else ''),fontsize=12,color='#aa3333' if development else '#222222')
    save(fig,'paired_ablations',dict(scope='Test accuracy and actual cost; target budget selected on validation',
                                    sign='comparison minus reference',records=wanted,sources=[source_record],
                                    uncertainty='Three independent seeds; sample standard deviation, not significance testing',
                                    development=development),output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paired',default='results/final/paired_effects.json')
    p.add_argument('--manifest',default='configs/main_queue.json');p.add_argument('--budget',type=float,default=.75)
    p.add_argument('--output',default='figures/final');a=p.parse_args();torch.set_num_threads(6)
    data=read_audited(a.paired,a.manifest);src,_=source(a.paired);plot(data['comparisons'],a.budget,a.output,src)
