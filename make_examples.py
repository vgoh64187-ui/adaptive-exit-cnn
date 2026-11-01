"""Illustrate deterministic validation cases with verified real dynamic paths.

One smallest-index example per (exit, correct/incorrect) stratum. These cases
explain execution; they do not estimate overall accuracy or test performance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from make_figures import save, NAMES
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from inference import load_run, load_policy, file_sha256
from src.models import route_cached, profile_macs
from src.data import MEAN, STD


@torch.no_grad()
def build(run, budget=.75, consensus=False, data='data', output='figures/application'):
    torch.set_num_threads(2)
    run=Path(run);data=Path(data)
    if not (run/'completed.json').exists():
        raise RuntimeError('Examples require completed training')
    model,config,weights=load_run(run,'cpu')
    if config['smoke_only'] or config['head']=='baseline':
        raise RuntimeError('Use a complete multi-exit model')
    policy,selected=load_policy(run,weights,budget,consensus)
    split=json.loads((data/'split_cifar100.json').read_text())
    meta=json.loads((run/'validation_predictions.metadata.json').read_text())
    cache=run/'validation_predictions.npz'
    if meta['weights_sha256']!=weights or meta['prediction_sha256']!=file_sha256(cache) or meta['split_sha256']!=split['sha256']:
        raise RuntimeError('Validation cache provenance mismatch')
    if meta['split']!='validation' or split['sha256']!=config['split_sha256']:
        raise RuntimeError('Example split mismatch')
    with np.load(cache,allow_pickle=False) as f:
        mask=np.isin(f['indices'],split['selection'])
        indices=f['indices'][mask];labels=torch.from_numpy(f['labels'][mask])
        logits=[torch.from_numpy(f[f'logits_{i}'][mask]) for i in range(3)]
    chosen,ids=route_cached(logits,policy)
    correct=chosen.argmax(1).eq(labels)
    strata=[];examples=[]
    for exit_id in range(3):
        for is_correct in [True,False]:
            positions=torch.where((ids==exit_id)&(correct==is_correct))[0].tolist()
            strata.append(dict(exit=exit_id+1,correct=is_correct,count=len(positions)))
            if positions:
                examples.append(min(positions,key=lambda p:int(indices[p])))
    ds=datasets.CIFAR100(data,train=True,download=False)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    costs=profile_macs(model);rows=[]
    fig,axes=plt.subplots(len(examples),2,figsize=(11,1.6*len(examples)),
                          gridspec_kw={'width_ratios':[1,6]},squeeze=False)
    for axes_row,pos in zip(axes,examples):
        image,label=ds[int(indices[pos])];x=tf(image).unsqueeze(0)
        if label!=int(labels[pos]):raise RuntimeError('Dataset label differs from prediction cache')
        visits=[0]*4
        def hook(i):
            def count(m,inputs,out):visits[i]+=len(inputs[0])
            return count
        handles=[s.register_forward_hook(hook(i)) for i,s in enumerate(model.stages)]
        actual,actual_id=model.predict_dynamic(x,policy)
        for h in handles:h.remove()
        e=int(actual_id.item());prediction=int(actual.argmax(1).item())
        if e!=int(ids[pos]) or prediction!=int(chosen[pos].argmax()):
            raise RuntimeError('CPU dynamic example differs from cached route; inspect precision before plotting')
        expected=[1,1,int(e>=1),int(e>=2)]
        if visits!=expected:raise RuntimeError('Stage visitation mismatch')
        probability=float((actual/policy.temperatures[e]).softmax(1).max())
        macs=costs['exit_macs'][e]
        rows.append(dict(index=int(indices[pos]),truth=ds.classes[label],prediction=ds.classes[prediction],
                         correct=prediction==label,exit=e+1,probability=probability,stage_visits=visits,
                         cumulative_macs=macs,cost_ratio=macs/costs['backbone_macs']))
        left,right=axes_row;left.imshow(image,interpolation='nearest');left.axis('off')
        left.set_title(f"验证索引 {indices[pos]}",fontsize=10)
        right.set_xlim(0,10);right.set_ylim(0,2);right.axis('off')
        right.text(.1,1.7,f"真实类别：{ds.classes[label]}    预测：{ds.classes[prediction]}    {'正确' if prediction==label else '错误'}",fontsize=11)
        for i,visited in enumerate(visits):
            color='#dceaf5' if visited else '#f1f1f1';edge='#397eb5' if visited else '#bbbbbb'
            right.add_patch(Rectangle((.15+i*1.55,.65),1.3,.6,facecolor=color,edgecolor=edge))
            right.text(.8+i*1.55,.95,f'Stage {i+1}',ha='center',va='center',color='#222222' if visited else '#999999')
            if i<3:right.text(1.58+i*1.55,.95,'→',ha='center',va='center',color='#397eb5' if visits[i+1] else '#bbbbbb')
        right.text(6.7,.95,f'出口 {e+1} 返回',color='#26734d',va='center',fontsize=12)
        right.text(.15,.15,f'预测概率 {probability:.3f}；累计 MACs {macs/1e6:.3f} M；完整骨干成本的 {100*macs/costs["backbone_macs"]:.2f}%',fontsize=10)
    key=config['head']+('_kd' if config['kd'] else '')
    fig.suptitle(f'{NAMES[key]}：验证样本动态路径（平均预算目标 {budget:.0%}）\n蓝色为实际执行，灰色为跳过；按出口与正误分层展示，不代表总体准确率',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94])
    evidence=dict(scope='validation selection examples; not aggregate accuracy or official test',
                  selection_rule='Smallest original index within every nonempty (exit, correctness) stratum',
                  strata=strata,rows=rows,weights_sha256=weights,policy_sha256=file_sha256(run/'policies.json'),
                  prediction_sha256=meta['prediction_sha256'],split_sha256=split['sha256'],
                  selected_policy=selected,execution_device='cpu',timing_measured=False)
    save(fig,f'{run.name}_dynamic_examples',evidence,output)
    print(json.dumps(dict(examples=len(rows),strata=strata,all_dynamic_paths_verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--budget',type=float,default=.75)
    p.add_argument('--consensus',action='store_true');p.add_argument('--data',default='data')
    p.add_argument('--output',default='figures/application');a=p.parse_args()
    build(a.run,a.budget,a.consensus,a.data,a.output)
