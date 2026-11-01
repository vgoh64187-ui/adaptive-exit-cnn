"""Visualize actual local-global pooling weights and channel gates on fixed images."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets,transforms
from make_figures import plt,save
from inference import load_run,file_sha256
from src.data import MEAN,STD


@torch.no_grad()
def build(run,data='data',output='figures/validation'):
    torch.set_num_threads(2);run=Path(run);data=Path(data)
    model,config,weights=load_run(run,'cpu')
    if config['head']!='local_global' or config['smoke_only'] or not (run/'completed.json').exists():
        raise RuntimeError('Completed local-global model required')
    split=json.loads((data/'split_cifar100.json').read_text())
    if split['sha256']!=config['split_sha256']:raise RuntimeError('Split mismatch')
    ds=datasets.CIFAR100(data,train=True,download=False)
    indices=sorted(split['selection'])[:4]
    samples=[ds[i] for i in indices]
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    x=torch.stack([tf(im) for im,_ in samples]);captured={};handles=[]
    for i,head in enumerate(model.exits):
        def capture(name):
            def hook(module,inputs,result):captured[name]=result.detach().cpu()
            return hook
        handles.append(head.pool.score.register_forward_hook(capture(f'score{i}')))
        handles.append(head.gate.register_forward_hook(capture(f'gate{i}')))
    model(x)
    for h in handles:h.remove()
    maps=[];gates=[]
    for i in range(2):
        score=captured[f'score{i}'];b,_,h,w=score.shape
        weights_map=score.flatten(2).softmax(-1).reshape(b,h,w)
        if not torch.allclose(weights_map.sum((1,2)),torch.ones(b),atol=1e-6):raise RuntimeError('Pool weights are not normalised')
        maps.append(weights_map.numpy());gates.append(captured[f'gate{i}'].sigmoid().numpy())
    # Relative weight one equals uniform pooling, enabling consistent colour scales.
    relative=[w*w.shape[-1]*w.shape[-2] for w in maps];vmax=max(float(a.max()) for a in relative)
    fig,axes=plt.subplots(4,5,figsize=(11,7.2),layout='constrained',
                          gridspec_kw={'width_ratios':[1,1,1,1.3,1.3]})
    titles=['输入图像','出口 1 空间权重','出口 2 空间权重','出口 1 门控','出口 2 门控']
    for col,title in enumerate(titles):axes[0,col].set_title(title)
    for row,(im,label) in enumerate(samples):
        axes[row,0].imshow(im,interpolation='nearest');axes[row,0].set_ylabel(f'索引 {indices[row]}\n{ds.classes[label]}')
        for col in [1,2]:
            heat=axes[row,col].imshow(relative[col-1][row],vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
        for col in [3,4]:
            gate=axes[row,col].imshow(gates[col-3][row][None,:],aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            axes[row,col].set_xlabel('通道索引')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(heat,ax=axes[:,1:3].ravel().tolist(),fraction=.02,pad=.015,label='空间权重 / 均匀权重')
    fig.colorbar(gate,ax=axes[:,3:5].ravel().tolist(),fraction=.02,pad=.015,label='Sigmoid 门控值')
    fig.suptitle('固定验证样本的出口内部响应（完整前向，仅作机制诊断）',fontsize=12)
    records=[]
    for i,idx in enumerate(indices):
        records.append(dict(index=idx,label=ds.classes[samples[i][1]],spatial_weights=[maps[j][i].tolist() for j in range(2)],
                            channel_gates=[gates[j][i].tolist() for j in range(2)]))
    save(fig,f'{run.name}_head_responses',dict(scope='Four lowest-index validation selection images; full forward; not a causal explanation or accuracy estimate',
         weights_sha256=weights,split_sha256=split['sha256'],model_source_sha256=file_sha256('src/models.py'),
         spatial_colour='Weight divided by uniform 1/(H*W); common scale across both exits',records=records),output)
    print(json.dumps(dict(samples=indices,weight_shapes=[list(w.shape) for w in maps],gate_shapes=[list(g.shape) for g in gates])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--data',default='data');p.add_argument('--output',default='figures/validation')
    a=p.parse_args();build(a.run,a.data,a.output)
