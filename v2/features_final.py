"""Describe branch scales on a fixed class-balanced validation sample."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.data import MEAN, STD
from src.models import ResNet18 as OriginalResNet18
from v2.src.models import ResNet18 as V2ResNet18
from v2.evaluate_v2 import sha

torch.set_num_threads(2)
root=Path.cwd()
selection=json.loads((root/'runs_v2/screening_selection.json').read_text(encoding='utf-8'))
selected=selection['selected_kd']
if selected not in ('residual_kd_seed42','residual_delayed_kd_seed42'):
    raise RuntimeError('Unexpected selected model')
prefix='residual_kd' if selected=='residual_kd_seed42' else 'residual_delayed_kd'
lock=json.loads((root/'runs_v2/v2_screening_lock.json').read_text(encoding='utf-8'))
split=json.loads((root/'data/split_cifar100.json').read_text(encoding='utf-8'))
dataset=datasets.CIFAR100(root/'data',train=True,download=False,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)]))
labels=np.asarray(dataset.targets)
selection_ids=np.asarray(split['selection'])
indices=[int(selection_ids[np.flatnonzero(labels[selection_ids]==cls)[0]]) for cls in range(100)]
loader=DataLoader(Subset(dataset,indices),batch_size=20,shuffle=False,num_workers=0)
groups=[('V1 local-global',root/'runs','lg',OriginalResNet18),
        ('V1 local-global + KD',root/'runs','lg_kd',OriginalResNet18),
        ('V2 residual',root/'runs_v2','residual',V2ResNet18),
        ('V2 residual + selected KD',root/'runs_v2',prefix,V2ResNet18)]
out=dict(scope='exploratory description of 100 selection images, one per class; no causal inference',
         n=len(indices),split_sha256=split['sha256'],
         indices_sha256=hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(),
         selection_sha256=sha(root/'runs_v2/screening_selection.json'),
         protocol_sha256=sha(root/'runs_v2/v2_screening_lock.json'),runs=[])
for label,base,prefix,cls in groups:
    for seed in (17,42,73):
        run=base/f'{prefix}_seed{seed}'
        weight=run/'best.pt'
        weight_sha=sha(weight)
        if base.name=='runs_v2':
            if lock['runs'][run.name]['best.pt']!=weight_sha:
                raise RuntimeError('V2 checkpoint differs from freeze')
        else:
            previous=json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
            if previous['weights_sha256']!=weight_sha:
                raise RuntimeError('V1 checkpoint differs from original metrics')
        checkpoint=torch.load(weight,map_location='cpu',weights_only=False)
        model=cls(**checkpoint['config']['model']).eval()
        model.load_state_dict(checkpoint['model'])
        collected=[[],[]]
        with torch.no_grad():
            for images,_ in loader:
                x=model.stem(images)
                for stage in range(3):
                    x=model.stages[stage](x)
                    if stage not in (1,2):
                        continue
                    head=model.exits[stage-1]
                    g=x.mean((2,3))
                    local=head.local(x)
                    weights=head.pool.score(local).flatten(2).softmax(-1)
                    l=(local.flatten(2)*weights).sum(-1)
                    gate=head.gate(torch.cat((g,l),1)).sigmoid()
                    gamma=head.gamma if hasattr(head,'gamma') else 1.
                    correction=gamma*gate*(l-g)
                    fused=g+correction
                    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1).squeeze(1)/np.log(weights.shape[-1])
                    values=torch.stack((g.norm(dim=1),l.norm(dim=1),correction.norm(dim=1),
                                        fused.norm(dim=1),gate.mean(dim=1),entropy),dim=1)
                    collected[stage-1].append(values)
        heads=[]
        for i,values in enumerate(collected):
            matrix=torch.cat(values).numpy()
            means=dict(zip(('global_norm','local_norm','correction_norm','fused_norm',
                            'gate_mean','normalized_spatial_entropy'),
                           (float(v) for v in matrix.mean(0))))
            means['correction_over_global']=means['correction_norm']/max(means['global_norm'],1e-12)
            means['local_over_global']=means['local_norm']/max(means['global_norm'],1e-12)
            means['gamma']=float(model.exits[i].gamma.detach()) if hasattr(model.exits[i],'gamma') else 1.
            heads.append(means)
        out['runs'].append(dict(group=label,seed=seed,weights_sha256=weight_sha,exits=heads))
        print(label,seed,flush=True)
target=root/'results/v2/final_feature_analysis.json'
if target.exists():
    raise RuntimeError('Output exists; refusing to overwrite')
with target.open('x',encoding='utf-8') as f:
    json.dump(out,f,indent=2)
print(target)
