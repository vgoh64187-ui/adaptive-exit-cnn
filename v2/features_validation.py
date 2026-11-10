"""Descriptive V1 feature diagnostics on an equal-class validation sample."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.data import MEAN, STD
from src.models import ResNet18

torch.set_num_threads(2)
split=json.loads(Path("data/split_cifar100.json").read_text(encoding="utf-8"))
dataset=datasets.CIFAR100("data",train=True,download=False,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)]))
labels=np.asarray(dataset.targets)
selection=np.asarray(split["selection"])
ids=[int(selection[np.flatnonzero(labels[selection]==c)[0]]) for c in range(100)]
loader=DataLoader(Subset(dataset,ids),batch_size=20,shuffle=False,num_workers=0)
out={"scope":"exploratory description on 100 selection images, one per class; not a causal test",
     "indices_sha256":hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
     "sample_count":len(ids),"runs":[]}
for name in ("lg","lg_kd"):
 for seed in (17,42,73):
    path=Path("runs")/f"{name}_seed{seed}"/"best.pt"
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    model=ResNet18(**ckpt["config"]["model"]).eval()
    model.load_state_dict(ckpt["model"])
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
          entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1).squeeze(1)/np.log(weights.shape[-1])
          cosine=torch.nn.functional.cosine_similarity(g,l,dim=1)
          values=torch.stack((g.norm(dim=1),l.norm(dim=1),cosine,gate.mean(dim=1),entropy),dim=1)
          collected[stage-1].append(values)
    desc=[]
    for values in collected:
      matrix=torch.cat(values).numpy()
      desc.append(dict(zip(("global_norm","local_norm","branch_cosine","gate_mean","normalized_spatial_entropy"),
                           (float(a) for a in matrix.mean(0)))))
    out["runs"].append(dict(model=name,seed=seed,weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),exit_means=desc))
    print(name,seed,flush=True)
Path("results/v2/features_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
