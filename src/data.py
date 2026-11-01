"""Fixed stratified CIFAR splits; no test set access during training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def prepare(root='data', split_seed=20261031):
    root = Path(root)
    ds = datasets.CIFAR100(root, train=True, download=True)
    rng = np.random.default_rng(split_seed)
    targets = np.asarray(ds.targets)
    train, calibration, selection = [], [], []
    for cls in range(100):
        indices = np.flatnonzero(targets == cls)
        rng.shuffle(indices)
        calibration.extend(indices[:25].tolist())
        selection.extend(indices[25:50].tolist())
        train.extend(indices[50:].tolist())
    split = dict(seed=split_seed, train=sorted(train), calibration=sorted(calibration),
                 selection=sorted(selection))
    payload = json.dumps(split, sort_keys=True).encode()
    split['sha256'] = hashlib.sha256(payload).hexdigest()
    path = root / 'split_cifar100.json'
    if path.exists() and json.loads(path.read_text()) != split:
        raise RuntimeError('Existing split differs; refusing to overwrite')
    path.write_text(json.dumps(split), encoding='utf-8')
    return split


def loaders(root='data', batch_size=128, workers=2, seed=42, limit=0):
    import torch
    split = prepare(root)
    normalise = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalise])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalise])
    train_ds = datasets.CIFAR100(root, train=True, transform=train_tf)
    val_ds = datasets.CIFAR100(root, train=True, transform=eval_tf)
    train_ids = split['train']
    val_ids = sorted(split['calibration'] + split['selection'])
    if limit:
        rng = np.random.default_rng(seed)
        train_ids = rng.choice(train_ids, min(limit, len(train_ids)), replace=False).tolist()
        val_ids = rng.choice(val_ids, min(limit, len(val_ids)), replace=False).tolist()
    gen = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0)
    return (DataLoader(Subset(train_ds, train_ids), shuffle=True, generator=gen, **common),
            DataLoader(Subset(val_ds, val_ids), shuffle=False, **common), split)


if __name__ == '__main__':
    s = prepare()
    print({k: len(v) if isinstance(v, list) else v for k, v in s.items()})
