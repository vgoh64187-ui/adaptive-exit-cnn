"""Cache predictions and select policies using validation only.

Usage: evaluate.py RUN --split validation
       evaluate.py RUN --split test --protocol-lock docs/protocol_lock.json
No timing is measured by this cached evaluation.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.data import MEAN, STD
from src.models import ResNet18, ExitPolicy, profile_macs, route_cached
from protocol import validate_frozen_run


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def temperature_fit(logits, labels):
    # Positive, bounded post-hoc scale, fitted to calibration NLL only.
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=.1, max_iter=50)
    def closure():
        optimizer.zero_grad()
        temperature = log_t.clamp(-3, 3).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return log_t.detach().clamp(-3, 3).exp().item()


def metrics(outputs, labels, policy, costs):
    chosen, ids = route_cached(outputs, policy)
    correct = chosen.argmax(1).eq(labels)
    final_correct = outputs[-1].argmax(1).eq(labels)
    early = ids < 2
    macs = torch.tensor(costs['exit_macs'], dtype=torch.float64)[ids]
    count = torch.bincount(ids, minlength=3)
    return dict(accuracy=correct.float().mean().item(),
                mean_macs=macs.mean().item(),
                cost_ratio=macs.mean().item() / costs['backbone_macs'],
                exit_counts=count.tolist(), n=len(labels),
                early_exit_error_rate=(~correct[early]).float().mean().item() if early.any() else None,
                harmful_early_exit_fraction=(early & ~correct & final_correct).float().mean().item(),
                beneficial_early_exit_fraction=(early & correct & ~final_correct).float().mean().item())


def choose_policies(outputs, labels, costs, temperatures):
    if len(outputs) == 1:
        policy = ExitPolicy(temperatures=(1., 1., temperatures[0]))
        return [dict(policy=asdict(policy), selection=metrics(outputs, labels, policy, costs),
                     criterion='full_depth')]
    grids = []
    for i in range(2):
        confidence = (outputs[i] / temperatures[i]).softmax(1).max(1).values
        grids.append(sorted(set([0., 1.1] + torch.quantile(confidence, torch.linspace(0, 1, 21)).tolist())))
    selected = []
    # Fixed-depth, shared-training controls. Earlier heads are still executed,
    # matching the sequential dynamic implementation and cumulative path MACs.
    for criterion, thresholds in [('fixed_exit_1', (0., 1.1)),
                                  ('fixed_exit_2', (1.1, 0.)),
                                  ('full_depth', (1.1, 1.1))]:
        policy = ExitPolicy(thresholds, tuple(temperatures))
        selected.append(dict(criterion=criterion, policy=asdict(policy),
                             selection=metrics(outputs, labels, policy, costs)))
    for consensus in [False, True]:
        candidates = []
        for a in grids[0]:
            for b in grids[1]:
                policy = ExitPolicy((a, b), tuple(temperatures), consensus=consensus)
                result = metrics(outputs, labels, policy, costs)
                candidates.append(dict(policy=asdict(policy), selection=result))
        for budget in [.55, .65, .75, .85, .95, 1.01]:
            feasible = [r for r in candidates if r['selection']['cost_ratio'] <= budget]
            if feasible:
                best = max(feasible, key=lambda r: (r['selection']['accuracy'], -r['selection']['cost_ratio']))
                selected.append(best | dict(criterion='budget', budget=budget))
            else:
                selected.append(dict(criterion='budget', budget=budget, consensus=consensus,
                                     status='infeasible_on_validation'))
        final_accuracy = outputs[-1].argmax(1).eq(labels).float().mean().item()
        for drop in [.005, .01, .02]:
            feasible = [r for r in candidates if r['selection']['accuracy'] >= final_accuracy - drop]
            best = min(feasible, key=lambda r: (r['selection']['cost_ratio'], -r['selection']['accuracy']))
            selected.append(best | dict(criterion='accuracy_constraint', allowed_drop=drop))
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--protocol-lock', default=None)
    p.add_argument('--data', default='data')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(6)
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    if args.split == 'test':
        if config['smoke_only']:
            raise RuntimeError('Smoke runs cannot enter final testing')
        if not args.protocol_lock or not Path(args.protocol_lock).exists():
            raise RuntimeError('Freeze the protocol before test evaluation')
        if not (run / 'policies.json').exists():
            raise RuntimeError('Validation-selected policies required before testing')
        if not (run / 'completed.json').exists():
            raise RuntimeError('Training must be complete before final testing')
        lock = json.loads(Path(args.protocol_lock).read_text(encoding='utf-8'))
        validate_frozen_run(run, config, lock)
    split = json.loads((Path(args.data) / 'split_cifar100.json').read_text())
    if split['sha256'] != config['split_sha256']:
        raise RuntimeError('Data split hash mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_hash = sha(run / 'best.pt')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    if sha(run / 'best.pt') != weights_hash:
        raise RuntimeError('Checkpoint changed during loading')
    model = ResNet18(**checkpoint['config']['model']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    costs = profile_macs(model)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR100(args.data, train=args.split != 'test', transform=transform, download=False)
    indices = sorted(split['calibration'] + split['selection']) if args.split == 'validation' else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == 'cuda')
    buffers, labels = None, []
    with torch.no_grad():
        for x, y in loader:
            # FP32 cache and FP32 dynamic benchmark use the same precision.
            outputs = model(x.to(device, non_blocking=True))
            if buffers is None:
                buffers = [[] for _ in outputs]
            for buffer, output in zip(buffers, outputs):
                buffer.append(output.cpu())
            labels.append(y)
    outputs = [torch.cat(b) for b in buffers]
    labels = torch.cat(labels)
    np.savez_compressed(run / f'{args.split}_predictions.npz',
                        **{f'logits_{i}': z.numpy() for i, z in enumerate(outputs)},
                        labels=labels.numpy(), indices=np.array(indices))
    cache_metadata = dict(weights_sha256=weights_hash, split_sha256=split['sha256'],
                          prediction_sha256=sha(run / f'{args.split}_predictions.npz'),
                          split=args.split, n=len(labels), precision='FP32',
                          smoke_only=config['smoke_only'])
    (run / f'{args.split}_predictions.metadata.json').write_text(json.dumps(cache_metadata, indent=2))
    if args.split == 'validation':
        calibration_set = set(split['calibration'])
        calibration = torch.tensor([i in calibration_set for i in indices])
        temperatures = [temperature_fit(z[calibration], labels[calibration]) for z in outputs]
        policies = choose_policies([z[~calibration] for z in outputs], labels[~calibration], costs, temperatures)
        payload = dict(weights_sha256=weights_hash, split_sha256=split['sha256'],
                       temperatures=temperatures, policies=policies, smoke_only=config['smoke_only'])
        (run / 'policies.json').write_text(json.dumps(payload, indent=2))
    else:
        payload = json.loads((run / 'policies.json').read_text())
        if payload['weights_sha256'] != weights_hash:
            raise RuntimeError('Policies belong to another checkpoint')
        rows = []
        for record in payload['policies']:
            if 'policy' in record:
                result = metrics(outputs, labels, ExitPolicy(**record['policy']), costs)
                rows.append(record | dict(test=result))
        result = dict(weights_sha256=weights_hash, split_sha256=split['sha256'],
                      protocol_sha256=sha(args.protocol_lock),
                      policy_sha256=sha(run / 'policies.json'),
                      exit_accuracy=[z.argmax(1).eq(labels).float().mean().item() for z in outputs],
                      profile=costs, policies=rows, n=len(labels), seed=config['seed'])
        (run / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(run=str(run), split=args.split, samples=len(labels),
                          exit_accuracy=[z.argmax(1).eq(labels).float().mean().item() for z in outputs])))


if __name__ == '__main__':
    main()
