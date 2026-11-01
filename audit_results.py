"""Reconstruct reported test numbers from bound per-image prediction caches."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from protocol import digest, validate_frozen_run
from src.models import ExitPolicy, ResNet18, profile_macs, route_cached


def require_close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f'{label}: null mismatch')
    elif not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError(f'{label}: reported {actual}, reconstructed {expected}')


def audit_policy(outputs, labels, policy, costs, reported):
    chosen, exits = route_cached(outputs, ExitPolicy(**policy))
    n = len(labels)
    correct = chosen.argmax(1).eq(labels)
    final = outputs[-1].argmax(1).eq(labels)
    early = exits < 2
    counts = torch.bincount(exits, minlength=3).tolist()
    if reported['n'] != n or reported['exit_counts'] != counts:
        raise RuntimeError('Reported population or exit counts disagree with cache')
    mean_macs = sum(count * macs for count, macs in zip(counts, costs['exit_macs'])) / n
    exact = dict(accuracy=int(correct.sum())/n, mean_macs=mean_macs,
                 cost_ratio=mean_macs/costs['backbone_macs'],
                 harmful_early_exit_fraction=int((early & ~correct & final).sum())/n,
                 beneficial_early_exit_fraction=int((early & correct & ~final).sum())/n,
                 early_exit_error_rate=int((early & ~correct).sum())/int(early.sum()) if early.any() else None)
    for key, value in exact.items():
        require_close(reported[key], value, key)
    # Integer outcomes also prove the error-transfer identity without rounding.
    assert int(correct.sum())-int(final.sum()) == int((early & correct & ~final).sum())-int((early & ~correct & final).sum())
    return dict(n=n, correct=int(correct.sum()), exit_counts=counts, **exact)


def audit_run(run, protocol_lock='docs/protocol_lock.json'):
    run = Path(run)
    read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
    config, result, policies = read('config.json'), read('test_metrics.json'), read('policies.json')
    lock = json.loads(Path(protocol_lock).read_text(encoding='utf-8'))
    validate_frozen_run(run, config, lock)
    if result['protocol_sha256'] != digest(protocol_lock):
        raise RuntimeError('Result belongs to a different protocol lock')
    if result['policy_sha256'] != digest(run/'policies.json'):
        raise RuntimeError('Result policy hash mismatch')
    metadata = read('test_predictions.metadata.json')
    for key in ['weights_sha256', 'split_sha256']:
        if metadata[key] != result[key] or policies[key] != result[key]:
            raise RuntimeError(f'Prediction/result mismatch: {key}')
    if result['weights_sha256'] != digest(run/'best.pt'):
        raise RuntimeError('Test result checkpoint mismatch')
    cache = run/'test_predictions.npz'
    if metadata['prediction_sha256'] != digest(cache) or metadata['split'] != 'test' or metadata['n'] != 10000 or metadata['smoke_only']:
        raise RuntimeError('Invalid test cache metadata')
    with np.load(cache, allow_pickle=False) as data:
        if not np.array_equal(data['indices'], np.arange(10000)):
            raise RuntimeError('Test cache must contain all ordered official test indices')
        labels = torch.from_numpy(data['labels'].copy())
        outputs = [torch.from_numpy(data[f'logits_{i}'].copy()) for i in range(1 if config['head']=='baseline' else 3)]
    if labels.shape != (10000,) or not ((labels >= 0) & (labels < 100)).all():
        raise RuntimeError('Invalid test label population')
    if any(z.shape != (10000,100) or not torch.isfinite(z).all() for z in outputs):
        raise RuntimeError('Invalid test logits')
    costs = profile_macs(ResNet18(**config['model']).eval())
    if costs != result['profile']:
        raise RuntimeError('Reported profile differs from current model structure')
    expected_records = [p for p in policies['policies'] if 'policy' in p]
    if len(result['policies']) != len(expected_records):
        raise RuntimeError('Missing or extra reported policies')
    if len(result['exit_accuracy']) != len(outputs):
        raise RuntimeError('Wrong number of exit accuracies')
    for i,z in enumerate(outputs):
        require_close(result['exit_accuracy'][i], int(z.argmax(1).eq(labels).sum())/len(labels), f'exit {i+1}')
    audits=[]
    for record, expected in zip(result['policies'], expected_records):
        if {k:v for k,v in record.items() if k!='test'} != expected:
            raise RuntimeError('Reported policy differs from pre-test validation selection')
        audits.append(audit_policy(outputs, labels, record['policy'], costs, record['test']))
    return dict(run=run.name, policies_checked=len(audits), n=len(labels),
                prediction_sha256=digest(cache), metrics_sha256=digest(run/'test_metrics.json'),
                protocol_sha256=digest(protocol_lock), exact_outcomes=audits)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--protocol-lock',default='docs/protocol_lock.json')
    args=p.parse_args();torch.set_num_threads(6)
    print(json.dumps(audit_run(args.run,args.protocol_lock),indent=2))
