"""Freeze the complete comparison manifest before the first official test.

This command records actual existing artifacts; it cannot be used as an empty
permission flag. It does not perform test evaluation or change experiment data.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from protocol import digest, EVALUATION_FILES, TRAINING_KEYS, FROZEN_ARTIFACTS


def build_lock(manifest, runs_root=Path('runs')):
    from train import source_hash
    manifest = Path(manifest); runs_root = Path(runs_root)
    spec = json.loads(manifest.read_text())
    if list(runs_root.glob('*/test_predictions.npz')) or list(runs_root.glob('*/test_metrics.json')):
        raise RuntimeError('Official test artifacts already exist; cannot establish a new pre-test freeze')
    lock = dict(status='frozen', frozen_at=datetime.now(timezone.utc).isoformat(),
                manifest_sha256=digest(manifest), runs={}, artifacts={},
                evaluation_sources={p: digest(p) for p in EVALUATION_FILES})
    seeds = defaultdict(list)
    current_source = source_hash()
    for job in spec['runs']:
        name = job['name']
        if Path(name).name != name or name in lock['runs']:
            raise RuntimeError('Invalid or repeated run name')
        run = runs_root / name
        config = json.loads((run/'config.json').read_text())
        finished = json.loads((run/'completed.json').read_text())
        policies = json.loads((run/'policies.json').read_text())
        metadata = json.loads((run/'validation_predictions.metadata.json').read_text())
        expected = {k: job[k] for k in ['head','seed','kd']}
        if any(config[k] != v for k,v in expected.items()):
            raise RuntimeError(f'{name}: manifest/config mismatch')
        if config['smoke_only'] or config['limit'] or policies['smoke_only']:
            raise RuntimeError(f'{name}: development run')
        if finished['epochs'] != config['epochs'] or config['epochs'] != spec['epochs']:
            raise RuntimeError(f'{name}: incomplete training or epoch mismatch')
        if any(config[k] != spec[k] for k in ['batch_size','workers']):
            raise RuntimeError(f'{name}: queue/config training mismatch')
        if config['head']=='baseline' and finished['best_validation_score'] < spec['minimum_baseline_validation']:
            raise RuntimeError(f'{name}: baseline requires validation review before freezing')
        if config['source_sha256'] != current_source:
            raise RuntimeError(f'{name}: training source differs from current source')
        history = [json.loads(s) for s in (run/'history.jsonl').read_text().splitlines() if s.strip()]
        if [r['epoch'] for r in history] != list(range(1, config['epochs']+1)):
            raise RuntimeError(f'{name}: missing or repeated epoch history')
        weights = digest(run/'best.pt')
        if policies['weights_sha256'] != weights or metadata['weights_sha256'] != weights:
            raise RuntimeError(f'{name}: stale validation artifacts')
        if metadata['prediction_sha256'] != digest(run/'validation_predictions.npz'):
            raise RuntimeError(f'{name}: corrupt validation prediction cache')
        if metadata['n'] != 5000 or metadata['split'] != 'validation' or metadata['smoke_only']:
            raise RuntimeError(f'{name}: invalid validation population')
        if policies['split_sha256'] != config['split_sha256'] or metadata['split_sha256'] != config['split_sha256']:
            raise RuntimeError(f'{name}: validation split mismatch')
        training = {k: config[k] for k in TRAINING_KEYS}
        if not lock['runs']:
            lock.update(training=training, training_source_sha256=current_source,
                        split_sha256=config['split_sha256'])
        elif training != lock['training'] or config['split_sha256'] != lock['split_sha256']:
            raise RuntimeError(f'{name}: comparison settings differ')
        criteria = {r['criterion'] for r in policies['policies']}
        required = {'full_depth'} if config['head']=='baseline' else {'full_depth','fixed_exit_1','fixed_exit_2','budget','accuracy_constraint'}
        if not required <= criteria:
            raise RuntimeError(f'{name}: missing predeclared comparison policies')
        lock['runs'][name] = expected
        lock['artifacts'][name] = {p: digest(run/p) for p in FROZEN_ARTIFACTS}
        seeds[(config['head'],config['kd'])].append(config['seed'])
    if not lock['runs']:
        raise RuntimeError('Empty comparison manifest')
    for group, values in seeds.items():
        if sorted(values) != [17,42,73]:
            raise RuntimeError(f'{group}: all three predefined independent seeds are required')
    return lock


def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',nargs='?',default='configs/main_queue.json')
    p.add_argument('--output',default='docs/protocol_lock.json')
    args=p.parse_args()
    if Path(args.output).exists():
        raise RuntimeError('Protocol lock already exists; refusing to overwrite')
    result=build_lock(args.manifest)
    with Path(args.output).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(path=args.output,runs=len(result['runs']),sha256=digest(args.output))))


if __name__=='__main__':main()
