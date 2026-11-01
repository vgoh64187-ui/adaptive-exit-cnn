"""Bind final evaluation to the exact pre-test weights, policies and evaluator."""
import hashlib
import json
from pathlib import Path

EVALUATION_FILES = ('evaluate.py', 'protocol.py', 'audit_results.py', 'summarise.py')
TRAINING_KEYS = ('epochs', 'batch_size', 'workers', 'width', 'lr', 'weight_decay',
                 'temperature', 'limit')
FROZEN_ARTIFACTS = ('best.pt', 'config.json', 'policies.json', 'completed.json',
                    'history.jsonl', 'validation_predictions.metadata.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate_frozen_run(run, config, lock):
    run = Path(run)
    if lock.get('status') != 'frozen' or run.name not in lock.get('runs', {}):
        raise RuntimeError('Frozen protocol must explicitly include this run')
    for key in ['source_sha256', 'split_sha256']:
        frozen_key = 'training_source_sha256' if key == 'source_sha256' else key
        if lock.get(frozen_key) != config[key]:
            raise RuntimeError(f'Frozen protocol {key} mismatch')
    expected = lock['runs'][run.name] | lock['training']
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f'Frozen protocol config mismatch: {key}')
    hashes = lock.get('artifacts', {}).get(run.name, {})
    for filename in FROZEN_ARTIFACTS:
        if hashes.get(filename) != digest(run / filename):
            raise RuntimeError(f'Frozen run artifact changed: {filename}')
    for filename in EVALUATION_FILES:
        if lock.get('evaluation_sources', {}).get(filename) != digest(filename):
            raise RuntimeError(f'Frozen evaluation source changed: {filename}')
    if config['smoke_only'] or config['limit']:
        raise RuntimeError('Smoke or truncated training cannot enter final testing')
    finished = json.loads((run / 'completed.json').read_text())
    if finished['epochs'] != config['epochs']:
        raise RuntimeError('Training epoch count is incomplete')

