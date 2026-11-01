"""Shared inference helpers, independent of the frozen training source."""
import hashlib
import json
from pathlib import Path
import torch
from src.models import ExitPolicy, ResNet18


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load_run(run, device='cpu'):
    run = Path(run)
    weight_path = run / 'best.pt'
    before = file_sha256(weight_path)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if file_sha256(weight_path) != before:
        raise RuntimeError('Checkpoint changed during read; use a finished run')
    model = ResNet18(**checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint['config'], before


def load_policy(run, weight_sha, budget=.75, consensus=False):
    payload = json.loads((Path(run) / 'policies.json').read_text())
    if payload['weights_sha256'] != weight_sha:
        raise RuntimeError('Policy/checkpoint hash mismatch; recalibrate validation policies')
    full = [p for p in payload['policies'] if p.get('criterion') == 'full_depth']
    if full and not any(p.get('criterion') == 'budget' for p in payload['policies']):
        return ExitPolicy(**full[0]['policy']), full[0]
    eligible = [p for p in payload['policies'] if p.get('criterion') == 'budget'
                and abs(p['budget'] - budget) < 1e-8 and 'policy' in p
                and p['policy']['consensus'] == consensus]
    if len(eligible) != 1:
        raise ValueError(f'No feasible validation policy for budget={budget}, consensus={consensus}')
    return ExitPolicy(**eligible[0]['policy']), eligible[0]


@torch.no_grad()
def backbone_only(model, x):
    """Evaluate the final classifier without executing auxiliary heads."""
    x = model.stem(x)
    for stage in model.stages:
        x = stage(x)
    logits = model.final_head(x)
    return logits, torch.full((len(x),), 2, dtype=torch.long, device=x.device)


def synchronise(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
