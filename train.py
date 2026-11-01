"""Train from scratch, select checkpoints on validation, never evaluate test here."""
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from src.data import loaders
from src.models import ResNet18, profile_macs


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_hash():
    h = hashlib.sha256()
    for p in sorted([Path(__file__), *Path('src').glob('*.py')]):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    correct, total = None, 0
    loss_sum = 0.
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(x)
        if correct is None:
            correct = [0] * len(outputs)
        for i, logits in enumerate(outputs):
            correct[i] += logits.argmax(1).eq(y).sum().item()
        loss_sum += F.cross_entropy(outputs[-1].float(), y, reduction='sum').item()
        total += len(y)
    return dict(accuracy=[c / total for c in correct], final_loss=loss_sum / total, n=total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--head', choices=['baseline', 'gap', 'local_global', 'mlp'], default='baseline')
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--kd', type=float, default=0.)
    p.add_argument('--temperature', type=float, default=3.)
    p.add_argument('--limit', type=int, default=0, help='SMOKE ONLY: subset size')
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.kd < 0 or args.temperature <= 0:
        p.error('Invalid training settings')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and not args.resume:
        raise RuntimeError('Run already exists; choose a new output or --resume')
    seed_all(args.seed)
    torch.set_num_threads(6)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    amp = device.type == 'cuda'
    train_loader, val_loader, split = loaders(args.data, args.batch_size, args.workers, args.seed, args.limit)
    model = ResNet18(head=args.head, width=args.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    config = vars(args) | dict(split_sha256=split['sha256'], source_sha256=source_hash(),
                               model=model.config, smoke_only=bool(args.limit))
    start, best = 0, -1.
    if args.resume:
        saved_cfg = json.loads((out / 'config.json').read_text())
        for key in ['head','epochs','batch_size','seed','width','lr','weight_decay','kd','temperature','limit','split_sha256','source_sha256']:
            if config[key] != saved_cfg[key]:
                raise RuntimeError(f'Resume mismatch: {key}')
        checkpoint = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch'] + 1, checkpoint['best']
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'].cpu())
        if amp:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['rng']['cuda']])
        train_loader.generator.set_state(checkpoint['rng']['loader'].cpu())
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (out / 'profile.json').write_text(json.dumps(profile_macs(model), indent=2))
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           torchvision=torchvision.__version__, numpy=np.__version__,
                           device=str(device), gpu=torch.cuda.get_device_name() if amp else None,
                           cuda=torch.version.cuda, platform=platform.platform(),
                           resume_note='Epoch-boundary recovery; worker augmentation RNG may differ on resumed runs.')
        (out / 'environment.json').write_text(json.dumps(environment, indent=2))
        try:
            (out / 'pip-freeze.txt').write_text(subprocess.check_output(
                [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
        except subprocess.CalledProcessError:
            pass
    print(json.dumps(dict(event='start', config=config, device=str(device))), flush=True)
    for epoch in range(start, args.epochs):
        begun = time.perf_counter()
        model.train()
        # Warmup then cosine. Same schedule across methods, step count preserved.
        warmup = min(5, max(1, args.epochs // 10))
        lr = args.lr * (epoch + 1) / warmup if epoch < warmup else args.lr * .5 * (
            1 + math.cos(math.pi * (epoch - warmup) / max(1, args.epochs - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss_sum, count = 0., 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(x)
                loss = sum(F.cross_entropy(z, y) for z in outputs) / len(outputs)
                if args.kd and len(outputs) > 1:
                    temperature = args.temperature
                    teacher = (outputs[-1].detach().float() / temperature).softmax(1)
                    kd = sum(F.kl_div((z.float() / temperature).log_softmax(1), teacher,
                                     reduction='batchmean') * temperature ** 2 for z in outputs[:-1]) / (len(outputs) - 1)
                    loss = loss + args.kd * kd
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            count += len(y)
        val = validate(model, val_loader, device, amp)
        # Predefined selection score: mean exit accuracy for multi-exit, final accuracy for baseline.
        score = sum(val['accuracy']) / len(val['accuracy'])
        improved = score > best
        best = max(best, score)
        row = dict(epoch=epoch + 1, lr=lr, train_loss=loss_sum / count,
                   validation=val, selection_score=score, seconds=time.perf_counter() - begun,
                   peak_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if amp else 0.)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch, best=best, config=config,
                          rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                   torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if amp else [],
                                   loader=train_loader.generator.get_state()))
        torch.save(checkpoint, out / 'last.tmp')
        (out / 'last.tmp').replace(out / 'last.pt')
        if improved:
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            validation=val), out / 'best.pt')
        with (out / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    (out / 'completed.json').write_text(json.dumps(dict(epochs=args.epochs, best_validation_score=best)))


if __name__ == '__main__':
    main()
