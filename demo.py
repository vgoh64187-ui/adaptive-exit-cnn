"""A local budget-controlled classifier, using real conditional network execution."""
import argparse
import json
import time
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from src.data import MEAN, STD
from src.models import profile_macs
from inference import load_run, load_policy, backbone_only, synchronise


def classify(model, tensor, policy, classes, device, full_depth=False):
    # Input is already resized and normalised; this measurement excludes preprocessing.
    x = tensor.unsqueeze(0).to(device)
    fn = (lambda: backbone_only(model, x)) if full_depth else (lambda: model.predict_dynamic(x, policy))
    with torch.no_grad():
        synchronise(device)
        start = time.perf_counter()
        logits, ids = fn()
        synchronise(device)
        latency = (time.perf_counter() - start) * 1000
    exit_id = ids.item()
    temperature = 1. if full_depth else policy.temperatures[exit_id]
    probs = (logits.float() / temperature).softmax(1)[0]
    top = probs.topk(5)
    return dict(exit=exit_id + 1,
                top5=[dict(class_id=int(i), label=classes[int(i)], probability=float(p))
                      for p, i in zip(top.values.cpu(), top.indices.cpu())],
                observed_model_latency_ms=latency)


def main():
    p = argparse.ArgumentParser(description='计算预算可调的 CIFAR-100 图像分类演示')
    p.add_argument('run')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--image', help='本地图像；按 CIFAR-100 的 100 类输出')
    source.add_argument('--sample-index', type=int, help='固定验证集中的序号，非测试集')
    p.add_argument('--budget', type=float, default=.75)
    p.add_argument('--consensus', action='store_true')
    p.add_argument('--full-depth', action='store_true')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--data', default='data')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    torch.set_num_threads(6)
    model, config, weight_sha = load_run(args.run, args.device)
    if args.full_depth:
        policy, selected = None, None
    else:
        policy, selected = load_policy(args.run, weight_sha, args.budget, args.consensus)
    dataset = datasets.CIFAR100(args.data, train=True, download=False)
    truth = None
    if args.image:
        with Image.open(args.image) as f:
            im = ImageOps.exif_transpose(f).convert('RGB')
        source_info = dict(type='local_image', path=str(Path(args.image).resolve()))
    else:
        split = json.loads((Path(args.data) / 'split_cifar100.json').read_text())
        candidates = sorted(split['selection'])
        sample_index = 0 if args.sample_index is None else args.sample_index
        if not 0 <= sample_index < len(candidates):
            p.error(f'sample-index must be between 0 and {len(candidates)-1}')
        original_index = candidates[sample_index]
        im, label = dataset[original_index]
        truth = dict(class_id=label, label=dataset.classes[label])
        source_info = dict(type='validation_selection', index=original_index,
                           split_sha256=split['sha256'])
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])
    result = classify(model, tf(im), policy, dataset.classes, args.device, args.full_depth)
    costs = profile_macs(model)
    macs = costs['backbone_macs'] if args.full_depth else costs['exit_macs'][result['exit']-1]
    result.update(source=source_info, ground_truth=truth, weights_sha256=weight_sha,
                  model=config['model'], cumulative_macs=macs,
                  cost_ratio=macs / costs['backbone_macs'], selected_policy=selected,
                  scope='CIFAR-100 100-class classifier; budget constrains validation mean, not each sample',
                  timing_note='Single observation including first-call overhead; not benchmark evidence',
                  smoke_only=config['smoke_only'])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        target=Path(args.output)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
