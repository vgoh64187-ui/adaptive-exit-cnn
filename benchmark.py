"""Measure genuine conditional execution, including routing/synchronisation costs.

Final benchmarks require completed training and validation-selected policies.
Use --smoke only for engineering checks, never for reported performance.
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from src.data import MEAN, STD
from src.models import route_cached, profile_macs
from inference import load_run, load_policy, backbone_only, synchronise, file_sha256


@torch.no_grad()
def measure(model, batches, policy, device, repeats, warmup, full_depth=False):
    fn = (lambda x: backbone_only(model,x)) if full_depth else (lambda x: model.predict_dynamic(x,policy))
    for i in range(warmup):
        fn(batches[i % len(batches)])
    synchronise(device)
    visits = [0] * 4
    def count(stage):
        def hook(module, inputs, output):
            visits[stage] += len(inputs[0])
        return hook
    handles = [m.register_forward_hook(count(i)) for i,m in enumerate(model.stages)]
    predicted, exits, disagreements = [], [], 0
    max_logit_error = 0.
    # Untimed correctness and stage-visit audit. Hooks removed before timing.
    for x in batches:
        actual, ids = fn(x)
        expected, expected_ids = (backbone_only(model,x) if full_depth else route_cached(model(x),policy))
        disagreements += int((ids!=expected_ids).sum())
        same = ids == expected_ids
        if same.any():
            max_logit_error = max(max_logit_error,float((actual[same]-expected[same]).abs().max()))
        predicted.extend(actual.argmax(1).cpu().tolist())
        exits.extend(ids.cpu().tolist())
    for h in handles:
        h.remove()
    # Auditing reference forwards visits every stage once; subtract these visits.
    n = sum(len(x) for x in batches)
    visits = [v-n for v in visits]
    expected_visits = [n,n,sum(e>=1 for e in exits),sum(e>=2 for e in exits)]
    if full_depth or not model.exits:
        expected_visits = [n]*4
    if visits != expected_visits:
        raise AssertionError(f'Unexpected stage visits: {visits} vs {expected_visits}')
    elapsed, sizes = [], []
    for _ in range(repeats):
        for x in batches:
            synchronise(device)
            start=time.perf_counter_ns()
            fn(x)
            synchronise(device)
            elapsed.append((time.perf_counter_ns()-start)/1e6)
            sizes.append(len(x))
    return dict(batch_latency_ms=elapsed, batch_sizes=sizes,
                batch_latency_p50_ms=float(np.percentile(elapsed,50)),
                batch_latency_p95_ms=float(np.percentile(elapsed,95)),
                effective_mean_ms_per_image=float(sum(elapsed)/sum(sizes)),
                throughput_images_per_second=float(1000*sum(sizes)/sum(elapsed)),
                predictions=predicted, exits=exits, stage_sample_visits=visits,
                cached_route_disagreements=disagreements, max_same_exit_logit_difference=max_logit_error)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--budget',type=float,default=.75)
    p.add_argument('--consensus',action='store_true')
    p.add_argument('--full-depth',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--data',default='data')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if min(args.batch_size,args.samples,args.repeats,args.threads)<1 or args.warmup<0:
        p.error('Invalid measurement dimensions')
    run=Path(args.run)
    if not args.smoke and not (run/'completed.json').exists():
        raise RuntimeError('Final timing requires a completed run')
    torch.set_num_threads(args.threads)
    model,config,weights_sha=load_run(run,args.device)
    if config['smoke_only'] and not args.smoke:
        raise RuntimeError('Cannot report a smoke model as a final benchmark')
    if args.full_depth:
        policy,selected=None,None
    else:
        policy,selected=load_policy(run,weights_sha,args.budget,args.consensus)
    split=json.loads((Path(args.data)/'split_cifar100.json').read_text())
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=datasets.CIFAR100(args.data,train=True,transform=tf)
    # Fixed random subset of the selection split, chosen without labels or outcomes.
    indices=np.random.default_rng(8128).permutation(sorted(split['selection']))[:args.samples].tolist()
    samples=[ds[i] for i in indices]
    inputs=torch.stack([x for x,y in samples]).to(args.device)
    labels=[y for x,y in samples]
    batches=list(inputs.split(args.batch_size))
    results=measure(model,batches,policy,args.device,args.repeats,args.warmup,args.full_depth)
    costs=profile_macs(model)
    mean_macs=costs['backbone_macs'] if args.full_depth else float(np.mean([costs['exit_macs'][e] for e in results['exits']]))
    results['actual_mean_macs']=mean_macs
    results['actual_cost_ratio']=mean_macs/costs['backbone_macs']
    results['accuracy_on_timing_subset']=float(np.mean(np.array(results['predictions'])==labels))
    payload=dict(settings=vars(args),weights_sha256=weights_sha,split_sha256=split['sha256'],
                 measured_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256={p:file_sha256(p) for p in ['benchmark.py','inference.py','src/models.py']},
                 policy_sha256=file_sha256(run/'policies.json'),
                 sample_indices=indices,labels=labels,selected_policy=selected,
                 policy=asdict(policy) if policy else None,results=results,
                 environment=dict(torch=torch.__version__,platform=platform.platform(),
                                  gpu=torch.cuda.get_device_name() if args.device=='cuda' else None),
                 precision='FP32',smoke_only=args.smoke,
                 scope='validation selection subset; accuracy is not official test accuracy',
                 timing_scope='device-resident normalised inputs; includes routing and CUDA host synchronisation; excludes disk, decoding and host-to-device copy')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in results.items() if k not in ['predictions','exits','batch_latency_ms','batch_sizes']},indent=2))


if __name__=='__main__':
    main()
