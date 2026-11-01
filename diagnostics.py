"""Probability reliability and exit-risk diagnostics from verified predictions.

These diagnostics describe fixed predictions. They never fit parameters to test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.models import ExitPolicy, route_cached
from inference import file_sha256


def reliability(logits,labels,temperature=1.,bins=15):
    scaled=logits.float()/temperature
    probs=scaled.softmax(1)
    confidence,pred=probs.max(1)
    correct=pred.eq(labels)
    bin_id=torch.clamp((confidence*bins).long(),max=bins-1)
    table=[]
    ece=0.
    for i in range(bins):
        mask=bin_id==i
        n=int(mask.sum())
        acc=float(correct[mask].float().mean()) if n else None
        conf=float(confidence[mask].mean()) if n else None
        if n:ece+=n/len(labels)*abs(acc-conf)
        table.append(dict(lower=i/bins,upper=(i+1)/bins,n=n,accuracy=acc,confidence=conf))
    onehot=F.one_hot(labels,num_classes=logits.shape[1]).float()
    order=torch.argsort(confidence,descending=True,stable=True)
    errors=(~correct[order]).float().cumsum(0)
    sample_counts=np.unique(np.linspace(1,len(labels),100).astype(int))
    curve=[dict(coverage=int(k)/len(labels),risk=float(errors[k-1]/int(k)),n=int(k)) for k in sample_counts]
    return dict(n=len(labels),temperature=temperature,ece=ece,
                nll=float(F.cross_entropy(scaled,labels)),
                multiclass_brier=float(((probs-onehot)**2).sum(1).mean()),
                accuracy=float(correct.float().mean()),bins=table,risk_coverage=curve)


def route_diagnostic(outputs,labels,policy,costs):
    selected,ids=route_cached(outputs,policy)
    correct=selected.argmax(1).eq(labels)
    final_correct=outputs[-1].argmax(1).eq(labels)
    early=ids<2
    return dict(accuracy=float(correct.float().mean()),exit_counts=torch.bincount(ids,minlength=3).tolist(),
                mean_cost_ratio=float(torch.tensor(costs['exit_macs'],dtype=torch.float64)[ids].mean()/costs['backbone_macs']),
                harmful_early_fraction=float((early & ~correct & final_correct).float().mean()),
                beneficial_early_fraction=float((early & correct & ~final_correct).float().mean()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--development',action='store_true')
    p.add_argument('--data',default='data')
    args=p.parse_args();torch.set_num_threads(4)
    run=Path(args.run)
    config=json.loads((run/'config.json').read_text())
    if not args.development and (config['smoke_only'] or not (run/'completed.json').exists()):
        raise RuntimeError('Final diagnostics require completed non-smoke runs')
    path=run/f'{args.split}_predictions.npz'
    metadata=json.loads(path.with_suffix('.metadata.json').read_text())
    policies=json.loads((run/'policies.json').read_text())
    weight_sha=file_sha256(run/'best.pt')
    if metadata['prediction_sha256']!=file_sha256(path):
        raise RuntimeError('Prediction cache checksum mismatch')
    if metadata['weights_sha256']!=weight_sha or policies['weights_sha256']!=weight_sha:
        raise RuntimeError('Stale predictions or policy')
    if args.split=='test' and not (run/'test_metrics.json').exists():
        raise RuntimeError('Frozen final evaluation required before test diagnostics')
    data=np.load(path)
    labels=torch.from_numpy(data['labels'])
    outputs=[torch.from_numpy(data[f'logits_{i}']) for i in range(len(policies['temperatures']))]
    if args.split=='validation':
        # Reliability is assessed outside the temperature-fitting calibration subset.
        split=json.loads((Path(args.data)/'split_cifar100.json').read_text())
        selected=set(split['selection'])
        mask=torch.tensor([int(i) in selected for i in data['indices']])
        outputs=[z[mask] for z in outputs];labels=labels[mask]
    stats=[dict(exit=i+1,raw=reliability(z,labels),calibrated=reliability(z,labels,policies['temperatures'][i]))
           for i,z in enumerate(outputs)]
    controls={}
    if len(outputs)==3:
        costs=json.loads((run/'profile.json').read_text())
        # Predeclared non-optimised control: first possible agreement is at exit 2.
        controls['pure_adjacent_agreement']=route_diagnostic(outputs,labels,
            ExitPolicy((1.1,0.),tuple(policies['temperatures']),True),costs)
        controls['fixed_second_exit_with_sequential_heads']=route_diagnostic(outputs,labels,
            ExitPolicy((1.1,0.),tuple(policies['temperatures']),False),costs)
    result=dict(weights_sha256=weight_sha,prediction_sha256=metadata['prediction_sha256'],
                split=args.split,scope='selection subset' if args.split=='validation' else 'official test',
                development=args.development,exits=stats,fixed_strategy_controls=controls)
    target=run/f'{args.split}_diagnostics.json'
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(path=str(target),n=len(labels),raw_ece=[s['raw']['ece'] for s in stats],
                          calibrated_ece=[s['calibrated']['ece'] for s in stats])))


if __name__=='__main__':main()
