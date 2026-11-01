"""Pair methods by training seed at the same validation target budget.

The actual TEST compute difference is retained; a shared validation budget is
never silently treated as exact matched test compute. No significance claims
are inferred from only three repeats.
"""
import argparse
import json
from pathlib import Path
import statistics
from inference import file_sha256
from summarise import collect

CONTRASTS = [
    ('local_features_vs_gap', ('gap',0.,False), ('local_global',0.,False)),
    ('fusion_vs_matched_capacity', ('mlp',0.,False), ('local_global',0.,False)),
    ('distillation', ('local_global',0.,False), ('local_global',.5,False)),
    ('consensus_without_distillation', ('local_global',0.,False), ('local_global',0.,True)),
    ('consensus_with_distillation', ('local_global',.5,False), ('local_global',.5,True)),
]
METRICS = ('accuracy','cost_ratio','harmful_early_exit_fraction','beneficial_early_exit_fraction')


def compare(rows):
    lookup={}
    for r in rows:
        if r['criterion']!='budget':continue
        key=(r['head'],r['kd'],r['consensus'],r['budget'],r['seed'])
        if key in lookup:raise RuntimeError('Duplicate seed in paired comparison')
        lookup[key]=r
    result=[]
    for name,a,b in CONTRASTS:
        for budget in [.55,.65,.75,.85,.95,1.01]:
            pairs=[]
            for seed in [17,42,73]:
                left,right=lookup.get((*a,budget,seed)),lookup.get((*b,budget,seed))
                if left is None or right is None:
                    raise RuntimeError(f'Missing paired evidence for {name}, budget {budget}, seed {seed}')
                delta={m+'_delta_pp':100*(right[m]-left[m]) for m in METRICS}
                pairs.append(dict(seed=seed,reference_run=left['run'],comparison_run=right['run'],
                                  reference_accuracy=left['accuracy'],comparison_accuracy=right['accuracy'],
                                  reference_actual_cost_ratio=left['cost_ratio'],
                                  comparison_actual_cost_ratio=right['cost_ratio'],**delta))
            summary={}
            for metric in METRICS:
                values=[p[metric+'_delta_pp'] for p in pairs]
                summary[metric+'_delta_pp_mean']=statistics.mean(values)
                summary[metric+'_delta_pp_std']=statistics.stdev(values)
            result.append(dict(contrast=name,reference=a,comparison=b,target_validation_budget=budget,
                               pairs=pairs,summary=summary))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',nargs='?',default='configs/main_queue.json')
    p.add_argument('--output',default='results/final/paired_effects.json');args=p.parse_args()
    rows,provenance=collect(args.manifest)
    payload=dict(manifest_sha256=file_sha256(args.manifest),provenance=provenance,comparisons=compare(rows),
                 interpretation='comparison minus reference, paired by seed; test cost may differ despite common validation target',
                 uncertainty='sample standard deviation of the three paired differences; not a significance test')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(path=str(path),comparisons=len(payload['comparisons']))))


if __name__=='__main__':main()
