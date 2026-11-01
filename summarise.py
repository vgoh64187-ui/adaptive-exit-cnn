"""Audited final test aggregation; refuses missing runs, smoke results and stale hashes."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
from inference import file_sha256
from audit_results import audit_run


def collect(manifest):
    spec=json.loads(Path(manifest).read_text())
    names=[r['name'] for r in spec['runs']]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate run names')
    rows=[]
    provenance=[]
    protocol_hashes=set()
    split_hashes=set()
    for name in names:
        run=Path('runs')/name
        required=['config.json','completed.json','test_metrics.json','policies.json','best.pt']
        for filename in required:
            if not (run/filename).exists():
                raise RuntimeError(f'Missing required result: {run/filename}')
        config=json.loads((run/'config.json').read_text())
        job=next(j for j in spec['runs'] if j['name']==name)
        if any(config[key]!=job[key] for key in ['head','kd','seed']) or config['epochs']!=spec['epochs']:
            raise RuntimeError(f'{run}: manifest/config mismatch')
        finished=json.loads((run/'completed.json').read_text())
        result=json.loads((run/'test_metrics.json').read_text())
        audit=audit_run(run)
        policies=json.loads((run/'policies.json').read_text())
        if config['smoke_only'] or policies['smoke_only'] or finished['epochs']!=config['epochs']:
            raise RuntimeError(f'{run}: incomplete or smoke run')
        weights=file_sha256(run/'best.pt')
        if result['weights_sha256']!=weights or policies['weights_sha256']!=weights:
            raise RuntimeError(f'{run}: stale metrics/policy checkpoint')
        if result.get('policy_sha256')!=file_sha256(run/'policies.json'):
            raise RuntimeError(f'{run}: policies changed after test evaluation')
        if result['split_sha256']!=config['split_sha256'] or result['n']!=10000:
            raise RuntimeError(f'{run}: split or test population mismatch')
        if result['seed']!=config['seed']:
            raise RuntimeError(f'{run}: seed mismatch')
        split_hashes.add(result['split_sha256'])
        protocol_hashes.add(result['protocol_sha256'])
        for record in result['policies']:
            test=record['test']
            if sum(test['exit_counts'])!=10000 or test['n']!=10000:
                raise RuntimeError(f'{run}: invalid exit counts')
            base=dict(run=name,head=config['head'],kd=config['kd'],seed=config['seed'],
                      criterion=record['criterion'],budget=record.get('budget'),
                      allowed_drop=record.get('allowed_drop'),
                      consensus=record['policy']['consensus'],parameters=result['profile']['parameters'],
                      exit1_accuracy=result['exit_accuracy'][0] if config['head']!='baseline' else None,
                      exit2_accuracy=result['exit_accuracy'][1] if config['head']!='baseline' else None,
                      final_exit_accuracy=result['exit_accuracy'][-1])
            rows.append(base|{k:test[k] for k in ['accuracy','mean_macs','cost_ratio',
                                                'harmful_early_exit_fraction','beneficial_early_exit_fraction']}|
                        {f'exit{i+1}_fraction':test['exit_counts'][i]/test['n'] for i in range(3)}|
                        dict(test_budget_met=(test['cost_ratio']<=record['budget']) if 'budget' in record else None))
        provenance.append(dict(run=name,metrics_sha256=file_sha256(run/'test_metrics.json'),
                               checkpoint_sha256=weights,config_sha256=file_sha256(run/'config.json'),
                               prediction_sha256=audit['prediction_sha256'],
                               audited_policies=audit['policies_checked']))
    if len(split_hashes)!=1 or len(protocol_hashes)!=1:
        raise RuntimeError('All results must use the same split and frozen protocol')
    return rows,provenance


def aggregate(rows,minimum_seeds=3):
    keys=['head','kd','criterion','budget','allowed_drop','consensus']
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    summary=[]
    for key,group in groups.items():
        seeds=[r['seed'] for r in group]
        if len(set(seeds))!=len(seeds):
            raise RuntimeError(f'Duplicate seed within comparison group: {key}')
        if len(seeds)<minimum_seeds:
            raise RuntimeError(f'Need {minimum_seeds} independent training seeds for {key}; got {seeds}')
        record=dict(zip(keys,key))|dict(seeds=sorted(seeds),n_seeds=len(seeds))
        for metric in ['exit1_accuracy','exit2_accuracy']:
            if record['head']=='baseline':
                record[metric+'_mean']=record[metric+'_std']=None
            else:
                vals=[r[metric] for r in group]
                record[metric+'_mean']=statistics.mean(vals)
                record[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else None
        for metric in ['accuracy','cost_ratio','mean_macs','final_exit_accuracy',
                       'harmful_early_exit_fraction','beneficial_early_exit_fraction',
                       'exit1_fraction','exit2_fraction','exit3_fraction']:
            vals=[r[metric] for r in group]
            record[metric+'_mean']=statistics.mean(vals)
            record[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else None
        summary.append(record)
    return summary


def main():
    import torch
    torch.set_num_threads(6)
    p=argparse.ArgumentParser()
    p.add_argument('manifest')
    p.add_argument('--output',default='results/final')
    p.add_argument('--exploratory',action='store_true',help='Allow fewer than 3 seeds; output marked exploratory')
    args=p.parse_args()
    rows,provenance=collect(args.manifest)
    summary=aggregate(rows,minimum_seeds=1 if args.exploratory else 3)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(exploratory=args.exploratory,manifest_sha256=file_sha256(args.manifest),
                 provenance=provenance,groups=summary)
    (out/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=len(provenance),comparison_groups=len(summary),exploratory=args.exploratory)))


if __name__=='__main__':
    main()
