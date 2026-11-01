"""Rebuild final summary from audited results before producing paper artifacts."""
import json
from pathlib import Path
from inference import file_sha256
from summarise import aggregate,collect


def required_groups():
    variants=[('gap',0.),('mlp',0.),('local_global',0.),('local_global',.5)]
    result={('baseline',0.,'full_depth',None,None,False)}
    for head,kd in variants:
        for criterion in ['full_depth','fixed_exit_1','fixed_exit_2']:
            result.add((head,kd,criterion,None,None,False))
        for consensus in [False,True]:
            for budget in [.55,.65,.75,.85,.95,1.01]:
                result.add((head,kd,'budget',budget,None,consensus))
            for drop in [.005,.01,.02]:
                result.add((head,kd,'accuracy_constraint',None,drop,consensus))
    return result


def verify_summary(summary_path,manifest='configs/main_queue.json'):
    result=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    if result.get('exploratory') or result['manifest_sha256']!=file_sha256(manifest):
        raise RuntimeError('Summary is exploratory or uses a different manifest')
    rows,provenance=collect(manifest)
    actual=aggregate(rows)
    fields=['head','kd','criterion','budget','allowed_drop','consensus']
    def key(g):return tuple(g[k] for k in fields)
    expected={key(g):g for g in actual};reported={key(g):g for g in result['groups']}
    if len(expected)!=len(actual) or len(reported)!=len(result['groups']) or set(expected)!=required_groups():
        raise RuntimeError('Missing, duplicate, or unexpected final comparison group')
    if reported!=expected or result['provenance']!=provenance:
        raise RuntimeError('Summary differs from freshly audited per-sample results')
    return result
