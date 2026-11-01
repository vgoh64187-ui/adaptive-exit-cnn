"""Audit numerical statements in the assembled report against stored outputs."""
from pathlib import Path
import json, hashlib, re

root=Path(__file__).resolve().parents[1]
report=(root/'paper/report_draft.md').read_text(encoding='utf-8')
summary=json.loads((root/'results/final/summary.json').read_text(encoding='utf-8'))
profiles=json.loads((root/'results/architecture/profiles.json').read_text(encoding='utf-8'))['profiles']
checks=[]
def present(label,value,text):
    if value not in text:raise RuntimeError(f'{label}: missing expected {value!r}')
    checks.append({'label':label,'expected':value})
def section(start,end):return report.split(start,1)[1].split(end,1)[0]
def group(head,kd,criterion,budget=None,consensus=False):
    matches=[g for g in summary['groups'] if g['head']==head and g['kd']==kd and g['criterion']==criterion and g['budget']==budget and g['consensus']==consensus]
    if len(matches)!=1:raise RuntimeError('Expected one group')
    return matches[0]
def mean_std(g,key):return f"{100*g[key+'_mean']:.2f}% ± {100*g[key+'_std']:.2f}%"
abstract=section('摘要：','关键词：')
base=group('baseline',0.,'full_depth');kd65=group('local_global',.5,'budget',.65)
for label,g,key in [('摘要基线',base,'accuracy'),('摘要预算准确率',kd65,'accuracy'),('摘要实际成本',kd65,'cost_ratio')]:present(label,mean_std(g,key),abstract)
budget=section('## 4.1 按预算分配计算','## 4.2 出口特征与蒸馏')
for head,kd,b in [('local_global',.5,.65),('gap',0.,.65),('local_global',0.,.75),('local_global',.5,.75)]:
 g=group(head,kd,'budget',b)
 for key in ['accuracy','cost_ratio']:present(f'{head}/{kd}/{b}/{key}',mean_std(g,key),budget)
present('完整基线',mean_std(base,'accuracy'),budget)
exits=section('## 4.2 出口特征与蒸馏','## 4.3 退出策略与实际执行')
for head,kd in [('gap',0.),('mlp',0.),('local_global',0.),('local_global',.5)]:
 g=group(head,kd,'full_depth')
 for key in ['exit2_accuracy','final_exit_accuracy']:present(f'{head}/{kd}/{key}',mean_std(g,key),exits)
for kd in [0.,.5]:present(f'first exit kd={kd}',mean_std(group('local_global',kd,'full_depth'),'exit1_accuracy'),exits)
routing=section('## 4.3 退出策略与实际执行','## 4.4 预算约束下的分类应用示例')
pair=json.loads((root/'results/final/paired_effects.json').read_text(encoding='utf-8'))
pair75=next(x['summary'] for x in pair['comparisons'] if x['contrast']=='local_features_vs_gap' and x['target_validation_budget']==.75)
for metric in ['accuracy','cost_ratio']:
 present(f'paired {metric}',f"{pair75[metric+'_delta_pp_mean']:+.2f} ± {pair75[metric+'_delta_pp_std']:.2f}",budget)
g=group('local_global',.5,'budget',.65)
for metric in ['exit1_fraction','exit2_fraction','exit3_fraction','harmful_early_exit_fraction','beneficial_early_exit_fraction']:
 present(f'routing {metric}',f"{100*g[metric+'_mean']:.2f}%",routing)
for consensus in [False,True]:
 g=group('local_global',.5,'budget',.65,consensus)
 for key in ['accuracy','cost_ratio']:present(f'consensus={consensus}/{key}',mean_std(g,key),routing)
for file in ['baseline_confidence75_cuda_b1','lg_kd_confidence75_cuda_b1','baseline_confidence75_cpu_b1']:
 d=json.loads((root/'results/latency/main'/f'{file}.json').read_text(encoding='utf-8'))['results'];present(file,f"{d['batch_latency_p50_ms']:.4f} ms",routing)
d=json.loads((root/'results/latency/baseline_cpu_b1_repeat.json').read_text(encoding='utf-8'))['results'];present('CPU repeat',f"{d['batch_latency_p50_ms']:.4f} ms",routing)
d=json.loads((root/'results/latency/main/lg_kd_confidence75_cuda_b1.json').read_text(encoding='utf-8'))['results'];present('GPU actual cost',f"{100*d['actual_cost_ratio']:.2f}%",routing)
application=section('## 4.4 预算约束下的分类应用示例','## 5 结论')
for cost in profiles['local_global']['exit_macs']:present('application exit MACs',f'{cost/1e6:.3f} M',application)
for path in re.findall(r'!\[.*?\]\((.*?)\)',report):
 if not (root/'paper'/path).is_file():raise RuntimeError(f'Missing figure {path}')
record={'report_sha256':hashlib.sha256(report.encode('utf-8')).hexdigest(),'checks':checks,'scope':'Narrative accuracy, costs, measured latency and referenced images; generated tables have separate provenance ledger.'}
out=root/'paper/generated/report_audit.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'narrative_numeric_checks':len(checks),'referenced_figures':len(re.findall(r'!\[.*?\]\((.*?)\)',report)),'output':str(out)},ensure_ascii=False))
