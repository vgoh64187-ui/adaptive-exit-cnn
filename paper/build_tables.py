"""Generate manuscript tables directly from architecture and audited test results.

Every formatted numeric cell retains its input path, JSON pointer and raw value.
Architecture-only mode is available before final tests; no test values are guessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

LABELS={'baseline':'单出口 ResNet-18','gap':'GAP 多出口','mlp':'近等参数 MLP',
        'local_global':'局部—全局融合','local_global_kd':'融合 + 蒸馏'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_table(path,caption,headers,rows):
    text=caption+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    text+=''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)
    path.write_text(text,encoding='utf-8')


def build(architecture,summary,output,architecture_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    architecture=Path(architecture);data=json.loads(architecture.read_text())
    if sha('src/models.py')!=data['model_source_sha256']:
        raise RuntimeError('Architecture report is stale')
    bindings=[];sources={str(architecture):sha(architecture)}
    def number(value,pointer,source,scale=1,places=2):
        formatted=f'{value*scale:,.{places}f}'
        bindings.append(dict(source=str(source),pointer=pointer,raw=value,scale=scale,
                             decimal_places=places,formatted=formatted))
        return formatted
    rows=[]
    for head in ['baseline','gap','mlp','local_global']:
        p=data['profiles'][head];prefix=f'/profiles/{head}'
        params=number(p['parameters'],prefix+'/parameters',architecture,places=0)
        costs=['—' if head=='baseline' and i<2 else number(v,prefix+f'/exit_macs/{i}',architecture,scale=1e-6,places=3)
               for i,v in enumerate(p['exit_macs'])]
        rows.append([LABELS[head],params,*costs])
    outputs=[output/'architecture_table.md']
    write_table(outputs[-1],'表 1 网络参数量与累计路径计算量',['结构','参数量','出口 1 MACs/M','出口 2 MACs/M','完整 MACs/M'],rows)
    if not architecture_only:
        from summary_guard import verify_summary
        summary=Path(summary);results=verify_summary(summary)
        sources[str(summary)]=sha(summary)
        for record in results['provenance']:
            if sha(Path('runs')/record['run']/'test_metrics.json')!=record['metrics_sha256']:
                raise RuntimeError('Final test summary is stale')
        groups=results['groups']
        def find(head,kd,criterion,budget=None,consensus=False):
            chosen=[(i,g) for i,g in enumerate(groups) if g['head']==head and g['kd']==kd and
                    g['criterion']==criterion and g['budget']==budget and g['consensus']==consensus]
            if len(chosen)!=1:raise RuntimeError('Missing or duplicate final comparison group')
            i,g=chosen[0]
            if g['seeds']!=[17,42,73]:raise RuntimeError('Expected all three predetermined seeds')
            return i,g
        def mean_std(i,g,metric,scale=100,places=2):
            return number(g[metric+'_mean'],f'/groups/{i}/{metric}_mean',summary,scale,places)+' ± '+number(
                   g[metric+'_std'],f'/groups/{i}/{metric}_std',summary,scale,places)
        variants=[('baseline',0.),('gap',0.),('mlp',0.),('local_global',0.),('local_global',.5)]
        rows=[]
        for head,kd in variants:
            i,g=find(head,kd,'full_depth')
            key=head+('_kd' if kd else '')
            values=['—','—'] if head=='baseline' else [mean_std(i,g,m) for m in ['exit1_accuracy','exit2_accuracy']]
            rows.append([LABELS[key],*values,mean_std(i,g,'final_exit_accuracy')])
        outputs.append(output/'exit_accuracy_table.md')
        write_table(outputs[-1],'表 2 各出口测试准确率（%，均值 ± 样本标准差）',
                    ['结构及训练方式','出口 1','出口 2','最终出口'],rows)
        rows=[]
        i,g=find('baseline',0.,'full_depth')
        rows.append(['单出口基线','完整网络','—',mean_std(i,g,'accuracy'),mean_std(i,g,'cost_ratio')])
        for budget in [.65,.75,.85]:
            for head,kd,consensus in [('gap',0.,False),('mlp',0.,False),('local_global',0.,False),
                                       ('local_global',.5,False),('local_global',.5,True)]:
                i,g=find(head,kd,'budget',budget,consensus)
                key=head+('_kd' if kd else '')
                target=number(budget,f'/groups/{i}/budget',summary,scale=100,places=0)+'%'
                rows.append([LABELS[key],'置信度+一致性' if consensus else '置信度',target,
                             mean_std(i,g,'accuracy'),mean_std(i,g,'cost_ratio')])
        outputs.append(output/'budget_accuracy_table.md')
        write_table(outputs[-1],'表 3 预设预算下的分类与实测计算量（均值 ± 样本标准差）',
                    ['模型','策略','验证目标预算','测试准确率/%','测试实际计算比/%'],rows)
    ledger=dict(architecture_only=architecture_only,sources=sources,bindings=bindings,
                output_sha256={str(path):sha(path) for path in outputs},
                notes='Architecture counts are not measured latency. Test mean cost is reported, not assumed equal to validation target.')
    (output/'table_provenance.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(tables=[str(p) for p in outputs],numeric_bindings=len(bindings),architecture_only=architecture_only)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',default='results/architecture/profiles.json')
    p.add_argument('--summary',default='results/final/summary.json');p.add_argument('--output',default='paper/generated')
    p.add_argument('--architecture-only',action='store_true');a=p.parse_args()
    build(a.architecture,a.summary,a.output,a.architecture_only)
