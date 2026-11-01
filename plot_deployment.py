"""Plot measured compute/latency tradeoffs, requiring a complete campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
from make_figures import plt, save, PALETTE, NAMES
from inference import file_sha256
from run_benchmarks import jobs


def validate_measurement(data):
    result=data['results'];times=result['batch_latency_ms'];sizes=result['batch_sizes']
    if not times or len(times)!=len(sizes) or any(x<=0 for x in times):raise RuntimeError('Invalid timing series')
    checks={'batch_latency_p50_ms':float(np.percentile(times,50)),
            'batch_latency_p95_ms':float(np.percentile(times,95)),
            'throughput_images_per_second':1000*sum(sizes)/sum(times)}
    for k,v in checks.items():
        if not np.isclose(result[k],v,rtol=1e-10):raise RuntimeError(f'Timing aggregate mismatch: {k}')
    if result['cached_route_disagreements'] or len(result['exits'])!=len(data['sample_indices']):
        raise RuntimeError('Dynamic routing audit mismatch')


def read_campaign(manifest,lock_path):
    spec=json.loads(Path(manifest).read_text());out=Path(spec['output']);lock=Path(lock_path)
    completion=json.loads((out/'completed.json').read_text())
    matrix=jobs(spec)
    if completion['jobs']!=len(matrix) or completion['manifest_sha256']!=file_sha256(manifest) or completion['protocol_sha256']!=file_sha256(lock):
        raise RuntimeError('Timing campaign incomplete or changed')
    records=[];indices=None
    for job in matrix:
        path=out/(job['name']+'.json');r=json.loads(path.read_text());c=r['campaign'];run=Path('runs')/job['run']
        if (c['job']!=job or c['manifest_sha256']!=file_sha256(manifest) or c['protocol_sha256']!=file_sha256(lock)
            or r['weights_sha256']!=file_sha256(run/'best.pt') or r['policy_sha256']!=file_sha256(run/'policies.json')
            or r['smoke_only'] or any(file_sha256(p)!=h for p,h in r['source_sha256'].items())):
            raise RuntimeError(f'Stale benchmark artifact: {path}')
        validate_measurement(r)
        if indices is None:indices=r['sample_indices']
        if r['sample_indices']!=indices:raise RuntimeError('Benchmark sample order differs across cases')
        records.append((path,r,job))
    return records


def plot(records,output,development=False):
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    evidence=[];legend={}
    for path,r,job in records:
        s=r['settings'];res=r['results'];device=s['device'];batch=s['batch_size']
        if device not in ('cpu','cuda') or batch not in (1,32):
            raise RuntimeError('Unexpected device or batch size')
        key=next((k for prefix,k in [('lg_kd','local_global_kd'),('baseline','baseline'),('gap','gap'),('mlp','mlp'),('lg','local_global')] if job['run'].startswith(prefix)),None)
        if key is None:raise RuntimeError('Unknown model in benchmark')
        variant=('单出口完整执行' if key=='baseline' else '骨干完整执行' if s['full_depth'] else
                 f"一致性（预算{s['budget']:.0%}）" if s['consensus'] else '置信度（预算75%）')
        label=NAMES[key]+'：'+variant;marker='s' if s['full_depth'] else ('^' if s['consensus'] else 'o')
        ax=axes[0 if batch==1 else 1,0 if device=='cpu' else 1]
        x=100*res['actual_cost_ratio'];y=res['batch_latency_p50_ms'] if batch==1 else res['throughput_images_per_second']
        h=ax.scatter(x,y,color=PALETTE[key],marker=marker,s=35,zorder=3,label=label)
        legend[label]=h
        if batch==1:
            ax.vlines(x,y,res['batch_latency_p95_ms'],color=PALETTE[key],alpha=.6,lw=.8)
        evidence.append(dict(path=str(path),sha256=file_sha256(path),case=job['name'],device=device,batch_size=s['batch_size'],
                             mean_cost_ratio=res['actual_cost_ratio'],p50_ms=res['batch_latency_p50_ms'],
                             p95_ms=res['batch_latency_p95_ms'],throughput=res['throughput_images_per_second']))
    for row in range(2):
        for col in range(2):
            ax=axes[row,col];ax.grid(alpha=.2);ax.set_xlabel('固定验证样本实际平均计算比（%）')
            ax.set_title([['(a) CPU','(b) GPU'],['(c) CPU','(d) GPU']][row][col])
            ax.set_ylabel('单张批处理延迟（ms，p50–p95）' if row==0 else '吞吐量（图像/秒，batch 32）')
    fig.legend(legend.values(),legend.keys(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    if development:fig.suptitle('工程流程检查：冒烟样本，不能作为正式性能结果',color='#aa3333')
    save(fig,'deployment_tradeoffs',dict(scope='development only' if development else 'seed42; fixed validation timing subset, not test accuracy',
                                         duty='Measure latency and throughput instead of inferring speed from MACs',records=evidence),output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/benchmark_matrix.json');p.add_argument('--lock',default='docs/protocol_lock.json')
    p.add_argument('--output',default='figures/final');p.add_argument('--development-file');a=p.parse_args()
    if a.development_file:
        path=Path(a.development_file);r=json.loads(path.read_text());validate_measurement(r)
        if not r['smoke_only']:raise RuntimeError('Development path accepts explicitly marked smoke results only')
        records=[(path,r,dict(name='smoke',run=Path(r['settings']['run']).name))]
        plot(records,a.output,True)
    else:plot(read_campaign(a.manifest,a.lock),a.output)
