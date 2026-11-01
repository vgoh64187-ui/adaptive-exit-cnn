"""Execute the declared deployment matrix after final testing and without training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid
import psutil
from inference import file_sha256
from protocol import validate_frozen_run

ROOT=Path(__file__).resolve().parent


def jobs(spec):
    result=[]
    for device in spec['devices']:
        for batch in spec['batch_sizes']:
            for case in spec['cases']:
                name=f"{case['name']}_{device}_b{batch}"
                result.append(case|dict(name=name,device=device,batch_size=batch))
    if len({x['name'] for x in result})!=len(result):
        raise RuntimeError('Duplicate benchmark names')
    return result


def require_idle_training():
    for p in psutil.process_iter(['pid','cmdline','cwd']):
        try:
            if p.info['cwd'] and Path(p.info['cwd']).resolve()==ROOT and any(Path(a).name in ['train.py','run_queue.py'] for a in (p.info['cmdline'] or [])):
                raise RuntimeError(f'Training or queue is still active (PID {p.pid}); defer formal timing')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/benchmark_matrix.json')
    p.add_argument('--protocol-lock',default='docs/protocol_lock.json');p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();spec=json.loads(Path(args.manifest).read_text());matrix=jobs(spec)
    if args.plan_only:
        print(json.dumps(dict(count=len(matrix),jobs=matrix),indent=2));return
    require_idle_training()
    lock=json.loads(Path(args.protocol_lock).read_text())
    protocol_sha=file_sha256(args.protocol_lock)
    for name in lock['runs']:
        run=Path('runs')/name;config=json.loads((run/'config.json').read_text())
        validate_frozen_run(run,config,lock)
        result=json.loads((run/'test_metrics.json').read_text())
        if result['protocol_sha256']!=protocol_sha:raise RuntimeError('Complete frozen test evaluation before timing')
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    for job in matrix:
        if (Path('runs')/'BENCHMARK_PAUSE').exists():
            print('Paused before next timing job');return
        require_idle_training()
        target=out/(job['name']+'.json');run=Path('runs')/job['run']
        if target.exists():
            prior=json.loads(target.read_text())
            campaign=prior.get('campaign',{})
            sources=['benchmark.py','inference.py','src/models.py']
            valid=(campaign.get('manifest_sha256')==file_sha256(args.manifest)
                   and campaign.get('protocol_sha256')==protocol_sha and campaign.get('job')==job
                   and prior['weights_sha256']==file_sha256(run/'best.pt')
                   and prior['policy_sha256']==file_sha256(run/'policies.json')
                   and not prior['smoke_only'] and not prior['results']['cached_route_disagreements']
                   and all(prior['source_sha256'].get(s)==file_sha256(s) for s in sources))
            if not valid:raise RuntimeError(f'{target}: existing measurement has incompatible provenance; inspect manually')
            print(json.dumps(dict(job=job['name'],status='reused_verified_measurement')),flush=True)
            continue
        # A terminated benchmark must not leave an incomplete final-named result.
        pending=out/(job['name']+'.pending-'+uuid.uuid4().hex+'.json')
        cmd=[sys.executable,'benchmark.py',str(run),'--device',job['device'],'--batch-size',str(job['batch_size']),
             '--samples',str(spec['samples']),'--warmup',str(spec['warmup']),'--repeats',str(spec['repeats']),
             '--threads',str(spec['threads']),'--budget',str(job['budget']),'--output',str(pending)]
        for key,flag in [('consensus','--consensus'),('full_depth','--full-depth')]:
            if job.get(key):cmd.append(flag)
        print(json.dumps(dict(job=job['name'],command=cmd)),flush=True)
        with (out/(job['name']+'.log')).open('w',encoding='utf-8') as stream:
            subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
        artifact=json.loads(pending.read_text())
        if artifact['results']['cached_route_disagreements']:
            raise RuntimeError('Dynamic/cache route mismatch; inspect before continuing timing')
        artifact['campaign']=dict(manifest_sha256=file_sha256(args.manifest),protocol_sha256=protocol_sha,
                                 job=job,scope='seed42 deployment study; repetition measures execution, not independent training')
        artifact['settings']['output']=str(target)
        pending.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
        if target.exists():raise RuntimeError(f'{target} appeared during timing; preserve both files for inspection')
        pending.replace(target)
    (out/'completed.json').write_text(json.dumps(dict(jobs=len(matrix),manifest_sha256=file_sha256(args.manifest),
                                                        protocol_sha256=protocol_sha),indent=2))


if __name__=='__main__':main()
