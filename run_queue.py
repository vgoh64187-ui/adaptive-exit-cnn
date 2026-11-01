"""Run a serial, restartable training/validation queue, attaching to live jobs.

An observation timeout never causes a restart. No final test evaluation is run.
Create runs/QUEUE_PAUSE to prevent the next launch; active training is preserved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import psutil
from inference import file_sha256

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'runs/queue_state.json'
LOCK=ROOT/'runs/queue.lock'
PAUSE=ROOT/'runs/QUEUE_PAUSE'


def record(**kwargs):
    state=dict(updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kwargs)
    temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temp.replace(STATE)
    print(json.dumps(state),flush=True)


def live_training(run):
    target=(ROOT/run).resolve()
    found=[]
    for proc in psutil.process_iter(['pid','cmdline','cwd','create_time']):
        try:
            args=proc.info['cmdline'] or []
            if not any(Path(s).name=='train.py' for s in args) or '--out' not in args:
                continue
            out=args[args.index('--out')+1]
            cwd=proc.info['cwd']
            if cwd and (Path(cwd)/out).resolve()==target:
                found.append(proc)
        except (psutil.NoSuchProcess,psutil.AccessDenied,IndexError):
            pass
    return found


def acquire_lock():
    LOCK.parent.mkdir(exist_ok=True)
    if LOCK.exists():
        old=json.loads(LOCK.read_text())
        try:
            existing=psutil.Process(old['pid'])
            if abs(existing.create_time()-old['create_time'])<.01:
                raise RuntimeError(f'Queue is already running as PID {old["pid"]}')
        except psutil.NoSuchProcess:
            pass
        LOCK.replace(LOCK.with_suffix(f'.stale-{time.time_ns()}'))
    with LOCK.open('x') as f:
        json.dump(dict(pid=os.getpid(),create_time=psutil.Process().create_time()),f)


def check_config(run, job, queue):
    path=run/'config.json'
    if not path.exists():
        return
    saved=json.loads(path.read_text())
    expected=dict(head=job['head'],seed=job['seed'],kd=job['kd'],epochs=queue['epochs'],
                  batch_size=queue['batch_size'],limit=0,width=64)
    for key,value in expected.items():
        if saved.get(key)!=value:
            raise RuntimeError(f'{run.name}: config mismatch for {key}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',nargs='?',default='configs/seed42_queue.json')
    args=p.parse_args()
    os.chdir(ROOT)
    manifest=Path(args.manifest)
    queue=json.loads(manifest.read_text())
    acquire_lock()
    try:
        for job in queue['runs']:
            if PAUSE.exists():
                record(status='paused_before_launch',next_run=job['name'])
                return
            run=Path('runs')/job['name']
            check_config(run,job,queue)
            active=live_training(run)
            if active:
                record(status='attached_to_training',run=job['name'],
                       training_pids=[p.pid for p in active])
                # Revalidate process identity via psutil object, not lockfile/time elapsed.
                while any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in active):
                    time.sleep(5)
                if not (run/'completed.json').exists():
                    record(status='external_training_stopped',run=job['name'])
                    return
            if not (run/'completed.json').exists():
                if PAUSE.exists():
                    record(status='paused_before_launch',next_run=job['name'])
                    return
                run.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'train.py','--head',job['head'],'--seed',str(job['seed']),
                     '--kd',str(job['kd']),'--epochs',str(queue['epochs']),
                     '--batch-size',str(queue['batch_size']),'--workers',str(queue['workers']),
                     '--out',str(run)]
                if (run/'last.pt').exists():
                    cmd.append('--resume')
                elif (run/'config.json').exists():
                    raise RuntimeError(f'{run}: incomplete start without checkpoint; inspect manually')
                with (run/'console.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    record(status='training',run=job['name'],training_pid=child.pid,command=cmd)
                    code=child.wait()
                if code!=0:
                    record(status='training_failed',run=job['name'],exit_code=code)
                    return
            finished=json.loads((run/'completed.json').read_text())
            if finished['epochs']!=queue['epochs']:
                raise RuntimeError(f'{run}: completed epoch mismatch')
            if job['head']=='baseline' and finished['best_validation_score']<queue['minimum_baseline_validation']:
                record(status='baseline_requires_review',run=job['name'],validation=finished)
                return
            if PAUSE.exists():
                record(status='paused_after_training',run=job['name'])
                return
            policy=run/'policies.json'
            needs_evaluation=not policy.exists() or json.loads(policy.read_text())['weights_sha256']!=file_sha256(run/'best.pt')
            if needs_evaluation:
                record(status='validation',run=job['name'])
                with (run/'validation_console.log').open('a',encoding='utf-8') as log:
                    code=subprocess.call([sys.executable,'evaluate.py',str(run),'--split','validation'],stdout=log,stderr=subprocess.STDOUT)
                if code!=0:
                    record(status='validation_failed',run=job['name'],exit_code=code)
                    return
            record(status='run_complete',run=job['name'])
        record(status='queue_complete',manifest=str(manifest),manifest_sha256=file_sha256(manifest))
    finally:
        if LOCK.exists() and json.loads(LOCK.read_text()).get('pid')==os.getpid():
            LOCK.unlink()


if __name__=='__main__':
    main()
