"""Observe one existing full queue and start gated evaluation only on success."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from protocol import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-pid',type=int,required=True)
    p.add_argument('--manifest',default='configs/main_queue.json');args=p.parse_args()
    queue=psutil.Process(args.queue_pid);cmd=queue.cmdline()
    queue_cwd=Path(queue.cwd()).resolve()
    if (queue_cwd!=Path.cwd().resolve() or 'run_queue.py' not in [Path(s).name for s in cmd]
        or not any((queue_cwd/Path(s)).resolve()==Path(args.manifest).resolve() for s in cmd[1:])):
        raise RuntimeError('PID is not the requested training queue')
    identity=queue.create_time();manifest_hash=digest(args.manifest)
    lock=Path('runs/post_training.lock')
    if lock.exists():
        previous=json.loads(lock.read_text())
        try:
            old=psutil.Process(previous['pid'])
            if abs(old.create_time()-previous['created'])<.01:raise RuntimeError('A watcher is already active')
        except psutil.NoSuchProcess:pass
        raise RuntimeError('Stale watcher lock requires inspection before continuing')
    with lock.open('x') as f:json.dump(dict(pid=psutil.Process().pid,created=psutil.Process().create_time(),queue_pid=args.queue_pid,queue_created=identity,manifest_sha256=manifest_hash),f)
    try:
        print(json.dumps(dict(event='watching',queue_pid=args.queue_pid,queue_created=identity,manifest_sha256=manifest_hash)),flush=True)
        while True:
            if Path('runs/QUEUE_PAUSE').exists():
                print(json.dumps(dict(event='paused_by_marker')),flush=True);return
            try:
                if not queue.is_running() or queue.status()==psutil.STATUS_ZOMBIE or queue.create_time()!=identity:break
            except psutil.NoSuchProcess:break
            time.sleep(5)
        if Path('runs/QUEUE_PAUSE').exists():return
        state=json.loads(Path('runs/queue_state.json').read_text())
        if state.get('status')!='queue_complete' or state.get('manifest_sha256')!=manifest_hash or digest(args.manifest)!=manifest_hash:
            print(json.dumps(dict(event='queue_not_complete',queue_status=state.get('status'))),flush=True);return
        print(json.dumps(dict(event='starting_gated_final_tests')),flush=True)
        subprocess.run([sys.executable,'run_final_tests.py','--manifest',args.manifest],check=True)
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid')==psutil.Process().pid:lock.unlink()


if __name__=='__main__':main()
