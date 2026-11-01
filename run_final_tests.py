"""Freeze completed experiments, evaluate each fixed policy, and audit results.

Run only after the full training queue has completed. Interrupted evaluations
can resume using the existing lock; existing results must pass full cache audit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from audit_results import audit_run
from protocol import digest, validate_frozen_run


def queue_complete(manifest, queue_state):
    state=json.loads(Path(queue_state).read_text())
    if state.get('status')!='queue_complete' or state.get('manifest_sha256')!=digest(manifest):
        raise RuntimeError('The exact full training queue must complete before final tests')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/main_queue.json')
    p.add_argument('--queue-state',default='runs/queue_state.json')
    args=p.parse_args();torch.set_num_threads(6)
    if Path('runs/QUEUE_PAUSE').exists():raise RuntimeError('Queue is paused')
    queue_complete(args.manifest,args.queue_state)
    manifest=json.loads(Path(args.manifest).read_text())
    # The summariser and all downstream audits use this same canonical lock.
    lock_path=Path('docs/protocol_lock.json')
    if not lock_path.exists():
        subprocess.run([sys.executable,'freeze_protocol.py',args.manifest,'--output',str(lock_path)],check=True)
    lock=json.loads(lock_path.read_text())
    if lock.get('manifest_sha256')!=digest(args.manifest) or set(lock.get('runs',{}))!={j['name'] for j in manifest['runs']}:
        raise RuntimeError('Frozen lock and comparison manifest disagree')
    for job in manifest['runs']:
        if Path('runs/QUEUE_PAUSE').exists():
            print('Paused before next final evaluation',flush=True);return
        run=Path('runs')/job['name']
        config=json.loads((run/'config.json').read_text())
        validate_frozen_run(run,config,lock)
        if not (run/'test_metrics.json').exists():
            print(json.dumps(dict(event='test_evaluation',run=job['name'])),flush=True)
            with (run/'test_console.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'evaluate.py',str(run),'--split','test','--protocol-lock',str(lock_path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=audit_run(run,str(lock_path))
        print(json.dumps(dict(event='audited',run=job['name'],policies=audit['policies_checked'])),flush=True)
    subprocess.run([sys.executable,'summarise.py',args.manifest],check=True)
    subprocess.run([sys.executable,'paired_effects.py',args.manifest],check=True)
    print(json.dumps(dict(event='final_tests_complete',runs=len(manifest['runs']),protocol_sha256=digest(lock_path))),flush=True)


if __name__=='__main__':main()
