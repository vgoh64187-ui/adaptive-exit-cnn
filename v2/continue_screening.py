"""Continue V2 screening, validation selection, paired seeds, lock, and test."""
import json
import subprocess
import sys
import time
from pathlib import Path

root=Path.cwd()
runs=root/"runs_v2"
screen=("residual_seed42","residual_kd_seed42","residual_delayed_kd_seed42")


def call(name,args):
    target=runs/name
    target.mkdir(parents=True,exist_ok=True)
    with (target/"pipeline.log").open("a",encoding="utf-8") as log:
        print("RUN",name," ".join(args),flush=True)
        subprocess.run([sys.executable,"-u",*args],stdout=log,stderr=subprocess.STDOUT,check=True)


def validate(name):
    path=runs/name
    required=[path/"validation_predictions.npz",path/"validation_predictions.metadata.json",path/"policies.json"]
    if all(p.exists() for p in required):
        return
    if any(p.exists() for p in required):
        raise RuntimeError(f"{name}: incomplete validation output requires inspection")
    call(name,["-m","v2.evaluate_v2",str(path),"--split","validation"])


def train(name,kd,start,ramp,seed):
    path=runs/name
    if (path/"completed.json").exists():
        return
    resume=(path/"config.json").exists()
    if resume and not (path/"last.pt").exists():
        raise RuntimeError(f"{name}: no checkpoint for resume")
    args=["-m","v2.train_v2","--head","residual_local_global","--epochs","160",
          "--batch-size","128","--workers","2","--seed",str(seed),"--kd",str(kd),
          "--kd-start",str(start),"--kd-ramp",str(ramp),"--out",str(path)]
    if resume:
        args.append("--resume")
    call(name,args)


while not all((runs/name/"completed.json").exists() for name in screen):
    time.sleep(30)
for name in screen:
    validate(name)
selection_path=runs/"screening_selection.json"
if not selection_path.exists():
    call("screening",["-m","v2.select_screening"])
selection=json.loads(selection_path.read_text(encoding="utf-8"))
selected=selection["selected_kd"]
if selected not in ("residual_kd_seed42","residual_delayed_kd_seed42"):
    raise RuntimeError("Unexpected screening choice")
prefix="residual_kd" if selected=="residual_kd_seed42" else "residual_delayed_kd"
start,ramp=(0,0) if prefix=="residual_kd" else (60,20)
expanded=[]
for seed in (17,73):
    control=f"residual_seed{seed}"
    intervention=f"{prefix}_seed{seed}"
    train(control,0.0,0,0,seed)
    train(intervention,0.5,start,ramp,seed)
    expanded.extend([control,intervention])
for name in expanded:
    validate(name)
all_runs=[*screen,*expanded]
lock=runs/"v2_screening_lock.json"
if not lock.exists():
    call("screening",["-m","v2.freeze_v2","--output",str(lock),*[str(runs/n) for n in all_runs]])
locked=json.loads(lock.read_text(encoding="utf-8"))
if set(locked["runs"])!=set(all_runs):
    raise RuntimeError("Existing lock does not contain exactly the expected runs")
for name in all_runs:
    path=runs/name
    outputs=[path/"test_predictions.npz",path/"test_predictions.metadata.json",path/"test_metrics.json"]
    if all(p.exists() for p in outputs):
        continue
    if any(p.exists() for p in outputs):
        raise RuntimeError(f"{name}: incomplete test output requires inspection")
    call(name,["-m","v2.evaluate_v2",str(path),"--split","test","--lock",str(lock)])
print("V2 screening and exploratory test completed",flush=True)
