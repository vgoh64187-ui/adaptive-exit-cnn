"""Audit all frozen V2 test caches before final deployment timing."""
import json
import time
from pathlib import Path
import torch
from v2.audit_v2 import audit

lock=Path("runs_v2/v2_screening_lock.json")
while True:
    if lock.exists():
        data=json.loads(lock.read_text(encoding="utf-8"))
        if all((Path("runs_v2")/name/"test_metrics.json").exists() for name in data["runs"]):
            break
    time.sleep(30)
out=Path("results/v2/v2_full_audit.json")
if out.exists():
    raise RuntimeError("Audit output already exists; inspect rather than overwrite")
torch.set_num_threads(2)
records=[]
for name in data["runs"]:
    records.append(audit(Path("runs_v2")/name,lock))
    print(name,"OK",flush=True)
with out.open("x",encoding="utf-8") as stream:
    json.dump(dict(scope="exploratory V2 frozen per-sample audit",runs=records),stream,indent=2)
print(out,flush=True)
