"""Continue predeclared single-GPU screening after the running first job."""
import json
import subprocess
import sys
import time
from pathlib import Path

root=Path.cwd()
manifest=json.loads((root/"v2/screening_manifest.json").read_text(encoding="utf-8-sig"))
first=root/"runs_v2/residual_seed42"
while not (first/"completed.json").exists():
    time.sleep(20)
for job in manifest["runs"][1:]:
    target=root/"runs_v2"/job["name"]
    if (target/"completed.json").exists():
        continue
    options=["--resume"] if (target/"config.json").exists() else []
    if options and not (target/"last.pt").exists():
        raise RuntimeError(f"{target}: partial run has no resumable checkpoint")
    args=[sys.executable,"-u","-m","v2.train_v2","--head","residual_local_global",
          "--epochs","160","--batch-size","128","--workers","2","--seed","42",
          "--kd",str(job["kd"]),"--kd-start",str(job["kd_start"]),
          "--kd-ramp",str(job["kd_ramp"]),"--out",str(target),*options]
    target.mkdir(parents=True,exist_ok=True)
    with (target/"queue.log").open("a",encoding="utf-8") as log:
        print("Starting",job["name"],flush=True)
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
print("V2 one-seed training screen completed",flush=True)
