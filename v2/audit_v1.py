"""Read-only audit of frozen V1 and GAP-KD per-sample prediction caches."""
import json
from pathlib import Path
import torch
from audit_results import audit_run

torch.set_num_threads(2)
runs=[(Path("runs")/f"{model}_seed{seed}","docs/protocol_lock.json")
      for model in ("baseline","gap","mlp","lg","lg_kd") for seed in (17,42,73)]
runs += [(Path("runs/revision")/f"gap_kd_seed{seed}","runs/revision/protocol_lock.json")
         for seed in (17,42,73)]
records=[]
for path,lock in runs:
    result=audit_run(path,lock)
    records.append({k:result[k] for k in ("run","policies_checked","n","prediction_sha256","metrics_sha256","protocol_sha256")})
    print(path, "OK",flush=True)
out=Path("results/v2/v1_full_audit.json")
out.write_text(json.dumps({"scope":"V1 frozen per-sample cache reconstruction","runs":records},indent=2),encoding="utf-8")
print(out)
