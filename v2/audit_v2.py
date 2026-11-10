"""Reconstruct a frozen V2 test result from its per-sample predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from audit_results import audit_policy, require_close
from v2.evaluate_v2 import sha
from v2.src.models import ResNet18, profile_macs


def audit(run, lock_path):
    run, lock_path = Path(run), Path(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen" or run.name not in lock["runs"]:
        raise RuntimeError("Run is absent from frozen V2 lock")
    for name, expected in lock["runs"][run.name].items():
        if sha(run / name) != expected:
            raise RuntimeError("Frozen artifact mismatch: " + name)
    for name, expected in lock["sources"].items():
        if sha(name) != expected:
            raise RuntimeError("Frozen source mismatch: " + name)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "test_predictions.metadata.json").read_text(encoding="utf-8"))
    policies = json.loads((run / "policies.json").read_text(encoding="utf-8"))
    if result["protocol_sha256"] != sha(lock_path) or result["policy_sha256"] != sha(run / "policies.json"):
        raise RuntimeError("Result not bound to lock and validation-selected policies")
    weights = sha(run / "best.pt")
    if any(record["weights_sha256"] != weights for record in (result, metadata, policies)):
        raise RuntimeError("Weight hash mismatch")
    if any(record["split_sha256"] != config["split_sha256"] for record in (result, metadata, policies)):
        raise RuntimeError("Split hash mismatch")
    cache=run/"test_predictions.npz"
    if metadata["prediction_sha256"] != sha(cache) or metadata["split"] != "test" or metadata["n"] != 10000:
        raise RuntimeError("Test prediction metadata mismatch")
    with np.load(cache,allow_pickle=False) as data:
        if not np.array_equal(data["indices"],np.arange(10000)):
            raise RuntimeError("Unexpected test sample ordering")
        labels=torch.from_numpy(data["labels"].copy())
        outputs=[torch.from_numpy(data[f"logits_{i}"].copy()) for i in range(3)]
    if labels.shape!=(10000,) or not ((labels>=0)&(labels<100)).all():
        raise RuntimeError("Unexpected label population")
    if any(x.shape!=(10000,100) or not torch.isfinite(x).all() for x in outputs):
        raise RuntimeError("Unexpected logits")
    costs=profile_macs(ResNet18(**config["model"]).eval())
    if costs!=result["profile"]:
        raise RuntimeError("Profile mismatch")
    for i,logits in enumerate(outputs):
        require_close(result["exit_accuracy"][i],int(logits.argmax(1).eq(labels).sum())/len(labels),
                      f"exit {i+1}")
    expected=[p for p in policies["policies"] if "policy" in p]
    if len(expected)!=len(result["policies"]):
        raise RuntimeError("Policy count mismatch")
    for observed, selected in zip(result["policies"],expected):
        if {k:v for k,v in observed.items() if k!="test"}!=selected:
            raise RuntimeError("Policy is not validation-selected")
        audit_policy(outputs,labels,observed["policy"],costs,observed["test"])
    return dict(run=run.name,n=len(labels),policies_checked=len(expected),
                prediction_sha256=sha(cache),metrics_sha256=sha(run/"test_metrics.json"),
                protocol_sha256=sha(lock_path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--lock",required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit(args.run,args.lock),indent=2))


if __name__=="__main__":
    main()
