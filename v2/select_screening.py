"""Choose a seed-42 distillation schedule using only predeclared validation scores."""
import json
from pathlib import Path
from v2.evaluate_v2 import sha

ROOT=Path.cwd()
MANIFEST=ROOT/"v2/screening_manifest.json"
OUTPUT=ROOT/"runs_v2/screening_selection.json"


def scores(name):
    path=ROOT/"runs_v2"/name/"policies.json"
    payload=json.loads(path.read_text(encoding="utf-8"))
    records=[]
    for target in (.55,.65,.75):
        candidates=[r for r in payload["policies"] if r.get("criterion")=="budget"
                    and r.get("budget")==target and "policy" in r and not r["policy"]["consensus"]]
        if len(candidates)!=1 or candidates[0]["selection"]["n"]!=2500:
            raise RuntimeError(f"Missing unique confidence policy at {target} for {name}")
        records.append(candidates[0]["selection"])
    return dict(run=name, policy_sha256=sha(path),
                mean_accuracy=sum(r["accuracy"] for r in records)/len(records),
                mean_cost=sum(r["cost_ratio"] for r in records)/len(records),
                per_budget=records)


def main():
    if OUTPUT.exists():
        raise RuntimeError("Selection already fixed; refusing to overwrite")
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidates=[scores("residual_kd_seed42"),scores("residual_delayed_kd_seed42")]
    # Tuple order gives the predeclared immediate-KD tie break.
    selected=max(candidates,key=lambda x:(x["mean_accuracy"],-x["mean_cost"],
                                           x["run"]=="residual_kd_seed42"))
    control=scores("residual_seed42")
    payload=dict(scope="exploratory seed-42 validation selection before V2 test",
                 manifest_sha256=sha(MANIFEST),selection_rule=manifest["selection_rule"],
                 control=control,candidates=candidates,selected_kd=selected["run"])
    with OUTPUT.open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2)
    print(json.dumps(dict(selection=str(OUTPUT),selected_kd=selected["run"],sha256=sha(OUTPUT))))


if __name__=="__main__":
    main()
