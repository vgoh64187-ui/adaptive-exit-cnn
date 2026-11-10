"""Aggregate audited frozen comparisons without choosing methods on test."""
import json
import statistics
from pathlib import Path
from v2.evaluate_v2 import sha

ROOT=Path.cwd()
V2ROOT=ROOT/"runs_v2"
OUTPUT=ROOT/"results/v2/final_comparison.json"
BUDGETS=(.55,.65,.75,.85)


def summarize(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None)


def read_record(path,budget):
    data=json.loads((path/"test_metrics.json").read_text(encoding="utf-8"))
    choices=[p for p in data["policies"] if p.get("criterion")=="budget"
             and p.get("budget")==budget and "policy" in p and not p["policy"]["consensus"]]
    if len(choices)!=1 or data["n"]!=10000:
        raise RuntimeError(f"Expected one confidence policy at {budget} for {path}")
    record=choices[0]["test"]
    return dict(seed=data["seed"],accuracy=record["accuracy"],cost_ratio=record["cost_ratio"],
                exit_counts=record["exit_counts"],n=record["n"],
                harmful_early_exit_fraction=record["harmful_early_exit_fraction"],
                beneficial_early_exit_fraction=record["beneficial_early_exit_fraction"],
                final_exit_accuracy=data["exit_accuracy"][-1],
                metrics_sha256=sha(path/"test_metrics.json"))


def main():
    lock=V2ROOT/"v2_screening_lock.json"
    selection=json.loads((V2ROOT/"screening_selection.json").read_text(encoding="utf-8"))
    audit=json.loads((ROOT/"results/v2/v2_full_audit.json").read_text(encoding="utf-8"))
    locked=json.loads(lock.read_text(encoding="utf-8"))
    if {r["run"] for r in audit["runs"]}!=set(locked["runs"]):
        raise RuntimeError("V2 audit does not cover locked runs")
    for r in audit["runs"]:
        if r["metrics_sha256"]!=sha(V2ROOT/r["run"]/"test_metrics.json") or r["protocol_sha256"]!=sha(lock):
            raise RuntimeError("Audited results changed")
    chosen=selection["selected_kd"]
    prefix="residual_kd" if chosen=="residual_kd_seed42" else "residual_delayed_kd"
    groups={"V1 GAP":(ROOT/"runs","gap"),
            "V1 local-global":(ROOT/"runs","lg"),
            "V1 local-global + KD":(ROOT/"runs","lg_kd"),
            "V2 residual":(V2ROOT,"residual"),
            "V2 residual + selected KD":(V2ROOT,prefix)}
    result=dict(scope="exploratory comparison on previously inspected official CIFAR-100 test set",
                locked_v2_protocol_sha256=sha(lock),selection_sha256=sha(V2ROOT/"screening_selection.json"),
                selection=selection["selected_kd"],groups={},paired_differences={})
    for label,(base,prefix) in groups.items():
        result["groups"][label]={}
        for budget in BUDGETS:
            rows=[read_record(base/f"{prefix}_seed{seed}",budget) for seed in (17,42,73)]
            result["groups"][label][str(budget)]=dict(per_seed=rows,
                accuracy=summarize([x["accuracy"] for x in rows]),
                cost_ratio=summarize([x["cost_ratio"] for x in rows]),
                final_exit_accuracy=summarize([x["final_exit_accuracy"] for x in rows]),
                compliant=sum(x["cost_ratio"]<=budget for x in rows))
    comparisons=(("V2 residual","V1 local-global"),
                 ("V2 residual + selected KD","V2 residual"),
                 ("V2 residual + selected KD","V1 GAP"),
                 ("V2 residual + selected KD","V1 local-global + KD"))
    for left,right in comparisons:
        key=left+" minus "+right
        result["paired_differences"][key]={}
        for budget in BUDGETS:
            a=result["groups"][left][str(budget)]["per_seed"]
            b=result["groups"][right][str(budget)]["per_seed"]
            if [x["seed"] for x in a]!=[x["seed"] for x in b]:
                raise RuntimeError("Seed pairing mismatch")
            da=[x["accuracy"]-y["accuracy"] for x,y in zip(a,b)]
            dc=[x["cost_ratio"]-y["cost_ratio"] for x,y in zip(a,b)]
            result["paired_differences"][key][str(budget)]=dict(
                accuracy=summarize(da),cost_ratio=summarize(dc),
                per_seed=[dict(seed=x["seed"],accuracy_delta=x["accuracy"]-y["accuracy"],
                               cost_delta=x["cost_ratio"]-y["cost_ratio"]) for x,y in zip(a,b)])
    if OUTPUT.exists():
        raise RuntimeError("Result exists, refusing overwrite")
    with OUTPUT.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(OUTPUT)


if __name__=="__main__":
    main()
