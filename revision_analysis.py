"""Post-review analyses of saved logits. Policy fitting uses validation only."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from diagnostics import reliability
from evaluate import choose_policies, metrics
from inference import file_sha256
from src.models import ExitPolicy, route_cached

ROOT = Path("runs")
OUT = Path("results/revision")
SEEDS = (17, 42, 73)
MODELS = ("gap", "mlp", "lg", "lg_kd")
BUDGETS = (.55, .65, .75, .85, .95, 1.01)


def load_predictions(run, split, weights):
    path = run / f"{split}_predictions.npz"
    meta = json.loads((run / f"{split}_predictions.metadata.json").read_text())
    if meta["weights_sha256"] != weights or meta["prediction_sha256"] != file_sha256(path):
        raise RuntimeError(f"Cache provenance mismatch for {run} {split}")
    with np.load(path) as data:
        logits = [torch.from_numpy(data[f"logits_{i}"]) for i in range(3)]
        return logits, torch.from_numpy(data["labels"]), data["indices"]


def find_policy(policies, budget):
    return next(ExitPolicy(**r["policy"]) for r in policies
                if r["criterion"] == "budget" and r["budget"] == budget
                and "policy" in r and not r["policy"]["consensus"])


def matched_policies(outputs, labels, costs, temperatures):
    # Compare policies that use nearly the same cost on the policy-selection split.
    conf = [(z / t).softmax(1).max(1).values for z, t in zip(outputs[:2], temperatures)]
    grids = [sorted(set([0., 1.1] + torch.quantile(c, torch.linspace(0, 1, 21)).tolist())) for c in conf]
    correct = [(z.argmax(1) == labels).numpy() for z in outputs]
    cost = np.asarray(costs["exit_macs"]) / costs["backbone_macs"]
    results = {}
    for a in grids[0]:
        early = (conf[0] >= a).numpy()
        for b in grids[1]:
            middle = (conf[1] >= b).numpy() & ~early
            exits = np.where(early, 0, np.where(middle, 1, 2))
            rho = float(cost[exits].mean())
            acc = float(np.choose(exits, correct).mean())
            for cap in (.65, .75):
                if cap - .005 <= rho <= cap:
                    candidate = (acc, -rho, a, b)
                    if cap not in results or candidate > results[cap]:
                        results[cap] = candidate
    return {cap: (ExitPolicy((a, b), tuple(temperatures)), -neg_rho)
            for cap, (_, neg_rho, a, b) in results.items()}


def exit_risk(outputs, labels, policy):
    pred, ids = route_cached(outputs, policy)
    correct = pred.argmax(1).eq(labels)
    full = outputs[-1].argmax(1).eq(labels)
    rows = []
    for index in range(3):
        mask = ids == index
        count = int(mask.sum())
        rows.append(dict(exit=index + 1, count=count, coverage=count / len(labels),
                         error=float((~correct[mask]).float().mean()) if count else None,
                         harmful=int((mask & ~correct & full).sum()),
                         beneficial=int((mask & correct & ~full).sum())))
    random = sum(row["coverage"] * float((z.argmax(1) == labels).float().mean())
                 for row, z in zip(rows, outputs))
    return dict(exits=rows, accuracy=float(correct.float().mean()), random_same_exit_counts_expectation=random)


def agreement_bins(outputs, labels, policy):
    _, ids = route_cached(outputs, policy)
    reached = ids > 0
    confidence, pred = (outputs[1] / policy.temperatures[1]).softmax(1).max(1)
    agree = pred.eq(outputs[0].argmax(1))
    correct = pred.eq(labels)
    rows = []
    for lo, hi in zip((0, .5, .7, .85, .95), (.5, .7, .85, .95, 1.01)):
        group = reached & (confidence >= lo) & (confidence < hi)
        for same in (False, True):
            mask = group & (agree == same)
            rows.append(dict(lower=lo, upper=hi, agree=same, count=int(mask.sum()),
                             correct=int(correct[mask].sum())))
    return rows


def main():
    torch.set_num_threads(2)
    split = json.loads(Path("data/split_cifar100.json").read_text())
    output = {"scope": "Post-review diagnostic on published CIFAR-100 test predictions; validation-only policy selection.",
              "selection_band": "validation MAC ratio within [target-0.005, target]; maximize validation accuracy",
              "runs": []}
    for name in MODELS:
        for seed in SEEDS:
            run = ROOT / f"{name}_seed{seed}"
            frozen = json.loads((run / "policies.json").read_text())
            costs = json.loads((run / "profile.json").read_text())
            v_logits, v_labels, indices = load_predictions(run, "validation", frozen["weights_sha256"])
            mask = np.isin(indices, split["selection"])
            v_logits = [z[mask] for z in v_logits]
            v_labels = v_labels[mask]
            t_logits, t_labels, t_indices = load_predictions(run, "test", frozen["weights_sha256"])
            if not np.array_equal(t_indices, np.arange(10000)):
                raise RuntimeError(f"Unexpected test indices for {run}")
            calibrated = frozen["temperatures"]
            no_calibration = choose_policies(v_logits, v_labels, costs, (1., 1., 1.))
            budget_metrics = []
            for budget in BUDGETS:
                original = find_policy(frozen["policies"], budget)
                raw = find_policy(no_calibration, budget)
                budget_metrics.append(dict(budget=budget,
                    calibrated=metrics(t_logits, t_labels, original, costs),
                    raw=metrics(t_logits, t_labels, raw, costs)))
            matched = [dict(target=cap, validation_cost=rho, test=metrics(t_logits, t_labels, p, costs))
                       for cap, (p, rho) in matched_policies(v_logits, v_labels, costs, calibrated).items()]
            policy = find_policy(frozen["policies"], .65)
            risk = exit_risk(t_logits, t_labels, policy)
            diagnostics = []
            for z, t in zip(v_logits, calibrated):
                values = []
                for temperature in (1., t):
                    stats = reliability(z, v_labels, temperature)
                    values.append({k: v for k, v in stats.items() if k not in ("bins", "risk_coverage")})
                diagnostics.append(dict(raw=values[0], calibrated=values[1]))
            output["runs"].append(dict(model=name, seed=seed, weights_sha256=frozen["weights_sha256"],
                policy_sha256=file_sha256(run / "policies.json"), budget_metrics=budget_metrics,
                matched=matched, risk65=risk, agreement65=agreement_bins(t_logits, t_labels, policy),
                validation_reliability=diagnostics))
            print(run, "complete", flush=True)
    with Path("results/final/per_seed_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["criterion"] == "budget"]
    output["budget_compliance"] = [dict(run=r["run"], seed=int(r["seed"]), budget=float(r["budget"]),
        consensus=r["consensus"] == "True", actual=float(r["cost_ratio"]), accuracy=float(r["accuracy"]),
        met=r["test_budget_met"] == "True") for r in rows]
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "routing_analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

