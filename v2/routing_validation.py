"""Validation-only, held-out routing diagnostics for previously trained models."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / "results/v2/routing_validation.json"
SPLIT = json.loads((ROOT / "data/split_cifar100.json").read_text(encoding="utf-8"))
BUDGETS = (0.55, 0.65, 0.75, 0.85)
VARIANTS = (
    ("confidence21", 21, 0.0, 0.0),
    ("confidence101", 101, 0.0, 0.0),
    ("margin005", 101, 0.05, 0.0),
    ("margin015", 101, 0.15, 0.0),
    ("safety005", 101, 0.0, 0.005),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(logits, temperature):
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    x = np.exp(z)
    return x / x.sum(axis=1, keepdims=True)


def evaluate(scores, predictions, labels, costs, thresholds, margin, indices):
    c1, m1, c2, m2 = (x[indices] for x in scores)
    exits = np.full(len(indices), 2, dtype=np.int64)
    first = (c1 >= thresholds[0]) & (m1 >= margin)
    second = ~first & (c2 >= thresholds[1]) & (m2 >= margin)
    exits[first] = 0
    exits[second] = 1
    pred = np.stack([x[indices] for x in predictions], axis=1)
    correct = pred[np.arange(len(indices)), exits] == labels[indices]
    counts = np.bincount(exits, minlength=3)
    return dict(n=len(indices), correct=int(correct.sum()),
                accuracy=float(correct.mean()),
                cost_ratio=float(np.dot(counts, costs["exit_macs"]) /
                                 (len(indices) * costs["backbone_macs"])),
                exit_counts=counts.tolist())


def select(scores, predictions, labels, costs, ngrid, margin, budget, indices):
    c1, m1, c2, m2 = (x[indices] for x in scores)
    labels = labels[indices]
    q = np.linspace(0, 1, ngrid)
    thresholds = [np.unique(np.r_[0.0, 1.1, np.quantile(c, q)]) for c in (c1, c2)]
    first = ((c1[:, None] >= thresholds[0]) & (m1[:, None] >= margin)).astype(np.int64)
    second = ((c2[:, None] >= thresholds[1]) & (m2[:, None] >= margin)).astype(np.int64)
    remaining = 1 - first
    right = [(p[indices] == labels).astype(np.int64) for p in predictions]
    nfirst = first.sum(axis=0)[:, None]
    nsecond = remaining.T @ second
    nthird = len(indices) - nfirst - nsecond
    correct = (first.T @ right[0])[:, None] + remaining.T @ right[2][:, None] + (
        remaining.T @ (second * (right[1] - right[2])[:, None]))
    ratio = (nfirst * costs["exit_macs"][0] + nsecond * costs["exit_macs"][1]
             + nthird * costs["exit_macs"][2]) / (len(indices) * costs["backbone_macs"])
    feasible = ratio <= budget + 1e-12
    if not feasible.any():
        return None
    best_accuracy = correct[feasible].max()
    candidates = feasible & (correct == best_accuracy)
    minimum_cost = ratio[candidates].min()
    row, col = np.argwhere(candidates & np.isclose(ratio, minimum_cost, rtol=0, atol=1e-12))[0]
    return [float(thresholds[0][row]), float(thresholds[1][col])]


def main():
    result = dict(scope="validation exploratory; V1 checkpoints and temperatures already selected using validation",
                  split_sha256=SPLIT["sha256"], budgets=list(BUDGETS), runs=[])
    selected_indices = np.asarray(SPLIT["selection"])
    for model in ("gap", "lg_kd"):
        for seed in (17, 42, 73):
            run = ROOT / "runs" / f"{model}_seed{seed}"
            data_path = run / "validation_predictions.npz"
            metadata = json.loads((run / "validation_predictions.metadata.json").read_text())
            config = json.loads((run / "config.json").read_text())
            policies = json.loads((run / "policies.json").read_text())
            assert metadata["prediction_sha256"] == sha(data_path)
            assert metadata["weights_sha256"] == sha(run / "best.pt") == policies["weights_sha256"]
            assert metadata["split_sha256"] == config["split_sha256"] == SPLIT["sha256"]
            costs = json.loads((run / "profile.json").read_text())
            with np.load(data_path, allow_pickle=False) as data:
                indices, labels = data["indices"], data["labels"]
                take = np.flatnonzero(np.isin(indices, selected_indices))
                assert len(take) == 2500
                logits = [data[f"logits_{i}"][take] for i in range(3)]
                labels = labels[take]
                source_ids = indices[take]
            probs = [probabilities(x, t) for x, t in zip(logits, policies["temperatures"])]
            top = [np.partition(p, -2, axis=1)[:, -2:] for p in probs[:2]]
            scores = (top[0].max(1), top[0].max(1)-top[0].min(1),
                      top[1].max(1), top[1].max(1)-top[1].min(1))
            predictions = [x.argmax(axis=1) for x in probs]
            # Deterministic, class-stratified split shared by all six models.
            fit, held = [], []
            for cls in range(100):
                group = np.flatnonzero(labels == cls)
                assert len(group) == 25
                rng = np.random.default_rng(20261110 + cls)
                group = rng.permutation(group)
                fit.extend(group[:12])
                held.extend(group[12:])
            fit, held = np.array(sorted(fit)), np.array(sorted(held))
            entry = dict(model=model, seed=seed, source_cache_sha256=sha(data_path),
                         fit_n=len(fit), held_n=len(held),
                         fit_ids_sha256=hashlib.sha256(source_ids[fit].tobytes()).hexdigest(),
                         held_ids_sha256=hashlib.sha256(source_ids[held].tobytes()).hexdigest(),
                         results=[])
            for name, ngrid, margin, safety in VARIANTS:
                for budget in BUDGETS:
                    threshold = select(scores, predictions, labels, costs, ngrid, margin,
                                       budget-safety, fit)
                    record = dict(variant=name, budget=budget, grid=ngrid, margin=margin,
                                  safety=safety, thresholds=threshold)
                    if threshold is not None:
                        record["fit"] = evaluate(scores, predictions, labels, costs, threshold, margin, fit)
                        record["held"] = evaluate(scores, predictions, labels, costs, threshold, margin, held)
                    entry["results"].append(record)
            result["runs"].append(entry)
            print(model, seed, "complete", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
