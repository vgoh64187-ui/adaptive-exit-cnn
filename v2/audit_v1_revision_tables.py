"""Recalculate V1 revision tables 5-11 from their recorded numeric sources."""
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, stdev

root = Path.cwd()
paper = root / "paper/report_logic_revision.md"
text = paper.read_text(encoding="utf-8")
paths = [root / "results/revision" / name for name in
         ("routing_analysis.json", "latency_interleaved.json", "gap_kd_factorial.json")]
routing, timing, factorial = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
labels = {"gap": "GAP", "mlp": "近等参数 MLP", "lg": "局部—全局", "lg_kd": "局部—全局＋蒸馏"}
runs = {model: [r for r in routing["runs"] if r["model"] == model] for model in labels}
pct = lambda value: f"{100 * value:.2f}"
expected = {}
expected[5] = []
for budget in (.55, .65, .75, .85):
    for model, label in labels.items():
        group = [r for r in routing["budget_compliance"] if r["budget"] == budget
                 and not r["consensus"] and r["run"].rsplit("_seed", 1)[0] == model]
        if len(group) != 3:
            raise RuntimeError("Expected three budget records")
        expected[5].append([f"{budget:.0%}", label, pct(mean(r["accuracy"] for r in group)),
                            pct(mean(r["actual"] for r in group)), f"{sum(r['met'] for r in group)}/3",
                            pct(max(0., max(r["actual"] for r in group)-budget))])
expected[6] = []
for budget in (.65, .75):
    for model, label in labels.items():
        group = [m for r in runs[model] for m in r["matched"] if m["target"] == budget]
        avg = lambda f: pct(mean(f(m) for m in group)) if group else "—"
        expected[6].append([f"{budget:.0%}", label, f"{len(group)}/3",
                            avg(lambda m: m["validation_cost"]), avg(lambda m: m["test"]["cost_ratio"]),
                            avg(lambda m: m["test"]["accuracy"])])
expected[7] = []
for budget in (.55, .65, .75):
    for model in ("gap", "lg_kd"):
        group = [next(m for m in r["budget_metrics"] if m["budget"] == budget) for r in runs[model]]
        expected[7].append([f"{budget:.0%}", labels[model]] +
                           [pct(mean(m[mode][metric] for m in group))
                            for mode in ("raw", "calibrated") for metric in ("accuracy", "cost_ratio")])
subset = runs["lg_kd"]
expected[8] = []
for i in range(3):
    diag = [r["validation_reliability"][i] for r in subset]
    exits = [r["risk65"]["exits"][i] for r in subset]
    errors = [e["error"] for e in exits if e["error"] is not None]
    expected[8].append([str(i+1), pct(mean(d["raw"]["ece"] for d in diag)),
                        pct(mean(d["calibrated"]["ece"] for d in diag)),
                        f'{mean(d["raw"]["nll"] for d in diag):.3f}',
                        f'{mean(d["calibrated"]["nll"] for d in diag):.3f}',
                        pct(mean(e["coverage"] for e in exits)), pct(mean(errors)) if errors else "—",
                        f"{len(errors)}/3"])
expected[9] = []
for i in range(0, 10, 2):
    groups = [[r["agreement65"][i+j] for r in subset] for j in (0, 1)]
    counts = [sum(v["count"] for v in group) for group in groups]
    rates = [pct(sum(v["correct"] for v in group)/n) if n else "—" for group, n in zip(groups, counts)]
    label = f'{groups[0][0]["lower"]:.2f}–{min(1., groups[0][0]["upper"]):.2f}'
    expected[9].append([label, str(counts[0]), rates[0], str(counts[1]), rates[1]])
timing_names = {"single_exit_baseline": "单出口基线", "full_same_weight": "同权重完整骨干",
                "fixed_exit_1": "固定出口 1", "fixed_exit_2": "固定出口 2", "dynamic65": "动态 65%"}
expected[10] = []
for device, batch in (("cpu", "1"), ("cuda", "1"), ("cpu", "32"), ("cuda", "32")):
    for name, label in timing_names.items():
        group = timing["devices"][device][batch]["rounds"][name]
        if len(group) != 5:
            raise RuntimeError("Expected five timing rounds")
        expected[10].append([device.upper(), batch, label, f'{mean(x["p50"] for x in group):.3f}',
                             f'{mean(x["p95"] for x in group):.3f}',
                             f'{mean(x["throughput"] for x in group):.0f}'])
expected[11] = []
for name, label in (("gap", "GAP"), ("gap_kd", "GAP＋蒸馏"), ("lg", "局部—全局"),
                    ("lg_kd", "局部—全局＋蒸馏")):
    group = list(factorial["groups"][name].values())
    if len(group) != 3:
        raise RuntimeError("Expected three factorial seeds")
    acc = [100*r["accuracy"] for r in group]
    cost = [100*r["cost_ratio"] for r in group]
    expected[11].append([label, f"{mean(acc):.2f} ± {stdev(acc):.2f}",
                         f"{mean(cost):.2f} ± {stdev(cost):.2f}", f"{sum(c <= 65 for c in cost)}/3"])

for number, wanted in expected.items():
    match = re.search(rf"(?m)^表 {number} [^\n]+\n\s*((?:\|[^\n]+\n)+)", text)
    if not match:
        raise RuntimeError(f"Missing table {number}")
    lines = [line for line in match.group(1).splitlines() if not re.fullmatch(r"[|:\-\s]+", line)]
    actual = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines][1:]
    if actual != wanted:
        raise RuntimeError(f"Table {number} differs from numeric source: {actual} vs {wanted}")

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = root / "results/v2/v1_revision_table_audit.json"
if out.exists():
    raise RuntimeError("Audit output exists")
result = {"scope": "V1 revision tables 5-11 recomputed from revision summaries; not narrative or figures",
          "paper_sha256": sha(paper), "sources": {str(p.relative_to(root)): sha(p) for p in paths},
          "tables_verified": list(expected), "data_rows_verified": sum(map(len, expected.values()))}
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
