"""Check selected V1 figure data records against the cited tables and sources."""
import hashlib
import json
import re
from pathlib import Path

root = Path.cwd()
paper = root / "paper/report_logic_revision.md"
text = paper.read_text(encoding="utf-8")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def table(number):
    match = re.search(rf"(?m)^表 {number} [^\n]+\n\s*((?:\|[^\n]+\n)+)", text)
    if not match:
        raise RuntimeError(f"Missing table {number}")
    lines = [line for line in match.group(1).splitlines() if not re.fullmatch(r"[|:\-\s]+", line)]
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines][1:]


checked = []
for name in ("budget_accuracy", "exit_ablations", "paired_ablations", "routing_local_global_kd_confidence"):
    path = root / f"figures/print/{name}.data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data["sources"]) != 1:
        raise RuntimeError(f"Unexpected sources for {name}")
    source = data["sources"][0]
    source_path = root / source["path"].replace("\\", "/")
    if sha(source_path) != source["sha256"]:
        raise RuntimeError(f"Figure source changed: {name}")
    original = json.loads(source_path.read_text(encoding="utf-8"))
    pool = original["comparisons"] if "comparisons" in original else original["groups"]
    if any(record not in pool for record in data["records"]):
        raise RuntimeError(f"Figure records differ from source: {name}")
    checked.append({"name": name, "records": len(data["records"]), "data_sha256": sha(path)})

for name, number in (("budget_compliance", 5), ("routing_reliability", 8),
                     ("latency_interleaved", 10), ("factorial_gap_kd", 11)):
    path = root / f"figures/revision/{name}.data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    source_path = root / data["source"].replace("\\", "/")
    rows = [[str(item) for item in row] for row in data["rows"]]
    if rows != table(number):
        raise RuntimeError(f"Figure data differs from table {number}: {name}")
    checked.append({"name": name, "rows": len(rows), "data_sha256": sha(path),
                    "source_sha256": sha(source_path)})

new = root / "figures/v2/v1_routing_reliability_valid_seeds.data.json"
data = json.loads(new.read_text(encoding="utf-8"))
source = root / "results/revision/routing_analysis.json"
if sha(source) != data["source_sha256"]:
    raise RuntimeError("Revised reliability figure source changed")
for row in table(8):
    for metric, col in (("coverage", 5), ("error", 6)):
        record = next(r for r in data["records"] if r["exit"] == int(row[0]) and r["metric"] == metric)
        n = int(row[7].split("/")[0]) if metric == "error" else 3
        if f'{record["mean_percent"]:.2f}' != row[col] or record["n"] != n:
            raise RuntimeError("Revised reliability figure disagrees with table 8")
        if n == 1 and record["sample_sd_percent"] is not None:
            raise RuntimeError("Single seed cannot have a sample standard deviation")
checked.append({"name": "v1_routing_reliability_valid_seeds", "data_sha256": sha(new)})
result = {"scope": "figure data and cited-table/source consistency; visual marks are not machine-verified",
          "paper_sha256": sha(paper), "checks": checked}
out = root / "results/v2/v1_figure_data_audit.json"
if out.exists():
    raise RuntimeError("Audit already exists")
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"figure_data_files_verified": len(checked), "output": str(out)}))
