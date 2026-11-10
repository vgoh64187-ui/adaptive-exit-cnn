"""Produce diagnostic validation summaries without modifying frozen V1 runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
OUTPUT = ROOT / "results/v2/history_diagnostics.json"
rows = []
for model in ("gap", "lg", "lg_kd"):
    for seed in (17, 42, 73):
        run = ROOT / "runs" / f"{model}_seed{seed}"
        path = run / "history.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        selected = max(records, key=lambda record: record["selection_score"])
        final_best = max(records, key=lambda record: record["validation"]["accuracy"][-1])
        rows.append(dict(model=model, seed=seed, history_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         selected_epoch=selected["epoch"], selected_final=selected["validation"]["accuracy"][-1],
                         final_best_epoch=final_best["epoch"], final_best_validation=final_best["validation"]["accuracy"][-1],
                         difference=final_best["validation"]["accuracy"][-1]-selected["validation"]["accuracy"][-1]))
OUTPUT.write_text(json.dumps(dict(scope="within-run validation diagnostics, no test", records=rows), indent=2),encoding="utf-8")
print(OUTPUT)
