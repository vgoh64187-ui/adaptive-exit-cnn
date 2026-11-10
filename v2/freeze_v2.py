"""Freeze completed V2 artifacts before exploratory reuse of the V1 test set."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from v2.evaluate_v2 import sha
from v2.train_v2 import source_hash

SOURCES = ("v2/train_v2.py", "v2/src/data.py", "v2/src/models.py",
           "v2/evaluate_v2.py", "v2/audit_v2.py", "v2/freeze_v2.py", "v2/select_screening.py",
           "v2/screening_manifest.json", "v2/screen_queue.py", "v2/continue_screening.py", "v2/benchmark_v2.py", "v2/benchmark_after_screening.py", "v2/audit_after_screening.py",
           "evaluate.py", "audit_results.py", "protocol.py", "inference.py", "src/data.py", "src/models.py", "data/split_cifar100.json")
ARTIFACTS = ("best.pt", "config.json", "history.jsonl", "completed.json",
             "validation_predictions.npz", "validation_predictions.metadata.json", "policies.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    root = (Path.cwd() / "runs_v2").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise RuntimeError("Choose an unused lock path in runs_v2")
    entries = {}
    reference = None
    for arg in args.runs:
        run = arg.resolve()
        if not run.is_relative_to(root) or run.name in entries:
            raise RuntimeError("Invalid or duplicate V2 run")
        if (run / "test_metrics.json").exists() or (run / "test_predictions.npz").exists():
            raise RuntimeError("This run has already been evaluated on test")
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if cfg["source_sha256"] != source_hash():
            raise RuntimeError("Training source does not match completed run")
        shared = {k: cfg[k] for k in ("epochs", "batch_size", "workers", "width",
                                       "lr", "weight_decay", "temperature", "split_sha256", "head")}
        if reference is None:
            reference = shared
        elif shared != reference:
            raise RuntimeError("V2 runs use different common training settings")
        if cfg["smoke_only"] or cfg["limit"]:
            raise RuntimeError("Smoke runs cannot be locked")
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line]
        if [r["epoch"] for r in history] != list(range(1, cfg["epochs"] + 1)):
            raise RuntimeError("Incomplete or repeated epochs")
        if json.loads((run / "completed.json").read_text())["epochs"] != cfg["epochs"]:
            raise RuntimeError("Incomplete training")
        meta = json.loads((run / "validation_predictions.metadata.json").read_text())
        policy = json.loads((run / "policies.json").read_text())
        if (meta["prediction_sha256"] != sha(run / "validation_predictions.npz")
                or meta["weights_sha256"] != sha(run / "best.pt")
                or policy["weights_sha256"] != meta["weights_sha256"]
                or meta["split_sha256"] != cfg["split_sha256"]
                or meta["n"] != 5000 or meta["split"] != "validation"):
            raise RuntimeError("Validation artifacts not bound to this run")
        entries[run.name] = {name: sha(run / name) for name in ARTIFACTS}
    sources = {path: sha(path) for path in SOURCES}
    selection = Path("runs_v2/screening_selection.json")
    if selection.exists():
        sources[str(selection)] = sha(selection)
    lock = dict(status="frozen", frozen_at=datetime.now(timezone.utc).isoformat(),
                scope="exploratory follow-up; CIFAR-100 official test results already known",
                sources=sources, runs=entries)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
    print(json.dumps(dict(output=str(output), runs=list(entries), sha256=sha(output))))


if __name__ == "__main__":
    main()
