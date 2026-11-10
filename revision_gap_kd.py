"""Train, freeze and evaluate a post-review control in its own run directory."""
import json
from pathlib import Path
import subprocess
import sys
from freeze_protocol import build_lock
from protocol import digest

MANIFEST = Path("configs/revision_gap_kd.json")
ROOT = Path("runs/revision")
LOCK = ROOT / "protocol_lock.json"


def run(script, *args, log):
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run([sys.executable, script, *map(str, args)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ROOT.mkdir(parents=True, exist_ok=True)
    for job in spec["runs"]:
        target = ROOT / job["name"]
        target.mkdir(exist_ok=True)
        if not (target / "completed.json").exists():
            print("Training", target, flush=True)
            options = ["--resume"] if (target / "last.pt").exists() else []
            run("train.py", "--head", job["head"], "--kd", job["kd"], "--seed", job["seed"],
                "--epochs", spec["epochs"], "--batch-size", spec["batch_size"],
                "--workers", spec["workers"], "--out", target, *options, log=target / "console.log")
        if not (target / "policies.json").exists():
            run("evaluate.py", target, "--split", "validation", log=target / "validation.log")
    if not LOCK.exists():
        lock = build_lock(MANIFEST, ROOT)
        lock["scope"] = "Post-review control, original CIFAR-100 test already reported. Freeze applies only to these runs."
        with LOCK.open("x", encoding="utf-8") as stream:
            json.dump(lock, stream, indent=2)
    if json.loads(LOCK.read_text(encoding="utf-8"))["manifest_sha256"] != digest(MANIFEST):
        raise RuntimeError("Supplemental manifest changed")
    for job in spec["runs"]:
        target = ROOT / job["name"]
        if not (target / "test_metrics.json").exists():
            run("evaluate.py", target, "--split", "test", "--protocol-lock", LOCK, log=target / "test.log")
    print("Supplemental runs complete", flush=True)


if __name__ == "__main__":
    main()
