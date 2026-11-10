"""Run the interleaved deployment benchmark after every locked V2 test completes."""
import json
import subprocess
import sys
import time
from pathlib import Path

lock=Path("runs_v2/v2_screening_lock.json")
while True:
    if lock.exists():
        data=json.loads(lock.read_text(encoding="utf-8"))
        if (Path("results/v2/v2_full_audit.json").exists() and
                all((Path("runs_v2")/name/"test_metrics.json").exists() for name in data["runs"])):
            break
    time.sleep(30)
selected=json.loads(Path("runs_v2/screening_selection.json").read_text(encoding="utf-8"))["selected_kd"]
output=Path("results/v2")/(selected+"_latency_interleaved.json")
if not output.exists():
    with Path("results/v2/benchmark_pipeline.log").open("a",encoding="utf-8") as log:
        subprocess.run([sys.executable,"-u","-m","v2.benchmark_v2"],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
print("Benchmark complete",flush=True)
