"""Summarize audited results and build comparison plots after benchmark completion."""
import json
import subprocess
import sys
import time
from pathlib import Path

selection=Path("runs_v2/screening_selection.json")
while True:
    if selection.exists():
        chosen=json.loads(selection.read_text(encoding="utf-8"))["selected_kd"]
        benchmark=Path("results/v2")/(chosen+"_latency_interleaved.json")
        if benchmark.exists() and Path("results/v2/v2_full_audit.json").exists():
            break
    time.sleep(30)
output=Path("results/v2/final_comparison.json")
if not output.exists():
    subprocess.run([sys.executable,"-u","-m","v2.summarise_v2"],check=True)
subprocess.run([sys.executable,"-u","v2/plot_final_comparison.py"],check=True)
if not Path("paper/report_v2.md").exists():
    subprocess.run([sys.executable,"-u","-m","v2.assemble_paper_v2"],check=True)
print("Audited comparison, plots and paper source complete",flush=True)
