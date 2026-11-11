"""Wait for the verified V2 paper and generate its self-contained explainer."""
import subprocess
import sys
import time
from pathlib import Path

root=Path.cwd()
ready=(root/"results/v2/paper_verification.json", root/"results/v2/final_comparison.json",
       root/"output/基于ResNet18的提前退出CNN课程论文-V2实验版.docx")
while not all(path.is_file() for path in ready):
    time.sleep(30)
output=root/"output/提前退出CNN论文导读-V2.html"
if output.exists():
    raise RuntimeError("Existing V2 explainer requires inspection")
with (root/"results/v2/explainer_pipeline.log").open("a",encoding="utf-8") as stream:
    subprocess.run([sys.executable,"-u","-m","v2.build_explainer"],stdout=stream,
                   stderr=subprocess.STDOUT,check=True)
print("Verified V2 explainer generated; visual inspection remains",flush=True)
