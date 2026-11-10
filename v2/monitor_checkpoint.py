"""Sample completed best checkpoints without affecting training."""
import json
import sys
import time
from pathlib import Path
import torch

run=Path(sys.argv[1])
output=Path("results/v2")/(run.name+"_gamma_trace.jsonl")
last=None
while True:
    try:
        checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
        epoch=int(checkpoint["epoch"])+1
        if epoch!=last:
            row=dict(epoch=epoch,gamma=[float(checkpoint["model"][f"exits.{i}.gamma"]) for i in range(2)],
                     validation=checkpoint["validation"]["accuracy"])
            with output.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            last=epoch
    except (OSError,RuntimeError,EOFError,KeyError,ValueError):
        pass
    if (run/"completed.json").exists():
        break
    time.sleep(30)
