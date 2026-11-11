"""Canonical metadata for the audited V2 accuracy and paired-difference plots."""
import hashlib
import json
from pathlib import Path

GROUPS=("V1 GAP", "V1 local-global", "V1 local-global + KD", "V2 residual", "V2 residual + selected KD")
PAIRS=("V2 residual minus V1 local-global", "V2 residual + selected KD minus V2 residual",
       "V2 residual + selected KD minus V1 GAP", "V2 residual + selected KD minus V1 local-global + KD")
BUDGETS=("0.55", "0.65", "0.75", "0.85")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure_metadata(name, summary_path, image_path):
    data=json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if name=="v2_budget_accuracy":
        points={group:{b:data["groups"][group][b] for b in BUDGETS} for group in GROUPS}
    elif name=="v2_paired_effects":
        points={pair:{b:data["paired_differences"][pair][b] for b in BUDGETS[:3]} for pair in PAIRS}
    elif name=="v2_exit_risk":
        points={group:data["groups"][group]["0.65"]["per_seed"] for group in GROUPS}
    else:
        raise ValueError("Unexpected figure")
    return {"scope":"exploratory test comparison from audited per-seed results",
            "summary_sha256":sha(summary_path),"image_sha256":sha(image_path),"points":points}
