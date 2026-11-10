"""Compare fixed structures and selected gamma values under the V1 counting convention."""
import hashlib
import json
from pathlib import Path
import torch

def digest(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
v1=Path('runs/lg_seed42')
v2=Path('runs_v2/residual_seed42')
profile1=json.loads((v1/'profile.json').read_text())
profile2=json.loads((v2/'profile.json').read_text())
ckpt=torch.load(v2/'best.pt',map_location='cpu',weights_only=False)
data={"scope":"seed-42 V2 selected checkpoint; validation-stage diagnosis",
      "convention":profile2["convention"],
      "excluded_operations_note":"Scalar and elementwise residual/gating operations, activation, normalization and dispatch are not included; identical counted MACs do not establish identical runtime.",
      "v1":{"parameters":profile1["parameters"],"exit_macs":profile1["exit_macs"],"profile_sha256":digest(v1/'profile.json')},
      "v2":{"parameters":profile2["parameters"],"exit_macs":profile2["exit_macs"],"profile_sha256":digest(v2/'profile.json'),
            "checkpoint_sha256":digest(v2/'best.pt'),"selected_epoch":ckpt["epoch"]+1,
            "gamma":[float(ckpt['model'][f'exits.{i}.gamma']) for i in range(2)]}}
out=Path('results/v2/structure_profile_seed42.json')
out.write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data,ensure_ascii=False))
