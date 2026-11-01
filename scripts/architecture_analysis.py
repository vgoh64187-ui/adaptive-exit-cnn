"""Export exact architecture dimensions and compute without trained-result claims."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.models import ResNet18,profile_macs


def main():
    torch.set_num_threads(2)
    profiles={}
    for head in ['baseline','gap','mlp','local_global']:
        model=ResNet18(head=head).eval()
        shapes=[]
        hooks=[stage.register_forward_hook(lambda m,x,y: shapes.append(list(y.shape[1:]))) for stage in model.stages]
        with torch.no_grad():outputs=model(torch.zeros(1,3,32,32))
        for hook in hooks:hook.remove()
        assert shapes==[[64,32,32],[128,16,16],[256,8,8],[512,4,4]]
        assert all(list(z.shape)==[1,100] for z in outputs)
        profile=profile_macs(model)
        profile['stage_output_shapes']=shapes
        profile['auxiliary_parameters']=sum(p.numel() for p in model.exits.parameters())
        profile['auxiliary_full_path_macs']=profile['exit_macs'][-1]-profile['backbone_macs']
        # Independent closed-form check of the measured module counts.
        audit=[]
        for i,module in enumerate(model.exits):
            c,h,w=shapes[i+1];k=100
            if head=='gap':
                parameters=c*k+k;macs=c*k
            elif head=='mlp':
                hidden=module.fc[0].out_features
                parameters=(c+k+1)*hidden+k;macs=(c+k)*hidden
            else:
                parameters=2*c*c+(k+13)*c+k+1
                macs=11*c*h*w+2*c*c+c*k
            measured_parameters=sum(p.numel() for p in module.parameters())
            measured_macs=sum(v for name,v in profile['components'].items()
                              if name.startswith(f'exits.{i}.'))
            assert parameters==measured_parameters,(head,i,parameters,measured_parameters)
            assert macs==measured_macs,(head,i,macs,measured_macs)
            audit.append(dict(exit=i+1,channels=c,height=h,width=w,
                              parameters=parameters,macs=macs,analytic_match=True))
        profile['head_formula_audit']=audit
        profiles[head]=profile
    reference=profiles['baseline']
    for profile in profiles.values():
        profile['parameter_increase_percent']=100*(profile['parameters']-reference['parameters'])/reference['parameters']
        profile['full_path_mac_increase_percent']=100*profile['auxiliary_full_path_macs']/reference['backbone_macs']
    out=ROOT/'results'/'architecture';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'src'/'models.py'
    data=dict(scope='Architecture counts at input 1x3x32x32; not trained accuracy or measured latency',
              model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profiles=profiles,
              matched_capacity_difference_parameters=abs(profiles['mlp']['parameters']-profiles['local_global']['parameters']))
    (out/'profiles.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({k:{p:v[p] for p in ['parameters','auxiliary_parameters','full_path_mac_increase_percent']} for k,v in profiles.items()}))


if __name__=='__main__':main()
