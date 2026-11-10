from pathlib import Path
root=Path.cwd()
pkg=root/'v2'/'src'
pkg.mkdir(exist_ok=True)
(root/'v2/__init__.py').write_text('',encoding='utf-8')
(pkg/'__init__.py').write_text('',encoding='utf-8')
data=(root/'src/data.py').read_text(encoding='utf-8')
data=data.replace("    path.write_text(json.dumps(split), encoding='utf-8')", "    if not path.exists():\n        path.write_text(json.dumps(split), encoding='utf-8')")
(pkg/'data.py').write_text(data,encoding='utf-8')
models=(root/'src/models.py').read_text(encoding='utf-8')
before='\n\nclass MLPHead(nn.Module):'
addition='''

class ResidualLocalGlobalHead(LocalGlobalHead):
    """GAP-preserving initial state with a learnable local correction."""
    def __init__(self, channels, classes):
        super().__init__(channels, classes)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        global_feature = x.mean((2, 3))
        local_feature = self.pool(self.local(x))
        gate = self.gate(torch.cat([global_feature, local_feature], 1)).sigmoid()
        fused = global_feature + self.gamma * gate * (local_feature - global_feature)
        return self.fc(fused)
'''
assert before in models
models=models.replace(before,addition+before)
models=models.replace("('baseline', 'gap', 'local_global', 'mlp')","('baseline', 'gap', 'local_global', 'mlp', 'residual_local_global')")
models=models.replace("'local_global': LocalGlobalHead, 'mlp': MLPHead}","'local_global': LocalGlobalHead, 'mlp': MLPHead,\n                     'residual_local_global': ResidualLocalGlobalHead}")
(pkg/'models.py').write_text(models,encoding='utf-8')
train=(root/'train.py').read_text(encoding='utf-8')
train=train.replace('from src.data import loaders','from v2.src.data import loaders')
train=train.replace('from src.models import ResNet18, profile_macs','from v2.src.models import ResNet18, profile_macs')
train=train.replace("*Path('src').glob('*.py')","*Path('v2/src').glob('*.py')")
train=train.replace("choices=['baseline', 'gap', 'local_global', 'mlp']","choices=['baseline', 'gap', 'local_global', 'mlp', 'residual_local_global']")
train=train.replace("    p.add_argument('--temperature', type=float, default=3.)","    p.add_argument('--temperature', type=float, default=3.)\n    p.add_argument('--kd-start', type=int, default=0)\n    p.add_argument('--kd-ramp', type=int, default=0)")
train=train.replace("if args.epochs < 1 or args.kd < 0 or args.temperature <= 0:","if args.epochs < 1 or args.kd < 0 or args.temperature <= 0 or args.kd_start < 0 or args.kd_ramp < 0:")
train=train.replace("'kd','temperature','limit'","'kd','temperature','kd_start','kd_ramp','limit'")
train=train.replace("                if args.kd and len(outputs) > 1:\n                    temperature",'''                effective_kd = args.kd if epoch >= args.kd_start else 0.
                if effective_kd and args.kd_ramp:
                    effective_kd *= min(1., (epoch - args.kd_start + 1) / args.kd_ramp)
                if effective_kd and len(outputs) > 1:
                    temperature''')
train=train.replace("loss = loss + args.kd * kd","loss = loss + effective_kd * kd")
train=train.replace("row = dict(epoch=epoch + 1, lr=lr, train_loss=loss_sum / count,","row = dict(epoch=epoch + 1, lr=lr, effective_kd=effective_kd, train_loss=loss_sum / count,")
(root/'v2/train_v2.py').write_text(train,encoding='utf-8')
