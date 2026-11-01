"""CIFAR ResNet-18 with causal, genuinely conditional early exits.

The stem is adapted to 32x32 inputs. Width=64 and [2,2,2,2] blocks define
the primary baseline; smaller widths are explicitly labelled controls.
"""
from dataclasses import dataclass
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class GAPHead(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.fc = nn.Linear(channels, classes)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


class AttentionPool(nn.Module):
    """Spatially weighted pooling; its weighted sum is counted separately in MACs."""
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        weights = self.score(x).flatten(2).softmax(-1)
        return (x.flatten(2) * weights).sum(-1)


class LocalGlobalHead(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1,
                                           groups=channels, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.pool = AttentionPool(channels)
        self.gate = nn.Linear(2 * channels, channels)
        self.fc = nn.Linear(channels, classes)

    def forward(self, x):
        global_feature = x.mean((2, 3))
        local_feature = self.pool(self.local(x))
        gate = self.gate(torch.cat([global_feature, local_feature], 1)).sigmoid()
        fused = global_feature + gate * (local_feature - global_feature)
        return self.fc(fused)


class MLPHead(nn.Module):
    """Capacity control, approximately matched to the local-global head."""
    def __init__(self, channels, classes):
        super().__init__()
        target = sum(p.numel() for p in LocalGlobalHead(channels, classes).parameters())
        hidden = max(1, round((target - classes) / (channels + classes + 1)))
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(),
                                nn.Linear(hidden, classes))

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


@dataclass
class ExitPolicy:
    thresholds: tuple = (1.1, 1.1)
    temperatures: tuple = (1., 1., 1.)
    consensus: bool = False
    margin: float = 0.

    def __post_init__(self):
        if len(self.thresholds) != 2 or len(self.temperatures) != 3:
            raise ValueError('Two early thresholds and three temperatures required')
        if any(t <= 0 for t in self.temperatures):
            raise ValueError('Temperatures must be positive')


def exit_decision(logits, exit_index, policy, previous=None):
    probs = (logits.float() / policy.temperatures[exit_index]).softmax(1)
    top = probs.topk(2, dim=1)
    prediction = top.indices[:, 0]
    accept = (top.values[:, 0] >= policy.thresholds[exit_index]) & (
        top.values[:, 0] - top.values[:, 1] >= policy.margin)
    if policy.consensus and exit_index > 0:
        if previous is None:
            raise ValueError('Consensus requires the preceding causal prediction')
        accept &= prediction.eq(previous)
    return accept, prediction


class ResNet18(nn.Module):
    def __init__(self, classes=100, head='baseline', width=64):
        super().__init__()
        if head not in ('baseline', 'gap', 'local_global', 'mlp'):
            raise ValueError(head)
        self.config = dict(classes=classes, head=head, width=width)
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.stages = nn.ModuleList()
        cin = width
        for i in range(4):
            cout = width * 2 ** i
            self.stages.append(nn.Sequential(BasicBlock(cin, cout, 1 if i == 0 else 2),
                                             BasicBlock(cout, cout)))
            cin = cout
        self.final_head = GAPHead(cin, classes)
        head_type = {'gap': GAPHead, 'local_global': LocalGlobalHead, 'mlp': MLPHead}
        self.exits = nn.ModuleList([] if head == 'baseline' else [
            head_type[head](width * 2, classes), head_type[head](width * 4, classes)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """All exits for training / prediction caching (not dynamic timing)."""
        x = self.stem(x)
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.exits and i in (1, 2):
                outputs.append(self.exits[i - 1](x))
        outputs.append(self.final_head(x))
        return outputs

    @torch.no_grad()
    def predict_dynamic(self, x, policy):
        """Compact the active batch so accepted samples never reach later stages.

        CUDA conditional dispatch includes host synchronization. Timing must
        include this overhead, and must be reported separately by batch size.
        """
        if self.training:
            raise RuntimeError('Dynamic inference requires model.eval()')
        if not self.exits:
            logits = self(x)[0]
            return logits, torch.full((len(x),), 2, dtype=torch.long, device=x.device)
        active = torch.arange(len(x), device=x.device)
        result = x.new_empty((len(x), self.config['classes']))
        exit_ids = torch.full((len(x),), 2, dtype=torch.long, device=x.device)
        previous = None
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in (1, 2):
                k = i - 1
                logits = self.exits[k](x)
                accept, pred = exit_decision(logits, k, policy, previous)
                result[active[accept]] = logits[accept].to(result.dtype)
                exit_ids[active[accept]] = k
                keep = ~accept
                if not keep.any():
                    return result, exit_ids
                active, x, previous = active[keep], x[keep], pred[keep]
        result[active] = self.final_head(x).to(result.dtype)
        return result, exit_ids


@torch.no_grad()
def route_cached(outputs, policy):
    """Policy simulation only; never used to claim measured inference speed."""
    if len(outputs) == 1:
        return outputs[0], torch.full((len(outputs[0]),), 2, dtype=torch.long,
                                     device=outputs[0].device)
    chosen = outputs[-1].clone()
    ids = torch.full((len(chosen),), 2, dtype=torch.long, device=chosen.device)
    active = torch.ones_like(ids, dtype=torch.bool)
    previous = None
    for i in range(2):
        accept, prediction = exit_decision(outputs[i], i, policy, previous)
        accept &= active
        chosen[accept] = outputs[i][accept]
        ids[accept] = i
        active &= ~accept
        previous = prediction
    return chosen, ids


@torch.no_grad()
def profile_macs(model, size=32):
    """Count Conv/Linear + spatial attention weighted-sum MACs, per image.

    BN/activation/pooling/softmax and dispatch costs are excluded; report this
    convention and measured latency alongside MACs. FLOPs=2*MACs is a convention.
    """
    counts = {}
    handles = []
    def hook(name):
        def count(m, inputs, output):
            if isinstance(m, nn.Conv2d):
                counts[name] = output.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                counts[name] = output.numel() * m.in_features
            else:
                counts[name] = inputs[0].numel()
        return count
    old = model.training
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, AttentionPool)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        model.eval()(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(old)
    stem = sum(v for k, v in counts.items() if k.startswith('stem.'))
    stages = [sum(v for k, v in counts.items() if k.startswith(f'stages.{i}.')) for i in range(4)]
    heads = [sum(v for k, v in counts.items() if k.startswith(f'exits.{i}.')) for i in range(2)]
    final = sum(v for k, v in counts.items() if k.startswith('final_head.'))
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                backbone_macs=stem + sum(stages) + final,
                exit_macs=[stem + sum(stages[:2]) + heads[0],
                           stem + sum(stages[:3]) + sum(heads),
                           stem + sum(stages) + sum(heads) + final],
                components=counts, convention='Conv/Linear and attention weighted sums; 1 MAC = 2 FLOPs')
