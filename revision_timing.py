"""Interleaved timings for the frozen 65% confidence policy and static controls."""
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import torch
from torchvision import datasets, transforms
from inference import backbone_only, file_sha256, load_policy, load_run, synchronise
from src.data import MEAN, STD
from src.models import profile_macs, route_cached

OUT = Path("results/revision")


def fixed_exit(model, inputs, exit_index):
    features = model.stem(inputs)
    for stage in model.stages[:exit_index + 2]:
        features = stage(features)
    return model.exits[exit_index](features)


@torch.inference_mode()
def benchmark():
    torch.set_num_threads(6)
    run = Path("runs/lg_kd_seed42")
    split = json.loads(Path("data/split_cifar100.json").read_text())
    samples = np.random.default_rng(8128).permutation(sorted(split["selection"]))[:256]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR100("data", train=True, transform=transform)
    images = torch.stack([dataset[int(i)][0] for i in samples])
    result = {"indices": samples.tolist(), "threads": 6, "warmup_batches": 32, "rounds": 5,
              "split_sha256": split["sha256"], "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "devices": {}}
    for device in ("cuda", "cpu"):
        model, _, weights = load_run(run, device)
        baseline, _, base_weights = load_run("runs/baseline_seed42", device)
        policy, _ = load_policy(run, weights, .65)
        costs = profile_macs(model)
        result.update(weights_sha256=weights, baseline_weights_sha256=base_weights,
                      policy_sha256=file_sha256(run / "policies.json"))
        controls = {
            "dynamic65": lambda x: model.predict_dynamic(x, policy),
            "full_same_weight": lambda x: backbone_only(model, x),
            "fixed_exit_1": lambda x: fixed_exit(model, x, 0),
            "fixed_exit_2": lambda x: fixed_exit(model, x, 1),
            "single_exit_baseline": lambda x: backbone_only(baseline, x),
        }
        names = list(controls)
        result["devices"][device] = {}
        for batch_size in (1, 32):
            batches = list(images.to(device).split(batch_size))
            exits = []
            for batch in batches:
                actual, ids = controls["dynamic65"](batch)
                expected, expected_ids = route_cached(model(batch), policy)
                if not torch.equal(ids, expected_ids) or not torch.equal(actual.argmax(1), expected.argmax(1)):
                    raise RuntimeError("Dynamic route disagrees with full-forward cache")
                exits.extend(ids.cpu().tolist())
            for name in names:
                for k in range(32):
                    controls[name](batches[k % len(batches)])
                synchronise(device)
            rounds = {name: [] for name in names}
            for index in range(5):
                for name in names[index:] + names[:index]:
                    elapsed = []
                    for batch in batches:
                        synchronise(device)
                        start = perf_counter_ns()
                        controls[name](batch)
                        synchronise(device)
                        elapsed.append((perf_counter_ns() - start) / 1e6)
                    rounds[name].append(dict(batch_ms=elapsed, p50=float(np.median(elapsed)),
                                            p95=float(np.percentile(elapsed, 95)),
                                            throughput=float(256000 / sum(elapsed))))
            result["devices"][device][str(batch_size)] = dict(
                rounds=rounds, dynamic_mean_cost_ratio=float(np.mean([costs["exit_macs"][i] for i in exits]) / costs["backbone_macs"]),
                exit_counts=np.bincount(exits, minlength=3).tolist(), route_disagreements=0)
            print(device, batch_size, "complete", flush=True)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "latency_interleaved.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    benchmark()
