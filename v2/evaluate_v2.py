"""Evaluate a completed V2 run using separately frozen policies and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from evaluate import choose_policies, metrics, temperature_fit
from src.data import MEAN, STD
from src.models import ExitPolicy
from v2.src.models import ResNet18, profile_macs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((Path.cwd() / "runs_v2").resolve()):
        raise RuntimeError("V2 evaluator only accepts runs_v2 paths")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if config["smoke_only"] or config["limit"] or completed["epochs"] != config["epochs"]:
        raise RuntimeError("Complete full-data training required")
    names = [run / f"{args.split}_predictions.npz", run / f"{args.split}_predictions.metadata.json",
             run / ("policies.json" if args.split == "validation" else "test_metrics.json")]
    if any(path.exists() for path in names):
        raise RuntimeError("Output exists; refusing to overwrite")
    if args.split == "test":
        if args.lock is None:
            raise RuntimeError("Test evaluation requires a V2 lock")
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        entry = lock["runs"].get(run.name)
        if lock.get("status") != "frozen" or entry is None:
            raise RuntimeError("Run not in frozen V2 lock")
        for name, digest in entry.items():
            if sha(run / name) != digest:
                raise RuntimeError("Frozen artifact changed: " + name)
        for path, digest in lock["sources"].items():
            if sha(path) != digest:
                raise RuntimeError("Frozen evaluation source changed: " + path)
    split = json.loads(Path("data/split_cifar100.json").read_text(encoding="utf-8"))
    if split["sha256"] != config["split_sha256"]:
        raise RuntimeError("Unexpected split")
    checkpoint_hash = sha(run / "best.pt")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(**checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    costs = profile_macs(model)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR100("data", train=args.split == "validation", transform=transform, download=False)
    indices = sorted(split["calibration"] + split["selection"]) if args.split == "validation" else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    buffers, labels = None, []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device, non_blocking=True))
            if buffers is None:
                buffers = [[] for _ in outputs]
            for buffer, output in zip(buffers, outputs):
                buffer.append(output.cpu())
            labels.append(y)
    outputs, labels = [torch.cat(b) for b in buffers], torch.cat(labels)
    if sha(run / "best.pt") != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during inference")
    np.savez_compressed(names[0], **{f"logits_{i}": z.numpy() for i, z in enumerate(outputs)},
                        labels=labels.numpy(), indices=np.asarray(indices))
    metadata = dict(weights_sha256=checkpoint_hash, split_sha256=split["sha256"],
                    prediction_sha256=sha(names[0]), split=args.split, n=len(labels),
                    precision="FP32", smoke_only=False)
    names[1].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.split == "validation":
        calibration_set = set(split["calibration"])
        calibration = torch.tensor([i in calibration_set for i in indices])
        temperatures = [temperature_fit(z[calibration], labels[calibration]) for z in outputs]
        policies = choose_policies([z[~calibration] for z in outputs], labels[~calibration], costs, temperatures)
        payload = dict(weights_sha256=checkpoint_hash, split_sha256=split["sha256"],
                       temperatures=temperatures, policies=policies, smoke_only=False)
    else:
        frozen_policies = json.loads((run / "policies.json").read_text(encoding="utf-8"))
        if frozen_policies["weights_sha256"] != checkpoint_hash:
            raise RuntimeError("Policies are not bound to checkpoint")
        rows = []
        for record in frozen_policies["policies"]:
            if "policy" in record:
                rows.append(record | dict(test=metrics(outputs, labels, ExitPolicy(**record["policy"]), costs)))
        payload = dict(weights_sha256=checkpoint_hash, split_sha256=split["sha256"],
                       protocol_sha256=sha(args.lock), policy_sha256=sha(run / "policies.json"),
                       exit_accuracy=[z.argmax(1).eq(labels).float().mean().item() for z in outputs],
                       profile=costs, policies=rows, n=len(labels), seed=config["seed"],
                       scope="exploratory follow-up on previously examined official test set")
    names[2].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(dict(run=run.name, split=args.split, samples=len(labels),
                          exit_accuracy=[z.argmax(1).eq(labels).float().mean().item() for z in outputs])))


if __name__ == "__main__":
    main()
