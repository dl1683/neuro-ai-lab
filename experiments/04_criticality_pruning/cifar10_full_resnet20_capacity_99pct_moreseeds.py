from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cifar10_resnet20_capacity_99pct as r20
import cifar10_tiny_resnet_capacity_transfer as base
from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask

SEEDS = [233, 234, 235, 236]
SPARSITY = 0.99
RESERVE = 0.60
BATCH = 256
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "tuned_40_35_25"]


def full_loaders(seed: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(base.DATA, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(base.DATA, train=False, download=True, transform=eval_tf)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


def summarize(rows):
    summary = {}
    paired = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
            "after_std": float(np.std([r["after_accuracy"] for r in selected])),
            "projection_min_mean": float(np.mean([r["route_quality"]["projection_min"] for r in selected])),
            "fc_score_mean": float(np.mean([r["route_quality"]["fc_score"] for r in selected])),
            "main_path_min_mean": float(np.mean([r["route_quality"]["main_path_min"] for r in selected])),
            "dead_outputs_mean": float(np.mean([r["route_quality"]["total_dead_outputs"] for r in selected])),
        }
        if method != "magnitude":
            deltas = []
            for seed in SEEDS:
                mag_row = next(r for r in rows if r["seed"] == seed and r["method"] == "magnitude")
                alt = next(r for r in rows if r["seed"] == seed and r["method"] == method)
                deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag_row["after_accuracy"]})
            summary[method]["after_delta_mean"] = float(np.mean([d["after_delta"] for d in deltas]))
            summary[method]["after_wins"] = int(sum(d["after_delta"] > 0 for d in deltas))
            paired.append({"method": method, "after_delta_mean": summary[method]["after_delta_mean"], "after_delta_std": float(np.std([d["after_delta"] for d in deltas])), "after_wins": summary[method]["after_wins"], "paired_rows": deltas})
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_full_resnet20_capacity_99pct_moreseeds.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_FULL_RESNET20_CAPACITY_99PCT_MORESEEDS.md"
    lines = ["# CIFAR-10 Full ResNet-20 Capacity at 99%", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", f"Seeds: `{result['seeds']}`", "", "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = result["summary"][method]
        if method == "magnitude":
            delta = "baseline"
            wins = "baseline"
        else:
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Interpretation", "", "This tests the ResNet-20-style capacity result on full CIFAR-10 train/test rather than the 20k/5k subset."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense full CIFAR", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = full_loaders(seed)
        model = r20.CifarResNet20().to(base.DEVICE)
        r20.train_model(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        masks_by_label = {
            "magnitude": base.global_mask(mag, SPARSITY),
            "global_synflow": base.global_mask(syn, SPARSITY),
            "reserve_0.60": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "tuned_40_35_25": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, TUNED_SPLIT),
        }
        for label, masks in masks_by_label.items():
            before, after = r20.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = r20.route_quality(masks)
            rows.append({"seed": seed, "method": label, "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "route_quality": quality})
            print(f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {"experiment": "04_cifar10_full_resnet20_capacity_99pct_moreseeds", "setup": "Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity.", "device": base.DEVICE, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsity": SPARSITY, "reserve": RESERVE, "summary": summary, "paired_deltas": paired, "rows": rows}
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))

