from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cifar10_tiny_resnet_capacity_transfer as base
from shared.residual_route_capacity import RouteFamilySplit, route_family, route_split_capacity_mask

SEEDS = [225, 226]
SPARSITY = 0.99
RESERVE = 0.60
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "tuned_40_35_25"]
TRAIN_EPOCHS = 5
FT_EPOCHS = 2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 16
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.fc = nn.Linear(64, 10, bias=False)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def train_model(model, loader):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(TRAIN_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def masked_finetune(model, loader, masks):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    base.apply_mask(model, masks)
    for _ in range(FT_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            base.apply_mask(model, masks)


def eval_method(model, dense_state, train_loader, test_loader, masks):
    before = base.evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune(model, train_loader, masks)
    after = base.evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def output_profile(mask: torch.Tensor):
    flat = mask.detach().reshape(mask.shape[0], -1).float()
    fanin = flat.sum(dim=1)
    live = fanin > 0
    return {"score": float(live.float().mean().item() * torch.log1p(fanin.mean()).item()), "dead_outputs": int((~live).sum().item()), "outputs": int(flat.shape[0])}


def route_quality(masks):
    family_scores = {"main": [], "projection": [], "readout": []}
    dead = 0
    for name, mask in masks.items():
        profile = output_profile(mask)
        dead += int(profile["dead_outputs"])
        if name == "stem.weight":
            continue
        family_scores[route_family(name)].append(float(profile["score"]))
    main = min(family_scores["main"]) if family_scores["main"] else 0.0
    projection = min(family_scores["projection"]) if family_scores["projection"] else 1.0
    readout = min(family_scores["readout"]) if family_scores["readout"] else 0.0
    return {"main_path_min": main, "projection_min": projection, "fc_score": readout, "route_min": min(main, projection, readout), "total_dead_outputs": dead}


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
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_resnet20_capacity_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_RESNET20_CAPACITY_99PCT.md"
    lines = ["# CIFAR-10 ResNet-20 Capacity at 99%", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", f"Seeds: `{result['seeds']}`", "", "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = result["summary"][method]
        if method == "magnitude":
            delta = "baseline"
            wins = "baseline"
        else:
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Interpretation", "", "This is a CIFAR ResNet-20-style transfer check for path-capacity pruning at 99% sparsity."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.loaders(seed)
        model = CifarResNet20().to(base.DEVICE)
        train_model(model, train_loader)
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
            before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = route_quality(masks)
            rows.append({"seed": seed, "method": label, "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "route_quality": quality})
            print(f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {"experiment": "04_cifar10_resnet20_capacity_99pct", "setup": "CIFAR-10 ResNet-20-style transfer check for path-capacity pruning at 99% sparsity.", "device": base.DEVICE, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsity": SPARSITY, "reserve": RESERVE, "summary": summary, "paired_deltas": paired, "rows": rows}
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
