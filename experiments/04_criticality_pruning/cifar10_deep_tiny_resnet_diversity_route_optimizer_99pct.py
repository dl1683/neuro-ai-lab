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

SEEDS = [219, 220]
SPARSITY = 0.99
RESERVE = 0.60
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "tuned_40_35_25", "diversity_target_optimizer"]
TRAIN_EPOCHS = 5
FT_EPOCHS = 2


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class DeepTinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.b1a = BasicBlock(16, 16)
        self.b1b = BasicBlock(16, 16)
        self.b2a = BasicBlock(16, 32, stride=2)
        self.b2b = BasicBlock(32, 32)
        self.b3a = BasicBlock(32, 64, stride=2)
        self.b3b = BasicBlock(64, 64)
        self.fc = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.b1a(x)
        x = self.b1b(x)
        x = self.b2a(x)
        x = self.b2b(x)
        x = self.b3a(x)
        x = self.b3b(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def train_model(model, loader, epochs=TRAIN_EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def masked_finetune_model(model, loader, masks):
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
    masked_finetune_model(model, train_loader, masks)
    after = base.evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def output_profile(mask: torch.Tensor):
    flat = mask.detach().reshape(mask.shape[0], -1).float()
    fanin = flat.sum(dim=1)
    live = fanin > 0
    return {
        "score": float(live.float().mean().item() * torch.log1p(fanin.mean()).item()),
        "dead_outputs": int((~live).sum().item()),
        "outputs": int(flat.shape[0]),
    }


def route_quality(masks):
    profiles = {name: output_profile(mask) for name, mask in masks.items()}
    family_scores = {"main": [], "projection": [], "readout": []}
    dead = 0
    for name, profile in profiles.items():
        dead += int(profile["dead_outputs"])
        if name == "stem.weight":
            continue
        family_scores[route_family(name)].append(float(profile["score"]))
    main = min(family_scores["main"]) if family_scores["main"] else 0.0
    projection = min(family_scores["projection"]) if family_scores["projection"] else 1.0
    readout = min(family_scores["readout"]) if family_scores["readout"] else 0.0
    return {"main_path_min": main, "projection_min": projection, "fc_score": readout, "route_min": min(main, projection, readout), "total_dead_outputs": dead}


def candidate_splits():
    values = [round(x, 2) for x in np.arange(0.20, 0.56, 0.05)]
    for main in values:
        for projection in values:
            readout = round(1.0 - main - projection, 2)
            if 0.20 <= readout <= 0.40:
                yield RouteFamilySplit(main=main, projection=projection, readout=readout).normalized()


def target_loss(quality, template_quality, reserve_quality, split):
    projection_target = max(1e-6, float(template_quality["projection_min"]))
    readout_target = max(1e-6, float(template_quality["fc_score"]))
    main_floor = max(1e-6, float(reserve_quality["main_path_min"]))
    projection_loss = ((quality["projection_min"] - projection_target) / projection_target) ** 2
    readout_loss = ((quality["fc_score"] - readout_target) / readout_target) ** 2
    main_loss = 0.0 if quality["main_path_min"] >= main_floor else ((main_floor - quality["main_path_min"]) / main_floor) ** 2
    dead_penalty = 0.004 * float(quality["total_dead_outputs"])
    concentration_penalty = 0.35 * (split.main**2 + split.projection**2 + split.readout**2)
    projection_overuse = 0.0 if split.projection <= 0.40 else 0.50 * (split.projection - 0.40) ** 2
    return projection_loss + readout_loss + main_loss + dead_penalty + concentration_penalty + projection_overuse


def optimize_split(syn, mag, template_quality, reserve_quality):
    best = None
    for split in candidate_splits():
        masks = route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split)
        quality = route_quality(masks)
        loss = target_loss(quality, template_quality, reserve_quality, split)
        item = {"split": split, "loss": float(loss), "quality": quality}
        if best is None or loss < best["loss"]:
            best = item
    return best


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
            paired.append({"method": method, "after_delta_mean": summary[method]["after_delta_mean"], "after_wins": summary[method]["after_wins"], "paired_rows": deltas})
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_DEEP_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT.md"
    lines = ["# CIFAR-10 DeepTinyResNet Diversity Route Optimizer at 99%", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", f"Seeds: `{result['seeds']}`", "", "## Chosen splits", "", "| Seed | Main | Projection | Readout | Pre-FT loss |", "|---:|---:|---:|---:|---:|"]
    for item in result["chosen_splits"]:
        lines.append(f"| `{item['seed']}` | `{item['main']:.2f}` | `{item['projection']:.2f}` | `{item['readout']:.2f}` | `{item['loss']:.4f}` |")
    lines.extend(["", "## Results", "", "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        item = result["summary"][method]
        if method == "magnitude":
            delta = "baseline"
            wins = "baseline"
        else:
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Interpretation", "", "This transfers the diversity route optimizer from the original TinyResNet to a deeper residual model with two blocks per stage."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    chosen = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.loaders(seed)
        model = DeepTinyResNet().to(base.DEVICE)
        train_model(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        mag_mask = base.global_mask(mag, SPARSITY)
        syn_mask = base.global_mask(syn, SPARSITY)
        reserve_mask = base.capacity_mask(syn, mag, SPARSITY, RESERVE)
        template_quality = route_quality(mag_mask)
        reserve_quality = route_quality(reserve_mask)
        best = optimize_split(syn, mag, template_quality, reserve_quality)
        chosen.append({"seed": seed, "main": float(best["split"].main), "projection": float(best["split"].projection), "readout": float(best["split"].readout), "loss": float(best["loss"])})
        masks_by_label = {"magnitude": mag_mask, "global_synflow": syn_mask, "reserve_0.60": reserve_mask, "tuned_40_35_25": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, TUNED_SPLIT), "diversity_target_optimizer": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, best["split"])}
        for label, masks in masks_by_label.items():
            before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = route_quality(masks)
            rows.append({"seed": seed, "method": label, "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "route_quality": quality})
            print(f"seed {seed} {label}: after={after:.4f} split={best['split'] if label == 'diversity_target_optimizer' else ''} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {"experiment": "04_cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct", "setup": "CIFAR-10 DeepTinyResNet transfer test for diversity-penalized route-capacity pruning at 99% sparsity.", "device": base.DEVICE, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsity": SPARSITY, "reserve": RESERVE, "chosen_splits": chosen, "summary": summary, "paired_deltas": paired, "rows": rows}
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"chosen_splits": result["chosen_splits"], "summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
