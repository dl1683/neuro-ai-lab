from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

import cifar10_tiny_resnet_capacity_transfer as base
import tinyimagenet_resnet20_ecology_selector_99pct as tiny
from shared.circuit_viability_selector import choose_ecology_aware_method, split_dict
from shared.residual_route_capacity import route_split_capacity_mask


SEED = 272
TRAIN_N = 12000
VAL_N = 3000
BATCH = 64
SPARSITY = 0.99
RESERVE = 0.60
DENSE_EPOCHS = 4
FT_EPOCHS = 2
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_policy"]


def loaders(seed: int):
    tiny.ensure_data()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_ds = tiny.TinyImageNet(tiny.TINY_ROOT, "train", train_tf)
    val_ds = tiny.TinyImageNet(tiny.TINY_ROOT, "val", eval_tf)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_ds))[:TRAIN_N]
    val_idx = np.arange(min(VAL_N, len(val_ds)))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(Subset(train_ds, train_idx), batch_size=BATCH, shuffle=True, num_workers=2, generator=generator),
        DataLoader(Subset(val_ds, val_idx), batch_size=128, shuffle=False, num_workers=2),
    )


def make_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 200)
    return model.to(base.DEVICE)


def train_model(model, loader):
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 2e-5},
            {"params": model.fc.parameters(), "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    for epoch in range(DENSE_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"  dense_epoch={epoch + 1}/{DENSE_EPOCHS}", flush=True)


def masked_finetune(model, loader, masks):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
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


def evaluate_method(model, dense_state, train_loader, val_loader, masks):
    before = base.evaluate(model, val_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune(model, train_loader, masks)
    after = base.evaluate(model, val_loader)
    model.load_state_dict(dense_state)
    return before, after


def synflow_scores_224(model):
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 3, 224, 224, device=base.DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in base.params(model).items()}
    for name, p in model.named_parameters():
        p.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def route_quality(masks):
    return tiny.r20.route_quality(masks)


def summarize(rows):
    summary = {}
    mag_after = next(row["after_accuracy"] for row in rows if row["method"] == "magnitude")
    for method in METHODS:
        row = next(row for row in rows if row["method"] == method)
        item = {
            "after_mean": row["after_accuracy"],
            "projection_min_mean": row["route_quality"]["projection_min"],
            "fc_score_mean": row["route_quality"]["fc_score"],
            "main_path_min_mean": row["route_quality"]["main_path_min"],
            "dead_outputs_mean": row["route_quality"]["total_dead_outputs"],
        }
        if method != "magnitude":
            item["after_delta_mean"] = row["after_accuracy"] - mag_after
            item["after_wins"] = int(row["after_accuracy"] > mag_after)
        summary[method] = item
    return summary


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_ECOLOGY_SELECTOR_99PCT.md"
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Ecology Selector at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seed: `{result['seed']}`",
        f"Train subset: `{TRAIN_N}`; validation subset: `{VAL_N}`",
        f"Dense epochs: `{DENSE_EPOCHS}`; masked fine-tune epochs: `{FT_EPOCHS}`",
        f"Dense accuracy: `{result['dense_accuracy']:.4f}`",
        "",
        "| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        delta = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected method: `{result['decision']['selected_method']}`",
            f"Readout ratio: `{result['decision']['plain_readout_ratio']:.4f}`",
            f"Selected split: `{result['decision']['selected_split']}`",
            "",
            "## Interpretation",
            "",
            "This upgrades the TinyImageNet proxy to an ImageNet-pretrained ResNet-18 and keeps the selector threshold fixed. It tests whether the boundary condition was caused by weak dense training rather than the viability idea itself.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    print(f"tinyimagenet pretrained resnet18 seed {SEED}: train dense", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = loaders(SEED)
    model = make_model()
    train_model(model, train_loader)
    dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dense_accuracy = base.evaluate(model, val_loader)
    print(f"tinyimagenet pretrained resnet18 seed {SEED}: dense_accuracy={dense_accuracy:.4f}", flush=True)
    mag = base.magnitude_scores(model)
    syn = synflow_scores_224(model)
    decision = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, route_quality)
    split = decision["best_split"]["split"]
    method_masks = {
        "magnitude": base.global_mask(mag, SPARSITY),
        "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
        "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split, skip_names=set()),
    }
    print(
        f"tinyimagenet pretrained resnet18 seed {SEED}: selected={decision['selected_method']} "
        f"readout_ratio={decision['plain_readout_ratio']:.4f} best_split={split_dict(split)}",
        flush=True,
    )
    rows = []
    evaluated = {}
    for label, masks in method_masks.items():
        before, after = evaluate_method(model, dense_state, train_loader, val_loader, masks)
        quality = route_quality(masks)
        row = {
            "seed": SEED,
            "method": label,
            "dense_accuracy": dense_accuracy,
            "before_accuracy": before,
            "after_accuracy": after,
            "route_quality": quality,
        }
        rows.append(row)
        evaluated[label] = row
        print(
            f"tinyimagenet pretrained {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
            f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
            flush=True,
        )
    policy_source = decision["selected_method"]
    policy_row = dict(evaluated[policy_source])
    policy_row["method"] = "ecology_policy"
    policy_row["policy_source_method"] = policy_source
    rows.append(policy_row)
    result = {
        "experiment": "04_tinyimagenet_resnet18_pretrained_ecology_selector_99pct",
        "setup": "TinyImageNet-200 external-proxy subset stress test using an ImageNet-pretrained ResNet-18 adapted to 200 classes, 99% sparsity, and the fixed ecology-aware selector.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": SEED,
        "train_subset": TRAIN_N,
        "val_subset": VAL_N,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "dense_epochs": DENSE_EPOCHS,
        "finetune_epochs": FT_EPOCHS,
        "dense_accuracy": dense_accuracy,
        "summary": summarize(rows),
        "decision": {
            "selected_method": decision["selected_method"],
            "selected_split": decision["selected_split"],
            "plain_readout_ratio": decision["plain_readout_ratio"],
            "readout_ratio_threshold": decision["readout_ratio_threshold"],
            "best_split": split_dict(split),
            "plain_quality": decision["plain_quality"],
            "magnitude_quality": decision["magnitude_quality"],
            "best_split_quality": decision["best_split"]["quality"],
        },
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dense_accuracy": result["dense_accuracy"], "summary": result["summary"], "decision": result["decision"]}, indent=2))
