from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

import cifar10_full_resnet20_capacity_99pct as full
import cifar10_resnet20_capacity_99pct as r20
import cifar10_tiny_resnet_capacity_transfer as base
from shared.circuit_viability_selector import choose_ecology_aware_method, split_dict
from shared.residual_route_capacity import route_split_capacity_mask


SEEDS = [269, 270]
SPARSITY = 0.99
RESERVE = 0.60
DENSE_EPOCHS = 40
FT_EPOCHS = 8
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_policy"]


def train_sgd40(model, loader):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DENSE_EPOCHS)
    for epoch in range(DENSE_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
        if epoch in {9, 19, 39}:
            print(f"  dense_epoch={epoch + 1}/{DENSE_EPOCHS}", flush=True)


def masked_finetune_sgd(model, loader, masks):
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
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
        scheduler.step()


def eval_method(model, dense_state, train_loader, test_loader, masks):
    before = base.evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune_sgd(model, train_loader, masks)
    after = base.evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def summarize(rows):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected])),
            "projection_min_mean": float(np.mean([row["route_quality"]["projection_min"] for row in selected])),
            "fc_score_mean": float(np.mean([row["route_quality"]["fc_score"] for row in selected])),
            "main_path_min_mean": float(np.mean([row["route_quality"]["main_path_min"] for row in selected])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
        }
    paired = []
    for method in METHODS:
        if method == "magnitude":
            continue
        deltas = []
        for seed in SEEDS:
            mag = next(row for row in rows if row["seed"] == seed and row["method"] == "magnitude")
            alt = next(row for row in rows if row["seed"] == seed and row["method"] == method)
            deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
        summary[method]["after_delta_mean"] = float(np.mean([row["after_delta"] for row in deltas]))
        summary[method]["after_wins"] = int(sum(row["after_delta"] > 0 for row in deltas))
        paired.append(
            {
                "method": method,
                "after_delta_mean": float(np.mean([row["after_delta"] for row in deltas])),
                "after_delta_std": float(np.std([row["after_delta"] for row in deltas])),
                "after_wins": int(sum(row["after_delta"] > 0 for row in deltas)),
                "paired_rows": deltas,
            }
        )
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_FULL_RESNET20_ECOLOGY_SELECTOR_99PCT_SGD40.md"
    lines = [
        "# CIFAR-10 Full ResNet-20 Ecology Selector at 99%: SGD-40 Stress Test",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Dense epochs: `{DENSE_EPOCHS}`; masked fine-tune epochs: `{FT_EPOCHS}`",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Decisions", ""])
    for item in result["decisions"]:
        lines.append(
            f"- seed `{item['seed']}`: selected `{item['selected_method']}` "
            f"readout_ratio `{item['plain_readout_ratio']:.4f}` split `{item['selected_split']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This stress test doubles dense training relative to the 20-epoch recipe and keeps the ecology selector threshold fixed. It is not a canonical 160-epoch CIFAR schedule, but it checks whether the selector survives a better-trained dense model without retuning.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    decisions = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense full CIFAR-10 SGD40", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = full.full_loaders(seed)
        model = r20.CifarResNet20().to(base.DEVICE)
        train_sgd40(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        decision = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, r20.route_quality)
        split = decision["best_split"]["split"]
        method_masks = {
            "magnitude": base.global_mask(mag, SPARSITY),
            "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split),
        }
        decisions.append(
            {
                "seed": seed,
                "selected_method": decision["selected_method"],
                "selected_split": decision["selected_split"],
                "plain_readout_ratio": decision["plain_readout_ratio"],
                "readout_ratio_threshold": decision["readout_ratio_threshold"],
                "best_split": split_dict(split),
                "plain_quality": decision["plain_quality"],
                "magnitude_quality": decision["magnitude_quality"],
                "best_split_quality": decision["best_split"]["quality"],
            }
        )
        print(
            f"seed {seed}: selected={decision['selected_method']} "
            f"readout_ratio={decision['plain_readout_ratio']:.4f} best_split={split_dict(split)}",
            flush=True,
        )
        evaluated = {}
        for label, masks in method_masks.items():
            before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = r20.route_quality(masks)
            row = {
                "seed": seed,
                "method": label,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "route_quality": quality,
            }
            rows.append(row)
            evaluated[label] = row
            print(
                f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
                f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
                flush=True,
            )
        policy_source = decision["selected_method"]
        policy_row = dict(evaluated[policy_source])
        policy_row["method"] = "ecology_policy"
        policy_row["policy_source_method"] = policy_source
        rows.append(policy_row)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_full_resnet20_ecology_selector_99pct_sgd40",
        "setup": "Full CIFAR-10 ResNet-20-style 99% sparsity stress test with 40 dense SGD/cosine epochs, 8 masked fine-tune epochs, and the fixed ecology-aware selector.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "dense_epochs": DENSE_EPOCHS,
        "finetune_epochs": FT_EPOCHS,
        "summary": summary,
        "paired_deltas": paired,
        "decisions": decisions,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "decisions": result["decisions"]}, indent=2))
