from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cifar10_tiny_resnet_capacity_transfer as base
import cifar10_tiny_resnet_balanced_route_capacity as balanced
import cifar10_tiny_resnet_route_quality_audit as audit

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [203, 204, 205, 206]
SPARSITY = 0.99
RESERVE = 0.60
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "balanced_route_0.60"]


def summarize(rows):
    summary = {}
    paired = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {
            "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
            "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
            "after_std": float(np.std([r["after_accuracy"] for r in selected])),
            "route_min_mean": float(np.mean([r["route_quality"]["route_min"] for r in selected])),
            "projection_min_mean": float(np.mean([r["route_quality"]["projection_min"] for r in selected])),
            "fc_score_mean": float(np.mean([r["route_quality"]["fc_score"] for r in selected])),
            "dead_outputs_mean": float(np.mean([r["route_quality"]["total_dead_outputs"] for r in selected])),
        }
        if method != "magnitude":
            deltas = []
            for seed in SEEDS:
                mag = next(r for r in rows if r["seed"] == seed and r["method"] == "magnitude")
                alt = next(r for r in rows if r["seed"] == seed and r["method"] == method)
                deltas.append({"seed": seed, "before_delta": alt["before_accuracy"] - mag["before_accuracy"], "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
            paired.append({
                "method": method,
                "before_delta_mean": float(np.mean([d["before_delta"] for d in deltas])),
                "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                "after_delta_std": float(np.std([d["after_delta"] for d in deltas])),
                "before_wins": int(sum(d["before_delta"] > 0 for d in deltas)),
                "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                "paired_rows": deltas,
            })
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_balanced_route_99pct_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_BALANCED_ROUTE_99PCT_REPLICATE.md"
    lines = [
        "# CIFAR-10 TinyResNet Balanced Route 99% Replicate",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | Before FT | After FT | After std | Route min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        lines.append(f"| `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | `{item['route_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Method | Before delta | Before wins | After delta | After wins |", "|---|---:|---:|---:|---:|"])
    for item in result["paired_deltas"]:
        lines.append(f"| `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(SEEDS)}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This replicate tests whether the two-seed TinyResNet `99%` balanced-route gain survives fresh seeds.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.loaders(seed)
        model = base.TinyResNet().to(base.DEVICE)
        base.train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        masks_by_label = {
            "magnitude": base.global_mask(mag, SPARSITY),
            "global_synflow": base.global_mask(syn, SPARSITY),
            "reserve_0.60": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "balanced_route_0.60": balanced.balanced_route_mask(syn, mag, SPARSITY, RESERVE),
        }
        for label, masks in masks_by_label.items():
            before, after = base.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = audit.route_quality(masks)
            rows.append({
                "seed": seed,
                "method": label,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "route_quality": quality,
            })
            print(f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_balanced_route_99pct_replicate",
        "setup": "Four fresh seed TinyResNet 99% sparsity replicate for balanced residual route-capacity pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
