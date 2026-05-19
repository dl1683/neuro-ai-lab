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
SPLITS = {
    "balanced_50_25_25": {"main": 0.50, "projection": 0.25, "readout": 0.25},
    "proj_heavy_45_35_20": {"main": 0.45, "projection": 0.35, "readout": 0.20},
    "proj_readout_40_35_25": {"main": 0.40, "projection": 0.35, "readout": 0.25},
}
METHODS = ["magnitude", "reserve_0.60"] + list(SPLITS.keys())


def mask_for_split(syn, mag, split):
    old = dict(balanced.GROUP_SPLIT)
    try:
        balanced.GROUP_SPLIT.clear()
        balanced.GROUP_SPLIT.update(split)
        return balanced.balanced_route_mask(syn, mag, SPARSITY, RESERVE)
    finally:
        balanced.GROUP_SPLIT.clear()
        balanced.GROUP_SPLIT.update(old)


def summarize(rows):
    summary = {}
    paired = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {
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
                mag_row = next(r for r in rows if r["seed"] == seed and r["method"] == "magnitude")
                alt = next(r for r in rows if r["seed"] == seed and r["method"] == method)
                deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag_row["after_accuracy"]})
            paired.append({
                "method": method,
                "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                "after_delta_std": float(np.std([d["after_delta"] for d in deltas])),
                "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                "paired_rows": deltas,
            })
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_route_split_sweep_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_ROUTE_SPLIT_SWEEP_99PCT.md"
    lines = [
        "# CIFAR-10 TinyResNet Route Split Sweep at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Route min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = {item["method"]: item for item in result["paired_deltas"]}
    for method in METHODS:
        item = result["summary"][method]
        if method == "magnitude":
            delta = "baseline"
            wins = "baseline"
        else:
            delta = f"`{deltas[method]['after_delta_mean']:+.4f}`"
            wins = f"`{deltas[method]['after_wins']}/{len(SEEDS)}`"
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['route_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a targeted projection/readout tradeoff test. The previous balanced split improved readout but still trailed magnitude, so this sweep increases projection share while preserving readout capacity.",
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
            "reserve_0.60": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
        }
        for label, split in SPLITS.items():
            masks_by_label[label] = mask_for_split(syn, mag, split)
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
        "experiment": "04_cifar10_tiny_resnet_route_split_sweep_99pct",
        "setup": "Four-seed targeted TinyResNet 99% projection/readout split sweep for balanced route-capacity pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "splits": SPLITS,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
