from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct as deep
import cifar10_tiny_resnet_capacity_transfer as base
from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask

SEEDS = [221, 222, 223, 224]
SPARSITY = 0.99
RESERVE = 0.60
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "tuned_40_35_25", "diversity_target_optimizer"]


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
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_DEEP_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT_REPLICATE.md"
    lines = ["# CIFAR-10 DeepTinyResNet Diversity Route Optimizer 99% Replicate", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", f"Seeds: `{result['seeds']}`", "", "## Chosen splits", "", "| Seed | Main | Projection | Readout | Pre-FT loss |", "|---:|---:|---:|---:|---:|"]
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
    lines.extend(["", "## Interpretation", "", "This is a fresh four-seed replicate for the deeper residual transfer result."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    chosen = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.loaders(seed)
        model = deep.DeepTinyResNet().to(base.DEVICE)
        deep.train_model(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        mag_mask = base.global_mask(mag, SPARSITY)
        syn_mask = base.global_mask(syn, SPARSITY)
        reserve_mask = base.capacity_mask(syn, mag, SPARSITY, RESERVE)
        template_quality = deep.route_quality(mag_mask)
        reserve_quality = deep.route_quality(reserve_mask)
        best = deep.optimize_split(syn, mag, template_quality, reserve_quality)
        chosen.append({"seed": seed, "main": float(best["split"].main), "projection": float(best["split"].projection), "readout": float(best["split"].readout), "loss": float(best["loss"])})
        masks_by_label = {"magnitude": mag_mask, "global_synflow": syn_mask, "reserve_0.60": reserve_mask, "tuned_40_35_25": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, TUNED_SPLIT), "diversity_target_optimizer": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, best["split"])}
        for label, masks in masks_by_label.items():
            before, after = deep.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = deep.route_quality(masks)
            rows.append({"seed": seed, "method": label, "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "route_quality": quality})
            print(f"seed {seed} {label}: after={after:.4f} split={best['split'] if label == 'diversity_target_optimizer' else ''} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {"experiment": "04_cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate", "setup": "Fresh four-seed DeepTinyResNet 99% replicate for diversity-penalized route-capacity pruning.", "device": base.DEVICE, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsity": SPARSITY, "reserve": RESERVE, "chosen_splits": chosen, "summary": summary, "paired_deltas": paired, "rows": rows}
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"chosen_splits": result["chosen_splits"], "summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
