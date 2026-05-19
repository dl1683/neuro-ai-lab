from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import cifar10_tiny_resnet_capacity_transfer as base
import cifar10_tiny_resnet_activation_capacity as activation
import cifar10_tiny_resnet_backbone_capacity as backbone

ROOT = Path(__file__).resolve().parents[2]
RESERVE = 0.60
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "activation_reserve_0.60", "backbone_reserve_0.60"]
BLOCKS = ["b1", "b2", "b3"]


def output_profile(mask: torch.Tensor) -> dict[str, float | int]:
    flat = mask.detach().reshape(mask.shape[0], -1).float()
    fanin = flat.sum(dim=1)
    alive = fanin > 0
    mean_fanin = float(fanin.mean().item())
    live_fraction = float(alive.float().mean().item())
    score = live_fraction * math.log1p(mean_fanin)
    return {
        "keep_rate": float(flat.mean().item()),
        "dead_outputs": int((~alive).sum().item()),
        "outputs": int(flat.shape[0]),
        "live_fraction": live_fraction,
        "mean_fanin": mean_fanin,
        "min_fanin": float(fanin.min().item()),
        "score": score,
    }


def get_score(layer_profiles: dict[str, dict], name: str, default: float | None = None) -> float:
    if name in layer_profiles:
        return float(layer_profiles[name]["score"])
    if default is None:
        raise KeyError(name)
    return default


def route_quality(masks: dict[str, torch.Tensor]) -> dict:
    layer_profiles = {name: output_profile(mask) for name, mask in masks.items()}
    block_scores = {}
    projection_scores = []
    main_scores = []
    for block in BLOCKS:
        conv1 = get_score(layer_profiles, f"{block}.conv1.weight")
        conv2 = get_score(layer_profiles, f"{block}.conv2.weight")
        main = min(conv1, conv2)
        shortcut_name = f"{block}.shortcut.0.weight"
        shortcut = get_score(layer_profiles, shortcut_name, 1.0)
        block_score = max(main, shortcut) if shortcut_name in layer_profiles else main
        block_scores[block] = {
            "conv1_score": conv1,
            "conv2_score": conv2,
            "main_score": main,
            "shortcut_score": shortcut,
            "block_score": block_score,
            "main_shortcut_balance": main / max(shortcut, 1e-8) if shortcut_name in layer_profiles else 1.0,
        }
        main_scores.append(main)
        if shortcut_name in layer_profiles:
            projection_scores.append(shortcut)

    route_components = [get_score(layer_profiles, "stem.weight")] + [block_scores[b]["block_score"] for b in BLOCKS] + [get_score(layer_profiles, "fc.weight")]
    strict_components = [get_score(layer_profiles, "stem.weight")] + [block_scores[b]["main_score"] for b in BLOCKS] + [get_score(layer_profiles, "fc.weight")]
    projection_min = min(projection_scores) if projection_scores else 1.0
    route_min = min(route_components)
    strict_route_min = min(strict_components)
    route_mean = float(np.mean(route_components))
    route_balance = route_min / max(route_mean, 1e-8)
    total_dead_outputs = int(sum(int(p["dead_outputs"]) for p in layer_profiles.values()))
    weighted_dead_outputs = float(sum(float(p["dead_outputs"]) / max(1, int(p["outputs"])) for p in layer_profiles.values()))
    return {
        "layer_profiles": layer_profiles,
        "block_scores": block_scores,
        "route_min": route_min,
        "strict_route_min": strict_route_min,
        "route_mean": route_mean,
        "route_balance": route_balance,
        "projection_min": projection_min,
        "fc_score": get_score(layer_profiles, "fc.weight"),
        "main_path_min": min(main_scores),
        "total_dead_outputs": total_dead_outputs,
        "weighted_dead_outputs": weighted_dead_outputs,
    }


def pearson(xs, ys) -> float | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(rows):
    metrics = [
        "route_min",
        "strict_route_min",
        "route_mean",
        "route_balance",
        "projection_min",
        "fc_score",
        "main_path_min",
        "total_dead_outputs",
        "weighted_dead_outputs",
    ]
    out = {}
    for sparsity in base.SPARSITIES:
        selected = [r for r in rows if r["sparsity"] == sparsity]
        ys = [r["after_accuracy"] for r in selected]
        out[str(sparsity)] = {metric: pearson([r["route_quality"][metric] for r in selected], ys) for metric in metrics}
    ys = [r["after_accuracy"] for r in rows]
    out["all"] = {metric: pearson([r["route_quality"][metric] for r in rows], ys) for metric in metrics}
    return out


def summarize(rows):
    summary = {}
    for sparsity in base.SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "route_min_mean": float(np.mean([r["route_quality"]["route_min"] for r in selected])),
                "strict_route_min_mean": float(np.mean([r["route_quality"]["strict_route_min"] for r in selected])),
                "route_balance_mean": float(np.mean([r["route_quality"]["route_balance"] for r in selected])),
                "projection_min_mean": float(np.mean([r["route_quality"]["projection_min"] for r in selected])),
                "fc_score_mean": float(np.mean([r["route_quality"]["fc_score"] for r in selected])),
                "dead_outputs_mean": float(np.mean([r["route_quality"]["total_dead_outputs"] for r in selected])),
            }
    return summary


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_route_quality_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_ROUTE_QUALITY_AUDIT.md"
    lines = [
        "# CIFAR-10 TinyResNet Route-Quality Audit",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        "",
        "## Summary by method",
        "",
        "| Sparsity | Method | After FT | Route min | Strict route min | Route balance | Projection min | FC score | Dead outputs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sparsity in base.SPARSITIES:
        for method in METHODS:
            item = result["summary"][str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['after_mean']:.4f}` | `{item['route_min_mean']:.4f}` | `{item['strict_route_min_mean']:.4f}` | `{item['route_balance_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Metric correlations with after-FT accuracy", "", "| Scope | Route min | Strict route min | Route balance | Projection min | FC score | Dead outputs |", "|---|---:|---:|---:|---:|---:|---:|"])
    for scope, corr in result["correlations"].items():
        def fmt(value):
            return "" if value is None else f"`{value:+.3f}`"
        lines.append(f"| `{scope}` | {fmt(corr['route_min'])} | {fmt(corr['strict_route_min'])} | {fmt(corr['route_balance'])} | {fmt(corr['projection_min'])} | {fmt(corr['fc_score'])} | {fmt(corr['total_dead_outputs'])} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This audit tests whether route-quality features explain recoverability better than total dead-output count. It treats each residual block as a composed route with a main path and, when present, a projection shortcut.",
        "",
        "A useful next pruning method should optimize the route-quality metric that tracks after-FT accuracy, not merely keep every output unit alive.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    for seed in base.SEEDS:
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
        act_rank = activation.activation_supported_scores(model, train_loader, syn, mag)
        for sparsity in base.SPARSITIES:
            masks_by_label = {
                "magnitude": base.global_mask(mag, sparsity),
                "global_synflow": base.global_mask(syn, sparsity),
                "reserve_0.60": base.capacity_mask(syn, mag, sparsity, RESERVE),
                "activation_reserve_0.60": activation.capacity_mask_with_rank(syn, mag, act_rank, sparsity, RESERVE),
                "backbone_reserve_0.60": backbone.backbone_capacity_mask(syn, mag, sparsity, RESERVE),
            }
            for label, masks in masks_by_label.items():
                before, after = base.eval_method(model, dense_state, train_loader, test_loader, masks)
                quality = route_quality(masks)
                rows.append({
                    "seed": seed,
                    "sparsity": sparsity,
                    "method": label,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "route_quality": quality,
                })
                print(f"seed {seed} sparsity {sparsity:.2f} {label}: after={after:.4f} route_min={quality['route_min']:.4f} strict={quality['strict_route_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    result = {
        "experiment": "04_cifar10_tiny_resnet_route_quality_audit",
        "setup": "CIFAR-10 TinyResNet audit of route-quality metrics for residual path-capacity pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": base.SEEDS,
        "sparsities": base.SPARSITIES,
        "methods": METHODS,
        "summary": summarize(rows),
        "correlations": correlations(rows),
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "correlations": result["correlations"]}, indent=2))
