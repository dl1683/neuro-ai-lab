from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cifar10_tiny_resnet_capacity_transfer as base
import cifar10_tiny_resnet_route_quality_audit as audit

ROOT = Path(__file__).resolve().parents[2]
RESERVE = 0.60
GROUP_SPLIT = {"main": 0.50, "projection": 0.25, "readout": 0.25}
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "balanced_route_0.60"]


def group_name(name: str) -> str:
    if name == "fc.weight" or name.endswith("fc.weight"):
        return "readout"
    if ".shortcut.0.weight" in name:
        return "projection"
    return "main"


def rank_for(name: str, synflow, magnitude):
    if group_name(name) in {"readout", "projection"}:
        return magnitude[name]
    return synflow[name]


def select_group_budget(names, group_budget, synflow, magnitude, masks):
    if not names or group_budget <= 0:
        return
    output_weights = {name: base.output_units(synflow[name]) for name in names}
    total_weight = sum(output_weights.values())
    for name in names:
        budget = max(base.output_units(synflow[name]), int(round(group_budget * output_weights[name] / max(1, total_weight))))
        budget = min(budget, synflow[name].numel())
        masks[name] = torch.maximum(masks[name], base.topk(rank_for(name, synflow, magnitude), budget))


def enforce_liveness(names, synflow, magnitude, masks, protected):
    for name in names:
        rank = rank_for(name, synflow, magnitude)
        layer_mask = masks[name].clone()
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_rank = rank.reshape(rank.shape[0], -1)
        flat_protected = protected[name].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_rank[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[name] = flat_outputs.reshape_as(layer_mask)
        protected[name] = flat_protected.reshape_as(layer_mask)


def exact_budget(masks, protected, synflow, magnitude, total_keep):
    selected = int(sum(m.sum().item() for m in masks.values()))
    remaining = total_keep - selected
    if remaining > 0:
        chunks = []
        values = []
        for name, score in synflow.items():
            available = masks[name].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                vals = score.flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining, flat_values.numel()), largest=True).values.min()
        left = remaining
        for name, idx, vals in chunks:
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 1
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    elif remaining < 0:
        excess = -remaining
        chunks = []
        values = []
        for name in synflow:
            removable = masks[name].bool() & ~protected[name]
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = rank_for(name, synflow, magnitude).flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, excess, largest=False).values.max()
        left = excess
        for name, idx, vals in chunks:
            chosen = idx[vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 0
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    return masks


def balanced_route_mask(synflow, magnitude, sparsity: float, reserve: float):
    total_params = sum(v.numel() for v in synflow.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    reserve_keep = int(round(reserve * total_keep))
    critical = [name for name in synflow if name != "stem.weight"]
    masks = {name: torch.zeros_like(score) for name, score in synflow.items()}
    protected = {name: torch.zeros_like(score, dtype=torch.bool) for name, score in synflow.items()}
    groups = {group: [name for name in critical if group_name(name) == group] for group in GROUP_SPLIT}

    for group, fraction in GROUP_SPLIT.items():
        select_group_budget(groups[group], int(round(reserve_keep * fraction)), synflow, magnitude, masks)
    enforce_liveness(critical, synflow, magnitude, masks, protected)
    return exact_budget(masks, protected, synflow, magnitude, total_keep)


def summarize(rows):
    summary = {}
    paired = []
    for sparsity in base.SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {
                "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "route_min_mean": float(np.mean([r["route_quality"]["route_min"] for r in selected])),
                "projection_min_mean": float(np.mean([r["route_quality"]["projection_min"] for r in selected])),
                "fc_score_mean": float(np.mean([r["route_quality"]["fc_score"] for r in selected])),
                "dead_outputs_mean": float(np.mean([r["route_quality"]["total_dead_outputs"] for r in selected])),
            }
            if method != "magnitude":
                deltas = []
                for seed in base.SEEDS:
                    mag = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
                    alt = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == method)
                    deltas.append({"seed": seed, "before_delta": alt["before_accuracy"] - mag["before_accuracy"], "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
                paired.append({
                    "sparsity": sparsity,
                    "method": method,
                    "before_delta_mean": float(np.mean([d["before_delta"] for d in deltas])),
                    "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                    "before_wins": int(sum(d["before_delta"] > 0 for d in deltas)),
                    "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                    "paired_rows": deltas,
                })
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_balanced_route_capacity.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_BALANCED_ROUTE_CAPACITY.md"
    lines = [
        "# CIFAR-10 TinyResNet Balanced Route Capacity",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Protected split: `{result['group_split']}`",
        "",
        "| Sparsity | Method | Before FT | After FT | Route min | Projection min | FC score | Dead outputs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sparsity in base.SPARSITIES:
        for method in METHODS:
            item = result["summary"][str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['route_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Method | Before delta | Before wins | After delta | After wins |", "|---:|---|---:|---:|---:|---:|"])
    for item in result["paired_deltas"]:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(base.SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(base.SEEDS)}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This tests whether residual recovery improves when protected capacity is explicitly balanced across main transformations, projection shortcuts, and classifier readout.",
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
        for sparsity in base.SPARSITIES:
            masks_by_label = {
                "magnitude": base.global_mask(mag, sparsity),
                "global_synflow": base.global_mask(syn, sparsity),
                "reserve_0.60": base.capacity_mask(syn, mag, sparsity, RESERVE),
                "balanced_route_0.60": balanced_route_mask(syn, mag, sparsity, RESERVE),
            }
            for label, masks in masks_by_label.items():
                before, after = base.eval_method(model, dense_state, train_loader, test_loader, masks)
                quality = audit.route_quality(masks)
                rows.append({
                    "seed": seed,
                    "sparsity": sparsity,
                    "method": label,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "route_quality": quality,
                })
                print(f"seed {seed} sparsity {sparsity:.2f} {label}: after={after:.4f} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_balanced_route_capacity",
        "setup": "CIFAR-10 TinyResNet test of balanced residual route-capacity pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": base.SEEDS,
        "sparsities": base.SPARSITIES,
        "reserve": RESERVE,
        "group_split": GROUP_SPLIT,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
