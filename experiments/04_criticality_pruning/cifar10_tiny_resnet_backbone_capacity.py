from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cifar10_tiny_resnet_capacity_transfer as base

ROOT = Path(__file__).resolve().parents[2]
RESERVE = 0.60
METHODS = ["magnitude", "global_synflow", "reserve_0.60", "backbone_reserve_0.60"]


def rank_for(name: str, synflow, magnitude):
    if name.endswith("fc.weight") or name == "fc.weight" or ".shortcut.0.weight" in name:
        return magnitude[name]
    return synflow[name]


def cut_weight(name: str, score) -> int:
    units = base.output_units(score)
    if ".shortcut.0.weight" in name:
        return units * 4
    if name.endswith("fc.weight") or name == "fc.weight":
        return units * 2
    return units


def backbone_capacity_mask(synflow, magnitude, sparsity: float, reserve: float):
    total_params = sum(v.numel() for v in synflow.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    reserve_keep = int(round(reserve * total_keep))
    critical = [k for k in synflow if k != "stem.weight"]
    weights = {k: cut_weight(k, synflow[k]) for k in critical}
    total_weight = sum(weights.values())
    masks = {k: torch.zeros_like(v) for k, v in synflow.items()}
    protected = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in synflow.items()}

    for k in critical:
        budget = max(base.output_units(synflow[k]), int(round(reserve_keep * weights[k] / total_weight)))
        masks[k] = torch.maximum(masks[k], base.topk(rank_for(k, synflow, magnitude), budget))

    for k in critical:
        layer_mask = masks[k].clone()
        rank = rank_for(k, synflow, magnitude)
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_rank = rank.reshape(rank.shape[0], -1)
        flat_protected = protected[k].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_rank[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[k] = flat_outputs.reshape_as(layer_mask)
        protected[k] = flat_protected.reshape_as(layer_mask)

    selected = int(sum(m.sum().item() for m in masks.values()))
    remaining = total_keep - selected
    if remaining > 0:
        chunks = []
        values = []
        for k, score in synflow.items():
            available = masks[k].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                vals = score.flatten()[idx]
                chunks.append((k, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining, flat_values.numel()), largest=True).values.min()
        left = remaining
        for k, idx, vals in chunks:
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[k].flatten()
            flat[chosen] = 1
            masks[k] = flat.reshape_as(masks[k])
            left -= int(chosen.numel())
            if left <= 0:
                break
    elif remaining < 0:
        excess = -remaining
        chunks = []
        values = []
        for k in synflow:
            removable = masks[k].bool() & ~protected[k]
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = rank_for(k, synflow, magnitude).flatten()[idx]
                chunks.append((k, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, excess, largest=False).values.max()
        left = excess
        for k, idx, vals in chunks:
            chosen = idx[vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[k].flatten()
            flat[chosen] = 0
            masks[k] = flat.reshape_as(masks[k])
            left -= int(chosen.numel())
            if left <= 0:
                break
    return masks


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
                "dead_outputs_mean": float(np.mean([sum(v["dead_outputs"] for v in r["damage"].values()) for r in selected])),
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
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_backbone_capacity.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_BACKBONE_CAPACITY.md"
    lines = [
        "# CIFAR-10 TinyResNet Backbone Capacity",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        "",
        "| Sparsity | Method | Before FT | After FT | Dead outputs |",
        "|---:|---|---:|---:|---:|",
    ]
    for sparsity in base.SPARSITIES:
        for method in METHODS:
            item = result["summary"][str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Method | Before delta | Before wins | After delta | After wins |", "|---:|---|---:|---:|---:|---:|"])
    for item in result["paired_deltas"]:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(base.SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(base.SEEDS)}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This tests a residual-specific hypothesis: projection shortcuts and classifier routes are communication backbones, so they receive extra protected capacity and use magnitude ranking inside protected budgets.",
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
                "backbone_reserve_0.60": backbone_capacity_mask(syn, mag, sparsity, RESERVE),
            }
            for label, masks in masks_by_label.items():
                before, after = base.eval_method(model, dense_state, train_loader, test_loader, masks)
                dmg = base.damage(masks)
                rows.append({
                    "seed": seed,
                    "sparsity": sparsity,
                    "method": label,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "before_retention": before / dense_accuracy,
                    "after_retention": after / dense_accuracy,
                    "damage": dmg,
                })
                dead_total = sum(v["dead_outputs"] for v in dmg.values())
                print(f"seed {seed} sparsity {sparsity:.2f} {label}: before={before:.4f} after={after:.4f} dead_outputs={dead_total}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_backbone_capacity",
        "setup": "CIFAR-10 TinyResNet test of residual-backbone-aware path-capacity pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": base.SEEDS,
        "sparsities": base.SPARSITIES,
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
