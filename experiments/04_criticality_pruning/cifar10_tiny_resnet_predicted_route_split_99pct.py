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
TUNED_SPLIT = {"main": 0.40, "projection": 0.35, "readout": 0.25}
METHODS = ["magnitude", "reserve_0.60", "tuned_40_35_25", "predicted_deficit_split"]


def with_split_mask(syn, mag, split):
    old = dict(balanced.GROUP_SPLIT)
    try:
        balanced.GROUP_SPLIT.clear()
        balanced.GROUP_SPLIT.update(split)
        return balanced.balanced_route_mask(syn, mag, SPARSITY, RESERVE)
    finally:
        balanced.GROUP_SPLIT.clear()
        balanced.GROUP_SPLIT.update(old)


def predicted_split(mag_mask, reserve_mask):
    template = audit.route_quality(mag_mask)
    candidate = audit.route_quality(reserve_mask)
    projection_deficit = max(1e-6, template["projection_min"] - candidate["projection_min"])
    readout_deficit = max(1e-6, template["fc_score"] - candidate["fc_score"])

    # The route audit showed projection capacity is the strongest residual-99 correlate,
    # while plain reserve mainly starves readout. Keep a main-path floor, then allocate
    # the remaining reserve from measured projection/readout deficits with a projection
    # reliability weight. This predicts the route-family split before fine-tuning.
    main = 0.40
    projection_signal = 2.0 * projection_deficit
    readout_signal = readout_deficit
    projection = (1.0 - main) * projection_signal / (projection_signal + readout_signal)
    readout = 1.0 - main - projection

    projection = min(0.40, max(0.20, projection))
    readout = min(0.35, max(0.20, readout))
    main = 1.0 - projection - readout
    return {
        "main": float(main),
        "projection": float(projection),
        "readout": float(readout),
        "projection_deficit": float(projection_deficit),
        "readout_deficit": float(readout_deficit),
        "template_projection_min": float(template["projection_min"]),
        "candidate_projection_min": float(candidate["projection_min"]),
        "template_fc_score": float(template["fc_score"]),
        "candidate_fc_score": float(candidate["fc_score"]),
    }


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
            summary[method]["after_delta_mean"] = float(np.mean([d["after_delta"] for d in deltas]))
            summary[method]["after_wins"] = int(sum(d["after_delta"] > 0 for d in deltas))
            paired.append({
                "method": method,
                "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                "after_delta_std": float(np.std([d["after_delta"] for d in deltas])),
                "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                "paired_rows": deltas,
            })
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_predicted_route_split_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_PREDICTED_ROUTE_SPLIT_99PCT.md"
    lines = [
        "# CIFAR-10 TinyResNet Predicted Route Split at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "## Predicted splits",
        "",
        "| Seed | Main | Projection | Readout | Projection deficit | Readout deficit |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["predicted_splits"]:
        lines.append(f"| `{item['seed']}` | `{item['main']:.3f}` | `{item['projection']:.3f}` | `{item['readout']:.3f}` | `{item['projection_deficit']:.4f}` | `{item['readout_deficit']:.4f}` |")
    lines.extend([
        "",
        "## Results",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Route min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        item = result["summary"][method]
        if method == "magnitude":
            delta = "baseline"
            wins = "baseline"
        else:
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['route_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This tests whether the route-family split can be derived from pre-finetune route deficits instead of swept by hand. The split compares a magnitude viability template to the plain reserve candidate, then allocates protected capacity across projection and readout deficits with a main-path floor.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    predicted = []
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
        mag_mask = base.global_mask(mag, SPARSITY)
        reserve_mask = base.capacity_mask(syn, mag, SPARSITY, RESERVE)
        split_info = predicted_split(mag_mask, reserve_mask)
        predicted.append({"seed": seed, **split_info})
        split = {"main": split_info["main"], "projection": split_info["projection"], "readout": split_info["readout"]}
        masks_by_label = {
            "magnitude": mag_mask,
            "reserve_0.60": reserve_mask,
            "tuned_40_35_25": with_split_mask(syn, mag, TUNED_SPLIT),
            "predicted_deficit_split": with_split_mask(syn, mag, split),
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
            print(f"seed {seed} {label}: after={after:.4f} split={split if label == 'predicted_deficit_split' else ''} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_predicted_route_split_99pct",
        "setup": "Four-seed TinyResNet 99% test of route-deficit-predicted capacity splits.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "tuned_split": TUNED_SPLIT,
        "predicted_splits": predicted,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"predicted_splits": result["predicted_splits"], "summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
