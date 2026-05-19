from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cifar10_tiny_resnet_capacity_transfer as base
import cifar10_tiny_resnet_route_quality_audit as audit
from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask, predict_route_split

SEEDS = [211, 212, 213, 214]
SPARSITY = 0.99
RESERVE = 0.60
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "reserve_0.60", "tuned_40_35_25", "fixed_deficit_predictor", "target_matched_optimizer"]


def candidate_splits():
    values = [round(x, 2) for x in np.arange(0.20, 0.56, 0.05)]
    for main in values:
        for projection in values:
            readout = round(1.0 - main - projection, 2)
            if 0.20 <= readout <= 0.40:
                yield RouteFamilySplit(main=main, projection=projection, readout=readout).normalized()


def target_loss(quality, template_quality, reserve_quality):
    projection_target = max(1e-6, float(template_quality["projection_min"]))
    readout_target = max(1e-6, float(template_quality["fc_score"]))
    main_floor = max(1e-6, float(reserve_quality["main_path_min"]))
    projection = float(quality["projection_min"])
    readout = float(quality["fc_score"])
    main = float(quality["main_path_min"])
    projection_loss = ((projection - projection_target) / projection_target) ** 2
    readout_loss = ((readout - readout_target) / readout_target) ** 2
    main_loss = 0.0 if main >= main_floor else ((main_floor - main) / main_floor) ** 2
    dead_penalty = 0.002 * float(quality["total_dead_outputs"])
    return projection_loss + readout_loss + main_loss + dead_penalty


def optimize_split(syn, mag, template_quality, reserve_quality):
    best = None
    for split in candidate_splits():
        masks = route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split)
        quality = audit.route_quality(masks)
        loss = target_loss(quality, template_quality, reserve_quality)
        item = {"split": split, "loss": loss, "quality": quality}
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
            "route_min_mean": float(np.mean([r["route_quality"]["route_min"] for r in selected])),
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
            paired.append({
                "method": method,
                "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                "after_delta_std": float(np.std([d["after_delta"] for d in deltas])),
                "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                "paired_rows": deltas,
            })
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_target_matched_route_optimizer_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_TARGET_MATCHED_ROUTE_OPTIMIZER_99PCT.md"
    lines = [
        "# CIFAR-10 TinyResNet Target-Matched Route Optimizer at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "## Chosen splits",
        "",
        "| Seed | Main | Projection | Readout | Pre-FT target loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in result["chosen_splits"]:
        lines.append(f"| `{item['seed']}` | `{item['main']:.2f}` | `{item['projection']:.2f}` | `{item['readout']:.2f}` | `{item['loss']:.4f}` |")
    lines.extend([
        "",
        "## Results",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
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
        lines.append(f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | `{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | `{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is the first optimizer-style residual route allocator. It searches route-family splits before fine-tuning and selects the mask that best matches projection/readout targets from magnitude while preserving the plain-reserve main-path floor.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    chosen = []
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
        template_quality = audit.route_quality(mag_mask)
        reserve_quality = audit.route_quality(reserve_mask)
        fixed_prediction = predict_route_split(
            template_projection=template_quality["projection_min"],
            candidate_projection=reserve_quality["projection_min"],
            template_readout=template_quality["fc_score"],
            candidate_readout=reserve_quality["fc_score"],
        ).split
        best = optimize_split(syn, mag, template_quality, reserve_quality)
        chosen.append({"seed": seed, "main": best["split"].main, "projection": best["split"].projection, "readout": best["split"].readout, "loss": best["loss"]})
        masks_by_label = {
            "magnitude": mag_mask,
            "reserve_0.60": reserve_mask,
            "tuned_40_35_25": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, TUNED_SPLIT),
            "fixed_deficit_predictor": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, fixed_prediction),
            "target_matched_optimizer": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, best["split"]),
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
            print(f"seed {seed} {label}: after={after:.4f} split={best['split'] if label == 'target_matched_optimizer' else ''} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_target_matched_route_optimizer_99pct",
        "setup": "Fresh four-seed TinyResNet 99% target-matched route-family split optimizer.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "chosen_splits": chosen,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"chosen_splits": result["chosen_splits"], "summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
