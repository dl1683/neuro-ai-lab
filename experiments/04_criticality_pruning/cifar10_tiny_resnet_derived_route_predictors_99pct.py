from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cifar10_tiny_resnet_capacity_transfer as base
import cifar10_tiny_resnet_route_quality_audit as audit
from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask, predict_route_split

SEEDS = [207, 208, 209, 210]
SPARSITY = 0.99
RESERVE = 0.60
TUNED_SPLIT = RouteFamilySplit(main=0.40, projection=0.35, readout=0.25)
METHODS = ["magnitude", "reserve_0.60", "tuned_40_35_25", "fixed_deficit_predictor", "relative_deficit_predictor", "sqrt_width_deficit_predictor"]


def output_units_by_family(scores):
    out = {"main": 0, "projection": 0, "readout": 0}
    for name, score in scores.items():
        if name == "stem.weight":
            continue
        units = int(score.shape[0]) if score.ndim >= 2 else 1
        if name == "fc.weight" or name.endswith("fc.weight"):
            out["readout"] += units
        elif ".shortcut.0.weight" in name:
            out["projection"] += units
        else:
            out["main"] += units
    return out


def family_metrics(quality):
    return {
        "main": float(quality["main_path_min"]),
        "projection": float(quality["projection_min"]),
        "readout": float(quality["fc_score"]),
    }


def relative_deficit_split(template_quality, candidate_quality):
    template = family_metrics(template_quality)
    candidate = family_metrics(candidate_quality)
    signals = {}
    for family in ["main", "projection", "readout"]:
        denom = max(abs(template[family]), 1e-6)
        deficit_ratio = max(0.0, template[family] - candidate[family]) / denom
        signals[family] = 1.0 + deficit_ratio
    total = sum(signals.values())
    return RouteFamilySplit(main=signals["main"] / total, projection=signals["projection"] / total, readout=signals["readout"] / total), signals


def sqrt_width_deficit_split(template_quality, candidate_quality, widths):
    template = family_metrics(template_quality)
    candidate = family_metrics(candidate_quality)
    signals = {}
    for family in ["main", "projection", "readout"]:
        denom = max(abs(template[family]), 1e-6)
        deficit_ratio = max(0.0, template[family] - candidate[family]) / denom
        signals[family] = math.sqrt(max(1, widths[family])) * (1.0 + deficit_ratio)
    total = sum(signals.values())
    return RouteFamilySplit(main=signals["main"] / total, projection=signals["projection"] / total, readout=signals["readout"] / total), signals


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
                mag = next(r for r in rows if r["seed"] == seed and r["method"] == "magnitude")
                alt = next(r for r in rows if r["seed"] == seed and r["method"] == method)
                deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
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
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_derived_route_predictors_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_DERIVED_ROUTE_PREDICTORS_99PCT.md"
    lines = [
        "# CIFAR-10 TinyResNet Derived Route Predictors at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "## Predicted splits",
        "",
        "| Seed | Predictor | Main | Projection | Readout |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in result["predicted_splits"]:
        lines.append(f"| `{item['seed']}` | `{item['predictor']}` | `{item['main']:.3f}` | `{item['projection']:.3f}` | `{item['readout']:.3f}` |")
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
        "This experiment removes or weakens the hand-set constants in the first route-deficit predictor. `relative_deficit_predictor` uses equal route-family priors. `sqrt_width_deficit_predictor` derives the family prior from route-family output width. Both select the split before recovery fine-tuning.",
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
        template_quality = audit.route_quality(mag_mask)
        candidate_quality = audit.route_quality(reserve_mask)
        fixed = predict_route_split(
            template_projection=template_quality["projection_min"],
            candidate_projection=candidate_quality["projection_min"],
            template_readout=template_quality["fc_score"],
            candidate_readout=candidate_quality["fc_score"],
        ).split
        relative, relative_signals = relative_deficit_split(template_quality, candidate_quality)
        sqrt_width, sqrt_signals = sqrt_width_deficit_split(template_quality, candidate_quality, output_units_by_family(syn))
        split_by_method = {
            "tuned_40_35_25": TUNED_SPLIT,
            "fixed_deficit_predictor": fixed,
            "relative_deficit_predictor": relative,
            "sqrt_width_deficit_predictor": sqrt_width,
        }
        for name, split in split_by_method.items():
            predicted.append({"seed": seed, "predictor": name, "main": split.main, "projection": split.projection, "readout": split.readout})
        masks_by_label = {
            "magnitude": mag_mask,
            "reserve_0.60": reserve_mask,
        }
        for label, split in split_by_method.items():
            masks_by_label[label] = route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split)
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
            print(f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}", flush=True)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_resnet_derived_route_predictors_99pct",
        "setup": "Fresh four-seed TinyResNet 99% test of derived route-family split predictors.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "predicted_splits": predicted,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
