from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cifar100_full_resnet20_capacity_99pct_sgd_recipe as c100
import cifar10_tiny_resnet_capacity_transfer as base
from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [255, 256]
SPARSITY = 0.99
RESERVE = 0.60
METHODS = ["magnitude", "plain_reserve", "predicted_route_split"]


def candidate_splits():
    splits = []
    for main in np.arange(0.30, 0.56, 0.05):
        for projection in np.arange(0.10, 0.41, 0.05):
            readout = 1.0 - float(main) - float(projection)
            if 0.20 <= readout <= 0.55:
                splits.append(RouteFamilySplit(main=float(main), projection=float(projection), readout=float(readout)))
    return splits


def route_score(quality, plain_quality, magnitude_quality):
    main_floor = 0.80 * plain_quality["main_path_min"]
    projection_floor = 0.90 * plain_quality["projection_min"]
    readout_target = max(plain_quality["fc_score"], 0.66 * magnitude_quality["fc_score"])
    main_ratio = quality["main_path_min"] / max(main_floor, 1e-8)
    projection_ratio = quality["projection_min"] / max(projection_floor, 1e-8)
    readout_ratio = quality["fc_score"] / max(readout_target, 1e-8)
    floor_score = min(main_ratio, projection_ratio, readout_ratio)
    route_balance = 0.15 * min(main_ratio, 1.5) + 0.10 * min(projection_ratio, 1.5) + 0.25 * min(readout_ratio, 1.5)
    dead_penalty = 0.002 * quality["total_dead_outputs"]
    return floor_score + route_balance - dead_penalty


def choose_split(syn, mag):
    plain_mask = base.capacity_mask(syn, mag, SPARSITY, RESERVE)
    mag_mask = base.global_mask(mag, SPARSITY)
    plain_quality = c100.r20.route_quality(plain_mask)
    magnitude_quality = c100.r20.route_quality(mag_mask)
    scored = []
    for split in candidate_splits():
        masks = route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split)
        quality = c100.r20.route_quality(masks)
        scored.append(
            {
                "split": split,
                "score": route_score(quality, plain_quality, magnitude_quality),
                "quality": quality,
            }
        )
    best = max(scored, key=lambda item: item["score"])
    top = sorted(scored, key=lambda item: item["score"], reverse=True)[:8]
    return best, top, plain_quality, magnitude_quality


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


def split_dict(split):
    return {"main": split.main, "projection": split.projection, "readout": split.readout}


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR100_FULL_RESNET20_PREDICTED_ROUTE_SPLIT_99PCT_SGD_RECIPE.md"
    lines = [
        "# CIFAR-100 Full ResNet-20 Predicted Route Split at 99%: SGD Recipe",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
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
    lines.extend(
        [
            "",
            "## Chosen splits",
            "",
        ]
    )
    for item in result["chosen_splits"]:
        lines.append(f"- seed `{item['seed']}`: `{item['split']}` score `{item['score']:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The split is selected before fine-tuning from route-quality deficits: preserve a main-path floor, avoid projection collapse, and raise readout capacity toward the magnitude readout template. This tests whether the CIFAR-100 readout result can be predicted rather than hand-picked.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    chosen = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense full CIFAR-100 predicted split", flush=True)
        c100.torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = c100.full_loaders(seed)
        model = c100.make_model()
        c100.train_sgd(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        best, top, plain_quality, magnitude_quality = choose_split(syn, mag)
        chosen.append(
            {
                "seed": seed,
                "split": split_dict(best["split"]),
                "score": best["score"],
                "quality": best["quality"],
                "plain_quality": plain_quality,
                "magnitude_quality": magnitude_quality,
                "top_splits": [{"split": split_dict(item["split"]), "score": item["score"], "quality": item["quality"]} for item in top],
            }
        )
        masks_by_label = {
            "magnitude": base.global_mask(mag, SPARSITY),
            "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, best["split"]),
        }
        print(f"seed {seed}: predicted_split={split_dict(best['split'])} score={best['score']:.4f}", flush=True)
        for label, masks in masks_by_label.items():
            before, after = c100.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = c100.r20.route_quality(masks)
            rows.append(
                {
                    "seed": seed,
                    "method": label,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "route_quality": quality,
                }
            )
            print(
                f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
                f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
                flush=True,
            )
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe",
        "setup": "Full CIFAR-100 train/test ResNet-20-style predicted route split at 99% sparsity using pre-finetune route-quality deficits.",
        "device": base.DEVICE,
        "gpu_name": c100.torch.cuda.get_device_name(0) if c100.torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "summary": summary,
        "paired_deltas": paired,
        "chosen_splits": chosen,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "chosen_splits": result["chosen_splits"]}, indent=2))
