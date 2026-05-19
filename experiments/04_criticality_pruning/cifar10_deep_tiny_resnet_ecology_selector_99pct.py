from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

import cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct as deep
import cifar10_tiny_resnet_capacity_transfer as base
from shared.circuit_viability_selector import choose_ecology_aware_method, split_dict
from shared.residual_route_capacity import route_split_capacity_mask


SEEDS = [267, 268]
SPARSITY = 0.99
RESERVE = 0.60
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_selected"]


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


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_deep_tiny_resnet_ecology_selector_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_DEEP_TINY_RESNET_ECOLOGY_SELECTOR_99PCT.md"
    lines = [
        "# CIFAR-10 DeepTinyResNet Ecology-Aware Selector at 99%",
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
    lines.extend(["", "## Decisions", ""])
    for item in result["decisions"]:
        lines.append(
            f"- seed `{item['seed']}`: selected `{item['selected_method']}` "
            f"readout_ratio `{item['plain_readout_ratio']:.4f}` split `{item['selected_split']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This tests whether the ecology-aware selector transfers to a deeper residual architecture without changing the readout-ratio threshold. The dataset is still the existing CIFAR-10 subset harness, so this is an architecture-transfer check rather than a full benchmark.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    decisions = []
    for seed in SEEDS:
        print(f"deep seed {seed}: train dense DeepTinyResNet", flush=True)
        deep.torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.loaders(seed)
        model = deep.DeepTinyResNet().to(base.DEVICE)
        deep.train_model(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base.evaluate(model, test_loader)
        print(f"deep seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = base.magnitude_scores(model)
        syn = base.synflow_scores(model)
        decision = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, deep.route_quality)
        split = decision["best_split"]["split"]
        masks_by_label = {
            "magnitude": base.global_mask(mag, SPARSITY),
            "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split),
            "ecology_selected": decision["selected_mask"],
        }
        decisions.append(
            {
                "seed": seed,
                "selected_method": decision["selected_method"],
                "selected_split": decision["selected_split"],
                "plain_readout_ratio": decision["plain_readout_ratio"],
                "readout_ratio_threshold": decision["readout_ratio_threshold"],
                "best_split": split_dict(split),
                "plain_quality": decision["plain_quality"],
                "magnitude_quality": decision["magnitude_quality"],
                "best_split_quality": decision["best_split"]["quality"],
            }
        )
        print(
            f"deep seed {seed}: selected={decision['selected_method']} "
            f"readout_ratio={decision['plain_readout_ratio']:.4f} best_split={split_dict(split)}",
            flush=True,
        )
        for label, masks in masks_by_label.items():
            before, after = deep.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = deep.route_quality(masks)
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
                f"deep seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
                f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
                flush=True,
            )
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar10_deep_tiny_resnet_ecology_selector_99pct",
        "setup": "CIFAR-10 subset DeepTinyResNet architecture-transfer validation of the ecology-aware pre-finetune selector at 99% sparsity.",
        "device": base.DEVICE,
        "gpu_name": deep.torch.cuda.get_device_name(0) if deep.torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "summary": summary,
        "paired_deltas": paired,
        "decisions": decisions,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "decisions": result["decisions"]}, indent=2))
