from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cifar100_full_resnet20_capacity_99pct_sgd_recipe as c100
import cifar10_tiny_resnet_capacity_transfer as base
from shared.circuit_viability_selector import choose_ecology_aware_method, choose_tradeoff_viability_method, split_dict
from shared.feature_preserving_viability import magnitude_with_liveness_repair
from shared.residual_route_capacity import route_split_capacity_mask


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [282, 283]
SPARSITY = 0.99
RESERVE = 0.60
METHODS = [
    "magnitude",
    "feature_viability_repair",
    "plain_reserve",
    "predicted_route_split",
    "tradeoff_policy",
]


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
    out = ROOT / "results" / "04_criticality_pruning" / "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_99PCT_SGD20.md"
    lines = [
        "# CIFAR-100 Full ResNet-20 Tradeoff Selector at 99%",
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
    lines.extend(["", "## Selector decisions", ""])
    for item in result["decisions"]:
        ranked = ", ".join(f"{rank['method']}={rank['score']:.3f}" for rank in item["ranked_methods"])
        lines.append(f"- seed `{item['seed']}`: selected `{item['selected_method']}`; scores {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This tests whether the feature-preservation/liveness tradeoff selector handles output-diverse CIFAR-100, where previous evidence favored stronger readout preservation. A failure here is useful: it would show the score still needs task-ecology terms rather than a fixed feature-overlap weight.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    decisions = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense full CIFAR-100 tradeoff-selector", flush=True)
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
        mag_mask = base.global_mask(mag, SPARSITY)
        ecology = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, c100.r20.route_quality)
        split = ecology["best_split"]["split"]
        candidate_masks = {
            "magnitude": mag_mask,
            "feature_viability_repair": magnitude_with_liveness_repair(mag, SPARSITY),
            "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
            "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split),
        }
        tradeoff = choose_tradeoff_viability_method(candidate_masks=candidate_masks, route_quality_fn=c100.r20.route_quality)
        source_method = tradeoff["selected_method"]
        decisions.append(
            {
                "seed": seed,
                "selected_method": source_method,
                "ecology_selected_method": ecology["selected_method"],
                "ecology_split": split_dict(split),
                "ranked_methods": [
                    {
                        "method": item["method"],
                        "score": item["score"],
                        "feature_overlap_with_magnitude": item["feature_overlap_with_magnitude"],
                        "liveness": item["liveness"],
                        "readout_ratio": item["readout_ratio"],
                        "dead_penalty": item["dead_penalty"],
                    }
                    for item in tradeoff["ranked_methods"]
                ],
            }
        )
        print(f"seed {seed}: tradeoff_selected={source_method}", flush=True)

        evaluated = {}
        for label, masks in candidate_masks.items():
            before, after = c100.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = c100.r20.route_quality(masks)
            row = {
                "seed": seed,
                "method": label,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "route_quality": quality,
            }
            rows.append(row)
            evaluated[label] = row
            print(
                f"seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
                f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
                flush=True,
            )
        policy_row = dict(evaluated[source_method])
        policy_row["method"] = "tradeoff_policy"
        policy_row["policy_source_method"] = source_method
        rows.append(policy_row)
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar100_full_resnet20_tradeoff_selector_99pct_sgd20",
        "setup": "Fresh full-CIFAR-100 ResNet-20 99% sparsity test of the feature-preservation / liveness tradeoff selector.",
        "device": base.DEVICE,
        "gpu_name": c100.torch.cuda.get_device_name(0) if c100.torch.cuda.is_available() else None,
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
