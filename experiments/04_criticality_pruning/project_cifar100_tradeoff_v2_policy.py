from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SOURCE = RESULTS / "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json"
OUT = RESULTS / "cifar100_full_resnet20_tradeoff_selector_v2_policy.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_V2_POLICY.md"


def score(item: dict, magnitude_quality: dict) -> float:
    magnitude_fc = float(magnitude_quality["fc_score"])
    magnitude_dead = float(magnitude_quality["total_dead_outputs"])
    readout_pressure = min(max((3.55 - magnitude_fc) / 0.70, 0.0), 1.0)
    death_pressure = min(max((magnitude_dead - 500.0) / 250.0, 0.0), 1.0)
    ecology_pressure = max(readout_pressure, death_pressure)
    feature_weight = 0.58 - 0.45 * ecology_pressure
    liveness_weight = 0.34 + 0.30 * ecology_pressure
    readout_weight = 0.16 + 0.15 * ecology_pressure
    return (
        feature_weight * item["feature_overlap_with_magnitude"]
        + liveness_weight * item["liveness"]
        + readout_weight * item["readout_ratio"]
        - 0.06 * item["dead_penalty"]
    )


def run():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    projected = []
    policy_rows = []
    for decision in source["decisions"]:
        seed = decision["seed"]
        mag_row = next(row for row in rows if row["seed"] == seed and row["method"] == "magnitude")
        rescored = []
        for item in decision["ranked_methods"]:
            updated = dict(item)
            updated["v2_score"] = score(item, mag_row["route_quality"])
            rescored.append(updated)
        selected = max(rescored, key=lambda item: item["v2_score"])["method"]
        selected_row = next(row for row in rows if row["seed"] == seed and row["method"] == selected)
        projected.append(
            {
                "seed": seed,
                "v1_selected_method": decision["selected_method"],
                "v2_selected_method": selected,
                "v2_ranked_methods": sorted(rescored, key=lambda item: item["v2_score"], reverse=True),
            }
        )
        policy_row = dict(selected_row)
        policy_row["method"] = "tradeoff_v2_policy"
        policy_row["policy_source_method"] = selected
        policy_rows.append(policy_row)

    all_rows = rows + policy_rows
    methods = ["magnitude", "feature_viability_repair", "plain_reserve", "predicted_route_split", "tradeoff_policy", "tradeoff_v2_policy"]
    summary = {}
    for method in methods:
        selected_rows = [row for row in all_rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected_rows])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected_rows])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected_rows])),
            "main_path_min_mean": float(np.mean([row["route_quality"]["main_path_min"] for row in selected_rows])),
            "projection_min_mean": float(np.mean([row["route_quality"]["projection_min"] for row in selected_rows])),
            "fc_score_mean": float(np.mean([row["route_quality"]["fc_score"] for row in selected_rows])),
        }
    for method in methods:
        if method == "magnitude":
            continue
        deltas = []
        for seed in source["seeds"]:
            mag = next(row for row in all_rows if row["seed"] == seed and row["method"] == "magnitude")
            alt = next(row for row in all_rows if row["seed"] == seed and row["method"] == method)
            deltas.append(alt["after_accuracy"] - mag["after_accuracy"])
        summary[method]["after_delta_mean"] = float(np.mean(deltas))
        summary[method]["after_wins"] = int(sum(delta > 0 for delta in deltas))

    result = {
        "experiment": "04_cifar100_full_resnet20_tradeoff_selector_v2_policy",
        "source": SOURCE.name,
        "setup": "Policy projection over the fresh CIFAR-100 tradeoff-selector candidate run. V2 lowers feature-overlap weight under high readout/output-diversity pressure.",
        "seeds": source["seeds"],
        "sparsity": source["sparsity"],
        "summary": summary,
        "decisions": projected,
        "rows": all_rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CIFAR-100 Full ResNet-20 Tradeoff Selector V2 Policy",
        "",
        result["setup"],
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summary[method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(source['seeds'])}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Decisions", ""])
    for item in projected:
        ranked = ", ".join(f"{rank['method']}={rank['v2_score']:.3f}" for rank in item["v2_ranked_methods"])
        lines.append(f"- seed `{item['seed']}`: v1 `{item['v1_selected_method']}` -> v2 `{item['v2_selected_method']}`; scores {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a policy projection, not a new training run. The candidate masks and post-finetune outcomes come from the fresh CIFAR-100 tradeoff experiment. The V2 selector changes only the pre-finetune selection rule, reducing feature-overlap weight when the magnitude mask shows high output/readout pressure.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
