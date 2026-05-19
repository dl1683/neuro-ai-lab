from __future__ import annotations

import json

import cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe as pred


pred.SEEDS = [257, 258]


def conservative_route_score(quality, plain_quality, magnitude_quality):
    main_floor = 0.84 * plain_quality["main_path_min"]
    projection_floor = 0.90 * plain_quality["projection_min"]
    readout_target = max(plain_quality["fc_score"], 0.66 * magnitude_quality["fc_score"])
    main_ratio = quality["main_path_min"] / max(main_floor, 1e-8)
    projection_ratio = quality["projection_min"] / max(projection_floor, 1e-8)
    readout_ratio = quality["fc_score"] / max(readout_target, 1e-8)
    floor_score = min(main_ratio, projection_ratio, readout_ratio)
    route_balance = 0.30 * min(main_ratio, 1.5) + 0.10 * min(projection_ratio, 1.5) + 0.18 * min(readout_ratio, 1.5)
    dead_penalty = 0.002 * quality["total_dead_outputs"]
    return floor_score + route_balance - dead_penalty


def split_dict(split):
    return {"main": split.main, "projection": split.projection, "readout": split.readout}


def write_report(result):
    result["experiment"] = "04_cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe"
    result["setup"] = "Full CIFAR-100 train/test ResNet-20-style conservative predicted route split at 99% sparsity using pre-finetune route-quality deficits."
    out = pred.ROOT / "results" / "04_criticality_pruning" / "cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = pred.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR100_FULL_RESNET20_CONSERVATIVE_PREDICTED_ROUTE_SPLIT_99PCT_SGD_RECIPE.md"
    lines = [
        "# CIFAR-100 Full ResNet-20 Conservative Predicted Route Split at 99%: SGD Recipe",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in pred.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(pred.SEEDS)}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Chosen splits", ""])
    for item in result["chosen_splits"]:
        lines.append(f"- seed `{item['seed']}`: `{item['split']}` score `{item['score']:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This follow-up tightens the automatic selector after the first predictor over-weighted readout on one seed. It requires a stronger main-path floor before spending capacity on the CIFAR-100 readout.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


pred.route_score = conservative_route_score
pred.write_report = write_report


if __name__ == "__main__":
    result = pred.run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "chosen_splits": result["chosen_splits"]}, indent=2))
