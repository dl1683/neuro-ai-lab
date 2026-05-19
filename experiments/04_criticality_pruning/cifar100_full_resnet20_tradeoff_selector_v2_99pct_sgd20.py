from __future__ import annotations

import json

import cifar100_full_resnet20_tradeoff_selector_99pct_sgd20 as base_exp


base_exp.SEEDS = [284, 285]


def write_report(result):
    result["experiment"] = "04_cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20"
    result["setup"] = "Fresh prospective full-CIFAR-100 ResNet-20 99% sparsity validation of the V2 feature-preservation / liveness tradeoff selector with task-ecology pressure."
    out = base_exp.ROOT / "results" / "04_criticality_pruning" / "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base_exp.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_V2_99PCT_SGD20.md"
    lines = [
        "# CIFAR-100 Full ResNet-20 Tradeoff Selector V2 at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in base_exp.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(base_exp.SEEDS)}`"
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
            "This is the prospective validation that the V2 task-ecology pressure term was intended to enable. Unlike the V2 policy projection, this run retrains fresh dense models and selects the sparse method before masked fine-tuning.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base_exp.write_report = write_report


if __name__ == "__main__":
    result = base_exp.run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "decisions": result["decisions"]}, indent=2))
