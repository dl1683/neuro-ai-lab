from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_90pct as base


base.base.SEEDS = [298]
base.base.SPARSITY = 0.90
base.base.tinyvit.TRAIN_N = 50000
base.base.tinyvit.TEST_N = 10000
base.base.tinyvit.DENSE_EPOCHS = 20
base.base.tinyvit.FT_EPOCHS = 5


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_90pct_strong"
    result["setup"] = "TinyViT CIFAR-10 full-train stronger-recipe 90% sparsity feature-route margin selector pilot. Uses full CIFAR-10 train/test, 20 dense epochs, and 5 masked fine-tune epochs."
    out = base.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_90PCT_STRONG.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector at 90%: Stronger Recipe",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Dense accuracy mean: `{result['dense_accuracy_mean']:.4f}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in base.base.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(base.base.SEEDS)}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['centered_cls_cosine_mean']:.4f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Selector decisions", ""])
    for item in result["decisions"]:
        ranked = ", ".join(
            f"{rank['method']}={rank['centered_cls_cosine_mean']:.4f}/dead{rank['route_dead']}"
            for rank in item["ranked_candidates"]
        )
        lines.append(f"- seed `{item['seed']}`: selected `{item['selected_method']}` via `{item['reason']}`; {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a one-seed pilot to test whether the TinyViT feature-route margin selector remains meaningful when dense training is stronger and the evaluation uses the full CIFAR-10 train/test split. If positive, this should be expanded to more seeds.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.base.write_report = write_report


if __name__ == "__main__":
    result = base.base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
