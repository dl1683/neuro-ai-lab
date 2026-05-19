from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong as base


base.base.SEEDS = [300]


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate"
    result["setup"] = "Fresh replicate of the full-train TinyViT CIFAR-10 90% sparsity V2 feature-route selector. Uses the same 20 dense epochs and 5 masked fine-tune epochs as the first strong V2 pilot."
    out = base.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V2_90PCT_STRONG_REPLICATE.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V2 at 90%: Strong Replicate",
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
            "This replicate checks whether the V2 trainable-capacity guardrail generalizes to another strong TinyViT seed. The claim is selector behavior, not benchmark strength.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.base.write_report = write_report


if __name__ == "__main__":
    result = base.base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
