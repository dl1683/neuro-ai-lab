from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong as v3


v3.v2.base.SEEDS = [302, 303]


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate"
    result["setup"] = "Two-seed fresh replicate of the full-train TinyViT CIFAR-10 90% sparsity V3 feature-route selector. Tests whether the three-way feature/liveness/trainability rule survives new strong TinyViT seeds."
    out = v3.v2.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = v3.v2.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V3_90PCT_STRONG_REPLICATE.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V3 at 90%: Strong Replicate",
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
    for method in v3.v2.base.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(v3.v2.base.SEEDS)}`"
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
            "This replicate tests the selector boundary rather than headline benchmark strength. A win means V3 is a better rule than V2; a miss identifies which feature/liveness/trainability term is still under-modeled.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


v3.v2.base.write_report = write_report


if __name__ == "__main__":
    result = v3.v2.base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
