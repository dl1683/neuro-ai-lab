from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_95pct as base


base.SEEDS = [296, 297]
base.SPARSITY = 0.90


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_90pct"
    result["setup"] = "Fresh TinyViT CIFAR-10 subset 90% sparsity prospective feature-route margin selector. This tests the transformer selector before recovery is fully floor-dominated."
    out = base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_90PCT.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector at 90%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in base.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(base.SEEDS)}`"
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
            "This is the same feature-route margin policy as the `95%` TinyViT selector, evaluated at `90%` sparsity. The purpose is to test whether representation-preserving circuit viability remains useful when the sparse transformer is not already close to chance.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.write_report = write_report


if __name__ == "__main__":
    result = base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
