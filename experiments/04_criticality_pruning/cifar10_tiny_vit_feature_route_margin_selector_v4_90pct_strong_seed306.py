from __future__ import annotations

import json

import numpy as np

import cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong as v4


v4.base.SEEDS = [306]


def write_report(result):
    out = v4.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306"
    result["setup"] = "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V4 selector on seed 306, which the branch scanner flagged as a non-SynFlow feature-argmax decision."
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = v4.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V4_90PCT_STRONG_SEED306.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V4 at 90%: Seed 306",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Dense accuracy mean: `{result['dense_accuracy_mean']:.4f}`",
        "",
        "| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in v4.base.METHODS:
        item = result["summary"][method]
        before = np.mean([row["before_accuracy"] for row in result["rows"] if row["method"] == method])
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(v4.base.SEEDS)}`"
        lines.append(
            f"| `{method}` | `{before:.4f}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['centered_cls_cosine_mean']:.4f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Selector decisions", ""])
    for item in result["decisions"]:
        ranked = ", ".join(
            f"{rank['method']}={rank['centered_cls_cosine_mean']:.4f}/before{rank['before_accuracy']:.4f}/dead{rank['route_dead']}"
            for rank in item["ranked_candidates"]
        )
        lines.append(f"- seed `{item['seed']}`: selected `{item['selected_method']}` via `{item['reason']}`; {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This validates whether V4 can choose a liveness/repair-style mask prospectively when residual-stream feature alignment favors it, rather than only selecting SynFlow in feature-dominant seeds.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


v4.write_report = write_report


if __name__ == "__main__":
    result = v4.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
