from __future__ import annotations

import json

import numpy as np

import cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong as v5


v5.v4.base.SEEDS = [310]


def write_report(result):
    out = v5.v4.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong_seed310.json"
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong_seed310"
    result["setup"] = (
        "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector on seed 310, "
        "chosen by a dense-only branch scan because V5 selected global SynFlow through the "
        "SynFlow masked-recovery-prior branch before any masked fine-tuning."
    )
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = (
        v5.v4.base.ROOT
        / "experiments"
        / "04_criticality_pruning"
        / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG_SEED310.md"
    )
    lines = [
        "# CIFAR-10 TinyViT V5 SynFlow-Prior Branch Validation at 90%",
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
    for method in v5.v4.base.METHODS:
        item = result["summary"][method]
        before = np.mean([row["before_accuracy"] for row in result["rows"] if row["method"] == method])
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(v5.v4.base.SEEDS)}`"
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
            "This run tests the V5 rule's most important unresolved branch. The seed was selected only from dense-model diagnostics, then evaluated with full masked fine-tuning across all candidates.",
            "",
            "The neuroscientific interpretation is the trainability side of circuit viability: a sparse circuit can preserve live pathways and still fail if the remaining masked network cannot recover function. The SynFlow prior is a developmental-stability heuristic for preserving globally connected signal paths before task-specific repair.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


v5.v4.write_report = write_report


if __name__ == "__main__":
    result = v5.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
