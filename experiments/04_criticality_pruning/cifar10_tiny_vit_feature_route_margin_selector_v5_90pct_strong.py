from __future__ import annotations

import json

import numpy as np

import cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong as v4


v4.base.SEEDS = [307]
ORIGINAL_CHOOSE_MARGIN_POLICY_V4 = v4.choose_margin_policy_v4


def choose_margin_policy_v5(alignments, qualities, before_scores):
    selected, reason, ranked = ORIGINAL_CHOOSE_MARGIN_POLICY_V4(alignments, qualities, before_scores)
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    selected_item = next(item for item in ranked if item["method"] == selected)
    if selected != "global_synflow" and syn["before_accuracy"] >= mag["before_accuracy"] and selected_item["before_accuracy"] - syn["before_accuracy"] < 0.008:
        return "global_synflow", "synflow_masked_recovery_prior", ranked
    return selected, reason, ranked


def run():
    v4.choose_margin_policy_v4 = choose_margin_policy_v5
    try:
        result = v4.run()
    finally:
        v4.choose_margin_policy_v4 = ORIGINAL_CHOOSE_MARGIN_POLICY_V4
    return result


def write_report(result):
    out = v4.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json"
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong"
    result["setup"] = "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector. V5 adds a SynFlow masked-recovery prior: when SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow despite lower centered CLS alignment."
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = v4.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V5 at 90%: Stronger Recipe",
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
            "This is the first prospective test of the V5 SynFlow recovery-prior rule. The decision is made before masked fine-tuning from feature alignment, route liveness, and masked-before accuracy.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


v4.write_report = write_report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
