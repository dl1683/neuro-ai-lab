from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong as v2


v2.base.SEEDS = [301]


def choose_margin_policy_v3(alignments, qualities):
    ranked = sorted(
        [
            {
                "method": method,
                "centered_cls_cosine_mean": alignments[method]["centered_cls_cosine_mean"],
                "route_dead": v2.base.route_dead(qualities[method]),
            }
            for method in v2.base.CANDIDATES
        ],
        key=lambda item: item["centered_cls_cosine_mean"],
        reverse=True,
    )
    top = ranked[0]
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    all_live = next(item for item in ranked if item["method"] == "all_route_liveness_floor")
    if top["method"] == "global_synflow" and (syn["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) < 0.012 and syn["route_dead"] > mag["route_dead"] * 1.5:
        if (all_live["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) > -0.002 and all_live["route_dead"] < mag["route_dead"]:
            return "all_route_liveness_floor", "all_route_liveness_capacity_guardrail", ranked
        return "magnitude", "magnitude_trainable_capacity_guardrail", ranked
    if top["method"] != "global_synflow" and (top["centered_cls_cosine_mean"] - syn["centered_cls_cosine_mean"]) < 0.006 and top["route_dead"] > syn["route_dead"] * 2:
        return syn["method"], "synflow_margin_route_risk", ranked
    return top["method"], "feature_argmax", ranked


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong"
    result["setup"] = "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V3 feature-route selector. V3 adds an all-route liveness guardrail when liveness repair preserves feature alignment within margin and removes much more route death than magnitude."
    out = v2.base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = v2.base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V3_90PCT_STRONG.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V3 at 90%: Stronger Recipe",
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
    for method in v2.base.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(v2.base.SEEDS)}`"
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
            "This is a one-seed fresh validation of the V3 transformer selector after the V2 replicate failure. The selector now has three possible regimes: feature-preserving SynFlow, magnitude trainable-capacity, or all-route liveness.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


v2.base.choose_margin_policy = choose_margin_policy_v3
v2.base.write_report = write_report


if __name__ == "__main__":
    result = v2.base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
