from __future__ import annotations

import json

import cifar10_tiny_vit_feature_route_margin_selector_95pct as base


base.SEEDS = [299]
base.SPARSITY = 0.90
base.tinyvit.TRAIN_N = 50000
base.tinyvit.TEST_N = 10000
base.tinyvit.DENSE_EPOCHS = 20
base.tinyvit.FT_EPOCHS = 5


def choose_margin_policy_v2(alignments, qualities):
    ranked = sorted(
        [
            {
                "method": method,
                "centered_cls_cosine_mean": alignments[method]["centered_cls_cosine_mean"],
                "route_dead": base.route_dead(qualities[method]),
            }
            for method in base.CANDIDATES
        ],
        key=lambda item: item["centered_cls_cosine_mean"],
        reverse=True,
    )
    top = ranked[0]
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    if top["method"] == "global_synflow" and (syn["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) < 0.012 and syn["route_dead"] > mag["route_dead"] * 1.5:
        return "magnitude", "magnitude_trainable_capacity_guardrail", ranked
    if top["method"] != "global_synflow" and (top["centered_cls_cosine_mean"] - syn["centered_cls_cosine_mean"]) < 0.006 and top["route_dead"] > syn["route_dead"] * 2:
        return syn["method"], "synflow_margin_route_risk", ranked
    return top["method"], "feature_argmax", ranked


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong"
    result["setup"] = "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V2 feature-route selector. V2 adds a trainable-capacity guardrail: do not choose SynFlow when its feature advantage is small and it creates much more route death than magnitude."
    out = base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V2_90PCT_STRONG.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V2 at 90%: Stronger Recipe",
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
            "This is a one-seed fresh validation of the V2 transformer selector after the strong-recipe V1 failure. It tests whether adding a trainable-capacity guardrail can avoid overselecting SynFlow when magnitude is already comparatively viable.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.choose_margin_policy = choose_margin_policy_v2
base.write_report = write_report


if __name__ == "__main__":
    result = base.run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
