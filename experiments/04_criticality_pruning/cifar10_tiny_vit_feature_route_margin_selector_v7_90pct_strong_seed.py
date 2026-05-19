from __future__ import annotations

import argparse
import json

import cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed as v6


def choose_margin_policy_v7(alignments, qualities, before_scores):
    selected, reason, ranked = v6.choose_margin_policy_v6(alignments, qualities, before_scores)
    if selected not in v6.LIVE_REPAIR_METHODS or reason != "feature_argmax":
        return selected, reason, ranked
    selected_item = next(item for item in ranked if item["method"] == selected)
    mag = next(item for item in ranked if item["method"] == "magnitude")
    if selected_item["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"] < 0.001:
        return "magnitude", "magnitude_live_repair_tiny_feature_guardrail", ranked
    return selected, reason, ranked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    v6.v5.v4.base.SEEDS = [args.seed]
    original = v6.v5.v4.choose_margin_policy_v4
    v6.v5.v4.choose_margin_policy_v4 = choose_margin_policy_v7
    v6.v5.v4.write_report = v6.make_writer(args.seed)
    try:
        result = v6.v5.v4.run()
    finally:
        v6.v5.v4.choose_margin_policy_v4 = original

    old_result = v6.v5.v4.base.RESULTS / f"cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed{args.seed}.json"
    new_result = v6.v5.v4.base.RESULTS / f"cifar10_tiny_vit_feature_route_margin_selector_v7_90pct_strong_seed{args.seed}.json"
    data = json.loads(old_result.read_text(encoding="utf-8"))
    data["experiment"] = f"04_cifar10_tiny_vit_feature_route_margin_selector_v7_90pct_strong_seed{args.seed}"
    data["setup"] = (
        f"Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V7 selector on seed {args.seed}. "
        "V7 keeps V6 and adds a magnitude-vs-live-repair guardrail: when direct live-repair feature advantage "
        "over magnitude is tiny, keep the magnitude sparse template."
    )
    new_result.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    old_result.unlink()

    old_md = (
        v6.v5.v4.base.ROOT
        / "experiments"
        / "04_criticality_pruning"
        / f"CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V6_90PCT_STRONG_SEED{args.seed}.md"
    )
    new_md = (
        v6.v5.v4.base.ROOT
        / "experiments"
        / "04_criticality_pruning"
        / f"CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V7_90PCT_STRONG_SEED{args.seed}.md"
    )
    text = old_md.read_text(encoding="utf-8")
    text = text.replace("V6", "V7").replace("v6", "v7")
    text = text.replace(
        "V7 keeps the V5 SynFlow masked-recovery prior and adds a live-repair tie-breaker: when live-repair feature margins are tiny, choose the repair with higher masked-before trainability.",
        data["setup"],
    )
    new_md.write_text(text, encoding="utf-8")
    old_md.unlink()

    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
