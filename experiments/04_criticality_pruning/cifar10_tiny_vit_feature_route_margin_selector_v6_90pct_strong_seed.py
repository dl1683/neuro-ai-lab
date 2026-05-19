from __future__ import annotations

import argparse
import json

import numpy as np

import cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong as v5


LIVE_REPAIR_METHODS = {
    "minimal_liveness_repair",
    "attn_mlp_readout_repair",
    "all_route_liveness_floor",
}


def choose_margin_policy_v6(alignments, qualities, before_scores):
    selected, reason, ranked = v5.choose_margin_policy_v5(alignments, qualities, before_scores)
    if selected not in LIVE_REPAIR_METHODS:
        return selected, reason, ranked

    live = [item for item in ranked if item["method"] in LIVE_REPAIR_METHODS]
    selected_item = next(item for item in live if item["method"] == selected)
    best_before = max(live, key=lambda item: item["before_accuracy"])
    feature_spread = max(item["centered_cls_cosine_mean"] for item in live) - min(
        item["centered_cls_cosine_mean"] for item in live
    )

    if (
        feature_spread < 0.003
        and best_before["before_accuracy"] >= selected_item["before_accuracy"] + 0.0005
    ):
        return best_before["method"], "live_repair_masked_before_tiebreak", ranked
    return selected, reason, ranked


def make_writer(seed: int):
    def write_report(result):
        out = v5.v4.base.RESULTS / f"cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed{seed}.json"
        result["experiment"] = f"04_cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed{seed}"
        result["setup"] = (
            f"Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V6 selector on seed {seed}. "
            "V6 keeps the V5 SynFlow masked-recovery prior and adds a live-repair tie-breaker: when live-repair "
            "feature margins are tiny, choose the repair with higher masked-before trainability."
        )
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        md = (
            v5.v4.base.ROOT
            / "experiments"
            / "04_criticality_pruning"
            / f"CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V6_90PCT_STRONG_SEED{seed}.md"
        )
        lines = [
            f"# CIFAR-10 TinyViT V6 Prospective Validation at 90%: Seed {seed}",
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
                "This run tests the V6 correction motivated by seed 312. V6 does not change the SynFlow branches. It only asks whether masked-before trainability is a better tie-breaker than a tiny feature-alignment margin inside the live-repair family.",
            ]
        )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    v5.v4.base.SEEDS = [args.seed]
    original = v5.v4.choose_margin_policy_v4
    v5.v4.choose_margin_policy_v4 = choose_margin_policy_v6
    v5.v4.write_report = make_writer(args.seed)
    try:
        result = v5.v4.run()
    finally:
        v5.v4.choose_margin_policy_v4 = original
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
