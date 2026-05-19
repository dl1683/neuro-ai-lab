from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SOURCE = RESULTS / "cifar10_tiny_vit_feature_subspace_selector_95pct.json"
OUT = RESULTS / "cifar10_tiny_vit_feature_route_margin_policy_95pct.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_POLICY_95PCT.md"


def choose(seed_rows):
    ranked = sorted(seed_rows, key=lambda row: row["feature_alignment"]["centered_cls_cosine_mean"], reverse=True)
    top = ranked[0]
    syn = next(row for row in seed_rows if row["method"] == "global_synflow")
    top_score = top["feature_alignment"]["centered_cls_cosine_mean"]
    syn_score = syn["feature_alignment"]["centered_cls_cosine_mean"]
    top_dead = top["route_quality"]["mlp_down_dead_outputs"] + top["route_quality"]["attn_out_dead_outputs"]
    syn_dead = syn["route_quality"]["mlp_down_dead_outputs"] + syn["route_quality"]["attn_out_dead_outputs"]
    if top["method"] != "global_synflow" and (top_score - syn_score) < 0.006 and top_dead > syn_dead * 2:
        return syn, "synflow_margin_route_risk"
    return top, "feature_argmax"


def run():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    policy_rows = []
    decisions = []
    for seed in source["seeds"]:
        seed_rows = [row for row in rows if row["seed"] == seed and row["method"] != "feature_subspace_policy"]
        selected, reason = choose(seed_rows)
        policy = dict(selected)
        policy["method"] = "feature_route_margin_policy"
        policy["policy_source_method"] = selected["method"]
        policy_rows.append(policy)
        decisions.append(
            {
                "seed": seed,
                "selected_method": selected["method"],
                "reason": reason,
                "selected_centered_cls": selected["feature_alignment"]["centered_cls_cosine_mean"],
                "selected_route_dead": selected["route_quality"]["mlp_down_dead_outputs"] + selected["route_quality"]["attn_out_dead_outputs"],
            }
        )
    all_rows = rows + policy_rows
    methods = ["magnitude", "global_synflow", "feature_subspace_policy", "feature_route_margin_policy"]
    summary = {}
    for method in methods:
        selected_rows = [row for row in all_rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected_rows])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected_rows])),
            "centered_cls_cosine_mean": float(np.mean([row["feature_alignment"]["centered_cls_cosine_mean"] for row in selected_rows])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected_rows])),
            "mlp_down_dead_outputs_mean": float(np.mean([row["route_quality"]["mlp_down_dead_outputs"] for row in selected_rows])),
            "attn_out_dead_outputs_mean": float(np.mean([row["route_quality"]["attn_out_dead_outputs"] for row in selected_rows])),
        }
    for method in methods:
        if method == "magnitude":
            continue
        deltas = []
        for seed in source["seeds"]:
            mag = next(row for row in all_rows if row["seed"] == seed and row["method"] == "magnitude")
            alt = next(row for row in all_rows if row["seed"] == seed and row["method"] == method)
            deltas.append(alt["after_accuracy"] - mag["after_accuracy"])
        summary[method]["after_delta_mean"] = float(np.mean(deltas))
        summary[method]["after_wins"] = int(sum(delta > 0 for delta in deltas))
    result = {
        "experiment": "04_cifar10_tiny_vit_feature_route_margin_policy_95pct",
        "source": SOURCE.name,
        "setup": "Policy projection over the fresh TinyViT feature-subspace selector run. If feature alignment is within a small margin, prefer the candidate with lower transformer route death.",
        "seeds": source["seeds"],
        "sparsity": source["sparsity"],
        "summary": summary,
        "decisions": decisions,
        "rows": all_rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Policy at 95%",
        "",
        result["setup"],
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summary[method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(source['seeds'])}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['centered_cls_cosine_mean']:.4f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Decisions", ""])
    for item in decisions:
        lines.append(
            f"- seed `{item['seed']}`: selected `{item['selected_method']}` via `{item['reason']}` "
            f"centered CLS `{item['selected_centered_cls']:.4f}` route dead `{item['selected_route_dead']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a policy projection, not a fresh training run. It uses the already evaluated candidate masks from the prospective feature-subspace selector. The result shows why argmax feature alignment is not enough: when scores are close, route-death risk can resolve the ambiguity.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
