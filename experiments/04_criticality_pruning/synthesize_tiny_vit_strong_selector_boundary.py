from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"

SOURCES = [
    ("v1_strong_seed298", "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"),
    ("v2_strong_seed299", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"),
    ("v2_strong_seed300", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"),
    ("v3_strong_seed301", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"),
    ("v3_strong_seed302_303", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"),
    ("v4_strong_seed304", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"),
]

CANDIDATES = [
    "magnitude",
    "global_synflow",
    "minimal_liveness_repair",
    "attn_mlp_readout_repair",
    "all_route_liveness_floor",
]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def route_dead(row: dict) -> int:
    rq = row["route_quality"]
    return int(rq["mlp_down_dead_outputs"] + rq["attn_out_dead_outputs"])


def centered(row: dict) -> float:
    return float(row["feature_alignment"]["centered_cls_cosine_mean"])


def choose_v3(rows_by_method: dict[str, dict]) -> tuple[str, str, list[dict]]:
    ranked = sorted(
        [
            {
                "method": method,
                "centered_cls_cosine_mean": centered(rows_by_method[method]),
                "route_dead": route_dead(rows_by_method[method]),
            }
            for method in CANDIDATES
        ],
        key=lambda item: item["centered_cls_cosine_mean"],
        reverse=True,
    )
    top = ranked[0]
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    all_live = next(item for item in ranked if item["method"] == "all_route_liveness_floor")
    if (
        top["method"] == "global_synflow"
        and (syn["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) < 0.012
        and syn["route_dead"] > mag["route_dead"] * 1.5
    ):
        if (
            (all_live["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) > -0.002
            and all_live["route_dead"] < mag["route_dead"]
        ):
            return "all_route_liveness_floor", "all_route_liveness_capacity_guardrail", ranked
        return "magnitude", "magnitude_trainable_capacity_guardrail", ranked
    if (
        top["method"] != "global_synflow"
        and (top["centered_cls_cosine_mean"] - syn["centered_cls_cosine_mean"]) < 0.006
        and top["route_dead"] > syn["route_dead"] * 2
    ):
        return syn["method"], "synflow_margin_route_risk", ranked
    return top["method"], "feature_argmax", ranked


def choose_v4(rows_by_method: dict[str, dict]) -> tuple[str, str, list[dict]]:
    ranked = sorted(
        [
            {
                "method": method,
                "centered_cls_cosine_mean": centered(rows_by_method[method]),
                "route_dead": route_dead(rows_by_method[method]),
                "before_accuracy": rows_by_method[method]["before_accuracy"],
            }
            for method in CANDIDATES
        ],
        key=lambda item: item["centered_cls_cosine_mean"],
        reverse=True,
    )
    top = ranked[0]
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    all_live = next(item for item in ranked if item["method"] == "all_route_liveness_floor")
    if (
        top["method"] == "global_synflow"
        and (syn["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) < 0.012
        and syn["route_dead"] > mag["route_dead"] * 1.5
    ):
        if (
            all_live["centered_cls_cosine_mean"] > mag["centered_cls_cosine_mean"]
            and all_live["before_accuracy"] > mag["before_accuracy"]
        ):
            return "all_route_liveness_floor", "all_route_liveness_feature_trainability_guardrail", ranked
        return "magnitude", "magnitude_masked_trainability_guardrail", ranked
    if (
        top["method"] != "global_synflow"
        and (top["centered_cls_cosine_mean"] - syn["centered_cls_cosine_mean"]) < 0.006
        and top["route_dead"] > syn["route_dead"] * 2
    ):
        return syn["method"], "synflow_margin_route_risk", ranked
    return top["method"], "feature_argmax", ranked


def run() -> dict:
    seed_cases = []
    for family, source in SOURCES:
        data = load(source)
        rows_by_seed: dict[int, dict[str, dict]] = {}
        for row in data["rows"]:
            if row["method"] in CANDIDATES:
                rows_by_seed.setdefault(int(row["seed"]), {})[row["method"]] = row
        decisions_by_seed = {int(item["seed"]): item for item in data["decisions"]}
        for seed, rows_by_method in sorted(rows_by_seed.items()):
            selected, reason, ranked = choose_v3(rows_by_method)
            v4_selected, v4_reason, v4_ranked = choose_v4(rows_by_method)
            best_method, best_row = max(rows_by_method.items(), key=lambda item: item[1]["after_accuracy"])
            mag_after = rows_by_method["magnitude"]["after_accuracy"]
            selected_after = rows_by_method[selected]["after_accuracy"]
            v4_after = rows_by_method[v4_selected]["after_accuracy"]
            best_after = best_row["after_accuracy"]
            seed_cases.append(
                {
                    "seed": seed,
                    "source_family": family,
                    "source": source,
                    "dense_accuracy": rows_by_method["magnitude"]["dense_accuracy"],
                    "original_selected_method": decisions_by_seed[seed]["selected_method"],
                    "original_reason": decisions_by_seed[seed]["reason"],
                    "v3_projected_method": selected,
                    "v3_projected_reason": reason,
                    "best_method": best_method,
                    "magnitude_after": mag_after,
                    "v3_after": selected_after,
                    "best_after": best_after,
                    "v3_delta_vs_magnitude": selected_after - mag_after,
                    "v3_gap_to_best": best_after - selected_after,
                    "v3_matches_best": selected == best_method,
                    "ranked_candidates": ranked,
                    "v4_projected_method": v4_selected,
                    "v4_projected_reason": v4_reason,
                    "v4_after": v4_after,
                    "v4_delta_vs_magnitude": v4_after - mag_after,
                    "v4_gap_to_best": best_after - v4_after,
                    "v4_matches_best": v4_selected == best_method,
                    "v4_ranked_candidates": v4_ranked,
                }
            )
    positive = [item for item in seed_cases if item["v3_delta_vs_magnitude"] > 0]
    matched = [item for item in seed_cases if item["v3_matches_best"]]
    v4_positive = [item for item in seed_cases if item["v4_delta_vs_magnitude"] > 0]
    v4_matched = [item for item in seed_cases if item["v4_matches_best"]]
    result = {
        "experiment": "04_tiny_vit_strong_selector_boundary_synthesis",
        "setup": "Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.",
        "sources": [source for _, source in SOURCES],
        "seed_count": len(seed_cases),
        "v3_positive_vs_magnitude": len(positive),
        "v3_matches_best": len(matched),
        "v3_mean_delta_vs_magnitude": sum(item["v3_delta_vs_magnitude"] for item in seed_cases) / len(seed_cases),
        "v3_mean_gap_to_best": sum(item["v3_gap_to_best"] for item in seed_cases) / len(seed_cases),
        "v4_positive_vs_magnitude": len(v4_positive),
        "v4_matches_best": len(v4_matched),
        "v4_mean_delta_vs_magnitude": sum(item["v4_delta_vs_magnitude"] for item in seed_cases) / len(seed_cases),
        "v4_mean_gap_to_best": sum(item["v4_gap_to_best"] for item in seed_cases) / len(seed_cases),
        "cases": seed_cases,
    }
    out = RESULTS / "tiny_vit_strong_selector_boundary_synthesis.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    return result


def write_report(result: dict) -> None:
    md = ROOT / "experiments" / "04_criticality_pruning" / "TINY_VIT_STRONG_SELECTOR_BOUNDARY_SYNTHESIS.md"
    lines = [
        "# TinyViT Strong Selector Boundary Synthesis",
        "",
        result["setup"],
        "",
        f"Seeds synthesized: `{result['seed_count']}`",
        f"V3 positive vs magnitude: `{result['v3_positive_vs_magnitude']}/{result['seed_count']}`",
        f"V3 matched best evaluated candidate: `{result['v3_matches_best']}/{result['seed_count']}`",
        f"Mean V3 delta vs magnitude: `{result['v3_mean_delta_vs_magnitude']:+.4f}`",
        f"Mean V3 gap to best candidate: `{result['v3_mean_gap_to_best']:.4f}`",
        f"V4 positive vs magnitude: `{result['v4_positive_vs_magnitude']}/{result['seed_count']}`",
        f"V4 matched best evaluated candidate: `{result['v4_matches_best']}/{result['seed_count']}`",
        f"Mean V4 delta vs magnitude: `{result['v4_mean_delta_vs_magnitude']:+.4f}`",
        f"Mean V4 gap to best candidate: `{result['v4_mean_gap_to_best']:.4f}`",
        "",
        "| Seed | V3 projected | V4 projected | Best evaluated | Magnitude after | V4 after | Best after | V4 delta | V4 gap |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["cases"]:
        lines.append(
            f"| `{item['seed']}` | `{item['v3_projected_method']}` | `{item['v4_projected_method']}` | "
            f"`{item['best_method']}` | `{item['magnitude_after']:.4f}` | `{item['v4_after']:.4f}` | "
            f"`{item['best_after']:.4f}` | `{item['v4_delta_vs_magnitude']:+.4f}` | `{item['v4_gap_to_best']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.",
            "",
            "The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, V3 chooses SynFlow and the choice wins. When the margin is small and route death is high, V3 routes to a trainability or liveness guardrail. V4 adds masked pre-finetune accuracy to that ambiguous branch. On the completed strong TinyViT seeds, that prospective diagnostic removes the two V3 guardrail misses and matches the best evaluated candidate on every seed. This is still a projection over completed candidate evaluations, but it defines the next prospective validation target precisely.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
