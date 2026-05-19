from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
RESULTS = ROOT / "results" / "04_criticality_pruning"
OUT = RESULTS / "unified_viability_selector_retrospective.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "UNIFIED_VIABILITY_SELECTOR_RETROSPECTIVE.md"

from shared.circuit_viability_selector import choose_unified_viability_family


CASES = [
    {
        "case": "full_cifar10_resnet20_sgd40",
        "source": "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json",
        "mode": "flat",
        "expected_family": "ecology_selector",
        "family_method": "ecology_policy",
    },
    {
        "case": "deep_tinyresnet_cifar10",
        "source": "cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json",
        "mode": "flat",
        "expected_family": "ecology_selector",
        "family_method": "ecology_policy",
    },
    {
        "case": "ecology_cifar10",
        "source": "cifar_ecology_selector_99pct_sgd_recipe.json",
        "mode": "nested",
        "task": "cifar10",
        "expected_family": "ecology_selector",
        "family_method": "ecology_selected",
    },
    {
        "case": "ecology_cifar100",
        "source": "cifar_ecology_selector_99pct_sgd_recipe.json",
        "mode": "nested",
        "task": "cifar100",
        "expected_family": "ecology_selector",
        "family_method": "ecology_selected",
    },
    {
        "case": "pretrained_tinyimagenet_95",
        "source": "tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json",
        "mode": "flat",
        "expected_family": "feature_viability_repair",
        "family_method": "feature_viability_repair",
    },
    {
        "case": "pretrained_tinyimagenet_99",
        "source": "tinyimagenet_resnet18_pretrained_feature_viability_99pct.json",
        "mode": "flat",
        "expected_family": "feature_viability_repair",
        "family_method": "feature_viability_repair",
    },
]


def load(source: str) -> dict:
    return json.loads((RESULTS / source).read_text(encoding="utf-8"))


def summary_for(data: dict, case: dict) -> dict:
    if case["mode"] == "nested":
        return data["summary"][case["task"]]
    return data["summary"]


def magnitude_quality(data: dict, case: dict, summary: dict) -> dict:
    if "decisions" in data:
        decisions = data["decisions"]
        if case["mode"] == "nested":
            decisions = [item for item in decisions if item.get("task") == case["task"]]
        if decisions:
            values = [item["magnitude_quality"]["main_path_min"] for item in decisions]
            return {"main_path_min": float(np.mean(values))}
    return {"main_path_min": float(summary["magnitude"]["main_path_min_mean"])}


def evaluate_case(case: dict) -> dict:
    data = load(case["source"])
    summary = summary_for(data, case)
    mag_quality = magnitude_quality(data, case, summary)
    family = choose_unified_viability_family(magnitude_quality=mag_quality)
    selected = summary[case["family_method"]]
    magnitude = summary["magnitude"]
    return {
        "case": case["case"],
        "source": case["source"],
        "selected_family": family["selected_family"],
        "expected_family": case["expected_family"],
        "family_correct": family["selected_family"] == case["expected_family"],
        "family_method": case["family_method"],
        "magnitude_route_floor": family["magnitude_route_floor"],
        "route_floor_threshold": family["route_floor_threshold"],
        "magnitude_after": magnitude["after_mean"],
        "selected_after": selected["after_mean"],
        "selected_delta_vs_magnitude": selected["after_mean"] - magnitude["after_mean"],
        "selected_dead_outputs": selected["dead_outputs_mean"],
        "magnitude_dead_outputs": magnitude["dead_outputs_mean"],
    }


def run():
    cases = [evaluate_case(case) for case in CASES]
    result = {
        "experiment": "04_unified_viability_selector_retrospective",
        "setup": "Retrospective validation of a unified pre-finetune family selector: use ecology/homeostatic selection when magnitude has a dead route floor, use feature-preserving liveness repair when magnitude retains a route floor.",
        "cases": cases,
        "family_accuracy": sum(item["family_correct"] for item in cases) / len(cases),
        "selected_delta_mean": float(np.mean([item["selected_delta_vs_magnitude"] for item in cases])),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    return result


def write_markdown(result: dict) -> None:
    lines = [
        "# Unified Viability Selector Retrospective",
        "",
        result["setup"],
        "",
        f"Family-selection accuracy: `{result['family_accuracy']:.2f}`",
        "",
        "| Case | Selected family | Route floor | Selected method | Magnitude | Selected | Delta | Mag dead | Selected dead |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["cases"]:
        lines.append(
            f"| `{item['case']}` | `{item['selected_family']}` | `{item['magnitude_route_floor']:.4f}` | "
            f"`{item['family_method']}` | `{item['magnitude_after']:.4f}` | `{item['selected_after']:.4f}` | "
            f"`{item['selected_delta_vs_magnitude']:+.4f}` | `{item['magnitude_dead_outputs']:.1f}` | `{item['selected_dead_outputs']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The unified selector does not claim one pruning mask is universally best. It first chooses the family: dead route floor implies homeostatic/ecology repair; preserved route floor implies feature-preserving liveness repair. This cleanly separates from-scratch severe-pruning collapse from pretrained feature-subspace preservation.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"family_accuracy": result["family_accuracy"], "cases": result["cases"]}, indent=2))
