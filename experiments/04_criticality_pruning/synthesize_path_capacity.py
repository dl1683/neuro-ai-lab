from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
OUT_JSON = RESULTS / "path_capacity_synthesis.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "PATH_CAPACITY_SYNTHESIS.md"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def extract_case(source: str, sparsity: str | None, method: str, label: str, dead_key: str = "dead_fc1_hidden_mean"):
    data = load(source)
    summary = data["summary"] if sparsity is None else data["summary"][sparsity]
    mag = summary["magnitude"]
    syn = summary.get("global_synflow")
    item = summary[method]
    syn_after = None if syn is None else syn["after_mean"]
    return {
        "source": source,
        "label": label,
        "sparsity": data.get("sparsity", float(sparsity) if sparsity is not None else None),
        "method": method,
        "magnitude_after": mag["after_mean"],
        "global_synflow_after": syn_after,
        "method_after": item["after_mean"],
        "method_minus_magnitude_after": item["after_mean"] - mag["after_mean"],
        "method_minus_global_synflow_after": None if syn_after is None else item["after_mean"] - syn_after,
        "dead_metric": dead_key,
        "method_dead_units": item[dead_key],
        "global_synflow_dead_units": None if syn is None else syn[dead_key],
        "after_wins_vs_magnitude": None,
    }


def extract_nested_case(source: str, task: str, method: str, label: str, dead_key: str = "dead_outputs_mean"):
    data = load(source)
    summary = data["summary"][task]
    mag = summary["magnitude"]
    item = summary[method]
    return {
        "source": source,
        "label": label,
        "sparsity": data.get("sparsity"),
        "method": method,
        "magnitude_after": mag["after_mean"],
        "global_synflow_after": None,
        "method_after": item["after_mean"],
        "method_minus_magnitude_after": item["after_mean"] - mag["after_mean"],
        "method_minus_global_synflow_after": None,
        "dead_metric": dead_key,
        "method_dead_units": item[dead_key],
        "global_synflow_dead_units": None,
        "after_wins_vs_magnitude": next((p["after_wins"] for p in data["paired_deltas"][task] if p["method"] == method), None),
        "before_wins_vs_magnitude": None,
        "paired_seed_count": len(next((p["paired_rows"] for p in data["paired_deltas"][task] if p["method"] == method), [])),
    }


def extract_unified_selector_case(source: str, label: str):
    data = load(source)
    return {
        "source": source,
        "label": label,
        "sparsity": None,
        "method": "unified_selector",
        "magnitude_after": float(np.mean([item["magnitude_after"] for item in data["cases"]])),
        "global_synflow_after": None,
        "method_after": float(np.mean([item["selected_after"] for item in data["cases"]])),
        "method_minus_magnitude_after": data["selected_delta_mean"],
        "method_minus_global_synflow_after": None,
        "dead_metric": "selected_dead_outputs",
        "method_dead_units": float(np.mean([item["selected_dead_outputs"] for item in data["cases"]])),
        "global_synflow_dead_units": None,
        "after_wins_vs_magnitude": int(sum(item["selected_delta_vs_magnitude"] > 0 for item in data["cases"])),
        "before_wins_vs_magnitude": None,
        "paired_seed_count": len(data["cases"]),
    }


def attach_wins(item: dict, source: str, method: str, sparsity: float | None = None) -> dict:
    data = load(source)
    wins = next((
        p for p in data.get("paired_deltas", [])
        if p["method"] == method and (sparsity is None or float(p.get("sparsity", sparsity)) == sparsity)
    ), None)
    if wins is not None:
        item["after_wins_vs_magnitude"] = wins["after_wins"]
        item["before_wins_vs_magnitude"] = wins.get("before_wins")
        item["paired_seed_count"] = len(wins.get("paired_rows", []))
    elif method in data.get("summary", {}) and "after_wins" in data["summary"][method]:
        item["after_wins_vs_magnitude"] = data["summary"][method]["after_wins"]
        item["before_wins_vs_magnitude"] = None
        item["paired_seed_count"] = len(data.get("seeds", []))
    return item


def run():
    cases = [
        extract_case("cifar10_cnn_path_capacity_pruning.json", "0.98", "pathcap_synflow_bridge_mag", "single bridge capacity"),
        extract_case("cifar10_cnn_multicut_capacity_pruning.json", "0.98", "multicut_capacity", "multi-cut fixed floors"),
        extract_case("cifar10_cnn_multicut_capacity_pruning.json", "0.99", "multicut_capacity", "multi-cut fixed floors"),
        extract_case("cifar10_cnn_adaptive_capacity_pruning.json", "0.98", "adaptive_capacity", "adaptive output-count capacity"),
        extract_case("cifar10_cnn_adaptive_capacity_pruning.json", "0.99", "adaptive_capacity", "adaptive output-count capacity"),
        extract_case("cifar10_cnn_risk_adaptive_capacity_pruning.json", "0.99", "fixed_capacity", "fixed capacity replicate"),
        extract_case("cifar10_cnn_risk_adaptive_capacity_pruning.json", "0.99", "risk_adaptive_capacity", "dead-risk adaptive capacity"),
        extract_case("cifar10_cnn_mass_risk_capacity_pruning.json", "0.99", "fixed_capacity", "fixed capacity replicate 2"),
        extract_case("cifar10_cnn_mass_risk_capacity_pruning.json", "0.99", "mass_risk_capacity", "mass-risk adaptive capacity"),
        attach_wins(extract_case("cifar10_cnn_capacity_reserve_sweep_99pct.json", None, "reserve_0.60", "cifar reserve sweep best"), "cifar10_cnn_capacity_reserve_sweep_99pct.json", "reserve_0.60"),
        attach_wins(extract_case("cifar10_cnn_capacity_reserve_sweep_99pct.json", None, "reserve_0.50", "cifar reserve sweep lower band"), "cifar10_cnn_capacity_reserve_sweep_99pct.json", "reserve_0.50"),
        attach_wins(extract_case("cifar10_cnn_capacity_reserve_sweep_99pct.json", None, "reserve_0.65", "cifar reserve sweep upper band"), "cifar10_cnn_capacity_reserve_sweep_99pct.json", "reserve_0.65"),
        extract_case("fashion_mnist_cnn_capacity_transfer.json", "0.98", "reserve_0.60", "fashion transfer 98"),
        extract_case("fashion_mnist_cnn_capacity_transfer.json", "0.99", "reserve_0.60", "fashion transfer 99"),
        attach_wins(extract_case("cifar10_tiny_resnet_capacity_transfer.json", "0.98", "reserve_0.60", "tinyresnet transfer 98", "dead_outputs_mean"), "cifar10_tiny_resnet_capacity_transfer.json", "reserve_0.60", 0.98),
        attach_wins(extract_case("cifar10_tiny_resnet_capacity_transfer.json", "0.99", "reserve_0.60", "tinyresnet transfer 99", "dead_outputs_mean"), "cifar10_tiny_resnet_capacity_transfer.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_activation_capacity.json", "0.98", "activation_reserve_0.60", "tinyresnet activation reserve 98", "dead_outputs_mean"), "cifar10_tiny_resnet_activation_capacity.json", "activation_reserve_0.60", 0.98),
        attach_wins(extract_case("cifar10_tiny_resnet_activation_capacity.json", "0.99", "activation_reserve_0.60", "tinyresnet activation reserve 99", "dead_outputs_mean"), "cifar10_tiny_resnet_activation_capacity.json", "activation_reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_backbone_capacity.json", "0.98", "backbone_reserve_0.60", "tinyresnet backbone reserve 98", "dead_outputs_mean"), "cifar10_tiny_resnet_backbone_capacity.json", "backbone_reserve_0.60", 0.98),
        attach_wins(extract_case("cifar10_tiny_resnet_backbone_capacity.json", "0.99", "backbone_reserve_0.60", "tinyresnet backbone reserve 99", "dead_outputs_mean"), "cifar10_tiny_resnet_backbone_capacity.json", "backbone_reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_balanced_route_capacity.json", "0.98", "balanced_route_0.60", "tinyresnet balanced route 98", "dead_outputs_mean"), "cifar10_tiny_resnet_balanced_route_capacity.json", "balanced_route_0.60", 0.98),
        attach_wins(extract_case("cifar10_tiny_resnet_balanced_route_capacity.json", "0.99", "balanced_route_0.60", "tinyresnet balanced route 99", "dead_outputs_mean"), "cifar10_tiny_resnet_balanced_route_capacity.json", "balanced_route_0.60", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_balanced_route_99pct_replicate.json", None, "balanced_route_0.60", "tinyresnet balanced route 99 replicate", "dead_outputs_mean"), "cifar10_tiny_resnet_balanced_route_99pct_replicate.json", "balanced_route_0.60", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_route_split_sweep_99pct.json", None, "proj_readout_40_35_25", "tinyresnet projection-readout split 99", "dead_outputs_mean"), "cifar10_tiny_resnet_route_split_sweep_99pct.json", "proj_readout_40_35_25", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_predicted_route_split_99pct.json", None, "predicted_deficit_split", "tinyresnet predicted route split 99", "dead_outputs_mean"), "cifar10_tiny_resnet_predicted_route_split_99pct.json", "predicted_deficit_split", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_derived_route_predictors_99pct.json", None, "fixed_deficit_predictor", "tinyresnet fixed predictor fresh 99", "dead_outputs_mean"), "cifar10_tiny_resnet_derived_route_predictors_99pct.json", "fixed_deficit_predictor", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_target_matched_route_optimizer_99pct.json", None, "target_matched_optimizer", "tinyresnet target-matched optimizer 99", "dead_outputs_mean"), "cifar10_tiny_resnet_target_matched_route_optimizer_99pct.json", "target_matched_optimizer", 0.99),
        attach_wins(extract_case("cifar10_tiny_resnet_diversity_route_optimizer_99pct.json", None, "diversity_target_optimizer", "tinyresnet diversity optimizer 99", "dead_outputs_mean"), "cifar10_tiny_resnet_diversity_route_optimizer_99pct.json", "diversity_target_optimizer", 0.99),
        attach_wins(extract_case("cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct.json", None, "diversity_target_optimizer", "deep tinyresnet diversity optimizer 99", "dead_outputs_mean"), "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct.json", "diversity_target_optimizer", 0.99),
        attach_wins(extract_case("cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json", None, "reserve_0.60", "deep tinyresnet reserve replicate 99", "dead_outputs_mean"), "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_resnet20_capacity_99pct.json", None, "reserve_0.60", "resnet20-style reserve 99", "dead_outputs_mean"), "cifar10_resnet20_capacity_99pct.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_resnet20_capacity_99pct_replicate.json", None, "reserve_0.60", "resnet20-style reserve replicate 99", "dead_outputs_mean"), "cifar10_resnet20_capacity_99pct_replicate.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_capacity_99pct.json", None, "reserve_0.60", "full cifar resnet20-style reserve 99", "dead_outputs_mean"), "cifar10_full_resnet20_capacity_99pct.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_capacity_99pct_sixseed.json", None, "reserve_0.60", "full cifar resnet20-style reserve sixseed 99", "dead_outputs_mean"), "cifar10_full_resnet20_capacity_99pct_sixseed.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json", None, "reserve_0.60", "full cifar resnet20-style SGD reserve fourseed 99", "dead_outputs_mean"), "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json", None, "predicted_route_split", "full cifar resnet20-style predicted route split 99", "dead_outputs_mean"), "cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json", "predicted_route_split", 0.99),
        attach_wins(extract_case("cifar100_full_resnet20_capacity_99pct_sgd_recipe.json", None, "reserve_0.60", "full cifar100 resnet20-style SGD reserve 99", "dead_outputs_mean"), "cifar100_full_resnet20_capacity_99pct_sgd_recipe.json", "reserve_0.60", 0.99),
        attach_wins(extract_case("cifar100_full_resnet20_route_split_99pct_sgd_recipe.json", None, "readout_main_45_15_40", "full cifar100 readout-main route split 99", "dead_outputs_mean"), "cifar100_full_resnet20_route_split_99pct_sgd_recipe.json", "readout_main_45_15_40", 0.99),
        attach_wins(extract_case("cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json", None, "predicted_route_split", "full cifar100 predicted route split 99", "dead_outputs_mean"), "cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json", "predicted_route_split", 0.99),
        attach_wins(extract_case("cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json", None, "predicted_route_split", "full cifar100 conservative predicted route split 99", "dead_outputs_mean"), "cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json", "predicted_route_split", 0.99),
        attach_wins(extract_case("cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json", None, "tradeoff_policy", "full cifar100 tradeoff selector v1 99", "dead_outputs_mean"), "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json", "tradeoff_policy", 0.99),
        extract_case("cifar100_full_resnet20_tradeoff_selector_v2_policy.json", None, "tradeoff_v2_policy", "full cifar100 tradeoff selector v2 policy 99", "dead_outputs_mean"),
        attach_wins(extract_case("cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json", None, "tradeoff_policy", "full cifar100 tradeoff selector v2 prospective 99", "dead_outputs_mean"), "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json", "tradeoff_policy", 0.99),
        extract_nested_case("cifar_ecology_selector_99pct_sgd_recipe.json", "cifar10", "ecology_selected", "ecology selector cifar10 selected reserve 99"),
        extract_nested_case("cifar_ecology_selector_99pct_sgd_recipe.json", "cifar100", "ecology_selected", "ecology selector cifar100 selected split 99"),
        attach_wins(extract_case("cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json", None, "ecology_policy", "deep tinyresnet ecology selector policy 99", "dead_outputs_mean"), "cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json", "ecology_policy", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_ecology_selector_99pct_sgd40.json", None, "ecology_policy", "full cifar resnet20 ecology selector SGD40 99", "dead_outputs_mean"), "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json", "ecology_policy", 0.99),
        attach_wins(extract_case("cifar10_tiny_vit_circuit_viability_98pct.json", None, "minimal_liveness_repair", "tinyvit minimal liveness repair 98", "dead_outputs_mean"), "cifar10_tiny_vit_circuit_viability_98pct.json", "minimal_liveness_repair", 0.98),
        attach_wins(extract_case("cifar10_tiny_vit_circuit_viability_98pct.json", None, "mlp_readout_reserve", "tinyvit mlp-readout reserve 98", "dead_outputs_mean"), "cifar10_tiny_vit_circuit_viability_98pct.json", "mlp_readout_reserve", 0.98),
        attach_wins(extract_case("cifar10_tiny_vit_circuit_viability_95pct.json", None, "minimal_liveness_repair", "tinyvit minimal liveness repair 95", "dead_outputs_mean"), "cifar10_tiny_vit_circuit_viability_95pct.json", "minimal_liveness_repair", 0.95),
        attach_wins(extract_case("cifar10_tiny_vit_circuit_viability_95pct.json", None, "selective_mlp_readout_repair", "tinyvit selective mlp-readout repair 95", "dead_outputs_mean"), "cifar10_tiny_vit_circuit_viability_95pct.json", "selective_mlp_readout_repair", 0.95),
        attach_wins(extract_case("cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json", None, "global_synflow", "tinyvit feature-subspace diagnostic synflow 95", "dead_outputs_mean"), "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json", "global_synflow", 0.95),
        attach_wins(extract_case("cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json", None, "minimal_liveness_repair", "tinyvit feature-subspace diagnostic liveness 95", "dead_outputs_mean"), "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json", "minimal_liveness_repair", 0.95),
        attach_wins(extract_case("cifar10_tiny_vit_feature_subspace_selector_95pct.json", None, "feature_subspace_policy", "tinyvit feature-subspace selector 95", "dead_outputs_mean"), "cifar10_tiny_vit_feature_subspace_selector_95pct.json", "feature_subspace_policy", 0.95),
        extract_case("cifar10_tiny_vit_feature_route_margin_policy_95pct.json", None, "feature_route_margin_policy", "tinyvit feature-route margin policy 95", "dead_outputs_mean"),
        attach_wins(extract_case("cifar10_tiny_vit_feature_route_margin_selector_95pct.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector prospective 95", "dead_outputs_mean"), "cifar10_tiny_vit_feature_route_margin_selector_95pct.json", "feature_route_margin_policy", 0.95),
        attach_wins(extract_case("cifar10_tiny_vit_feature_route_margin_selector_90pct.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector prospective 90", "dead_outputs_mean"), "cifar10_tiny_vit_feature_route_margin_selector_90pct.json", "feature_route_margin_policy", 0.90),
        extract_case("cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector strong pilot 90", "dead_outputs_mean"),
        extract_case("cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v2 strong 90", "dead_outputs_mean"),
        extract_case("cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v2 strong replicate 90", "dead_outputs_mean"),
        extract_case("cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v3 strong 90", "dead_outputs_mean"),
        attach_wins(extract_case("cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v3 strong replicate 90", "dead_outputs_mean"), "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json", "feature_route_margin_policy", 0.90),
        attach_wins(extract_case("cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v4 strong 90", "dead_outputs_mean"), "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json", "feature_route_margin_policy", 0.90),
        attach_wins(extract_case("cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json", None, "feature_route_margin_policy", "tinyvit feature-route margin selector v4 seed306 90", "dead_outputs_mean"), "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json", "feature_route_margin_policy", 0.90),
        extract_case("tinyimagenet_resnet20_ecology_selector_99pct.json", None, "ecology_policy", "tinyimagenet ecology selector external proxy 99", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json", None, "ecology_policy", "tinyimagenet pretrained resnet18 ecology selector 99", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json", None, "ecology_policy", "tinyimagenet pretrained resnet18 ecology selector 95", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_feature_viability_95pct.json", None, "feature_viability_repair", "tinyimagenet pretrained feature-viability repair 95", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json", None, "feature_viability_repair", "tinyimagenet pretrained feature-viability repair twoseed 95", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_feature_viability_99pct.json", None, "feature_viability_repair", "tinyimagenet pretrained feature-viability repair 99", "dead_outputs_mean"),
        extract_case("tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json", None, "tradeoff_policy", "tinyimagenet pretrained tradeoff selector 95", "dead_outputs_mean"),
        attach_wins(extract_case("cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json", None, "feature_viability_repair", "full cifar resnet20 feature-vs-homeostasis 99", "dead_outputs_mean"), "cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json", "feature_viability_repair", 0.99),
        attach_wins(extract_case("cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json", None, "tradeoff_policy", "full cifar resnet20 tradeoff selector 99", "dead_outputs_mean"), "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json", "tradeoff_policy", 0.99),
        extract_unified_selector_case("unified_viability_selector_retrospective.json", "unified viability selector retrospective"),
    ]
    best_vs_magnitude = max(cases, key=lambda item: item["method_minus_magnitude_after"])
    rescue_cases = [case for case in cases if case["method_minus_global_synflow_after"] is not None]
    best_rescue = max(rescue_cases, key=lambda item: item["method_minus_global_synflow_after"])
    positive_cases = [case for case in cases if case["method_minus_magnitude_after"] > 0]
    result = {
        "experiment": "04_path_capacity_synthesis",
        "claim": "Circuit-capacity constraints can convert global SynFlow's severe-sparsity cutset collapse into trainable masks under the same parameter budget. The strongest current CIFAR-10 CNN result is a 99% reserve sweep where reserve_0.60 beats magnitude by +2.56 points after fine-tuning and wins 4/4 paired seeds; a broad 0.45-0.65 reserve band beats magnitude on mean. Fashion-MNIST CNN transfer also beats magnitude on mean at 98% and 99%. Residual transfer is now positive in custom and standard-style settings: route-quality targets plus a degeneracy penalty beat magnitude on TinyResNet, broad reserve capacity beats magnitude on DeepTinyResNet, reserve beats magnitude by +5.97 points on a ResNet-20-style subset replicate, and remains positive on full CIFAR-10. The latest full-CIFAR tradeoff selector replaces the old route-floor-only family rule with a feature-preservation/liveness score and selects the best method on two fresh seeds.",
        "cases": cases,
        "positive_vs_magnitude_cases": len(positive_cases),
        "best_vs_magnitude": best_vs_magnitude,
        "best_rescue_vs_global_synflow": best_rescue,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Path-Capacity Synthesis", "", result["claim"], "", "## Cases", "", "| Label | Sparsity | Method | Magnitude after FT | Global SynFlow after FT | Method after FT | Delta vs magnitude | Rescue vs SynFlow | Wins | Dead units | Dead metric |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def fmt_optional(value):
        return "n/a" if value is None else f"{value:.4f}"

    def fmt_signed_optional(value):
        return "n/a" if value is None else f"{value:+.4f}"

    def fmt_sparsity(value):
        return "n/a" if value is None else f"{value:.2f}"

    for item in cases:
        paired_seed_count = item.get("paired_seed_count", 4)
        wins = "" if item.get("after_wins_vs_magnitude") is None else f"{item['after_wins_vs_magnitude']}/{paired_seed_count}"
        lines.append(f"| {item['label']} | `{fmt_sparsity(item['sparsity'])}` | `{item['method']}` | `{item['magnitude_after']:.4f}` | `{fmt_optional(item['global_synflow_after'])}` | `{item['method_after']:.4f}` | `{item['method_minus_magnitude_after']:+.4f}` | `{fmt_signed_optional(item['method_minus_global_synflow_after'])}` | `{wins}` | `{item['method_dead_units']:.1f}` | `{item['dead_metric']}` |")
    lines.extend([
        "",
        "## Best current result",
        "",
        f"Best delta vs magnitude: `{best_vs_magnitude['label']}` at `{best_vs_magnitude['sparsity']:.2f}` with `{best_vs_magnitude['method_minus_magnitude_after']:+.4f}` after-FT accuracy.",
        f"Best rescue vs global SynFlow: `{best_rescue['label']}` at `{best_rescue['sparsity']:.2f}` with `{best_rescue['method_minus_global_synflow_after']:+.4f}` after-FT accuracy.",
        f"Positive cases vs magnitude in this synthesis: `{len(positive_cases)}/{len(cases)}`.",
        "",
        "## Readout",
        "",
        "Path-capacity constraints reliably prevent dense-bridge death and rescue SynFlow collapse. The strongest CIFAR evidence is the `99%` reserve sweep: a broad reserve band from `0.45` through `0.65` beats magnitude on mean, and `reserve_0.60` beats magnitude by `+2.56` points after fine-tuning with `4/4` paired wins.",
        "",
        "The constructive result also transfers to Fashion-MNIST CNN: `reserve_0.60` beats magnitude on mean at both `98%` and `99%`, with the larger gain at the harsher `99%` cliff.",
        "",
        "TinyResNet is the first residual transfer test and gives a mixed result: capacity reserve beats magnitude at `98%` but trails at `99%` even while nearly eliminating dead outputs. That is a useful boundary condition: the next method must estimate residual-route quality, not merely output liveness.",
        "",
        "The activation-supported TinyResNet follow-up is a negative result. Multiplying protected-capacity rank by presynaptic activation did not outperform plain reserve capacity and worsened the `99%` gap. Simple use-dependent stabilization is therefore too local; the residual case likely needs path interaction, block-level cut geometry, or post-residual route diversity.",
        "",
        "The residual-backbone TinyResNet follow-up improves the `98%` residual case but worsens the `99%` cliff. Projection shortcuts matter, but simply overprotecting them is not enough; at extreme sparsity it appears to steal capacity from other required routes.",
        "",
        "The balanced residual route allocator is a useful but incomplete residual advance. It uses the route-quality audit to split protected capacity across main transformations, projection shortcuts, and classifier readout. In the first two-seed TinyResNet run it beat magnitude at `99%`, but a four-seed fresh replicate did not confirm that win: balanced route capacity improved over plain reserve by `+4.01` points but still trailed magnitude by `-1.53` points. The residual gap is narrowed, not solved.",
        "",
        "A targeted projection/readout split sweep then closed the residual gap in the same four-seed `99%` setting. The `40/35/25` main/projection/readout split beats magnitude by `+0.745` points and wins `3/4` seeds, while plain reserve trails magnitude by `-4.36` points. This is the strongest current evidence that residual path-capacity needs balanced route-family allocation, not output liveness alone.",
        "",
        "The first route-deficit predictor is small but important. It compares a magnitude viability template to the plain reserve candidate before fine-tuning, then allocates projection/readout capacity from the measured deficits. Follow-ups showed that target matching alone overconcentrates projection capacity and trails magnitude. Adding a degeneracy-style diversity penalty fixes that failure in the latest batch: the diversity optimizer beats magnitude by `+1.05` points and wins `4/4` seeds while plain reserve trails by `-4.67` points. The residual result is now a concrete route-target plus degeneracy mechanism, though penalty weights are still hand-set.",
        "",
        "The deeper residual transfer is also positive and now has a four-seed replicate. On DeepTinyResNet at `99%`, magnitude leaves hundreds of dead outputs. The replicate shows plain reserve at `30.20%` versus magnitude at `28.48%`, with `3/4` wins. This suggests the mechanism is not confined to the original TinyResNet, but the hierarchy is clearer: broad homeostatic capacity dominates in deeper residuals, while route-family diversity matters most once liveness is no longer the limiting factor.",
        "",
        "The feature-vs-homeostasis tests correct the earlier selector. Route-floor-only family selection is too crude: feature-preserving liveness repair can beat broad reserve even in a from-scratch full-CIFAR ResNet-20 setting where magnitude has a dead main-path floor. A new tradeoff selector scores realized candidate masks by feature overlap, liveness, readout preservation, and dead-output penalty. On two fresh full-CIFAR seeds, it selected feature repair before fine-tuning and beat magnitude by `+5.37` points with `2/2` wins, while also beating plain reserve on mean. On a fresh pretrained TinyImageNet ResNet-18 seed at `95%`, the same selector avoided the homeostatic masks that collapsed performance, selected feature repair, removed all dead outputs, and slightly beat magnitude. CIFAR-100 exposed the next correction: V1 still overweighted feature preservation, while a V2 task-ecology pressure term selected route split on both projection and fresh prospective seeds. In the fresh prospective run, V2 selected the best mean candidate and beat magnitude by `+2.72` points with `2/2` wins. TinyViT now defines the transformer analogue but also the limitation: row-liveness repairs remove MLP/attention route death but do not dominate. In weak dense TinyViT runs, feature-route margin selectors select SynFlow and beat magnitude at both `95%` and `90%`. Stronger full-train TinyViT pilots show a different regime: V1 overselects SynFlow, V2 corrects one seed by selecting magnitude, a replicate favors all-route liveness, and V3 identifies the complementary feature-dominant regime. Across three fresh strong V3 seeds, the feature-margin branch selects SynFlow and beats magnitude; the two-seed replicate improves mean recovery by `+6.98` points with `2/2` wins. V4 adds masked pre-finetune accuracy as a trainability diagnostic and on its first fresh seed again selects the feature-dominant SynFlow branch, beating magnitude by `+6.87` points. A fresh non-SynFlow V4 seed falsifies the perfect projection: the selected attention+MLP repair beats magnitude by `+0.59` points, but SynFlow beats it by `+1.71` points. The strong-transformer selector is improving but remains unsolved: it needs a predictive feature/liveness/trainability policy that can recognize when low feature alignment SynFlow still has stronger recovery.",
        "",
        "This is still not a final theory. The next scientific step is to predict the useful reserve band and feature/liveness tradeoff from route-quality features instead of sweeping or hand-weighting it."
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["best_vs_magnitude"], indent=2))
