from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
OUT = ROOT / "docs" / "CLAIM_AUDIT.md"


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    source: str
    actual: float | int | str
    expected: float | int | str
    tolerance: float = 1e-4

    def passed(self) -> bool:
        if isinstance(self.actual, str) or isinstance(self.expected, str):
            return self.actual == self.expected
        return abs(float(self.actual) - float(self.expected)) <= self.tolerance


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def close(actual, expected, name, source, tol=1e-4):
    return ClaimCheck(name=name, source=source, actual=actual, expected=expected, tolerance=tol)


def checks():
    out: list[ClaimCheck] = []

    syn = load("synflow_pathology_synthesis.json")
    out.append(close(syn["aggregate"]["global_synflow_zero_fc1_cases"], 3, "SynFlow zero bridge cases", "synflow_pathology_synthesis.json"))
    out.append(close(round(syn["aggregate"]["global_synflow_after_delta_mean"] * 100, 2), -42.80, "SynFlow mean after-FT delta vs magnitude points", "synflow_pathology_synthesis.json", 0.01))

    cifar_sweep = load("cifar10_cnn_capacity_reserve_sweep_99pct.json")
    out.append(close(cifar_sweep["summary"]["magnitude"]["after_mean"], 0.32315, "CIFAR CNN 99 magnitude after", "cifar10_cnn_capacity_reserve_sweep_99pct.json"))
    out.append(close(cifar_sweep["summary"]["reserve_0.60"]["after_mean"], 0.34875, "CIFAR CNN 99 reserve 0.60 after", "cifar10_cnn_capacity_reserve_sweep_99pct.json"))
    out.append(close(next(x for x in cifar_sweep["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 4, "CIFAR CNN 99 reserve 0.60 wins", "cifar10_cnn_capacity_reserve_sweep_99pct.json"))

    fashion = load("fashion_mnist_cnn_capacity_transfer.json")
    out.append(close(fashion["summary"]["0.99"]["magnitude"]["after_mean"], 0.8024375, "Fashion 99 magnitude after", "fashion_mnist_cnn_capacity_transfer.json"))
    out.append(close(fashion["summary"]["0.99"]["reserve_0.60"]["after_mean"], 0.8175, "Fashion 99 reserve 0.60 after", "fashion_mnist_cnn_capacity_transfer.json"))

    tiny_div = load("cifar10_tiny_resnet_diversity_route_optimizer_99pct.json")
    out.append(close(tiny_div["summary"]["magnitude"]["after_mean"], 0.24795, "TinyResNet diversity batch magnitude after", "cifar10_tiny_resnet_diversity_route_optimizer_99pct.json"))
    out.append(close(tiny_div["summary"]["diversity_target_optimizer"]["after_mean"], 0.25845, "TinyResNet diversity optimizer after", "cifar10_tiny_resnet_diversity_route_optimizer_99pct.json"))
    out.append(close(next(x for x in tiny_div["paired_deltas"] if x["method"] == "diversity_target_optimizer")["after_wins"], 4, "TinyResNet diversity optimizer wins", "cifar10_tiny_resnet_diversity_route_optimizer_99pct.json"))

    deep_rep = load("cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json")
    out.append(close(deep_rep["summary"]["magnitude"]["after_mean"], 0.28475, "DeepTinyResNet replicate magnitude after", "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json"))
    out.append(close(deep_rep["summary"]["reserve_0.60"]["after_mean"], 0.302, "DeepTinyResNet replicate reserve after", "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json"))
    out.append(close(next(x for x in deep_rep["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 3, "DeepTinyResNet replicate reserve wins", "cifar10_deep_tiny_resnet_diversity_route_optimizer_99pct_replicate.json"))

    resnet20 = load("cifar10_resnet20_capacity_99pct.json")
    out.append(close(resnet20["summary"]["magnitude"]["after_mean"], 0.2848, "ResNet20-style magnitude after", "cifar10_resnet20_capacity_99pct.json"))
    out.append(close(resnet20["summary"]["reserve_0.60"]["after_mean"], 0.3241, "ResNet20-style reserve after", "cifar10_resnet20_capacity_99pct.json"))
    out.append(close(resnet20["summary"]["reserve_0.60"]["dead_outputs_mean"], 0.0, "ResNet20-style reserve dead outputs", "cifar10_resnet20_capacity_99pct.json"))
    out.append(close(next(x for x in resnet20["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 2, "ResNet20-style reserve wins", "cifar10_resnet20_capacity_99pct.json"))

    resnet20_rep = load("cifar10_resnet20_capacity_99pct_replicate.json")
    out.append(close(resnet20_rep["summary"]["magnitude"]["after_mean"], 0.27245, "ResNet20-style replicate magnitude after", "cifar10_resnet20_capacity_99pct_replicate.json"))
    out.append(close(resnet20_rep["summary"]["reserve_0.60"]["after_mean"], 0.33215, "ResNet20-style replicate reserve after", "cifar10_resnet20_capacity_99pct_replicate.json"))
    out.append(close(resnet20_rep["summary"]["reserve_0.60"]["dead_outputs_mean"], 0.25, "ResNet20-style replicate reserve dead outputs", "cifar10_resnet20_capacity_99pct_replicate.json"))
    out.append(close(next(x for x in resnet20_rep["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 4, "ResNet20-style replicate reserve wins", "cifar10_resnet20_capacity_99pct_replicate.json"))

    full_resnet20 = load("cifar10_full_resnet20_capacity_99pct.json")
    out.append(close(full_resnet20["summary"]["magnitude"]["after_mean"], 0.3801, "Full CIFAR ResNet20-style magnitude after", "cifar10_full_resnet20_capacity_99pct.json"))
    out.append(close(full_resnet20["summary"]["reserve_0.60"]["after_mean"], 0.39605, "Full CIFAR ResNet20-style reserve after", "cifar10_full_resnet20_capacity_99pct.json"))
    out.append(close(full_resnet20["summary"]["reserve_0.60"]["dead_outputs_mean"], 0.0, "Full CIFAR ResNet20-style reserve dead outputs", "cifar10_full_resnet20_capacity_99pct.json"))
    out.append(close(next(x for x in full_resnet20["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 2, "Full CIFAR ResNet20-style reserve wins", "cifar10_full_resnet20_capacity_99pct.json"))
    full_resnet20_sixseed = load("cifar10_full_resnet20_capacity_99pct_sixseed.json")
    out.append(close(full_resnet20_sixseed["summary"]["magnitude"]["after_mean"], 0.37716666666666665, "Full CIFAR ResNet20-style sixseed magnitude after", "cifar10_full_resnet20_capacity_99pct_sixseed.json"))
    out.append(close(full_resnet20_sixseed["summary"]["reserve_0.60"]["after_mean"], 0.3921333333333334, "Full CIFAR ResNet20-style sixseed reserve after", "cifar10_full_resnet20_capacity_99pct_sixseed.json"))
    out.append(close(full_resnet20_sixseed["summary"]["reserve_0.60"]["dead_outputs_mean"], 0.0, "Full CIFAR ResNet20-style sixseed reserve dead outputs", "cifar10_full_resnet20_capacity_99pct_sixseed.json"))
    out.append(close(next(x for x in full_resnet20_sixseed["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 6, "Full CIFAR ResNet20-style sixseed reserve wins", "cifar10_full_resnet20_capacity_99pct_sixseed.json"))
    full_resnet20_sgd = load("cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json")
    out.append(close(full_resnet20_sgd["summary"]["magnitude"]["after_mean"], 0.42867500000000003, "Full CIFAR ResNet20-style SGD fourseed magnitude after", "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json"))
    out.append(close(full_resnet20_sgd["summary"]["reserve_0.60"]["after_mean"], 0.49434999999999996, "Full CIFAR ResNet20-style SGD fourseed reserve after", "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json"))
    out.append(close(full_resnet20_sgd["summary"]["reserve_0.60"]["dead_outputs_mean"], 1.75, "Full CIFAR ResNet20-style SGD fourseed reserve dead outputs", "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json"))
    out.append(close(next(x for x in full_resnet20_sgd["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 4, "Full CIFAR ResNet20-style SGD fourseed reserve wins", "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json"))
    cifar10_pred = load("cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json")
    out.append(close(cifar10_pred["summary"]["plain_reserve"]["after_mean"], 0.47875, "CIFAR10 full ResNet20-style predicted batch plain reserve after", "cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"))
    out.append(close(cifar10_pred["summary"]["predicted_route_split"]["after_mean"], 0.46915, "CIFAR10 full ResNet20-style predicted route split after", "cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in cifar10_pred["paired_deltas"] if x["method"] == "predicted_route_split")["after_wins"], 2, "CIFAR10 full ResNet20-style predicted route split wins", "cifar10_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"))
    cifar100 = load("cifar100_full_resnet20_capacity_99pct_sgd_recipe.json")
    out.append(close(cifar100["summary"]["magnitude"]["after_mean"], 0.0658, "CIFAR100 full ResNet20-style SGD magnitude after", "cifar100_full_resnet20_capacity_99pct_sgd_recipe.json"))
    out.append(close(cifar100["summary"]["reserve_0.60"]["after_mean"], 0.0764, "CIFAR100 full ResNet20-style SGD reserve after", "cifar100_full_resnet20_capacity_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in cifar100["paired_deltas"] if x["method"] == "reserve_0.60")["after_wins"], 2, "CIFAR100 full ResNet20-style SGD reserve wins", "cifar100_full_resnet20_capacity_99pct_sgd_recipe.json"))
    cifar100_route = load("cifar100_full_resnet20_route_split_99pct_sgd_recipe.json")
    out.append(close(cifar100_route["summary"]["readout_main_45_15_40"]["after_mean"], 0.09115000000000001, "CIFAR100 readout-main route split after", "cifar100_full_resnet20_route_split_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in cifar100_route["paired_deltas"] if x["method"] == "readout_main_45_15_40")["after_wins"], 2, "CIFAR100 readout-main route split wins", "cifar100_full_resnet20_route_split_99pct_sgd_recipe.json"))
    cifar100_pred = load("cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json")
    out.append(close(cifar100_pred["summary"]["predicted_route_split"]["after_mean"], 0.09805, "CIFAR100 predicted route split after", "cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in cifar100_pred["paired_deltas"] if x["method"] == "predicted_route_split")["after_wins"], 2, "CIFAR100 predicted route split wins", "cifar100_full_resnet20_predicted_route_split_99pct_sgd_recipe.json"))
    cifar100_conservative = load("cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json")
    out.append(close(cifar100_conservative["summary"]["plain_reserve"]["after_mean"], 0.07680000000000001, "CIFAR100 conservative predictor plain reserve after", "cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json"))
    out.append(close(cifar100_conservative["summary"]["predicted_route_split"]["after_mean"], 0.08785, "CIFAR100 conservative predicted route split after", "cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in cifar100_conservative["paired_deltas"] if x["method"] == "predicted_route_split")["after_wins"], 2, "CIFAR100 conservative predicted route split wins", "cifar100_full_resnet20_conservative_predicted_route_split_99pct_sgd_recipe.json"))
    ecology = load("cifar_ecology_selector_99pct_sgd_recipe.json")
    out.append(close(ecology["summary"]["cifar10"]["ecology_selected"]["after_mean"], 0.46740000000000004, "Ecology selector CIFAR10 selected after", "cifar_ecology_selector_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in ecology["paired_deltas"]["cifar10"] if x["method"] == "ecology_selected")["after_wins"], 2, "Ecology selector CIFAR10 selected wins", "cifar_ecology_selector_99pct_sgd_recipe.json"))
    out.append(close(ecology["summary"]["cifar100"]["ecology_selected"]["after_mean"], 0.09079999999999999, "Ecology selector CIFAR100 selected after", "cifar_ecology_selector_99pct_sgd_recipe.json"))
    out.append(close(next(x for x in ecology["paired_deltas"]["cifar100"] if x["method"] == "ecology_selected")["after_wins"], 2, "Ecology selector CIFAR100 selected wins", "cifar_ecology_selector_99pct_sgd_recipe.json"))
    deep_policy = load("cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json")
    out.append(close(deep_policy["summary"]["ecology_policy"]["after_mean"], 0.315, "DeepTinyResNet ecology policy after", "cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json"))
    out.append(close(next(x for x in deep_policy["paired_deltas"] if x["method"] == "ecology_policy")["after_wins"], 2, "DeepTinyResNet ecology policy wins", "cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json"))
    sgd40 = load("cifar10_full_resnet20_ecology_selector_99pct_sgd40.json")
    out.append(close(sgd40["summary"]["magnitude"]["after_mean"], 0.48729999999999996, "CIFAR10 full ResNet20 SGD40 magnitude after", "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json"))
    out.append(close(sgd40["summary"]["ecology_policy"]["after_mean"], 0.52625, "CIFAR10 full ResNet20 SGD40 ecology policy after", "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json"))
    out.append(close(next(x for x in sgd40["paired_deltas"] if x["method"] == "ecology_policy")["after_wins"], 2, "CIFAR10 full ResNet20 SGD40 ecology policy wins", "cifar10_full_resnet20_ecology_selector_99pct_sgd40.json"))
    tiny = load("tinyimagenet_resnet20_ecology_selector_99pct.json")
    out.append(close(tiny["summary"]["magnitude"]["after_mean"], 0.0232, "TinyImageNet external proxy magnitude after", "tinyimagenet_resnet20_ecology_selector_99pct.json"))
    out.append(close(tiny["summary"]["plain_reserve"]["after_mean"], 0.0308, "TinyImageNet external proxy reserve after", "tinyimagenet_resnet20_ecology_selector_99pct.json"))
    out.append(close(tiny["summary"]["ecology_policy"]["after_mean"], 0.029, "TinyImageNet external proxy ecology policy after", "tinyimagenet_resnet20_ecology_selector_99pct.json"))
    tiny_pre_99 = load("tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json")
    out.append(close(tiny_pre_99["dense_accuracy"], 0.5983333333333334, "TinyImageNet pretrained ResNet18 99 dense", "tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json"))
    out.append(close(tiny_pre_99["summary"]["magnitude"]["after_mean"], 0.010666666666666666, "TinyImageNet pretrained ResNet18 99 magnitude after", "tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json"))
    out.append(close(tiny_pre_99["summary"]["ecology_policy"]["after_mean"], 0.008, "TinyImageNet pretrained ResNet18 99 ecology policy after", "tinyimagenet_resnet18_pretrained_ecology_selector_99pct.json"))
    tiny_pre_95 = load("tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json")
    out.append(close(tiny_pre_95["dense_accuracy"], 0.6076666666666667, "TinyImageNet pretrained ResNet18 95 dense", "tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json"))
    out.append(close(tiny_pre_95["summary"]["magnitude"]["after_mean"], 0.15233333333333332, "TinyImageNet pretrained ResNet18 95 magnitude after", "tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json"))
    out.append(close(tiny_pre_95["summary"]["ecology_policy"]["after_mean"], 0.03966666666666667, "TinyImageNet pretrained ResNet18 95 ecology policy after", "tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json"))
    tiny_feature = load("tinyimagenet_resnet18_pretrained_feature_viability_95pct.json")
    out.append(close(tiny_feature["dense_accuracy"], 0.6106666666666667, "TinyImageNet pretrained feature viability dense", "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json"))
    out.append(close(tiny_feature["summary"]["magnitude"]["after_mean"], 0.15133333333333332, "TinyImageNet pretrained feature viability magnitude after", "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json"))
    out.append(close(tiny_feature["summary"]["feature_viability_repair"]["after_mean"], 0.15033333333333335, "TinyImageNet pretrained feature viability repair after", "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json"))
    out.append(close(tiny_feature["summary"]["feature_viability_repair"]["dead_outputs_mean"], 0, "TinyImageNet pretrained feature viability repair dead outputs", "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json"))
    tiny_feature_99 = load("tinyimagenet_resnet18_pretrained_feature_viability_99pct.json")
    out.append(close(tiny_feature_99["dense_accuracy"], 0.595, "TinyImageNet pretrained feature viability 99 dense", "tinyimagenet_resnet18_pretrained_feature_viability_99pct.json"))
    out.append(close(tiny_feature_99["summary"]["magnitude"]["after_mean"], 0.012, "TinyImageNet pretrained feature viability 99 magnitude after", "tinyimagenet_resnet18_pretrained_feature_viability_99pct.json"))
    out.append(close(tiny_feature_99["summary"]["feature_viability_repair"]["after_mean"], 0.014333333333333333, "TinyImageNet pretrained feature viability 99 repair after", "tinyimagenet_resnet18_pretrained_feature_viability_99pct.json"))
    out.append(close(tiny_feature_99["summary"]["feature_viability_repair"]["dead_outputs_mean"], 4, "TinyImageNet pretrained feature viability 99 repair dead outputs", "tinyimagenet_resnet18_pretrained_feature_viability_99pct.json"))
    tiny_feature_95_two = load("tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json")
    out.append(close(tiny_feature_95_two["summary"]["magnitude"]["after_mean"], 0.14866666666666667, "TinyImageNet pretrained feature viability 95 twoseed magnitude after", "tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json"))
    out.append(close(tiny_feature_95_two["summary"]["feature_viability_repair"]["after_mean"], 0.14816666666666667, "TinyImageNet pretrained feature viability 95 twoseed repair after", "tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json"))
    out.append(close(tiny_feature_95_two["summary"]["feature_viability_repair"]["dead_outputs_mean"], 0.0, "TinyImageNet pretrained feature viability 95 twoseed repair dead outputs", "tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json"))
    unified = load("unified_viability_selector_retrospective.json")
    out.append(close(unified["family_accuracy"], 1.0, "Unified viability selector family accuracy", "unified_viability_selector_retrospective.json"))
    out.append(close(len(unified["cases"]), 6, "Unified viability selector case count", "unified_viability_selector_retrospective.json"))
    feature_vs_homeostasis = load("cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json")
    out.append(close(feature_vs_homeostasis["summary"]["feature_viability_repair"]["after_mean"], 0.49260000000000004, "CIFAR10 feature-vs-homeostasis feature repair after", "cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json"))
    out.append(close(feature_vs_homeostasis["summary"]["plain_reserve"]["after_mean"], 0.4778, "CIFAR10 feature-vs-homeostasis plain reserve after", "cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json"))
    out.append(close(next(x for x in feature_vs_homeostasis["paired_deltas"] if x["method"] == "feature_viability_repair")["after_wins"], 2, "CIFAR10 feature-vs-homeostasis feature repair wins", "cifar10_full_resnet20_feature_vs_homeostasis_99pct_sgd20.json"))
    tradeoff = load("cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json")
    out.append(close(tradeoff["summary"]["tradeoff_policy"]["after_mean"], 0.49970000000000003, "CIFAR10 tradeoff selector policy after", "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(tradeoff["summary"]["tradeoff_policy"]["after_delta_mean"], 0.05374999999999999, "CIFAR10 tradeoff selector policy delta", "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(next(x for x in tradeoff["paired_deltas"] if x["method"] == "tradeoff_policy")["after_wins"], 2, "CIFAR10 tradeoff selector policy wins", "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(tradeoff["decisions"][0]["selected_method"], "feature_viability_repair", "CIFAR10 tradeoff selector seed 279 decision", "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(tradeoff["decisions"][1]["selected_method"], "feature_viability_repair", "CIFAR10 tradeoff selector seed 280 decision", "cifar10_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    tiny_tradeoff = load("tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json")
    out.append(close(tiny_tradeoff["summary"]["tradeoff_policy"]["after_mean"], 0.157, "TinyImageNet pretrained tradeoff selector policy after", "tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json"))
    out.append(close(tiny_tradeoff["summary"]["tradeoff_policy"]["after_delta_mean"], 0.0003333333333333244, "TinyImageNet pretrained tradeoff selector policy delta", "tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json"))
    out.append(close(tiny_tradeoff["summary"]["tradeoff_policy"]["dead_outputs_mean"], 0, "TinyImageNet pretrained tradeoff selector dead outputs", "tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json"))
    out.append(close(tiny_tradeoff["decision"]["selected_method"], "feature_viability_repair", "TinyImageNet pretrained tradeoff selector decision", "tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json"))
    cifar100_tradeoff = load("cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json")
    out.append(close(cifar100_tradeoff["summary"]["tradeoff_policy"]["after_mean"], 0.0821, "CIFAR100 tradeoff selector v1 policy after", "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(cifar100_tradeoff["summary"]["predicted_route_split"]["after_mean"], 0.08865, "CIFAR100 tradeoff selector route split candidate after", "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    out.append(close(cifar100_tradeoff["decisions"][0]["selected_method"], "feature_viability_repair", "CIFAR100 tradeoff selector v1 seed 282 decision", "cifar100_full_resnet20_tradeoff_selector_99pct_sgd20.json"))
    cifar100_v2 = load("cifar100_full_resnet20_tradeoff_selector_v2_policy.json")
    out.append(close(cifar100_v2["summary"]["tradeoff_v2_policy"]["after_mean"], 0.08865, "CIFAR100 tradeoff selector v2 policy after", "cifar100_full_resnet20_tradeoff_selector_v2_policy.json"))
    out.append(close(cifar100_v2["summary"]["tradeoff_v2_policy"]["after_delta_mean"], 0.0189, "CIFAR100 tradeoff selector v2 policy delta", "cifar100_full_resnet20_tradeoff_selector_v2_policy.json"))
    out.append(close(cifar100_v2["decisions"][0]["v2_selected_method"], "predicted_route_split", "CIFAR100 tradeoff selector v2 seed 282 decision", "cifar100_full_resnet20_tradeoff_selector_v2_policy.json"))
    out.append(close(cifar100_v2["decisions"][1]["v2_selected_method"], "predicted_route_split", "CIFAR100 tradeoff selector v2 seed 283 decision", "cifar100_full_resnet20_tradeoff_selector_v2_policy.json"))
    cifar100_v2_fresh = load("cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json")
    out.append(close(cifar100_v2_fresh["summary"]["tradeoff_policy"]["after_mean"], 0.0926, "CIFAR100 tradeoff selector v2 fresh policy after", "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"))
    out.append(close(cifar100_v2_fresh["summary"]["tradeoff_policy"]["after_delta_mean"], 0.027249999999999996, "CIFAR100 tradeoff selector v2 fresh policy delta", "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"))
    out.append(close(cifar100_v2_fresh["summary"]["tradeoff_policy"]["after_wins"], 2, "CIFAR100 tradeoff selector v2 fresh wins", "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"))
    out.append(close(cifar100_v2_fresh["decisions"][0]["selected_method"], "predicted_route_split", "CIFAR100 tradeoff selector v2 fresh seed 284 decision", "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"))
    out.append(close(cifar100_v2_fresh["decisions"][1]["selected_method"], "predicted_route_split", "CIFAR100 tradeoff selector v2 fresh seed 285 decision", "cifar100_full_resnet20_tradeoff_selector_v2_99pct_sgd20.json"))
    tinyvit = load("cifar10_tiny_vit_circuit_viability_98pct.json")
    out.append(close(tinyvit["dense_accuracy_mean"], 0.5467, "TinyViT dense accuracy mean", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["magnitude"]["mlp_down_dead_outputs_mean"], 512.0, "TinyViT magnitude MLP-down dead outputs", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["minimal_liveness_repair"]["after_mean"], 0.1131, "TinyViT minimal liveness repair after", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["minimal_liveness_repair"]["mlp_down_dead_outputs_mean"], 0.0, "TinyViT minimal liveness repair MLP-down dead outputs", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["selective_mlp_readout_repair"]["after_mean"], 0.0959, "TinyViT selective MLP-readout repair after", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["selective_mlp_readout_repair"]["mlp_down_dead_outputs_mean"], 0.0, "TinyViT selective MLP-readout repair MLP-down dead outputs", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["all_route_liveness_floor"]["after_mean"], 0.0996, "TinyViT all-route liveness floor after", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["all_route_liveness_floor"]["dead_outputs_mean"], 0.0, "TinyViT all-route liveness floor dead outputs", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    out.append(close(tinyvit["summary"]["mlp_readout_reserve"]["after_delta_mean"], -0.0011999999999999997, "TinyViT MLP-readout reserve delta", "cifar10_tiny_vit_circuit_viability_98pct.json"))
    tinyvit95 = load("cifar10_tiny_vit_circuit_viability_95pct.json")
    out.append(close(tinyvit95["summary"]["magnitude"]["mlp_down_dead_outputs_mean"], 512.0, "TinyViT 95 magnitude MLP-down dead outputs", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["magnitude"]["attn_out_dead_outputs_mean"], 314.5, "TinyViT 95 magnitude attention-output dead outputs", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["minimal_liveness_repair"]["after_mean"], 0.10869999999999999, "TinyViT 95 minimal liveness repair after", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["minimal_liveness_repair"]["after_wins"], 2, "TinyViT 95 minimal liveness wins", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["attn_mlp_readout_repair"]["attn_out_dead_outputs_mean"], 17.5, "TinyViT 95 attention-MLP repair attention dead outputs", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["attn_mlp_readout_repair"]["after_mean"], 0.1031, "TinyViT 95 attention-MLP repair after", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    out.append(close(tinyvit95["summary"]["selective_mlp_readout_repair"]["after_delta_mean"], 0.003600000000000006, "TinyViT 95 selective repair delta", "cifar10_tiny_vit_circuit_viability_95pct.json"))
    tinyvit_feature = load("cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json")
    out.append(close(tinyvit_feature["summary"]["global_synflow"]["after_mean"], 0.1686, "TinyViT feature diagnostic SynFlow after", "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json"))
    out.append(close(tinyvit_feature["summary"]["global_synflow"]["after_wins"], 2, "TinyViT feature diagnostic SynFlow wins", "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json"))
    out.append(close(tinyvit_feature["summary"]["minimal_liveness_repair"]["after_delta_mean"], -0.007099999999999995, "TinyViT feature diagnostic liveness delta", "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json"))
    out.append(close(tinyvit_feature["diagnostics"]["centered_cls_cosine_vs_after_corr"], 0.5829819280466383, "TinyViT centered CLS cosine correlation", "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json"))
    tinyvit_selector = load("cifar10_tiny_vit_feature_subspace_selector_95pct.json")
    out.append(close(tinyvit_selector["summary"]["feature_subspace_policy"]["after_mean"], 0.1224, "TinyViT feature-subspace policy after", "cifar10_tiny_vit_feature_subspace_selector_95pct.json"))
    out.append(close(tinyvit_selector["summary"]["feature_subspace_policy"]["after_delta_mean"], 0.034199999999999994, "TinyViT feature-subspace policy delta", "cifar10_tiny_vit_feature_subspace_selector_95pct.json"))
    out.append(close(tinyvit_selector["decisions"][0]["selected_method"], "global_synflow", "TinyViT feature-subspace selector seed 292 decision", "cifar10_tiny_vit_feature_subspace_selector_95pct.json"))
    tinyvit_margin = load("cifar10_tiny_vit_feature_route_margin_policy_95pct.json")
    out.append(close(tinyvit_margin["summary"]["feature_route_margin_policy"]["after_mean"], 0.1357, "TinyViT feature-route margin policy after", "cifar10_tiny_vit_feature_route_margin_policy_95pct.json"))
    out.append(close(tinyvit_margin["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.0475, "TinyViT feature-route margin policy delta", "cifar10_tiny_vit_feature_route_margin_policy_95pct.json"))
    out.append(close(tinyvit_margin["decisions"][1]["selected_method"], "global_synflow", "TinyViT feature-route margin seed 293 decision", "cifar10_tiny_vit_feature_route_margin_policy_95pct.json"))
    tinyvit_margin_fresh = load("cifar10_tiny_vit_feature_route_margin_selector_95pct.json")
    out.append(close(tinyvit_margin_fresh["summary"]["feature_route_margin_policy"]["after_mean"], 0.14600000000000002, "TinyViT feature-route margin fresh policy after", "cifar10_tiny_vit_feature_route_margin_selector_95pct.json"))
    out.append(close(tinyvit_margin_fresh["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.0565, "TinyViT feature-route margin fresh policy delta", "cifar10_tiny_vit_feature_route_margin_selector_95pct.json"))
    out.append(close(tinyvit_margin_fresh["summary"]["feature_route_margin_policy"]["after_wins"], 2, "TinyViT feature-route margin fresh wins", "cifar10_tiny_vit_feature_route_margin_selector_95pct.json"))
    out.append(close(tinyvit_margin_fresh["decisions"][0]["selected_method"], "global_synflow", "TinyViT feature-route margin fresh seed 294 decision", "cifar10_tiny_vit_feature_route_margin_selector_95pct.json"))
    out.append(close(tinyvit_margin_fresh["decisions"][1]["selected_method"], "global_synflow", "TinyViT feature-route margin fresh seed 295 decision", "cifar10_tiny_vit_feature_route_margin_selector_95pct.json"))
    tinyvit_margin_90 = load("cifar10_tiny_vit_feature_route_margin_selector_90pct.json")
    out.append(close(tinyvit_margin_90["summary"]["feature_route_margin_policy"]["after_mean"], 0.1456, "TinyViT feature-route margin 90 policy after", "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"))
    out.append(close(tinyvit_margin_90["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.04239999999999999, "TinyViT feature-route margin 90 policy delta", "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"))
    out.append(close(tinyvit_margin_90["summary"]["feature_route_margin_policy"]["after_wins"], 2, "TinyViT feature-route margin 90 wins", "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"))
    out.append(close(tinyvit_margin_90["decisions"][0]["selected_method"], "global_synflow", "TinyViT feature-route margin 90 seed 296 decision", "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"))
    out.append(close(tinyvit_margin_90["decisions"][1]["selected_method"], "global_synflow", "TinyViT feature-route margin 90 seed 297 decision", "cifar10_tiny_vit_feature_route_margin_selector_90pct.json"))
    tinyvit_strong = load("cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json")
    out.append(close(tinyvit_strong["dense_accuracy_mean"], 0.7162, "TinyViT strong dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"))
    out.append(close(tinyvit_strong["summary"]["magnitude"]["after_mean"], 0.167, "TinyViT strong magnitude after", "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"))
    out.append(close(tinyvit_strong["summary"]["feature_route_margin_policy"]["after_delta_mean"], -0.02350000000000002, "TinyViT strong feature-route policy delta", "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"))
    out.append(close(tinyvit_strong["decisions"][0]["selected_method"], "global_synflow", "TinyViT strong selector decision", "cifar10_tiny_vit_feature_route_margin_selector_90pct_strong.json"))
    tinyvit_v2_strong = load("cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json")
    out.append(close(tinyvit_v2_strong["dense_accuracy_mean"], 0.7248, "TinyViT V2 strong dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"))
    out.append(close(tinyvit_v2_strong["summary"]["feature_route_margin_policy"]["after_mean"], 0.1218, "TinyViT V2 strong policy after", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"))
    out.append(close(tinyvit_v2_strong["decisions"][0]["selected_method"], "magnitude", "TinyViT V2 strong selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"))
    out.append(close(tinyvit_v2_strong["decisions"][0]["reason"], "magnitude_trainable_capacity_guardrail", "TinyViT V2 strong selector reason", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong.json"))
    tinyvit_v2_rep = load("cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json")
    out.append(close(tinyvit_v2_rep["dense_accuracy_mean"], 0.7275, "TinyViT V2 strong replicate dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v2_rep["summary"]["all_route_liveness_floor"]["after_mean"], 0.1484, "TinyViT V2 strong replicate all-route after", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v2_rep["summary"]["feature_route_margin_policy"]["after_mean"], 0.1382, "TinyViT V2 strong replicate policy after", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v2_rep["decisions"][0]["selected_method"], "magnitude", "TinyViT V2 strong replicate selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v2_90pct_strong_replicate.json"))
    tinyvit_v3_strong = load("cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json")
    out.append(close(tinyvit_v3_strong["dense_accuracy_mean"], 0.7282, "TinyViT V3 strong dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    out.append(close(tinyvit_v3_strong["summary"]["feature_route_margin_policy"]["after_mean"], 0.1452, "TinyViT V3 strong policy after", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    out.append(close(tinyvit_v3_strong["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.031299999999999994, "TinyViT V3 strong policy delta", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    out.append(close(tinyvit_v3_strong["summary"]["all_route_liveness_floor"]["dead_outputs_mean"], 0.0, "TinyViT V3 strong all-route dead outputs", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    out.append(close(tinyvit_v3_strong["decisions"][0]["selected_method"], "global_synflow", "TinyViT V3 strong selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    out.append(close(tinyvit_v3_strong["decisions"][0]["reason"], "feature_argmax", "TinyViT V3 strong selector reason", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong.json"))
    tinyvit_v3_rep = load("cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json")
    out.append(close(tinyvit_v3_rep["dense_accuracy_mean"], 0.71765, "TinyViT V3 strong replicate dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["summary"]["feature_route_margin_policy"]["after_mean"], 0.15045, "TinyViT V3 strong replicate policy after", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.0698, "TinyViT V3 strong replicate policy delta", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["summary"]["feature_route_margin_policy"]["after_wins"], 2, "TinyViT V3 strong replicate wins", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["summary"]["all_route_liveness_floor"]["dead_outputs_mean"], 0.0, "TinyViT V3 strong replicate all-route dead outputs", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["decisions"][0]["selected_method"], "global_synflow", "TinyViT V3 strong replicate seed 302 decision", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    out.append(close(tinyvit_v3_rep["decisions"][1]["selected_method"], "global_synflow", "TinyViT V3 strong replicate seed 303 decision", "cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong_replicate.json"))
    tinyvit_v4 = load("cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json")
    out.append(close(tinyvit_v4["dense_accuracy_mean"], 0.7263, "TinyViT V4 strong dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"))
    out.append(close(tinyvit_v4["summary"]["feature_route_margin_policy"]["after_mean"], 0.1709, "TinyViT V4 strong policy after", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"))
    out.append(close(tinyvit_v4["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.0687, "TinyViT V4 strong policy delta", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"))
    out.append(close(tinyvit_v4["decisions"][0]["selected_method"], "global_synflow", "TinyViT V4 strong selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"))
    out.append(close(tinyvit_v4["decisions"][0]["reason"], "feature_argmax", "TinyViT V4 strong selector reason", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"))
    tinyvit_v4_seed306 = load("cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json")
    out.append(close(tinyvit_v4_seed306["dense_accuracy_mean"], 0.7299, "TinyViT V4 seed306 dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"))
    out.append(close(tinyvit_v4_seed306["summary"]["feature_route_margin_policy"]["after_mean"], 0.1158, "TinyViT V4 seed306 policy after", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"))
    out.append(close(tinyvit_v4_seed306["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.0059000000000000025, "TinyViT V4 seed306 policy delta", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"))
    out.append(close(tinyvit_v4_seed306["summary"]["global_synflow"]["after_mean"], 0.1329, "TinyViT V4 seed306 SynFlow after", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"))
    out.append(close(tinyvit_v4_seed306["decisions"][0]["selected_method"], "attn_mlp_readout_repair", "TinyViT V4 seed306 selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong_seed306.json"))
    tinyvit_v5 = load("cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json")
    out.append(close(tinyvit_v5["dense_accuracy_mean"], 0.7254, "TinyViT V5 strong dense accuracy", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json"))
    out.append(close(tinyvit_v5["summary"]["feature_route_margin_policy"]["after_mean"], 0.1018, "TinyViT V5 strong policy after", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json"))
    out.append(close(tinyvit_v5["summary"]["feature_route_margin_policy"]["after_delta_mean"], 0.029300000000000007, "TinyViT V5 strong policy delta", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json"))
    out.append(close(tinyvit_v5["decisions"][0]["selected_method"], "global_synflow", "TinyViT V5 strong selector decision", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json"))
    tinyvit_boundary = load("tiny_vit_strong_selector_boundary_synthesis.json")
    out.append(close(tinyvit_boundary["seed_count"], 9, "TinyViT strong boundary seed count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v3_positive_vs_magnitude"], 7, "TinyViT strong boundary positive count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v3_matches_best"], 6, "TinyViT strong boundary best-match count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v3_mean_delta_vs_magnitude"], 0.031122222222222225, "TinyViT strong boundary mean delta", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v3_mean_gap_to_best"], 0.002444444444444445, "TinyViT strong boundary mean oracle gap", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v4_positive_vs_magnitude"], 7, "TinyViT V4 boundary positive count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v4_matches_best"], 8, "TinyViT V4 boundary best-match count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v4_mean_delta_vs_magnitude"], 0.03166666666666667, "TinyViT V4 boundary mean delta", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v4_mean_gap_to_best"], 0.001899999999999999, "TinyViT V4 boundary mean oracle gap", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v5_positive_vs_magnitude"], 7, "TinyViT V5 boundary positive count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v5_matches_best"], 9, "TinyViT V5 boundary best-match count", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v5_mean_delta_vs_magnitude"], 0.033566666666666665, "TinyViT V5 boundary mean delta", "tiny_vit_strong_selector_boundary_synthesis.json"))
    out.append(close(tinyvit_boundary["v5_mean_gap_to_best"], 0.0, "TinyViT V5 boundary mean oracle gap", "tiny_vit_strong_selector_boundary_synthesis.json"))

    synthesis = load("path_capacity_synthesis.json")
    out.append(close(synthesis["best_vs_magnitude"]["label"], "tinyvit feature-route margin selector v3 strong replicate 90", "Path-capacity synthesis best label", "path_capacity_synthesis.json"))
    out.append(close(synthesis["best_vs_magnitude"]["method_minus_magnitude_after"], 0.0698, "Path-capacity synthesis best delta", "path_capacity_synthesis.json"))
    return out


def write_report(items: list[ClaimCheck]):
    passed = [item for item in items if item.passed()]
    failed = [item for item in items if not item.passed()]
    lines = [
        "# Claim Audit",
        "",
        "This file is generated by `experiments/04_criticality_pruning/audit_circuit_viability_claims.py`.",
        "",
        f"Passed: `{len(passed)}/{len(items)}`",
        "",
        "| Claim | Source | Expected | Actual | Status |",
        "|---|---|---:|---:|---|",
    ]
    for item in items:
        status = "PASS" if item.passed() else "FAIL"
        lines.append(f"| {item.name} | `{item.source}` | `{item.expected}` | `{item.actual}` | `{status}` |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return failed


def main():
    items = checks()
    failed = write_report(items)
    print(json.dumps({"passed": len(items) - len(failed), "total": len(items), "failed": [item.name for item in failed]}, indent=2))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
