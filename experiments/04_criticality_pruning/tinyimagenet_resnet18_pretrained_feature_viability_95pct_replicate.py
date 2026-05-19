from __future__ import annotations

import json

import tinyimagenet_resnet18_pretrained_feature_viability_95pct as fv
import tinyimagenet_resnet18_pretrained_ecology_selector_99pct as pre


pre.SEED = 276
pre.SPARSITY = 0.95


def write_report(result):
    result["experiment"] = "04_tinyimagenet_resnet18_pretrained_feature_viability_95pct_replicate"
    result["setup"] = "Fresh-seed TinyImageNet-200 pretrained ResNet-18 95% sparsity replicate for magnitude-first feature-subspace preservation plus minimal liveness repair."
    out = pre.ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet18_pretrained_feature_viability_95pct_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = pre.ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_95PCT_REPLICATE.md"
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%: Replicate",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seed: `{result['seed']}`",
        f"Dense accuracy: `{result['dense_accuracy']:.4f}`",
        "",
        "| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in pre.METHODS:
        item = result["summary"][method]
        delta = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This fresh seed tests whether feature-viability repair consistently preserves pretrained magnitude-level recovery while eliminating dead outputs, or whether the first 95% result was seed-specific.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


fv.write_report = write_report


if __name__ == "__main__":
    result = fv.run()
    print(json.dumps({"dense_accuracy": result["dense_accuracy"], "summary": result["summary"]}, indent=2))
