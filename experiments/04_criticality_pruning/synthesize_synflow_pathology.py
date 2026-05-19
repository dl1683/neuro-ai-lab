from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
OUT_JSON = RESULTS / "synflow_pathology_synthesis.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "SYNFLOW_PATHOLOGY_SYNTHESIS.md"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fashion_cnn_items():
    one_shot = load("fashion_mnist_cnn_synflow_comparison.json")
    ft = load("synflow_finetune_98pct_comparison.json")
    forensic = load("synflow_cnn_mask_forensics_98pct.json")
    rescue = load("synflow_cnn_layerwise_rescue_98pct.json")
    item = {
        "dataset_model": "Fashion-MNIST CNN",
        "sparsity": 0.98,
        "global_synflow_before": ft["summary"]["Fashion-MNIST CNN"]["synflow"]["before_mean"],
        "global_synflow_after": ft["summary"]["Fashion-MNIST CNN"]["synflow"]["after_mean"],
        "magnitude_before": ft["summary"]["Fashion-MNIST CNN"]["magnitude"]["before_mean"],
        "magnitude_after": ft["summary"]["Fashion-MNIST CNN"]["magnitude"]["after_mean"],
        "adaptive_before": ft["summary"]["Fashion-MNIST CNN"]["adaptive_path"]["before_mean"],
        "adaptive_after": ft["summary"]["Fashion-MNIST CNN"]["adaptive_path"]["after_mean"],
        "global_synflow_fc1_keep_rate": forensic["summary"]["synflow"]["layer_keep_rate_mean"]["fc1"],
        "global_synflow_dead_fc1_hidden": forensic["summary"]["synflow"]["damage_mean"]["dead_fc1_hidden"],
        "fc1_hidden_units": 128,
        "layerwise_synflow_after": rescue["summary"]["layerwise_synflow"]["after_mean"],
        "layerwise_synflow_fc1_keep_rate": rescue["summary"]["layerwise_synflow"]["fc1_keep_rate_mean"],
        "layerwise_synflow_dead_fc1_hidden": rescue["summary"]["layerwise_synflow"]["fc1_dead_hidden_mean"],
    }
    item["global_synflow_after_delta_vs_magnitude"] = item["global_synflow_after"] - item["magnitude_after"]
    item["layerwise_synflow_after_delta_vs_magnitude"] = item["layerwise_synflow_after"] - item["magnitude_after"]
    return [item]


def cifar_items():
    data = load("cifar10_cnn_synflow_pathology.json")
    items = []
    for sparsity in ["0.98", "0.99"]:
        summary = data["summary"][sparsity]
        item = {
            "dataset_model": "CIFAR-10 CNN",
            "sparsity": float(sparsity),
            "global_synflow_before": summary["global_synflow"]["before_mean"],
            "global_synflow_after": summary["global_synflow"]["after_mean"],
            "magnitude_before": summary["magnitude"]["before_mean"],
            "magnitude_after": summary["magnitude"]["after_mean"],
            "adaptive_before": None,
            "adaptive_after": None,
            "global_synflow_fc1_keep_rate": summary["global_synflow"]["fc1_keep_rate_mean"],
            "global_synflow_dead_fc1_hidden": summary["global_synflow"]["dead_fc1_hidden_mean"],
            "fc1_hidden_units": 192,
            "layerwise_synflow_after": summary["layerwise_synflow"]["after_mean"],
            "layerwise_synflow_fc1_keep_rate": summary["layerwise_synflow"]["fc1_keep_rate_mean"],
            "layerwise_synflow_dead_fc1_hidden": summary["layerwise_synflow"]["dead_fc1_hidden_mean"],
        }
        item["global_synflow_after_delta_vs_magnitude"] = item["global_synflow_after"] - item["magnitude_after"]
        item["layerwise_synflow_after_delta_vs_magnitude"] = item["layerwise_synflow_after"] - item["magnitude_after"]
        items.append(item)
    return items


def run():
    items = fashion_cnn_items() + cifar_items()
    global_after_deltas = [item["global_synflow_after_delta_vs_magnitude"] for item in items]
    layerwise_after_deltas = [item["layerwise_synflow_after_delta_vs_magnitude"] for item in items]
    result = {
        "experiment": "04_synflow_pathology_synthesis",
        "claim": "Global SynFlow can catastrophically starve dense classifier bridges in CNNs at severe global sparsity; layerwise SynFlow partially repairs allocation but remains far below magnitude after masked fine-tuning in the tested CNNs.",
        "items": items,
        "aggregate": {
            "cases": len(items),
            "global_synflow_zero_fc1_cases": int(sum(item["global_synflow_fc1_keep_rate"] == 0 for item in items)),
            "global_synflow_after_delta_mean": float(np.mean(global_after_deltas)),
            "global_synflow_after_delta_min": float(np.min(global_after_deltas)),
            "global_synflow_after_delta_max": float(np.max(global_after_deltas)),
            "layerwise_synflow_after_delta_mean": float(np.mean(layerwise_after_deltas)),
            "layerwise_synflow_after_delta_min": float(np.min(layerwise_after_deltas)),
            "layerwise_synflow_after_delta_max": float(np.max(layerwise_after_deltas)),
        },
        "source_files": [
            "fashion_mnist_cnn_synflow_comparison.json",
            "synflow_finetune_98pct_comparison.json",
            "synflow_cnn_mask_forensics_98pct.json",
            "synflow_cnn_layerwise_rescue_98pct.json",
            "cifar10_cnn_synflow_pathology.json",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# SynFlow Pathology Synthesis", "", result["claim"], "", "## Cross-dataset cases", "", "| Dataset/model | Sparsity | Magnitude after FT | Global SynFlow after FT | Global delta | Global fc1 keep | Global dead fc1 | Layerwise after FT | Layerwise delta |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in items:
        lines.append(
            f"| {item['dataset_model']} | `{item['sparsity']:.2f}` | `{item['magnitude_after']:.4f}` | `{item['global_synflow_after']:.4f}` | `{item['global_synflow_after_delta_vs_magnitude']:+.4f}` | `{item['global_synflow_fc1_keep_rate']:.4f}` | `{item['global_synflow_dead_fc1_hidden']:.1f}/{item['fc1_hidden_units']}` | `{item['layerwise_synflow_after']:.4f}` | `{item['layerwise_synflow_after_delta_vs_magnitude']:+.4f}` |"
        )
    agg = result["aggregate"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Cases: `{agg['cases']}`.",
        f"- Global SynFlow zero-fc1 cases: `{agg['global_synflow_zero_fc1_cases']}/{agg['cases']}`.",
        f"- Mean global SynFlow after-FT delta vs magnitude: `{agg['global_synflow_after_delta_mean']:+.4f}`.",
        f"- Mean layerwise SynFlow after-FT delta vs magnitude: `{agg['layerwise_synflow_after_delta_mean']:+.4f}`.",
        "",
        "## Interpretation",
        "",
        "The failure is structural, not just a modest score-quality difference. In every synthesized severe-sparsity CNN case, global SynFlow assigns zero parameters to `fc1`, so the dense classifier bridge is absent and masked fine-tuning cannot recover. Layerwise SynFlow restores a nominal per-layer budget, but still trails magnitude badly, which means the SynFlow ranking inside the dense bridge is also weak in these settings.",
        "",
        "Practical guardrail: severe global pruning methods should emit dense-bridge diagnostics before their scores are trusted: per-layer keep rate, dead hidden bridge units, and after-prune reachability."
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["aggregate"], indent=2))
