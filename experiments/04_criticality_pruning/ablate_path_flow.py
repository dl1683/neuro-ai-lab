from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("criticality_run", RUN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {RUN_PATH}")
criticality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(criticality)

SEEDS = [23, 24, 25, 26, 27]
SPARSITIES = [0.90, 0.95]
VARIANTS = [
    "magnitude",
    "gradient_saliency",
    "path_flow_full",
    "no_input_activity",
    "no_hidden_strength",
    "no_hidden_balance",
    "no_output_strength",
]


def _variant_scores(model, x_train, y_train, seed):
    input_flow, hidden_strength, hidden_balance, path_importance = criticality._feature_stats(model, x_train)
    output_strength = np.mean(np.abs(model["W2"]), axis=1) + 1e-6
    grads = criticality._gradients(model, x_train, y_train)

    def score(input_term, strength_term, balance_term, output_term):
        hidden = strength_term * balance_term * output_term
        return {
            "W1": np.abs(model["W1"]) * input_term[:, None] * hidden[None, :],
            "W2": np.abs(model["W2"]) * hidden[:, None],
        }

    ones_input = np.ones_like(input_flow)
    ones_hidden = np.ones_like(hidden_strength)
    return {
        "magnitude": {
            "W1": np.abs(model["W1"]),
            "W2": np.abs(model["W2"]),
        },
        "gradient_saliency": {
            "W1": np.abs(model["W1"] * grads["W1"]),
            "W2": np.abs(model["W2"] * grads["W2"]),
        },
        "path_flow_full": score(input_flow, hidden_strength, hidden_balance, output_strength),
        "no_input_activity": score(ones_input, hidden_strength, hidden_balance, output_strength),
        "no_hidden_strength": score(input_flow, ones_hidden, hidden_balance, output_strength),
        "no_hidden_balance": score(input_flow, hidden_strength, ones_hidden, output_strength),
        "no_output_strength": score(input_flow, hidden_strength, hidden_balance, ones_hidden),
    }


def run() -> dict:
    rows = []
    for seed in SEEDS:
        model, x_train, x_test, y_train, y_test = criticality._train(seed)
        dense_accuracy = criticality._accuracy(model, x_test, y_test)
        scores = _variant_scores(model, x_train, y_train, seed)
        for sparsity in SPARSITIES:
            for variant in VARIANTS:
                mask = criticality._mask_from_scores(scores[variant], sparsity, "path_flow")
                accuracy = criticality._accuracy(model, x_test, y_test, mask)
                rows.append(
                    {
                        "seed": seed,
                        "sparsity": sparsity,
                        "variant": variant,
                        "dense_accuracy": dense_accuracy,
                        "accuracy": accuracy,
                        "accuracy_retention": accuracy / dense_accuracy,
                        "hidden_coverage": criticality._hidden_coverage(mask),
                    }
                )

    summary = {}
    for variant in VARIANTS:
        variant_rows = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in variant_rows])),
            "accuracy_std": float(np.std([r["accuracy"] for r in variant_rows])),
            "retention_mean": float(np.mean([r["accuracy_retention"] for r in variant_rows])),
            "hidden_coverage_mean": float(np.mean([r["hidden_coverage"] for r in variant_rows])),
        }

    full = summary["path_flow_full"]["accuracy_mean"]
    ablation_drops = {
        variant: float(full - summary[variant]["accuracy_mean"])
        for variant in VARIANTS
        if variant.startswith("no_")
    }
    best_variant = max(summary, key=lambda v: summary[v]["accuracy_mean"])
    result = {
        "experiment": "04_path_flow_ablation",
        "setup": "Ablate label-free path-flow score ingredients at 90% and 95% sparsity across five trained sklearn-digits MLP seeds.",
        "variants": VARIANTS,
        "summary": summary,
        "ablation_drops_vs_full": ablation_drops,
        "best_variant": best_variant,
        "rows": rows,
        "interpretation": "Positive ablation_drop means removing that ingredient hurts path-flow accuracy. Compare path_flow_full to gradient_saliency to see whether the label-free score remains competitive with a label-dependent gradient baseline.",
    }
    out = ROOT / "results" / "04_criticality_pruning" / "path_flow_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = ROOT / "experiments" / "04_criticality_pruning" / "PATH_FLOW_ABLATION.md"
    lines = [
        "# Path-Flow Ablation",
        "",
        "This ablation tests which terms in the label-free path-flow pruning score matter at `90%` and `95%` sparsity across five trained sklearn-digits MLP seeds.",
        "",
        "## Mean accuracy by variant",
        "",
        "| Variant | Mean accuracy | Std | Mean retention | Mean hidden coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summary[variant]
        lines.append(f"| `{variant}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` | `{item['hidden_coverage_mean']:.4f}` |")
    lines.extend([
        "",
        "## Drops from full path-flow",
        "",
        "| Removed ingredient | Accuracy drop |",
        "|---|---:|",
    ])
    for variant, drop in ablation_drops.items():
        lines.append(f"| `{variant}` | `{drop:.4f}` |")
    lines.extend([
        "",
        f"Best variant: `{best_variant}`.",
        "",
        "Interpretation: if a removal improves accuracy, that ingredient is not helping in this benchmark. If removal hurts, it is carrying useful signal.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_variant": result["best_variant"], "summary": result["summary"], "ablation_drops_vs_full": result["ablation_drops_vs_full"]}, indent=2))
