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
METHODS = ["magnitude", "gradient_saliency", "path_flow"]


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _masked_train(model, masks, x_train, y_train, epochs=55):
    lr = 0.075
    for _ in range(epochs):
        h, logits = criticality._forward(model, x_train, masks)
        probs = _softmax(logits)
        yoh = np.zeros_like(probs)
        yoh[np.arange(len(y_train)), y_train] = 1.0
        dlogits = (probs - yoh) / len(y_train)
        dW2 = h.T @ dlogits + 0.0002 * model["W2"]
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ (model["W2"] * masks["W2"]).T
        dz = dh * (h > 0.0)
        dW1 = x_train.T @ dz + 0.0002 * model["W1"]
        db1 = dz.sum(axis=0)
        model["W1"] -= lr * dW1 * masks["W1"]
        model["b1"] -= lr * db1
        model["W2"] -= lr * dW2 * masks["W2"]
        model["b2"] -= lr * db2
        lr *= 0.99
    return model


def _copy_model(model):
    return {k: v.copy() for k, v in model.items()}


def run() -> dict:
    rows = []
    for seed in SEEDS:
        model, x_train, x_test, y_train, y_test = criticality._train(seed)
        dense_accuracy = criticality._accuracy(model, x_test, y_test)
        scores = criticality._scores(model, x_train, y_train, seed)
        for sparsity in SPARSITIES:
            for method in METHODS:
                score_key = method if method != "path_flow" else "path_flow"
                masks = criticality._mask_from_scores(scores[score_key], sparsity, "path_flow")
                before = criticality._accuracy(model, x_test, y_test, masks)
                tuned = _masked_train(_copy_model(model), masks, x_train, y_train)
                after = criticality._accuracy(tuned, x_test, y_test, masks)
                rows.append(
                    {
                        "seed": seed,
                        "sparsity": sparsity,
                        "method": method,
                        "dense_accuracy": dense_accuracy,
                        "accuracy_before_finetune": before,
                        "accuracy_after_finetune": after,
                        "recovery_gain": after - before,
                        "retention_after_finetune": after / dense_accuracy,
                    }
                )

    summary = {}
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        summary[method] = {
            "before_mean": float(np.mean([r["accuracy_before_finetune"] for r in method_rows])),
            "after_mean": float(np.mean([r["accuracy_after_finetune"] for r in method_rows])),
            "recovery_gain_mean": float(np.mean([r["recovery_gain"] for r in method_rows])),
            "retention_after_mean": float(np.mean([r["retention_after_finetune"] for r in method_rows])),
        }

    path_vs_mag = summary["path_flow"]["after_mean"] - summary["magnitude"]["after_mean"]
    path_vs_grad = summary["path_flow"]["after_mean"] - summary["gradient_saliency"]["after_mean"]
    result = {
        "experiment": "04_path_flow_finetune",
        "setup": "Compare masked post-prune fine-tuning for magnitude, gradient saliency, and label-free path-flow at 90% and 95% sparsity across five trained sklearn-digits MLP seeds.",
        "summary": summary,
        "path_flow_after_finetune_gain_vs_magnitude": float(path_vs_mag),
        "path_flow_after_finetune_gain_vs_gradient_saliency": float(path_vs_grad),
        "rows": rows,
    }

    out = ROOT / "results" / "04_criticality_pruning" / "path_flow_finetune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = ROOT / "experiments" / "04_criticality_pruning" / "PATH_FLOW_FINETUNE.md"
    lines = [
        "# Path-Flow Post-Prune Fine-Tuning",
        "",
        "This experiment checks whether path-flow masks remain good after masked fine-tuning, not just immediately after one-shot pruning.",
        "",
        "Setup: trained sklearn-digits MLPs, `90%` and `95%` sparsity, five seeds, 55 masked fine-tuning epochs.",
        "",
        "| Method | Before FT | After FT | Recovery gain | Retention after FT |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['recovery_gain_mean']:.4f}` | `{item['retention_after_mean']:.4f}` |")
    lines.extend([
        "",
        f"Path-flow after-FT gain vs magnitude: `{path_vs_mag:.4f}`.",
        f"Path-flow after-FT gain vs gradient saliency: `{path_vs_grad:.4f}`.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "path_flow_after_finetune_gain_vs_magnitude": result["path_flow_after_finetune_gain_vs_magnitude"], "path_flow_after_finetune_gain_vs_gradient_saliency": result["path_flow_after_finetune_gain_vs_gradient_saliency"]}, indent=2))
