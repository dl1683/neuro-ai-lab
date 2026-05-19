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
SPARSITIES = [0.97, 0.98, 0.99]
METHODS = ["magnitude", "gradient_saliency", "path_flow"]


def run():
    rows = []
    for seed in SEEDS:
        model, x_train, x_test, y_train, y_test = criticality._train(seed)
        dense_accuracy = criticality._accuracy(model, x_test, y_test)
        scores = criticality._scores(model, x_train, y_train, seed)
        for sparsity in SPARSITIES:
            for method in METHODS:
                mask = criticality._mask_from_scores(scores[method], sparsity, "path_flow")
                accuracy = criticality._accuracy(model, x_test, y_test, mask)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy, "hidden_coverage": criticality._hidden_coverage(mask)})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected])), "hidden_coverage_mean": float(np.mean([r["hidden_coverage"] for r in selected]))}
    result = {"experiment": "04_path_flow_extreme_sparsity", "setup": "One-shot pruning at 97%, 98%, and 99% sparsity on trained sklearn-digits MLPs across five seeds.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "path_flow_extreme_sparsity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "PATH_FLOW_EXTREME_SPARSITY.md"
    lines = ["# Path-Flow Extreme Sparsity", "", "One-shot pruning at `97%`, `98%`, and `99%` sparsity on trained sklearn-digits MLPs across five seeds.", "", "| Sparsity | Method | Mean accuracy | Std | Mean retention | Hidden coverage |", "|---:|---|---:|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` | `{item['hidden_coverage_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
