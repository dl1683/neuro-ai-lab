from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "torch_fashion_mnist_path_flow.py"
spec = importlib.util.spec_from_file_location("fashion_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SPARSITIES = [0.90, 0.95, 0.98]
SEEDS = [31, 32]
METHODS = ["magnitude_global", "path_flow_global", "magnitude_layerwise", "gradient_layerwise", "path_flow_layerwise"]


def _layerwise_mask(scores, sparsity):
    masks = {}
    for key in ["fc1", "fc2"]:
        flat = scores[key].flatten()
        keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
        threshold = torch.topk(flat, keep, largest=True).values.min()
        masks[key] = (scores[key] >= threshold).float()
    return masks


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.MLP().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        scores = {
            "magnitude": base._magnitude_scores(model),
            "gradient": base._gradient_scores(model, calib_loader),
            "path_flow": base._path_flow_scores(model, calib_loader),
        }
        for sparsity in SPARSITIES:
            method_masks = {
                "magnitude_global": base._mask(scores["magnitude"], sparsity),
                "path_flow_global": base._mask(scores["path_flow"], sparsity),
                "magnitude_layerwise": _layerwise_mask(scores["magnitude"], sparsity),
                "gradient_layerwise": _layerwise_mask(scores["gradient"], sparsity),
                "path_flow_layerwise": _layerwise_mask(scores["path_flow"], sparsity),
            }
            for method, masks in method_masks.items():
                acc = base._evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "accuracy": acc, "retention": acc / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    result = {"experiment": "04_torch_fashion_mnist_layerwise", "setup": "Check whether Fashion-MNIST path-flow failure is caused by global pruning allocation by comparing global vs layerwise masks.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_layerwise.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_LAYERWISE.md"
    lines = ["# Torch Fashion-MNIST Layerwise Pruning Check", "", result["setup"], "", "| Sparsity | Method | Mean accuracy | Std | Retention |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
