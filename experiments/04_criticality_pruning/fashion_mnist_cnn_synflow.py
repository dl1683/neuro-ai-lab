from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PATH = Path(__file__).resolve().parent / "torch_fashion_mnist_cnn_dense_hybrid.py"
spec = importlib.util.spec_from_file_location("hybrid_base", HYBRID_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {HYBRID_PATH}")
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)
base = hybrid.base

SEEDS = [51, 52]
SPARSITIES = [0.90, 0.95, 0.98]
ALPHA_BY_SPARSITY = {0.90: 0.0, 0.95: 0.20, 0.98: 0.20}
METHODS = ["magnitude", "synflow", "adaptive_dense_hybrid"]


def _synflow_scores(model):
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 1, 28, 28, device=base.DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in base._params(model).items()}
    for name, param in model.named_parameters():
        param.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        fixed_scores = {
            "magnitude": {k: p.detach().abs().clone() for k, p in base._params(model).items()},
            "synflow": _synflow_scores(model),
        }
        for sparsity in SPARSITIES:
            score_by_method = {**fixed_scores, "adaptive_dense_hybrid": hybrid._hybrid_scores(model, calib_loader, ALPHA_BY_SPARSITY[sparsity])}
            for method in METHODS:
                masks = base._mask(score_by_method[method], sparsity)
                acc = base._evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "accuracy": acc, "retention": acc / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    paired = []
    for seed in SEEDS:
        for sparsity in SPARSITIES:
            adaptive = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "adaptive_dense_hybrid")
            synflow = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "synflow")
            magnitude = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
            paired.append({"seed": seed, "sparsity": sparsity, "adaptive_minus_synflow": adaptive["accuracy"] - synflow["accuracy"], "adaptive_minus_magnitude": adaptive["accuracy"] - magnitude["accuracy"]})
    result = {"experiment": "04_fashion_mnist_cnn_synflow_comparison", "setup": "Fashion-MNIST CNN comparison against SynFlow. Adaptive dense-hybrid uses magnitude conv layers and path-corrected dense layers at 95/98% sparsity.", "summary": summary, "paired_deltas": paired, "mean_adaptive_minus_synflow": float(np.mean([p["adaptive_minus_synflow"] for p in paired])), "mean_adaptive_minus_magnitude": float(np.mean([p["adaptive_minus_magnitude"] for p in paired])), "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "fashion_mnist_cnn_synflow_comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "FASHION_MNIST_CNN_SYNFLOW.md"
    lines = ["# Fashion-MNIST CNN SynFlow Comparison", "", result["setup"], "", "| Sparsity | Method | Mean accuracy | Std | Retention |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` |")
    lines.extend(["", f"Mean adaptive minus SynFlow: `{result['mean_adaptive_minus_synflow']:.4f}`.", f"Mean adaptive minus magnitude: `{result['mean_adaptive_minus_magnitude']:.4f}`."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "mean_adaptive_minus_synflow": result["mean_adaptive_minus_synflow"], "mean_adaptive_minus_magnitude": result["mean_adaptive_minus_magnitude"]}, indent=2))
