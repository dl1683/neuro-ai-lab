from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "torch_fashion_mnist_cnn_path_flow.py"
spec = importlib.util.spec_from_file_location("cnn_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEEDS = [51, 52]
SPARSITIES = [0.90, 0.95, 0.98]
ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]


def _hybrid_scores(model, calib_loader, alpha: float):
    # Conv layers stay pure magnitude. Dense layers get corrected no-balance path modulation.
    model.eval()
    flats = []
    hiddens = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(base.DEVICE)
            _, acts = model(x, return_acts=True)
            flats.append(acts["flat"].detach().cpu())
            hiddens.append(acts["h"].detach().cpu())
    flat_all = torch.cat(flats, dim=0).to(base.DEVICE)
    h_all = torch.cat(hiddens, dim=0).to(base.DEVICE)
    flat_signal = flat_all.std(dim=0) + 1e-6
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    hidden = hidden_strength * output_strength
    return {
        "conv1": model.conv1.weight.detach().abs().clone(),
        "conv2": model.conv2.weight.detach().abs().clone(),
        "fc1": model.fc1.weight.detach().abs() * torch.pow(hidden[:, None] * flat_signal[None, :], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(hidden[None, :], alpha),
    }


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        for alpha in ALPHAS:
            scores = _hybrid_scores(model, calib_loader, alpha)
            for sparsity in SPARSITIES:
                masks = base._mask(scores, sparsity)
                accuracy = base._evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "alpha": alpha, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for alpha in ALPHAS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["alpha"] == alpha]
            summary[str(sparsity)][str(alpha)] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    alpha_scores = {str(alpha): float(np.mean([r["accuracy"] for r in rows if r["alpha"] == alpha])) for alpha in ALPHAS}
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    result = {"experiment": "04_torch_fashion_mnist_cnn_dense_hybrid", "setup": "Fashion-MNIST CNN hybrid pruning: conv layers use magnitude; dense layers use corrected no-balance path-flow blend. alpha=0 is pure magnitude everywhere.", "alpha_scores": alpha_scores, "best_alpha": best_alpha, "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_cnn_dense_hybrid.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_CNN_DENSE_HYBRID.md"
    lines = ["# Torch Fashion-MNIST CNN Dense-Hybrid Path Flow", "", result["setup"], "", f"Best alpha: `{best_alpha}`", "", "| Alpha | Mean accuracy over 90/95/98% |", "|---:|---:|"]
    for alpha in ALPHAS:
        lines.append(f"| `{alpha:.2f}` | `{alpha_scores[str(alpha)]:.4f}` |")
    lines.extend(["", "## Per-sparsity", "", "| Sparsity | Alpha | Accuracy | Std | Retention |", "|---:|---:|---:|---:|---:|"])
    for sparsity in SPARSITIES:
        for alpha in ALPHAS:
            item = summary[str(sparsity)][str(alpha)]
            lines.append(f"| `{sparsity:.2f}` | `{alpha:.2f}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_alpha": result["best_alpha"], "alpha_scores": result["alpha_scores"], "summary": result["summary"]}, indent=2))
