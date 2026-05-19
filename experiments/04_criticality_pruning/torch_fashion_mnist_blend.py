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
ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


def _blended_scores(model, calib_loader, alpha):
    # alpha=0 is pure magnitude. alpha=1 is full path-flow modulation.
    model.eval()
    inputs = []
    hidden = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(base.DEVICE)
            _, flat, h = model(x, return_hidden=True)
            inputs.append(flat.abs().cpu())
            hidden.append(h.cpu())
    input_activity = torch.cat(inputs, dim=0).mean(dim=0).to(base.DEVICE) + 1e-6
    h_all = torch.cat(hidden, dim=0).to(base.DEVICE)
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    hidden_fire = (h_all > 0).float().mean(dim=0) + 1e-6
    hidden_balance = torch.exp(-torch.abs(hidden_fire - 0.35) / 0.22)
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    path = hidden_strength * hidden_balance * output_strength
    return {
        "fc1": model.fc1.weight.detach().abs() * torch.pow(input_activity[None, :] * path[:, None], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(path[None, :], alpha),
    }


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.MLP().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        for alpha in ALPHAS:
            scores = _blended_scores(model, calib_loader, alpha)
            for sparsity in SPARSITIES:
                mask = base._mask(scores, sparsity)
                acc = base._evaluate(model, test_loader, mask)
                rows.append({"seed": seed, "alpha": alpha, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": acc, "retention": acc / dense_accuracy})
    summary = {}
    for alpha in ALPHAS:
        summary[str(alpha)] = {}
        for sparsity in SPARSITIES:
            selected = [r for r in rows if r["alpha"] == alpha and r["sparsity"] == sparsity]
            summary[str(alpha)][str(sparsity)] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    high = [r for r in rows if r["sparsity"] in [0.90, 0.95, 0.98]]
    alpha_scores = {str(alpha): float(np.mean([r["accuracy"] for r in high if r["alpha"] == alpha])) for alpha in ALPHAS}
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    result = {"experiment": "04_torch_fashion_mnist_path_magnitude_blend", "setup": "Search conservative blends between magnitude and path-flow on Fashion-MNIST MLP. alpha=0 is magnitude; alpha=1 is full path-flow modulation.", "alpha_scores_all_sparsities": alpha_scores, "best_alpha": best_alpha, "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_blend.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_BLEND.md"
    lines = ["# Torch Fashion-MNIST Magnitude/Path-Flow Blend", "", result["setup"], "", f"Best alpha: `{best_alpha}`", "", "| Alpha | Mean accuracy over 90/95/98% |", "|---:|---:|"]
    for alpha in ALPHAS:
        lines.append(f"| `{alpha:.2f}` | `{alpha_scores[str(alpha)]:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_alpha": result["best_alpha"], "alpha_scores_all_sparsities": result["alpha_scores_all_sparsities"]}, indent=2))
