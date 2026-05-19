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
ALPHAS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.65, 1.0]


def _scores(model, calib_loader, alpha):
    model.eval()
    flats = []
    hiddens = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(base.DEVICE)
            _, flat, h = model(x, return_hidden=True)
            flats.append(flat.cpu())
            hiddens.append(h.cpu())
    flat_all = torch.cat(flats, dim=0).to(base.DEVICE)
    h_all = torch.cat(hiddens, dim=0).to(base.DEVICE)
    input_signal = flat_all.std(dim=0) + 1e-6
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    hidden = hidden_strength * output_strength
    return {
        "fc1": model.fc1.weight.detach().abs() * torch.pow(input_signal[None, :] * hidden[:, None], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(hidden[None, :], alpha),
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
            scores = _scores(model, calib_loader, alpha)
            for sparsity in SPARSITIES:
                mask = base._mask(scores, sparsity)
                accuracy = base._evaluate(model, test_loader, mask)
                rows.append({"seed": seed, "alpha": alpha, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})
    scores = {str(alpha): float(np.mean([r["accuracy"] for r in rows if r["alpha"] == alpha])) for alpha in ALPHAS}
    best_alpha = max(scores, key=scores.get)
    result = {"experiment": "04_torch_fashion_mnist_no_balance_blend", "setup": "Search conservative blends between magnitude and no-balance path-flow on Fashion-MNIST. alpha=0 is magnitude.", "alpha_scores_all_sparsities": scores, "best_alpha": best_alpha, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_no_balance_blend.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_NO_BALANCE_BLEND.md"
    lines = ["# Torch Fashion-MNIST No-Balance Blend", "", result["setup"], "", f"Best alpha: `{best_alpha}`", "", "| Alpha | Mean accuracy over 90/95/98% |", "|---:|---:|"]
    for alpha in ALPHAS:
        lines.append(f"| `{alpha:.2f}` | `{scores[str(alpha)]:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_alpha": result["best_alpha"], "alpha_scores_all_sparsities": result["alpha_scores_all_sparsities"]}, indent=2))
