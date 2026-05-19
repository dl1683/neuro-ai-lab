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
METHODS = ["magnitude", "path_abs_input", "path_std_input", "path_std_no_balance"]


def _scores(model, calib_loader, method):
    if method == "magnitude":
        return base._magnitude_scores(model)
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
    if method == "path_abs_input":
        input_signal = flat_all.abs().mean(dim=0) + 1e-6
    else:
        input_signal = flat_all.std(dim=0) + 1e-6
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    hidden_fire = (h_all > 0).float().mean(dim=0) + 1e-6
    if method == "path_std_no_balance":
        hidden_balance = torch.ones_like(hidden_strength)
    else:
        hidden_balance = torch.exp(-torch.abs(hidden_fire - 0.35) / 0.22)
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    hidden = hidden_strength * hidden_balance * output_strength
    return {
        "fc1": model.fc1.weight.detach().abs() * hidden[:, None] * input_signal[None, :],
        "fc2": model.fc2.weight.detach().abs() * hidden[None, :],
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
        score_cache = {method: _scores(model, calib_loader, method) for method in METHODS}
        for sparsity in SPARSITIES:
            for method in METHODS:
                mask = base._mask(score_cache[method], sparsity)
                acc = base._evaluate(model, test_loader, mask)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "accuracy": acc, "retention": acc / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    result = {"experiment": "04_torch_fashion_mnist_input_signal", "setup": "Test whether Fashion-MNIST path-flow failure is caused by bad input activity estimates on normalized images. Compare abs normalized input vs per-pixel std input signal.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_input_signal.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_INPUT_SIGNAL.md"
    lines = ["# Torch Fashion-MNIST Input Signal Check", "", result["setup"], "", "| Sparsity | Method | Mean accuracy | Std | Retention |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
