from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "torch_fashion_mnist_cnn_dense_hybrid.py"
spec = importlib.util.spec_from_file_location("hybrid_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)
base = hybrid.base

SEEDS = [51, 52]
CONFIGS = [
    {"sparsity": 0.95, "alpha": 0.0, "method": "magnitude"},
    {"sparsity": 0.95, "alpha": 0.20, "method": "dense_hybrid"},
    {"sparsity": 0.98, "alpha": 0.0, "method": "magnitude"},
    {"sparsity": 0.98, "alpha": 0.20, "method": "dense_hybrid"},
]


def _copy_model(model):
    clone = base.SmallCNN().to(base.DEVICE)
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def _masked_finetune(model, masks, train_loader, epochs=2):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    masks = {k: v.to(base.DEVICE) for k, v in masks.items()}
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(base.DEVICE), y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            original = base._apply_masks(model, masks)
            loss = loss_fn(model(x), y)
            loss.backward()
            base._restore(model, original)
            for key, param in base._params(model).items():
                if param.grad is not None:
                    param.grad.mul_(masks[key])
            opt.step()
            base._apply_masks(model, masks)
    return model


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        for cfg in CONFIGS:
            scores = hybrid._hybrid_scores(model, calib_loader, cfg["alpha"])
            masks = base._mask(scores, cfg["sparsity"])
            before = base._evaluate(model, test_loader, masks)
            tuned = _masked_finetune(_copy_model(model), masks, train_loader)
            after = base._evaluate(tuned, test_loader, masks)
            rows.append({"seed": seed, "sparsity": cfg["sparsity"], "method": cfg["method"], "alpha": cfg["alpha"], "dense_accuracy": dense_accuracy, "before_finetune": before, "after_finetune": after, "retention_after": after / dense_accuracy})
    summary = {}
    for sparsity in [0.95, 0.98]:
        summary[str(sparsity)] = {}
        for method in ["magnitude", "dense_hybrid"]:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"before_mean": float(np.mean([r["before_finetune"] for r in selected])), "after_mean": float(np.mean([r["after_finetune"] for r in selected])), "retention_after_mean": float(np.mean([r["retention_after"] for r in selected]))}
    result = {"experiment": "04_torch_fashion_mnist_cnn_dense_hybrid_finetune", "setup": "Masked fine-tuning after pruning Fashion-MNIST CNNs. Conv layers are magnitude-pruned; dense layers are either magnitude or dense-hybrid path-flow alpha=0.20.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_cnn_dense_hybrid_finetune.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_CNN_DENSE_HYBRID_FINETUNE.md"
    lines = ["# Torch Fashion-MNIST CNN Dense-Hybrid Fine-Tuning", "", result["setup"], "", "| Sparsity | Method | Before FT | After FT | Retention after FT |", "|---:|---|---:|---:|---:|"]
    for sparsity in [0.95, 0.98]:
        for method in ["magnitude", "dense_hybrid"]:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['retention_after_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
