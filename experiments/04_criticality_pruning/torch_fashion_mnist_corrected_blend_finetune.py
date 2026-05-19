from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "torch_fashion_mnist_path_flow.py"
spec = importlib.util.spec_from_file_location("fashion_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEEDS = [31, 32]
SPARSITIES = [0.95, 0.98]
METHODS = ["magnitude", "corrected_path_blend"]
ALPHA = 0.25


def _corrected_scores(model, calib_loader):
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
        "fc1": model.fc1.weight.detach().abs() * torch.pow(input_signal[None, :] * hidden[:, None], ALPHA),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(hidden[None, :], ALPHA),
    }


def _copy_model(model):
    clone = base.MLP().to(base.DEVICE)
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def _masked_finetune(model, masks, train_loader, epochs=2):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    original_masks = {k: v.to(base.DEVICE) for k, v in masks.items()}
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(base.DEVICE), y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            original = base._apply_masks(model, original_masks)
            loss = loss_fn(model(x), y)
            loss.backward()
            base._restore(model, original)
            model.fc1.weight.grad.mul_(original_masks["fc1"])
            model.fc2.weight.grad.mul_(original_masks["fc2"])
            opt.step()
            base._apply_masks(model, original_masks)
    return model


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.MLP().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        score_by_method = {"magnitude": base._magnitude_scores(model), "corrected_path_blend": _corrected_scores(model, calib_loader)}
        for sparsity in SPARSITIES:
            for method in METHODS:
                masks = base._mask(score_by_method[method], sparsity)
                before = base._evaluate(model, test_loader, masks)
                tuned = _masked_finetune(_copy_model(model), masks, train_loader)
                after = base._evaluate(tuned, test_loader, masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "before_finetune": before, "after_finetune": after, "retention_after": after / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"before_mean": float(np.mean([r["before_finetune"] for r in selected])), "after_mean": float(np.mean([r["after_finetune"] for r in selected])), "retention_after_mean": float(np.mean([r["retention_after"] for r in selected]))}
    result = {"experiment": "04_torch_fashion_mnist_corrected_blend_finetune", "setup": "Masked fine-tuning after pruning Fashion-MNIST MLPs with magnitude vs corrected no-balance path blend at 95% and 98% sparsity.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_corrected_blend_finetune.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_CORRECTED_BLEND_FINETUNE.md"
    lines = ["# Torch Fashion-MNIST Corrected Blend Fine-Tuning", "", result["setup"], "", "| Sparsity | Method | Before FT | After FT | Retention after FT |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['retention_after_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
