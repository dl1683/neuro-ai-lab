from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
MLP_PATH = Path(__file__).resolve().parent / "fashion_mnist_mlp_synflow.py"
CNN_PATH = Path(__file__).resolve().parent / "fashion_mnist_cnn_synflow.py"
mlp_spec = importlib.util.spec_from_file_location("mlp_syn", MLP_PATH)
cnn_spec = importlib.util.spec_from_file_location("cnn_syn", CNN_PATH)
if mlp_spec is None or mlp_spec.loader is None or cnn_spec is None or cnn_spec.loader is None:
    raise RuntimeError("Could not load synflow modules")
mlp_syn = importlib.util.module_from_spec(mlp_spec)
cnn_syn = importlib.util.module_from_spec(cnn_spec)
mlp_spec.loader.exec_module(mlp_syn)
cnn_spec.loader.exec_module(cnn_syn)
mlp_base = mlp_syn.base
cnn_hybrid = cnn_syn.hybrid
cnn_base = cnn_syn.base

SEEDS = [31, 32]
CNN_SEEDS = [51, 52]
SPARSITY = 0.98
METHODS = ["magnitude", "synflow", "adaptive_path"]


def _copy_mlp(model):
    clone = mlp_base.MLP().to(mlp_base.DEVICE)
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def _copy_cnn(model):
    clone = cnn_base.SmallCNN().to(cnn_base.DEVICE)
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def _masked_finetune_mlp(model, masks, train_loader, epochs=2):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    masks = {k: v.to(mlp_base.DEVICE) for k, v in masks.items()}
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(mlp_base.DEVICE), y.to(mlp_base.DEVICE)
            opt.zero_grad(set_to_none=True)
            original = mlp_base._apply_masks(model, masks)
            loss = loss_fn(model(x), y)
            loss.backward()
            mlp_base._restore(model, original)
            model.fc1.weight.grad.mul_(masks["fc1"])
            model.fc2.weight.grad.mul_(masks["fc2"])
            opt.step()
            mlp_base._apply_masks(model, masks)
    return model


def _masked_finetune_cnn(model, masks, train_loader, epochs=2):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    masks = {k: v.to(cnn_base.DEVICE) for k, v in masks.items()}
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cnn_base.DEVICE), y.to(cnn_base.DEVICE)
            opt.zero_grad(set_to_none=True)
            original = cnn_base._apply_masks(model, masks)
            loss = loss_fn(model(x), y)
            loss.backward()
            cnn_base._restore(model, original)
            for key, param in cnn_base._params(model).items():
                if param.grad is not None:
                    param.grad.mul_(masks[key])
            opt.step()
            cnn_base._apply_masks(model, masks)
    return model


def _run_mlp():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = mlp_base._loaders(seed)
        model = mlp_base.MLP().to(mlp_base.DEVICE)
        mlp_base._train(model, train_loader)
        dense = mlp_base._evaluate(model, test_loader)
        scores = {
            "magnitude": mlp_base._magnitude_scores(model),
            "synflow": mlp_syn._synflow_scores(model),
            "adaptive_path": mlp_syn._corrected_path_scores(model, calib_loader, 0.50),
        }
        for method in METHODS:
            masks = mlp_base._mask(scores[method], SPARSITY)
            before = mlp_base._evaluate(model, test_loader, masks)
            tuned = _masked_finetune_mlp(_copy_mlp(model), masks, train_loader)
            after = mlp_base._evaluate(tuned, test_loader, masks)
            rows.append({"study": "Fashion-MNIST MLP", "seed": seed, "method": method, "dense_accuracy": dense, "before_finetune": before, "after_finetune": after, "retention_after": after / dense})
    return rows


def _run_cnn():
    rows = []
    for seed in CNN_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = cnn_base._loaders(seed)
        model = cnn_base.SmallCNN().to(cnn_base.DEVICE)
        cnn_base._train(model, train_loader)
        dense = cnn_base._evaluate(model, test_loader)
        scores = {
            "magnitude": {k: p.detach().abs().clone() for k, p in cnn_base._params(model).items()},
            "synflow": cnn_syn._synflow_scores(model),
            "adaptive_path": cnn_hybrid._hybrid_scores(model, calib_loader, 0.20),
        }
        for method in METHODS:
            masks = cnn_base._mask(scores[method], SPARSITY)
            before = cnn_base._evaluate(model, test_loader, masks)
            tuned = _masked_finetune_cnn(_copy_cnn(model), masks, train_loader)
            after = cnn_base._evaluate(tuned, test_loader, masks)
            rows.append({"study": "Fashion-MNIST CNN", "seed": seed, "method": method, "dense_accuracy": dense, "before_finetune": before, "after_finetune": after, "retention_after": after / dense})
    return rows


def run():
    rows = _run_mlp() + _run_cnn()
    summary = {}
    for study in sorted({r["study"] for r in rows}):
        summary[study] = {}
        for method in METHODS:
            selected = [r for r in rows if r["study"] == study and r["method"] == method]
            summary[study][method] = {"before_mean": float(np.mean([r["before_finetune"] for r in selected])), "after_mean": float(np.mean([r["after_finetune"] for r in selected])), "retention_after_mean": float(np.mean([r["retention_after"] for r in selected]))}
    paired = []
    for study in sorted({r["study"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows if r["study"] == study}):
            adaptive = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["method"] == "adaptive_path")
            synflow = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["method"] == "synflow")
            magnitude = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["method"] == "magnitude")
            paired.append({"study": study, "seed": seed, "adaptive_minus_synflow_after": adaptive["after_finetune"] - synflow["after_finetune"], "adaptive_minus_magnitude_after": adaptive["after_finetune"] - magnitude["after_finetune"]})
    result = {"experiment": "04_98pct_synflow_finetune_comparison", "setup": "At 98% sparsity, compare magnitude, SynFlow, and adaptive path before/after masked fine-tuning on Fashion-MNIST MLP and CNN.", "summary": summary, "paired_deltas_after_finetune": paired, "mean_adaptive_minus_synflow_after": float(np.mean([p["adaptive_minus_synflow_after"] for p in paired])), "mean_adaptive_minus_magnitude_after": float(np.mean([p["adaptive_minus_magnitude_after"] for p in paired])), "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "synflow_finetune_98pct_comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "SYNFLOW_FINETUNE_98PCT.md"
    lines = ["# 98% SynFlow Fine-Tuning Comparison", "", result["setup"], "", "| Study | Method | Before FT | After FT | Retention after FT |", "|---|---|---:|---:|---:|"]
    for study in summary:
        for method in METHODS:
            item = summary[study][method]
            lines.append(f"| {study} | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['retention_after_mean']:.4f}` |")
    lines.extend(["", f"Mean adaptive minus SynFlow after FT: `{result['mean_adaptive_minus_synflow_after']:.4f}`.", f"Mean adaptive minus magnitude after FT: `{result['mean_adaptive_minus_magnitude_after']:.4f}`."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "mean_adaptive_minus_synflow_after": result["mean_adaptive_minus_synflow_after"], "mean_adaptive_minus_magnitude_after": result["mean_adaptive_minus_magnitude_after"]}, indent=2))
