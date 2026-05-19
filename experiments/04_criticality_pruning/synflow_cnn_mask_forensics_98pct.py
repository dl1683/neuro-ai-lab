from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CNN_PATH = Path(__file__).resolve().parent / "fashion_mnist_cnn_synflow.py"
spec = importlib.util.spec_from_file_location("cnn_synflow", CNN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {CNN_PATH}")
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
base = cnn.base
hybrid = cnn.hybrid

SPARSITY = 0.98
METHODS = ["magnitude", "synflow", "adaptive_dense_hybrid"]


def _tensor_stats(mask: torch.Tensor) -> dict:
    kept = int(mask.sum().item())
    total = int(mask.numel())
    return {"kept": kept, "total": total, "keep_rate": kept / total if total else 0.0}


def _layer_forensics(masks: dict[str, torch.Tensor]) -> dict:
    out = {}
    for name, mask in masks.items():
        item = _tensor_stats(mask)
        if mask.ndim >= 2:
            flat_out = mask.reshape(mask.shape[0], -1)
            flat_in = mask.reshape(mask.shape[0], -1)
            item["dead_output_units"] = int((flat_out.sum(dim=1) == 0).sum().item())
            item["output_units"] = int(flat_out.shape[0])
            if mask.ndim == 2:
                item["dead_input_units"] = int((mask.sum(dim=0) == 0).sum().item())
                item["input_units"] = int(mask.shape[1])
        out[name] = item
    return out


def _damage_scores(masks: dict[str, torch.Tensor]) -> dict:
    fc2 = masks.get("fc2")
    fc1 = masks.get("fc1")
    conv2 = masks.get("conv2")
    conv1 = masks.get("conv1")
    result = {}
    if fc2 is not None:
        result["dead_classes"] = int((fc2.reshape(fc2.shape[0], -1).sum(dim=1) == 0).sum().item())
        result["min_class_fanin"] = int(fc2.sum(dim=1).min().item())
        result["max_class_fanin"] = int(fc2.sum(dim=1).max().item())
    if fc1 is not None:
        result["dead_fc1_hidden"] = int((fc1.sum(dim=1) == 0).sum().item())
        result["live_fc1_hidden"] = int((fc1.sum(dim=1) > 0).sum().item())
    if conv2 is not None:
        result["dead_conv2_filters"] = int((conv2.reshape(conv2.shape[0], -1).sum(dim=1) == 0).sum().item())
    if conv1 is not None:
        result["dead_conv1_filters"] = int((conv1.reshape(conv1.shape[0], -1).sum(dim=1) == 0).sum().item())
    return result


def run() -> dict:
    rows = []
    for seed in cnn.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_accuracy = base._evaluate(model, test_loader)
        scores = {
            "magnitude": {k: p.detach().abs().clone() for k, p in base._params(model).items()},
            "synflow": cnn._synflow_scores(model),
            "adaptive_dense_hybrid": hybrid._hybrid_scores(model, calib_loader, cnn.ALPHA_BY_SPARSITY[SPARSITY]),
        }
        for method in METHODS:
            masks = base._mask(scores[method], SPARSITY)
            accuracy = base._evaluate(model, test_loader, masks)
            rows.append({
                "seed": seed,
                "method": method,
                "dense_accuracy": dense_accuracy,
                "accuracy": accuracy,
                "retention": accuracy / dense_accuracy,
                "damage": _damage_scores(masks),
                "layers": _layer_forensics(masks),
            })

    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
            "retention_mean": float(np.mean([row["retention"] for row in selected])),
            "damage_mean": {},
        }
        for key in sorted({k for row in selected for k in row["damage"]}):
            summary[method]["damage_mean"][key] = float(np.mean([row["damage"].get(key, 0) for row in selected]))
        layer_names = sorted({name for row in selected for name in row["layers"]})
        summary[method]["layer_keep_rate_mean"] = {
            name: float(np.mean([row["layers"][name]["keep_rate"] for row in selected])) for name in layer_names
        }

    result = {
        "experiment": "04_synflow_cnn_mask_forensics_98pct",
        "setup": "Fashion-MNIST CNN 98% global pruning mask forensics. Measures structural mask damage for magnitude, SynFlow, and adaptive dense-hybrid pruning.",
        "sparsity": SPARSITY,
        "summary": summary,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "synflow_cnn_mask_forensics_98pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "SYNFLOW_CNN_MASK_FORENSICS_98PCT.md"
    lines = ["# SynFlow CNN Mask Forensics at 98% Sparsity", "", result["setup"], "", "| Method | Accuracy | Retention | Dead conv1 | Dead conv2 | Dead fc1 hidden | Dead classes | Min class fan-in |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = summary[method]
        damage = item["damage_mean"]
        lines.append(
            f"| `{method}` | `{item['accuracy_mean']:.4f}` | `{item['retention_mean']:.4f}` | `{damage.get('dead_conv1_filters', 0):.1f}` | `{damage.get('dead_conv2_filters', 0):.1f}` | `{damage.get('dead_fc1_hidden', 0):.1f}` | `{damage.get('dead_classes', 0):.1f}` | `{damage.get('min_class_fanin', 0):.1f}` |"
        )
    lines.extend(["", "## Mean layer keep rates", "", "| Method | conv1 | conv2 | fc1 | fc2 |", "|---|---:|---:|---:|---:|"])
    for method in METHODS:
        rates = summary[method]["layer_keep_rate_mean"]
        lines.append(f"| `{method}` | `{rates.get('conv1.weight', 0):.4f}` | `{rates.get('conv2.weight', 0):.4f}` | `{rates.get('fc1.weight', 0):.4f}` | `{rates.get('fc2.weight', 0):.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["summary"], indent=2))
