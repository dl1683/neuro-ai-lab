from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
EPOCHS = 3
METHODS = ["magnitude", "global_synflow", "layerwise_synflow", "adaptive_dense_hybrid"]


def layerwise_mask(scores: dict[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    masks = {}
    for name, score in scores.items():
        flat = score.detach().flatten()
        keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
        threshold = torch.topk(flat, keep, largest=True).values.min()
        masks[name] = (score >= threshold).to(score.dtype)
    return masks


def apply_mask(model, masks):
    for name, param in base._params(model).items():
        param.data.mul_(masks[name].to(param.device))


def masked_finetune(model, train_loader, masks, epochs: int = EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    apply_mask(model, masks)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            apply_mask(model, masks)


def layer_damage(masks):
    out = {}
    for name, mask in masks.items():
        flat_out = mask.reshape(mask.shape[0], -1) if mask.ndim >= 2 else mask.reshape(1, -1)
        out[name] = {
            "keep_rate": float(mask.mean().item()),
            "dead_output_units": int((flat_out.sum(dim=1) == 0).sum().item()),
            "output_units": int(flat_out.shape[0]),
        }
    return out


def run():
    rows = []
    for seed in cnn.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base._evaluate(model, test_loader)
        score_maps = {
            "magnitude": {k: p.detach().abs().clone() for k, p in base._params(model).items()},
            "global_synflow": cnn._synflow_scores(model),
            "adaptive_dense_hybrid": hybrid._hybrid_scores(model, calib_loader, cnn.ALPHA_BY_SPARSITY[SPARSITY]),
        }
        masks_by_method = {
            "magnitude": base._mask(score_maps["magnitude"], SPARSITY),
            "global_synflow": base._mask(score_maps["global_synflow"], SPARSITY),
            "layerwise_synflow": layerwise_mask(score_maps["global_synflow"], SPARSITY),
            "adaptive_dense_hybrid": base._mask(score_maps["adaptive_dense_hybrid"], SPARSITY),
        }
        for method in METHODS:
            masks = masks_by_method[method]
            before = base._evaluate(model, test_loader, masks)
            model.load_state_dict(dense_state)
            masked_finetune(model, train_loader, masks)
            after = base._evaluate(model, test_loader)
            rows.append({
                "seed": seed,
                "method": method,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "before_retention": before / dense_accuracy,
                "after_retention": after / dense_accuracy,
                "damage": layer_damage(masks),
            })
            model.load_state_dict(dense_state)
    summary = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {
            "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
            "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
            "after_retention_mean": float(np.mean([r["after_retention"] for r in selected])),
            "fc1_keep_rate_mean": float(np.mean([r["damage"]["fc1"]["keep_rate"] for r in selected])),
            "fc1_dead_hidden_mean": float(np.mean([r["damage"]["fc1"]["dead_output_units"] for r in selected])),
        }
    result = {
        "experiment": "04_synflow_cnn_layerwise_rescue_98pct",
        "setup": "Fashion-MNIST CNN 98% pruning rescue test. Layerwise SynFlow keeps 2% inside each layer instead of allowing global SynFlow to starve fc1.",
        "sparsity": SPARSITY,
        "finetune_epochs": EPOCHS,
        "summary": summary,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "synflow_cnn_layerwise_rescue_98pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "SYNFLOW_CNN_LAYERWISE_RESCUE_98PCT.md"
    lines = ["# SynFlow CNN Layerwise Rescue at 98% Sparsity", "", result["setup"], "", "| Method | Before FT | After FT | After retention | fc1 keep rate | Dead fc1 hidden |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['after_retention_mean']:.4f}` | `{item['fc1_keep_rate_mean']:.4f}` | `{item['fc1_dead_hidden_mean']:.1f}` |")
    lines.extend(["", "Interpretation: this isolates global allocation from the SynFlow score itself. If layerwise SynFlow recovers while global SynFlow stays at chance, the failure is not simply `SynFlow bad`; it is global score allocation starving the dense bridge."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["summary"], indent=2))
