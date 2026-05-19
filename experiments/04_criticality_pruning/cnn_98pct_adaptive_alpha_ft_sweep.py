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
ALPHAS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]


def apply_mask(model, masks):
    for name, param in base._params(model).items():
        param.data.mul_(masks[name].to(param.device))


def masked_finetune(model, train_loader, masks):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    apply_mask(model, masks)
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            apply_mask(model, masks)


def fc1_stats(masks):
    fc1 = masks["fc1"]
    return {
        "fc1_keep_rate": float(fc1.mean().item()),
        "dead_fc1_hidden": int((fc1.sum(dim=1) == 0).sum().item()),
    }


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
        for alpha in ALPHAS:
            scores = hybrid._hybrid_scores(model, calib_loader, alpha)
            masks = base._mask(scores, SPARSITY)
            before = base._evaluate(model, test_loader, masks)
            model.load_state_dict(dense_state)
            masked_finetune(model, train_loader, masks)
            after = base._evaluate(model, test_loader)
            item = {
                "seed": seed,
                "alpha": alpha,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "before_retention": before / dense_accuracy,
                "after_retention": after / dense_accuracy,
            }
            item.update(fc1_stats(masks))
            rows.append(item)
            model.load_state_dict(dense_state)
    summary = {}
    for alpha in ALPHAS:
        selected = [r for r in rows if r["alpha"] == alpha]
        summary[str(alpha)] = {
            "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
            "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
            "before_retention_mean": float(np.mean([r["before_retention"] for r in selected])),
            "after_retention_mean": float(np.mean([r["after_retention"] for r in selected])),
            "fc1_keep_rate_mean": float(np.mean([r["fc1_keep_rate"] for r in selected])),
            "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
        }
    best_before = max(summary.items(), key=lambda kv: kv[1]["before_mean"])
    best_after = max(summary.items(), key=lambda kv: kv[1]["after_mean"])
    best_balanced = max(summary.items(), key=lambda kv: kv[1]["before_mean"] + kv[1]["after_mean"])
    result = {
        "experiment": "04_cnn_98pct_adaptive_alpha_ft_sweep",
        "setup": "Fashion-MNIST CNN 98% adaptive dense-hybrid alpha sweep before and after 3-epoch masked fine-tuning.",
        "sparsity": SPARSITY,
        "finetune_epochs": EPOCHS,
        "summary": summary,
        "best_before": {"alpha": float(best_before[0]), **best_before[1]},
        "best_after": {"alpha": float(best_after[0]), **best_after[1]},
        "best_balanced": {"alpha": float(best_balanced[0]), **best_balanced[1]},
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cnn_98pct_adaptive_alpha_ft_sweep.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CNN_98PCT_ADAPTIVE_ALPHA_FT_SWEEP.md"
    lines = ["# CNN 98% Adaptive Alpha Fine-Tuning Sweep", "", result["setup"], "", "| Alpha | Before FT | After FT | fc1 keep rate | Dead fc1 hidden |", "|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        item = summary[str(alpha)]
        lines.append(f"| `{alpha:.2f}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['fc1_keep_rate_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` |")
    lines.extend(["", f"Best one-shot alpha: `{result['best_before']['alpha']:.2f}` with `{result['best_before']['before_mean']:.4f}` before FT and `{result['best_before']['after_mean']:.4f}` after FT.", f"Best after-FT alpha: `{result['best_after']['alpha']:.2f}` with `{result['best_after']['after_mean']:.4f}` after FT.", f"Best balanced alpha: `{result['best_balanced']['alpha']:.2f}`."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_before": result["best_before"], "best_after": result["best_after"], "best_balanced": result["best_balanced"], "summary": result["summary"]}, indent=2))
