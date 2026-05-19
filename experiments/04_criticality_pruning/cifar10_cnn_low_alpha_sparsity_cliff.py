from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "cifar10_cnn_98pct_adaptive_alpha_ft_sweep.py"
spec = importlib.util.spec_from_file_location("cifar98", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SPARSITIES = [0.95, 0.98, 0.99]
ALPHAS = [0.0, 0.03, 0.05]


def run():
    rows = []
    for seed in base.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCifarCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base._evaluate(model, test_loader)
        score_cache = {alpha: base._hybrid_scores(model, calib_loader, alpha) for alpha in ALPHAS}
        for sparsity in SPARSITIES:
            for alpha in ALPHAS:
                masks = base._mask(score_cache[alpha], sparsity)
                before = base._evaluate(model, test_loader, masks)
                model.load_state_dict(dense_state)
                base._masked_finetune(model, train_loader, masks)
                after = base._evaluate(model, test_loader)
                stats = base._fc1_stats(masks)
                rows.append({
                    "seed": seed,
                    "sparsity": sparsity,
                    "alpha": alpha,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "before_retention": before / dense_accuracy,
                    "after_retention": after / dense_accuracy,
                    **stats,
                })
                model.load_state_dict(dense_state)
    summary = {}
    paired = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for alpha in ALPHAS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["alpha"] == alpha]
            summary[str(sparsity)][str(alpha)] = {
                "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "before_retention_mean": float(np.mean([r["before_retention"] for r in selected])),
                "after_retention_mean": float(np.mean([r["after_retention"] for r in selected])),
                "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
            }
        mag = summary[str(sparsity)]["0.0"]
        best_before_alpha = max(ALPHAS, key=lambda a: summary[str(sparsity)][str(a)]["before_mean"])
        best_after_alpha = max(ALPHAS, key=lambda a: summary[str(sparsity)][str(a)]["after_mean"])
        paired.append({
            "sparsity": sparsity,
            "best_before_alpha": best_before_alpha,
            "best_before_delta_vs_magnitude": summary[str(sparsity)][str(best_before_alpha)]["before_mean"] - mag["before_mean"],
            "best_after_alpha": best_after_alpha,
            "best_after_delta_vs_magnitude": summary[str(sparsity)][str(best_after_alpha)]["after_mean"] - mag["after_mean"],
        })
    result = {
        "experiment": "04_cifar10_cnn_low_alpha_sparsity_cliff",
        "setup": "CIFAR-10 small CNN low-alpha path-correction sweep across 95/98/99% sparsity. Real CIFAR-10 images, 20k train subset, 5k test subset, two seeds.",
        "sparsities": SPARSITIES,
        "alphas": ALPHAS,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_cnn_low_alpha_sparsity_cliff.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_CNN_LOW_ALPHA_SPARSITY_CLIFF.md"
    lines = ["# CIFAR-10 CNN Low-Alpha Sparsity Cliff", "", result["setup"], "", "| Sparsity | Alpha | Before FT | After FT | Dead fc1 hidden |", "|---:|---:|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for alpha in ALPHAS:
            item = summary[str(sparsity)][str(alpha)]
            lines.append(f"| `{sparsity:.2f}` | `{alpha:.2f}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` |")
    lines.extend(["", "## Best deltas vs magnitude", "", "| Sparsity | Best before alpha | Before delta | Best after alpha | After delta |", "|---:|---:|---:|---:|---:|"])
    for item in paired:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['best_before_alpha']:.2f}` | `{item['best_before_delta_vs_magnitude']:+.4f}` | `{item['best_after_alpha']:.2f}` | `{item['best_after_delta_vs_magnitude']:+.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"paired_deltas": result["paired_deltas"], "summary": result["summary"]}, indent=2))
