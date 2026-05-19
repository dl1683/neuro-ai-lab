from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "04_criticality_pruning"
DOC = ROOT / "experiments" / "04_criticality_pruning"

IMAGE_RUNS = [
    {
        "study": "Fashion-MNIST MLP",
        "path": OUT / "torch_fashion_mnist_no_balance_blend.json",
        "schedule": {0.90: 0.0, 0.95: 0.25, 0.98: 0.50},
    },
    {
        "study": "CIFAR-10 MLP",
        "path": OUT / "torch_cifar10_mlp_path_flow.json",
        "schedule": {0.90: 0.0, 0.95: 0.25, 0.98: 0.50},
    },
    {
        "study": "Fashion-MNIST CNN dense-hybrid",
        "path": OUT / "torch_fashion_mnist_cnn_dense_hybrid.json",
        "schedule": {0.90: 0.0, 0.95: 0.20, 0.98: 0.20},
    },
]


def _nearest_alpha(rows, sparsity, target_alpha):
    alphas = sorted({float(r["alpha"]) for r in rows if r.get("alpha") is not None and float(r["sparsity"]) == sparsity})
    return min(alphas, key=lambda a: abs(a - target_alpha))


def run():
    rows = []
    for spec in IMAGE_RUNS:
        data = json.loads(spec["path"].read_text(encoding="utf-8"))
        source_rows = data["rows"]
        seeds = sorted({r["seed"] for r in source_rows})
        for seed in seeds:
            for sparsity, target_alpha in spec["schedule"].items():
                chosen_alpha = _nearest_alpha(source_rows, sparsity, target_alpha)
                chosen = next(r for r in source_rows if r["seed"] == seed and float(r["sparsity"]) == sparsity and r.get("alpha") is not None and float(r["alpha"]) == chosen_alpha)
                magnitude = next(r for r in source_rows if r["seed"] == seed and float(r["sparsity"]) == sparsity and r.get("alpha") is not None and float(r["alpha"]) == 0.0)
                rows.append({
                    "study": spec["study"],
                    "seed": seed,
                    "sparsity": sparsity,
                    "target_alpha": target_alpha,
                    "chosen_alpha": chosen_alpha,
                    "adaptive_accuracy": chosen["accuracy"],
                    "magnitude_accuracy": magnitude["accuracy"],
                    "delta": chosen["accuracy"] - magnitude["accuracy"],
                })

    by_study = {}
    for study in sorted({r["study"] for r in rows}):
        selected = [r for r in rows if r["study"] == study]
        by_study[study] = {
            "mean_delta": float(np.mean([r["delta"] for r in selected])),
            "wins": int(sum(r["delta"] > 0 for r in selected)),
            "ties": int(sum(r["delta"] == 0 for r in selected)),
            "losses": int(sum(r["delta"] < 0 for r in selected)),
            "n": len(selected),
        }
    total = {
        "mean_delta": float(np.mean([r["delta"] for r in rows])),
        "wins": int(sum(r["delta"] > 0 for r in rows)),
        "ties": int(sum(r["delta"] == 0 for r in rows)),
        "losses": int(sum(r["delta"] < 0 for r in rows)),
        "n": len(rows),
    }
    result = {
        "experiment": "04_adaptive_path_rule_image_eval",
        "rule": "Use magnitude at 90% sparsity; use a weak path correction at 95%; use a stronger path correction at 98%, with CNNs using dense-layer-only path correction.",
        "by_study": by_study,
        "overall": total,
        "rows": rows,
    }
    (OUT / "adaptive_path_rule_image_eval.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Adaptive Path Rule Image Evaluation",
        "",
        "This document evaluates the current practical rule on completed image-model runs.",
        "",
        "## Rule",
        "",
        result["rule"],
        "",
        "## Aggregate",
        "",
        f"Overall mean delta vs pure magnitude: `{total['mean_delta']:.4f}` over `{total['n']}` paired cases; wins `{total['wins']}`, ties `{total['ties']}`, losses `{total['losses']}`.",
        "",
        "| Study | Mean delta | Wins | Ties | Losses | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for study, item in by_study.items():
        lines.append(f"| {study} | `{item['mean_delta']:.4f}` | `{item['wins']}` | `{item['ties']}` | `{item['losses']}` | `{item['n']}` |")
    lines.extend(["", "## Paired rows", "", "| Study | Seed | Sparsity | Alpha | Adaptive accuracy | Magnitude accuracy | Delta |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['study']} | `{row['seed']}` | `{row['sparsity']:.2f}` | `{row['chosen_alpha']:.2f}` | `{row['adaptive_accuracy']:.4f}` | `{row['magnitude_accuracy']:.4f}` | `{row['delta']:.4f}` |")
    (DOC / "ADAPTIVE_PATH_RULE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"overall": result["overall"], "by_study": result["by_study"]}, indent=2))
