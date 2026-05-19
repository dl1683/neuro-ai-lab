from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "04_criticality_pruning"
DOC_DIR = ROOT / "experiments" / "04_criticality_pruning"


def _bootstrap_ci(values, n=10000, seed=123, alpha=0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [None, None]
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n)]
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def _paired_deltas(rows, method, baseline, sparsities):
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for sparsity in sparsities:
        for seed in seeds:
            left = next(r for r in rows if r["seed"] == seed and r.get("target_sparsity", r.get("sparsity")) == sparsity and r["method"] == method)
            right = next(r for r in rows if r["seed"] == seed and r.get("target_sparsity", r.get("sparsity")) == sparsity and r["method"] == baseline)
            out.append(left["accuracy"] - right["accuracy"])
    return out


def _sign_summary(deltas):
    deltas = np.asarray(deltas, dtype=float)
    return {
        "n": int(len(deltas)),
        "wins": int(np.sum(deltas > 0)),
        "losses": int(np.sum(deltas < 0)),
        "ties": int(np.sum(deltas == 0)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "bootstrap_95_ci_mean_delta": _bootstrap_ci(deltas),
    }


def synthesize() -> dict:
    core = json.loads((OUT_DIR / "pilot_result.json").read_text(encoding="utf-8"))
    ablation = json.loads((OUT_DIR / "path_flow_ablation.json").read_text(encoding="utf-8"))
    calibration = json.loads((OUT_DIR / "path_flow_calibration_sweep.json").read_text(encoding="utf-8"))
    finetune = json.loads((OUT_DIR / "path_flow_finetune.json").read_text(encoding="utf-8"))
    extreme = json.loads((OUT_DIR / "path_flow_extreme_sparsity.json").read_text(encoding="utf-8"))
    fashion_blend = json.loads((OUT_DIR / "torch_fashion_mnist_no_balance_blend.json").read_text(encoding="utf-8"))
    fashion_ft = json.loads((OUT_DIR / "torch_fashion_mnist_corrected_blend_finetune.json").read_text(encoding="utf-8"))

    core_rows = core["rows"]
    high = [0.90, 0.95]
    core_stats = {
        "path_flow_vs_magnitude": _sign_summary(_paired_deltas(core_rows, "path_flow", "magnitude", high)),
        "path_flow_vs_gradient_saliency": _sign_summary(_paired_deltas(core_rows, "path_flow", "gradient_saliency", high)),
        "activation_flow_vs_magnitude": _sign_summary(_paired_deltas(core_rows, "activation_flow", "magnitude", high)),
    }

    extreme_rows = extreme["rows"]
    extreme_stats = {
        "path_flow_vs_magnitude_97_99": _sign_summary(_paired_deltas(extreme_rows, "path_flow", "magnitude", [0.97, 0.98, 0.99])),
        "path_flow_vs_gradient_97_99": _sign_summary(_paired_deltas(extreme_rows, "path_flow", "gradient_saliency", [0.97, 0.98, 0.99])),
    }

    fashion_rows = fashion_blend["rows"]
    alpha = float(fashion_blend["best_alpha"])
    adaptive_rows = []
    for seed in sorted({r["seed"] for r in fashion_rows}):
        for sparsity in sorted({r["sparsity"] for r in fashion_rows}):
            chosen_alpha = 0.0 if sparsity <= 0.90 else alpha
            chosen = next(r for r in fashion_rows if r["seed"] == seed and r["sparsity"] == sparsity and r["alpha"] == chosen_alpha)
            mag = next(r for r in fashion_rows if r["seed"] == seed and r["sparsity"] == sparsity and r["alpha"] == 0.0)
            adaptive_rows.append({
                "seed": seed,
                "sparsity": sparsity,
                "chosen_alpha": chosen_alpha,
                "adaptive_accuracy": chosen["accuracy"],
                "magnitude_accuracy": mag["accuracy"],
                "delta": chosen["accuracy"] - mag["accuracy"],
            })
    adaptive_deltas = [r["delta"] for r in adaptive_rows]

    result = {
        "experiment": "04_path_flow_evidence_synthesis",
        "claim_v3": "Path-aware pruning is not universally better than magnitude. It is strongest as a label-free severe-sparsity path-preservation correction. On sklearn digits MLPs, full path-flow dominates magnitude and slightly beats gradient saliency. On Fashion-MNIST MLPs, naive path-flow fails, but a no-balance weak path/magnitude blend improves severe-sparsity masks and keeps a small edge after fine-tuning.",
        "core_sklearn_digits_paired_stats": core_stats,
        "extreme_sparsity_paired_stats": extreme_stats,
        "ablation_drops_vs_full": ablation["ablation_drops_vs_full"],
        "calibration_accuracy_drop_vs_full": calibration["accuracy_drop_vs_full_calibration"],
        "sklearn_finetune_summary": finetune["summary"],
        "fashion_mnist_adaptive_rule": {
            "rule": "use alpha=0 magnitude at 90% sparsity; use corrected no-balance path blend alpha=0.25 at 95% and 98% sparsity",
            "rows": adaptive_rows,
            "paired_delta_vs_magnitude": _sign_summary(adaptive_deltas),
        },
        "fashion_mnist_corrected_blend_finetune": fashion_ft["summary"],
    }
    (OUT_DIR / "evidence_synthesis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Path-Flow Evidence Synthesis",
        "",
        "This document consolidates the real experiment evidence for the pruning thread. It intentionally narrows the claim to what survived cross-dataset testing.",
        "",
        "## Claim v3",
        "",
        result["claim_v3"],
        "",
        "## Sklearn digits paired evidence",
        "",
        "High-sparsity regime: `90%` and `95%`, five seeds, paired by seed and sparsity.",
        "",
        "| Comparison | Mean delta | 95% bootstrap CI | Wins / N | Median delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in core_stats.items():
        ci = item["bootstrap_95_ci_mean_delta"]
        lines.append(f"| `{name}` | `{item['mean_delta']:.4f}` | `[{ci[0]:.4f}, {ci[1]:.4f}]` | `{item['wins']} / {item['n']}` | `{item['median_delta']:.4f}` |")

    lines.extend([
        "",
        "## Extreme sparsity evidence",
        "",
        "Extreme regime: `97%`, `98%`, `99%`, five seeds, one-shot pruning.",
        "",
        "| Comparison | Mean delta | 95% bootstrap CI | Wins / N | Median delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in extreme_stats.items():
        ci = item["bootstrap_95_ci_mean_delta"]
        lines.append(f"| `{name}` | `{item['mean_delta']:.4f}` | `[{ci[0]:.4f}, {ci[1]:.4f}]` | `{item['wins']} / {item['n']}` | `{item['median_delta']:.4f}` |")

    lines.extend([
        "",
        "## Ingredient ablation",
        "",
        "Positive drops mean removing the ingredient hurt full path-flow.",
        "",
        "| Removed ingredient | Accuracy drop |",
        "|---|---:|",
    ])
    for key, val in ablation["ablation_drops_vs_full"].items():
        lines.append(f"| `{key}` | `{val:.4f}` |")

    lines.extend([
        "",
        "## Calibration requirement",
        "",
        "Accuracy drop versus full unlabeled calibration on sklearn digits high-sparsity masks.",
        "",
        "| Calibration fraction | Accuracy drop |",
        "|---:|---:|",
    ])
    for key, val in calibration["accuracy_drop_vs_full_calibration"].items():
        lines.append(f"| `{key}` | `{val:.4f}` |")

    fashion_item = result["fashion_mnist_adaptive_rule"]["paired_delta_vs_magnitude"]
    ci = fashion_item["bootstrap_95_ci_mean_delta"]
    lines.extend([
        "",
        "## Fashion-MNIST adaptive rule",
        "",
        result["fashion_mnist_adaptive_rule"]["rule"],
        "",
        f"Paired delta vs magnitude across two seeds and three sparsities: mean `{fashion_item['mean_delta']:.4f}`, 95% bootstrap CI `[{ci[0]:.4f}, {ci[1]:.4f}]`, wins `{fashion_item['wins']} / {fashion_item['n']}`.",
        "",
        "Rows:",
        "",
        "| Seed | Sparsity | Chosen alpha | Adaptive accuracy | Magnitude accuracy | Delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in adaptive_rows:
        lines.append(f"| `{row['seed']}` | `{row['sparsity']:.2f}` | `{row['chosen_alpha']:.2f}` | `{row['adaptive_accuracy']:.4f}` | `{row['magnitude_accuracy']:.4f}` | `{row['delta']:.4f}` |")

    lines.extend([
        "",
        "## Bottom line",
        "",
        "The exceptional part is not the original criticality story. It is the severe-sparsity path-preservation principle. The strongest evidence is on sklearn digits; Fashion-MNIST forces a narrower rule: keep magnitude dominant at moderate sparsity, add a weak no-balance path correction only near the sparsity cliff.",
    ])
    (DOC_DIR / "EVIDENCE_SYNTHESIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    summary = synthesize()
    print(json.dumps({
        "claim_v3": summary["claim_v3"],
        "core": summary["core_sklearn_digits_paired_stats"],
        "extreme": summary["extreme_sparsity_paired_stats"],
        "fashion_adaptive": summary["fashion_mnist_adaptive_rule"]["paired_delta_vs_magnitude"],
    }, indent=2))
