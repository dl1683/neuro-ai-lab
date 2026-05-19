from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "04_criticality_pruning"
DOC = ROOT / "experiments" / "04_criticality_pruning"

SOURCES = [
    {
        "name": "Fashion-MNIST MLP corrected blend",
        "path": OUT / "torch_fashion_mnist_no_balance_blend.json",
        "alpha_key": "alpha",
        "score_label": "accuracy",
    },
    {
        "name": "CIFAR-10 MLP corrected blend",
        "path": OUT / "torch_cifar10_mlp_path_flow.json",
        "alpha_key": "alpha",
        "score_label": "accuracy",
    },
    {
        "name": "Fashion-MNIST CNN dense-hybrid",
        "path": OUT / "torch_fashion_mnist_cnn_dense_hybrid.json",
        "alpha_key": "alpha",
        "score_label": "accuracy",
    },
]


def _best_by_sparsity(rows):
    out = []
    sparsities = sorted({float(r["sparsity"]) for r in rows})
    for sparsity in sparsities:
        rows_s = [r for r in rows if float(r["sparsity"]) == sparsity and r.get("alpha") is not None]
        alphas = sorted({float(r["alpha"]) for r in rows_s})
        means = {}
        for alpha in alphas:
            vals = [float(r["accuracy"]) for r in rows_s if float(r["alpha"]) == alpha]
            means[alpha] = float(np.mean(vals))
        best_alpha = max(means, key=means.get)
        magnitude = means.get(0.0)
        out.append({
            "sparsity": sparsity,
            "best_alpha": best_alpha,
            "best_accuracy": means[best_alpha],
            "magnitude_accuracy": magnitude,
            "delta_vs_magnitude": None if magnitude is None else means[best_alpha] - magnitude,
            "all_alpha_means": {str(k): v for k, v in means.items()},
        })
    return out


def run():
    studies = []
    all_points = []
    for source in SOURCES:
        data = json.loads(source["path"].read_text(encoding="utf-8"))
        rows = data["rows"]
        best = _best_by_sparsity(rows)
        studies.append({"name": source["name"], "path": str(source["path"].relative_to(ROOT)), "best_by_sparsity": best})
        for item in best:
            all_points.append({"study": source["name"], **item})

    positive_points = [p for p in all_points if p["delta_vs_magnitude"] is not None and p["delta_vs_magnitude"] > 0]
    by_sparsity = {}
    for sparsity in sorted({p["sparsity"] for p in all_points}):
        pts = [p for p in all_points if p["sparsity"] == sparsity]
        by_sparsity[str(sparsity)] = {
            "mean_best_alpha": float(np.mean([p["best_alpha"] for p in pts])),
            "mean_delta_vs_magnitude": float(np.mean([p["delta_vs_magnitude"] for p in pts if p["delta_vs_magnitude"] is not None])),
            "positive_deltas": int(sum(1 for p in pts if p["delta_vs_magnitude"] is not None and p["delta_vs_magnitude"] > 0)),
            "n": len(pts),
        }

    result = {
        "experiment": "04_sparsity_cliff_meta_analysis",
        "claim": "Across the current MLP and CNN transfer checks, path correction should be treated as a sparsity-cliff intervention: keep alpha near zero at moderate sparsity and increase it only when magnitude pruning begins to collapse.",
        "studies": studies,
        "by_sparsity": by_sparsity,
        "positive_points": positive_points,
    }
    (OUT / "sparsity_cliff_meta_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Sparsity-Cliff Path Correction",
        "",
        "This meta-analysis extracts the best path-correction strength (`alpha`) from the completed image-model transfer checks.",
        "",
        "## Claim",
        "",
        result["claim"],
        "",
        "## Aggregate by sparsity",
        "",
        "| Sparsity | Mean best alpha | Mean delta vs magnitude | Positive studies | N |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sparsity, item in by_sparsity.items():
        lines.append(f"| `{float(sparsity):.2f}` | `{item['mean_best_alpha']:.3f}` | `{item['mean_delta_vs_magnitude']:.4f}` | `{item['positive_deltas']}` | `{item['n']}` |")

    lines.extend(["", "## Study-level best alpha", "", "| Study | Sparsity | Best alpha | Best accuracy | Magnitude accuracy | Delta |", "|---|---:|---:|---:|---:|---:|"])
    for study in studies:
        for item in study["best_by_sparsity"]:
            lines.append(f"| {study['name']} | `{item['sparsity']:.2f}` | `{item['best_alpha']:.2f}` | `{item['best_accuracy']:.4f}` | `{item['magnitude_accuracy']:.4f}` | `{item['delta_vs_magnitude']:.4f}` |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The useful path signal is not a replacement for magnitude at ordinary sparsity. It becomes useful as the model approaches the compression cliff, especially at `95-98%` sparsity. The practical rule is to keep magnitude dominant and introduce a weak path correction only in severe pruning regimes.",
    ])
    (DOC / "SPARSITY_CLIFF_LAW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"claim": result["claim"], "by_sparsity": result["by_sparsity"]}, indent=2))
