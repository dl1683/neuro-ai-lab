from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "04_criticality_pruning"
DOC = ROOT / "experiments" / "04_criticality_pruning"

COMPARISONS = [
    {"name": "Fashion-MNIST MLP", "path": OUT / "fashion_mnist_mlp_synflow_comparison.json", "adaptive": "adaptive_path"},
    {"name": "Fashion-MNIST CNN", "path": OUT / "fashion_mnist_cnn_synflow_comparison.json", "adaptive": "adaptive_dense_hybrid"},
]


def run():
    rows = []
    for spec in COMPARISONS:
        data = json.loads(spec["path"].read_text(encoding="utf-8"))
        for r in data["rows"]:
            if r["method"] in ["magnitude", "synflow", spec["adaptive"]]:
                rows.append({"study": spec["name"], "adaptive_method": spec["adaptive"], **r})

    paired = []
    for study in sorted({r["study"] for r in rows}):
        adaptive_name = next(r["adaptive_method"] for r in rows if r["study"] == study)
        for seed in sorted({r["seed"] for r in rows if r["study"] == study}):
            for sparsity in sorted({r["sparsity"] for r in rows if r["study"] == study}):
                adaptive = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == adaptive_name)
                synflow = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "synflow")
                magnitude = next(r for r in rows if r["study"] == study and r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
                paired.append({"study": study, "seed": seed, "sparsity": sparsity, "adaptive_accuracy": adaptive["accuracy"], "synflow_accuracy": synflow["accuracy"], "magnitude_accuracy": magnitude["accuracy"], "adaptive_minus_synflow": adaptive["accuracy"] - synflow["accuracy"], "adaptive_minus_magnitude": adaptive["accuracy"] - magnitude["accuracy"]})

    def summary(rows_selected, key):
        vals = [r[key] for r in rows_selected]
        return {"mean": float(np.mean(vals)), "wins": int(sum(v > 0 for v in vals)), "ties": int(sum(v == 0 for v in vals)), "losses": int(sum(v < 0 for v in vals)), "n": len(vals)}

    by_sparsity = {}
    for sparsity in sorted({r["sparsity"] for r in paired}):
        selected = [r for r in paired if r["sparsity"] == sparsity]
        by_sparsity[str(sparsity)] = {"vs_synflow": summary(selected, "adaptive_minus_synflow"), "vs_magnitude": summary(selected, "adaptive_minus_magnitude")}
    overall = {"vs_synflow": summary(paired, "adaptive_minus_synflow"), "vs_magnitude": summary(paired, "adaptive_minus_magnitude")}
    severe = [r for r in paired if r["sparsity"] == 0.98]
    result = {"experiment": "04_label_free_baseline_synthesis", "claim": "Against SynFlow and magnitude on Fashion-MNIST MLP/CNN, adaptive path correction is most valuable at the severe sparsity cliff. It is not always best at 90-95%, but at 98% it beats both baselines in every paired case currently tested.", "overall": overall, "by_sparsity": by_sparsity, "severe_sparsity_98": {"vs_synflow": summary(severe, "adaptive_minus_synflow"), "vs_magnitude": summary(severe, "adaptive_minus_magnitude")}, "paired_rows": paired}
    (OUT / "label_free_baseline_synthesis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# Label-Free Baseline Synthesis", "", result["claim"], "", "## Overall", "", "| Comparison | Mean delta | Wins | Ties | Losses | N |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in overall.items():
        lines.append(f"| `{name}` | `{item['mean']:.4f}` | `{item['wins']}` | `{item['ties']}` | `{item['losses']}` | `{item['n']}` |")
    lines.extend(["", "## By sparsity", "", "| Sparsity | Baseline | Mean delta | Wins | Ties | Losses | N |", "|---:|---|---:|---:|---:|---:|---:|"])
    for sparsity, comps in by_sparsity.items():
        for baseline, item in comps.items():
            lines.append(f"| `{float(sparsity):.2f}` | `{baseline}` | `{item['mean']:.4f}` | `{item['wins']}` | `{item['ties']}` | `{item['losses']}` | `{item['n']}` |")
    lines.extend(["", "## Paired rows", "", "| Study | Seed | Sparsity | Adaptive | SynFlow | Magnitude | Delta vs SynFlow | Delta vs Magnitude |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in paired:
        lines.append(f"| {r['study']} | `{r['seed']}` | `{r['sparsity']:.2f}` | `{r['adaptive_accuracy']:.4f}` | `{r['synflow_accuracy']:.4f}` | `{r['magnitude_accuracy']:.4f}` | `{r['adaptive_minus_synflow']:.4f}` | `{r['adaptive_minus_magnitude']:.4f}` |")
    (DOC / "LABEL_FREE_BASELINE_SYNTHESIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"claim": result["claim"], "overall": result["overall"], "severe_sparsity_98": result["severe_sparsity_98"]}, indent=2))
