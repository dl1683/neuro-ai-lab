from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("criticality_run", RUN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {RUN_PATH}")
criticality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(criticality)

SEEDS = [23, 24, 25, 26, 27]
SPARSITIES = [0.90, 0.95]
CALIBRATION_FRACTIONS = [0.02, 0.05, 0.10, 0.25, 1.00]


def _path_flow_scores(model, x_calib):
    input_flow, hidden_strength, hidden_balance, _ = criticality._feature_stats(model, x_calib)
    output_strength = np.mean(np.abs(model["W2"]), axis=1) + 1e-6
    hidden = hidden_strength * hidden_balance * output_strength
    return {
        "W1": np.abs(model["W1"]) * input_flow[:, None] * hidden[None, :],
        "W2": np.abs(model["W2"]) * hidden[:, None],
    }


def run() -> dict:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 404)
        model, x_train, x_test, y_train, y_test = criticality._train(seed)
        dense_accuracy = criticality._accuracy(model, x_test, y_test)
        order = rng.permutation(len(x_train))
        for frac in CALIBRATION_FRACTIONS:
            n = max(1, int(round(frac * len(x_train))))
            x_calib = x_train[order[:n]]
            scores = _path_flow_scores(model, x_calib)
            for sparsity in SPARSITIES:
                mask = criticality._mask_from_scores(scores, sparsity, "path_flow")
                accuracy = criticality._accuracy(model, x_test, y_test, mask)
                rows.append(
                    {
                        "seed": seed,
                        "calibration_fraction": frac,
                        "calibration_examples": n,
                        "sparsity": sparsity,
                        "dense_accuracy": dense_accuracy,
                        "accuracy": accuracy,
                        "accuracy_retention": accuracy / dense_accuracy,
                        "hidden_coverage": criticality._hidden_coverage(mask),
                    }
                )

    summary = {}
    for frac in CALIBRATION_FRACTIONS:
        rows_f = [r for r in rows if r["calibration_fraction"] == frac]
        summary[str(frac)] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows_f])),
            "accuracy_std": float(np.std([r["accuracy"] for r in rows_f])),
            "retention_mean": float(np.mean([r["accuracy_retention"] for r in rows_f])),
            "hidden_coverage_mean": float(np.mean([r["hidden_coverage"] for r in rows_f])),
            "mean_calibration_examples": float(np.mean([r["calibration_examples"] for r in rows_f])),
        }

    full = summary["1.0"]["accuracy_mean"]
    result = {
        "experiment": "04_path_flow_calibration_sweep",
        "setup": "Evaluate label-free path-flow pruning using only a small unlabeled calibration subset to estimate activation statistics. Accuracy is averaged over 90% and 95% sparsity across five trained sklearn-digits MLP seeds.",
        "summary": summary,
        "accuracy_drop_vs_full_calibration": {frac: float(full - summary[str(frac)]["accuracy_mean"]) for frac in CALIBRATION_FRACTIONS if frac != 1.0},
        "rows": rows,
    }

    out = ROOT / "results" / "04_criticality_pruning" / "path_flow_calibration_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = ROOT / "experiments" / "04_criticality_pruning" / "PATH_FLOW_CALIBRATION.md"
    lines = [
        "# Path-Flow Calibration Sweep",
        "",
        "This experiment asks whether label-free path-flow pruning needs the full training set to estimate activation statistics, or whether a tiny unlabeled calibration batch is enough.",
        "",
        "Setup: trained sklearn-digits MLPs, `90%` and `95%` sparsity, five seeds.",
        "",
        "| Calibration fraction | Mean examples | Mean accuracy | Std | Mean retention | Drop vs full calibration |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for frac in CALIBRATION_FRACTIONS:
        item = summary[str(frac)]
        drop = 0.0 if frac == 1.0 else full - item["accuracy_mean"]
        lines.append(f"| `{frac:.2f}` | `{item['mean_calibration_examples']:.1f}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` | `{drop:.4f}` |")
    lines.extend([
        "",
        "Interpretation: small drops mean the pruning score is deployable with only a small unlabeled calibration batch. Large drops mean path-flow depends on stable activation estimates from broad data coverage.",
    ])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "accuracy_drop_vs_full_calibration": result["accuracy_drop_vs_full_calibration"]}, indent=2))
