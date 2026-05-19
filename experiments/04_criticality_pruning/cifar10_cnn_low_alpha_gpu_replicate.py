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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

SEEDS = [101, 102, 103, 104, 105, 106]
SPARSITIES = [0.95, 0.98, 0.99]
ALPHAS = [0.0, 0.03, 0.05]

# Keep this comparable to the earlier CIFAR probes while increasing seed count.
base.TRAIN_N = 20000
base.TEST_N = 5000
base.CALIB_N = 1024
base.BATCH = 256
base.TRAIN_EPOCHS = 4
base.FT_EPOCHS = 3


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCifarCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base._evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        score_cache = {alpha: base._hybrid_scores(model, calib_loader, alpha) for alpha in ALPHAS}
        for sparsity in SPARSITIES:
            for alpha in ALPHAS:
                masks = base._mask(score_cache[alpha], sparsity)
                before = base._evaluate(model, test_loader, masks)
                model.load_state_dict(dense_state)
                base._masked_finetune(model, train_loader, masks)
                after = base._evaluate(model, test_loader)
                stats = base._fc1_stats(masks)
                row = {
                    "seed": seed,
                    "sparsity": sparsity,
                    "alpha": alpha,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "before_retention": before / dense_accuracy,
                    "after_retention": after / dense_accuracy,
                    **stats,
                }
                rows.append(row)
                print(f"seed {seed} sparsity {sparsity:.2f} alpha {alpha:.2f}: before={before:.4f} after={after:.4f}", flush=True)
                model.load_state_dict(dense_state)
    summary = {}
    deltas = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for alpha in ALPHAS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["alpha"] == alpha]
            summary[str(sparsity)][str(alpha)] = {
                "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
                "before_std": float(np.std([r["before_accuracy"] for r in selected])),
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "after_std": float(np.std([r["after_accuracy"] for r in selected])),
                "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
            }
        for alpha in [0.03, 0.05]:
            paired_rows = []
            for seed in SEEDS:
                mag = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["alpha"] == 0.0)
                path = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["alpha"] == alpha)
                paired_rows.append({
                    "seed": seed,
                    "before_delta": path["before_accuracy"] - mag["before_accuracy"],
                    "after_delta": path["after_accuracy"] - mag["after_accuracy"],
                })
            deltas.append({
                "sparsity": sparsity,
                "alpha": alpha,
                "before_delta_mean": float(np.mean([r["before_delta"] for r in paired_rows])),
                "before_delta_std": float(np.std([r["before_delta"] for r in paired_rows])),
                "after_delta_mean": float(np.mean([r["after_delta"] for r in paired_rows])),
                "after_delta_std": float(np.std([r["after_delta"] for r in paired_rows])),
                "before_wins": int(sum(r["before_delta"] > 0 for r in paired_rows)),
                "after_wins": int(sum(r["after_delta"] > 0 for r in paired_rows)),
                "paired_rows": paired_rows,
            })
    result = {
        "experiment": "04_cifar10_cnn_low_alpha_gpu_replicate",
        "setup": "CIFAR-10 small CNN low-alpha path-correction GPU replicate. Six seeds, real CIFAR-10 images, 20k train subset, 5k test subset, paired magnitude vs alpha masks across 95/98/99% sparsity.",
        "device": base.DEVICE,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsities": SPARSITIES,
        "alphas": ALPHAS,
        "train_n": base.TRAIN_N,
        "test_n": base.TEST_N,
        "train_epochs": base.TRAIN_EPOCHS,
        "finetune_epochs": base.FT_EPOCHS,
        "summary": summary,
        "paired_deltas": deltas,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_cnn_low_alpha_gpu_replicate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_CNN_LOW_ALPHA_GPU_REPLICATE.md"
    lines = ["# CIFAR-10 CNN Low-Alpha GPU Replicate", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", "", "## Means", "", "| Sparsity | Alpha | Before FT | After FT | Dead fc1 hidden |", "|---:|---:|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for alpha in ALPHAS:
            item = summary[str(sparsity)][str(alpha)]
            lines.append(f"| `{sparsity:.2f}` | `{alpha:.2f}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Alpha | Before delta | Before wins | After delta | After wins |", "|---:|---:|---:|---:|---:|---:|"])
    for item in deltas:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['alpha']:.2f}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(SEEDS)}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"device": result["device"], "gpu_name": result["gpu_name"], "paired_deltas": result["paired_deltas"], "summary": result["summary"]}, indent=2))
