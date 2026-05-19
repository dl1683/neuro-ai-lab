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

SEEDS = [111, 112, 113, 114]
SPARSITIES = [0.98, 0.99]
METHODS = ["magnitude", "alpha03", "bridge_floor1", "bridge_floor2"]

base.TRAIN_N = 20000
base.TEST_N = 5000
base.CALIB_N = 1024
base.BATCH = 256
base.TRAIN_EPOCHS = 4
base.FT_EPOCHS = 3


def magnitude_scores(model):
    return {k: p.detach().abs().clone() for k, p in base._params(model).items()}


def bridge_floor_mask(scores, sparsity: float, min_fc1_fanin: int):
    masks = base._mask(scores, sparsity)
    target_keep = sum(int(m.sum().item()) for m in masks.values())
    fc1_scores = scores["fc1"]
    fc1_mask = masks["fc1"].clone()
    protected = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in masks.items()}

    for row in range(fc1_mask.shape[0]):
        current = int(fc1_mask[row].sum().item())
        if current < min_fc1_fanin:
            need = min_fc1_fanin - current
            row_scores = fc1_scores[row].clone()
            row_scores[fc1_mask[row].bool()] = -1
            add_idx = torch.topk(row_scores, need).indices
            fc1_mask[row, add_idx] = 1
        kept_idx = torch.nonzero(fc1_mask[row].bool(), as_tuple=False).flatten()
        if kept_idx.numel() > 0:
            top_kept = kept_idx[torch.topk(fc1_scores[row, kept_idx], min(min_fc1_fanin, kept_idx.numel())).indices]
            protected["fc1"][row, top_kept] = True
    masks["fc1"] = fc1_mask

    current_keep = sum(int(m.sum().item()) for m in masks.values())
    excess = current_keep - target_keep
    if excess > 0:
        candidates = []
        for name, mask in masks.items():
            removable = mask.bool() & ~protected[name]
            if removable.any():
                idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
                vals = scores[name].flatten()[idx]
                candidates.append((name, idx, vals))
        all_vals = torch.cat([c[2] for c in candidates])
        remove_threshold = torch.topk(all_vals, excess, largest=False).values.max()
        remaining = excess
        for name, idx, vals in candidates:
            remove_local = idx[vals <= remove_threshold]
            if remove_local.numel() > remaining:
                remove_local = remove_local[:remaining]
            flat = masks[name].flatten()
            flat[remove_local] = 0
            masks[name] = flat.reshape_as(masks[name])
            remaining -= int(remove_local.numel())
            if remaining <= 0:
                break
    return masks


def evaluate_method(model, dense_state, train_loader, test_loader, calib_loader, sparsity, method):
    if method == "magnitude":
        scores = magnitude_scores(model)
        masks = base._mask(scores, sparsity)
    elif method == "alpha03":
        scores = base._hybrid_scores(model, calib_loader, 0.03)
        masks = base._mask(scores, sparsity)
    elif method == "bridge_floor1":
        scores = magnitude_scores(model)
        masks = bridge_floor_mask(scores, sparsity, 1)
    elif method == "bridge_floor2":
        scores = magnitude_scores(model)
        masks = bridge_floor_mask(scores, sparsity, 2)
    else:
        raise ValueError(method)
    before = base._evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    base._masked_finetune(model, train_loader, masks)
    after = base._evaluate(model, test_loader)
    stats = base._fc1_stats(masks)
    model.load_state_dict(dense_state)
    return before, after, stats


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
        for sparsity in SPARSITIES:
            for method in METHODS:
                before, after, stats = evaluate_method(model, dense_state, train_loader, test_loader, calib_loader, sparsity, method)
                rows.append({
                    "seed": seed,
                    "sparsity": sparsity,
                    "method": method,
                    "dense_accuracy": dense_accuracy,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "before_retention": before / dense_accuracy,
                    "after_retention": after / dense_accuracy,
                    **stats,
                })
                print(f"seed {seed} sparsity {sparsity:.2f} {method}: before={before:.4f} after={after:.4f} dead_fc1={stats['dead_fc1_hidden']}", flush=True)
    summary = {}
    paired = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {
                "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
                "fc1_keep_rate_mean": float(np.mean([r["fc1_keep_rate"] for r in selected])),
            }
        for method in [m for m in METHODS if m != "magnitude"]:
            deltas = []
            for seed in SEEDS:
                mag = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
                alt = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == method)
                deltas.append({"seed": seed, "before_delta": alt["before_accuracy"] - mag["before_accuracy"], "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
            paired.append({
                "sparsity": sparsity,
                "method": method,
                "before_delta_mean": float(np.mean([d["before_delta"] for d in deltas])),
                "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])),
                "before_wins": int(sum(d["before_delta"] > 0 for d in deltas)),
                "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)),
                "paired_rows": deltas,
            })
    result = {
        "experiment": "04_cifar10_cnn_bridge_floor",
        "setup": "CIFAR-10 CNN structural bridge-floor test. Repairs magnitude masks so each fc1 hidden unit has at least 1 or 2 incoming weights, preserving total keep count by removing low-score unprotected weights elsewhere.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsities": SPARSITIES,
        "methods": METHODS,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_cnn_bridge_floor.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_CNN_BRIDGE_FLOOR.md"
    lines = ["# CIFAR-10 CNN Bridge-Floor Test", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", "", "## Means", "", "| Sparsity | Method | Before FT | After FT | Dead fc1 hidden | fc1 keep rate |", "|---:|---|---:|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` | `{item['fc1_keep_rate_mean']:.4f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Method | Before delta | Before wins | After delta | After wins |", "|---:|---|---:|---:|---:|---:|"])
    for item in paired:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(SEEDS)}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
