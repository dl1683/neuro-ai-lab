from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).resolve().parent / "cifar10_cnn_98pct_adaptive_alpha_ft_sweep.py"
spec = importlib.util.spec_from_file_location("cifar98", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from shared.path_capacity_pruning import DenseBridgeConstraint, dense_bridge_capacity_mask, global_topk_mask

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

SEEDS = [131, 132, 133, 134]
SPARSITIES = [0.98, 0.99]
METHODS = [
    "magnitude",
    "global_synflow",
    "pathcap_synflow_bridge_mag",
    "pathcap_synflow_bridge_hybrid",
]

base.TRAIN_N = 20000
base.TEST_N = 5000
base.CALIB_N = 1024
base.BATCH = 256
base.TRAIN_EPOCHS = 4
base.FT_EPOCHS = 3


def synflow_scores(model):
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 3, 32, 32, device=base.DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in base._params(model).items()}
    for name, param in model.named_parameters():
        param.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def magnitude_scores(model):
    return {k: p.detach().abs().clone() for k, p in base._params(model).items()}


def mask_for_method(model, calib_loader, method: str, sparsity: float):
    magnitude = magnitude_scores(model)
    if method == "magnitude":
        return global_topk_mask(magnitude, sparsity)

    synflow = synflow_scores(model)
    if method == "global_synflow":
        return global_topk_mask(synflow, sparsity)

    if sparsity >= 0.99:
        bridge_floor = 0.004
    else:
        bridge_floor = 0.008
    constraint = DenseBridgeConstraint(name="fc1", min_keep_rate=bridge_floor, min_fanin_per_hidden=1)

    if method == "pathcap_synflow_bridge_mag":
        return dense_bridge_capacity_mask(synflow, sparsity, constraint, bridge_scores=magnitude)

    if method == "pathcap_synflow_bridge_hybrid":
        hybrid = base._hybrid_scores(model, calib_loader, 0.03)
        return dense_bridge_capacity_mask(synflow, sparsity, constraint, bridge_scores=hybrid)

    raise ValueError(method)


def layer_damage(masks):
    out = {}
    for name, mask in masks.items():
        flat_out = mask.reshape(mask.shape[0], -1) if mask.ndim >= 2 else mask.reshape(1, -1)
        out[name] = {
            "keep_rate": float(mask.float().mean().item()),
            "dead_output_units": int((flat_out.sum(dim=1) == 0).sum().item()),
            "output_units": int(flat_out.shape[0]),
        }
    return out


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
                model.load_state_dict(dense_state)
                masks = mask_for_method(model, calib_loader, method, sparsity)
                before = base._evaluate(model, test_loader, masks)
                model.load_state_dict(dense_state)
                base._masked_finetune(model, train_loader, masks)
                after = base._evaluate(model, test_loader)
                stats = base._fc1_stats(masks)
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
                    "damage": layer_damage(masks),
                })
                print(f"seed {seed} sparsity {sparsity:.2f} {method}: before={before:.4f} after={after:.4f} fc1_keep={stats['fc1_keep_rate']:.4f} dead_fc1={stats['dead_fc1_hidden']}", flush=True)
    summary = {}
    paired = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {
                "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
                "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
                "fc1_keep_rate_mean": float(np.mean([r["fc1_keep_rate"] for r in selected])),
                "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
                "conv1_keep_rate_mean": float(np.mean([r["damage"]["conv1"]["keep_rate"] for r in selected])),
                "conv2_keep_rate_mean": float(np.mean([r["damage"]["conv2"]["keep_rate"] for r in selected])),
                "conv3_keep_rate_mean": float(np.mean([r["damage"]["conv3"]["keep_rate"] for r in selected])),
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
        "experiment": "04_cifar10_cnn_path_capacity_pruning",
        "setup": "CIFAR-10 CNN first path-capacity pruning test. SynFlow saliency is constrained so fc1 keeps a minimum bridge capacity; bridge ranking uses magnitude or activation-supported hybrid scores. Same global parameter budget as ordinary global pruning.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsities": SPARSITIES,
        "methods": METHODS,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_cnn_path_capacity_pruning.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_CNN_PATH_CAPACITY_PRUNING.md"
    lines = ["# CIFAR-10 CNN Path-Capacity Pruning", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", "", "## Means", "", "| Sparsity | Method | Before FT | After FT | fc1 keep rate | Dead fc1 hidden | conv keep rates |", "|---:|---|---:|---:|---:|---:|---|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            convs = f"{item['conv1_keep_rate_mean']:.2f}/{item['conv2_keep_rate_mean']:.2f}/{item['conv3_keep_rate_mean']:.2f}"
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['fc1_keep_rate_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` | `{convs}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Method | Before delta | Before wins | After delta | After wins |", "|---:|---|---:|---:|---:|---:|"])
    for item in paired:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(SEEDS)}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))


