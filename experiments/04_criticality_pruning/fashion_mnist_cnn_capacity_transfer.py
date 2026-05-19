from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_PATH = Path(__file__).resolve().parent / "fashion_mnist_cnn_synflow.py"
spec = importlib.util.spec_from_file_location("fashion_syn", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_PATH}")
fashion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fashion)
base = fashion.base

from shared.path_capacity_pruning import global_topk_mask

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

SEEDS = [191, 192, 193, 194]
SPARSITIES = [0.98, 0.99]
RESERVES = [0.45, 0.55, 0.60]
CRITICAL_LAYERS = ["conv2", "fc1", "fc2"]


def synflow_scores(model):
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 1, 28, 28, device=base.DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in base._params(model).items()}
    for name, param in model.named_parameters():
        param.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def magnitude_scores(model):
    return {k: p.detach().abs().clone() for k, p in base._params(model).items()}


def topk(score, keep):
    flat = score.detach().flatten()
    keep = max(0, min(int(keep), flat.numel()))
    if keep == 0:
        return torch.zeros_like(score)
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return (score >= threshold).to(score.dtype)


def capacity_mask(synflow, magnitude, sparsity, reserve):
    total_params = sum(score.numel() for score in synflow.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    reserve_keep = int(round(reserve * total_keep))
    masks = {name: torch.zeros_like(score) for name, score in synflow.items()}
    protected = {name: torch.zeros_like(score, dtype=torch.bool) for name, score in synflow.items()}
    weights = {"conv2": synflow["conv2"].shape[0], "fc1": synflow["fc1"].shape[0], "fc2": synflow["fc2"].shape[0]}
    total_weight = sum(weights.values())

    for name in CRITICAL_LAYERS:
        budget = max(weights[name], int(round(reserve_keep * weights[name] / total_weight)))
        rank = magnitude[name] if name in {"fc1", "fc2"} else synflow[name]
        masks[name] = torch.maximum(masks[name], topk(rank, budget))

    for name in CRITICAL_LAYERS:
        rank = magnitude[name] if name in {"fc1", "fc2"} else synflow[name]
        layer_mask = masks[name].clone()
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_rank = rank.reshape(rank.shape[0], -1)
        flat_protected = protected[name].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_rank[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[name] = flat_outputs.reshape_as(layer_mask)
        protected[name] = flat_protected.reshape_as(layer_mask)

    selected = int(sum(mask.sum().item() for mask in masks.values()))
    remaining = total_keep - selected
    if remaining > 0:
        chunks = []
        values = []
        for name, score in synflow.items():
            available = masks[name].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                vals = score.flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining, flat_values.numel()), largest=True).values.min()
        left = remaining
        for name, idx, vals in chunks:
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 1
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    elif remaining < 0:
        excess = -remaining
        chunks = []
        values = []
        for name, score in synflow.items():
            removable = masks[name].bool() & ~protected[name]
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = synflow[name].flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, excess, largest=False).values.max()
        left = excess
        for name, idx, vals in chunks:
            chosen = idx[vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 0
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    return masks


def layer_damage(masks):
    out = {}
    for name, mask in masks.items():
        flat_out = mask.reshape(mask.shape[0], -1) if mask.ndim >= 2 else mask.reshape(1, -1)
        out[name] = {"keep_rate": float(mask.float().mean().item()), "dead_output_units": int((flat_out.sum(dim=1) == 0).sum().item()), "output_units": int(flat_out.shape[0])}
    return out


def apply_mask(model, masks):
    for name, param in base._params(model).items():
        param.data.mul_(masks[name].to(param.device))


def masked_finetune(model, train_loader, masks, epochs=3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    apply_mask(model, masks)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            apply_mask(model, masks)


def eval_mask(model, dense_state, train_loader, test_loader, masks):
    before = base._evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune(model, train_loader, masks)
    after = base._evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def fc1_stats(masks):
    fc1 = masks["fc1"]
    return {"fc1_keep_rate": float(fc1.float().mean().item()), "dead_fc1_hidden": int((fc1.sum(dim=1) == 0).sum().item())}


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = base._loaders(seed)
        model = base.SmallCNN().to(base.DEVICE)
        base._train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = base._evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
        mag = magnitude_scores(model)
        syn = synflow_scores(model)
        for sparsity in SPARSITIES:
            masks_by_label = {"magnitude": global_topk_mask(mag, sparsity), "global_synflow": global_topk_mask(syn, sparsity)}
            for reserve in RESERVES:
                masks_by_label[f"reserve_{reserve:.2f}"] = capacity_mask(syn, mag, sparsity, reserve)
            for label, masks in masks_by_label.items():
                before, after = eval_mask(model, dense_state, train_loader, test_loader, masks)
                stats = fc1_stats(masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": label, "reserve": None if not label.startswith("reserve_") else float(label.split("_")[1]), "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "before_retention": before / dense_accuracy, "after_retention": after / dense_accuracy, **stats, "damage": layer_damage(masks)})
                print(f"seed {seed} sparsity {sparsity:.2f} {label}: before={before:.4f} after={after:.4f} dead_fc1={stats['dead_fc1_hidden']}", flush=True)
    methods = ["magnitude", "global_synflow"] + [f"reserve_{r:.2f}" for r in RESERVES]
    summary = {}
    paired = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in methods:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"before_mean": float(np.mean([r["before_accuracy"] for r in selected])), "after_mean": float(np.mean([r["after_accuracy"] for r in selected])), "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])), "fc1_keep_rate_mean": float(np.mean([r["fc1_keep_rate"] for r in selected]))}
            if method != "magnitude":
                deltas = []
                for seed in SEEDS:
                    mag_row = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
                    alt = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == method)
                    deltas.append({"seed": seed, "before_delta": alt["before_accuracy"] - mag_row["before_accuracy"], "after_delta": alt["after_accuracy"] - mag_row["after_accuracy"]})
                paired.append({"sparsity": sparsity, "method": method, "before_delta_mean": float(np.mean([d["before_delta"] for d in deltas])), "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])), "before_wins": int(sum(d["before_delta"] > 0 for d in deltas)), "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)), "paired_rows": deltas})
    result = {"experiment": "04_fashion_mnist_cnn_capacity_transfer", "setup": "Fashion-MNIST CNN transfer test for path-capacity reserve pruning. Same parameter budget, reserves 0.45/0.55/0.60, sparsities 98/99%.", "device": str(base.DEVICE), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsities": SPARSITIES, "reserves": RESERVES, "summary": summary, "paired_deltas": paired, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "fashion_mnist_cnn_capacity_transfer.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "FASHION_MNIST_CNN_CAPACITY_TRANSFER.md"
    lines = ["# Fashion-MNIST CNN Capacity Transfer", "", result["setup"], "", f"Device: `{result['device']}` / `{result['gpu_name']}`", "", "| Sparsity | Method | Before FT | After FT | Dead fc1 | fc1 keep |", "|---:|---|---:|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in methods:
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

