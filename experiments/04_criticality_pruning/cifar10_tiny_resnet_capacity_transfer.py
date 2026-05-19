from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [201, 202]
SPARSITIES = [0.98, 0.99]
RESERVES = [0.50, 0.60]
TRAIN_N = 20000
TEST_N = 5000
TRAIN_EPOCHS = 4
FT_EPOCHS = 2
BATCH = 256

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.b1 = BasicBlock(16, 16)
        self.b2 = BasicBlock(16, 32, stride=2)
        self.b3 = BasicBlock(32, 64, stride=2)
        self.fc = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def loaders(seed: int):
    g = torch.Generator().manual_seed(seed)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(DATA, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(DATA, train=False, download=True, transform=eval_tf)
    idx = torch.randperm(len(train_ds), generator=g).tolist()
    train_loader = DataLoader(Subset(train_ds, idx[:TRAIN_N]), batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test_ds, list(range(TEST_N))), batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


def params(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = f"{name}.weight" if name else "weight"
            out[key] = module.weight
    return out


def train(model, loader, epochs=TRAIN_EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def evaluate(model, loader, masks=None):
    originals = None
    if masks is not None:
        originals = {k: p.data.clone() for k, p in params(model).items()}
        for k, p in params(model).items():
            p.data.mul_(masks[k].to(p.device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            correct += int((model(x).argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    if originals is not None:
        for k, p in params(model).items():
            p.data.copy_(originals[k])
    return correct / total


def apply_mask(model, masks):
    for k, p in params(model).items():
        p.data.mul_(masks[k].to(p.device))


def masked_finetune(model, loader, masks):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    apply_mask(model, masks)
    for _ in range(FT_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            apply_mask(model, masks)


def magnitude_scores(model):
    return {k: p.detach().abs().clone() for k, p in params(model).items()}


def synflow_scores(model):
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 3, 32, 32, device=DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in params(model).items()}
    for name, p in model.named_parameters():
        p.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def global_mask(scores, sparsity: float):
    flat = torch.cat([v.detach().flatten() for v in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {k: (v >= threshold).to(v.dtype) for k, v in scores.items()}


def topk(score, keep):
    flat = score.detach().flatten()
    keep = max(0, min(int(keep), flat.numel()))
    if keep == 0:
        return torch.zeros_like(score)
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return (score >= threshold).to(score.dtype)


def output_units(score):
    return int(score.shape[0]) if score.ndim >= 2 else 1


def capacity_mask(synflow, magnitude, sparsity: float, reserve: float):
    total_params = sum(v.numel() for v in synflow.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    reserve_keep = int(round(reserve * total_keep))
    critical = [k for k in synflow if k != "stem.weight"]
    weights = {k: output_units(synflow[k]) for k in critical}
    total_weight = sum(weights.values())
    masks = {k: torch.zeros_like(v) for k, v in synflow.items()}
    protected = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in synflow.items()}

    for k in critical:
        budget = max(weights[k], int(round(reserve_keep * weights[k] / total_weight)))
        rank = magnitude[k] if k.endswith("fc.weight") or k == "fc.weight" else synflow[k]
        masks[k] = torch.maximum(masks[k], topk(rank, budget))

    for k in critical:
        rank = magnitude[k] if k.endswith("fc.weight") or k == "fc.weight" else synflow[k]
        layer_mask = masks[k].clone()
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_rank = rank.reshape(rank.shape[0], -1)
        flat_protected = protected[k].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_rank[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[k] = flat_outputs.reshape_as(layer_mask)
        protected[k] = flat_protected.reshape_as(layer_mask)

    selected = int(sum(m.sum().item() for m in masks.values()))
    remaining = total_keep - selected
    if remaining > 0:
        chunks = []
        values = []
        for k, score in synflow.items():
            available = masks[k].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                vals = score.flatten()[idx]
                chunks.append((k, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining, flat_values.numel()), largest=True).values.min()
        left = remaining
        for k, idx, vals in chunks:
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[k].flatten()
            flat[chosen] = 1
            masks[k] = flat.reshape_as(masks[k])
            left -= int(chosen.numel())
            if left <= 0:
                break
    elif remaining < 0:
        excess = -remaining
        chunks = []
        values = []
        for k, score in synflow.items():
            removable = masks[k].bool() & ~protected[k]
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = synflow[k].flatten()[idx]
                chunks.append((k, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, excess, largest=False).values.max()
        left = excess
        for k, idx, vals in chunks:
            chosen = idx[vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[k].flatten()
            flat[chosen] = 0
            masks[k] = flat.reshape_as(masks[k])
            left -= int(chosen.numel())
            if left <= 0:
                break
    return masks


def damage(masks):
    out = {}
    for k, mask in masks.items():
        flat = mask.reshape(mask.shape[0], -1)
        out[k] = {"keep_rate": float(mask.float().mean().item()), "dead_outputs": int((flat.sum(dim=1) == 0).sum().item()), "outputs": int(flat.shape[0])}
    return out


def eval_method(model, dense_state, train_loader, test_loader, masks):
    before = evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune(model, train_loader, masks)
    after = evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def run():
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = loaders(seed)
        model = TinyResNet().to(DEVICE)
        train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = evaluate(model, test_loader)
        print(f"seed {seed}: dense_accuracy={dense_accuracy:.4f} device={DEVICE}", flush=True)
        mag = magnitude_scores(model)
        syn = synflow_scores(model)
        for sparsity in SPARSITIES:
            masks_by_label = {"magnitude": global_mask(mag, sparsity), "global_synflow": global_mask(syn, sparsity)}
            for reserve in RESERVES:
                masks_by_label[f"reserve_{reserve:.2f}"] = capacity_mask(syn, mag, sparsity, reserve)
            for label, masks in masks_by_label.items():
                before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
                dmg = damage(masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": label, "reserve": None if not label.startswith("reserve_") else float(label.split("_")[1]), "dense_accuracy": dense_accuracy, "before_accuracy": before, "after_accuracy": after, "before_retention": before / dense_accuracy, "after_retention": after / dense_accuracy, "damage": dmg})
                dead_total = sum(v["dead_outputs"] for v in dmg.values())
                print(f"seed {seed} sparsity {sparsity:.2f} {label}: before={before:.4f} after={after:.4f} dead_outputs={dead_total}", flush=True)
    methods = ["magnitude", "global_synflow"] + [f"reserve_{r:.2f}" for r in RESERVES]
    summary = {}
    paired = []
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in methods:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"before_mean": float(np.mean([r["before_accuracy"] for r in selected])), "after_mean": float(np.mean([r["after_accuracy"] for r in selected])), "dead_outputs_mean": float(np.mean([sum(v["dead_outputs"] for v in r["damage"].values()) for r in selected]))}
            if method != "magnitude":
                deltas = []
                for seed in SEEDS:
                    mag_row = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == "magnitude")
                    alt = next(r for r in rows if r["seed"] == seed and r["sparsity"] == sparsity and r["method"] == method)
                    deltas.append({"seed": seed, "before_delta": alt["before_accuracy"] - mag_row["before_accuracy"], "after_delta": alt["after_accuracy"] - mag_row["after_accuracy"]})
                paired.append({"sparsity": sparsity, "method": method, "before_delta_mean": float(np.mean([d["before_delta"] for d in deltas])), "after_delta_mean": float(np.mean([d["after_delta"] for d in deltas])), "before_wins": int(sum(d["before_delta"] > 0 for d in deltas)), "after_wins": int(sum(d["after_delta"] > 0 for d in deltas)), "paired_rows": deltas})
    result = {"experiment": "04_cifar10_tiny_resnet_capacity_transfer", "setup": "CIFAR-10 TinyResNet transfer test for path-capacity reserve pruning under residual connections.", "device": DEVICE, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seeds": SEEDS, "sparsities": SPARSITIES, "reserves": RESERVES, "summary": summary, "paired_deltas": paired, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_tiny_resnet_capacity_transfer.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_RESNET_CAPACITY_TRANSFER.md"
    lines = ["# CIFAR-10 TinyResNet Capacity Transfer", "", result["setup"], "", f"Device: `{DEVICE}` / `{result['gpu_name']}`", "", "| Sparsity | Method | Before FT | After FT | Dead outputs |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in methods:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |")
    lines.extend(["", "## Paired deltas vs magnitude", "", "| Sparsity | Method | Before delta | Before wins | After delta | After wins |", "|---:|---|---:|---:|---:|---:|"])
    for item in paired:
        lines.append(f"| `{item['sparsity']:.2f}` | `{item['method']}` | `{item['before_delta_mean']:+.4f}` | `{item['before_wins']}/{len(SEEDS)}` | `{item['after_delta_mean']:+.4f}` | `{item['after_wins']}/{len(SEEDS)}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
