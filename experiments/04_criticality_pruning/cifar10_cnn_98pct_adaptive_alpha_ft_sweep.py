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
SEEDS = [61, 62]
SPARSITY = 0.98
ALPHAS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
TRAIN_N = 20000
TEST_N = 5000
CALIB_N = 1024
TRAIN_EPOCHS = 4
FT_EPOCHS = 3
BATCH = 128


class SmallCifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(64 * 4 * 4, 192, bias=False)
        self.fc2 = nn.Linear(192, 10, bias=False)

    def forward(self, x, return_acts: bool = False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        flat = torch.flatten(x, 1)
        h = F.relu(self.fc1(flat))
        out = self.fc2(h)
        if return_acts:
            return out, {"flat": flat, "h": h}
        return out


def _loaders(seed: int):
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
    calib_ds = datasets.CIFAR10(DATA, train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10(DATA, train=False, download=True, transform=eval_tf)
    idx = torch.randperm(len(train_ds), generator=g).tolist()
    train_idx = idx[:TRAIN_N]
    calib_idx = idx[TRAIN_N:TRAIN_N + CALIB_N]
    test_idx = torch.arange(TEST_N).tolist()
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH, shuffle=True, num_workers=0)
    calib_loader = DataLoader(Subset(calib_ds, calib_idx), batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader, calib_loader


def _params(model):
    return {
        "conv1": model.conv1.weight,
        "conv2": model.conv2.weight,
        "conv3": model.conv3.weight,
        "fc1": model.fc1.weight,
        "fc2": model.fc2.weight,
    }


def _train(model, loader, epochs=TRAIN_EPOCHS):
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


def _evaluate(model, loader, masks=None):
    originals = None
    if masks is not None:
        originals = {k: p.data.clone() for k, p in _params(model).items()}
        for k, p in _params(model).items():
            p.data.mul_(masks[k].to(p.device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    if originals is not None:
        for k, p in _params(model).items():
            p.data.copy_(originals[k])
    return correct / total


def _mask(scores, sparsity: float):
    flat = torch.cat([v.detach().flatten() for v in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {k: (v >= threshold).to(v.dtype) for k, v in scores.items()}


def _apply_mask(model, masks):
    for k, p in _params(model).items():
        p.data.mul_(masks[k].to(p.device))


def _masked_finetune(model, train_loader, masks):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _apply_mask(model, masks)
    for _ in range(FT_EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            _apply_mask(model, masks)


def _hybrid_scores(model, calib_loader, alpha: float):
    model.eval()
    flats = []
    hiddens = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(DEVICE)
            _, acts = model(x, return_acts=True)
            flats.append(acts["flat"].detach().cpu())
            hiddens.append(acts["h"].detach().cpu())
    flat_all = torch.cat(flats, dim=0).to(DEVICE)
    h_all = torch.cat(hiddens, dim=0).to(DEVICE)
    flat_signal = flat_all.std(dim=0).clamp_min(1e-6)
    hidden_strength = h_all.abs().mean(dim=0).clamp_min(1e-6)
    output_strength = model.fc2.weight.detach().abs().mean(dim=0).clamp_min(1e-6)
    hidden = hidden_strength * output_strength
    return {
        "conv1": model.conv1.weight.detach().abs().clone(),
        "conv2": model.conv2.weight.detach().abs().clone(),
        "conv3": model.conv3.weight.detach().abs().clone(),
        "fc1": model.fc1.weight.detach().abs() * torch.pow(hidden[:, None] * flat_signal[None, :], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(hidden[None, :], alpha),
    }


def _fc1_stats(masks):
    fc1 = masks["fc1"]
    return {
        "fc1_keep_rate": float(fc1.mean().item()),
        "dead_fc1_hidden": int((fc1.sum(dim=1) == 0).sum().item()),
    }


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = _loaders(seed)
        model = SmallCifarCNN().to(DEVICE)
        _train(model, train_loader)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_accuracy = _evaluate(model, test_loader)
        for alpha in ALPHAS:
            scores = _hybrid_scores(model, calib_loader, alpha)
            masks = _mask(scores, SPARSITY)
            before = _evaluate(model, test_loader, masks)
            model.load_state_dict(dense_state)
            _masked_finetune(model, train_loader, masks)
            after = _evaluate(model, test_loader)
            item = {
                "seed": seed,
                "alpha": alpha,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "before_retention": before / dense_accuracy,
                "after_retention": after / dense_accuracy,
            }
            item.update(_fc1_stats(masks))
            rows.append(item)
            model.load_state_dict(dense_state)
    summary = {}
    for alpha in ALPHAS:
        selected = [r for r in rows if r["alpha"] == alpha]
        summary[str(alpha)] = {
            "before_mean": float(np.mean([r["before_accuracy"] for r in selected])),
            "after_mean": float(np.mean([r["after_accuracy"] for r in selected])),
            "before_retention_mean": float(np.mean([r["before_retention"] for r in selected])),
            "after_retention_mean": float(np.mean([r["after_retention"] for r in selected])),
            "fc1_keep_rate_mean": float(np.mean([r["fc1_keep_rate"] for r in selected])),
            "dead_fc1_hidden_mean": float(np.mean([r["dead_fc1_hidden"] for r in selected])),
        }
    best_before = max(summary.items(), key=lambda kv: kv[1]["before_mean"])
    best_after = max(summary.items(), key=lambda kv: kv[1]["after_mean"])
    best_balanced = max(summary.items(), key=lambda kv: kv[1]["before_mean"] + kv[1]["after_mean"])
    result = {
        "experiment": "04_cifar10_cnn_98pct_adaptive_alpha_ft_sweep",
        "setup": "CIFAR-10 small CNN transfer test for 98% adaptive dense-tail path correction. Conv layers use magnitude; dense tail uses low-alpha path modulation. Real CIFAR-10 images, 20k train subset, 5k test subset, 2 seeds.",
        "device": DEVICE,
        "train_n": TRAIN_N,
        "test_n": TEST_N,
        "calib_n": CALIB_N,
        "sparsity": SPARSITY,
        "train_epochs": TRAIN_EPOCHS,
        "finetune_epochs": FT_EPOCHS,
        "summary": summary,
        "best_before": {"alpha": float(best_before[0]), **best_before[1]},
        "best_after": {"alpha": float(best_after[0]), **best_after[1]},
        "best_balanced": {"alpha": float(best_balanced[0]), **best_balanced[1]},
        "rows": rows,
    }
    out = ROOT / "results" / "04_criticality_pruning" / "cifar10_cnn_98pct_adaptive_alpha_ft_sweep.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_CNN_98PCT_ADAPTIVE_ALPHA_FT_SWEEP.md"
    lines = ["# CIFAR-10 CNN 98% Adaptive Alpha Fine-Tuning Sweep", "", result["setup"], "", f"Device: `{DEVICE}`", "", "| Alpha | Before FT | After FT | Before retention | After retention | fc1 keep rate | Dead fc1 hidden |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        item = summary[str(alpha)]
        lines.append(f"| `{alpha:.2f}` | `{item['before_mean']:.4f}` | `{item['after_mean']:.4f}` | `{item['before_retention_mean']:.4f}` | `{item['after_retention_mean']:.4f}` | `{item['fc1_keep_rate_mean']:.4f}` | `{item['dead_fc1_hidden_mean']:.1f}` |")
    lines.extend(["", f"Best one-shot alpha: `{result['best_before']['alpha']:.2f}`.", f"Best after-FT alpha: `{result['best_after']['alpha']:.2f}`.", f"Best balanced alpha: `{result['best_balanced']['alpha']:.2f}`."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"best_before": result["best_before"], "best_after": result["best_after"], "best_balanced": result["best_balanced"], "summary": result["summary"]}, indent=2))
