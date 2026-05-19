from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [51, 52]
SPARSITIES = [0.90, 0.95, 0.98]
ALPHAS = [0.0, 0.10, 0.25, 0.50, 1.0]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_acts=False):
        a1 = self.relu(self.conv1(x))
        p1 = self.pool(a1)
        a2 = self.relu(self.conv2(p1))
        p2 = self.pool(a2)
        flat = p2.view(p2.shape[0], -1)
        h = self.relu(self.fc1(flat))
        logits = self.fc2(h)
        if return_acts:
            return logits, {"input": x, "a1": a1, "a2": a2, "flat": flat, "h": h}
        return logits


def _loaders(seed):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    data_dir = ROOT / "data" / "fashion_mnist"
    train = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train))[:20000]
    test_idx = rng.permutation(len(test))[:4000]
    calib_idx = train_idx[:4096]
    return (
        DataLoader(Subset(train, train_idx), batch_size=256, shuffle=True, num_workers=0),
        DataLoader(Subset(test, test_idx), batch_size=512, shuffle=False, num_workers=0),
        DataLoader(Subset(train, calib_idx), batch_size=512, shuffle=False, num_workers=0),
    )


def _train(model, loader):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(4):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


def _params(model):
    return {"conv1": model.conv1.weight, "conv2": model.conv2.weight, "fc1": model.fc1.weight, "fc2": model.fc2.weight}


def _apply_masks(model, masks):
    original = {k: p.data.clone() for k, p in _params(model).items()}
    for k, p in _params(model).items():
        p.data.mul_(masks[k])
    return original


def _restore(model, original):
    for k, p in _params(model).items():
        p.data.copy_(original[k])


def _evaluate(model, loader, masks=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        original = None
        if masks is not None:
            original = _apply_masks(model, masks)
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
        if original is not None:
            _restore(model, original)
    return correct / total


def _activation_stats(model, calib_loader):
    acts = {"input": [], "a1": [], "a2": [], "flat": [], "h": []}
    model.eval()
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(DEVICE)
            _, batch = model(x, return_acts=True)
            for k in acts:
                acts[k].append(batch[k].detach().cpu())
    input_std = torch.cat(acts["input"], dim=0).std(dim=(0, 2, 3)).to(DEVICE) + 1e-6
    a1_strength = torch.cat(acts["a1"], dim=0).abs().mean(dim=(0, 2, 3)).to(DEVICE) + 1e-6
    a2_strength = torch.cat(acts["a2"], dim=0).abs().mean(dim=(0, 2, 3)).to(DEVICE) + 1e-6
    flat_strength = torch.cat(acts["flat"], dim=0).abs().mean(dim=0).to(DEVICE) + 1e-6
    h_strength = torch.cat(acts["h"], dim=0).abs().mean(dim=0).to(DEVICE) + 1e-6
    return input_std, a1_strength, a2_strength, flat_strength, h_strength


def _scores(model, calib_loader, alpha):
    if alpha == 0.0:
        return {k: p.detach().abs().clone() for k, p in _params(model).items()}
    input_std, a1, a2, flat, h = _activation_stats(model, calib_loader)
    fc2_out = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    fc1_importance = h * fc2_out
    # Map fc1 input features back to conv2 channels.
    fc1_in_strength = model.fc1.weight.detach().abs().mean(dim=0)
    conv2_downstream = fc1_in_strength.view(32, 7, 7).mean(dim=(1, 2)) + 1e-6
    conv2_importance = a2 * conv2_downstream
    conv1_downstream = model.conv2.weight.detach().abs().mean(dim=(0, 2, 3)) + 1e-6
    conv1_importance = a1 * conv1_downstream
    return {
        "conv1": model.conv1.weight.detach().abs() * torch.pow(conv1_importance[:, None, None, None] * input_std[None, :, None, None], alpha),
        "conv2": model.conv2.weight.detach().abs() * torch.pow(conv2_importance[:, None, None, None] * conv1_importance[None, :, None, None], alpha),
        "fc1": model.fc1.weight.detach().abs() * torch.pow(fc1_importance[:, None] * flat[None, :], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(fc1_importance[None, :], alpha),
    }


def _mask(scores, sparsity):
    flat = torch.cat([v.flatten() for v in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {k: (v >= threshold).float() for k, v in scores.items()}


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = _loaders(seed)
        model = SmallCNN().to(DEVICE)
        _train(model, train_loader)
        dense_accuracy = _evaluate(model, test_loader)
        for alpha in ALPHAS:
            scores = _scores(model, calib_loader, alpha)
            for sparsity in SPARSITIES:
                masks = _mask(scores, sparsity)
                accuracy = _evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "alpha": alpha, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for alpha in ALPHAS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["alpha"] == alpha]
            summary[str(sparsity)][str(alpha)] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    alpha_scores = {str(alpha): float(np.mean([r["accuracy"] for r in rows if r["alpha"] == alpha])) for alpha in ALPHAS}
    result = {"experiment": "04_torch_fashion_mnist_cnn_path_flow", "device": str(DEVICE), "setup": "Exploratory Fashion-MNIST CNN path-flow transfer. alpha=0 is magnitude; alpha>0 applies channel/path activation correction across conv and fc layers.", "alpha_scores": alpha_scores, "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_cnn_path_flow.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST_CNN.md"
    lines = ["# Torch Fashion-MNIST CNN Path-Flow Exploration", "", result["setup"], "", f"Device: `{DEVICE}`", "", "| Alpha | Mean accuracy over sparsities |", "|---:|---:|"]
    for alpha in ALPHAS:
        lines.append(f"| `{alpha:.2f}` | `{alpha_scores[str(alpha)]:.4f}` |")
    lines.extend(["", "## Per-sparsity", "", "| Sparsity | Alpha | Accuracy | Retention |", "|---:|---:|---:|---:|"])
    for sparsity in SPARSITIES:
        for alpha in ALPHAS:
            item = summary[str(sparsity)][str(alpha)]
            lines.append(f"| `{sparsity:.2f}` | `{alpha:.2f}` | `{item['accuracy_mean']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"device": result["device"], "alpha_scores": result["alpha_scores"], "summary": result["summary"]}, indent=2))
