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
SPARSITIES = [0.90, 0.95, 0.98]
ALPHAS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
SEEDS = [41, 42]


class MLP(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x, return_hidden: bool = False):
        flat = x.view(x.shape[0], -1)
        h = self.relu(self.fc1(flat))
        logits = self.fc2(h)
        if return_hidden:
            return logits, flat, h
        return logits


def _loaders(seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    data_dir = ROOT / "data" / "cifar10"
    train = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train))[:20000]
    test_idx = rng.permutation(len(test))[:5000]
    calib_idx = train_idx[:4096]
    return (
        DataLoader(Subset(train, train_idx), batch_size=512, shuffle=True, num_workers=0),
        DataLoader(Subset(test, test_idx), batch_size=512, shuffle=False, num_workers=0),
        DataLoader(Subset(train, calib_idx), batch_size=512, shuffle=False, num_workers=0),
    )


def _train(model, loader):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(7):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


def _apply_masks(model, masks):
    original = {"fc1": model.fc1.weight.data.clone(), "fc2": model.fc2.weight.data.clone()}
    model.fc1.weight.data.mul_(masks["fc1"])
    model.fc2.weight.data.mul_(masks["fc2"])
    return original


def _restore(model, original):
    model.fc1.weight.data.copy_(original["fc1"])
    model.fc2.weight.data.copy_(original["fc2"])


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


def _scores(model, calib_loader, alpha: float):
    model.eval()
    flats = []
    hiddens = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(DEVICE)
            _, flat, h = model(x, return_hidden=True)
            flats.append(flat.cpu())
            hiddens.append(h.cpu())
    flat_all = torch.cat(flats, dim=0).to(DEVICE)
    h_all = torch.cat(hiddens, dim=0).to(DEVICE)
    input_signal = flat_all.std(dim=0) + 1e-6
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    hidden = hidden_strength * output_strength
    return {
        "fc1": model.fc1.weight.detach().abs() * torch.pow(input_signal[None, :] * hidden[:, None], alpha),
        "fc2": model.fc2.weight.detach().abs() * torch.pow(hidden[None, :], alpha),
    }


def _gradient_scores(model, calib_loader):
    model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in calib_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(x), y)
        loss.backward()
        break
    return {
        "fc1": (model.fc1.weight.grad * model.fc1.weight).abs().detach().clone(),
        "fc2": (model.fc2.weight.grad * model.fc2.weight).abs().detach().clone(),
    }


def _mask(scores, sparsity):
    flat = torch.cat([scores["fc1"].flatten(), scores["fc2"].flatten()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {"fc1": (scores["fc1"] >= threshold).float(), "fc2": (scores["fc2"] >= threshold).float()}


def run():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = _loaders(seed)
        model = MLP().to(DEVICE)
        _train(model, train_loader)
        dense_accuracy = _evaluate(model, test_loader)
        gradient = _gradient_scores(model, calib_loader)
        for alpha in ALPHAS:
            scores = _scores(model, calib_loader, alpha)
            for sparsity in SPARSITIES:
                masks = _mask(scores, sparsity)
                accuracy = _evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "method": f"path_alpha_{alpha}", "alpha": alpha, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})
        for sparsity in SPARSITIES:
            masks = _mask(gradient, sparsity)
            accuracy = _evaluate(model, test_loader, masks)
            rows.append({"seed": seed, "method": "gradient_saliency", "alpha": None, "sparsity": sparsity, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})
    summary = {}
    methods = sorted(set(r["method"] for r in rows))
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in methods:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            if selected:
                summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}
    alpha_scores = {str(alpha): float(np.mean([r["accuracy"] for r in rows if r["alpha"] == alpha])) for alpha in ALPHAS}
    result = {"experiment": "04_torch_cifar10_mlp_path_flow", "device": str(DEVICE), "setup": "Exploratory CIFAR-10 MLP test. Train on 20k examples, evaluate on 5k, compare magnitude/path blends alpha=0..1 and gradient saliency at 90/95/98% sparsity.", "alpha_scores": alpha_scores, "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_cifar10_mlp_path_flow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_CIFAR10_MLP.md"
    lines = ["# Torch CIFAR-10 MLP Path-Flow Exploration", "", result["setup"], "", f"Device: `{DEVICE}`", "", "| Alpha | Mean accuracy over 90/95/98% |", "|---:|---:|"]
    for alpha in ALPHAS:
        lines.append(f"| `{alpha:.2f}` | `{alpha_scores[str(alpha)]:.4f}` |")
    lines.extend(["", "## Per-sparsity summary", "", "| Sparsity | Method | Accuracy | Retention |", "|---:|---|---:|---:|"])
    for sparsity in SPARSITIES:
        for method, item in summary[str(sparsity)].items():
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"device": result["device"], "alpha_scores": result["alpha_scores"], "summary": result["summary"]}, indent=2))
