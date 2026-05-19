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
METHODS = ["magnitude", "gradient_saliency", "path_flow"]
SEEDS = [31, 32]


class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x, return_hidden: bool = False):
        x = x.view(x.shape[0], -1)
        h = self.relu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, x, h
        return logits


def _loaders(seed: int):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    data_dir = ROOT / "data" / "fashion_mnist"
    train = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train))[:16000]
    test_idx = rng.permutation(len(test))[:3000]
    calib_idx = train_idx[:2048]
    train_loader = DataLoader(Subset(train, train_idx), batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test, test_idx), batch_size=512, shuffle=False, num_workers=0)
    calib_loader = DataLoader(Subset(train, calib_idx), batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader, calib_loader


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
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        if original is not None:
            _restore(model, original)
    return correct / total


def _apply_masks(model, masks):
    original = {"fc1": model.fc1.weight.data.clone(), "fc2": model.fc2.weight.data.clone()}
    model.fc1.weight.data.mul_(masks["fc1"])
    model.fc2.weight.data.mul_(masks["fc2"])
    return original


def _restore(model, original):
    model.fc1.weight.data.copy_(original["fc1"])
    model.fc2.weight.data.copy_(original["fc2"])


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


def _path_flow_scores(model, calib_loader):
    model.eval()
    inputs = []
    hidden = []
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(DEVICE)
            _, flat, h = model(x, return_hidden=True)
            inputs.append(flat.abs().cpu())
            hidden.append(h.cpu())
    input_activity = torch.cat(inputs, dim=0).mean(dim=0).to(DEVICE) + 1e-6
    h_all = torch.cat(hidden, dim=0).to(DEVICE)
    hidden_strength = h_all.abs().mean(dim=0) + 1e-6
    hidden_fire = (h_all > 0).float().mean(dim=0) + 1e-6
    hidden_balance = torch.exp(-torch.abs(hidden_fire - 0.35) / 0.22)
    output_strength = model.fc2.weight.detach().abs().mean(dim=0) + 1e-6
    path_importance = hidden_strength * hidden_balance * output_strength
    return {
        "fc1": model.fc1.weight.detach().abs() * path_importance[:, None] * input_activity[None, :],
        "fc2": model.fc2.weight.detach().abs() * path_importance[None, :],
    }


def _magnitude_scores(model):
    return {"fc1": model.fc1.weight.detach().abs().clone(), "fc2": model.fc2.weight.detach().abs().clone()}


def _mask(scores, sparsity: float):
    flat = torch.cat([scores["fc1"].flatten(), scores["fc2"].flatten()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {"fc1": (scores["fc1"] >= threshold).float(), "fc2": (scores["fc2"] >= threshold).float()}


def run() -> dict:
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader, calib_loader = _loaders(seed)
        model = MLP().to(DEVICE)
        _train(model, train_loader)
        dense_accuracy = _evaluate(model, test_loader)
        score_by_method = {
            "magnitude": _magnitude_scores(model),
            "gradient_saliency": _gradient_scores(model, calib_loader),
            "path_flow": _path_flow_scores(model, calib_loader),
        }
        for sparsity in SPARSITIES:
            for method in METHODS:
                masks = _mask(score_by_method[method], sparsity)
                accuracy = _evaluate(model, test_loader, masks)
                rows.append({"seed": seed, "sparsity": sparsity, "method": method, "dense_accuracy": dense_accuracy, "accuracy": accuracy, "retention": accuracy / dense_accuracy})

    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["sparsity"] == sparsity and r["method"] == method]
            summary[str(sparsity)][method] = {"accuracy_mean": float(np.mean([r["accuracy"] for r in selected])), "accuracy_std": float(np.std([r["accuracy"] for r in selected])), "retention_mean": float(np.mean([r["retention"] for r in selected]))}

    result = {"experiment": "04_torch_fashion_mnist_path_flow", "device": str(DEVICE), "setup": "Torch MLP trained on 16k Fashion-MNIST examples, evaluated on 3k examples. One-shot pruning of fc1/fc2 at high sparsity using magnitude, gradient saliency, and label-free path-flow.", "summary": summary, "rows": rows}
    out = ROOT / "results" / "04_criticality_pruning" / "torch_fashion_mnist_path_flow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = ROOT / "experiments" / "04_criticality_pruning" / "TORCH_FASHION_MNIST.md"
    lines = ["# Torch Fashion-MNIST Path-Flow Check", "", result["setup"], "", f"Device: `{DEVICE}`", "", "| Sparsity | Method | Mean accuracy | Std | Mean retention |", "|---:|---|---:|---:|---:|"]
    for sparsity in SPARSITIES:
        for method in METHODS:
            item = summary[str(sparsity)][method]
            lines.append(f"| `{sparsity:.2f}` | `{method}` | `{item['accuracy_mean']:.4f}` | `{item['accuracy_std']:.4f}` | `{item['retention_mean']:.4f}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"device": result["device"], "summary": result["summary"]}, indent=2))
