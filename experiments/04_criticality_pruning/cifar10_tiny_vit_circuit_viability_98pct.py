from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [286, 287]
SPARSITY = 0.98
TRAIN_N = 20000
TEST_N = 5000
BATCH = 128
DENSE_EPOCHS = 8
FT_EPOCHS = 3
METHODS = [
    "magnitude",
    "global_synflow",
    "minimal_liveness_repair",
    "selective_mlp_readout_repair",
    "attn_mlp_readout_repair",
    "all_route_liveness_floor",
    "mlp_readout_reserve",
]


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, patch: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        tokens = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


def loaders(seed: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=test_tf)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train))[:TRAIN_N]
    test_idx = np.arange(TEST_N)
    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(Subset(train, train_idx), batch_size=BATCH, shuffle=True, num_workers=2, generator=gen),
        DataLoader(Subset(test, test_idx), batch_size=256, shuffle=False, num_workers=2),
    )


def params(model):
    return {name: p for name, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad}


def train(model, loader, epochs: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
        print(f"  epoch={epoch + 1}/{epochs}", flush=True)


@torch.no_grad()
def evaluate(model, loader, masks=None):
    if masks is not None:
        apply_mask(model, masks)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / total


def apply_mask(model, masks):
    with torch.no_grad():
        for name, p in params(model).items():
            p.mul_(masks[name])


def magnitude_scores(model):
    return {name: p.detach().abs().clone() for name, p in params(model).items()}


def synflow_scores(model):
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    model.zero_grad(set_to_none=True)
    torch.sum(model(torch.ones(1, 3, 32, 32, device=DEVICE))).backward()
    scores = {name: (p.grad * p).abs().detach().clone() for name, p in params(model).items()}
    for name, p in model.named_parameters():
        p.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def global_mask(scores: dict[str, torch.Tensor], sparsity: float):
    total = sum(score.numel() for score in scores.values())
    keep = max(1, int(round((1.0 - sparsity) * total)))
    flat = torch.cat([score.flatten() for score in scores.values()])
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {name: (score >= threshold).to(score.dtype) for name, score in scores.items()}


def budgeted_mask_with_protection(scores, sparsity, protected):
    total = sum(score.numel() for score in scores.values())
    keep = max(1, int(round((1.0 - sparsity) * total)))
    masks = {name: protected.get(name, torch.zeros_like(score)).clone().to(score.dtype) for name, score in scores.items()}
    protected_count = int(sum(mask.sum().item() for mask in masks.values()))
    remaining = max(0, keep - protected_count)
    if remaining > 0:
        masked_scores = []
        refs = []
        for name, score in scores.items():
            available = masks[name].flatten() == 0
            idx = torch.nonzero(available, as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = score.flatten()[idx]
                masked_scores.append(vals)
                refs.append((name, idx, vals))
        if masked_scores:
            all_vals = torch.cat(masked_scores)
            kth = torch.topk(all_vals, min(remaining, all_vals.numel()), largest=True).values.min()
            left = remaining
            for name, idx, vals in refs:
                chosen = idx[vals >= kth]
                if chosen.numel() > left:
                    chosen = chosen[:left]
                flat = masks[name].flatten()
                flat[chosen] = 1
                masks[name] = flat.reshape_as(masks[name])
                left -= int(chosen.numel())
                if left <= 0:
                    break
    selected = int(sum(mask.sum().item() for mask in masks.values()))
    if selected <= keep:
        return masks
    removable = []
    refs = []
    for name, score in scores.items():
        prot = protected.get(name, torch.zeros_like(score)).flatten() > 0
        kept = masks[name].flatten() > 0
        idx = torch.nonzero(kept & ~prot, as_tuple=False).flatten()
        if idx.numel() > 0:
            vals = score.flatten()[idx]
            removable.append(vals)
            refs.append((name, idx, vals))
    remove_n = selected - keep
    cutoff = torch.topk(torch.cat(removable), remove_n, largest=False).values.max()
    left = remove_n
    for name, idx, vals in refs:
        chosen = idx[vals <= cutoff]
        if chosen.numel() > left:
            chosen = chosen[:left]
        flat = masks[name].flatten()
        flat[chosen] = 0
        masks[name] = flat.reshape_as(masks[name])
        left -= int(chosen.numel())
        if left <= 0:
            break
    return masks


def minimal_liveness_repair(scores, sparsity):
    base = global_mask(scores, sparsity)
    protected = {name: torch.zeros_like(score) for name, score in scores.items()}
    for name, score in scores.items():
        if score.ndim < 2:
            continue
        rows = base[name].reshape(score.shape[0], -1)
        flat_score = score.reshape(score.shape[0], -1)
        flat_prot = protected[name].reshape(score.shape[0], -1)
        for row in range(rows.shape[0]):
            if int(rows[row].sum().item()) == 0:
                idx = torch.argmax(flat_score[row])
                rows[row, idx] = 1
                flat_prot[row, idx] = 1
        base[name] = rows.reshape_as(base[name])
        protected[name] = flat_prot.reshape_as(protected[name])
    return budgeted_mask_with_protection(scores, sparsity, protected)


def selective_mlp_readout_repair(scores, sparsity):
    base = global_mask(scores, sparsity)
    protected = {name: torch.zeros_like(score) for name, score in scores.items()}
    for name, score in scores.items():
        if score.ndim < 2:
            continue
        if not (name.endswith("mlp.net.2.weight") or name == "head.weight"):
            continue
        rows = base[name].reshape(score.shape[0], -1)
        flat_score = score.reshape(score.shape[0], -1)
        flat_prot = protected[name].reshape(score.shape[0], -1)
        for row in range(rows.shape[0]):
            if int(rows[row].sum().item()) == 0:
                idx = torch.argmax(flat_score[row])
                rows[row, idx] = 1
                flat_prot[row, idx] = 1
        base[name] = rows.reshape_as(base[name])
        protected[name] = flat_prot.reshape_as(protected[name])
    return budgeted_mask_with_protection(scores, sparsity, protected)


def attn_mlp_readout_repair(scores, sparsity):
    base = global_mask(scores, sparsity)
    protected = {name: torch.zeros_like(score) for name, score in scores.items()}
    for name, score in scores.items():
        if score.ndim < 2:
            continue
        is_route = name.endswith("attn.out_proj.weight") or name.endswith("mlp.net.2.weight") or name == "head.weight"
        if not is_route:
            continue
        rows = base[name].reshape(score.shape[0], -1)
        flat_score = score.reshape(score.shape[0], -1)
        flat_prot = protected[name].reshape(score.shape[0], -1)
        for row in range(rows.shape[0]):
            if int(rows[row].sum().item()) == 0:
                idx = torch.argmax(flat_score[row])
                rows[row, idx] = 1
                flat_prot[row, idx] = 1
        base[name] = rows.reshape_as(base[name])
        protected[name] = flat_prot.reshape_as(protected[name])
    return budgeted_mask_with_protection(scores, sparsity, protected)


def all_route_liveness_floor(scores, sparsity):
    protected = {name: torch.zeros_like(score) for name, score in scores.items()}
    for name, score in scores.items():
        if score.ndim < 2:
            continue
        rows = score.reshape(score.shape[0], -1)
        flat_prot = protected[name].reshape(score.shape[0], -1)
        for row in range(rows.shape[0]):
            idx = torch.argmax(rows[row])
            flat_prot[row, idx] = 1
        protected[name] = flat_prot.reshape_as(protected[name])
    return budgeted_mask_with_protection(scores, sparsity, protected)


def mlp_readout_reserve_mask(scores, sparsity):
    protected = {name: torch.zeros_like(score) for name, score in scores.items()}
    for name, score in scores.items():
        if score.ndim < 2:
            continue
        is_vulnerable = name.endswith("mlp.net.2.weight") or name == "head.weight"
        if not is_vulnerable:
            continue
        rows = score.reshape(score.shape[0], -1)
        flat_prot = protected[name].reshape(score.shape[0], -1)
        row_keep = max(1, int(round(rows.shape[1] * (1.0 - sparsity) * 3.0)))
        for row in range(rows.shape[0]):
            idx = torch.topk(rows[row], min(row_keep, rows.shape[1]), largest=True).indices
            flat_prot[row, idx] = 1
        protected[name] = flat_prot.reshape_as(protected[name])
    return budgeted_mask_with_protection(scores, sparsity, protected)


def route_quality(masks):
    linear = {}
    for name, mask in masks.items():
        if mask.ndim < 2:
            continue
        rows = mask.reshape(mask.shape[0], -1).sum(1).float()
        linear[name] = {
            "dead": int((rows == 0).sum().item()),
            "min": float(rows.min().item()),
            "mean": float(rows.mean().item()),
        }
    mlp_down = [v for k, v in linear.items() if k.endswith("mlp.net.2.weight")]
    attn_out = [v for k, v in linear.items() if k.endswith("attn.out_proj.weight")]
    head = linear.get("head.weight", {"dead": 0, "min": 0.0, "mean": 0.0})
    return {
        "total_dead_outputs": int(sum(v["dead"] for v in linear.values())),
        "mlp_down_dead_outputs": int(sum(v["dead"] for v in mlp_down)),
        "mlp_down_min": float(min([v["min"] for v in mlp_down] or [0.0])),
        "mlp_down_mean": float(np.mean([v["mean"] for v in mlp_down] or [0.0])),
        "attn_out_dead_outputs": int(sum(v["dead"] for v in attn_out)),
        "attn_out_min": float(min([v["min"] for v in attn_out] or [0.0])),
        "attn_out_mean": float(np.mean([v["mean"] for v in attn_out] or [0.0])),
        "head_dead_outputs": int(head["dead"]),
        "head_min": float(head["min"]),
        "head_mean": float(head["mean"]),
    }


def eval_method(model, dense_state, train_loader, test_loader, masks):
    before = evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    apply_mask(model, masks)
    train(model, train_loader, FT_EPOCHS, lr=5e-5)
    apply_mask(model, masks)
    after = evaluate(model, test_loader)
    model.load_state_dict(dense_state)
    return before, after


def summarize(rows):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
            "mlp_down_dead_outputs_mean": float(np.mean([row["route_quality"]["mlp_down_dead_outputs"] for row in selected])),
            "attn_out_dead_outputs_mean": float(np.mean([row["route_quality"]["attn_out_dead_outputs"] for row in selected])),
            "mlp_down_min_mean": float(np.mean([row["route_quality"]["mlp_down_min"] for row in selected])),
            "attn_out_min_mean": float(np.mean([row["route_quality"]["attn_out_min"] for row in selected])),
            "head_min_mean": float(np.mean([row["route_quality"]["head_min"] for row in selected])),
        }
    for method in METHODS:
        if method == "magnitude":
            continue
        deltas = []
        for seed in SEEDS:
            mag = next(row for row in rows if row["seed"] == seed and row["method"] == "magnitude")
            alt = next(row for row in rows if row["seed"] == seed and row["method"] == method)
            deltas.append(alt["after_accuracy"] - mag["after_accuracy"])
        summary[method]["after_delta_mean"] = float(np.mean(deltas))
        summary[method]["after_wins"] = int(sum(delta > 0 for delta in deltas))
    return summary


def write_report(result):
    out = RESULTS / "cifar10_tiny_vit_circuit_viability_98pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_CIRCUIT_VIABILITY_98PCT.md"
    lines = [
        "# CIFAR-10 TinyViT Circuit Viability at 98%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | Attn-out dead | MLP-down min | Attn-out min | Head min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(SEEDS)}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['dead_outputs_mean']:.1f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` | `{item['mlp_down_min_mean']:.1f}` | "
            f"`{item['attn_out_min_mean']:.1f}` | `{item['head_min_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the first transformer-style analogue in the repo. The vulnerable routes are TinyViT MLP down-projections and the classifier readout rather than CNN dense bridges. The test is intentionally small, but it is real: dense TinyViT models are trained on CIFAR-10, pruned at `98%`, then fine-tuned under fixed masks.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    dense = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense TinyViT", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = loaders(seed)
        model = TinyViT().to(DEVICE)
        train(model, train_loader, DENSE_EPOCHS, lr=3e-4)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_acc = evaluate(model, test_loader)
        dense.append(dense_acc)
        print(f"seed {seed}: dense_accuracy={dense_acc:.4f}", flush=True)
        mag = magnitude_scores(model)
        syn = synflow_scores(model)
        masks_by_method = {
            "magnitude": global_mask(mag, SPARSITY),
            "global_synflow": global_mask(syn, SPARSITY),
            "minimal_liveness_repair": minimal_liveness_repair(mag, SPARSITY),
            "selective_mlp_readout_repair": selective_mlp_readout_repair(mag, SPARSITY),
            "attn_mlp_readout_repair": attn_mlp_readout_repair(mag, SPARSITY),
            "all_route_liveness_floor": all_route_liveness_floor(mag, SPARSITY),
            "mlp_readout_reserve": mlp_readout_reserve_mask(mag, SPARSITY),
        }
        for label, masks in masks_by_method.items():
            before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = route_quality(masks)
            rows.append(
                {
                    "seed": seed,
                    "method": label,
                    "dense_accuracy": dense_acc,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "route_quality": quality,
                }
            )
            print(
                f"seed {seed} {label}: after={after:.4f} dead={quality['total_dead_outputs']} "
                f"mlp_dead={quality['mlp_down_dead_outputs']} attn_dead={quality['attn_out_dead_outputs']} "
                f"mlp_min={quality['mlp_down_min']:.1f} attn_min={quality['attn_out_min']:.1f} "
                f"head_min={quality['head_min']:.1f}",
                flush=True,
            )
    result = {
        "experiment": "04_cifar10_tiny_vit_circuit_viability_98pct",
        "setup": "TinyViT CIFAR-10 subset transformer-analogue severe-pruning test. MLP down-projection and classifier readout rows are treated as circuit bottlenecks.",
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "train_subset": TRAIN_N,
        "test_subset": TEST_N,
        "sparsity": SPARSITY,
        "dense_epochs": DENSE_EPOCHS,
        "finetune_epochs": FT_EPOCHS,
        "dense_accuracy_mean": float(np.mean(dense)),
        "summary": summarize(rows),
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dense_accuracy_mean": result["dense_accuracy_mean"], "summary": result["summary"]}, indent=2))
