from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

import cifar10_resnet20_capacity_99pct as r20
import cifar10_tiny_resnet_capacity_transfer as base
from shared.circuit_viability_selector import choose_ecology_aware_method, split_dict
from shared.residual_route_capacity import route_split_capacity_mask


URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATA_ROOT = ROOT / "data"
ZIP_PATH = DATA_ROOT / "tiny-imagenet-200.zip"
TINY_ROOT = DATA_ROOT / "tiny-imagenet-200"
SEED = 271
TRAIN_N = 20000
VAL_N = 5000
BATCH = 256
SPARSITY = 0.99
RESERVE = 0.60
DENSE_EPOCHS = 12
FT_EPOCHS = 4
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_policy"]


class TinyImageNet(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.classes = sorted([p.name for p in (root / "train").iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        if split == "train":
            for cls in self.classes:
                for path in sorted((root / "train" / cls / "images").glob("*.JPEG")):
                    self.samples.append((path, self.class_to_idx[cls]))
        elif split == "val":
            ann = {}
            for line in (root / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                ann[parts[0]] = self.class_to_idx[parts[1]]
            for path in sorted((root / "val" / "images").glob("*.JPEG")):
                self.samples.append((path, ann[path.name]))
        else:
            raise ValueError(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def ensure_data():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if TINY_ROOT.exists():
        return
    if not ZIP_PATH.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, ZIP_PATH)
    print("extracting TinyImageNet", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)


def loaders(seed: int):
    ensure_data()
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = TinyImageNet(TINY_ROOT, "train", train_tf)
    val_ds = TinyImageNet(TINY_ROOT, "val", eval_tf)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_ds))[:TRAIN_N]
    val_idx = np.arange(min(VAL_N, len(val_ds)))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH, shuffle=True, num_workers=2, generator=generator)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=512, shuffle=False, num_workers=2)
    return train_loader, val_loader


def make_model():
    model = r20.CifarResNet20()
    model.fc = nn.Linear(model.fc.in_features, 200)
    return model.to(base.DEVICE)


def train_model(model, loader):
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DENSE_EPOCHS)
    for epoch in range(DENSE_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
        if epoch in {3, 7, 11}:
            print(f"  dense_epoch={epoch + 1}/{DENSE_EPOCHS}", flush=True)


def masked_finetune(model, loader, masks):
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    base.apply_mask(model, masks)
    for _ in range(FT_EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(base.DEVICE)
            y = y.to(base.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            base.apply_mask(model, masks)
        scheduler.step()


def eval_method(model, dense_state, train_loader, val_loader, masks):
    before = base.evaluate(model, val_loader, masks)
    model.load_state_dict(dense_state)
    masked_finetune(model, train_loader, masks)
    after = base.evaluate(model, val_loader)
    model.load_state_dict(dense_state)
    return before, after


def synflow_scores_64(model):
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    model.zero_grad(set_to_none=True)
    ones = torch.ones(1, 3, 64, 64, device=base.DEVICE)
    torch.sum(model(ones)).backward()
    scores = {k: (p.grad * p).abs().detach().clone() for k, p in base.params(model).items()}
    for name, p in model.named_parameters():
        p.data.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def summarize(rows):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "projection_min_mean": float(np.mean([row["route_quality"]["projection_min"] for row in selected])),
            "fc_score_mean": float(np.mean([row["route_quality"]["fc_score"] for row in selected])),
            "main_path_min_mean": float(np.mean([row["route_quality"]["main_path_min"] for row in selected])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
        }
    mag = summary["magnitude"]["after_mean"]
    for method in METHODS:
        if method != "magnitude":
            summary[method]["after_delta_mean"] = summary[method]["after_mean"] - mag
            summary[method]["after_wins"] = int(summary[method]["after_delta_mean"] > 0)
    return summary


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet20_ecology_selector_99pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET20_ECOLOGY_SELECTOR_99PCT.md"
    lines = [
        "# TinyImageNet-200 ResNet-20 Ecology Selector at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seed: `{result['seed']}`",
        f"Train subset: `{TRAIN_N}`; validation subset: `{VAL_N}`",
        f"Dense epochs: `{DENSE_EPOCHS}`; masked fine-tune epochs: `{FT_EPOCHS}`",
        "",
        "| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        delta = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected method: `{result['decision']['selected_method']}`",
            f"Readout ratio: `{result['decision']['plain_readout_ratio']:.4f}`",
            f"Selected split: `{result['decision']['selected_split']}`",
            "",
            "## Interpretation",
            "",
            "This is a first external TinyImageNet-200 proxy, not a full publication benchmark. It tests whether the same fixed readout-ratio selector behaves coherently on a 200-class natural-image task outside CIFAR.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    print(f"tinyimagenet seed {SEED}: train dense", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = loaders(SEED)
    model = make_model()
    train_model(model, train_loader)
    dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dense_accuracy = base.evaluate(model, val_loader)
    print(f"tinyimagenet seed {SEED}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
    mag = base.magnitude_scores(model)
    syn = synflow_scores_64(model)
    decision = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, r20.route_quality)
    split = decision["best_split"]["split"]
    method_masks = {
        "magnitude": base.global_mask(mag, SPARSITY),
        "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
        "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split),
    }
    print(
        f"tinyimagenet seed {SEED}: selected={decision['selected_method']} "
        f"readout_ratio={decision['plain_readout_ratio']:.4f} best_split={split_dict(split)}",
        flush=True,
    )
    rows = []
    evaluated = {}
    for label, masks in method_masks.items():
        before, after = eval_method(model, dense_state, train_loader, val_loader, masks)
        quality = r20.route_quality(masks)
        row = {
            "seed": SEED,
            "method": label,
            "dense_accuracy": dense_accuracy,
            "before_accuracy": before,
            "after_accuracy": after,
            "route_quality": quality,
        }
        rows.append(row)
        evaluated[label] = row
        print(
            f"tinyimagenet {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
            f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
            flush=True,
        )
    policy_source = decision["selected_method"]
    policy_row = dict(evaluated[policy_source])
    policy_row["method"] = "ecology_policy"
    policy_row["policy_source_method"] = policy_source
    rows.append(policy_row)
    result = {
        "experiment": "04_tinyimagenet_resnet20_ecology_selector_99pct",
        "setup": "TinyImageNet-200 external-proxy subset stress test using a ResNet-20-style 200-class model, 99% sparsity, and the fixed ecology-aware selector.",
        "device": base.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": SEED,
        "train_subset": TRAIN_N,
        "val_subset": VAL_N,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "dense_epochs": DENSE_EPOCHS,
        "finetune_epochs": FT_EPOCHS,
        "summary": summarize(rows),
        "decision": {
            "selected_method": decision["selected_method"],
            "selected_split": decision["selected_split"],
            "plain_readout_ratio": decision["plain_readout_ratio"],
            "readout_ratio_threshold": decision["readout_ratio_threshold"],
            "best_split": split_dict(split),
            "plain_quality": decision["plain_quality"],
            "magnitude_quality": decision["magnitude_quality"],
            "best_split_quality": decision["best_split"]["quality"],
        },
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2))
