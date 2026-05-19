from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cifar10_tiny_vit_circuit_viability_98pct as tinyvit


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SEEDS = [290, 291]
SPARSITY = 0.95
METHODS = [
    "magnitude",
    "global_synflow",
    "minimal_liveness_repair",
    "attn_mlp_readout_repair",
    "all_route_liveness_floor",
]


def cls_features(model, x):
    x = model.patch_embed(x).flatten(2).transpose(1, 2)
    cls = model.cls.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1) + model.pos
    for block in model.blocks:
        x = block(x)
    return model.norm(x[:, 0])


@torch.no_grad()
def feature_alignment(model, dense_state, masks, loader, batches: int = 8):
    model.load_state_dict(dense_state)
    model.eval()
    dense_features = []
    masked_features = []
    for idx, (x, _) in enumerate(loader):
        if idx >= batches:
            break
        x = x.to(tinyvit.DEVICE)
        dense_features.append(cls_features(model, x).detach().cpu())
    model.load_state_dict(dense_state)
    tinyvit.apply_mask(model, masks)
    model.eval()
    for idx, (x, _) in enumerate(loader):
        if idx >= batches:
            break
        x = x.to(tinyvit.DEVICE)
        masked_features.append(cls_features(model, x).detach().cpu())
    model.load_state_dict(dense_state)
    dense = torch.cat(dense_features)
    masked = torch.cat(masked_features)
    cosine = F.cosine_similarity(dense, masked, dim=1)
    centered_dense = dense - dense.mean(0, keepdim=True)
    centered_masked = masked - masked.mean(0, keepdim=True)
    centered_cosine = F.cosine_similarity(centered_dense, centered_masked, dim=1)
    return {
        "cls_cosine_mean": float(cosine.mean().item()),
        "cls_cosine_std": float(cosine.std(unbiased=False).item()),
        "centered_cls_cosine_mean": float(centered_cosine.mean().item()),
        "feature_samples": int(dense.shape[0]),
    }


def summarize(rows):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected])),
            "cls_cosine_mean": float(np.mean([row["feature_alignment"]["cls_cosine_mean"] for row in selected])),
            "centered_cls_cosine_mean": float(np.mean([row["feature_alignment"]["centered_cls_cosine_mean"] for row in selected])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
            "mlp_down_dead_outputs_mean": float(np.mean([row["route_quality"]["mlp_down_dead_outputs"] for row in selected])),
            "attn_out_dead_outputs_mean": float(np.mean([row["route_quality"]["attn_out_dead_outputs"] for row in selected])),
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
    xs = np.array([row["feature_alignment"]["centered_cls_cosine_mean"] for row in rows], dtype=float)
    ys = np.array([row["after_accuracy"] for row in rows], dtype=float)
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 and float(np.std(xs)) > 0 and float(np.std(ys)) > 0 else 0.0
    return summary, {"centered_cls_cosine_vs_after_corr": corr}


def write_report(result):
    out = RESULTS / "cifar10_tiny_vit_feature_subspace_diagnostic_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_SUBSPACE_DIAGNOSTIC_95PCT.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Subspace Diagnostic at 95%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Centered CLS cosine vs after-FT correlation: `{result['diagnostics']['centered_cls_cosine_vs_after_corr']:.4f}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | CLS cosine | Centered CLS cosine | Dead outputs | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"`{item['cls_cosine_mean']:.4f}` | `{item['centered_cls_cosine_mean']:.4f}` | "
            f"`{item['dead_outputs_mean']:.1f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This diagnostic tests the transformer-specific hypothesis that sparse recovery depends on preserving the dense CLS/residual-stream representation, not only keeping rows alive. The feature score is measured before fine-tuning, so it is a candidate predictor rather than a post-hoc accuracy statistic.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    dense_scores = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense TinyViT feature-subspace diagnostic", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = tinyvit.loaders(seed)
        model = tinyvit.TinyViT().to(tinyvit.DEVICE)
        tinyvit.train(model, train_loader, tinyvit.DENSE_EPOCHS, lr=3e-4)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_acc = tinyvit.evaluate(model, test_loader)
        dense_scores.append(dense_acc)
        print(f"seed {seed}: dense_accuracy={dense_acc:.4f}", flush=True)
        mag = tinyvit.magnitude_scores(model)
        syn = tinyvit.synflow_scores(model)
        masks_by_method = {
            "magnitude": tinyvit.global_mask(mag, SPARSITY),
            "global_synflow": tinyvit.global_mask(syn, SPARSITY),
            "minimal_liveness_repair": tinyvit.minimal_liveness_repair(mag, SPARSITY),
            "attn_mlp_readout_repair": tinyvit.attn_mlp_readout_repair(mag, SPARSITY),
            "all_route_liveness_floor": tinyvit.all_route_liveness_floor(mag, SPARSITY),
        }
        for label, masks in masks_by_method.items():
            alignment = feature_alignment(model, dense_state, masks, test_loader)
            before, after = tinyvit.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = tinyvit.route_quality(masks)
            rows.append(
                {
                    "seed": seed,
                    "method": label,
                    "dense_accuracy": dense_acc,
                    "before_accuracy": before,
                    "after_accuracy": after,
                    "feature_alignment": alignment,
                    "route_quality": quality,
                }
            )
            print(
                f"seed {seed} {label}: after={after:.4f} centered_cls={alignment['centered_cls_cosine_mean']:.4f} "
                f"dead={quality['total_dead_outputs']} mlp_dead={quality['mlp_down_dead_outputs']} "
                f"attn_dead={quality['attn_out_dead_outputs']}",
                flush=True,
            )
    summary, diagnostics = summarize(rows)
    result = {
        "experiment": "04_cifar10_tiny_vit_feature_subspace_diagnostic_95pct",
        "setup": "TinyViT CIFAR-10 subset 95% sparsity diagnostic comparing pre-finetune CLS/residual-stream feature preservation against route-liveness metrics and masked recovery.",
        "device": tinyvit.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "dense_accuracy_mean": float(np.mean(dense_scores)),
        "summary": summary,
        "diagnostics": diagnostics,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "diagnostics": result["diagnostics"]}, indent=2))
