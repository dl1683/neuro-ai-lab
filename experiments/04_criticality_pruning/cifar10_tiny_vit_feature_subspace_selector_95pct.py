from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cifar10_tiny_vit_circuit_viability_98pct as tinyvit
import cifar10_tiny_vit_feature_subspace_diagnostic_95pct as diag


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SEEDS = [292, 293]
SPARSITY = 0.95
CANDIDATES = [
    "magnitude",
    "global_synflow",
    "minimal_liveness_repair",
    "attn_mlp_readout_repair",
    "all_route_liveness_floor",
]
METHODS = CANDIDATES + ["feature_subspace_policy"]


def summarize(rows):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected])),
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
    return summary


def write_report(result):
    out = RESULTS / "cifar10_tiny_vit_feature_subspace_selector_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_SUBSPACE_SELECTOR_95PCT.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Subspace Selector at 95%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | Dead outputs | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
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
            f"`{item['centered_cls_cosine_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` | "
            f"`{item['mlp_down_dead_outputs_mean']:.1f}` | `{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Selector decisions", ""])
    for item in result["decisions"]:
        ranked = ", ".join(f"{rank['method']}={rank['centered_cls_cosine_mean']:.4f}" for rank in item["ranked_candidates"])
        lines.append(f"- seed `{item['seed']}`: selected `{item['selected_method']}`; centered CLS scores {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a prospective selector test. The selected method is the candidate with the highest pre-finetune centered CLS/residual-stream feature alignment to the dense model. Post-finetune accuracy is measured only after selection.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    decisions = []
    dense_scores = []
    for seed in SEEDS:
        print(f"seed {seed}: train dense TinyViT feature-subspace selector", flush=True)
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
        alignments = {
            method: diag.feature_alignment(model, dense_state, masks, test_loader)
            for method, masks in masks_by_method.items()
        }
        ranked = sorted(
            [
                {"method": method, "centered_cls_cosine_mean": alignment["centered_cls_cosine_mean"]}
                for method, alignment in alignments.items()
            ],
            key=lambda item: item["centered_cls_cosine_mean"],
            reverse=True,
        )
        selected_method = ranked[0]["method"]
        decisions.append({"seed": seed, "selected_method": selected_method, "ranked_candidates": ranked})
        print(f"seed {seed}: selected={selected_method}", flush=True)
        evaluated = {}
        for label, masks in masks_by_method.items():
            before, after = tinyvit.eval_method(model, dense_state, train_loader, test_loader, masks)
            quality = tinyvit.route_quality(masks)
            row = {
                "seed": seed,
                "method": label,
                "dense_accuracy": dense_acc,
                "before_accuracy": before,
                "after_accuracy": after,
                "feature_alignment": alignments[label],
                "route_quality": quality,
            }
            rows.append(row)
            evaluated[label] = row
            print(
                f"seed {seed} {label}: after={after:.4f} centered_cls={alignments[label]['centered_cls_cosine_mean']:.4f} "
                f"dead={quality['total_dead_outputs']} mlp_dead={quality['mlp_down_dead_outputs']} "
                f"attn_dead={quality['attn_out_dead_outputs']}",
                flush=True,
            )
        policy_row = dict(evaluated[selected_method])
        policy_row["method"] = "feature_subspace_policy"
        policy_row["policy_source_method"] = selected_method
        rows.append(policy_row)
    result = {
        "experiment": "04_cifar10_tiny_vit_feature_subspace_selector_95pct",
        "setup": "Fresh TinyViT CIFAR-10 subset 95% sparsity prospective selector. The policy selects the mask with highest pre-finetune centered CLS/residual-stream feature alignment.",
        "device": tinyvit.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "sparsity": SPARSITY,
        "dense_accuracy_mean": float(np.mean(dense_scores)),
        "summary": summarize(rows),
        "decisions": decisions,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
