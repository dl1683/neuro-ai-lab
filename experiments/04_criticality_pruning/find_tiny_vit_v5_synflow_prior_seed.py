from __future__ import annotations

import json

import numpy as np
import torch

import cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong as v4
import cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong as v5


base = v4.base
SEEDS = [309, 310, 311, 312, 313, 314]


def scan_seed(seed: int) -> dict:
    print(f"seed {seed}: train dense TinyViT V5 branch scanner", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader = base.tinyvit.loaders(seed)
    model = base.tinyvit.TinyViT().to(base.tinyvit.DEVICE)
    base.tinyvit.train(model, train_loader, base.tinyvit.DENSE_EPOCHS, lr=3e-4)
    dense_state = {k: value.detach().clone() for k, value in model.state_dict().items()}
    dense_acc = base.tinyvit.evaluate(model, test_loader)
    mag = base.tinyvit.magnitude_scores(model)
    syn = base.tinyvit.synflow_scores(model)
    masks_by_method = {
        "magnitude": base.tinyvit.global_mask(mag, base.SPARSITY),
        "global_synflow": base.tinyvit.global_mask(syn, base.SPARSITY),
        "minimal_liveness_repair": base.tinyvit.minimal_liveness_repair(mag, base.SPARSITY),
        "attn_mlp_readout_repair": base.tinyvit.attn_mlp_readout_repair(mag, base.SPARSITY),
        "all_route_liveness_floor": base.tinyvit.all_route_liveness_floor(mag, base.SPARSITY),
    }
    alignments = {
        method: base.diag.feature_alignment(model, dense_state, masks, test_loader)
        for method, masks in masks_by_method.items()
    }
    qualities = {method: base.tinyvit.route_quality(masks) for method, masks in masks_by_method.items()}
    before_scores = v4.masked_before_scores(model, dense_state, test_loader, masks_by_method)
    selected, reason, ranked = v5.choose_margin_policy_v5(alignments, qualities, before_scores)
    print(f"seed {seed}: dense={dense_acc:.4f} selected={selected} reason={reason}", flush=True)
    return {
        "seed": seed,
        "dense_accuracy": dense_acc,
        "selected_method": selected,
        "reason": reason,
        "ranked_candidates": ranked,
    }


def run() -> dict:
    rows = []
    target = None
    for seed in SEEDS:
        item = scan_seed(seed)
        rows.append(item)
        if item["reason"] == "synflow_masked_recovery_prior":
            target = item
            break
    result = {
        "experiment": "04_find_tiny_vit_v5_synflow_prior_seed",
        "setup": "Dense-training branch scanner for the strong TinyViT V5 selector. Computes only pre-finetune selector diagnostics and stops when V5 enters the SynFlow masked-recovery-prior branch.",
        "device": base.tinyvit.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "candidate_seeds": SEEDS,
        "scanned_count": len(rows),
        "found_synflow_prior": target is not None,
        "synflow_prior_seed": None if target is None else target["seed"],
        "rows": rows,
    }
    out = base.RESULTS / "find_tiny_vit_v5_synflow_prior_seed.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
