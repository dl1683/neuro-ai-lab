from __future__ import annotations

import json

import numpy as np
import torch

import cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed as v6


SEEDS = [314, 315, 316, 317, 318, 319]


def run():
    base = v6.v5.v4.base
    rows = []
    found_seed = None
    for seed in SEEDS:
        print(f"seed {seed}: train dense TinyViT V6 live-repair scanner", flush=True)
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
        before_scores = v6.v5.v4.masked_before_scores(model, dense_state, test_loader, masks_by_method)
        selected_method, reason, ranked = v6.choose_margin_policy_v6(alignments, qualities, before_scores)
        row = {
            "seed": seed,
            "dense_accuracy": dense_acc,
            "selected_method": selected_method,
            "reason": reason,
            "ranked_candidates": ranked,
        }
        rows.append(row)
        print(f"seed {seed}: dense={dense_acc:.4f} selected={selected_method} reason={reason}", flush=True)
        if reason == "live_repair_masked_before_tiebreak":
            found_seed = seed
            break

    result = {
        "experiment": "04_find_tiny_v6_live_repair_tiebreak_seed",
        "setup": (
            "Dense-training branch scanner for the strong TinyViT V6 selector. Computes only pre-finetune "
            "selector diagnostics and stops when V6 enters the live-repair masked-before tie-breaker branch."
        ),
        "device": base.tinyvit.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "candidate_seeds": SEEDS,
        "scanned_count": len(rows),
        "found_live_repair_tiebreak": found_seed is not None,
        "live_repair_tiebreak_seed": found_seed,
        "rows": rows,
    }
    out = base.RESULTS / "find_tiny_v6_live_repair_tiebreak_seed.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
