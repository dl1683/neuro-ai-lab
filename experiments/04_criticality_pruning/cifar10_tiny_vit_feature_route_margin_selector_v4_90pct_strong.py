from __future__ import annotations

import json

import numpy as np
import torch

import cifar10_tiny_vit_feature_route_margin_selector_v3_90pct_strong as v3


base = v3.v2.base
base.SEEDS = [304]


def choose_margin_policy_v4(alignments, qualities, before_scores):
    ranked = sorted(
        [
            {
                "method": method,
                "centered_cls_cosine_mean": alignments[method]["centered_cls_cosine_mean"],
                "route_dead": base.route_dead(qualities[method]),
                "before_accuracy": before_scores[method],
            }
            for method in base.CANDIDATES
        ],
        key=lambda item: item["centered_cls_cosine_mean"],
        reverse=True,
    )
    top = ranked[0]
    syn = next(item for item in ranked if item["method"] == "global_synflow")
    mag = next(item for item in ranked if item["method"] == "magnitude")
    all_live = next(item for item in ranked if item["method"] == "all_route_liveness_floor")
    if top["method"] == "global_synflow" and (syn["centered_cls_cosine_mean"] - mag["centered_cls_cosine_mean"]) < 0.012 and syn["route_dead"] > mag["route_dead"] * 1.5:
        if all_live["centered_cls_cosine_mean"] > mag["centered_cls_cosine_mean"] and all_live["before_accuracy"] > mag["before_accuracy"]:
            return "all_route_liveness_floor", "all_route_liveness_feature_trainability_guardrail", ranked
        return "magnitude", "magnitude_masked_trainability_guardrail", ranked
    if top["method"] != "global_synflow" and (top["centered_cls_cosine_mean"] - syn["centered_cls_cosine_mean"]) < 0.006 and top["route_dead"] > syn["route_dead"] * 2:
        return syn["method"], "synflow_margin_route_risk", ranked
    return top["method"], "feature_argmax", ranked


def masked_before_scores(model, dense_state, test_loader, masks_by_method):
    scores = {}
    for method, masks in masks_by_method.items():
        model.load_state_dict(dense_state)
        scores[method] = base.tinyvit.evaluate(model, test_loader, masks)
    model.load_state_dict(dense_state)
    return scores


def run():
    rows = []
    decisions = []
    dense_scores = []
    for seed in base.SEEDS:
        print(f"seed {seed}: train dense TinyViT feature-route margin selector v4", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test_loader = base.tinyvit.loaders(seed)
        model = base.tinyvit.TinyViT().to(base.tinyvit.DEVICE)
        base.tinyvit.train(model, train_loader, base.tinyvit.DENSE_EPOCHS, lr=3e-4)
        dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        dense_acc = base.tinyvit.evaluate(model, test_loader)
        dense_scores.append(dense_acc)
        print(f"seed {seed}: dense_accuracy={dense_acc:.4f}", flush=True)
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
        before_scores = masked_before_scores(model, dense_state, test_loader, masks_by_method)
        selected_method, reason, ranked = choose_margin_policy_v4(alignments, qualities, before_scores)
        decisions.append({"seed": seed, "selected_method": selected_method, "reason": reason, "ranked_candidates": ranked})
        print(f"seed {seed}: selected={selected_method} reason={reason}", flush=True)
        evaluated = {}
        for label, masks in masks_by_method.items():
            before, after = base.tinyvit.eval_method(model, dense_state, train_loader, test_loader, masks)
            row = {
                "seed": seed,
                "method": label,
                "dense_accuracy": dense_acc,
                "before_accuracy": before,
                "after_accuracy": after,
                "feature_alignment": alignments[label],
                "route_quality": qualities[label],
            }
            rows.append(row)
            evaluated[label] = row
            print(
                f"seed {seed} {label}: after={after:.4f} centered_cls={alignments[label]['centered_cls_cosine_mean']:.4f} "
                f"before={before:.4f} dead={qualities[label]['total_dead_outputs']} mlp_dead={qualities[label]['mlp_down_dead_outputs']} "
                f"attn_dead={qualities[label]['attn_out_dead_outputs']}",
                flush=True,
            )
        policy_row = dict(evaluated[selected_method])
        policy_row["method"] = "feature_route_margin_policy"
        policy_row["policy_source_method"] = selected_method
        rows.append(policy_row)
    result = {
        "experiment": "04_cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong",
        "setup": "Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V4 feature-route selector. V4 adds masked pre-finetune accuracy as a trainability guardrail for the ambiguous liveness-vs-magnitude branch.",
        "device": base.tinyvit.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": base.SEEDS,
        "sparsity": base.SPARSITY,
        "dense_accuracy_mean": float(np.mean(dense_scores)),
        "summary": base.summarize(rows),
        "decisions": decisions,
        "rows": rows,
    }
    write_report(result)
    return result


def write_report(result):
    out = base.RESULTS / "cifar10_tiny_vit_feature_route_margin_selector_v4_90pct_strong.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V4_90PCT_STRONG.md"
    lines = [
        "# CIFAR-10 TinyViT Feature-Route Margin Selector V4 at 90%: Stronger Recipe",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Dense accuracy mean: `{result['dense_accuracy_mean']:.4f}`",
        "",
        "| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in base.METHODS:
        item = result["summary"][method]
        before = np.mean([row["before_accuracy"] for row in result["rows"] if row["method"] == method])
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(base.SEEDS)}`"
        lines.append(
            f"| `{method}` | `{before:.4f}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['centered_cls_cosine_mean']:.4f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['attn_out_dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Selector decisions", ""])
    for item in result["decisions"]:
        ranked = ", ".join(
            f"{rank['method']}={rank['centered_cls_cosine_mean']:.4f}/before{rank['before_accuracy']:.4f}/dead{rank['route_dead']}"
            for rank in item["ranked_candidates"]
        )
        lines.append(f"- seed `{item['seed']}`: selected `{item['selected_method']}` via `{item['reason']}`; {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "V4 tests whether masked pre-finetune accuracy can serve as the missing trainability term in the ambiguous liveness-vs-magnitude branch. The decision is made before masked fine-tuning; after-FT recovery is only used for evaluation.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "decisions": result["decisions"]}, indent=2))
