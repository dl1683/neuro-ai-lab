from __future__ import annotations

import json

import tinyimagenet_resnet18_pretrained_ecology_selector_99pct as pre
from shared.circuit_viability_selector import choose_ecology_aware_method, choose_tradeoff_viability_method, split_dict
from shared.feature_preserving_viability import magnitude_with_liveness_repair
from shared.residual_route_capacity import route_split_capacity_mask


pre.SEED = 281
pre.SPARSITY = 0.95
pre.METHODS = [
    "magnitude",
    "feature_viability_repair",
    "plain_reserve",
    "predicted_route_split",
    "tradeoff_policy",
]


def summarize(rows):
    summary = {}
    mag_after = next(row["after_accuracy"] for row in rows if row["method"] == "magnitude")
    for method in pre.METHODS:
        row = next(row for row in rows if row["method"] == method)
        item = {
            "after_mean": row["after_accuracy"],
            "projection_min_mean": row["route_quality"]["projection_min"],
            "fc_score_mean": row["route_quality"]["fc_score"],
            "main_path_min_mean": row["route_quality"]["main_path_min"],
            "dead_outputs_mean": row["route_quality"]["total_dead_outputs"],
        }
        if method != "magnitude":
            item["after_delta_mean"] = row["after_accuracy"] - mag_after
            item["after_wins"] = int(row["after_accuracy"] > mag_after)
        summary[method] = item
    return summary


def write_report(result):
    out = pre.ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = pre.ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_TRADEOFF_SELECTOR_95PCT.md"
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Tradeoff Selector at 95%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seed: `{result['seed']}`",
        f"Train subset: `{pre.TRAIN_N}`; validation subset: `{pre.VAL_N}`",
        f"Dense epochs: `{pre.DENSE_EPOCHS}`; masked fine-tune epochs: `{pre.FT_EPOCHS}`",
        f"Dense accuracy: `{result['dense_accuracy']:.4f}`",
        "",
        "| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in pre.METHODS:
        item = result["summary"][method]
        delta = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(["", "## Selector decision", ""])
    ranked = ", ".join(f"{item['method']}={item['score']:.3f}" for item in result["decision"]["ranked_methods"])
    lines.append(f"Selected method: `{result['decision']['selected_method']}`")
    lines.append(f"Scores: {ranked}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a pretrained/external validation of the feature-preservation versus liveness tradeoff selector. The selector is not allowed to use post-finetune accuracy; it ranks candidate masks from pre-finetune route-quality diagnostics and feature overlap with the magnitude template.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    print(f"tinyimagenet pretrained tradeoff seed {pre.SEED}: train dense", flush=True)
    pre.torch.manual_seed(pre.SEED)
    pre.np.random.seed(pre.SEED)
    train_loader, val_loader = pre.loaders(pre.SEED)
    model = pre.make_model()
    pre.train_model(model, train_loader)
    dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dense_accuracy = pre.base.evaluate(model, val_loader)
    print(f"tinyimagenet pretrained tradeoff seed {pre.SEED}: dense_accuracy={dense_accuracy:.4f}", flush=True)

    mag = pre.base.magnitude_scores(model)
    syn = pre.synflow_scores_224(model)
    ecology = choose_ecology_aware_method(syn, mag, pre.SPARSITY, pre.RESERVE, pre.base.capacity_mask, pre.base.global_mask, pre.route_quality)
    split = ecology["best_split"]["split"]
    candidate_masks = {
        "magnitude": pre.base.global_mask(mag, pre.SPARSITY),
        "feature_viability_repair": magnitude_with_liveness_repair(mag, pre.SPARSITY),
        "plain_reserve": pre.base.capacity_mask(syn, mag, pre.SPARSITY, pre.RESERVE),
        "predicted_route_split": route_split_capacity_mask(syn, mag, pre.SPARSITY, pre.RESERVE, split, skip_names=set()),
    }
    tradeoff = choose_tradeoff_viability_method(candidate_masks=candidate_masks, route_quality_fn=pre.route_quality)
    source_method = tradeoff["selected_method"]
    print(f"tinyimagenet pretrained tradeoff seed {pre.SEED}: selected={source_method}", flush=True)

    rows = []
    evaluated = {}
    for label, masks in candidate_masks.items():
        before, after = pre.evaluate_method(model, dense_state, train_loader, val_loader, masks)
        quality = pre.route_quality(masks)
        row = {
            "seed": pre.SEED,
            "method": label,
            "dense_accuracy": dense_accuracy,
            "before_accuracy": before,
            "after_accuracy": after,
            "route_quality": quality,
        }
        rows.append(row)
        evaluated[label] = row
        print(
            f"tinyimagenet pretrained tradeoff {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
            f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
            flush=True,
        )
    policy_row = dict(evaluated[source_method])
    policy_row["method"] = "tradeoff_policy"
    policy_row["policy_source_method"] = source_method
    rows.append(policy_row)
    result = {
        "experiment": "04_tinyimagenet_resnet18_pretrained_tradeoff_selector_95pct",
        "setup": "TinyImageNet-200 pretrained ResNet-18 95% sparsity validation of the feature-preservation / liveness tradeoff selector.",
        "device": pre.base.DEVICE,
        "gpu_name": pre.torch.cuda.get_device_name(0) if pre.torch.cuda.is_available() else None,
        "seed": pre.SEED,
        "train_subset": pre.TRAIN_N,
        "val_subset": pre.VAL_N,
        "sparsity": pre.SPARSITY,
        "reserve": pre.RESERVE,
        "dense_epochs": pre.DENSE_EPOCHS,
        "finetune_epochs": pre.FT_EPOCHS,
        "dense_accuracy": dense_accuracy,
        "summary": summarize(rows),
        "decision": {
            "selected_method": source_method,
            "ecology_selected_method": ecology["selected_method"],
            "ecology_split": split_dict(split),
            "ranked_methods": [
                {
                    "method": item["method"],
                    "score": item["score"],
                    "feature_overlap_with_magnitude": item["feature_overlap_with_magnitude"],
                    "liveness": item["liveness"],
                    "readout_ratio": item["readout_ratio"],
                    "dead_penalty": item["dead_penalty"],
                }
                for item in tradeoff["ranked_methods"]
            ],
        },
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dense_accuracy": result["dense_accuracy"], "summary": result["summary"], "decision": result["decision"]}, indent=2))
