from __future__ import annotations

import json

import tinyimagenet_resnet18_pretrained_ecology_selector_99pct as pre
from shared.feature_preserving_viability import magnitude_with_liveness_repair


pre.SEED = 274
pre.SPARSITY = 0.95
pre.METHODS = ["magnitude", "plain_reserve", "feature_viability_repair"]


def run():
    print(f"tinyimagenet pretrained feature-viability seed {pre.SEED}: train dense", flush=True)
    pre.torch.manual_seed(pre.SEED)
    pre.np.random.seed(pre.SEED)
    train_loader, val_loader = pre.loaders(pre.SEED)
    model = pre.make_model()
    pre.train_model(model, train_loader)
    dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dense_accuracy = pre.base.evaluate(model, val_loader)
    print(f"tinyimagenet pretrained feature-viability seed {pre.SEED}: dense_accuracy={dense_accuracy:.4f}", flush=True)
    mag = pre.base.magnitude_scores(model)
    syn = pre.synflow_scores_224(model)
    method_masks = {
        "magnitude": pre.base.global_mask(mag, pre.SPARSITY),
        "plain_reserve": pre.base.capacity_mask(syn, mag, pre.SPARSITY, pre.RESERVE),
        "feature_viability_repair": magnitude_with_liveness_repair(mag, pre.SPARSITY),
    }
    rows = []
    for label, masks in method_masks.items():
        before, after = pre.evaluate_method(model, dense_state, train_loader, val_loader, masks)
        quality = pre.route_quality(masks)
        rows.append(
            {
                "seed": pre.SEED,
                "method": label,
                "dense_accuracy": dense_accuracy,
                "before_accuracy": before,
                "after_accuracy": after,
                "route_quality": quality,
            }
        )
        print(
            f"tinyimagenet pretrained feature {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
            f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
            flush=True,
        )
    result = {
        "experiment": "04_tinyimagenet_resnet18_pretrained_feature_viability_95pct",
        "setup": "TinyImageNet-200 pretrained ResNet-18 95% sparsity test for magnitude-first feature-subspace preservation plus minimal liveness repair.",
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
        "rows": rows,
    }
    write_report(result)
    return result


def summarize(rows):
    summary = {}
    mag_after = next(row["after_accuracy"] for row in rows if row["method"] == "magnitude")
    for row in rows:
        method = row["method"]
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
    out = pre.ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = pre.ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_95PCT.md"
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seed: `{result['seed']}`",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This tests the new pretrained-network hypothesis: start from feature-preserving magnitude pruning, then only repair true dead output rows while preserving the global parameter budget. If this works better than reserve, the external failure is not a rejection of circuit viability; it means pretrained systems need viability constrained by feature-subspace preservation.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dense_accuracy": result["dense_accuracy"], "summary": result["summary"]}, indent=2))
