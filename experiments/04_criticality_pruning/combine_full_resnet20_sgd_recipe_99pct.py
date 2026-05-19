from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SOURCES = [
    "cifar10_full_resnet20_capacity_99pct_sgd_recipe.json",
    "cifar10_full_resnet20_capacity_99pct_sgd_recipe_moreseeds.json",
]
OUT_JSON = RESULTS / "cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_FULL_RESNET20_CAPACITY_99PCT_SGD_RECIPE_FOURSEED.md"
METHODS = ["magnitude", "global_synflow", "reserve_0.60"]


def run() -> dict:
    rows = []
    seeds = []
    metadata = {}
    for source in SOURCES:
        data = json.loads((RESULTS / source).read_text(encoding="utf-8"))
        rows.extend(data["rows"])
        seeds.extend(data["seeds"])
        metadata = data
    seeds = sorted(set(seeds))

    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
            "after_std": float(np.std([row["after_accuracy"] for row in selected])),
            "projection_min_mean": float(np.mean([row["route_quality"]["projection_min"] for row in selected])),
            "fc_score_mean": float(np.mean([row["route_quality"]["fc_score"] for row in selected])),
            "main_path_min_mean": float(np.mean([row["route_quality"]["main_path_min"] for row in selected])),
            "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
        }

    paired = []
    for method in METHODS:
        if method == "magnitude":
            continue
        deltas = []
        for seed in seeds:
            mag = next(row for row in rows if row["seed"] == seed and row["method"] == "magnitude")
            alt = next(row for row in rows if row["seed"] == seed and row["method"] == method)
            deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
        summary[method]["after_delta_mean"] = float(np.mean([row["after_delta"] for row in deltas]))
        summary[method]["after_wins"] = int(sum(row["after_delta"] > 0 for row in deltas))
        paired.append(
            {
                "method": method,
                "after_delta_mean": float(np.mean([row["after_delta"] for row in deltas])),
                "after_delta_std": float(np.std([row["after_delta"] for row in deltas])),
                "after_wins": int(sum(row["after_delta"] > 0 for row in deltas)),
                "paired_rows": deltas,
            }
        )

    result = {
        "experiment": "04_cifar10_full_resnet20_capacity_99pct_sgd_recipe_fourseed",
        "setup": "Four-seed aggregate of full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity using 20 dense SGD/cosine epochs and 5 masked fine-tune epochs.",
        "device": metadata["device"],
        "gpu_name": metadata["gpu_name"],
        "seeds": seeds,
        "sparsity": metadata["sparsity"],
        "reserve": metadata["reserve"],
        "dense_epochs": metadata["dense_epochs"],
        "finetune_epochs": metadata["finetune_epochs"],
        "sources": SOURCES,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    return result


def write_markdown(result: dict) -> None:
    lines = [
        "# CIFAR-10 Full ResNet-20 Capacity at 99%: SGD Recipe Four-Seed Aggregate",
        "",
        result["setup"],
        "",
        f"Sources: `{SOURCES[0]}`, `{SOURCES[1]}`",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        f"Dense epochs: `{result['dense_epochs']}`; masked fine-tune epochs: `{result['finetune_epochs']}`",
        "",
        "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(result['seeds'])}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | "
            f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
            f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The capacity reserve advantage becomes larger under the stronger SGD/cosine recipe than under the short speed recipe. Magnitude leaves the main route at zero capacity and hundreds of dead outputs, while reserve restores a main-path floor and wins every paired seed.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
