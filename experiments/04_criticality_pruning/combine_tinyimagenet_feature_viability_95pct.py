from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
SOURCES = [
    "tinyimagenet_resnet18_pretrained_feature_viability_95pct.json",
    "tinyimagenet_resnet18_pretrained_feature_viability_95pct_replicate.json",
]
OUT = RESULTS / "tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_95PCT_TWOSEED.md"
METHODS = ["magnitude", "plain_reserve", "feature_viability_repair"]


def run():
    rows = []
    seeds = []
    metadata = {}
    for source in SOURCES:
        data = json.loads((RESULTS / source).read_text(encoding="utf-8"))
        rows.extend(data["rows"])
        seeds.append(data["seed"])
        metadata = data
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
        "experiment": "04_tinyimagenet_resnet18_pretrained_feature_viability_95pct_twoseed",
        "setup": "Two-seed TinyImageNet-200 pretrained ResNet-18 95% sparsity aggregate for magnitude-first feature-subspace preservation plus minimal liveness repair.",
        "device": metadata["device"],
        "gpu_name": metadata["gpu_name"],
        "seeds": seeds,
        "sparsity": metadata["sparsity"],
        "sources": SOURCES,
        "summary": summary,
        "paired_deltas": paired,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    return result


def write_markdown(result):
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%: Two-Seed Aggregate",
        "",
        result["setup"],
        "",
        f"Seeds: `{result['seeds']}`",
        f"Sources: `{SOURCES[0]}`, `{SOURCES[1]}`",
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
            "Across two pretrained TinyImageNet seeds, feature-viability repair preserves magnitude-level accuracy while eliminating dead outputs. Broad reserve consistently destroys pretrained performance.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
