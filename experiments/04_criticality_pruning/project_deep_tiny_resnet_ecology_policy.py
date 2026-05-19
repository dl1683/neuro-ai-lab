from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "04_criticality_pruning" / "cifar10_deep_tiny_resnet_ecology_selector_99pct.json"
OUT = ROOT / "results" / "04_criticality_pruning" / "cifar10_deep_tiny_resnet_ecology_selector_99pct_policy.json"
OUT_MD = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_DEEP_TINY_RESNET_ECOLOGY_SELECTOR_99PCT_POLICY.md"
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_policy"]


def summarize(rows, seeds):
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
    return summary, paired


def run():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if row["method"] != "ecology_selected"]
    for decision in data["decisions"]:
        source_method = decision["selected_method"]
        row = dict(next(row for row in data["rows"] if row["seed"] == decision["seed"] and row["method"] == source_method))
        row["method"] = "ecology_policy"
        row["policy_source_method"] = source_method
        rows.append(row)
    summary, paired = summarize(rows, data["seeds"])
    result = {
        "experiment": "04_cifar10_deep_tiny_resnet_ecology_selector_99pct_policy",
        "setup": "Policy projection of the DeepTinyResNet ecology selector, using the already evaluated method chosen by the pre-finetune selector instead of duplicate stochastic fine-tuning.",
        "device": data["device"],
        "gpu_name": data["gpu_name"],
        "seeds": data["seeds"],
        "sparsity": data["sparsity"],
        "reserve": data["reserve"],
        "summary": summary,
        "paired_deltas": paired,
        "decisions": data["decisions"],
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    return result


def write_markdown(result):
    lines = [
        "# CIFAR-10 DeepTinyResNet Ecology Selector Policy Projection",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
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
            "The selector chooses plain reserve on both DeepTinyResNet seeds because the plain-reserve readout ratio is already above threshold. This projection removes duplicate fine-tune noise by assigning the selected-policy result to the already evaluated plain-reserve row.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"]}, indent=2))
