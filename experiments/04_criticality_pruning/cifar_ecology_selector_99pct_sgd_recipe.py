from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

import cifar10_full_resnet20_capacity_99pct_sgd_recipe as c10
import cifar100_full_resnet20_capacity_99pct_sgd_recipe as c100
import cifar10_tiny_resnet_capacity_transfer as base
from shared.circuit_viability_selector import choose_ecology_aware_method, split_dict
from shared.residual_route_capacity import route_split_capacity_mask


SPARSITY = 0.99
RESERVE = 0.60
TASKS = {
    "cifar10": {"seeds": [263, 264]},
    "cifar100": {"seeds": [265, 266]},
}
METHODS = ["magnitude", "plain_reserve", "predicted_route_split", "ecology_selected"]


def task_handles(task: str):
    if task == "cifar10":
        return c10.full.full_loaders, lambda: c10.r20.CifarResNet20().to(base.DEVICE), c10.train_sgd, c10.eval_method, c10.r20.route_quality, c10.torch
    if task == "cifar100":
        return c100.full_loaders, c100.make_model, c100.train_sgd, c100.eval_method, c100.r20.route_quality, c100.torch
    raise ValueError(task)


def summarize(rows):
    summary = {}
    paired = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        task_summary = {}
        for method in METHODS:
            selected = [row for row in task_rows if row["method"] == method]
            task_summary[method] = {
                "after_mean": float(np.mean([row["after_accuracy"] for row in selected])),
                "after_std": float(np.std([row["after_accuracy"] for row in selected])),
                "projection_min_mean": float(np.mean([row["route_quality"]["projection_min"] for row in selected])),
                "fc_score_mean": float(np.mean([row["route_quality"]["fc_score"] for row in selected])),
                "main_path_min_mean": float(np.mean([row["route_quality"]["main_path_min"] for row in selected])),
                "dead_outputs_mean": float(np.mean([row["route_quality"]["total_dead_outputs"] for row in selected])),
            }
        summary[task] = task_summary
        task_paired = []
        for method in METHODS:
            if method == "magnitude":
                continue
            deltas = []
            for seed in TASKS[task]["seeds"]:
                mag = next(row for row in task_rows if row["seed"] == seed and row["method"] == "magnitude")
                alt = next(row for row in task_rows if row["seed"] == seed and row["method"] == method)
                deltas.append({"seed": seed, "after_delta": alt["after_accuracy"] - mag["after_accuracy"]})
            task_summary[method]["after_delta_mean"] = float(np.mean([row["after_delta"] for row in deltas]))
            task_summary[method]["after_wins"] = int(sum(row["after_delta"] > 0 for row in deltas))
            task_paired.append(
                {
                    "method": method,
                    "after_delta_mean": float(np.mean([row["after_delta"] for row in deltas])),
                    "after_delta_std": float(np.std([row["after_delta"] for row in deltas])),
                    "after_wins": int(sum(row["after_delta"] > 0 for row in deltas)),
                    "paired_rows": deltas,
                }
            )
        paired[task] = task_paired
    return summary, paired


def write_report(result):
    out = ROOT / "results" / "04_criticality_pruning" / "cifar_ecology_selector_99pct_sgd_recipe.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments" / "04_criticality_pruning" / "CIFAR_ECOLOGY_SELECTOR_99PCT_SGD_RECIPE.md"
    lines = [
        "# CIFAR Ecology-Aware Circuit-Viability Selector at 99%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        "",
    ]
    for task in TASKS:
        lines.extend(
            [
                f"## {task}",
                "",
                f"Seeds: `{TASKS[task]['seeds']}`",
                "",
                "| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            item = result["summary"][task][method]
            delta = ""
            wins = ""
            if method != "magnitude":
                delta = f"`{item['after_delta_mean']:+.4f}`"
                wins = f"`{item['after_wins']}/{len(TASKS[task]['seeds'])}`"
            lines.append(
                f"| `{method}` | `{item['after_mean']:.4f}` | `{item['after_std']:.4f}` | {delta} | {wins} | "
                f"`{item['main_path_min_mean']:.4f}` | `{item['projection_min_mean']:.4f}` | "
                f"`{item['fc_score_mean']:.4f}` | `{item['dead_outputs_mean']:.1f}` |"
            )
        lines.extend(["", "Selections:", ""])
        for item in [x for x in result["decisions"] if x["task"] == task]:
            lines.append(
                f"- seed `{item['seed']}`: selected `{item['selected_method']}` "
                f"readout_ratio `{item['plain_readout_ratio']:.4f}` split `{item['selected_split']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The selector first measures the plain-reserve readout ratio against the magnitude readout template. If plain reserve has a large readout deficit, it uses the conservative predicted route split; otherwise it keeps broad reserve. This tests whether task ecology can choose the intervention family before fine-tuning.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    rows = []
    decisions = []
    gpu_name = None
    for task, cfg in TASKS.items():
        full_loaders, make_model, train_model, eval_method, route_quality, torch_mod = task_handles(task)
        if torch_mod.cuda.is_available():
            gpu_name = torch_mod.cuda.get_device_name(0)
        for seed in cfg["seeds"]:
            print(f"{task} seed {seed}: train dense SGD/cosine", flush=True)
            torch_mod.manual_seed(seed)
            np.random.seed(seed)
            train_loader, test_loader = full_loaders(seed)
            model = make_model()
            train_model(model, train_loader)
            dense_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            dense_accuracy = base.evaluate(model, test_loader)
            print(f"{task} seed {seed}: dense_accuracy={dense_accuracy:.4f} device={base.DEVICE}", flush=True)
            mag = base.magnitude_scores(model)
            syn = base.synflow_scores(model)
            decision = choose_ecology_aware_method(syn, mag, SPARSITY, RESERVE, base.capacity_mask, base.global_mask, route_quality)
            split = decision["best_split"]["split"]
            masks_by_label = {
                "magnitude": base.global_mask(mag, SPARSITY),
                "plain_reserve": base.capacity_mask(syn, mag, SPARSITY, RESERVE),
                "predicted_route_split": route_split_capacity_mask(syn, mag, SPARSITY, RESERVE, split),
                "ecology_selected": decision["selected_mask"],
            }
            decisions.append(
                {
                    "task": task,
                    "seed": seed,
                    "selected_method": decision["selected_method"],
                    "selected_split": decision["selected_split"],
                    "plain_readout_ratio": decision["plain_readout_ratio"],
                    "readout_ratio_threshold": decision["readout_ratio_threshold"],
                    "best_split": split_dict(split),
                    "plain_quality": decision["plain_quality"],
                    "magnitude_quality": decision["magnitude_quality"],
                    "best_split_quality": decision["best_split"]["quality"],
                }
            )
            print(
                f"{task} seed {seed}: selected={decision['selected_method']} "
                f"readout_ratio={decision['plain_readout_ratio']:.4f} best_split={split_dict(split)}",
                flush=True,
            )
            for label, masks in masks_by_label.items():
                before, after = eval_method(model, dense_state, train_loader, test_loader, masks)
                quality = route_quality(masks)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "method": label,
                        "dense_accuracy": dense_accuracy,
                        "before_accuracy": before,
                        "after_accuracy": after,
                        "route_quality": quality,
                    }
                )
                print(
                    f"{task} seed {seed} {label}: after={after:.4f} proj={quality['projection_min']:.4f} "
                    f"fc={quality['fc_score']:.4f} main={quality['main_path_min']:.4f} dead={quality['total_dead_outputs']}",
                    flush=True,
                )
    summary, paired = summarize(rows)
    result = {
        "experiment": "04_cifar_ecology_selector_99pct_sgd_recipe",
        "setup": "Fresh CIFAR-10 and CIFAR-100 full-dataset ResNet-20-style validation of an ecology-aware pre-finetune selector that chooses broad reserve or conservative route split from readout deficit.",
        "device": base.DEVICE,
        "gpu_name": gpu_name,
        "sparsity": SPARSITY,
        "reserve": RESERVE,
        "tasks": TASKS,
        "summary": summary,
        "paired_deltas": paired,
        "decisions": decisions,
        "rows": rows,
    }
    write_report(result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired_deltas": result["paired_deltas"], "decisions": result["decisions"]}, indent=2))
