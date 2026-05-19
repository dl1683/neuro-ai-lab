from __future__ import annotations

import json

import cifar10_tiny_vit_circuit_viability_98pct as base


base.SEEDS = [288, 289]
base.SPARSITY = 0.95


def write_report(result):
    result["experiment"] = "04_cifar10_tiny_vit_circuit_viability_95pct"
    result["setup"] = "TinyViT CIFAR-10 subset transformer-analogue pruning test at 95% sparsity. MLP down-projection and classifier readout rows are treated as circuit bottlenecks."
    out = base.RESULTS / "cifar10_tiny_vit_circuit_viability_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = base.ROOT / "experiments" / "04_criticality_pruning" / "CIFAR10_TINY_VIT_CIRCUIT_VIABILITY_95PCT.md"
    lines = [
        "# CIFAR-10 TinyViT Circuit Viability at 95%",
        "",
        result["setup"],
        "",
        f"Device: `{result['device']}` / `{result['gpu_name']}`",
        f"Seeds: `{result['seeds']}`",
        "",
        "| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | MLP-down min | Head min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in base.METHODS:
        item = result["summary"][method]
        delta = ""
        wins = ""
        if method != "magnitude":
            delta = f"`{item['after_delta_mean']:+.4f}`"
            wins = f"`{item['after_wins']}/{len(base.SEEDS)}`"
        lines.append(
            f"| `{method}` | `{item['after_mean']:.4f}` | {delta} | {wins} | "
            f"`{item['dead_outputs_mean']:.1f}` | `{item['mlp_down_dead_outputs_mean']:.1f}` | "
            f"`{item['mlp_down_min_mean']:.1f}` | `{item['head_min_mean']:.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This lowers TinyViT sparsity from `98%` to `95%` to test whether the transformer circuit-viability interventions matter before recovery reaches the chance floor. It uses fresh seeds and the same candidate mask families as the `98%` run.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.write_report = write_report


if __name__ == "__main__":
    result = base.run()
    print(json.dumps({"dense_accuracy_mean": result["dense_accuracy_mean"], "summary": result["summary"]}, indent=2))
