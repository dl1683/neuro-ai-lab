from __future__ import annotations

import json

import tinyimagenet_resnet18_pretrained_ecology_selector_99pct as pre


pre.SEED = 273
pre.SPARSITY = 0.95


def write_report(result):
    result["experiment"] = "04_tinyimagenet_resnet18_pretrained_ecology_selector_95pct"
    result["setup"] = "TinyImageNet-200 external-proxy subset stress test using an ImageNet-pretrained ResNet-18 adapted to 200 classes, 95% sparsity, and the fixed ecology-aware selector."
    out = pre.ROOT / "results" / "04_criticality_pruning" / "tinyimagenet_resnet18_pretrained_ecology_selector_95pct.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = pre.ROOT / "experiments" / "04_criticality_pruning" / "TINYIMAGENET_RESNET18_PRETRAINED_ECOLOGY_SELECTOR_95PCT.md"
    lines = [
        "# TinyImageNet-200 Pretrained ResNet-18 Ecology Selector at 95%",
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
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected method: `{result['decision']['selected_method']}`",
            f"Readout ratio: `{result['decision']['plain_readout_ratio']:.4f}`",
            f"Selected split: `{result['decision']['selected_split']}`",
            "",
            "## Interpretation",
            "",
            "This repeats the pretrained ResNet-18 TinyImageNet proxy at 95% sparsity to test whether the 99% external failure is a hard sparsity cliff rather than a selector-only failure.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


pre.write_report = write_report


if __name__ == "__main__":
    result = pre.run()
    print(json.dumps({"dense_accuracy": result["dense_accuracy"], "summary": result["summary"], "decision": result["decision"]}, indent=2))
