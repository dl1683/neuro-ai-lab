from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
FIGURES = ROOT / "figures" / "04_circuit_viability"


BG = "#0e1117"
PANEL = "#151b23"
FG = "#f0f6fc"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#ff7b72"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=FG, fontsize=16, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def pathology_chart():
    data = load("synflow_pathology_synthesis.json")
    items = data["items"]
    labels = [f"{item['dataset_model']}\n{int(item['sparsity'] * 100)}%" for item in items]
    magnitude = [item["magnitude_after"] * 100 for item in items]
    synflow = [item["global_synflow_after"] * 100 for item in items]
    layerwise = [item["layerwise_synflow_after"] * 100 for item in items]
    dead_frac = [item["global_synflow_dead_fc1_hidden"] / item["fc1_hidden_units"] * 100 for item in items]

    x = np.arange(len(items))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor=BG)
    style(
        ax,
        "Hidden cutsets: global SynFlow deletes the classifier bridge",
        "After masked fine-tuning, accuracy collapses exactly where the dense bridge has zero surviving capacity.",
    )
    bars1 = ax.bar(x - width, magnitude, width, label="Magnitude after FT", color=BLUE)
    bars2 = ax.bar(x, layerwise, width, label="Layerwise SynFlow after FT", color=PURPLE)
    bars3 = ax.bar(x + width, synflow, width, label="Global SynFlow after FT", color=RED)
    ax2 = ax.twinx()
    ax2.plot(x, dead_frac, color=YELLOW, marker="o", linewidth=2.5, label="Global SynFlow dead bridge units")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Dead bridge units (%)", color=YELLOW)
    ax2.tick_params(colors=YELLOW, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")
    ax.set_ylabel("Accuracy after FT (%)", color=FG)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=FG)
    ax.set_ylim(0, 90)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                color=FG,
                fontsize=8,
            )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False, labelcolor=FG)
    save(fig, "readme_synflow_bridge_collapse.svg")


def selector_evolution_chart():
    data = load("tiny_vit_strong_selector_boundary_synthesis.json")
    selectors = ["v3", "v4", "v5", "v6", "v7"]
    available = [s for s in selectors if f"{s}_matches_best" in data]
    seed_count = data["seed_count"]
    match_pct = [data[f"{s}_matches_best"] / seed_count * 100 for s in available]
    positive_pct = [data[f"{s}_positive_vs_magnitude"] / seed_count * 100 for s in available]
    mean_delta = [data[f"{s}_mean_delta_vs_magnitude"] * 100 for s in available]

    x = np.arange(len(available))
    fig, ax = plt.subplots(figsize=(11.5, 6.1), facecolor=BG)
    style(
        ax,
        "Selector evolution: from feature heuristic to circuit-viability policy",
        f"Boundary synthesis over {seed_count} full-CIFAR TinyViT strong seeds at 90% sparsity.",
    )
    ax.bar(x - 0.18, match_pct, 0.36, color=GREEN, label="Matches best evaluated candidate")
    ax.bar(x + 0.18, positive_pct, 0.36, color=BLUE, label="Positive vs magnitude")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Seeds (%)", color=FG)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in available], color=FG, fontsize=12, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, mean_delta, color=YELLOW, marker="o", linewidth=3, label="Mean delta vs magnitude")
    ax2.set_ylabel("Mean delta vs magnitude (points)", color=YELLOW)
    ax2.tick_params(colors=YELLOW, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")
    for i, value in enumerate(mean_delta):
        ax2.text(i, value + 0.2, f"+{value:.2f}", ha="center", color=YELLOW, fontsize=9)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", frameon=False, labelcolor=FG)
    save(fig, "readme_tinyvit_selector_evolution.svg")


def branch_case_chart():
    cases = [
        ("seed308", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong.json", "V5 feature\nSynFlow"),
        ("seed310", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong_seed310.json", "V5 recovery\nprior"),
        ("seed311", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong_seed311.json", "V5 unselected\nSynFlow"),
        ("seed312", "cifar10_tiny_vit_feature_route_margin_selector_v5_90pct_strong_seed312.json", "V5 live\nmiss"),
        ("seed313", "cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed313.json", "V6 magnitude\nguardrail"),
        ("seed315", "cifar10_tiny_vit_feature_route_margin_selector_v6_90pct_strong_seed315.json", "V6 live\nmiss"),
        ("seed320", "cifar10_tiny_vit_feature_route_margin_selector_v7_90pct_strong_seed320.json", "V7 SynFlow\nno-regression"),
    ]
    labels = []
    deltas = []
    reasons = []
    colors = []
    for _, filename, label in cases:
        item = load(filename)
        labels.append(label)
        deltas.append(item["summary"]["feature_route_margin_policy"]["after_delta_mean"] * 100)
        reasons.append(item["decisions"][0]["reason"].replace("_", " "))
        value = deltas[-1]
        colors.append(GREEN if value > 0.05 else RED if value < -0.05 else YELLOW)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 6.4), facecolor=BG)
    style(
        ax,
        "TinyViT branch evidence: the policy learns when liveness is not enough",
        "Each bar is a full masked fine-tuning run; height is selected-policy recovery over magnitude.",
    )
    bars = ax.bar(x, deltas, color=colors, width=0.62)
    ax.axhline(0, color=FG, linewidth=1)
    ax.set_ylabel("Selected policy delta vs magnitude (points)", color=FG)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=FG, fontsize=9)
    ymin = min(-1.0, min(deltas) - 0.8)
    ymax = max(deltas) + 1.1
    ax.set_ylim(ymin, ymax)
    for bar, delta, reason in zip(bars, deltas, reasons):
        va = "bottom" if delta >= 0 else "top"
        y = delta + (0.18 if delta >= 0 else -0.18)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{delta:+.2f}", ha="center", va=va, color=FG, fontsize=9)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ymin + 0.15,
            reason,
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=7,
            rotation=90,
        )
    save(fig, "readme_tinyvit_branch_cases.svg")


def main():
    pathology_chart()
    selector_evolution_chart()
    branch_case_chart()
    print(json.dumps({"written": [
        "figures/04_circuit_viability/readme_synflow_bridge_collapse.svg",
        "figures/04_circuit_viability/readme_tinyvit_selector_evolution.svg",
        "figures/04_circuit_viability/readme_tinyvit_branch_cases.svg",
    ]}, indent=2))


if __name__ == "__main__":
    main()
