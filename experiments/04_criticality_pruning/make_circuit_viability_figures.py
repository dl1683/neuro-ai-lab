from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
FIGURES = ROOT / "figures" / "04_circuit_viability"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "magnitude": "#4c566a",
    "synflow": "#bf616a",
    "capacity": "#5e81ac",
    "route": "#a3be8c",
    "optimizer": "#b48ead",
}


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name: str):
    out = FIGURES / name
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_synflow_collapse():
    labels = ["Fashion\n98%", "CIFAR\n98%", "CIFAR\n99%"]
    magnitude = [80.86, 44.08, 33.24]
    synflow = [10.28, 9.76, 9.76]
    dead = [128, 192, 192]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, magnitude, width, label="Magnitude after FT", color=COLORS["magnitude"])
    ax.bar(x + width / 2, synflow, width, label="Global SynFlow after FT", color=COLORS["synflow"])
    for idx, value in enumerate(synflow):
        ax.text(idx + width / 2, value + 1.5, f"dead bridge\n{dead[idx]}/{dead[idx]}", ha="center", va="bottom", fontsize=8, color="#5a1d1d")
    ax.set_title("Global SynFlow creates zero-capacity classifier bridges")
    ax.set_ylabel("Accuracy after masked fine-tuning (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 90)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "figure_01_synflow_bridge_collapse.png")


def figure_cnn_capacity():
    labels = ["CIFAR CNN\n99%", "Fashion CNN\n99%"]
    magnitude = [32.31, 80.24]
    synflow = [9.76, 9.89]
    capacity = [34.88, 81.75]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width, magnitude, width, label="Magnitude", color=COLORS["magnitude"])
    ax.bar(x, synflow, width, label="Global SynFlow", color=COLORS["synflow"])
    ax.bar(x + width, capacity, width, label="Path capacity", color=COLORS["capacity"])
    for idx, delta in enumerate([2.56, 1.51]):
        ax.text(idx + width, capacity[idx] + 2, f"+{delta:.2f} pts", ha="center", fontsize=9, color="#2d4a22")
    ax.set_title("Capacity constraints convert collapse into recoverable masks")
    ax.set_ylabel("Accuracy after masked fine-tuning (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "figure_02_cnn_capacity_rescue.png")


def figure_residual_transfer():
    groups = ["TinyResNet\nsubset 99%", "ResNet-20 style\nsubset 99%", "ResNet-20 style\nfull CIFAR 99%", "Full CIFAR\nSGD recipe 99%"]
    magnitude = [24.80, 27.25, 37.72, 42.87]
    synflow = [10.0, 10.03, 10.0, 10.0]
    reserve = [20.13, 33.22, 39.21, 49.43]
    best_route = [25.85, 33.22, 38.06, np.nan]
    x = np.arange(len(groups))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - 1.5 * width, magnitude, width, label="Magnitude", color=COLORS["magnitude"])
    ax.bar(x - 0.5 * width, synflow, width, label="Global SynFlow", color=COLORS["synflow"])
    ax.bar(x + 0.5 * width, reserve, width, label="Reserve", color=COLORS["capacity"])
    ax.bar(x + 1.5 * width, best_route, width, label="Route split/optimizer", color=COLORS["optimizer"])
    ax.text(x[0] + 1.5 * width, best_route[0] + 1.2, "+1.05 pts\n4/4 wins", ha="center", fontsize=8)
    ax.text(x[1] + 0.5 * width, reserve[1] + 1.2, "+5.97 pts\n4/4 wins", ha="center", fontsize=8)
    ax.text(x[2] + 0.5 * width, reserve[2] + 1.2, "+1.50 pts\n6/6 wins", ha="center", fontsize=8)
    ax.text(x[3] + 0.5 * width, reserve[3] + 1.2, "+6.57 pts\n4/4 wins", ha="center", fontsize=8)
    ax.set_title("Residual transfer: liveness first, route diversity when needed")
    ax.set_ylabel("Accuracy after masked fine-tuning (%)")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 56)
    ax.legend(frameon=False, ncols=4, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "figure_03_residual_transfer.png")


def figure_mechanism_hierarchy():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.axis("off")
    steps = [
        ("1", "Homeostasis", "prevent dead\nroutes/outputs"),
        ("2", "Route targets", "preserve projection,\nreadout, main paths"),
        ("3", "Degeneracy", "avoid single-family\noverconcentration"),
        ("4", "Prediction", "choose constraints\nfrom route deficits"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for idx, (num, title, body) in enumerate(steps):
        x = xs[idx]
        circle = plt.Circle((x, 0.65), 0.055, color=COLORS["capacity"] if idx < 2 else COLORS["optimizer"], transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x, 0.65, num, ha="center", va="center", color="white", weight="bold", transform=ax.transAxes)
        ax.text(x, 0.45, title, ha="center", va="center", fontsize=11, weight="bold", transform=ax.transAxes)
        ax.text(x, 0.25, body, ha="center", va="center", fontsize=9, transform=ax.transAxes)
        if idx < len(steps) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.08, 0.65), xytext=(x + 0.08, 0.65), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#4c566a"})
    ax.set_title("Circuit-viability hierarchy for severe pruning", fontsize=14, pad=10)
    return save(fig, "figure_04_mechanism_hierarchy.png")


def figure_output_diversity():
    groups = ["CIFAR-10\nSGD reserve", "CIFAR-100\nplain reserve", "CIFAR-100\necology-selected"]
    magnitude = [42.87, 6.58, 5.41]
    capacity = [49.43, 7.64, 9.08]
    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, magnitude, width, label="Magnitude", color=COLORS["magnitude"])
    ax.bar(x + width / 2, capacity, width, label="Viability-constrained", color=COLORS["capacity"])
    ax.text(x[0] + width / 2, capacity[0] + 1.0, "+6.57 pts\n4/4 wins", ha="center", fontsize=8)
    ax.text(x[1] + width / 2, capacity[1] + 1.0, "+1.06 pts\n2/2 wins", ha="center", fontsize=8)
    ax.text(x[2] + width / 2, capacity[2] + 1.0, "+3.68 pts\n2/2 wins", ha="center", fontsize=8)
    ax.set_title("Output diversity shifts the viable circuit constraint")
    ax.set_ylabel("Accuracy after masked fine-tuning (%)")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 56)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "figure_05_output_diversity.png")


def run():
    paths = [figure_synflow_collapse(), figure_cnn_capacity(), figure_residual_transfer(), figure_mechanism_hierarchy(), figure_output_diversity()]
    md = FIGURES / "README.md"
    lines = ["# Circuit Viability Figure Set", "", "Generated figures for the pruning-as-circuit-viability story.", ""]
    for path in paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([str(p.relative_to(ROOT)) for p in paths], indent=2))


if __name__ == "__main__":
    run()
