from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "04_criticality_pruning"
DOC = ROOT / "experiments" / "04_criticality_pruning"

REQUIRED_ARTIFACTS = [
    OUT / "fashion_mnist_mlp_synflow_comparison.json",
    OUT / "fashion_mnist_cnn_synflow_comparison.json",
    OUT / "label_free_baseline_synthesis.json",
]

REPRO_COMMANDS = [
    "python experiments/04_criticality_pruning/fashion_mnist_mlp_synflow.py",
    "python experiments/04_criticality_pruning/fashion_mnist_cnn_synflow.py",
    "python experiments/04_criticality_pruning/synthesize_label_free_baselines.py",
]


def _run(cmd: str) -> None:
    subprocess.run(cmd, cwd=ROOT, shell=True, check=True)


def _ensure_artifacts(refresh: bool) -> None:
    if refresh or not all(path.exists() for path in REQUIRED_ARTIFACTS):
        for cmd in REPRO_COMMANDS:
            _run(cmd)


def _load() -> dict:
    return json.loads((OUT / "label_free_baseline_synthesis.json").read_text(encoding="utf-8"))


def _grade_claim(synth: dict) -> str:
    severe = synth["severe_sparsity_98"]
    if severe["vs_synflow"]["wins"] == severe["vs_synflow"]["n"] and severe["vs_synflow"]["mean"] > 0.20:
        return "strong severe-sparsity guardrail evidence"
    if synth["overall"]["vs_synflow"]["mean"] > 0.05:
        return "moderate label-free baseline evidence"
    return "weak/inconclusive"


def build_claim_card(refresh: bool = False) -> dict:
    _ensure_artifacts(refresh)
    synth = _load()
    grade = _grade_claim(synth)
    card = {
        "title": "Adaptive path correction as a label-free severe-sparsity guardrail",
        "claim_grade": grade,
        "claim": synth["claim"],
        "primary_result": {
            "setting": "Fashion-MNIST MLP and CNN, SynFlow/magnitude baselines, 90/95/98% sparsity, two seeds each",
            "overall_vs_synflow_mean_delta": synth["overall"]["vs_synflow"]["mean"],
            "overall_vs_synflow_wins": synth["overall"]["vs_synflow"]["wins"],
            "overall_vs_synflow_n": synth["overall"]["vs_synflow"]["n"],
            "severe_98_vs_synflow_mean_delta": synth["severe_sparsity_98"]["vs_synflow"]["mean"],
            "severe_98_vs_synflow_wins": synth["severe_sparsity_98"]["vs_synflow"]["wins"],
            "severe_98_vs_synflow_n": synth["severe_sparsity_98"]["vs_synflow"]["n"],
            "severe_98_vs_magnitude_mean_delta": synth["severe_sparsity_98"]["vs_magnitude"]["mean"],
            "severe_98_vs_magnitude_wins": synth["severe_sparsity_98"]["vs_magnitude"]["wins"],
            "severe_98_vs_magnitude_n": synth["severe_sparsity_98"]["vs_magnitude"]["n"],
        },
        "what_it_is_not": [
            "Not evidence for the original branching-ratio criticality mechanism.",
            "Not a universal replacement for magnitude pruning.",
            "Not yet validated on large CNNs, transformers, or production-scale datasets.",
        ],
        "reproduce": REPRO_COMMANDS,
        "source_artifacts": [str(path.relative_to(ROOT)) for path in REQUIRED_ARTIFACTS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CLAIM_CARD.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Claim Card: Adaptive Path Correction",
        "",
        f"Claim grade: **{grade}**",
        "",
        "## Claim",
        "",
        card["claim"],
        "",
        "## Primary result",
        "",
        f"Setting: {card['primary_result']['setting']}.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Overall adaptive path vs SynFlow mean delta | `{card['primary_result']['overall_vs_synflow_mean_delta']:.4f}` |",
        f"| Overall adaptive path vs SynFlow wins | `{card['primary_result']['overall_vs_synflow_wins']} / {card['primary_result']['overall_vs_synflow_n']}` |",
        f"| 98% sparsity adaptive path vs SynFlow mean delta | `{card['primary_result']['severe_98_vs_synflow_mean_delta']:.4f}` |",
        f"| 98% sparsity adaptive path vs SynFlow wins | `{card['primary_result']['severe_98_vs_synflow_wins']} / {card['primary_result']['severe_98_vs_synflow_n']}` |",
        f"| 98% sparsity adaptive path vs magnitude mean delta | `{card['primary_result']['severe_98_vs_magnitude_mean_delta']:.4f}` |",
        f"| 98% sparsity adaptive path vs magnitude wins | `{card['primary_result']['severe_98_vs_magnitude_wins']} / {card['primary_result']['severe_98_vs_magnitude_n']}` |",
        "",
        "## What this is not",
        "",
    ]
    for item in card["what_it_is_not"]:
        md.append(f"- {item}")
    md.extend(["", "## Reproduce", ""])
    for cmd in REPRO_COMMANDS:
        md.append(f"- `{cmd}`")
    md.extend(["", "## Source artifacts", ""])
    for path in card["source_artifacts"]:
        md.append(f"- `{path}`")
    (DOC / "CLAIM_CARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return card


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    card = build_claim_card(refresh=refresh)
    print(json.dumps({"claim_grade": card["claim_grade"], "primary_result": card["primary_result"]}, indent=2))
