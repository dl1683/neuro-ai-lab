from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_criticality_pruning"
DOC = ROOT / "experiments" / "04_criticality_pruning" / "SYNFLOW_PATHOLOGY_SYNTHESIS.md"
SYNTHESIS = RESULTS / "synflow_pathology_synthesis.json"
SOURCE_FILES = [
    RESULTS / "fashion_mnist_cnn_synflow_comparison.json",
    RESULTS / "synflow_finetune_98pct_comparison.json",
    RESULTS / "synflow_cnn_mask_forensics_98pct.json",
    RESULTS / "synflow_cnn_layerwise_rescue_98pct.json",
    RESULTS / "cifar10_cnn_synflow_pathology.json",
]


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT FAILED: {message}")


def main() -> None:
    if not SYNTHESIS.exists():
        fail(f"missing synthesis JSON: {SYNTHESIS}")
    if not DOC.exists():
        fail(f"missing synthesis doc: {DOC}")
    missing_sources = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing_sources:
        fail(f"missing source result files: {missing_sources}")

    data = json.loads(SYNTHESIS.read_text(encoding="utf-8"))
    aggregate = data.get("aggregate", {})
    items = data.get("items", [])

    if aggregate.get("cases") != len(items):
        fail("aggregate case count does not match item count")
    if aggregate.get("cases", 0) < 3:
        fail("expected at least three synthesized CNN pathology cases")
    if aggregate.get("global_synflow_zero_fc1_cases") != aggregate.get("cases"):
        fail("not every synthesized case has zero global-SynFlow fc1 allocation")
    if aggregate.get("global_synflow_after_delta_mean", 0.0) > -0.25:
        fail("global SynFlow mean after-FT delta is not strongly negative")
    if aggregate.get("layerwise_synflow_after_delta_mean", 0.0) > -0.10:
        fail("layerwise SynFlow mean after-FT delta is not negative enough")

    for item in items:
        name = item["dataset_model"]
        sparsity = item["sparsity"]
        if item["global_synflow_fc1_keep_rate"] != 0.0:
            fail(f"{name} {sparsity}: global SynFlow fc1 keep rate is not zero")
        if item["global_synflow_dead_fc1_hidden"] != item["fc1_hidden_units"]:
            fail(f"{name} {sparsity}: global SynFlow did not kill every bridge unit")
        if item["global_synflow_after"] >= item["magnitude_after"]:
            fail(f"{name} {sparsity}: global SynFlow is not worse than magnitude after FT")
        if item["layerwise_synflow_after"] >= item["magnitude_after"]:
            fail(f"{name} {sparsity}: layerwise SynFlow is not worse than magnitude after FT")

    doc_text = DOC.read_text(encoding="utf-8")
    required_phrases = [
        "Global SynFlow zero-fc1 cases: `3/3`",
        "Mean global SynFlow after-FT delta vs magnitude: `-0.4280`",
        "dense classifier bridge",
    ]
    for phrase in required_phrases:
        if phrase not in doc_text:
            fail(f"synthesis doc missing phrase: {phrase}")

    print(json.dumps({"status": "ok", "cases": aggregate["cases"], "global_synflow_after_delta_mean": aggregate["global_synflow_after_delta_mean"]}, indent=2))


if __name__ == "__main__":
    main()
