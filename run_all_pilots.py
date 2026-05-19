from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "01_grokking_prediction",
    "02_sleep_training",
    "03_reconsolidation",
    "04_criticality_pruning",
    "05_ddm_depth",
]


def load_run_function(experiment: str):
    path = ROOT / "experiments" / experiment / "run.py"
    spec = importlib.util.spec_from_file_location(f"{experiment}_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def main() -> dict:
    results = {}
    for experiment in EXPERIMENTS:
        results[experiment] = load_run_function(experiment)()

    wins = sum(1 for result in results.values() if result.get("notable_win"))
    summary = {
        "pilot_suite": "neuro-ai-lab",
        "definition_of_win": "Each experiment exposes a fast metric for quality, efficiency, retention, sparsity, or compute savings.",
        "experiments_run": len(EXPERIMENTS),
        "notable_wins": wins,
        "win_rate": wins / len(EXPERIMENTS),
        "results": results,
    }
    out = ROOT / "results" / "pilot_suite_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
