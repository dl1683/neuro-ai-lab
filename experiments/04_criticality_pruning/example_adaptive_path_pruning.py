from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.adaptive_path_pruning import (
    PathPruningConfig,
    dense_calibration_stats,
    dense_path_scores,
    global_topk_mask,
    magnitude_scores,
)


def main() -> dict:
    torch.manual_seed(123)
    fc1 = torch.randn(16, 32)
    fc2 = torch.randn(10, 16)
    flat_inputs = torch.randn(128, 32)
    hidden = torch.relu(flat_inputs @ fc1.T)
    input_signal, hidden_strength = dense_calibration_stats(flat_inputs, hidden)

    config = PathPruningConfig(sparsity=0.98)
    path_scores = dense_path_scores(fc1, fc2, input_signal, hidden_strength, config)
    mag_scores = magnitude_scores({"fc1": fc1, "fc2": fc2})
    path_mask = global_topk_mask(path_scores, sparsity=0.98)
    mag_mask = global_topk_mask(mag_scores, sparsity=0.98)

    result = {
        "example": "adaptive path pruning dense score smoke example",
        "sparsity": 0.98,
        "alpha": config.resolved_alpha(),
        "path_kept_fc1": int(path_mask["fc1"].sum().item()),
        "path_kept_fc2": int(path_mask["fc2"].sum().item()),
        "magnitude_kept_fc1": int(mag_mask["fc1"].sum().item()),
        "magnitude_kept_fc2": int(mag_mask["fc2"].sum().item()),
        "path_mask_total": int(path_mask["fc1"].sum().item() + path_mask["fc2"].sum().item()),
        "magnitude_mask_total": int(mag_mask["fc1"].sum().item() + mag_mask["fc2"].sum().item()),
    }
    out = ROOT / "results" / "04_criticality_pruning" / "adaptive_path_pruning_example.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
