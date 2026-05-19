"""Cutset predictors for severe global pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class CutsetLayerReport:
    name: str
    keep_rate: float
    kept: int
    total: int
    output_units: int | None
    dead_output_units: int | None
    min_fanin: int | None
    max_fanin: int | None


def global_threshold(scores: Mapping[str, torch.Tensor], sparsity: float) -> torch.Tensor:
    flat = torch.cat([score.detach().flatten() for score in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    return torch.topk(flat, keep, largest=True).values.min()


def layer_report(name: str, score: torch.Tensor, threshold: torch.Tensor) -> CutsetLayerReport:
    mask = score.detach() >= threshold
    kept = int(mask.sum().item())
    total = int(mask.numel())
    if mask.ndim >= 2:
        outputs = mask.reshape(mask.shape[0], -1)
        fanin = outputs.sum(dim=1)
        return CutsetLayerReport(
            name=name,
            keep_rate=kept / total if total else 0.0,
            kept=kept,
            total=total,
            output_units=int(outputs.shape[0]),
            dead_output_units=int((fanin == 0).sum().item()),
            min_fanin=int(fanin.min().item()),
            max_fanin=int(fanin.max().item()),
        )
    return CutsetLayerReport(name=name, keep_rate=kept / total if total else 0.0, kept=kept, total=total, output_units=None, dead_output_units=None, min_fanin=None, max_fanin=None)


def predict_cutsets(scores: Mapping[str, torch.Tensor], sparsity: float) -> dict[str, CutsetLayerReport]:
    threshold = global_threshold(scores, sparsity)
    return {name: layer_report(name, score, threshold) for name, score in scores.items()}


def flagged_cutsets(reports: Mapping[str, CutsetLayerReport], max_dead_fraction: float = 0.0) -> list[str]:
    flagged = []
    for name, report in reports.items():
        if report.output_units is None or report.dead_output_units is None:
            continue
        dead_fraction = report.dead_output_units / report.output_units if report.output_units else 0.0
        if dead_fraction > max_dead_fraction:
            flagged.append(name)
    return flagged
