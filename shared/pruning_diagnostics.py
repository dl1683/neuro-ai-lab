"""Diagnostics for severe-sparsity pruning masks.

These helpers are intentionally method-agnostic. They inspect binary masks and
report structural failure modes that can make a pruning score unusable even
when the scalar saliency values look plausible.

The motivating failure is global SynFlow on CNNs with dense classifier tails:
at 98-99% global sparsity, the mask can allocate zero parameters to the first
dense bridge layer, making masked fine-tuning unrecoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class LayerMaskSummary:
    """Basic structural summary for one mask tensor."""

    name: str
    kept: int
    total: int
    keep_rate: float
    output_units: int | None = None
    dead_output_units: int | None = None


@dataclass(frozen=True)
class DenseBridgeSummary:
    """Reachability summary for a dense bridge mask."""

    name: str
    keep_rate: float
    hidden_units: int
    dead_hidden_units: int
    min_fanin: int
    max_fanin: int
    collapsed: bool


def summarize_layer_mask(name: str, mask: torch.Tensor) -> LayerMaskSummary:
    """Return keep-rate and dead-output diagnostics for one mask."""

    binary = mask.detach() != 0
    kept = int(binary.sum().item())
    total = int(binary.numel())
    if binary.ndim >= 2:
        flattened_outputs = binary.reshape(binary.shape[0], -1)
        output_units = int(flattened_outputs.shape[0])
        dead_output_units = int((flattened_outputs.sum(dim=1) == 0).sum().item())
    else:
        output_units = None
        dead_output_units = None
    return LayerMaskSummary(
        name=name,
        kept=kept,
        total=total,
        keep_rate=kept / total if total else 0.0,
        output_units=output_units,
        dead_output_units=dead_output_units,
    )


def summarize_masks(masks: Mapping[str, torch.Tensor]) -> dict[str, LayerMaskSummary]:
    """Summarize every mask in a mapping."""

    return {name: summarize_layer_mask(name, mask) for name, mask in masks.items()}


def summarize_dense_bridge(name: str, mask: torch.Tensor) -> DenseBridgeSummary:
    """Summarize a dense bridge mask shaped [hidden, input]."""

    if mask.ndim != 2:
        raise ValueError(f"dense bridge mask must be rank 2, got shape {tuple(mask.shape)}")
    binary = mask.detach() != 0
    fanin = binary.sum(dim=1)
    dead = int((fanin == 0).sum().item())
    hidden = int(binary.shape[0])
    kept = int(binary.sum().item())
    total = int(binary.numel())
    return DenseBridgeSummary(
        name=name,
        keep_rate=kept / total if total else 0.0,
        hidden_units=hidden,
        dead_hidden_units=dead,
        min_fanin=int(fanin.min().item()) if hidden else 0,
        max_fanin=int(fanin.max().item()) if hidden else 0,
        collapsed=dead == hidden,
    )


def bridge_collapse_report(
    masks: Mapping[str, torch.Tensor],
    bridge_name: str = "fc1",
    max_dead_fraction: float = 0.50,
) -> dict[str, float | int | bool | str]:
    """Return a compact guardrail report for a dense bridge layer.

    A bridge is flagged when every hidden unit is dead or when the fraction of
    dead hidden units exceeds `max_dead_fraction`.
    """

    if bridge_name not in masks:
        raise KeyError(f"bridge mask {bridge_name!r} not found")
    summary = summarize_dense_bridge(bridge_name, masks[bridge_name])
    dead_fraction = summary.dead_hidden_units / summary.hidden_units if summary.hidden_units else 0.0
    flagged = summary.collapsed or dead_fraction > max_dead_fraction
    return {
        "bridge_name": summary.name,
        "keep_rate": summary.keep_rate,
        "hidden_units": summary.hidden_units,
        "dead_hidden_units": summary.dead_hidden_units,
        "dead_fraction": dead_fraction,
        "min_fanin": summary.min_fanin,
        "max_fanin": summary.max_fanin,
        "collapsed": summary.collapsed,
        "flagged": flagged,
    }
