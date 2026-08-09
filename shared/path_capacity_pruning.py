"""Path-capacity constrained pruning utilities.

The goal is to preserve circuit viability under severe sparsity. A base saliency
score still ranks most parameters, but required bridge layers receive an explicit
capacity floor so global thresholding cannot create a zero-capacity cutset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class DenseBridgeConstraint:
    """Capacity constraint for a dense bridge shaped [hidden, input]."""

    name: str = "fc1"
    min_keep_rate: float = 0.005
    min_fanin_per_hidden: int = 1


def _topk_binary(score: torch.Tensor, keep: int) -> torch.Tensor:
    flat = score.detach().flatten()
    keep = max(0, min(int(keep), flat.numel()))
    if keep == 0:
        return torch.zeros_like(score)
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return (score >= threshold).to(score.dtype)


def global_topk_mask(scores: Mapping[str, torch.Tensor], sparsity: float) -> TensorDict:
    """Standard global top-k mask."""

    flat = torch.cat([score.detach().flatten() for score in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {name: (score >= threshold).to(score.dtype) for name, score in scores.items()}


def dense_bridge_capacity_mask(
    base_scores: Mapping[str, torch.Tensor],
    sparsity: float,
    bridge_constraint: DenseBridgeConstraint,
    bridge_scores: Mapping[str, torch.Tensor] | None = None,
) -> TensorDict:
    """Create a global mask with a reserved dense-bridge capacity floor.

    Args:
        base_scores: Scores used for the global fill.
        sparsity: Global sparsity target.
        bridge_constraint: Dense bridge liveness/capacity requirement.
        bridge_scores: Optional ranking scores for bridge selection. This lets a
            method use one saliency signal globally while using magnitude or an
            activation-supported score inside the bridge.

    Returns:
        A binary mask with the same total keep count as global top-k pruning.
    """

    if bridge_constraint.name not in base_scores:
        raise KeyError(f"missing bridge score {bridge_constraint.name!r}")
    scores = {name: score.detach() for name, score in base_scores.items()}
    bridge_rank_scores = bridge_scores or base_scores
    if bridge_constraint.name not in bridge_rank_scores:
        raise KeyError(f"missing bridge ranking score {bridge_constraint.name!r}")

    total_params = sum(score.numel() for score in scores.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    masks = {name: torch.zeros_like(score) for name, score in scores.items()}

    bridge_name = bridge_constraint.name
    bridge_score = bridge_rank_scores[bridge_name].detach()
    if bridge_score.ndim != 2:
        raise ValueError(f"bridge score must be rank 2, got {tuple(bridge_score.shape)}")

    bridge_keep = max(
        int(round(bridge_constraint.min_keep_rate * bridge_score.numel())),
        int(bridge_constraint.min_fanin_per_hidden * bridge_score.shape[0]),
    )
    bridge_keep = min(bridge_keep, bridge_score.numel(), total_keep)

    bridge_mask = torch.zeros_like(bridge_score)
    protected = torch.zeros_like(bridge_score, dtype=torch.bool)

    # First guarantee minimum fan-in for every hidden unit.
    for row in range(bridge_score.shape[0]):
        row_keep = min(bridge_constraint.min_fanin_per_hidden, bridge_score.shape[1])
        if row_keep <= 0:
            continue
        idx = torch.topk(bridge_score[row], row_keep, largest=True).indices
        bridge_mask[row, idx] = 1
        protected[row, idx] = True

    # Then add the highest-ranking bridge weights until the bridge capacity is met.
    current_bridge_keep = int(bridge_mask.sum().item())
    extra = bridge_keep - current_bridge_keep
    if extra > 0:
        remaining_scores = bridge_score.clone()
        remaining_scores[bridge_mask.bool()] = -torch.inf
        extra_mask = _topk_binary(remaining_scores, extra)
        bridge_mask = torch.maximum(bridge_mask, extra_mask.to(bridge_mask.dtype))

    masks[bridge_name] = bridge_mask
    selected = int(sum(mask.sum().item() for mask in masks.values()))
    remaining_keep = max(0, total_keep - selected)

    if remaining_keep > 0:
        names = []
        values = []
        indices = []
        for name, score in scores.items():
            available = masks[name].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                names.append(name)
                indices.append(idx)
                values.append(score.flatten()[idx])
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining_keep, flat_values.numel()), largest=True).values.min()
        left = remaining_keep
        for name, idx, vals in zip(names, indices, values):
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat_mask = masks[name].flatten()
            flat_mask[chosen] = 1
            masks[name] = flat_mask.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break

    # Exact correction for ties if we overshot the target.
    current_keep = int(sum(mask.sum().item() for mask in masks.values()))
    excess = current_keep - total_keep
    if excess > 0:
        removable_chunks = []
        for name, mask in masks.items():
            removable = mask.bool()
            if name == bridge_name:
                removable = removable & ~protected
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                removable_chunks.append((name, idx, scores[name].flatten()[idx]))
        vals = torch.cat([chunk[2] for chunk in removable_chunks])
        threshold = torch.topk(vals, excess, largest=False).values.max()
        left = excess
        for name, idx, chunk_vals in removable_chunks:
            chosen = idx[chunk_vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat_mask = masks[name].flatten()
            flat_mask[chosen] = 0
            masks[name] = flat_mask.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break

    return masks
