"""Residual route-capacity allocation utilities.

This module turns the TinyResNet route-quality experiments into a reusable method
surface. It predicts protected-capacity shares for residual networks from route
family deficits before recovery fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class RouteFamilySplit:
    main: float
    projection: float
    readout: float

    def normalized(self) -> "RouteFamilySplit":
        total = self.main + self.projection + self.readout
        if total <= 0:
            return RouteFamilySplit(main=1 / 3, projection=1 / 3, readout=1 / 3)
        return RouteFamilySplit(main=self.main / total, projection=self.projection / total, readout=self.readout / total)


@dataclass(frozen=True)
class RouteDeficitPrediction:
    split: RouteFamilySplit
    projection_deficit: float
    readout_deficit: float
    template_projection: float
    candidate_projection: float
    template_readout: float
    candidate_readout: float
    main_floor: float
    projection_weight: float


def route_family(name: str) -> str:
    if name == "fc.weight" or name.endswith("fc.weight"):
        return "readout"
    if ".shortcut.0.weight" in name or ".downsample.0.weight" in name:
        return "projection"
    return "main"


def output_units(tensor: torch.Tensor) -> int:
    return int(tensor.shape[0]) if tensor.ndim >= 2 else 1


def topk_binary(score: torch.Tensor, keep: int) -> torch.Tensor:
    flat = score.detach().flatten()
    keep = max(0, min(int(keep), flat.numel()))
    if keep == 0:
        return torch.zeros_like(score)
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return (score >= threshold).to(score.dtype)


def predict_route_split(
    *,
    template_projection: float,
    candidate_projection: float,
    template_readout: float,
    candidate_readout: float,
    main_floor: float = 0.40,
    projection_weight: float = 2.0,
    min_projection: float = 0.20,
    max_projection: float = 0.40,
    min_readout: float = 0.20,
    max_readout: float = 0.35,
) -> RouteDeficitPrediction:
    """Predict main/projection/readout capacity split from route deficits.

    The predictor compares a viability template, currently usually a magnitude
    mask, against a candidate capacity mask. It keeps a main-path floor and
    allocates the remaining protected budget to projection/readout families
    according to measured deficits.
    """

    projection_deficit = max(1e-6, float(template_projection) - float(candidate_projection))
    readout_deficit = max(1e-6, float(template_readout) - float(candidate_readout))
    remaining = max(0.0, 1.0 - main_floor)
    projection_signal = projection_weight * projection_deficit
    readout_signal = readout_deficit
    projection = remaining * projection_signal / (projection_signal + readout_signal)
    readout = remaining - projection
    projection = min(max_projection, max(min_projection, projection))
    readout = min(max_readout, max(min_readout, readout))
    main = 1.0 - projection - readout
    split = RouteFamilySplit(main=main, projection=projection, readout=readout).normalized()
    return RouteDeficitPrediction(
        split=split,
        projection_deficit=projection_deficit,
        readout_deficit=readout_deficit,
        template_projection=float(template_projection),
        candidate_projection=float(candidate_projection),
        template_readout=float(template_readout),
        candidate_readout=float(candidate_readout),
        main_floor=main_floor,
        projection_weight=projection_weight,
    )


def rank_scores_for_route(name: str, synflow: TensorDict, magnitude: TensorDict) -> torch.Tensor:
    family = route_family(name)
    if family in {"projection", "readout"}:
        return magnitude[name]
    return synflow[name]


def route_split_capacity_mask(
    synflow: TensorDict,
    magnitude: TensorDict,
    sparsity: float,
    reserve_fraction: float,
    split: RouteFamilySplit,
    skip_names: set[str] | None = None,
) -> TensorDict:
    """Build a residual route-family capacity mask.

    The mask reserves a fraction of the global keep budget across route families,
    enforces one live input per output unit for prunable tensors, then fills the
    remaining budget by global SynFlow score.
    """

    skip = skip_names or {"stem.weight"}
    split = split.normalized()
    total_params = sum(score.numel() for score in synflow.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    reserve_keep = int(round(reserve_fraction * total_keep))
    masks = {name: torch.zeros_like(score) for name, score in synflow.items()}
    protected = {name: torch.zeros_like(score, dtype=torch.bool) for name, score in synflow.items()}
    critical = [name for name in synflow if name not in skip]
    groups = {
        "main": [name for name in critical if route_family(name) == "main"],
        "projection": [name for name in critical if route_family(name) == "projection"],
        "readout": [name for name in critical if route_family(name) == "readout"],
    }
    fractions = {"main": split.main, "projection": split.projection, "readout": split.readout}

    for family, names in groups.items():
        if not names:
            continue
        group_budget = int(round(reserve_keep * fractions[family]))
        weights = {name: output_units(synflow[name]) for name in names}
        total_weight = sum(weights.values())
        for name in names:
            budget = max(output_units(synflow[name]), int(round(group_budget * weights[name] / max(1, total_weight))))
            masks[name] = torch.maximum(masks[name], topk_binary(rank_scores_for_route(name, synflow, magnitude), budget))

    for name in critical:
        rank = rank_scores_for_route(name, synflow, magnitude)
        layer_mask = masks[name].clone()
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_rank = rank.reshape(rank.shape[0], -1)
        flat_protected = protected[name].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_rank[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[name] = flat_outputs.reshape_as(layer_mask)
        protected[name] = flat_protected.reshape_as(layer_mask)

    selected = int(sum(mask.sum().item() for mask in masks.values()))
    remaining = total_keep - selected
    if remaining > 0:
        chunks = []
        values = []
        for name, score in synflow.items():
            available = masks[name].flatten() == 0
            if available.any():
                idx = torch.nonzero(available, as_tuple=False).flatten()
                vals = score.flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, min(remaining, flat_values.numel()), largest=True).values.min()
        left = remaining
        for name, idx, vals in chunks:
            chosen = idx[vals >= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 1
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    elif remaining < 0:
        excess = -remaining
        chunks = []
        values = []
        for name in synflow:
            removable = masks[name].bool() & ~protected[name]
            idx = torch.nonzero(removable.flatten(), as_tuple=False).flatten()
            if idx.numel() > 0:
                vals = rank_scores_for_route(name, synflow, magnitude).flatten()[idx]
                chunks.append((name, idx, vals))
                values.append(vals)
        flat_values = torch.cat(values)
        threshold = torch.topk(flat_values, excess, largest=False).values.max()
        left = excess
        for name, idx, vals in chunks:
            chosen = idx[vals <= threshold]
            if chosen.numel() > left:
                chosen = chosen[:left]
            flat = masks[name].flatten()
            flat[chosen] = 0
            masks[name] = flat.reshape_as(masks[name])
            left -= int(chosen.numel())
            if left <= 0:
                break
    return masks
