"""Adaptive path pruning utilities.

This module contains the reusable core behind the experiment-04 discovery:
path-aware pruning is most useful as a sparsity-cliff correction for dense
path matrices. The intended use is:

1. Compute standard magnitude scores.
2. Estimate input and hidden activation statistics on a small unlabeled
   calibration batch.
3. Apply a sparsity-dependent weak path modulation to dense layers.
4. Keep convolutional layers on magnitude unless a separate conv-specific
   path score is developed.

The utilities are deliberately small and framework-light beyond PyTorch. They
are not a full pruning framework; they are the scoring/masking kernel that the
experiment scripts can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class PathPruningConfig:
    """Configuration for dense path correction.

    alpha controls how strongly the path signal modulates magnitude. alpha=0 is
    exactly magnitude pruning. The default objective is deliberately
    conservative: use weak path modulation near the severe sparsity cliff rather
    than aggressive path weighting. Current CNN evidence shows this can preserve
    the one-shot benefit while retaining magnitude-level fine-tuning recovery.
    """

    sparsity: float
    alpha: float | None = None
    objective: str = "balanced"
    eps: float = 1e-6

    def resolved_alpha(self) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        return adaptive_alpha(self.sparsity, objective=self.objective)


def adaptive_alpha(sparsity: float, objective: str = "balanced") -> float:
    """Empirical alpha schedule from current image-model experiments.

    Supported objectives:

    - "balanced": tiny path correction intended mainly to improve one-shot
      severe pruning while keeping the recovery cost small.
    - "one_shot": tiny path correction for no-recovery pruning; sweep this
      value when possible because transfer tests show model-specific optima.
    - "recovery": no default correction. Current CIFAR evidence says magnitude
      remains the safest fine-tuning initializer unless a domain-specific sweep
      proves otherwise.
    """

    if objective not in {"balanced", "one_shot", "recovery"}:
        raise ValueError(f"unknown objective {objective!r}")
    if sparsity < 0.925:
        return 0.0
    if objective == "recovery":
        return 0.0
    if objective == "one_shot":
        return 0.03
    return 0.03


def dense_path_scores(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    input_signal: torch.Tensor,
    hidden_strength: torch.Tensor,
    config: PathPruningConfig,
) -> TensorDict:
    """Score a two-layer dense path `input -> hidden -> output`.

    Args:
        fc1_weight: Shape [hidden, input].
        fc2_weight: Shape [output, hidden].
        input_signal: Shape [input]. Recommended signal for normalized images is
            per-feature standard deviation, not mean absolute activation.
        hidden_strength: Shape [hidden]. Mean hidden activation magnitude on an
            unlabeled calibration set.
        config: Sparsity/alpha configuration.

    Returns:
        Scores keyed as `fc1` and `fc2` with the same shapes as the weights.
    """

    alpha = config.resolved_alpha()
    eps = config.eps
    input_signal = input_signal.to(fc1_weight.device).clamp_min(eps)
    hidden_strength = hidden_strength.to(fc1_weight.device).clamp_min(eps)
    output_strength = fc2_weight.detach().abs().mean(dim=0).clamp_min(eps)
    hidden_path = hidden_strength * output_strength

    return {
        "fc1": fc1_weight.detach().abs() * torch.pow(input_signal[None, :] * hidden_path[:, None], alpha),
        "fc2": fc2_weight.detach().abs() * torch.pow(hidden_path[None, :], alpha),
    }


def magnitude_scores(weights: Mapping[str, torch.Tensor]) -> TensorDict:
    """Return absolute-value scores for a named weight mapping."""

    return {name: weight.detach().abs().clone() for name, weight in weights.items()}


def merge_scores(*score_maps: Mapping[str, torch.Tensor]) -> TensorDict:
    """Merge score dictionaries, with later maps overriding earlier maps."""

    merged: TensorDict = {}
    for score_map in score_maps:
        merged.update({key: value for key, value in score_map.items()})
    return merged


def global_topk_mask(scores: Mapping[str, torch.Tensor], sparsity: float) -> TensorDict:
    """Create a global top-k binary mask from score tensors.

    The mask keeps approximately `(1 - sparsity)` of all scored parameters.
    """

    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    flat = torch.cat([score.detach().flatten() for score in scores.values()])
    keep = max(1, int(round((1.0 - sparsity) * flat.numel())))
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return {name: (score >= threshold).to(score.dtype) for name, score in scores.items()}


def apply_masks_in_place(weights: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]) -> TensorDict:
    """Apply masks in place and return cloned originals for later restoration."""

    originals = {name: weight.data.clone() for name, weight in weights.items()}
    for name, weight in weights.items():
        weight.data.mul_(masks[name].to(weight.device))
    return originals


def restore_weights_in_place(weights: Mapping[str, torch.Tensor], originals: Mapping[str, torch.Tensor]) -> None:
    """Restore tensors saved by apply_masks_in_place."""

    for name, weight in weights.items():
        weight.data.copy_(originals[name].to(weight.device))


def dense_calibration_stats(flat_inputs: torch.Tensor, hidden_activations: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute recommended calibration statistics for dense path scoring.

    For normalized image inputs, per-feature standard deviation is more reliable
    than mean absolute value because blank/background pixels can become large
    negative values after normalization.
    """

    input_signal = flat_inputs.detach().std(dim=0).clamp_min(eps)
    hidden_strength = hidden_activations.detach().abs().mean(dim=0).clamp_min(eps)
    return input_signal, hidden_strength


def score_dense_tail_with_magnitude_convs(
    conv_weights: Mapping[str, torch.Tensor],
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    flat_inputs: torch.Tensor,
    hidden_activations: torch.Tensor,
    config: PathPruningConfig,
) -> TensorDict:
    """Hybrid CNN rule: magnitude for conv weights, path correction for dense tail."""

    input_signal, hidden_strength = dense_calibration_stats(flat_inputs, hidden_activations, eps=config.eps)
    return merge_scores(
        magnitude_scores(conv_weights),
        dense_path_scores(fc1_weight, fc2_weight, input_signal, hidden_strength, config),
    )
