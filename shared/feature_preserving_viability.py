from __future__ import annotations

import torch


def topk_binary(score: torch.Tensor, keep: int) -> torch.Tensor:
    flat = score.detach().flatten()
    keep = max(0, min(int(keep), flat.numel()))
    if keep == 0:
        return torch.zeros_like(score)
    threshold = torch.topk(flat, keep, largest=True).values.min()
    return (score >= threshold).to(score.dtype)


def magnitude_with_liveness_repair(magnitude: dict, sparsity: float, skip_names: set[str] | None = None) -> dict:
    """Start from global magnitude and minimally repair dead output rows.

    This is designed for pretrained models where magnitude preserves useful
    feature subspaces better than aggressive homeostatic reallocation. The mask
    keeps the global parameter budget fixed, repairs dead output rows using
    within-row magnitude, and removes the weakest non-protected kept weights to
    pay for those repairs.
    """

    skip = skip_names or set()
    total_params = sum(score.numel() for score in magnitude.values())
    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    flat = torch.cat([score.detach().flatten() for score in magnitude.values()])
    threshold = torch.topk(flat, total_keep, largest=True).values.min()
    masks = {name: (score >= threshold).to(score.dtype) for name, score in magnitude.items()}
    protected = {name: torch.zeros_like(score, dtype=torch.bool) for name, score in magnitude.items()}

    for name, score in magnitude.items():
        if name in skip or score.ndim < 2:
            continue
        layer_mask = masks[name].clone()
        flat_outputs = layer_mask.reshape(layer_mask.shape[0], -1)
        flat_score = score.reshape(score.shape[0], -1)
        flat_protected = protected[name].reshape(layer_mask.shape[0], -1)
        for row in range(flat_outputs.shape[0]):
            if int(flat_outputs[row].sum().item()) == 0:
                idx = torch.argmax(flat_score[row])
                flat_outputs[row, idx] = 1
                flat_protected[row, idx] = True
        masks[name] = flat_outputs.reshape_as(layer_mask)
        protected[name] = flat_protected.reshape_as(layer_mask)

    selected = int(sum(mask.sum().item() for mask in masks.values()))
    excess = selected - total_keep
    if excess <= 0:
        return masks

    removable_scores = []
    chunks = []
    for name, score in magnitude.items():
        removable = (masks[name].flatten() > 0) & (~protected[name].flatten())
        if removable.any():
            idx = torch.nonzero(removable, as_tuple=False).flatten()
            vals = score.flatten()[idx]
            chunks.append((name, idx, vals))
            removable_scores.append(vals)
    if not removable_scores:
        return masks
    all_vals = torch.cat(removable_scores)
    remove_count = min(excess, all_vals.numel())
    cutoff = torch.topk(all_vals, remove_count, largest=False).values.max()
    left = remove_count
    for name, idx, vals in chunks:
        chosen = idx[vals <= cutoff]
        if chosen.numel() > left:
            chosen = chosen[:left]
        flat_mask = masks[name].flatten()
        flat_mask[chosen] = 0
        masks[name] = flat_mask.reshape_as(masks[name])
        left -= int(chosen.numel())
        if left <= 0:
            break
    return masks
