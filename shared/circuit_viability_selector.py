from __future__ import annotations

from dataclasses import asdict

import numpy as np

from shared.residual_route_capacity import RouteFamilySplit, route_split_capacity_mask


def candidate_route_splits(
    main_min: float = 0.30,
    main_max: float = 0.55,
    projection_min: float = 0.10,
    projection_max: float = 0.40,
    readout_min: float = 0.20,
    readout_max: float = 0.55,
    step: float = 0.05,
) -> list[RouteFamilySplit]:
    splits = []
    for main in np.arange(main_min, main_max + 0.001, step):
        for projection in np.arange(projection_min, projection_max + 0.001, step):
            readout = 1.0 - float(main) - float(projection)
            if readout_min <= readout <= readout_max:
                splits.append(RouteFamilySplit(main=float(main), projection=float(projection), readout=float(readout)))
    return splits


def conservative_route_score(quality: dict, plain_quality: dict, magnitude_quality: dict) -> float:
    main_floor = 0.84 * plain_quality["main_path_min"]
    projection_floor = 0.90 * plain_quality["projection_min"]
    readout_target = max(plain_quality["fc_score"], 0.66 * magnitude_quality["fc_score"])
    main_ratio = quality["main_path_min"] / max(main_floor, 1e-8)
    projection_ratio = quality["projection_min"] / max(projection_floor, 1e-8)
    readout_ratio = quality["fc_score"] / max(readout_target, 1e-8)
    floor_score = min(main_ratio, projection_ratio, readout_ratio)
    route_balance = 0.30 * min(main_ratio, 1.5) + 0.10 * min(projection_ratio, 1.5) + 0.18 * min(readout_ratio, 1.5)
    dead_penalty = 0.002 * quality["total_dead_outputs"]
    return floor_score + route_balance - dead_penalty


def split_dict(split: RouteFamilySplit) -> dict:
    return asdict(split)


def choose_conservative_route_split(
    syn_scores: dict,
    mag_scores: dict,
    sparsity: float,
    reserve: float,
    capacity_mask_fn,
    global_mask_fn,
    route_quality_fn,
) -> tuple[dict, list[dict]]:
    plain_mask = capacity_mask_fn(syn_scores, mag_scores, sparsity, reserve)
    mag_mask = global_mask_fn(mag_scores, sparsity)
    plain_quality = route_quality_fn(plain_mask)
    magnitude_quality = route_quality_fn(mag_mask)
    scored = []
    for split in candidate_route_splits():
        masks = route_split_capacity_mask(syn_scores, mag_scores, sparsity, reserve, split)
        quality = route_quality_fn(masks)
        scored.append(
            {
                "split": split,
                "split_dict": split_dict(split),
                "score": conservative_route_score(quality, plain_quality, magnitude_quality),
                "quality": quality,
                "plain_quality": plain_quality,
                "magnitude_quality": magnitude_quality,
            }
        )
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    return ranked[0], ranked[:8]


def choose_ecology_aware_method(
    syn_scores: dict,
    mag_scores: dict,
    sparsity: float,
    reserve: float,
    capacity_mask_fn,
    global_mask_fn,
    route_quality_fn,
    readout_ratio_threshold: float = 0.66,
) -> dict:
    best_split, top_splits = choose_conservative_route_split(
        syn_scores,
        mag_scores,
        sparsity,
        reserve,
        capacity_mask_fn,
        global_mask_fn,
        route_quality_fn,
    )
    plain_mask = capacity_mask_fn(syn_scores, mag_scores, sparsity, reserve)
    mag_mask = global_mask_fn(mag_scores, sparsity)
    plain_quality = route_quality_fn(plain_mask)
    magnitude_quality = route_quality_fn(mag_mask)
    readout_ratio = plain_quality["fc_score"] / max(magnitude_quality["fc_score"], 1e-8)
    if readout_ratio < readout_ratio_threshold:
        selected = "predicted_route_split"
        selected_split = best_split["split"]
        selected_mask = route_split_capacity_mask(syn_scores, mag_scores, sparsity, reserve, selected_split)
    else:
        selected = "plain_reserve"
        selected_split = None
        selected_mask = plain_mask
    return {
        "selected_method": selected,
        "selected_split": None if selected_split is None else split_dict(selected_split),
        "selected_mask": selected_mask,
        "plain_quality": plain_quality,
        "magnitude_quality": magnitude_quality,
        "plain_readout_ratio": readout_ratio,
        "readout_ratio_threshold": readout_ratio_threshold,
        "best_split": best_split,
        "top_splits": top_splits,
    }


def choose_unified_viability_family(
    *,
    magnitude_quality: dict,
    route_floor_threshold: float = 0.50,
) -> dict:
    """Choose the high-level viability family from pre-finetune route diagnostics.

    If magnitude already has a live route floor, as in pretrained networks, the
    safest intervention is feature-preserving liveness repair. If magnitude has
    a dead or near-dead route floor, as in the from-scratch severe-pruning
    failures, use the homeostatic/ecology selector.
    """

    route_floor = float(magnitude_quality.get("main_path_min", 0.0))
    if route_floor >= route_floor_threshold:
        family = "feature_viability_repair"
    else:
        family = "ecology_selector"
    return {
        "selected_family": family,
        "magnitude_route_floor": route_floor,
        "route_floor_threshold": route_floor_threshold,
    }


def _mask_keep_count(masks: dict) -> int:
    return int(sum(mask.detach().sum().item() for mask in masks.values()))


def _mask_overlap_fraction(candidate: dict, reference: dict) -> float:
    overlap = 0
    keep = 0
    for name, mask in candidate.items():
        if name not in reference:
            continue
        cand = mask.detach() > 0
        ref = reference[name].detach() > 0
        overlap += int((cand & ref).sum().item())
        keep += int(cand.sum().item())
    return float(overlap / max(keep, 1))


def _quality_value(quality: dict, key: str) -> float:
    return float(quality.get(key, 0.0) or 0.0)


def tradeoff_viability_score(
    *,
    method: str,
    quality: dict,
    candidate_mask: dict,
    magnitude_mask: dict,
    candidate_qualities: dict[str, dict],
    feature_weight: float | None = None,
    liveness_weight: float | None = None,
    readout_weight: float | None = None,
    dead_weight: float = 0.06,
) -> dict:
    """Score a sparse mask as a feature-preservation / liveness tradeoff.

    The older unified selector treated a dead magnitude route floor as a hard
    reason to choose broad homeostatic repair. The current evidence is more
    subtle: a magnitude-like mask can preserve useful computation even when it
    leaves some dead outputs, while broad reserve can over-rewrite learned
    feature subspaces. This score therefore keeps feature overlap load-bearing
    while still rewarding route floors and penalizing dead outputs.
    """

    feature_overlap = _mask_overlap_fraction(candidate_mask, magnitude_mask)
    magnitude_quality = candidate_qualities.get("magnitude", {})
    magnitude_fc = _quality_value(magnitude_quality, "fc_score")
    magnitude_dead = _quality_value(magnitude_quality, "total_dead_outputs")
    readout_pressure = min(max((3.55 - magnitude_fc) / 0.70, 0.0), 1.0)
    death_pressure = min(max((magnitude_dead - 500.0) / 250.0, 0.0), 1.0)
    ecology_pressure = max(readout_pressure, death_pressure)
    if feature_weight is None:
        feature_weight = 0.58 - 0.45 * ecology_pressure
    if liveness_weight is None:
        liveness_weight = 0.34 + 0.30 * ecology_pressure
    if readout_weight is None:
        readout_weight = 0.16 + 0.15 * ecology_pressure
    main_values = [_quality_value(item, "main_path_min") for item in candidate_qualities.values()]
    projection_values = [_quality_value(item, "projection_min") for item in candidate_qualities.values()]
    readout_values = [_quality_value(item, "fc_score") for item in candidate_qualities.values()]
    dead_values = [_quality_value(item, "total_dead_outputs") for item in candidate_qualities.values()]
    main_target = max(main_values + [1e-8])
    projection_target = max(projection_values + [1e-8])
    readout_target = max(readout_values + [1e-8])
    dead_target = max(dead_values + [1.0])

    main_ratio = min(_quality_value(quality, "main_path_min") / main_target, 1.0)
    projection_ratio = min(_quality_value(quality, "projection_min") / projection_target, 1.0)
    readout_ratio = min(_quality_value(quality, "fc_score") / readout_target, 1.0)
    liveness_floor = min(main_ratio, projection_ratio)
    liveness_mean = 0.50 * main_ratio + 0.25 * projection_ratio + 0.25 * readout_ratio
    liveness = 0.55 * liveness_floor + 0.45 * liveness_mean
    dead_penalty = min(_quality_value(quality, "total_dead_outputs") / dead_target, 1.0)
    score = feature_weight * feature_overlap + liveness_weight * liveness + readout_weight * readout_ratio - dead_weight * dead_penalty

    return {
        "method": method,
        "score": float(score),
        "feature_overlap_with_magnitude": float(feature_overlap),
        "liveness": float(liveness),
        "liveness_floor": float(liveness_floor),
        "main_ratio": float(main_ratio),
        "projection_ratio": float(projection_ratio),
        "readout_ratio": float(readout_ratio),
        "dead_penalty": float(dead_penalty),
        "ecology_pressure": float(ecology_pressure),
        "feature_weight": float(feature_weight),
        "liveness_weight": float(liveness_weight),
        "readout_weight": float(readout_weight),
        "quality": quality,
        "keep_count": _mask_keep_count(candidate_mask),
    }


def choose_tradeoff_viability_method(
    *,
    candidate_masks: dict[str, dict],
    route_quality_fn,
    magnitude_method: str = "magnitude",
) -> dict:
    """Choose among realized masks by balancing feature preservation and liveness."""

    if magnitude_method not in candidate_masks:
        raise KeyError(f"candidate_masks must include {magnitude_method!r}")
    magnitude_mask = candidate_masks[magnitude_method]
    candidate_qualities = {method: route_quality_fn(mask) for method, mask in candidate_masks.items()}
    scored = [
        tradeoff_viability_score(
            method=method,
            quality=candidate_qualities[method],
            candidate_mask=mask,
            magnitude_mask=magnitude_mask,
            candidate_qualities=candidate_qualities,
        )
        for method, mask in candidate_masks.items()
    ]
    ranked = sorted(scored, key=lambda item: (item["score"], item["feature_overlap_with_magnitude"]), reverse=True)
    selected = ranked[0]["method"]
    return {
        "selected_method": selected,
        "selected_mask": candidate_masks[selected],
        "selected_quality": candidate_qualities[selected],
        "ranked_methods": ranked,
        "candidate_qualities": candidate_qualities,
    }
