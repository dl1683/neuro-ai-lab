"""Lightweight metrics and reporting helpers for fast neuro-AI pilots.

The functions here are intentionally small and dependency-light. They let each
experiment produce an executable "first evidence" artifact before expensive
GPU-scale training is justified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def result_path(experiment_id: str) -> Path:
    path = repo_root() / "results" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(experiment_id: str, name: str, payload: dict[str, Any]) -> Path:
    path = result_path(experiment_id) / name
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def logistic(x: np.ndarray | float, center: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(x) - center) / scale))


def linear_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if denom == 0.0 else 1.0 - float(np.sum((y - pred) ** 2)) / denom
    return float(slope), float(intercept), float(r2)


def first_crossing(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(np.asarray(values) >= threshold)
    return int(hits[0]) if len(hits) else None


def effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    total = float(singular_values.sum())
    if total <= eps:
        return 0.0
    probs = singular_values / total
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    return float(np.exp(entropy))


def mean_pairwise_cosine(centroids: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    vectors = np.asarray(centroids, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    normalized = vectors / norms
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return float(upper.mean()), float(upper.std())


def branching_ratio(active_by_layer: list[np.ndarray], eps: float = 1e-12) -> float:
    ratios = []
    for left, right in zip(active_by_layer[:-1], active_by_layer[1:]):
        left_count = np.asarray(left).sum(axis=1)
        right_count = np.asarray(right).sum(axis=1)
        ratios.extend((right_count / np.maximum(left_count, eps)).tolist())
    return float(np.mean(ratios)) if ratios else 0.0


def synthetic_backbone_degree(weights: np.ndarray, mask: np.ndarray) -> float:
    active = np.abs(weights) * mask
    return float((active > 0).sum(axis=1).mean())


def summary_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(metrics[key] * weight for key, weight in weights.items()))

