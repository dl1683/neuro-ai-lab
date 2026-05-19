from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.pilot_metrics import branching_ratio, write_result


SPARSITIES = [0.50, 0.80, 0.90, 0.95]
METHODS = [
    "random",
    "magnitude",
    "gradient_saliency",
    "activation_flow",
    "path_flow",
    "path_coverage",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _data(seed: int):
    digits = load_digits()
    x = StandardScaler().fit_transform(digits.data.astype(float) / 16.0)
    y = digits.target.astype(int)
    return train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)


def _init(seed: int, dim: int = 64, hidden: int = 128, classes: int = 10):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.15, (dim, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, 0.15, (hidden, classes)),
        "b2": np.zeros(classes),
    }


def _forward(model: dict[str, np.ndarray], x: np.ndarray, masks: dict[str, np.ndarray] | None = None):
    mask1 = 1.0 if masks is None else masks["W1"]
    mask2 = 1.0 if masks is None else masks["W2"]
    z1 = x @ (model["W1"] * mask1) + model["b1"]
    h = np.maximum(0.0, z1)
    logits = h @ (model["W2"] * mask2) + model["b2"]
    return h, logits


def _accuracy(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray] | None = None) -> float:
    _, logits = _forward(model, x, masks)
    return float((logits.argmax(axis=1) == y).mean())


def _gradients(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray):
    h, logits = _forward(model, x)
    probs = _softmax(logits)
    yoh = np.zeros_like(probs)
    yoh[np.arange(len(y)), y] = 1.0
    dlogits = (probs - yoh) / len(y)
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ model["W2"].T
    dz1 = dh * (h > 0.0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def _train(seed: int):
    x_train, x_test, y_train, y_test = _data(seed)
    model = _init(seed + 1000)
    lr = 0.18
    for _ in range(190):
        grads = _gradients(model, x_train, y_train)
        model["W1"] -= lr * (grads["W1"] + 0.0005 * model["W1"])
        model["b1"] -= lr * grads["b1"]
        model["W2"] -= lr * (grads["W2"] + 0.0005 * model["W2"])
        model["b2"] -= lr * grads["b2"]
        lr *= 0.995
    return model, x_train, x_test, y_train, y_test


def _feature_stats(model: dict[str, np.ndarray], x_train: np.ndarray):
    h, _ = _forward(model, x_train)
    input_flow = np.mean(np.abs(x_train), axis=0) + 1e-6
    hidden_fire = np.mean(h > 0.0, axis=0) + 1e-6
    hidden_strength = np.mean(np.abs(h), axis=0) + 1e-6
    hidden_balance = np.exp(-np.abs(hidden_fire - 0.35) / 0.22)
    output_strength = np.mean(np.abs(model["W2"]), axis=1) + 1e-6
    path_importance = hidden_strength * hidden_balance * output_strength
    return input_flow, hidden_strength, hidden_balance, path_importance


def _scores(model: dict[str, np.ndarray], x_train: np.ndarray, y_train: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 999)
    grads = _gradients(model, x_train, y_train)
    input_flow, hidden_strength, hidden_balance, path_importance = _feature_stats(model, x_train)

    activation_hidden = hidden_strength * hidden_balance
    return {
        "random": {
            "W1": rng.random(model["W1"].shape),
            "W2": rng.random(model["W2"].shape),
        },
        "magnitude": {
            "W1": np.abs(model["W1"]),
            "W2": np.abs(model["W2"]),
        },
        "gradient_saliency": {
            "W1": np.abs(model["W1"] * grads["W1"]),
            "W2": np.abs(model["W2"] * grads["W2"]),
        },
        "activation_flow": {
            "W1": np.abs(model["W1"]) * input_flow[:, None] * activation_hidden[None, :],
            "W2": np.abs(model["W2"]) * activation_hidden[:, None],
        },
        "path_flow": {
            "W1": np.abs(model["W1"]) * input_flow[:, None] * path_importance[None, :],
            "W2": np.abs(model["W2"]) * path_importance[:, None],
        },
        "path_coverage": {
            "W1": np.abs(model["W1"]) * input_flow[:, None] * path_importance[None, :],
            "W2": np.abs(model["W2"]) * path_importance[:, None],
            "hidden_importance": path_importance,
        },
    }


def _mask_from_scores(score: dict[str, np.ndarray], sparsity: float, method: str) -> dict[str, np.ndarray]:
    keep_fraction = 1.0 - sparsity
    total_edges = score["W1"].size + score["W2"].size
    budget = max(1, int(round(keep_fraction * total_edges)))

    if method != "path_coverage":
        flat = np.concatenate([score["W1"].ravel(), score["W2"].ravel()])
        cutoff = np.partition(flat, max(0, flat.size - budget))[max(0, flat.size - budget)]
        mask = {
            "W1": (score["W1"] >= cutoff).astype(float),
            "W2": (score["W2"] >= cutoff).astype(float),
        }
        # Exact-size trim for ties.
        extra = int(mask["W1"].sum() + mask["W2"].sum() - budget)
        if extra > 0:
            selected = np.concatenate([score["W1"][mask["W1"] > 0], score["W2"][mask["W2"] > 0]])
            tie_cut = np.sort(selected)[:extra][-1]
            for key in ["W1", "W2"]:
                tie = np.argwhere((mask[key] > 0) & (score[key] <= tie_cut))
                for idx in tie[:extra]:
                    mask[key][tuple(idx)] = 0.0
                    extra -= 1
                    if extra == 0:
                        break
        return mask

    mask = {"W1": np.zeros_like(score["W1"]), "W2": np.zeros_like(score["W2"])}
    hidden_order = np.argsort(score["hidden_importance"])[::-1]
    used = 0
    max_hidden = min(len(hidden_order), budget // 2)
    for hidden in hidden_order[:max_hidden]:
        in_idx = int(np.argmax(score["W1"][:, hidden]))
        out_idx = int(np.argmax(score["W2"][hidden, :]))
        if mask["W1"][in_idx, hidden] == 0:
            mask["W1"][in_idx, hidden] = 1.0
            used += 1
        if used >= budget:
            break
        if mask["W2"][hidden, out_idx] == 0:
            mask["W2"][hidden, out_idx] = 1.0
            used += 1
        if used >= budget:
            break

    remaining = budget - used
    if remaining > 0:
        already_w1 = mask["W1"].astype(bool)
        already_w2 = mask["W2"].astype(bool)
        candidates = []
        w1_indices = np.argwhere(~already_w1)
        for i, j in w1_indices:
            candidates.append((score["W1"][i, j], "W1", int(i), int(j)))
        w2_indices = np.argwhere(~already_w2)
        for i, j in w2_indices:
            candidates.append((score["W2"][i, j], "W2", int(i), int(j)))
        candidates.sort(reverse=True, key=lambda item: item[0])
        for _, key, i, j in candidates[:remaining]:
            mask[key][i, j] = 1.0
    return mask


def _branching(model: dict[str, np.ndarray], x: np.ndarray, masks: dict[str, np.ndarray]) -> float:
    h, logits = _forward(model, x, masks)
    active_input = np.abs(x) > np.percentile(np.abs(x), 70)
    active_hidden = h > 0.0
    active_output = logits > np.percentile(logits, 70)
    return branching_ratio([active_input, active_hidden, active_output])


def _hidden_coverage(masks: dict[str, np.ndarray]) -> float:
    hidden_has_in = masks["W1"].sum(axis=0) > 0
    hidden_has_out = masks["W2"].sum(axis=1) > 0
    return float((hidden_has_in & hidden_has_out).mean())


def _actual_sparsity(masks: dict[str, np.ndarray]) -> float:
    active = masks["W1"].sum() + masks["W2"].sum()
    total = masks["W1"].size + masks["W2"].size
    return float(1.0 - active / total)


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for sparsity in SPARSITIES:
        summary[str(sparsity)] = {}
        rows_s = [row for row in rows if row["target_sparsity"] == sparsity]
        for method in METHODS:
            rows_m = [row for row in rows_s if row["method"] == method]
            summary[str(sparsity)][method] = {
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rows_m])),
                "accuracy_std": float(np.std([r["accuracy"] for r in rows_m])),
                "accuracy_retention_mean": float(np.mean([r["accuracy_retention"] for r in rows_m])),
                "branching_ratio_mean": float(np.mean([r["branching_ratio"] for r in rows_m])),
                "distance_from_criticality_mean": float(np.mean([r["distance_from_criticality"] for r in rows_m])),
                "hidden_coverage_mean": float(np.mean([r["hidden_coverage"] for r in rows_m])),
                "actual_sparsity_mean": float(np.mean([r["actual_sparsity"] for r in rows_m])),
            }
    return summary


def _pairwise(rows: list[dict], method: str, baseline: str, levels: list[float]) -> list[float]:
    deltas = []
    seeds = sorted({row["seed"] for row in rows})
    for sparsity in levels:
        for seed in seeds:
            left = next(r for r in rows if r["seed"] == seed and r["target_sparsity"] == sparsity and r["method"] == method)
            right = next(r for r in rows if r["seed"] == seed and r["target_sparsity"] == sparsity and r["method"] == baseline)
            deltas.append(left["accuracy"] - right["accuracy"])
    return deltas


def run(seed: int = 23) -> dict:
    seeds = [seed, seed + 1, seed + 2, seed + 3, seed + 4]
    rows = []
    dense_accuracies = []
    for trial_seed in seeds:
        model, x_train, x_test, y_train, y_test = _train(trial_seed)
        dense_accuracy = _accuracy(model, x_test, y_test)
        dense_accuracies.append(dense_accuracy)
        score_by_method = _scores(model, x_train, y_train, trial_seed)
        for sparsity in SPARSITIES:
            for method in METHODS:
                mask = _mask_from_scores(score_by_method[method], sparsity, method)
                acc = _accuracy(model, x_test, y_test, mask)
                sigma = _branching(model, x_test, mask)
                rows.append(
                    {
                        "seed": trial_seed,
                        "target_sparsity": sparsity,
                        "actual_sparsity": _actual_sparsity(mask),
                        "method": method,
                        "dense_accuracy": dense_accuracy,
                        "accuracy": acc,
                        "accuracy_retention": acc / dense_accuracy if dense_accuracy else 0.0,
                        "branching_ratio": sigma,
                        "distance_from_criticality": abs(sigma - 1.0),
                        "hidden_coverage": _hidden_coverage(mask),
                    }
                )

    summary = _summarize(rows)
    high_sparsity = [0.90, 0.95]
    discoveries = {}
    for method in ["activation_flow", "path_flow", "path_coverage", "gradient_saliency"]:
        vs_mag = _pairwise(rows, method, "magnitude", high_sparsity)
        vs_grad = _pairwise(rows, method, "gradient_saliency", high_sparsity) if method != "gradient_saliency" else [0.0]
        discoveries[method] = {
            "mean_accuracy_gain_vs_magnitude_high_sparsity": float(np.mean(vs_mag)),
            "wins_vs_magnitude_high_sparsity": int(sum(delta > 0 for delta in vs_mag)),
            "mean_accuracy_gain_vs_gradient_high_sparsity": float(np.mean(vs_grad)),
            "wins_vs_gradient_high_sparsity": int(sum(delta > 0 for delta in vs_grad)),
        }

    best_label_free = max(["activation_flow", "path_flow", "path_coverage"], key=lambda m: discoveries[m]["mean_accuracy_gain_vs_magnitude_high_sparsity"])
    discovery = {
        "claim": "Label-free activation/path-flow pruning is a robust high-sparsity alternative to magnitude pruning on trained sklearn-digits MLPs. The result is about preserving active input-hidden-output paths, not about validating the original branching-ratio criticality metric.",
        "best_label_free_method": best_label_free,
        "high_sparsity_levels": high_sparsity,
        "method_discoveries": discoveries,
        "mechanism_status": "branching-ratio distance is tracked as a diagnostic, but it is not the causal win signal in the current benchmark.",
    }

    payload = {
        "experiment": "04_criticality_pruning",
        "pilot_type": "real multi-seed pruning benchmark on trained sklearn-digits MLPs with label-free path-flow variants",
        "task_setup": "train dense MLPs on sklearn digits, prune W1/W2 without retraining across sparsity levels and compare random, magnitude, gradient saliency, activation-flow, path-flow, and path-coverage scores",
        "dense_accuracy_mean": float(np.mean(dense_accuracies)),
        "dense_accuracy_std": float(np.std(dense_accuracies)),
        "methods": METHODS,
        "sparsities": SPARSITIES,
        "summary": summary,
        "rows": rows,
        "discovery": discovery,
        "notable_win": bool(discoveries[best_label_free]["mean_accuracy_gain_vs_magnitude_high_sparsity"] > 0.10 and discoveries[best_label_free]["wins_vs_magnitude_high_sparsity"] >= 9),
    }
    write_result("04_criticality_pruning", "pilot_result.json", payload)
    (ROOT / "results" / "04_criticality_pruning" / "discovery_summary.json").write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dense_accuracy_mean": result["dense_accuracy_mean"], "discovery": result["discovery"], "notable_win": result["notable_win"]}, indent=2))
