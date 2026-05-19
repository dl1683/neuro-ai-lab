from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.pilot_metrics import linear_r2, logistic, write_result


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _make_data(seed: int):
    digits = load_digits()
    mask = digits.target < 5
    x = digits.data[mask].astype(float) / 16.0
    y = digits.target[mask].astype(int)
    x = StandardScaler().fit_transform(x)
    return train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)


def _init_model(seed: int, dim: int, classes: int, depth: int = 10):
    rng = np.random.default_rng(seed)
    return {
        "W": rng.normal(0.0, 0.02, size=(dim, classes)),
        "B": rng.normal(0.0, 0.02, size=(depth, dim, classes)),
    }


def _layer_logits(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    base = x @ model["W"]
    layers = []
    running = np.zeros_like(base)
    for layer in range(model["B"].shape[0]):
        running = running + x @ model["B"][layer]
        layers.append(base + running)
    return np.asarray(layers)


def _train(seed: int, epochs: int = 260):
    x_train, x_test, y_train, y_test = _make_data(seed)
    classes = len(np.unique(y_train))
    model = _init_model(seed + 100, x_train.shape[1], classes)
    one_hot = np.zeros((len(y_train), classes))
    one_hot[np.arange(len(y_train)), y_train] = 1.0
    lr = 0.18
    depth = model["B"].shape[0]
    layer_weights = np.linspace(0.15, 1.0, depth)
    layer_weights = layer_weights / layer_weights.sum()

    for _ in range(epochs):
        logits_by_layer = _layer_logits(model, x_train)
        grad_W = np.zeros_like(model["W"])
        grad_B = np.zeros_like(model["B"])
        for layer, weight in enumerate(layer_weights):
            probs = _softmax(logits_by_layer[layer])
            dlogits = weight * (probs - one_hot) / len(y_train)
            grad_W += x_train.T @ dlogits
            for prior in range(layer + 1):
                grad_B[prior] += x_train.T @ dlogits
        model["W"] -= lr * grad_W
        model["B"] -= lr * grad_B
        lr *= 0.996
    return model, x_test, y_test


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


def _exit_sweep(logits_by_layer: np.ndarray, y: np.ndarray, target_accuracy: float):
    sweep = []
    depth, n, _ = logits_by_layer.shape
    for margin_threshold in np.linspace(0.25, 3.0, 23):
        exits = []
        preds = []
        for sample in range(n):
            exit_layer = depth - 1
            pred = logits_by_layer[-1, sample].argmax()
            for layer in range(depth):
                logits = logits_by_layer[layer, sample]
                ordered = np.sort(logits)
                margin = ordered[-1] - ordered[-2]
                if margin >= margin_threshold:
                    exit_layer = layer
                    pred = logits.argmax()
                    break
            exits.append(exit_layer)
            preds.append(pred)
        accuracy = float((np.asarray(preds) == y).mean())
        sweep.append(
            {
                "margin_threshold": float(margin_threshold),
                "accuracy": accuracy,
                "compute_fraction": float((np.mean(exits) + 1.0) / depth),
                "mean_exit_layer": float(np.mean(exits)),
                "meets_target": bool(accuracy >= target_accuracy),
            }
        )
    feasible = [item for item in sweep if item["meets_target"]]
    best = min(feasible or sweep, key=lambda item: (item["compute_fraction"], -item["accuracy"]))
    return best, sweep


def run(seed: int = 29) -> dict:
    model, x_test, y_test = _train(seed)
    logits_by_layer = _layer_logits(model, x_test)
    depth = logits_by_layer.shape[0]
    final_logits = logits_by_layer[-1]
    final_accuracy = _accuracy(final_logits, y_test)
    layer_accuracy = np.asarray([_accuracy(logits_by_layer[layer], y_test) for layer in range(depth)])

    correct_direction = final_logits[np.arange(len(y_test)), y_test]
    evidence_paths = logits_by_layer[:, np.arange(len(y_test)), y_test].T
    linearity = [linear_r2(np.arange(depth), path)[2] for path in evidence_paths]
    increments = np.diff(evidence_paths, axis=1)
    diffusion_cv = float(increments.var(axis=0).std() / (increments.var(axis=0).mean() + 1e-12))
    mean_drift = float(np.mean(np.gradient(evidence_paths, axis=1)))

    ddm_pred = logistic(np.arange(1, depth + 1), center=4.2, scale=1.25)
    _, _, accuracy_depth_r2 = linear_r2(ddm_pred, layer_accuracy)
    target_accuracy = 0.95 * final_accuracy
    best_exit, sweep = _exit_sweep(logits_by_layer, y_test, target_accuracy)

    payload = {
        "experiment": "05_ddm_depth",
        "pilot_type": "real sklearn-digits residual-depth early-exit benchmark",
        "task_setup": "classify sklearn digits 0-4 using a 10-step additive residual linear classifier trained with intermediate losses",
        "win_metric": "intermediate residual states should support early exit with >=95% of final accuracy at <60% compute",
        "final_accuracy": final_accuracy,
        "layer_accuracy": layer_accuracy.tolist(),
        "mean_correct_class_evidence": float(correct_direction.mean()),
        "mean_drift_rate": mean_drift,
        "mean_linearity_r2": float(np.mean(linearity)),
        "diffusion_variance_cv_across_depth": diffusion_cv,
        "accuracy_depth_r2": accuracy_depth_r2,
        "target_retained_accuracy": target_accuracy,
        "best_early_exit": best_exit,
        "early_exit_sweep": sweep,
        "notable_win": bool(best_exit["accuracy"] >= target_accuracy and best_exit["compute_fraction"] < 0.60),
    }
    write_result("05_ddm_depth", "pilot_result.json", payload)
    return payload


if __name__ == "__main__":
    print(run())
