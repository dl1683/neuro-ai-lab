from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.pilot_metrics import write_result


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _data(seed):
    d = load_digits()
    x = StandardScaler().fit_transform(d.data.astype(float) / 16.0)
    y = d.target.astype(int)
    return train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)


def _init(seed, dim=64, hidden=80, classes=10):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.16, (dim, hidden)), "b1": np.zeros(hidden), "W2": rng.normal(0, 0.16, (hidden, classes)), "b2": np.zeros(classes)}


def _forward(m, x, masks=None):
    w1 = m["W1"] if masks is None else m["W1"] * masks["W1"]
    w2 = m["W2"] if masks is None else m["W2"] * masks["W2"]
    h = np.maximum(0, x @ w1 + m["b1"])
    return h, h @ w2 + m["b2"]


def _acc(m, x, y, masks=None):
    _, logits = _forward(m, x, masks)
    return float((logits.argmax(axis=1) == y).mean())


def _train_epoch(m, x, y, lr, wd, masks=None, noise=0.0, rng=None):
    if noise and rng is not None:
        x = x + rng.normal(0, noise, x.shape)
    yoh = np.zeros((len(y), 10)); yoh[np.arange(len(y)), y] = 1
    h, logits = _forward(m, x, masks)
    probs = _softmax(logits)
    dlogits = (probs - yoh) / len(y)
    w2_mask = 1 if masks is None else masks["W2"]
    w1_mask = 1 if masks is None else masks["W1"]
    dW2 = h.T @ dlogits + wd * m["W2"]
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ (m["W2"] * w2_mask).T
    dz = dh * (h > 0)
    dW1 = x.T @ dz + wd * m["W1"]
    db1 = dz.sum(axis=0)
    m["W1"] -= lr * dW1 * w1_mask
    m["b1"] -= lr * db1
    m["W2"] -= lr * dW2 * w2_mask
    m["b2"] -= lr * db2


def _sparsity(masks):
    total = masks["W1"].size + masks["W2"].size
    active = masks["W1"].sum() + masks["W2"].sum()
    return float(1 - active / total)


def run(seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    xtr, xte, ytr, yte = _data(seed)
    base = _init(seed)
    sleep = _init(seed)
    masks = {"W1": np.ones_like(sleep["W1"]), "W2": np.ones_like(sleep["W2"])}

    for epoch in range(90):
        _train_epoch(base, xtr, ytr, lr=0.14, wd=0.0005, rng=rng)

    for cycle in range(9):
        for _ in range(7):
            _train_epoch(sleep, xtr, ytr, lr=0.075, wd=0.004, masks=masks, rng=rng)
        # NREM pruning: remove weakest still-active weights.
        for key in ["W1", "W2"]:
            active_weights = np.abs(sleep[key][masks[key] > 0])
            cutoff = np.quantile(active_weights, 0.055) if active_weights.size else np.inf
            masks[key] *= (np.abs(sleep[key]) >= cutoff)
        for _ in range(3):
            _train_epoch(sleep, xtr, ytr, lr=0.12, wd=0.0005, masks=masks, noise=0.08, rng=rng)

    baseline_train = _acc(base, xtr, ytr)
    baseline_val = _acc(base, xte, yte)
    sleep_train = _acc(sleep, xtr, ytr, masks)
    sleep_val = _acc(sleep, xte, yte, masks)
    baseline_gap = baseline_train - baseline_val
    sleep_gap = sleep_train - sleep_val
    sparsity = _sparsity(masks)
    payload = {
        "experiment": "02_sleep_training",
        "pilot_type": "real sklearn-digits MLP baseline-vs-sleep-schedule run",
        "task_setup": "baseline full-batch MLP vs NREM weight-decay/pruning and REM noisy-replay schedule on sklearn digits",
        "baseline_train_accuracy": baseline_train,
        "baseline_validation_accuracy": baseline_val,
        "sleep_train_accuracy": sleep_train,
        "sleep_validation_accuracy": sleep_val,
        "baseline_generalization_gap": baseline_gap,
        "sleep_generalization_gap": sleep_gap,
        "sleep_sparsity": sparsity,
        "validation_delta_sleep_minus_baseline": sleep_val - baseline_val,
        "notable_win": bool(sleep_val >= baseline_val - 0.02 and sparsity >= 0.30 and sleep_gap <= baseline_gap + 0.01),
    }
    write_result("02_sleep_training", "pilot_result.json", payload)
    return payload


if __name__ == "__main__":
    print(run())
