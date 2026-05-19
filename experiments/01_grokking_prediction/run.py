from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.pilot_metrics import first_crossing, linear_r2, mean_pairwise_cosine, write_result


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _one_hot(y, classes):
    out = np.zeros((len(y), classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def _forward(model, a, b):
    ea = model["E"][a]
    eb = model["E"][b]
    x = np.concatenate([ea, eb], axis=1)
    z1 = x @ model["W1"] + model["b1"]
    h = np.tanh(z1)
    logits = h @ model["W2"] + model["b2"]
    return x, h, logits


def _representation_rho(model, pairs, labels, p):
    _, h, _ = _forward(model, pairs[:, 0], pairs[:, 1])
    centroids = []
    for cls in range(p):
        centroids.append(h[labels == cls].mean(axis=0))
    rho, rho_std = mean_pairwise_cosine(np.asarray(centroids))
    return rho, rho_std


def run(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    p = 19
    pairs = np.asarray([(a, b) for a in range(p) for b in range(p)], dtype=int)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    order = rng.permutation(len(pairs))
    train_size = int(0.45 * len(pairs))
    train_idx = order[:train_size]
    val_idx = order[train_size:]
    x_train, y_train = pairs[train_idx], labels[train_idx]
    x_val, y_val = pairs[val_idx], labels[val_idx]

    emb_dim = 24
    hidden = 96
    model = {
        "E": rng.normal(0.0, 0.15, size=(p, emb_dim)),
        "W1": rng.normal(0.0, 0.18, size=(emb_dim * 2, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0.0, 0.18, size=(hidden, p)),
        "b2": np.zeros(p),
    }
    yoh = _one_hot(y_train, p)
    lr = 0.42
    weight_decay = 0.0008
    history = []

    for epoch in range(1001):
        xcat, h, logits = _forward(model, x_train[:, 0], x_train[:, 1])
        probs = _softmax(logits)
        dlogits = (probs - yoh) / len(y_train)
        dW2 = h.T @ dlogits + weight_decay * model["W2"]
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ model["W2"].T
        dz1 = dh * (1.0 - h * h)
        dW1 = xcat.T @ dz1 + weight_decay * model["W1"]
        db1 = dz1.sum(axis=0)
        dx = dz1 @ model["W1"].T
        dEa = dx[:, :emb_dim]
        dEb = dx[:, emb_dim:]
        dE = weight_decay * model["E"]
        np.add.at(dE, x_train[:, 0], dEa)
        np.add.at(dE, x_train[:, 1], dEb)

        model["W2"] -= lr * dW2
        model["b2"] -= lr * db2
        model["W1"] -= lr * dW1
        model["b1"] -= lr * db1
        model["E"] -= lr * dE
        lr *= 0.9975

        if epoch % 20 == 0:
            _, _, train_logits = _forward(model, x_train[:, 0], x_train[:, 1])
            _, _, val_logits = _forward(model, x_val[:, 0], x_val[:, 1])
            train_acc = float((_softmax(train_logits).argmax(axis=1) == y_train).mean())
            val_acc = float((_softmax(val_logits).argmax(axis=1) == y_val).mean())
            rho, rho_std = _representation_rho(model, pairs, labels, p)
            history.append({"epoch": epoch, "train_accuracy": train_acc, "validation_accuracy": val_acc, "rho": rho, "rho_std": rho_std})

    epochs = np.asarray([h["epoch"] for h in history])
    val_acc = np.asarray([h["validation_accuracy"] for h in history])
    rho = np.asarray([h["rho"] for h in history])
    grokking_step = first_crossing(val_acc, 0.80)
    grokking_epoch = None if grokking_step is None else int(epochs[grokking_step])
    velocity = np.gradient(rho)
    early_step = int(np.argmax(velocity[:grokking_step])) if grokking_step and grokking_step > 2 else None
    early_epoch = None if early_step is None else int(epochs[early_step])
    lead = None if early_epoch is None or grokking_epoch is None else grokking_epoch - early_epoch
    _, _, rho_val_r2 = linear_r2(rho, val_acc)

    payload = {
        "experiment": "01_grokking_prediction",
        "pilot_type": "real modular-addition training run",
        "task_setup": "MLP with learned token embeddings trained on 45% of all a+b mod 19 pairs and evaluated on held-out pairs",
        "win_metric": "representation velocity should give an early warning before held-out modular-addition generalization",
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "best_validation_accuracy": float(val_acc.max()),
        "grokking_epoch_80pct_validation": grokking_epoch,
        "velocity_early_warning_epoch": early_epoch,
        "velocity_early_warning_lead_epochs": lead,
        "rho_validation_linear_r2": rho_val_r2,
        "history": history,
        "notable_win": bool(lead is not None and lead >= 40 and val_acc.max() >= 0.80),
    }
    write_result("01_grokking_prediction", "pilot_result.json", payload)
    return payload


if __name__ == "__main__":
    print(run())
