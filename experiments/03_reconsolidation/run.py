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


def _make_tasks(seed):
    d = load_digits()
    x = StandardScaler().fit_transform(d.data.astype(float) / 16.0)
    y = d.target.astype(int)
    tasks = []
    for lo in [0, 2, 4, 6, 8]:
        mask = (y == lo) | (y == lo + 1)
        x_task = x[mask]
        y_task = y[mask]
        tasks.append(train_test_split(x_task, y_task, test_size=0.35, random_state=seed + lo, stratify=y_task))
    return tasks


def _init(seed, dim=64, hidden=96, classes=10):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.15, (dim, hidden)), "b1": np.zeros(hidden), "W2": rng.normal(0, 0.15, (hidden, classes)), "b2": np.zeros(classes)}


def _forward(m, x):
    z = x @ m["W1"] + m["b1"]
    h = np.tanh(z)
    return h, h @ m["W2"] + m["b2"]


def _grads(m, x, y):
    h, logits = _forward(m, x)
    probs = _softmax(logits)
    yoh = np.zeros_like(probs); yoh[np.arange(len(y)), y] = 1
    dlogits = (probs - yoh) / len(y)
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ m["W2"].T
    dz = dh * (1 - h * h)
    dW1 = x.T @ dz
    db1 = dz.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, h


def _acc(m, x, y):
    _, logits = _forward(m, x)
    return float((logits.argmax(axis=1) == y).mean())


def _importance(m, x, y):
    g, _ = _grads(m, x, y)
    return {k: v * v for k, v in g.items()}


def _run_method(method, seed, params=None):
    tasks = _make_tasks(seed)
    m = _init(seed + 100)
    lr = 0.34
    importance = {k: np.zeros_like(v) for k, v in m.items()}
    anchor = {k: v.copy() for k, v in m.items()}
    hidden_mean = None
    hidden_std = None
    labilized = []
    history = []
    params = params or {"active_scale": 1.15, "inactive_scale": 0.45, "anchor_strength": 0.008, "min_frac": 0.16}

    for task_id, (xtr, xte, ytr, yte) in enumerate(tasks):
        for _ in range(90):
            g, h = _grads(m, xtr, ytr)
            if method == "naive":
                for k in m: m[k] -= lr * g[k]
            elif method == "ewc":
                for k in m: m[k] -= lr * (g[k] + 12.0 * importance[k] * (m[k] - anchor[k]))
            elif method == "reconsolidation":
                if hidden_mean is None:
                    retrieved = np.ones(m["W1"].shape[1], dtype=bool)
                else:
                    signal = np.abs(h).mean(axis=0)
                    retrieved = signal > hidden_mean + 0.15 * hidden_std
                    min_count = max(4, int(params["min_frac"] * len(retrieved)))
                    if retrieved.sum() < min_count:
                        retrieved[np.argsort(signal)[-min_count:]] = True
                labilized.append(float(retrieved.mean()))
                scale = np.where(retrieved, params["active_scale"], params["inactive_scale"])
                m["W1"] -= lr * g["W1"] * scale[None, :]
                m["b1"] -= lr * g["b1"] * scale
                m["W2"] -= lr * g["W2"] * scale[:, None]
                m["b2"] -= lr * g["b2"]
                if hidden_mean is not None:
                    stable = ~retrieved
                    m["W1"][:, stable] -= lr * params["anchor_strength"] * (m["W1"][:, stable] - anchor["W1"][:, stable])
                    m["W2"][stable, :] -= lr * params["anchor_strength"] * (m["W2"][stable, :] - anchor["W2"][stable, :])
            else:
                raise ValueError(method)
        if method in {"ewc", "reconsolidation"}:
            imp = _importance(m, xtr, ytr)
            for k in m:
                importance[k] = 0.65 * importance[k] + imp[k]
                anchor[k] = m[k].copy()
        htr, _ = _forward(m, xtr)
        cur_mean = np.abs(htr).mean(axis=0)
        cur_std = np.abs(htr).std(axis=0) + 1e-6
        hidden_mean = cur_mean if hidden_mean is None else 0.65 * hidden_mean + 0.35 * cur_mean
        hidden_std = cur_std if hidden_std is None else 0.65 * hidden_std + 0.35 * cur_std
        seen = []
        for prior in range(task_id + 1):
            _, xte2, _, yte2 = tasks[prior]
            seen.append(_acc(m, xte2, yte2))
        history.append(seen)
    final = np.asarray(history[-1])
    return {"final_average_accuracy": float(final.mean()), "final_min_accuracy": float(final.min()), "first_task_forgetting": float(history[0][0] - history[-1][0]), "mean_labilized_rate": float(np.mean(labilized)) if labilized else 0.0, "trajectory": history}


def run(seed: int = 19):
    seeds = [seed, seed + 1]
    methods = {name: [_run_method(name, s) for s in seeds] for name in ["naive", "ewc", "reconsolidation"]}
    agg = {}
    for name, runs in methods.items():
        agg[name] = {k: float(np.mean([r[k] for r in runs])) for k in ["final_average_accuracy", "final_min_accuracy", "first_task_forgetting", "mean_labilized_rate"]}
    payload = {
        "experiment": "03_reconsolidation",
        "pilot_type": "real split-sklearn-digits continual-learning benchmark",
        "task_setup": "sequential 10-way classifier trained on digit pairs 0/1, 2/3, 4/5, 6/7, 8/9 without replay",
        "win_metric": "reconsolidation should beat EWC or reduce first-task forgetting without replay",
        "methods": agg,
        "reconsolidation_vs_ewc_accuracy_delta": agg["reconsolidation"]["final_average_accuracy"] - agg["ewc"]["final_average_accuracy"],
        "forgetting_reduction_vs_ewc": agg["ewc"]["first_task_forgetting"] - agg["reconsolidation"]["first_task_forgetting"],
        "notable_win": bool(agg["reconsolidation"]["final_average_accuracy"] > agg["ewc"]["final_average_accuracy"] or agg["reconsolidation"]["first_task_forgetting"] < agg["ewc"]["first_task_forgetting"]),
    }
    write_result("03_reconsolidation", "pilot_result.json", payload)
    return payload


if __name__ == "__main__":
    print(run())
