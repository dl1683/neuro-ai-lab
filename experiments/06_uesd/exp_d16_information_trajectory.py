"""
Experiment D16: Information Accumulation Trajectory

Direct test of Axiom A4: "Thinking and generating are regions of the
same trajectory through S." If true, the state s_t should accumulate
information about the target y* monotonically through dynamics steps.

This goes beyond D7's "first-stable-correct step" in two ways:
1. Measures PARTIAL information via linear probe accuracy — a position
   might have high MI with the answer before it stabilizes
2. Measures information accumulation rate, not just threshold crossing

Protocol:
1. Train E5 and CE-dynamics models
2. Collect intermediate states s_0, s_1, ..., s_T on eval set
3. Train frozen linear probes from s_t → y* at each step t
4. Plot probe accuracy vs t for each result position
5. Stratify by carry-chain length

PREDICTIONS:
1. Information increases monotonically with t (no backtracking)
2. Rate of increase is higher for carry-independent positions
   (they can be solved earlier in the trajectory)
3. E5 shows smoother information accumulation (highway dynamics),
   CE-dynamics shows more variable/circuitous accumulation
4. For carry-dependent positions, information plateaus briefly
   then jumps when carry propagation resolves
5. At step t=0 (initial state), probe accuracy ≈ chance (1/V)
   At step t=T, probe accuracy ≈ model accuracy

If prediction 4 holds, it directly shows "thinking" — the model holds
partial information while working on the hard computational part.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch


def build_config(seed=42):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "warmup_steps": 5000,
        "seed": seed,
    }


def train_model(track, config, device):
    set_seed(config["seed"])
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]
    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        s = model.init_state(B, L, src.device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]

        if track == "dynamics_ce":
            loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                   tgt_r.reshape(-1))
        else:
            ce = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                 tgt_r.reshape(-1))
            s_next = model.dynamics(s, context)
            sc = (s_next - s).pow(2).sum(dim=-1).mean()
            eff_lam = min(step / config["warmup_steps"], 1.0)
            loss = ce + eff_lam * sc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    model.eval()
    return model


def compute_carries(src, vocab_size):
    B, L = src.shape
    half = L // 2
    a = src[:, 0::2][:, :half]
    b = src[:, 1::2][:, :half]
    carry = torch.zeros(B, dtype=torch.long, device=src.device)
    carry_in = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        carry = (s >= vocab_size).long()
        if i > 0:
            carry_in[:, i - 1] = carry
    max_chain = torch.zeros(B, dtype=torch.long, device=src.device)
    chain_len = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 2, -1, -1):
        has_carry = carry_in[:, i] == 1
        chain_len[has_carry, i] = 1 + chain_len[has_carry, i + 1]
    max_chain = chain_len.max(dim=1).values
    return max_chain, carry_in


@torch.no_grad()
def collect_intermediate_states(model, eval_src, config, device):
    """Collect s_0, s_1, ..., s_T for all eval examples."""
    T = config["T"]
    context = model.encode(eval_src)
    B, L = eval_src.shape
    s = model.init_state(B, L, device)

    states = [s.clone()]
    for t in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    return states  # list of T+1 tensors, each [B, L, d]


def make_example_split(B, train_frac=0.8, seed=1234):
    """Split examples (not tokens) into train/test. Returns index arrays."""
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B, generator=rng)
    n_train = int(train_frac * B)
    return perm[:n_train], perm[n_train:]


def train_linear_probe(states_t, targets, d_model, V, half, device,
                       train_idx, test_idx, n_epochs=50, lr=1e-3):
    """Train a linear probe from s_t to y* with example-level train/test split.

    train_idx/test_idx are example-level indices (not token-level), ensuring
    no positions from the same addition problem leak across splits.
    """
    train_feat = states_t[train_idx, :half, :].reshape(-1, d_model)
    train_labels = targets[train_idx, :half].reshape(-1)
    test_feat = states_t[test_idx, :half, :].reshape(-1, d_model)
    test_labels = targets[test_idx, :half].reshape(-1)

    probe = nn.Linear(d_model, V).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(n_epochs):
        logits = probe(train_feat)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = probe(test_feat)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == test_labels).float().mean().item()

        per_pos = []
        n_test = len(test_idx)
        for p in range(half):
            pos_preds = test_preds[p::half]
            pos_labels = test_labels[p::half]
            pos_acc = (pos_preds == pos_labels).float().mean().item()
            per_pos.append(pos_acc)

    return test_acc, per_pos


def train_shuffled_probe(states_t, targets, d_model, V, half, device,
                         train_idx, test_idx, n_epochs=50, lr=1e-3):
    """Shuffled-label control: same as train_linear_probe but permuted targets."""
    perm = torch.randperm(targets.shape[0], device=device)
    shuffled_tgt = targets[perm]
    return train_linear_probe(
        states_t, shuffled_tgt, d_model, V, half, device,
        train_idx, test_idx, n_epochs, lr
    )


@torch.no_grad()
def readout_accuracy_per_step(model, states, eval_tgt, config):
    """Direct readout accuracy at each step (no probe, uses model's readout)."""
    half = config["seq_len"] // 2
    targets = eval_tgt[:, :half]

    step_accs = []
    for t, s_t in enumerate(states):
        logits = model.readout_logits(s_t)
        preds = logits[:, :half, :].argmax(dim=-1)
        tok_acc = (preds == targets).float().mean().item()
        seq_acc = (preds == targets).all(dim=1).float().mean().item()

        per_pos = []
        for p in range(half):
            pos_acc = (preds[:, p] == targets[:, p]).float().mean().item()
            per_pos.append(pos_acc)

        step_accs.append({
            "step": t,
            "tok_acc": tok_acc,
            "seq_acc": seq_acc,
            "per_position": per_pos,
        })

    return step_accs


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config()
    V = config["vocab_size"]
    d_model = config["d_model"]
    T = config["T"]
    half = config["seq_len"] // 2

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Information accumulation trajectory: MI(s_t, y*) via probes",
        "config": config,
    }

    # Generate eval data
    set_seed(6666)
    eval_src, eval_tgt = generate_batch("addition", 4096, config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)
    max_chain, carry_in = compute_carries(eval_src, V)

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Training: {track}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model = train_model(track, config, device)

        # Collect intermediate states
        print(f"  Collecting intermediate states...", flush=True)
        states = collect_intermediate_states(model, eval_src, config, device)

        # Direct readout accuracy per step
        print(f"  Direct readout accuracy per step...", flush=True)
        readout_accs = readout_accuracy_per_step(model, states, eval_tgt, config)
        for ra in readout_accs:
            if ra["step"] in [0, 1, 3, 5, 7, 10]:
                print(f"    t={ra['step']}: tok={ra['tok_acc']:.4f} "
                      f"seq={ra['seq_acc']:.4f} "
                      f"pos={[f'{p:.3f}' for p in ra['per_position']]}",
                      flush=True)

        # Fixed example-level split (shared across all steps and controls)
        train_idx, test_idx = make_example_split(eval_src.shape[0])
        train_idx = train_idx.to(device)
        test_idx = test_idx.to(device)

        # Linear probe accuracy per step
        print(f"  Training linear probes per step...", flush=True)
        probe_results = []
        for t in range(T + 1):
            probe_acc, probe_per_pos = train_linear_probe(
                states[t], eval_tgt, d_model, V, half, device,
                train_idx, test_idx,
            )
            probe_results.append({
                "step": t,
                "probe_accuracy": probe_acc,
                "per_position": probe_per_pos,
            })
            if t in [0, 1, 3, 5, 7, 10]:
                print(f"    t={t}: probe={probe_acc:.4f} "
                      f"pos={[f'{p:.3f}' for p in probe_per_pos]}",
                      flush=True)

        # Shuffled-label control at t=0 and t=T
        print(f"  Shuffled-label control probes...", flush=True)
        shuffled_controls = {}
        for t in [0, T]:
            sh_acc, sh_pos = train_shuffled_probe(
                states[t], eval_tgt, d_model, V, half, device,
                train_idx, test_idx,
            )
            shuffled_controls[t] = {"probe_accuracy": sh_acc, "per_position": sh_pos}
            print(f"    t={t} shuffled: probe={sh_acc:.4f}", flush=True)

        # Check monotonicity
        probe_accs = [p["probe_accuracy"] for p in probe_results]
        monotonic = all(probe_accs[i] <= probe_accs[i + 1]
                        for i in range(len(probe_accs) - 1))
        max_backtrack = max(
            (probe_accs[i] - probe_accs[i + 1]
             for i in range(len(probe_accs) - 1)),
            default=0.0,
        )
        print(f"  Monotonic: {monotonic}, max backtrack: {max_backtrack:.4f}",
              flush=True)

        # Per-chain-length probe analysis
        print(f"  Per-chain probe analysis...", flush=True)
        chain_probe_results = {}
        for chain_len in range(half):
            mask = max_chain == chain_len
            n = mask.sum().item()
            if n < 100:
                continue

            chain_states = [s[mask] for s in states]
            chain_tgt = eval_tgt[mask]
            chain_train, chain_test = make_example_split(n)
            chain_train = chain_train.to(device)
            chain_test = chain_test.to(device)

            chain_probes = []
            for t in range(T + 1):
                acc, per_pos = train_linear_probe(
                    chain_states[t], chain_tgt, d_model, V, half, device,
                    chain_train, chain_test, n_epochs=30,
                )
                chain_probes.append({"step": t, "probe_accuracy": acc})

            chain_probe_results[int(chain_len)] = {
                "n": n,
                "probes": chain_probes,
            }
            accs_str = " → ".join(
                f"{p['probe_accuracy']:.3f}" for p in chain_probes
                if p["step"] in [0, 3, 5, 7, 10]
            )
            print(f"    chain={chain_len} (n={n}): {accs_str}", flush=True)

        all_results[track] = {
            "readout_accuracy": readout_accs,
            "probe_accuracy": probe_results,
            "shuffled_label_controls": {
                str(k): v for k, v in shuffled_controls.items()
            },
            "monotonic": monotonic,
            "max_backtrack": max_backtrack,
            "per_chain_probes": chain_probe_results,
        }

        del model, states
        torch.cuda.empty_cache()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d16_information_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
