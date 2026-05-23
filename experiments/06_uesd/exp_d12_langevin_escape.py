"""
Experiment D12: Langevin Dynamics — Noise-Driven Basin Escape

The UESD formalization (Section 4.2) proposes Langevin dynamics:
  s_{t+1} = s_t + F_theta(s_t, c) + sqrt(2*tau(t)) * epsilon_t

This experiment tests whether noise injection can:
1. Rescue E5 wrong-attractor failures by helping escape bad basins
2. Improve accuracy on hard examples (long carry chains) via exploration
3. Produce annealing dynamics: high noise early (exploration) → low noise
   late (convergence), matching the theoretical thinking→generating transition

Architecture: Standard UESD + noise injection during inference.
Training is UNCHANGED — we inject noise only at test time, which tests
whether the learned energy landscape has the right basin structure for
Langevin exploration.

Three noise schedules:
1. CONSTANT: tau(t) = tau_0 for all t (baseline)
2. LINEAR ANNEAL: tau(t) = tau_0 * (1 - t/T) (simple cooling)
3. COSINE ANNEAL: tau(t) = tau_0 * 0.5 * (1 + cos(pi*t/T)) (smooth cooling)

Key measurements:
- Accuracy vs noise level (tau_0) for each schedule
- Does annealing outperform constant noise?
- Per-carry-chain-length accuracy: does noise help hard examples more?
- Does noise rescue specific examples that deterministic dynamics get wrong?
- State trajectory analysis: do noisy trajectories explore then converge?

PREDICTIONS:
1. Moderate noise (tau_0 ~ 0.01-0.1) with annealing IMPROVES accuracy on
   hard examples (long carry chains) while maintaining accuracy on easy ones.
2. Annealing outperforms constant noise because the system needs exploration
   early and convergence late — the hallmark of true Langevin dynamics.
3. For E5 wrong-attractor seeds, noise injection rescues 10-30% of failed
   examples by allowing escape from the wrong basin.
4. Optimal tau_0 correlates with the energy barrier height found in D11.

If prediction 3 holds, it directly validates the energy landscape
interpretation and suggests a practical training improvement: add
annealed noise during training to broaden basin coverage.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params, train
from shared.data import generate_batch


TAU_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
N_LANGEVIN_SAMPLES = 8  # average over multiple noise realizations


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
        "seed": seed,
    }


def noise_schedule_constant(t, T, tau_0):
    return tau_0


def noise_schedule_linear(t, T, tau_0):
    return tau_0 * (1.0 - t / T)


def noise_schedule_cosine(t, T, tau_0):
    import math
    return tau_0 * 0.5 * (1.0 + math.cos(math.pi * t / T))


SCHEDULES = {
    "constant": noise_schedule_constant,
    "linear": noise_schedule_linear,
    "cosine": noise_schedule_cosine,
}


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
    return max_chain


@torch.no_grad()
def langevin_inference(model, src, T, tau_0, schedule_fn, n_samples=1):
    """Run Langevin dynamics: deterministic step + noise injection."""
    B, L = src.shape
    half = L // 2
    context = model.encode(src)

    all_preds = []
    all_energies = []

    for sample_idx in range(n_samples):
        s = model.init_state(B, L, src.device)
        energies = []

        for t in range(T):
            tau_t = schedule_fn(t, T, tau_0)

            # Deterministic dynamics step
            s_new, _ = model.dynamics_step(s, context)

            # Track energy BEFORE noise (clean dynamics residual)
            residual = s_new - s
            energy = (residual ** 2).sum(dim=-1).mean(dim=-1)
            energies.append(energy)

            # Add noise (Langevin)
            if tau_t > 0 and t < T - 1:  # no noise on last step
                noise = torch.randn_like(s_new) * (2 * tau_t) ** 0.5
                s_new = s_new + noise

            s = s_new

        logits = model.readout_logits(s)
        preds = logits[:, :half, :].argmax(dim=-1)
        all_preds.append(preds)
        all_energies.append(torch.stack(energies, dim=1))

    return all_preds, all_energies


@torch.no_grad()
def evaluate_langevin(model, src, tgt, T, tau_0, schedule_fn, max_chain, half):
    """Full evaluation with multiple noise samples and majority voting."""
    all_preds, all_energies = langevin_inference(
        model, src, T, tau_0, schedule_fn, n_samples=N_LANGEVIN_SAMPLES
    )
    targets = tgt[:, :half]
    B = src.shape[0]

    # Per-sample accuracy
    sample_accs = []
    for preds in all_preds:
        seq_correct = (preds == targets).all(dim=1)
        sample_accs.append(seq_correct.float().mean().item())

    # Majority vote across samples
    vote_preds = torch.stack(all_preds, dim=0)  # [n_samples, B, half]
    majority_preds = torch.zeros_like(all_preds[0])
    for b in range(B):
        for p in range(half):
            counts = torch.bincount(vote_preds[:, b, p], minlength=64)
            majority_preds[b, p] = counts.argmax()

    majority_correct = (majority_preds == targets).all(dim=1)
    majority_acc = majority_correct.float().mean().item()

    # Per-chain-length accuracy
    per_chain = {}
    for chain_len in range(half + 1):
        mask = max_chain == chain_len
        n = mask.sum().item()
        if n < 10:
            continue
        chain_acc = majority_correct[mask].float().mean().item()
        per_chain[int(chain_len)] = {"n": n, "accuracy": chain_acc}

    # Mean energy profile
    mean_energies = torch.stack(all_energies, dim=0).mean(dim=0)  # [B, T]
    energy_profile = mean_energies.mean(dim=0).cpu().tolist()

    return {
        "majority_seq_acc": majority_acc,
        "sample_accs": sample_accs,
        "mean_sample_acc": float(np.mean(sample_accs)),
        "per_chain_length": per_chain,
        "energy_profile": energy_profile,
    }


@torch.no_grad()
def rescue_analysis(model, src, tgt, T, config, device):
    """Identify examples where deterministic fails but Langevin succeeds."""
    half = config["seq_len"] // 2
    targets = tgt[:, :half]
    B = src.shape[0]

    # Deterministic baseline
    logits_det = model(src, T)
    preds_det = logits_det[:, :half, :].argmax(dim=-1)
    correct_det = (preds_det == targets).all(dim=1)

    # Best Langevin (cosine anneal, sweep tau)
    best_rescued = 0
    best_tau = 0
    best_details = None

    for tau_0 in [0.01, 0.05, 0.1]:
        all_preds, _ = langevin_inference(
            model, src, T, tau_0, noise_schedule_cosine, n_samples=N_LANGEVIN_SAMPLES
        )

        # Majority vote
        vote_preds = torch.stack(all_preds, dim=0)
        majority_preds = torch.zeros_like(all_preds[0])
        for b in range(B):
            for p in range(half):
                counts = torch.bincount(vote_preds[:, b, p], minlength=64)
                majority_preds[b, p] = counts.argmax()

        correct_lang = (majority_preds == targets).all(dim=1)

        # Rescued: wrong deterministically, correct with Langevin
        rescued = (~correct_det & correct_lang).sum().item()
        # Broken: correct deterministically, wrong with Langevin
        broken = (correct_det & ~correct_lang).sum().item()

        if rescued > best_rescued:
            best_rescued = rescued
            best_tau = tau_0
            best_details = {
                "tau_0": tau_0,
                "rescued": rescued,
                "broken": broken,
                "det_correct": correct_det.sum().item(),
                "lang_correct": correct_lang.sum().item(),
                "net_gain": rescued - broken,
            }

    return best_details if best_details else {
        "tau_0": 0, "rescued": 0, "broken": 0,
        "det_correct": correct_det.sum().item(),
        "lang_correct": correct_det.sum().item(), "net_gain": 0,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config()
    V = config["vocab_size"]
    T = config["T"]
    half = config["seq_len"] // 2

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Langevin dynamics: noise injection for basin escape and exploration",
        "config": config,
    }

    # Generate eval data
    set_seed(7777)
    eval_src, eval_tgt = generate_batch("addition", 2048, config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)
    max_chain = compute_carries(eval_src, V)

    print(f"\nCarry chain distribution:", flush=True)
    for cl in range(half + 1):
        n = (max_chain == cl).sum().item()
        print(f"  max_chain={cl}: {n}/2048 ({n/2048:.1%})", flush=True)

    for track in ["e5", "dynamics_ce"]:
        seed = 42
        label = f"{track}_seed{seed}"
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Training: {label}", flush=True)
        print(f"{'=' * 60}", flush=True)

        cfg = build_config(seed)
        set_seed(seed)
        model = UESDModel(
            V, cfg["d_model"], cfg["n_heads"],
            cfg["d_ff"], cfg["n_enc_layers"], cfg["max_len"],
        ).to(device)

        t0 = time.time()
        tr = train(model, "addition", track, cfg, device)
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s, final CE: {tr['ce_history'][-1]:.4f}",
              flush=True)

        # Deterministic baseline
        with torch.no_grad():
            logits = model(eval_src, T)
            preds = logits[:, :half, :].argmax(dim=-1)
            targets = eval_tgt[:, :half]
            det_tok = (preds == targets).float().mean().item()
            det_seq = (preds == targets).all(dim=1).float().mean().item()
        print(f"  Deterministic: tok={det_tok:.4f} seq={det_seq:.4f}", flush=True)

        track_results = {
            "train_time_s": train_time,
            "deterministic_tok_acc": det_tok,
            "deterministic_seq_acc": det_seq,
        }

        # Sweep tau and schedules
        for sched_name, sched_fn in SCHEDULES.items():
            print(f"\n  Schedule: {sched_name}", flush=True)
            sched_results = {}

            for tau_0 in TAU_VALUES:
                res = evaluate_langevin(
                    model, eval_src, eval_tgt, T, tau_0, sched_fn, max_chain, half
                )
                sched_results[f"tau_{tau_0}"] = res
                print(f"    tau={tau_0:.3f}: majority_acc={res['majority_seq_acc']:.4f} "
                      f"mean_sample={res['mean_sample_acc']:.4f}", flush=True)

            track_results[sched_name] = sched_results

        # Rescue analysis
        print(f"\n  Rescue analysis...", flush=True)
        rescue = rescue_analysis(model, eval_src, eval_tgt, T, cfg, device)
        track_results["rescue_analysis"] = rescue
        print(f"    Best tau={rescue['tau_0']}: rescued={rescue['rescued']}, "
              f"broken={rescue['broken']}, net={rescue['net_gain']}", flush=True)

        all_results[label] = track_results

        del model
        torch.cuda.empty_cache()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d12_langevin_escape.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
