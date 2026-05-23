"""
Experiment D22: Robust Dynamics — Variable-T Curriculum + Denoising Training

Codex falsification review identified two critical weaknesses at 4.5/10 confidence:
1. NO RECOVERY from perturbation (D17, D21 — all recovery values negative)
2. FINITE COMPUTE WINDOW (D19 — CE degrades at T>15, collapses at T=32)

This experiment tests whether training modifications can fix both:

Variant A — VARIABLE-T CURRICULUM:
  Sample T uniformly from {4,6,8,10,12,14,16} each batch instead of fixed T=10.
  Hypothesis: model learns to produce correct readouts at ANY step count,
  widening the compute window and potentially enabling adaptive stopping.

Variant B — DENOISING-ROBUST TRAINING:
  Standard T=10, but at a random intermediate step t* in {3,4,5,6,7},
  inject Gaussian noise (sigma=0.3 * state_norm) into the state, then
  continue dynamics. Train CE on the final output.
  Hypothesis: dynamics learn to recover from perturbation, creating
  genuine error-correcting behavior.

Variant C — COMBINED (Variable-T + Denoising):
  Both modifications together.

Baseline — STANDARD CE-DYNAMICS (T=10, no modifications).

Evaluation protocol (identical across all variants):
- Step ablation: accuracy at T=1,2,3,5,8,10,15,20,32
- Recovery test: perturb at T=10, run +1,+5,+10,+20 extra steps
- Noise robustness: tau=0.05 cosine-annealed noise during inference
- Per-carry-position accuracy at each T

3 seeds x 4 variants = 12 training runs.

PREDICTIONS:
1. Variable-T widens compute window: T=20 and T=32 accuracy >> baseline
2. Denoising enables positive recovery values (the big one!)
3. Combined achieves both, possibly with slight accuracy tradeoff
4. All variants maintain T=10 accuracy >= 99%
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEEDS = [42, 1337, 2024]
TRAINING_STEPS = 20000
BATCH_SIZE = 256
SEQ_LEN = 8
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 32

EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 32]
RECOVERY_EXTRA_STEPS = [1, 5, 10, 20]
NOISE_TAU = 0.05
EVAL_SAMPLES = 4096

VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
DENOISE_INJECT_STEPS = [3, 4, 5, 6, 7]
DENOISE_SIGMA_FRAC = 0.3


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


def train_baseline(model, device, seed):
    """Standard CE-dynamics training at fixed T=10."""
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, 10)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s", flush=True)
    return elapsed


def train_variable_t(model, device, seed):
    """Variable-T curriculum: sample T from {4,6,8,10,12,14,16} each batch."""
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(VARIABLE_T_RANGE)
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s", flush=True)
    return elapsed


def train_denoising(model, device, seed):
    """Denoising-robust: inject noise at random intermediate step, train on final output."""
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        inject_step = random.choice(DENOISE_INJECT_STEPS)
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(1, 11):
            s, _ = model.dynamics_step(s, context)
            if t == inject_step:
                noise_scale = DENOISE_SIGMA_FRAC * s.detach().norm(dim=-1, keepdim=True)
                noise = torch.randn_like(s) * noise_scale
                s = s + noise

        logits = model.readout_logits(s)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | inject@{inject_step} | Loss: {loss.item():.4f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s", flush=True)
    return elapsed


def train_combined(model, device, seed):
    """Combined: variable-T + denoising injection."""
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(VARIABLE_T_RANGE)
        inject_step = random.choice([t for t in DENOISE_INJECT_STEPS if t < T])

        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(1, T + 1):
            s, _ = model.dynamics_step(s, context)
            if t == inject_step:
                noise_scale = DENOISE_SIGMA_FRAC * s.detach().norm(dim=-1, keepdim=True)
                noise = torch.randn_like(s) * noise_scale
                s = s + noise

        logits = model.readout_logits(s)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} inject@{inject_step} | Loss: {loss.item():.4f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s", flush=True)
    return elapsed


@torch.no_grad()
def evaluate_step_ablation(model, device):
    """Measure accuracy at each T value, with per-carry-position breakdown."""
    model.eval()
    half = SEQ_LEN // 2
    set_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    results = {}
    for T in EVAL_T_VALUES:
        logits = model(src, T)
        preds = logits[:, :half].argmax(dim=-1)
        tok_acc = (preds == tgt_result).float().mean().item()
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()

        per_pos = {}
        for p in range(half):
            per_pos[f"c{p}"] = (preds[:, p] == tgt_result[:, p]).float().mean().item()

        results[f"T={T}"] = {
            "tok_acc": round(tok_acc, 4),
            "seq_acc": round(seq_acc, 4),
            **{k: round(v, 3) for k, v in per_pos.items()},
        }

    return results


@torch.no_grad()
def evaluate_recovery(model, device):
    """D21-style: perturb at T=10, run extra steps, measure recovery."""
    model.eval()
    half = SEQ_LEN // 2
    set_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    for _ in range(10):
        s, _ = model.dynamics_step(s, context)

    logits_clean = model.readout_logits(s)
    preds_clean = logits_clean[:, :half].argmax(dim=-1)
    clean_correct = (preds_clean == tgt_result).all(dim=1)

    results = {}
    for sigma in [0.01, 0.1, 0.5, 1.0]:
        noise = torch.randn_like(s) * sigma
        s_noisy = s + noise

        preds_noisy = model.readout_logits(s_noisy)[:, :half].argmax(dim=-1)
        wa_at_0 = 1.0 - (preds_noisy == tgt_result).all(dim=1).float().mean().item()

        s_extra = s_noisy.clone()
        extra_results = {}
        for extra in RECOVERY_EXTRA_STEPS:
            prev_steps = extra_results.get("prev_steps", 0)
            for _ in range(extra - prev_steps):
                s_extra, _ = model.dynamics_step(s_extra, context)
            preds_extra = model.readout_logits(s_extra)[:, :half].argmax(dim=-1)
            wa_after = 1.0 - (preds_extra == tgt_result).all(dim=1).float().mean().item()
            recovery = wa_at_0 - wa_after
            extra_results[f"+{extra}"] = {
                "WA": round(wa_after, 4),
                "recovery": round(recovery, 4),
            }
            extra_results["prev_steps"] = extra

        del extra_results["prev_steps"]
        results[f"sigma={sigma}"] = {
            "WA_at_0": round(wa_at_0, 4),
            **extra_results,
        }

    return results


@torch.no_grad()
def evaluate_noise_robustness(model, device):
    """D12-style: cosine-annealed noise during dynamics, measure accuracy."""
    model.eval()
    half = SEQ_LEN // 2
    set_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    import math
    T = 10
    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    for t in range(T):
        s, _ = model.dynamics_step(s, context)
        schedule = 0.5 * (1 + math.cos(math.pi * t / (T - 1)))
        noise = torch.randn_like(s) * NOISE_TAU * schedule
        s = s + noise

    logits = model.readout_logits(s)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()

    return {"tok_acc": round(tok_acc, 4), "seq_acc": round(seq_acc, 4)}


def run_variant(variant_name, train_fn, device, seed):
    """Train and evaluate one variant with one seed."""
    print(f"\n  {variant_name}, seed={seed}", flush=True)

    set_seed(seed)
    model = make_model(device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    train_time = train_fn(model, device, seed)

    print("    Evaluating step ablation...", flush=True)
    step_abl = evaluate_step_ablation(model, device)
    for k, v in step_abl.items():
        print(f"      {k}: tok={v['tok_acc']:.4f} seq={v['seq_acc']:.4f}", flush=True)

    print("    Evaluating recovery...", flush=True)
    recovery = evaluate_recovery(model, device)
    for sigma_key, sigma_data in recovery.items():
        best_rec = max(v["recovery"] for k, v in sigma_data.items() if k != "WA_at_0")
        print(f"      {sigma_key}: WA@0={sigma_data['WA_at_0']:.4f} best_recovery={best_rec:+.4f}",
              flush=True)

    print("    Evaluating noise robustness...", flush=True)
    noise_rob = evaluate_noise_robustness(model, device)
    print(f"      tau={NOISE_TAU}: seq_acc={noise_rob['seq_acc']:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "params": params,
        "train_time_s": round(train_time, 1),
        "step_ablation": step_abl,
        "recovery": recovery,
        "noise_robustness": noise_rob,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    variants = {
        "baseline": train_baseline,
        "variable_t": train_variable_t,
        "denoising": train_denoising,
        "combined": train_combined,
    }

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D22: Robust dynamics - variable-T curriculum + denoising training",
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE, "lr": LR,
            "seeds": SEEDS,
            "variable_t_range": VARIABLE_T_RANGE,
            "denoise_inject_steps": DENOISE_INJECT_STEPS,
            "denoise_sigma_frac": DENOISE_SIGMA_FRAC,
        },
    }

    for variant_name, train_fn in variants.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"  VARIANT: {variant_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        variant_runs = []
        for seed in SEEDS:
            result = run_variant(variant_name, train_fn, device, seed)
            variant_runs.append(result)

        # Compute means across seeds
        mean_t10 = sum(r["step_ablation"]["T=10"]["seq_acc"] for r in variant_runs) / len(variant_runs)
        mean_t20 = sum(r["step_ablation"]["T=20"]["seq_acc"] for r in variant_runs) / len(variant_runs)
        mean_t32 = sum(r["step_ablation"]["T=32"]["seq_acc"] for r in variant_runs) / len(variant_runs)

        rec_vals = []
        for r in variant_runs:
            for sigma_key, sigma_data in r["recovery"].items():
                for k, v in sigma_data.items():
                    if k != "WA_at_0":
                        rec_vals.append(v["recovery"])
        mean_recovery = sum(rec_vals) / len(rec_vals) if rec_vals else 0

        mean_noise = sum(r["noise_robustness"]["seq_acc"] for r in variant_runs) / len(variant_runs)

        summary = {
            "mean_T10_seq": round(mean_t10, 4),
            "mean_T20_seq": round(mean_t20, 4),
            "mean_T32_seq": round(mean_t32, 4),
            "mean_recovery": round(mean_recovery, 4),
            "mean_noise_robustness": round(mean_noise, 4),
            "compute_window_width": f"T=10:{mean_t10:.3f} T=20:{mean_t20:.3f} T=32:{mean_t32:.3f}",
        }

        all_results[variant_name] = {
            "runs": variant_runs,
            "summary": summary,
        }

        print(f"\n  {variant_name.upper()} SUMMARY:", flush=True)
        print(f"    T=10: {mean_t10:.4f}  T=20: {mean_t20:.4f}  T=32: {mean_t32:.4f}",
              flush=True)
        print(f"    Mean recovery: {mean_recovery:+.4f}", flush=True)
        print(f"    Noise robustness (tau={NOISE_TAU}): {mean_noise:.4f}", flush=True)

    # Final comparison
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D22 FINAL COMPARISON", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  {'Variant':<12} {'T=10':>8} {'T=20':>8} {'T=32':>8} {'Recovery':>10} {'Noise':>8}",
          flush=True)
    for vname in variants:
        s = all_results[vname]["summary"]
        print(f"  {vname:<12} {s['mean_T10_seq']:>8.4f} {s['mean_T20_seq']:>8.4f} "
              f"{s['mean_T32_seq']:>8.4f} {s['mean_recovery']:>+10.4f} "
              f"{s['mean_noise_robustness']:>8.4f}", flush=True)

    baseline_rec = all_results["baseline"]["summary"]["mean_recovery"]
    best_rec_variant = max(variants, key=lambda v: all_results[v]["summary"]["mean_recovery"])
    best_rec = all_results[best_rec_variant]["summary"]["mean_recovery"]

    print(f"\n  RECOVERY IMPROVEMENT: {best_rec_variant} ({best_rec:+.4f}) vs baseline ({baseline_rec:+.4f})",
          flush=True)
    if best_rec > 0:
        print(f"  >>> POSITIVE RECOVERY ACHIEVED! This is a breakthrough. <<<", flush=True)
    elif best_rec > baseline_rec:
        print(f"  >>> Recovery improved but still negative. Partial success. <<<", flush=True)
    else:
        print(f"  >>> No recovery improvement. Training modifications insufficient. <<<", flush=True)

    baseline_t32 = all_results["baseline"]["summary"]["mean_T32_seq"]
    best_t32_variant = max(variants, key=lambda v: all_results[v]["summary"]["mean_T32_seq"])
    best_t32 = all_results[best_t32_variant]["summary"]["mean_T32_seq"]

    print(f"  WINDOW WIDENING: {best_t32_variant} T=32={best_t32:.4f} vs baseline T=32={baseline_t32:.4f}",
          flush=True)
    if best_t32 > 0.95:
        print(f"  >>> Compute window extended to T=32! <<<", flush=True)

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d22_robust_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
