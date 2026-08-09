"""
D23b: Multi-seed carry-depth scaling at L=20,24.

Extends D23 (single seed=42) with seeds {42, 1337, 2024} at the large-L
configurations where generalizability is weakest. Variable-T only (the
critical variant for Prop 32). Adds spectral radius measurement.

Tests:
1. T_99 universality across seeds (is T_99=3 at L=20 seed-independent?)
2. T_99=5 crossover at L=24 across seeds (is this robust?)
3. rho consistency across seeds (Cor 30.1 replication)
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

TRAINING_STEPS = 30000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64

SEQ_LENS = [20, 24]
SEEDS = [42, 1337, 2024]
EVAL_T_VALUES = [1, 2, 3, 4, 5, 8, 10, 15, 20, 32, 48]
EVAL_SAMPLES = 4096

POWER_ITER_VECS = 10
POWER_ITER_STEPS = 50
FP_T = 100

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d23b_multiseed.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(results):
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def get_t_range(seq_len):
    base = [4, 6, 8, 10, 12, 14, 16]
    carry_depth = seq_len // 2
    if carry_depth > 8:
        extra = list(range(18, min(carry_depth * 3, 48) + 1, 2))
        return base + extra
    return base


def train_variable_t(model, device, seq_len, seed):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    t_range = get_t_range(seq_len)
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(t_range)
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
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
            with torch.no_grad():
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f} | {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


@torch.no_grad()
def evaluate_step_ablation(model, device, seq_len):
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    results = {}
    for T in EVAL_T_VALUES:
        logits = model(src, T)
        preds = logits[:, :half].argmax(dim=-1)
        tok_acc = (preds == tgt_result).float().mean().item()
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        results[f"T={T}"] = {
            "tok_acc": round(tok_acc, 6),
            "seq_acc": round(seq_acc, 6),
        }
    return results


def compute_t99(step_ablation):
    for T in sorted(EVAL_T_VALUES):
        key = f"T={T}"
        if key in step_ablation and step_ablation[key]["seq_acc"] >= 0.99:
            return T
    return None


@torch.no_grad()
def measure_spectral_radius(model, device, seq_len):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 32, seq_len, VOCAB_SIZE)
    src = src.to(device)

    context = model.encode(src)
    s = model.init_state(32, seq_len, device)
    for _ in range(FP_T):
        s, _ = model.dynamics_step(s, context)
    s_star = s.detach().clone()

    rhos = []
    for _ in range(POWER_ITER_VECS):
        v = torch.randn_like(s_star)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for _ in range(POWER_ITER_STEPS):
            with torch.enable_grad():
                s_pert = s_star.detach().requires_grad_(True)
                s_next, _ = model.dynamics_step(s_pert, context)
                jvp = torch.autograd.grad(
                    s_next, s_pert, grad_outputs=v, create_graph=False
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    return {
        "mean": round(float(np.mean(rhos)), 4),
        "std": round(float(np.std(rhos)), 4),
    }


def run_one_config(seq_len, seed, device):
    print(f"\n{'='*60}", flush=True)
    print(f"  L={seq_len} (D={seq_len//2}), seed={seed}, variable_t", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(seed)
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN).to(device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    train_time, best_acc = train_variable_t(model, device, seq_len, seed)

    print("    Evaluating step ablation...", flush=True)
    step_abl = evaluate_step_ablation(model, device, seq_len)
    t99 = compute_t99(step_abl)
    print(f"    T_99 = {t99}", flush=True)
    for k, v in sorted(step_abl.items(), key=lambda x: int(x[0].split('=')[1])):
        print(f"      {k}: seq={v['seq_acc']:.4f}", flush=True)

    print("    Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device, seq_len)
    print(f"    rho = {spectral['mean']:.4f} +/- {spectral['std']:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "carry_depth": seq_len // 2,
        "seed": seed,
        "variant": "variable_t",
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_acc, 6),
        "step_ablation": step_abl,
        "T_99": t99,
        "spectral_radius": spectral,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed = {(r["seq_len"], r["seed"]) for r in all_results["runs"]}
        print(f"\n  RESUMING: {len(completed)} configs already done", flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D23b: Multi-seed carry-depth scaling (L=20,24 x 3 seeds)",
            "architecture": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            },
            "training": {
                "steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                "lr": LR, "variant": "variable_t",
            },
            "runs": [],
        }
        completed = set()

    for seq_len in SEQ_LENS:
        for seed in SEEDS:
            if (seq_len, seed) in completed:
                print(f"\n  L={seq_len} seed={seed} already done, skipping", flush=True)
                continue
            result = run_one_config(seq_len, seed, device)
            all_results["runs"].append(result)
            save_checkpoint(all_results)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  D23b MULTI-SEED SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n  {'L':>3} {'Seed':>6} {'T_99':>5} {'rho':>7} {'Train Acc':>10}", flush=True)
    print(f"  {'-'*35}", flush=True)

    for r in sorted(all_results["runs"], key=lambda x: (x["seq_len"], x["seed"])):
        t99_str = str(r["T_99"]) if r["T_99"] is not None else "DNF"
        print(f"  {r['seq_len']:>3} {r['seed']:>6} {t99_str:>5} "
              f"{r['spectral_radius']['mean']:>7.4f} {r['best_train_acc']:>10.6f}", flush=True)

    # Cross-seed consistency
    for seq_len in SEQ_LENS:
        runs = [r for r in all_results["runs"] if r["seq_len"] == seq_len]
        if len(runs) >= 2:
            t99s = [r["T_99"] for r in runs if r["T_99"] is not None]
            rhos = [r["spectral_radius"]["mean"] for r in runs]
            if t99s:
                print(f"\n  L={seq_len}: T_99 = {t99s} (unanimous={len(set(t99s))==1}), "
                      f"rho = {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}", flush=True)

    save_checkpoint(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    run()
