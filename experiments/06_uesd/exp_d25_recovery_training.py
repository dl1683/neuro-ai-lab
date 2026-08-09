"""
Experiment D25: Recovery-First Training — Targeting T6 (Causal Repair)

Recovery is the #1 gap in the UESD thesis (T6 at 2/10). All prior experiments show
negative or zero recovery: dynamics cannot self-correct after perturbation.

Codex D22 review insight: recovery fails because no training objective enforces it.
Variable-T gives horizon robustness, but not basin recovery.

This experiment adds explicit perturbation-recovery training:
  1. Run dynamics for T steps (standard CE-dynamics or variable-T)
  2. Perturb the final state s_T with noise (sigma * ||s_T||)
  3. Run K extra dynamics steps from the perturbed state
  4. Add recovery loss: CE on the readout AFTER extra steps from perturbed state
  5. Total loss = alpha * task_loss + (1-alpha) * recovery_loss

Sigma curriculum: start very small (0.01) and linearly increase to sigma_max.
This avoids the D22 denoising failure (sigma=0.3 killed learning entirely).

Variants:
  A. Variable-T + recovery (sigma curriculum 0.01->0.1, K=5 extra steps)
  B. Variable-T + recovery (sigma curriculum 0.01->0.2, K=10 extra steps)
  C. Variable-T + recovery (sigma curriculum 0.01->0.1, K=5, alpha schedule 1.0->0.5)
  D. Variable-T only (D22-style baseline for comparison)

PREDICTIONS:
1. At least one variant achieves positive mean recovery (first time ever)
2. Sigma curriculum prevents the D22 denoising catastrophe
3. Recovery training slightly slows task accuracy convergence but preserves final accuracy
4. Longer extra steps (K=10) help recovery more than larger sigma
5. (Prop 21) Recovery training pushes spectral radius toward tanh(1/2) ≈ 0.462
6. (Prop 21) recovery_weighted variant should have ρ closest to tanh(1/2)
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEED = 42
SEEDS = [42, 1337, 2024]
TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
SEQ_LEN = 8
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 32
TRAIN_T = 10

T_RANGE = [4, 6, 8, 10, 12, 14, 16]

EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 32]
EVAL_SAMPLES = 4096
RECOVERY_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
RECOVERY_EXTRA_STEPS = [1, 5, 10, 20]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d25_recovery_training.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)


def save_checkpoint(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


def sigma_schedule(step, total_steps, sigma_start, sigma_end):
    """Linear sigma curriculum."""
    frac = min(step / total_steps, 1.0)
    return sigma_start + (sigma_end - sigma_start) * frac


def alpha_schedule(step, total_steps, alpha_start, alpha_end):
    """Linear alpha (task vs recovery weight) schedule."""
    frac = min(step / total_steps, 1.0)
    return alpha_start + (alpha_end - alpha_start) * frac


def train_variable_t_only(model, device, seed):
    """D22-style variable-T baseline (no recovery training)."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(T_RANGE)
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
            with torch.no_grad():
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    return elapsed, best_acc


def train_recovery(model, device, seed, sigma_end, extra_k, alpha_end):
    """Variable-T + perturbation-recovery training."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(T_RANGE)
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)
        tgt_result = tgt[:, :half]

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(T):
            s, _ = model.dynamics_step(s, context)

        # Task loss on clean state
        logits_clean = model.readout_logits(s)
        task_loss = F.cross_entropy(
            logits_clean[:, :half].reshape(-1, logits_clean.size(-1)),
            tgt_result.reshape(-1),
        )

        # Recovery: perturb and run extra steps
        sigma = sigma_schedule(step, TRAINING_STEPS, 0.01, sigma_end)
        state_norm = s.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(s) * (sigma * state_norm)
        s_perturbed = s + noise

        s_recovery = s_perturbed
        for _ in range(extra_k):
            s_recovery, _ = model.dynamics_step(s_recovery, context)

        logits_recovery = model.readout_logits(s_recovery)
        recovery_loss = F.cross_entropy(
            logits_recovery[:, :half].reshape(-1, logits_recovery.size(-1)),
            tgt_result.reshape(-1),
        )

        alpha = alpha_schedule(step, TRAINING_STEPS, 1.0, alpha_end)
        loss = alpha * task_loss + (1.0 - alpha) * recovery_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits_clean[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | sigma={sigma:.3f} | "
                  f"alpha={alpha:.2f} | task={task_loss.item():.4f} | "
                  f"rec={recovery_loss.item():.4f} | Seq: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    return elapsed, best_acc


VARIANTS = {
    "variable_t_only": {
        "desc": "D22-style variable-T baseline",
        "train_fn": "variable_t_only",
    },
    "recovery_gentle": {
        "desc": "Variable-T + recovery (sigma 0.01->0.1, K=5, alpha=1.0 fixed)",
        "sigma_end": 0.1,
        "extra_k": 5,
        "alpha_end": 1.0,
        "train_fn": "recovery",
    },
    "recovery_stronger": {
        "desc": "Variable-T + recovery (sigma 0.01->0.2, K=10, alpha=1.0 fixed)",
        "sigma_end": 0.2,
        "extra_k": 10,
        "alpha_end": 1.0,
        "train_fn": "recovery",
    },
    "recovery_weighted": {
        "desc": "Variable-T + recovery (sigma 0.01->0.1, K=5, alpha 1.0->0.5)",
        "sigma_end": 0.1,
        "extra_k": 5,
        "alpha_end": 0.5,
        "train_fn": "recovery",
    },
}


@torch.no_grad()
def evaluate_step_ablation(model, device):
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    results = {}
    for T in EVAL_T_VALUES:
        logits = model(src, T)
        preds = logits[:, :half].argmax(dim=-1)
        tok_acc = (preds == tgt_result).float().mean().item()
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        results[f"T={T}"] = {
            "tok_acc": round(tok_acc, 4),
            "seq_acc": round(seq_acc, 4),
        }
    return results


@torch.no_grad()
def evaluate_recovery(model, device):
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    for _ in range(10):
        s, _ = model.dynamics_step(s, context)

    state_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    results = {}
    for sigma in RECOVERY_SIGMAS:
        noise = torch.randn_like(s) * (sigma * state_norm)
        s_noisy = s + noise

        preds_noisy = model.readout_logits(s_noisy)[:, :half].argmax(dim=-1)
        wa_at_0 = 1.0 - (preds_noisy == tgt_result).all(dim=1).float().mean().item()

        s_extra = s_noisy.clone()
        extra_results = {}
        prev = 0
        for extra in RECOVERY_EXTRA_STEPS:
            for _ in range(extra - prev):
                s_extra, _ = model.dynamics_step(s_extra, context)
            prev = extra
            preds_extra = model.readout_logits(s_extra)[:, :half].argmax(dim=-1)
            wa_after = 1.0 - (preds_extra == tgt_result).all(dim=1).float().mean().item()
            recovery = wa_at_0 - wa_after
            extra_results[f"+{extra}"] = {
                "WA": round(wa_after, 4),
                "recovery": round(recovery, 4),
            }

        results[f"sigma={sigma}"] = {
            "WA_at_0": round(wa_at_0, 4),
            **extra_results,
        }

    return results


@torch.no_grad()
def estimate_spectral_radius(model, device, n_samples=512, n_measure=64, power_iters=20, eps=1e-4):
    """Estimate spectral radius of dynamics Jacobian at near-converged states via power iteration.
    Tests Proposition 21: recovery training should push rho toward tanh(1/2) ≈ 0.462.
    Note: measures at T=10 states (proxy for fixed point), not true s*. Sufficient for
    relative comparison between variants per Codex review."""
    model.eval()
    full_seed(8888)
    src, tgt = generate_batch("addition", n_samples, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    for _ in range(10):
        s, _ = model.dynamics_step(s, context)

    indices = torch.randperm(n_samples)[:n_measure]
    rhos = []
    for idx in indices:
        s_i = s[idx:idx+1]
        ctx_i = context[idx:idx+1]
        v = torch.randn_like(s_i)
        v = v / v.norm()

        lam = 0.0
        for _ in range(power_iters):
            s_plus = s_i + eps * v
            s_minus = s_i - eps * v
            out_plus, _ = model.dynamics_step(s_plus, ctx_i)
            out_minus, _ = model.dynamics_step(s_minus, ctx_i)
            Jv = (out_plus - out_minus) / (2 * eps)
            lam_new = Jv.norm().item()
            if lam_new < 1e-8:
                break
            v = Jv / Jv.norm()
            if abs(lam_new - lam) / max(lam_new, 1e-10) < 1e-4:
                lam = lam_new
                break
            lam = lam_new

        rhos.append(lam)

    mean_rho = sum(rhos) / len(rhos)
    std_rho = (sum((r - mean_rho) ** 2 for r in rhos) / len(rhos)) ** 0.5
    nishimori_dist = abs(mean_rho - 0.4621715)
    return {
        "mean_rho": round(mean_rho, 4),
        "std_rho": round(std_rho, 4),
        "nishimori_target": 0.4622,
        "distance_to_nishimori": round(nishimori_dist, 4),
        "n_samples": len(rhos),
    }


@torch.no_grad()
def evaluate_noise_robustness(model, device, tau=0.05):
    """Cosine-annealed noise during full inference."""
    import math
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    T = 10
    for t in range(1, T + 1):
        s, _ = model.dynamics_step(s, context)
        noise_scale = tau * (0.5 + 0.5 * math.cos(math.pi * t / T))
        s = s + torch.randn_like(s) * noise_scale

    logits = model.readout_logits(s)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
    return {"tok_acc": round(tok_acc, 4), "seq_acc": round(seq_acc, 4)}


def run_one(variant_name, variant_cfg, seed, device):
    """Train and evaluate one (variant, seed) configuration."""
    print(f"\n  variant={variant_name}, seed={seed}", flush=True)
    print(f"    {variant_cfg['desc']}", flush=True)

    model = make_model(device)
    params = count_params(model)

    if variant_cfg["train_fn"] == "variable_t_only":
        train_time, best_acc = train_variable_t_only(model, device, seed)
    else:
        train_time, best_acc = train_recovery(
            model, device, seed,
            sigma_end=variant_cfg["sigma_end"],
            extra_k=variant_cfg["extra_k"],
            alpha_end=variant_cfg["alpha_end"],
        )

    print("    Evaluating step ablation...", flush=True)
    step_abl = evaluate_step_ablation(model, device)
    for k, v in step_abl.items():
        print(f"      {k}: seq={v['seq_acc']:.4f}", flush=True)

    print("    Evaluating recovery...", flush=True)
    recovery = evaluate_recovery(model, device)
    for sigma_key, sigma_data in recovery.items():
        best_rec = max(v["recovery"] for k, v in sigma_data.items() if k != "WA_at_0")
        print(f"      {sigma_key}: WA@0={sigma_data['WA_at_0']:.4f} best_recovery={best_rec:+.4f}",
              flush=True)

    print("    Evaluating noise robustness...", flush=True)
    noise_rob = evaluate_noise_robustness(model, device)
    print(f"      noise_robustness: seq={noise_rob['seq_acc']:.4f}", flush=True)

    print("    Estimating spectral radius (Prop 21)...", flush=True)
    spec_radius = estimate_spectral_radius(model, device)
    print(f"      rho={spec_radius['mean_rho']:.4f} ± {spec_radius['std_rho']:.4f} "
          f"(Nishimori target={spec_radius['nishimori_target']}, "
          f"dist={spec_radius['distance_to_nishimori']:.4f})", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "variant": variant_name,
        "seed": seed,
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_acc, 4),
        "step_ablation": step_abl,
        "recovery": recovery,
        "noise_robustness": noise_rob,
        "spectral_radius": spec_radius,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed_keys = {(r["variant"], r["seed"]) for r in all_results["runs"]}
        print(f"\n  RESUMING: {len(all_results['runs'])} runs already done", flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D25: Recovery-first training — targeting T6 causal repair",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "batch_size": BATCH_SIZE, "lr": LR, "seq_len": SEQ_LEN,
                "t_range": T_RANGE,
            },
            "variants": {k: v["desc"] for k, v in VARIANTS.items()},
            "runs": [],
        }
        completed_keys = set()

    for variant_name, variant_cfg in VARIANTS.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"  VARIANT: {variant_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seed in SEEDS:
            if (variant_name, seed) in completed_keys:
                print(f"\n  {variant_name} seed={seed} — SKIPPED (already done)", flush=True)
                continue
            result = run_one(variant_name, variant_cfg, seed, device)
            all_results["runs"].append(result)
            save_checkpoint(all_results)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D25 RECOVERY TRAINING SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for variant_name in VARIANTS:
        runs = [r for r in all_results["runs"] if r["variant"] == variant_name]
        if not runs:
            continue

        t10_accs = [r["step_ablation"].get("T=10", {}).get("seq_acc", 0) for r in runs]
        t32_accs = [r["step_ablation"].get("T=32", {}).get("seq_acc", 0) for r in runs]

        all_recoveries = []
        for r in runs:
            for sigma_key, sigma_data in r["recovery"].items():
                for k, v in sigma_data.items():
                    if k != "WA_at_0":
                        all_recoveries.append(v["recovery"])

        mean_t10 = sum(t10_accs) / len(t10_accs)
        mean_t32 = sum(t32_accs) / len(t32_accs)
        mean_rec = sum(all_recoveries) / len(all_recoveries) if all_recoveries else 0
        max_rec = max(all_recoveries) if all_recoveries else 0
        pos_rec_pct = sum(1 for r in all_recoveries if r > 0) / len(all_recoveries) * 100 if all_recoveries else 0

        print(f"\n  {variant_name}:", flush=True)
        print(f"    T10 mean: {mean_t10:.4f}", flush=True)
        print(f"    T32 mean: {mean_t32:.4f}", flush=True)
        print(f"    Recovery mean: {mean_rec:+.4f}", flush=True)
        print(f"    Recovery max:  {max_rec:+.4f}", flush=True)
        print(f"    Positive recovery: {pos_rec_pct:.1f}%", flush=True)

    # Highlight: did ANY variant achieve positive mean recovery?
    print(f"\n  RECOVERY BREAKTHROUGH CHECK:", flush=True)
    any_positive = False
    for variant_name in VARIANTS:
        runs = [r for r in all_results["runs"] if r["variant"] == variant_name]
        if not runs:
            continue
        all_rec = []
        for r in runs:
            for sigma_key, sigma_data in r["recovery"].items():
                for k, v in sigma_data.items():
                    if k != "WA_at_0":
                        all_rec.append(v["recovery"])
        mean_rec = sum(all_rec) / len(all_rec) if all_rec else 0
        if mean_rec > 0:
            print(f"    {variant_name}: POSITIVE RECOVERY ({mean_rec:+.4f}) !!!", flush=True)
            any_positive = True
        else:
            print(f"    {variant_name}: {mean_rec:+.4f} (still negative)", flush=True)

    if not any_positive:
        print(f"\n    No variant achieved positive mean recovery.", flush=True)
        print(f"    Recovery remains the #1 gap in UESD thesis.", flush=True)
    else:
        print(f"\n    FIRST POSITIVE RECOVERY IN UESD HISTORY!", flush=True)

    save_checkpoint(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    return all_results


if __name__ == "__main__":
    run()
