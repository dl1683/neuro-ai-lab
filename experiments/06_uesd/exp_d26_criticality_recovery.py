"""
Experiment D26: Criticality-Targeted Recovery (Nishimori-Conditioned Training)

NOVEL CONTRIBUTION: First experiment to explicitly target the spectral radius
rho = tanh(1/2) = 0.462 during dynamics training via a differentiable loss term.

Theory (Proposition 21): Recovery capacity is maximized at the Nishimori critical
point where the dynamics Jacobian spectral radius equals tanh(1/2). CE-only training
naturally approaches this point (D6: rho ~0.49-0.53) but overshoots. D25's recovery
training may push rho toward this value implicitly. D26 tests whether EXPLICIT
targeting outperforms implicit.

Mechanism: The spectral penalty uses a differentiable approximation:
  1. At converged state s_T, inject small noise eps * v (v = unit vector)
  2. Compute amplification: ||G(s_T + eps*v) - G(s_T)|| / ||eps*v||
  3. This approximates ||J*v|| / ||v|| for the random direction v
  4. Penalty: L_spec = lambda_s * (amplification - tanh(1/2))^2

Three key mechanisms from _meta/Open Exploration research:
  1. Spectral radius targeting (Prop 21) — directly controls basin geometry
  2. Recovery loss (D25/Thm 14) — shapes dynamics at perturbed states
  3. Calibration self-consistency (Nishimori condition) — confidence = accuracy

Variants:
  A. Recovery + spectral targeting (primary Prop 21 test)
  B. Spectral targeting only (does criticality alone produce recovery?)
  C. Recovery + spectral + calibration (full Nishimori)
  D. Recovery only (D25-style baseline for controlled comparison)

Each x 3 seeds. SEQ_LEN=8 (same as D25 for direct comparison).

PREDICTIONS:
1. Variant A outperforms D on recovery metrics (spectral targeting helps)
2. Variant B shows SOME basin improvement but no positive recovery (Thm 13 still applies)
3. Variant C has best overall performance (full Nishimori alignment)
4. All spectral-targeted variants have rho closer to 0.462 than untargeted
5. Calibration error (confidence - accuracy) is smallest for variant C
"""
import json
import math
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

NISHIMORI_RHO = math.tanh(0.5)  # 0.46211715...

EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 32]
EVAL_SAMPLES = 4096
RECOVERY_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
RECOVERY_EXTRA_STEPS = [1, 5, 10, 20]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d26_criticality_recovery.json"

SPEC_EPS = 0.01
SPEC_EVERY = 5


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
    frac = min(step / total_steps, 1.0)
    return sigma_start + (sigma_end - sigma_start) * frac


def compute_spectral_penalty(model, s, context):
    """Differentiable spectral radius estimate via noise amplification.
    Returns penalty (rho_est - tanh(1/2))^2 and the estimated rho."""
    with torch.no_grad():
        v = torch.randn_like(s)
        noise_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_unit = v / noise_norm
        eps_v = SPEC_EPS * v_unit

    s_clean = s.detach()
    s_noisy = s_clean + eps_v

    out_noisy, _ = model.dynamics_step(s_noisy, context)
    out_clean, _ = model.dynamics_step(s_clean, context)

    diff = out_noisy - out_clean
    amplification = diff.norm(dim=-1) / (SPEC_EPS + 1e-10)
    rho_est = amplification.mean()

    penalty = (rho_est - NISHIMORI_RHO) ** 2
    return penalty, rho_est.item()


def compute_calibration_penalty(logits, targets, half):
    """Nishimori self-consistency: |mean_confidence - mean_accuracy|^2."""
    probs = F.softmax(logits[:, :half].reshape(-1, logits.size(-1)), dim=-1)
    confidence = probs.max(dim=-1).values.mean()

    preds = logits[:, :half].argmax(dim=-1)
    accuracy = (preds == targets).float().mean()

    return (confidence - accuracy) ** 2


def train_criticality_recovery(model, device, seed, cfg):
    """Train with optional recovery loss, spectral targeting, and calibration."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0
    rho_history = []

    use_recovery = cfg.get("sigma_end", 0) > 0
    use_spectral = cfg.get("lambda_s", 0) > 0
    use_calibration = cfg.get("lambda_c", 0) > 0
    sigma_end = cfg.get("sigma_end", 0.1)
    extra_k = cfg.get("extra_k", 5)
    lambda_s = cfg.get("lambda_s", 0.1)
    lambda_c = cfg.get("lambda_c", 0.01)

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(T_RANGE)
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)
        tgt_result = tgt[:, :half]

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(T):
            s, _ = model.dynamics_step(s, context)

        logits_clean = model.readout_logits(s)
        task_loss = F.cross_entropy(
            logits_clean[:, :half].reshape(-1, logits_clean.size(-1)),
            tgt_result.reshape(-1),
        )

        loss = task_loss

        if use_recovery:
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
            loss = loss + recovery_loss

        rho_est = 0.0
        if use_spectral and step % SPEC_EVERY == 0:
            spec_penalty, rho_est = compute_spectral_penalty(model, s, context)
            loss = loss + lambda_s * spec_penalty
            rho_history.append(rho_est)

        if use_calibration:
            cal_penalty = compute_calibration_penalty(logits_clean, tgt_result, half)
            loss = loss + lambda_c * cal_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits_clean[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            rho_str = f"rho={rho_est:.3f}" if use_spectral else "rho=N/A"
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | "
                  f"loss={loss.item():.4f} | task={task_loss.item():.4f} | "
                  f"{rho_str} | Seq: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    rho_final = sum(rho_history[-20:]) / max(len(rho_history[-20:]), 1) if rho_history else 0
    return elapsed, best_acc, rho_final, rho_history


VARIANTS = {
    "plain_control": {
        "desc": "Variable-T only, no recovery or spectral (D22-style baseline)",
        "sigma_end": 0.0,
        "extra_k": 0,
        "lambda_s": 0.0,
        "lambda_c": 0.0,
    },
    "recovery_only": {
        "desc": "Recovery only (D25-style baseline for controlled comparison)",
        "sigma_end": 0.1,
        "extra_k": 5,
        "lambda_s": 0.0,
        "lambda_c": 0.0,
    },
    "spectral_only": {
        "desc": "Spectral targeting only (does criticality alone help?)",
        "sigma_end": 0.0,
        "extra_k": 0,
        "lambda_s": 0.1,
        "lambda_c": 0.0,
    },
    "recovery_spectral": {
        "desc": "Recovery + spectral targeting (Prop 21 primary test)",
        "sigma_end": 0.1,
        "extra_k": 5,
        "lambda_s": 0.1,
        "lambda_c": 0.0,
    },
    "full_nishimori": {
        "desc": "Recovery + spectral + calibration (full Nishimori alignment)",
        "sigma_end": 0.1,
        "extra_k": 5,
        "lambda_s": 0.1,
        "lambda_c": 0.01,
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
        results[f"T={T}"] = {"tok_acc": round(tok_acc, 4), "seq_acc": round(seq_acc, 4)}
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
            extra_results[f"+{extra}"] = {"WA": round(wa_after, 4), "recovery": round(recovery, 4)}

        results[f"sigma={sigma}"] = {"WA_at_0": round(wa_at_0, 4), **extra_results}

    return results


@torch.no_grad()
def estimate_spectral_radius_eval(model, device, n_samples=512, n_measure=64, power_iters=20, eps=1e-4):
    """Power-iteration spectral radius estimate for evaluation (non-differentiable)."""
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
    return {
        "mean_rho": round(mean_rho, 4),
        "std_rho": round(std_rho, 4),
        "nishimori_target": round(NISHIMORI_RHO, 4),
        "distance_to_nishimori": round(abs(mean_rho - NISHIMORI_RHO), 4),
        "n_samples": len(rhos),
    }


def run_one(variant_name, variant_cfg, seed, device):
    print(f"\n  variant={variant_name}, seed={seed}", flush=True)
    print(f"    {variant_cfg['desc']}", flush=True)

    model = make_model(device)
    params = count_params(model)

    train_time, best_acc, rho_final, rho_history = train_criticality_recovery(
        model, device, seed, variant_cfg
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

    print("    Estimating spectral radius (power iteration)...", flush=True)
    spec_radius = estimate_spectral_radius_eval(model, device)
    print(f"      rho={spec_radius['mean_rho']:.4f} ± {spec_radius['std_rho']:.4f} "
          f"(target={spec_radius['nishimori_target']}, "
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
        "spectral_radius": spec_radius,
        "rho_training_final": round(rho_final, 4),
        "rho_training_history_last20": [round(r, 4) for r in rho_history[-20:]],
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
            "purpose": "D26: Criticality-targeted recovery — Nishimori-conditioned training",
            "nishimori_rho_target": round(NISHIMORI_RHO, 6),
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                "lr": LR, "seq_len": SEQ_LEN, "t_range": T_RANGE,
                "spec_eps": SPEC_EPS, "spec_every": SPEC_EVERY,
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

    print(f"\n{'=' * 60}", flush=True)
    print(f"  D26 CRITICALITY RECOVERY SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for variant_name in VARIANTS:
        runs = [r for r in all_results["runs"] if r["variant"] == variant_name]
        if not runs:
            continue

        mean_rho = sum(r["spectral_radius"]["mean_rho"] for r in runs) / len(runs)
        mean_dist = sum(r["spectral_radius"]["distance_to_nishimori"] for r in runs) / len(runs)
        t10_accs = [r["step_ablation"].get("T=10", {}).get("seq_acc", 0) for r in runs]

        all_recoveries = []
        for r in runs:
            for sigma_key, sigma_data in r["recovery"].items():
                for k, v in sigma_data.items():
                    if k != "WA_at_0":
                        all_recoveries.append(v["recovery"])

        mean_t10 = sum(t10_accs) / len(t10_accs)
        mean_rec = sum(all_recoveries) / len(all_recoveries) if all_recoveries else 0
        max_rec = max(all_recoveries) if all_recoveries else 0

        print(f"\n  {variant_name}:", flush=True)
        print(f"    T10 acc: {mean_t10:.4f}", flush=True)
        print(f"    rho: {mean_rho:.4f} (dist to Nishimori: {mean_dist:.4f})", flush=True)
        print(f"    Recovery mean: {mean_rec:+.4f}, max: {max_rec:+.4f}", flush=True)

    print(f"\n  PROPOSITION 21 TEST:", flush=True)
    for variant_name in ["recovery_spectral", "full_nishimori"]:
        targeted = [r for r in all_results["runs"] if r["variant"] == variant_name]
        baseline = [r for r in all_results["runs"] if r["variant"] == "recovery_only"]
        if targeted and baseline:
            t_rho = sum(r["spectral_radius"]["mean_rho"] for r in targeted) / len(targeted)
            b_rho = sum(r["spectral_radius"]["mean_rho"] for r in baseline) / len(baseline)
            print(f"    {variant_name} rho={t_rho:.4f} vs baseline rho={b_rho:.4f} "
                  f"(target={NISHIMORI_RHO:.4f})", flush=True)

    save_checkpoint(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    return all_results


if __name__ == "__main__":
    run()
