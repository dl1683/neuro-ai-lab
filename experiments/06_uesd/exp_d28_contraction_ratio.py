"""
Experiment D28: Banach Contraction Ratio Measurement

Measures per-step contraction ratio k_t = ||s_{t+1} - s*|| / ||s_t - s*|| for each carry depth.

Key question: Is the dynamics map G a contraction mapping with constant ratio k?
- If k ≈ 0.4 and constant across D → Banach theorem explains universal T_99=5
  (k^5 < 0.01 means 99% convergence in 5 steps)
- If k varies with D → contraction is problem-dependent, need separate T_99 explanation
- If k varies with t → non-geometric convergence, Banach model insufficient
- If k > 1 at early steps → transient expansion before contraction

Derived from research mining: Banach contraction mapping theorem + Edge-of-Stability.
Also measures edge-of-stability: is ρ slightly supercritical because generation benefits
from mild signal amplification? (Cohen et al. 2021, Priesemann et al. 2014)

Design:
- L = {4, 8, 12, 16, 20, 24} (carry depths 2, 4, 6, 8, 10, 12)
- TWO training variants per L:
  * fixed_t: CE-dynamics at T=10, 20K steps (original)
  * variable_t: T sampled from {4,6,8,10,12,14,16} each batch, 20K steps (controls for horizon bias)
- Fixed point: T=100 iterations from same s_0 (approximate s*)
- Contraction: track ||s_t - s*|| for t=0..30, compute k_t ratios
- 1 seed, 4096 eval samples

PREDICTIONS (Banach hypothesis):
1. k ≈ 0.35-0.45, constant across D=2-12
2. k is approximately constant across t (geometric convergence)
3. k^5 * ||s_0 - s*|| < readout margin (explains T_99=5)
4. Slight supercriticality (ρ > 1) at linearization, but global contraction
5. k_fixed ≈ k_variable (contraction is architectural, not horizon artifact)
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

FP_RELATIVE_THRESHOLD = 1e-4  # warn if fixed-point residual exceeds this

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEED = 42
TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
TRAIN_T = 10
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
VARIANTS = ["fixed_t", "variable_t"]

SEQ_LENS = [4, 8, 12, 16, 20, 24]
EVAL_SAMPLES = 4096
FP_T = 100  # steps to approximate fixed point
TRAJECTORY_T = 30  # steps to track contraction

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d28_contraction_ratio.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_results(results):
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


def train_model(model, device, seq_len, variant="fixed_t"):
    """Train with fixed or variable T."""
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if variant == "variable_t":
            t_steps = random.choice(VARIABLE_T_RANGE)
        else:
            t_steps = TRAIN_T

        logits = model(src, t_steps)
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
                eval_logits = model(src, TRAIN_T)
                preds = eval_logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            t_label = f"T={t_steps}" if variant == "variable_t" else f"T={TRAIN_T}"
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | {t_label} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


@torch.no_grad()
def measure_contraction(model, device, seq_len):
    """Core measurement: per-step contraction ratio toward fixed point.

    Returns:
        dict with per-step distances, contraction ratios, and summary stats.
    """
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    context = model.encode(src)

    # 1. Compute approximate fixed point s* by running T=FP_T steps
    s_star = model.init_state(src.size(0), src.size(1), device)
    for _ in range(FP_T):
        s_star, _ = model.dynamics_step(s_star, context)

    # Check convergence: run 10 more steps and measure change
    s_check = s_star.clone()
    for _ in range(10):
        s_check, _ = model.dynamics_step(s_check, context)
    # Use Frobenius norm over full state (B, L, d) flattened to (B, L*d)
    diff_flat = (s_check - s_star).reshape(src.size(0), -1)
    star_flat = s_star.reshape(src.size(0), -1)
    fp_residual = diff_flat.norm(dim=-1).mean().item()
    s_star_norm = star_flat.norm(dim=-1).mean().item()
    relative_residual = fp_residual / max(s_star_norm, 1e-8)
    fp_converged = relative_residual < FP_RELATIVE_THRESHOLD
    print(f"    Fixed-point residual: {fp_residual:.6f} (relative: {relative_residual:.6f})"
          f"{'' if fp_converged else ' *** WARNING: NOT CONVERGED ***'}")

    # 2. Track trajectory from s_0 and measure distances to s*
    # Use Frobenius norm over full state for Banach-consistent measurement
    s = model.init_state(src.size(0), src.size(1), device)
    distances = []  # ||s_t - s*|| (Frobenius over L*d)
    update_norms = []  # ||s_{t+1} - s_t||
    contraction_ratios = []  # ||s_{t+1} - s*|| / ||s_t - s*||

    # Distance at t=0
    dist_0 = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)  # [B]
    distances.append({
        "mean": dist_0.mean().item(),
        "std": dist_0.std().item(),
        "median": dist_0.median().item(),
    })

    for t in range(TRAJECTORY_T):
        s_prev = s.clone()
        s, update_norm = model.dynamics_step(s, context)

        dist_t = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)  # [B]
        upd_norm = (s - s_prev).reshape(src.size(0), -1).norm(dim=-1)  # [B]

        distances.append({
            "mean": dist_t.mean().item(),
            "std": dist_t.std().item(),
            "median": dist_t.median().item(),
        })
        update_norms.append({
            "mean": upd_norm.mean().item(),
            "std": upd_norm.std().item(),
        })

        # Contraction ratio from mean distances
        prev_dist = distances[-2]["mean"]
        curr_dist = dist_t.mean().item()
        if prev_dist > 1e-8:
            k = curr_dist / prev_dist
        else:
            k = 0.0
        contraction_ratios.append(round(k, 6))

    # 3. Per-sample contraction ratios (Frobenius norm)
    s = model.init_state(src.size(0), src.size(1), device)
    per_sample_k = []
    prev_dist_per_sample = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)  # [B]

    for t in range(min(TRAJECTORY_T, 20)):
        s, _ = model.dynamics_step(s, context)
        curr_dist_per_sample = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)  # [B]
        k_per = curr_dist_per_sample / prev_dist_per_sample.clamp(min=1e-8)
        per_sample_k.append({
            "mean": k_per.mean().item(),
            "std": k_per.std().item(),
            "min": k_per.min().item(),
            "max": k_per.max().item(),
            "q25": k_per.quantile(0.25).item(),
            "q75": k_per.quantile(0.75).item(),
        })
        prev_dist_per_sample = curr_dist_per_sample

    # 4. Check readout accuracy along trajectory
    s = model.init_state(src.size(0), src.size(1), device)
    readout_trajectory = []
    tgt_result = tgt[:, :half]
    for t in range(TRAJECTORY_T):
        s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        preds = logits[:, :half].argmax(dim=-1)
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        readout_trajectory.append(round(seq_acc, 4))

    # 5. Compute summary statistics
    # Mean k over steps 2-10 (avoid initialization transient at step 1)
    stable_k = contraction_ratios[1:10] if len(contraction_ratios) > 1 else contraction_ratios
    mean_k = float(np.mean(stable_k))
    std_k = float(np.std(stable_k))

    # Predicted T_99 from Banach: k^T < 0.01 → T > log(0.01) / log(k)
    if 0 < mean_k < 1:
        predicted_T99 = np.log(0.01) / np.log(mean_k)
    else:
        predicted_T99 = float("inf")

    # Find actual T_99 from readout trajectory
    actual_T99 = None
    for t, acc in enumerate(readout_trajectory, 1):
        if acc >= 0.99:
            actual_T99 = t
            break

    carry_depth = seq_len // 2
    is_contractive = mean_k < 1.0
    summary = {
        "carry_depth": carry_depth,
        "mean_k": round(mean_k, 4),
        "std_k": round(std_k, 4),
        "is_contractive": is_contractive,
        "predicted_T99": round(predicted_T99, 2) if predicted_T99 < 100 else "inf",
        "actual_T99": actual_T99,
        "fp_converged": fp_converged,
        "fp_residual": round(relative_residual, 8),
        "s_star_norm": round(s_star_norm, 4),
    }
    status = "CONTRACTIVE" if is_contractive else "NON-CONTRACTIVE"
    fp_status = "" if fp_converged else " [FP NOT CONVERGED]"
    print(f"    {status}: mean k={mean_k:.4f}±{std_k:.4f}, predicted T_99={predicted_T99:.1f}, "
          f"actual T_99={actual_T99}{fp_status}", flush=True)

    return {
        "distances": distances,
        "update_norms": update_norms,
        "contraction_ratios": contraction_ratios,
        "per_sample_k": per_sample_k,
        "readout_trajectory": readout_trajectory,
        "summary": summary,
    }


def measure_spectral_radius(model, device, seq_len, n_vecs=10, n_iter=50):
    """Estimate spectral radius via power iteration on the Jacobian at s*."""
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 256, seq_len, VOCAB_SIZE)
    src = src.to(device)

    with torch.no_grad():
        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)
        for _ in range(FP_T):
            s, _ = model.dynamics_step(s, context)
        s_star = s.detach().clone()

    rhos = []
    for _ in range(n_vecs):
        v = torch.randn_like(s_star[:32])
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for _ in range(n_iter):
            with torch.enable_grad():
                s_pert = s_star[:32].detach().requires_grad_(True)
                s_next, _ = model.dynamics_step(s_pert, context[:32])
                jvp = torch.autograd.grad(
                    s_next, s_pert,
                    grad_outputs=v,
                    create_graph=False,
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    std_rho = float(np.std(rhos))
    print(f"    Spectral radius: rho={mean_rho:.4f}+/-{std_rho:.4f}")
    return {"mean": round(mean_rho, 4), "std": round(std_rho, 4)}


def run_one_config(seq_len, device, variant="fixed_t"):
    """Train model, measure contraction ratio and spectral radius."""
    carry_depth = seq_len // 2
    print(f"\n  L={seq_len} (carry depth D={carry_depth}), variant={variant}", flush=True)

    full_seed(SEED)
    model = make_model(device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    # Train
    train_time, best_acc = train_model(model, device, seq_len, variant)

    # Measure contraction
    print("    Measuring contraction ratios...", flush=True)
    contraction = measure_contraction(model, device, seq_len)

    # Measure spectral radius
    print("    Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device, seq_len)

    return {
        "seq_len": seq_len,
        "carry_depth": carry_depth,
        "variant": variant,
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_acc, 4),
        "contraction": contraction,
        "spectral_radius": spectral,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    # Check for resume
    existing = load_checkpoint()
    completed_keys = set()
    if existing and "runs" in existing:
        for run in existing["runs"]:
            key = f"{run['seq_len']}_{run.get('variant', 'fixed_t')}"
            completed_keys.add(key)
        print(f"Resuming: {len(completed_keys)}/{len(SEQ_LENS) * len(VARIANTS)} configs complete")
    else:
        existing = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D28: Banach contraction ratio — does G contract geometrically toward s*? "
                       "Includes fixed-T vs variable-T control to rule out horizon bias.",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "variable_t_range": VARIABLE_T_RANGE,
                "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
                "fp_steps": FP_T, "trajectory_steps": TRAJECTORY_T,
            },
            "runs": [],
        }

    for seq_len in SEQ_LENS:
        for variant in VARIANTS:
            key = f"{seq_len}_{variant}"
            if key in completed_keys:
                print(f"\n  L={seq_len} {variant} already complete, skipping", flush=True)
                continue

            print(f"\n{'='*60}", flush=True)
            print(f"  L={seq_len} (D={seq_len // 2}), variant={variant}", flush=True)
            print(f"{'='*60}", flush=True)

            result = run_one_config(seq_len, device, variant)
            existing["runs"].append(result)
            save_results(existing)

    # Final summary — split by variant
    print(f"\n{'='*60}", flush=True)
    print(f"  D28 CONTRACTION RATIO SUMMARY", flush=True)
    print(f"{'='*60}\n", flush=True)

    for variant in VARIANTS:
        variant_runs = [r for r in existing["runs"] if r.get("variant", "fixed_t") == variant]
        if not variant_runs:
            continue

        print(f"  --- {variant.upper()} ---")
        print(f"  {'D':>3s}  {'k_mean':>7s}  {'k_std':>6s}  {'pred_T99':>9s}  {'actual_T99':>11s}  {'rho':>6s}  {'acc':>6s}")
        print(f"  {'---':>3s}  {'------':>7s}  {'-----':>6s}  {'--------':>9s}  {'----------':>11s}  {'---':>6s}  {'---':>6s}")

        for run in variant_runs:
            s = run["contraction"]["summary"]
            rho = run["spectral_radius"]["mean"]
            d = s["carry_depth"]
            acc = run["best_train_acc"]
            print(f"  {d:>3d}  {s['mean_k']:>7.4f}  {s['std_k']:>6.4f}  "
                  f"{str(s['predicted_T99']):>9s}  {str(s['actual_T99']):>11s}  "
                  f"{rho:>6.4f}  {acc:>6.4f}")

        k_values = [r["contraction"]["summary"]["mean_k"] for r in variant_runs]
        k_mean = float(np.mean(k_values))
        k_std = float(np.std(k_values))
        k_cv = k_std / max(k_mean, 1e-8)
        print(f"  Mean k = {k_mean:.4f} ± {k_std:.4f} (CV={k_cv:.3f})\n")

    # Cross-variant comparison
    fixed_runs = [r for r in existing["runs"] if r.get("variant", "fixed_t") == "fixed_t"]
    variable_runs = [r for r in existing["runs"] if r.get("variant") == "variable_t"]

    if fixed_runs and variable_runs:
        print(f"  --- FIXED vs VARIABLE T COMPARISON ---")
        fixed_by_d = {r["contraction"]["summary"]["carry_depth"]: r for r in fixed_runs}
        variable_by_d = {r["contraction"]["summary"]["carry_depth"]: r for r in variable_runs}

        print(f"  {'D':>3s}  {'k_fixed':>8s}  {'k_var':>8s}  {'dk':>8s}  {'rho_fix':>8s}  {'rho_var':>8s}")
        print(f"  {'---':>3s}  {'-------':>8s}  {'-----':>8s}  {'--':>8s}  {'-------':>8s}  {'-----':>8s}")

        deltas = []
        for d in sorted(set(fixed_by_d.keys()) & set(variable_by_d.keys())):
            k_f = fixed_by_d[d]["contraction"]["summary"]["mean_k"]
            k_v = variable_by_d[d]["contraction"]["summary"]["mean_k"]
            rho_f = fixed_by_d[d]["spectral_radius"]["mean"]
            rho_v = variable_by_d[d]["spectral_radius"]["mean"]
            dk = k_v - k_f
            deltas.append(dk)
            print(f"  {d:>3d}  {k_f:>8.4f}  {k_v:>8.4f}  {dk:>+8.4f}  {rho_f:>8.4f}  {rho_v:>8.4f}")

        mean_delta = float(np.mean(deltas))
        print(f"\n  Mean dk (variable - fixed) = {mean_delta:+.4f}")
        if abs(mean_delta) < 0.05:
            print(f"  >>> CONTRACTION IS ARCHITECTURAL: k does not depend on T_train <<<")
        else:
            direction = "STRONGER" if mean_delta < 0 else "WEAKER"
            print(f"  >>> CONTRACTION IS TRAINING-DEPENDENT: variable-T makes contraction {direction} <<<")

    # Overall Banach hypothesis (using fixed_t runs as primary)
    primary_runs = fixed_runs if fixed_runs else existing["runs"]
    k_values = [r["contraction"]["summary"]["mean_k"] for r in primary_runs]
    k_global_mean = float(np.mean(k_values))
    k_global_std = float(np.std(k_values))
    k_cv = k_global_std / max(k_global_mean, 1e-8)

    print(f"\n  Overall (fixed_t): mean k = {k_global_mean:.4f} ± {k_global_std:.4f} (CV={k_cv:.3f})")

    if k_cv < 0.15:
        print(f"  >>> BANACH HYPOTHESIS SUPPORTED: k is constant across D (CV < 15%) <<<")
    elif k_cv < 0.30:
        print(f"  >>> BANACH HYPOTHESIS PARTIAL: k varies moderately (CV 15-30%) <<<")
    else:
        print(f"  >>> BANACH HYPOTHESIS REJECTED: k varies significantly (CV > 30%) <<<")

    # Check predicted vs actual T_99
    for run in primary_runs:
        s = run["contraction"]["summary"]
        pred = s["predicted_T99"]
        actual = s["actual_T99"]
        if isinstance(pred, (int, float)) and actual is not None:
            ratio = pred / actual
            status = "MATCH" if abs(ratio - 1) < 0.3 else "MISMATCH"
            print(f"  D={s['carry_depth']}: predicted T_99={pred:.1f} vs actual={actual} — {status}")

    # Save summaries
    fixed_k = [r["contraction"]["summary"]["mean_k"] for r in fixed_runs] if fixed_runs else []
    var_k = [r["contraction"]["summary"]["mean_k"] for r in variable_runs] if variable_runs else []
    existing["banach_summary"] = {
        "global_mean_k": round(k_global_mean, 4),
        "global_std_k": round(k_global_std, 4),
        "coefficient_of_variation": round(k_cv, 4),
        "hypothesis_supported": k_cv < 0.15,
        "fixed_t_mean_k": round(float(np.mean(fixed_k)), 4) if fixed_k else None,
        "variable_t_mean_k": round(float(np.mean(var_k)), 4) if var_k else None,
        "horizon_independent": abs(float(np.mean(fixed_k)) - float(np.mean(var_k))) < 0.05 if (fixed_k and var_k) else None,
    }
    save_results(existing)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
