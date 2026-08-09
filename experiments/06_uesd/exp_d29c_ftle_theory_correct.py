"""
D29c: Theory-correct FTLE analysis — addresses all 3 HIGH-severity Codex issues from D29b.

Fixes vs D29b:
1. [HIGH] Readout normalization Jacobian: includes D_norm(h) in critical direction
   computation. The readout uses F.normalize(h), so margin gradient through s
   must account for (I - h_hat @ h_hat^T) / ||h||.
2. [HIGH] Theory-correct FTLE partition: instead of P @ Phi @ P (two-sided
   projection), we compute full SVD of Phi and partition singular vectors by
   their alignment with V_R, exactly as stated in Proposition 31.
3. [HIGH] Fixed-point convergence: increased to 500 steps with convergence
   gate. Samples where fixed point doesn't converge are flagged and excluded.

Also fixes:
- [MEDIUM] Checks ALL singular vectors for alignment, not just top-100
- [MEDIUM] Uses threshold 0.5 from Prop 31, not 0.3
- [LOW] Uses relative FP tolerance (1e-6) instead of absolute
- Reports both theory-correct and D29b-style metrics for comparison

Note on P_R definition: We use the margin-critical subspace (dim ~4) rather
than the full readout Jacobian row space (dim ~64 per position). The full row
space was D29's approach and it failed (covered 50% of state space, making
partition meaningless). The margin-critical subspace captures exactly the
directions that matter for classification correctness.
"""
import json
import math
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

TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
SEQ_LEN = 8
SEED = 42

FP_STEPS = 500
FP_REL_TOL = 1e-6
ANALYSIS_T_VALUES = [1, 2, 3, 4, 5, 8, 10, 15, 20]
N_ANALYSIS_SAMPLES = 8

ALIGNMENT_THRESHOLD = 0.5  # Prop 31 definition

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d29c_ftle_theory_correct.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)


def get_t_range(seq_len):
    return [4, 6, 8, 10, 12, 14, 16]


def train_variable_t(model, device):
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    t_range = get_t_range(SEQ_LEN)
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(t_range)
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
            elapsed = time.time() - t0
            print(f"  Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Acc: {seq_acc:.4f} | {elapsed:.0f}s", flush=True)

    return time.time() - t0


def compute_jacobian_single(model, s, context):
    L, d = s.shape[1], s.shape[2]
    n = L * d
    s_input = s.detach().clone().requires_grad_(True)
    s_new, _ = model.dynamics_step(s_input, context)
    s_new_flat = s_new.reshape(-1)

    J = torch.zeros(n, n, device=s.device)
    for i in range(n):
        grad = torch.autograd.grad(
            s_new_flat[i], s_input,
            retain_graph=(i < n - 1),
            create_graph=False,
        )[0]
        J[i] = grad.reshape(-1)
    return J


def compute_readout_subspace(model, s_star, tgt, device):
    """Compute readout-relevant subspace V_R with normalization Jacobian.

    For each output position l, computes the margin-critical direction:
        d_l = W_R^T @ D_norm(h_l)^T @ (e_correct - e_second)
    where D_norm is the Jacobian of the normalization operation.

    Returns:
        P_R: (n, n) projection onto readout-relevant subspace
        dim_R: dimension of the subspace
        d29b_P_R: (n, n) D29b-style projection (without D_norm) for comparison
    """
    L, d = s_star.shape[1], s_star.shape[2]
    n = L * d
    half = SEQ_LEN // 2

    with torch.no_grad():
        W_R = model.readout_proj.weight  # (d, d)
        b_R = model.readout_proj.bias    # (d,) or None
        E_norm = F.normalize(model.tok_emb.weight, dim=-1)  # (V, d)

        correct_dirs = []
        d29b_dirs = []

        for l in range(half):
            s_l = s_star[0, l]  # (d,)

            # Raw readout hidden state (before normalization)
            h_raw = W_R @ s_l
            if b_R is not None:
                h_raw = h_raw + b_R

            h_norm_val = h_raw.norm().clamp(min=1e-10)
            h_hat = h_raw / h_norm_val  # unit vector

            # Jacobian of normalize: D_norm(h) = (I - h_hat h_hat^T) / ||h||
            D_normalize = (torch.eye(d, device=device) - h_hat.outer(h_hat)) / h_norm_val

            # Logits after normalization
            h_normalized = F.normalize(h_raw.unsqueeze(0), dim=-1).squeeze(0)
            logits_l = E_norm @ h_normalized  # (V,)

            y_correct = tgt[l].item()
            logits_copy = logits_l.clone()
            logits_copy[y_correct] = -float('inf')
            y_second = logits_copy.argmax().item()

            delta_e = E_norm[y_correct] - E_norm[y_second]  # (d,)

            # Theory-correct critical direction (with D_norm)
            d_critical = W_R.T @ D_normalize.T @ delta_e
            d_critical = d_critical / d_critical.norm().clamp(min=1e-10)

            # D29b-style critical direction (without D_norm, for comparison)
            d_29b = W_R.T @ delta_e
            d_29b = d_29b / d_29b.norm().clamp(min=1e-10)

            # Embed into full state space
            full_dir = torch.zeros(n, device=device)
            full_dir[l * d:(l + 1) * d] = d_critical
            correct_dirs.append(full_dir)

            full_dir_29b = torch.zeros(n, device=device)
            full_dir_29b[l * d:(l + 1) * d] = d_29b
            d29b_dirs.append(full_dir_29b)

        if len(correct_dirs) == 0:
            I_zero = torch.zeros(n, n, device=device)
            return I_zero, 0, I_zero

        # Build orthonormal projections
        D_correct = torch.stack(correct_dirs)  # (half, n)
        Q, _ = torch.linalg.qr(D_correct.T)
        P_R = Q @ Q.T

        D_29b = torch.stack(d29b_dirs)
        Q_29b, _ = torch.linalg.qr(D_29b.T)
        P_R_29b = Q_29b @ Q_29b.T

    return P_R, Q.shape[1], P_R_29b


def find_fixed_point(model, device, src_single):
    """Find fixed point with convergence gate."""
    with torch.no_grad():
        context = model.encode(src_single.unsqueeze(0))
        s = model.init_state(1, SEQ_LEN, device)

        fp_steps_taken = 0
        fp_residual = float('inf')
        fp_rel_residual = float('inf')

        for fp_step in range(FP_STEPS):
            s_prev = s.clone()
            s, _ = model.dynamics_step(s, context)
            delta = (s - s_prev).norm().item()
            s_norm = s.norm().item()
            fp_residual = delta
            fp_rel_residual = delta / max(s_norm, 1e-10)
            fp_steps_taken = fp_step + 1

            if fp_rel_residual < FP_REL_TOL:
                break

        converged = fp_rel_residual < FP_REL_TOL
        s_star = s.detach().clone()

    return s_star, context, converged, fp_residual, fp_rel_residual, fp_steps_taken


def ftle_analysis_theory_correct(model, device, src, tgt_single, max_T):
    """Compute FTLE with theory-correct alignment partition (Prop 31)."""
    L, d = SEQ_LEN, D_MODEL
    n = L * d
    half = SEQ_LEN // 2

    # Find fixed point with convergence gate
    s_star, context, fp_converged, fp_residual, fp_rel_residual, fp_steps = find_fixed_point(
        model, device, src
    )

    if not fp_converged:
        print(f"    WARNING: FP not converged (rel_residual={fp_rel_residual:.2e}, "
              f"abs_residual={fp_residual:.4f}, steps={fp_steps})", flush=True)

    # Compute readout subspace (with and without D_norm correction)
    P_R, dim_R, P_R_29b = compute_readout_subspace(model, s_star, tgt_single, device)

    # Check readout correctness at s*
    with torch.no_grad():
        logits_star = model.readout_logits(s_star)
        preds = logits_star[0, :half].argmax(dim=-1)
        readout_correct = (preds == tgt_single[:half]).all().item()

        margins = []
        for l_pos in range(half):
            logits_l = logits_star[0, l_pos]
            y_c = tgt_single[l_pos].item()
            margin = logits_l[y_c] - logits_l.clone().scatter_(
                0, torch.tensor([y_c], device=device), -float('inf')
            ).max()
            margins.append(margin.item())

    # Accumulate transition matrix Phi_{0,T}
    model.eval()
    results_by_T = {}
    Phi = torch.eye(n, device=device)

    for T in range(1, max_T + 1):
        with torch.no_grad():
            s_curr = s_star.clone() if T == 1 else s_prev.clone()

        J_t = compute_jacobian_single(model, s_curr, context)
        Phi = J_t @ Phi

        with torch.no_grad():
            s_next, _ = model.dynamics_step(s_curr, context)
            s_prev = s_next.detach().clone()

        if T not in ANALYSIS_T_VALUES:
            continue

        with torch.no_grad():
            # Full SVD of transition matrix
            U, S_vals, Vh = torch.linalg.svd(Phi)
            ftle = torch.log(S_vals.clamp(min=1e-20)) / T

            # === THEORY-CORRECT PARTITION (Prop 31) ===
            # For each right singular vector v_i, compute alignment with V_R
            # v_i are ROWS of Vh
            readout_aligned_ftles = []
            null_aligned_ftles = []

            for i in range(n):
                v_i = Vh[i]  # right singular vector
                alignment = (P_R @ v_i).norm().item() / v_i.norm().clamp(min=1e-10).item()

                if alignment > ALIGNMENT_THRESHOLD:
                    readout_aligned_ftles.append(ftle[i].item())
                else:
                    null_aligned_ftles.append(ftle[i].item())

            lambda_R_theory = max(readout_aligned_ftles) if readout_aligned_ftles else float('nan')
            lambda_perp_theory = max(null_aligned_ftles) if null_aligned_ftles else float('nan')
            n_readout = len(readout_aligned_ftles)
            n_null = len(null_aligned_ftles)

            # === D29b-STYLE (without D_norm) for comparison ===
            readout_29b_ftles = []
            null_29b_ftles = []
            for i in range(n):
                v_i = Vh[i]
                alignment_29b = (P_R_29b @ v_i).norm().item() / v_i.norm().clamp(min=1e-10).item()
                if alignment_29b > ALIGNMENT_THRESHOLD:
                    readout_29b_ftles.append(ftle[i].item())
                else:
                    null_29b_ftles.append(ftle[i].item())

            lambda_R_29b = max(readout_29b_ftles) if readout_29b_ftles else float('nan')
            n_readout_29b = len(readout_29b_ftles)

            # === D29b-STYLE PROJECTION (P @ Phi @ P) for comparison ===
            Phi_proj = P_R @ Phi @ P_R
            U_p, S_p, _ = torch.linalg.svd(Phi_proj)
            S_p_nz = S_p[S_p > 1e-20]
            lambda_R_projected = torch.log(S_p_nz[0]).item() / T if len(S_p_nz) > 0 else float('nan')

            # Alignment distribution for top singular vectors
            top_alignments = []
            for i in range(min(20, n)):
                v_i = Vh[i]
                a = (P_R @ v_i).norm().item() / v_i.norm().clamp(min=1e-10).item()
                top_alignments.append(round(a, 4))

            results_by_T[T] = {
                "lambda_max": round(ftle[0].item(), 6),
                "theory_correct": {
                    "lambda_R": round(lambda_R_theory, 6) if not math.isnan(lambda_R_theory) else None,
                    "lambda_perp": round(lambda_perp_theory, 6) if not math.isnan(lambda_perp_theory) else None,
                    "gap": round(lambda_perp_theory - lambda_R_theory, 6) if not (math.isnan(lambda_R_theory) or math.isnan(lambda_perp_theory)) else None,
                    "n_readout_aligned": n_readout,
                    "n_null_aligned": n_null,
                },
                "d29b_comparison": {
                    "lambda_R_no_dnorm": round(lambda_R_29b, 6) if not math.isnan(lambda_R_29b) else None,
                    "n_readout_no_dnorm": n_readout_29b,
                    "lambda_R_projected": round(lambda_R_projected, 6) if not math.isnan(lambda_R_projected) else None,
                },
                "top_20_alignments": top_alignments,
                "top_5_ftle": [round(x, 6) for x in ftle[:5].tolist()],
            }

        lr_str = f"{lambda_R_theory:.6f}" if not math.isnan(lambda_R_theory) else "N/A"
        lp_str = f"{lambda_perp_theory:.6f}" if not math.isnan(lambda_perp_theory) else "N/A"
        print(f"    T={T:>2d}: lam_R={lr_str}, lam_perp={lp_str}, "
              f"n_R={n_readout}, n_null={n_null}, lam_max={ftle[0].item():.6f}",
              flush=True)

    results_by_T["_meta"] = {
        "fp_converged": fp_converged,
        "fp_residual": round(fp_residual, 8),
        "fp_rel_residual": round(fp_rel_residual, 10),
        "fp_steps": fp_steps,
        "readout_correct_at_sstar": readout_correct,
        "margins_at_sstar": [round(m, 4) for m in margins],
        "dim_R": dim_R,
        "alignment_threshold": ALIGNMENT_THRESHOLD,
    }
    return results_by_T


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  D29c: Theory-correct FTLE analysis", flush=True)
    print(f"  Fixes: D_norm Jacobian + alignment partition + FP convergence", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n  Phase 1: Training variable_t model (L={SEQ_LEN}, seed={SEED})", flush=True)

    full_seed(SEED)
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN).to(device)
    params = count_params(model)
    print(f"  Params: {params}", flush=True)

    train_time = train_variable_t(model, device)
    print(f"  Training done in {train_time:.1f}s", flush=True)

    print(f"\n  Phase 2: Theory-correct FTLE analysis ({N_ANALYSIS_SAMPLES} samples)", flush=True)
    print(f"  Alignment threshold: {ALIGNMENT_THRESHOLD} (Prop 31 definition)", flush=True)
    print(f"  FP rel tolerance: {FP_REL_TOL}, max steps: {FP_STEPS}", flush=True)

    full_seed(9999)
    src, tgt = generate_batch("addition", N_ANALYSIS_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    all_sample_results = []
    converged_count = 0
    for i in range(N_ANALYSIS_SAMPLES):
        print(f"\n  Sample {i+1}/{N_ANALYSIS_SAMPLES}:", flush=True)
        t0 = time.time()
        sample_result = ftle_analysis_theory_correct(
            model, device, src[i], tgt[i], max(ANALYSIS_T_VALUES)
        )
        elapsed = time.time() - t0
        all_sample_results.append(sample_result)
        if sample_result["_meta"]["fp_converged"]:
            converged_count += 1
        print(f"    Done in {elapsed:.1f}s (FP: {'converged' if sample_result['_meta']['fp_converged'] else 'NOT converged'})",
              flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 3: Aggregation ({converged_count}/{N_ANALYSIS_SAMPLES} samples converged)", flush=True)
    print(f"{'='*60}", flush=True)

    # Aggregate only converged samples
    converged_results = [r for r in all_sample_results if r["_meta"]["fp_converged"]]
    all_results_for_agg = all_sample_results  # also aggregate all for comparison

    aggregated = {}
    for T in ANALYSIS_T_VALUES:
        # Theory-correct (converged only)
        lam_R_vals = [r[T]["theory_correct"]["lambda_R"] for r in converged_results
                      if T in r and r[T]["theory_correct"]["lambda_R"] is not None]
        lam_perp_vals = [r[T]["theory_correct"]["lambda_perp"] for r in converged_results
                         if T in r and r[T]["theory_correct"]["lambda_perp"] is not None]

        # Theory-correct (all samples)
        lam_R_all = [r[T]["theory_correct"]["lambda_R"] for r in all_results_for_agg
                     if T in r and r[T]["theory_correct"]["lambda_R"] is not None]
        lam_perp_all = [r[T]["theory_correct"]["lambda_perp"] for r in all_results_for_agg
                        if T in r and r[T]["theory_correct"]["lambda_perp"] is not None]

        # D29b comparison (also converged only for fair comparison)
        lam_R_29b = [r[T]["d29b_comparison"]["lambda_R_projected"] for r in converged_results
                     if T in r and r[T]["d29b_comparison"]["lambda_R_projected"] is not None]

        aggregated[T] = {
            "theory_correct_converged": {
                "lambda_R_mean": round(float(np.mean(lam_R_vals)), 6) if lam_R_vals else None,
                "lambda_R_std": round(float(np.std(lam_R_vals)), 6) if lam_R_vals else None,
                "lambda_perp_mean": round(float(np.mean(lam_perp_vals)), 6) if lam_perp_vals else None,
                "lambda_perp_std": round(float(np.std(lam_perp_vals)), 6) if lam_perp_vals else None,
                "n": len(lam_R_vals),
                "all_R_negative": all(v < 0 for v in lam_R_vals) if lam_R_vals else None,
                "all_perp_positive": all(v > 0 for v in lam_perp_vals) if lam_perp_vals else None,
            },
            "theory_correct_all": {
                "lambda_R_mean": round(float(np.mean(lam_R_all)), 6) if lam_R_all else None,
                "lambda_R_std": round(float(np.std(lam_R_all)), 6) if lam_R_all else None,
                "n": len(lam_R_all),
                "all_R_negative": all(v < 0 for v in lam_R_all) if lam_R_all else None,
            },
            "d29b_projected_mean": round(float(np.mean(lam_R_29b)), 6) if lam_R_29b else None,
        }

        if lam_R_vals:
            mean_R = np.mean(lam_R_vals)
            mean_P = np.mean(lam_perp_vals) if lam_perp_vals else float('nan')
            print(f"  T={T:>2d}: lam_R={mean_R:+.6f} (n={len(lam_R_vals)}), "
                  f"lam_perp={mean_P:+.6f}, "
                  f"all_R<0={all(v < 0 for v in lam_R_vals)}", flush=True)

    # Prop 31 verdict
    # Check at T=5 (our reference point from D29b)
    ref_T = 5
    if ref_T in aggregated and aggregated[ref_T]["theory_correct_converged"]["lambda_R_mean"] is not None:
        tc = aggregated[ref_T]["theory_correct_converged"]
        lam_R = tc["lambda_R_mean"]
        lam_perp = tc["lambda_perp_mean"]
        all_neg = tc["all_R_negative"]
        all_pos = tc["all_perp_positive"]

        if all_neg and all_pos and lam_R < 0 and lam_perp > 0:
            verdict = "CONFIRMED"
        elif lam_R < 0 and lam_perp > 0:
            verdict = "SUPPORTED"
        elif lam_R < 0:
            verdict = "PARTIAL"
        else:
            verdict = "REFUTED"
    else:
        verdict = "INSUFFICIENT_DATA"

    print(f"\n  *** PROPOSITION 31 VERDICT (D29c theory-correct): {verdict} ***", flush=True)
    if ref_T in aggregated and aggregated[ref_T]["theory_correct_converged"]["lambda_R_mean"] is not None:
        tc = aggregated[ref_T]["theory_correct_converged"]
        print(f"  lambda_R = {tc['lambda_R_mean']:+.6f} +/- {tc['lambda_R_std']:.6f} "
              f"(n={tc['n']}, converged only)", flush=True)
        print(f"  lambda_perp = {tc['lambda_perp_mean']:+.6f} +/- {tc['lambda_perp_std']:.6f}", flush=True)

    # Compare with D29b
    if ref_T in aggregated and aggregated[ref_T]["d29b_projected_mean"] is not None:
        print(f"  D29b P@Phi@P = {aggregated[ref_T]['d29b_projected_mean']:+.6f} (for comparison)", flush=True)

    # Save
    final_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D29c: Theory-correct FTLE analysis (fixes D29b HIGH issues)",
        "fixes_applied": [
            "Readout normalization Jacobian D_norm(h) included in critical direction",
            "Theory-correct alignment partition (Prop 31 definition, threshold=0.5)",
            "Fixed-point convergence gate (tol=1e-4, max_steps=500)",
            "All singular vectors checked (not just top-100)",
        ],
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "seed": SEED,
            "variant": "variable_t", "L": SEQ_LEN,
            "fp_steps": FP_STEPS, "fp_rel_tol": FP_REL_TOL,
            "alignment_threshold": ALIGNMENT_THRESHOLD,
        },
        "converged_samples": converged_count,
        "total_samples": N_ANALYSIS_SAMPLES,
        "per_sample": all_sample_results,
        "aggregated": aggregated,
        "prop31_verdict": verdict,
    }

    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)

    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    return final_results


if __name__ == "__main__":
    run()
