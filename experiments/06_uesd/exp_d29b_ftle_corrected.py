"""
D29b: Corrected FTLE analysis with margin-critical direction decomposition.

D29 used alignment threshold 0.5 on a readout subspace that covers 50% of
state space (rank 64/128 per position). This misclassified ~99% of directions
as "readout-aligned", making lambda_R ≈ lambda_max (meaningless partition).

This script fixes the analysis by:
1. Using the MARGIN-CRITICAL direction (dim ~L/2) — the direction that would
   flip the readout argmax, not the full readout row space.
2. Computing FTLE projected onto these critical directions directly.
3. Also computing FTLE in the null space of the readout Jacobian (the
   directions with truly zero readout effect).

This reuses D29's trained model (loads from the checkpoint saved during D29,
or re-trains with same seed).
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

FP_STEPS = 100
ANALYSIS_T_VALUES = [1, 2, 3, 4, 5, 8, 10, 15, 20]
N_ANALYSIS_SAMPLES = 8

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d29b_ftle_corrected.json"


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


def compute_margin_critical_projection(model, s_star, tgt, device):
    """Compute projection onto MARGIN-CRITICAL directions.

    The margin-critical direction at position l is the state-space direction
    that most efficiently decreases the readout margin (the gap between the
    correct logit and the second-best logit).

    This gives dim = L/2 (one critical direction per output position),
    instead of dim = L*rank(M_eff) ~ L*64 from D29's row-space approach.
    """
    L, d = s_star.shape[1], s_star.shape[2]
    half = SEQ_LEN // 2

    with torch.no_grad():
        W_R = model.readout_proj.weight  # (d, d)
        E_norm = F.normalize(model.tok_emb.weight, dim=-1)  # (V, d)

        critical_dirs = []

        for l in range(half):
            s_l = s_star[0, l]  # (d,)
            h_l = W_R @ s_l  # (d,)
            if model.readout_proj.bias is not None:
                h_l = h_l + model.readout_proj.bias
            logits_l = E_norm @ h_l  # (V,)

            y_correct = tgt[l].item()
            logits_copy = logits_l.clone()
            logits_copy[y_correct] = -float('inf')
            y_second = logits_copy.argmax().item()

            e_correct = E_norm[y_correct]  # (d,)
            e_second = E_norm[y_second]  # (d,)
            delta_e = e_correct - e_second  # (d,)

            d_critical = W_R.T @ delta_e
            d_critical = d_critical / d_critical.norm().clamp(min=1e-10)

            full_dir = torch.zeros(L * d, device=device)
            full_dir[l * d:(l + 1) * d] = d_critical
            critical_dirs.append(full_dir)

        if len(critical_dirs) == 0:
            return torch.zeros(L * d, L * d, device=device), 0, []

        D_crit = torch.stack(critical_dirs)  # (half, n)
        Q, _ = torch.linalg.qr(D_crit.T)  # (n, half) orthonormal basis
        P_crit = Q @ Q.T  # (n, n) projection

    return P_crit, Q.shape[1], critical_dirs


def compute_null_space_projection(model, device):
    """Compute projection onto the NULL SPACE of the readout.

    The null space consists of directions with ZERO readout effect.
    dim(null) = d - rank(M_eff) per position.
    """
    L, d = SEQ_LEN, D_MODEL
    n = L * d

    with torch.no_grad():
        E_norm = F.normalize(model.tok_emb.weight, dim=-1)
        W_R = model.readout_proj.weight
        M_eff = E_norm @ W_R  # (V, d)

        U, S, Vh = torch.linalg.svd(M_eff, full_matrices=True)
        rank = (S > S[0] * 1e-6).sum().item()
        null_basis = Vh[rank:]  # (d - rank, d)
        P_null_per_pos = null_basis.T @ null_basis  # (d, d)

        P_null = torch.zeros(n, n, device=device)
        for l in range(L):
            start = l * d
            end = (l + 1) * d
            P_null[start:end, start:end] = P_null_per_pos

    null_dim = (d - rank) * L
    return P_null, null_dim, rank


def ftle_analysis_corrected(model, device, src, tgt_single, max_T):
    L, d = SEQ_LEN, D_MODEL
    n = L * d
    half = SEQ_LEN // 2

    with torch.no_grad():
        context = model.encode(src.unsqueeze(0))
        s = model.init_state(1, SEQ_LEN, device)
        fp_residual = float('inf')
        for fp_step in range(FP_STEPS):
            s_prev_fp = s.clone()
            s, _ = model.dynamics_step(s, context)
            fp_residual = (s - s_prev_fp).norm().item()
            if fp_residual < 1e-4:
                break
        s_star = s.detach().clone()

    P_crit, crit_dim, _ = compute_margin_critical_projection(
        model, s_star, tgt_single, device
    )
    P_null, null_dim, readout_rank = compute_null_space_projection(model, device)
    P_row = torch.eye(n, device=device) - P_null

    with torch.no_grad():
        logits_star = model.readout_logits(s_star)
        preds = logits_star[0, :half].argmax(dim=-1)
        readout_correct = (preds == tgt_single[:half]).all().item()

        margins = []
        for l in range(half):
            logits_l = logits_star[0, l]
            y_correct = tgt_single[l].item()
            margin = logits_l[y_correct] - logits_l.clone().scatter_(0, torch.tensor([y_correct], device=device), -float('inf')).max()
            margins.append(margin.item())

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

        if T in ANALYSIS_T_VALUES:
            with torch.no_grad():
                U, S_vals, Vh = torch.linalg.svd(Phi)
                ftle = torch.log(S_vals.clamp(min=1e-20)) / T

                V_dirs = Vh
                lambda_max = ftle[0].item()

                crit_alignments = torch.zeros(n, device=device)
                null_alignments = torch.zeros(n, device=device)
                row_alignments = torch.zeros(n, device=device)
                for i in range(min(n, 100)):
                    v_i = V_dirs[i]
                    crit_alignments[i] = (P_crit @ v_i).norm() / v_i.norm().clamp(min=1e-10)
                    null_alignments[i] = (P_null @ v_i).norm() / v_i.norm().clamp(min=1e-10)
                    row_alignments[i] = (P_row @ v_i).norm() / v_i.norm().clamp(min=1e-10)

                crit_threshold = 0.3
                null_threshold = 0.7

                crit_mask = crit_alignments[:100] > crit_threshold
                null_mask = null_alignments[:100] > null_threshold

                n_crit = crit_mask.sum().item()
                n_null = null_mask.sum().item()

                if n_crit > 0:
                    lambda_crit = ftle[:100][crit_mask].max().item()
                    lambda_crit_mean = ftle[:100][crit_mask].mean().item()
                else:
                    lambda_crit = lambda_crit_mean = float('nan')

                if n_null > 0:
                    lambda_null = ftle[:100][null_mask].max().item()
                    lambda_null_mean = ftle[:100][null_mask].mean().item()
                else:
                    lambda_null = lambda_null_mean = float('nan')

                Phi_crit = P_crit @ Phi @ P_crit
                U_c, S_c, _ = torch.linalg.svd(Phi_crit)
                S_c_nonzero = S_c[S_c > 1e-20]
                if len(S_c_nonzero) > 0:
                    lambda_R_direct = torch.log(S_c_nonzero[0]).item() / T
                    lambda_R_direct_min = torch.log(S_c_nonzero[-1]).item() / T
                else:
                    lambda_R_direct = lambda_R_direct_min = float('nan')

                Phi_null = P_null @ Phi @ P_null
                U_n, S_n, _ = torch.linalg.svd(Phi_null)
                S_n_nonzero = S_n[S_n > 1e-20]
                if len(S_n_nonzero) > 0:
                    lambda_null_direct = torch.log(S_n_nonzero[0]).item() / T
                else:
                    lambda_null_direct = float('nan')

                results_by_T[T] = {
                    "lambda_max": round(lambda_max, 6),
                    "rho_from_ftle": round(float(np.exp(lambda_max)), 6),

                    "margin_critical": {
                        "lambda_crit_max": round(lambda_crit, 6) if not np.isnan(lambda_crit) else None,
                        "lambda_crit_mean": round(lambda_crit_mean, 6) if not np.isnan(lambda_crit_mean) else None,
                        "n_crit_aligned": n_crit,
                        "lambda_R_direct": round(lambda_R_direct, 6) if not np.isnan(lambda_R_direct) else None,
                        "lambda_R_direct_min": round(lambda_R_direct_min, 6) if not np.isnan(lambda_R_direct_min) else None,
                        "dim_critical": crit_dim,
                    },

                    "readout_null": {
                        "lambda_null_max": round(lambda_null, 6) if not np.isnan(lambda_null) else None,
                        "lambda_null_mean": round(lambda_null_mean, 6) if not np.isnan(lambda_null_mean) else None,
                        "n_null_aligned": n_null,
                        "lambda_null_direct": round(lambda_null_direct, 6) if not np.isnan(lambda_null_direct) else None,
                        "dim_null": null_dim,
                    },

                    "top_10_crit_alignments": [round(x, 4) for x in crit_alignments[:10].tolist()],
                    "top_10_null_alignments": [round(x, 4) for x in null_alignments[:10].tolist()],
                    "top_5_ftle": [round(x, 6) for x in ftle[:5].tolist()],
                    "bottom_5_ftle": [round(x, 6) for x in ftle[-5:].tolist()],
                }

            crit_str = f"{lambda_crit:.4f}" if not np.isnan(lambda_crit) else "N/A"
            null_str = f"{lambda_null:.4f}" if not np.isnan(lambda_null) else "N/A"
            direct_str = f"{lambda_R_direct:.4f}" if not np.isnan(lambda_R_direct) else "N/A"
            print(f"    T={T:>2d}: lam_crit={crit_str}, lam_null={null_str}, "
                  f"lam_R_direct={direct_str}, lam_max={lambda_max:.4f}, "
                  f"n_crit={n_crit}, n_null={n_null}", flush=True)

    results_by_T["_meta"] = {
        "fp_residual": round(fp_residual, 8),
        "readout_correct_at_sstar": readout_correct,
        "margins_at_sstar": [round(m, 4) for m in margins],
        "crit_dim": crit_dim,
        "null_dim": null_dim,
        "readout_rank_per_pos": readout_rank,
    }
    return results_by_T


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 1: Training variable_t model (L={SEQ_LEN}, seed={SEED})", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(SEED)
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN).to(device)
    params = count_params(model)
    print(f"  Params: {params}", flush=True)

    train_time = train_variable_t(model, device)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 2: Corrected FTLE analysis ({N_ANALYSIS_SAMPLES} samples)", flush=True)
    print(f"  Margin-critical dim: ~{SEQ_LEN // 2}, "
          f"Null space dim: ~{(D_MODEL - 64) * SEQ_LEN}", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(9999)
    src, tgt = generate_batch("addition", N_ANALYSIS_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    half = SEQ_LEN // 2

    all_sample_results = []
    for i in range(N_ANALYSIS_SAMPLES):
        print(f"\n  Sample {i+1}/{N_ANALYSIS_SAMPLES}:", flush=True)
        t0 = time.time()
        sample_result = ftle_analysis_corrected(
            model, device, src[i], tgt[i], max(ANALYSIS_T_VALUES)
        )
        elapsed = time.time() - t0
        all_sample_results.append(sample_result)
        print(f"    Done in {elapsed:.1f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 3: Aggregation", flush=True)
    print(f"{'='*60}", flush=True)

    aggregated = {}
    for T in ANALYSIS_T_VALUES:
        lam_R_direct_vals = [
            r[T]["margin_critical"]["lambda_R_direct"]
            for r in all_sample_results
            if T in r and r[T]["margin_critical"]["lambda_R_direct"] is not None
        ]
        lam_null_direct_vals = [
            r[T]["readout_null"]["lambda_null_direct"]
            for r in all_sample_results
            if T in r and r[T]["readout_null"]["lambda_null_direct"] is not None
        ]
        lam_max_vals = [r[T]["lambda_max"] for r in all_sample_results if T in r]

        aggregated[T] = {
            "lambda_R_direct": {
                "mean": round(float(np.mean(lam_R_direct_vals)), 6) if lam_R_direct_vals else None,
                "std": round(float(np.std(lam_R_direct_vals)), 6) if lam_R_direct_vals else None,
                "n": len(lam_R_direct_vals),
            },
            "lambda_null_direct": {
                "mean": round(float(np.mean(lam_null_direct_vals)), 6) if lam_null_direct_vals else None,
                "std": round(float(np.std(lam_null_direct_vals)), 6) if lam_null_direct_vals else None,
                "n": len(lam_null_direct_vals),
            },
            "lambda_max": {
                "mean": round(float(np.mean(lam_max_vals)), 6) if lam_max_vals else None,
                "std": round(float(np.std(lam_max_vals)), 6) if lam_max_vals else None,
            },
        }

        if lam_R_direct_vals and lam_null_direct_vals:
            gap = np.mean(lam_null_direct_vals) - np.mean(lam_R_direct_vals)
            print(f"  T={T:>2d}: lam_R={np.mean(lam_R_direct_vals):.4f}+/-{np.std(lam_R_direct_vals):.4f}, "
                  f"lam_null={np.mean(lam_null_direct_vals):.4f}+/-{np.std(lam_null_direct_vals):.4f}, "
                  f"gap={gap:.4f}, lam_max={np.mean(lam_max_vals):.4f}", flush=True)

    prop31_result = "INCONCLUSIVE"
    lam_R_5 = aggregated.get(5, {}).get("lambda_R_direct", {}).get("mean")
    lam_null_5 = aggregated.get(5, {}).get("lambda_null_direct", {}).get("mean")
    if lam_R_5 is not None and lam_null_5 is not None:
        if lam_R_5 < 0 and lam_null_5 > 0:
            prop31_result = "CONFIRMED"
            print(f"\n  *** PROPOSITION 31 CONFIRMED (corrected analysis) ***", flush=True)
        elif lam_R_5 > 0 and lam_null_5 > 0:
            if lam_R_5 < lam_null_5:
                prop31_result = "PARTIAL — lambda_R > 0 but lambda_R < lambda_null"
            else:
                prop31_result = "REFUTED — lambda_R >= lambda_null"
        elif lam_R_5 < 0 and lam_null_5 < 0:
            prop31_result = "PARTIAL — both negative (global contraction)"
        print(f"  lambda_R_direct = {lam_R_5:.6f}", flush=True)
        print(f"  lambda_null_direct = {lam_null_5:.6f}", flush=True)
        print(f"  Verdict: {prop31_result}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D29b: Corrected FTLE analysis with margin-critical decomposition",
        "methodology_fix": (
            "D29 used readout row space (dim=512/1024) with threshold 0.5, "
            "classifying 99% of directions as readout-aligned. D29b uses "
            "margin-critical directions (dim~4) and readout null space "
            "(dim~512) for proper decomposition."
        ),
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "seed": SEED,
            "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
            "variant": "variable_t", "fp_steps": FP_STEPS,
            "n_analysis_samples": N_ANALYSIS_SAMPLES,
        },
        "ftle_analysis": {
            "per_sample": {str(i): r for i, r in enumerate(all_sample_results)},
            "aggregated": {str(k): v for k, v in aggregated.items()},
        },
        "prop31_verdict": {
            "result": prop31_result,
            "lambda_R_direct_at_T5": lam_R_5,
            "lambda_null_direct_at_T5": lam_null_5,
        },
    }

    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)

    print(f"\n  Results saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    run()
