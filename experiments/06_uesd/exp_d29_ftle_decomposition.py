"""
D29 FTLE Decomposition: Direct test of Proposition 31.

Measures Finite-Time Lyapunov Exponents partitioned by readout alignment.
Trains a variable_t model at L=8 (matching D25 setup), then:
1. Computes product Jacobian Phi_{0,T} at the readout-stable manifold
2. SVDs Phi to get FTLEs and their directions
3. Computes readout projection alignment for each FTLE direction
4. Partitions into lambda_R (readout-aligned) and lambda_perp (orthogonal)

Prop 31 predictions:
- lambda_R < 0 (readout subspace contracts)
- lambda_perp > 0 (orthogonal subspace expands)
- rho = exp(max FTLE) > 1, living in V_perp
- T_99 determined by lambda_R, not by rho
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

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d29_ftle_decomposition.json"


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
    """Compute full ds_new/ds Jacobian for a single sample.

    s: (1, L, d) state tensor
    context: (1, L, d) encoder context
    Returns: (n, n) Jacobian where n = L * d
    """
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


def compute_readout_projection(model, device):
    """Compute the readout-relevant subspace projection.

    The effective readout matrix is M_eff = E @ W_R (V x d).
    V_R per position = row_space(M_eff).
    Full projection is block-diagonal: P_R = I_L (x) P_r.
    """
    with torch.no_grad():
        E = model.tok_emb.weight  # (V, d)
        W_R = model.readout_proj.weight  # (d, d)

        E_norm = F.normalize(E, dim=-1)
        M_eff = E_norm @ W_R  # (V, d)

        U_r, S_r, Vh_r = torch.linalg.svd(M_eff, full_matrices=False)

        rank = (S_r > S_r[0] * 1e-6).sum().item()
        V_r = Vh_r[:rank]  # (rank, d) — basis for readout-relevant subspace

        P_r = V_r.T @ V_r  # (d, d) projection matrix

    return P_r, rank, S_r.cpu().numpy()


def compute_full_projection(P_r, L, d, device):
    """Build block-diagonal projection for full state (L*d, L*d)."""
    n = L * d
    P_full = torch.zeros(n, n, device=device)
    for l in range(L):
        start = l * d
        end = (l + 1) * d
        P_full[start:end, start:end] = P_r
    return P_full


def compute_readout_projection_at_state(model, s_star, device):
    """Compute readout-relevant subspace projection at manifold point s_star.

    Uses the analytic Jacobian of readout_logits, accounting for the
    state-dependent normalization: d(logits)/d(s) = W_norm @ D_norm(h) @ W_R / tau.
    """
    L, d = s_star.shape[1], s_star.shape[2]

    with torch.no_grad():
        W_R = model.readout_proj.weight
        b_R = model.readout_proj.bias
        W_norm = F.normalize(model.tok_emb.weight, dim=-1)

        P_r_blocks = []
        ranks = []

        for l in range(L):
            s_l = s_star[0, l]
            h_l = W_R @ s_l
            if b_R is not None:
                h_l = h_l + b_R

            h_norm_val = h_l.norm().clamp(min=1e-10)
            h_hat = h_l / h_norm_val

            D_norm_mat = (torch.eye(d, device=device)
                          - h_hat.unsqueeze(1) * h_hat.unsqueeze(0)) / h_norm_val

            J_l = W_norm @ D_norm_mat @ W_R

            U, S, Vh = torch.linalg.svd(J_l, full_matrices=False)
            rank = (S > S[0] * 1e-6).sum().item()
            V_r = Vh[:rank]
            P_r_l = V_r.T @ V_r

            P_r_blocks.append(P_r_l)
            ranks.append(rank)

        n = L * d
        P_full = torch.zeros(n, n, device=device)
        for l in range(L):
            start = l * d
            end = (l + 1) * d
            P_full[start:end, start:end] = P_r_blocks[l]

    return P_full, ranks


def ftle_analysis_single_sample(model, device, src, max_T):
    """Run FTLE decomposition for a single input sample."""
    L, d = SEQ_LEN, D_MODEL
    n = L * d
    half = SEQ_LEN // 2

    with torch.no_grad():
        context = model.encode(src.unsqueeze(0))
        s = model.init_state(1, SEQ_LEN, device)
        fp_converged = False
        fp_residual = float('inf')
        fp_steps_taken = FP_STEPS
        for fp_step in range(FP_STEPS):
            s_prev_fp = s.clone()
            s, _ = model.dynamics_step(s, context)
            fp_residual = (s - s_prev_fp).norm().item()
            if fp_residual < 1e-4:
                fp_converged = True
                fp_steps_taken = fp_step + 1
                break
        s_star = s.detach().clone()

    if not fp_converged:
        print(f"    WARNING: FP did not converge after {FP_STEPS} steps "
              f"(residual={fp_residual:.6f})", flush=True)

    P_full, readout_ranks = compute_readout_projection_at_state(model, s_star, device)

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

                V_dirs = Vh  # (n, n) right singular vectors (rows)
                alignments = torch.zeros(n, device=device)
                for i in range(n):
                    v_i = V_dirs[i]  # right singular vector
                    proj_v = P_full @ v_i
                    alignments[i] = proj_v.norm() / v_i.norm().clamp(min=1e-10)

                threshold = 0.5
                readout_mask = alignments > threshold
                ortho_mask = ~readout_mask

                n_readout = readout_mask.sum().item()
                n_ortho = ortho_mask.sum().item()

                if n_readout > 0:
                    lambda_R = ftle[readout_mask].max().item()
                    lambda_R_mean = ftle[readout_mask].mean().item()
                    lambda_R_min = ftle[readout_mask].min().item()
                else:
                    lambda_R = lambda_R_mean = lambda_R_min = float('nan')

                if n_ortho > 0:
                    lambda_perp = ftle[ortho_mask].max().item()
                    lambda_perp_mean = ftle[ortho_mask].mean().item()
                else:
                    lambda_perp = lambda_perp_mean = float('nan')

                lambda_max = ftle[0].item()
                lambda_min = ftle[-1].item()

                results_by_T[T] = {
                    "lambda_max": round(lambda_max, 6),
                    "lambda_min": round(lambda_min, 6),
                    "lambda_R_max": round(lambda_R, 6),
                    "lambda_R_mean": round(lambda_R_mean, 6),
                    "lambda_R_min": round(lambda_R_min, 6),
                    "lambda_perp_max": round(lambda_perp, 6),
                    "lambda_perp_mean": round(lambda_perp_mean, 6),
                    "n_readout_aligned": n_readout,
                    "n_ortho": n_ortho,
                    "gap": round(lambda_perp - lambda_R, 6) if not (
                        np.isnan(lambda_perp) or np.isnan(lambda_R)) else None,
                    "top_5_ftle": [round(x, 6) for x in ftle[:5].tolist()],
                    "bottom_5_ftle": [round(x, 6) for x in ftle[-5:].tolist()],
                    "alignment_histogram": {
                        "0.0-0.1": (alignments < 0.1).sum().item(),
                        "0.1-0.3": ((alignments >= 0.1) & (alignments < 0.3)).sum().item(),
                        "0.3-0.5": ((alignments >= 0.3) & (alignments < 0.5)).sum().item(),
                        "0.5-0.7": ((alignments >= 0.5) & (alignments < 0.7)).sum().item(),
                        "0.7-0.9": ((alignments >= 0.7) & (alignments < 0.9)).sum().item(),
                        "0.9-1.0": (alignments >= 0.9).sum().item(),
                    },
                    "rho_from_ftle": round(float(np.exp(lambda_max)), 6),
                }

            print(f"    T={T:>2d}: lam_R={lambda_R:.4f}, lam_perp={lambda_perp:.4f}, "
                  f"gap={lambda_perp - lambda_R:.4f}, n_R={n_readout}, n_orth={n_ortho}, "
                  f"rho={np.exp(lambda_max):.4f}", flush=True)

    results_by_T["_meta"] = {
        "fp_converged": fp_converged,
        "fp_residual": round(fp_residual, 8),
        "fp_steps_taken": fp_steps_taken,
        "readout_ranks": readout_ranks,
    }
    return results_by_T


def measure_spectral_radius(model, device, n_vecs=10, n_iter=50):
    """Standard rho measurement for comparison."""
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 64, SEQ_LEN, VOCAB_SIZE)
    src = src.to(device)

    with torch.no_grad():
        context = model.encode(src)
        s = model.init_state(src.size(0), SEQ_LEN, device)
        for _ in range(FP_STEPS):
            s, _ = model.dynamics_step(s, context)
        s_star = s.detach().clone()

    rhos = []
    for _ in range(n_vecs):
        v = torch.randn_like(s_star[:16])
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for _ in range(n_iter):
            with torch.enable_grad():
                s_pert = s_star[:16].detach().requires_grad_(True)
                s_next, _ = model.dynamics_step(s_pert, context[:16])
                jvp = torch.autograd.grad(s_next, s_pert, grad_outputs=v, create_graph=False)[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    return {"mean": round(float(np.mean(rhos)), 6), "std": round(float(np.std(rhos)), 6)}


def evaluate_step_ablation(model, device):
    """Measure T_99 for comparison."""
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", 2048, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    results = {}
    with torch.no_grad():
        context = model.encode(src)
        for T in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
            s = model.init_state(src.size(0), SEQ_LEN, device)
            for t in range(T):
                s, _ = model.dynamics_step(s, context)
            logits = model.readout_logits(s)
            preds = logits[:, :half].argmax(dim=-1)
            seq_correct = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
            results[T] = round(seq_correct, 4)
    return results


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
        if existing.get("ftle_analysis"):
            print("Results already exist — skipping.", flush=True)
            return

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 1: Training variable_t model (L={SEQ_LEN}, seed={SEED})", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(SEED)
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN).to(device)
    params = count_params(model)
    print(f"  Params: {params}", flush=True)

    train_time = train_variable_t(model, device)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 2: Standard diagnostics", flush=True)
    print(f"{'='*60}", flush=True)

    step_ablation = evaluate_step_ablation(model, device)
    T_99 = None
    for T in [1, 2, 3, 4, 5, 8, 10]:
        if step_ablation[T] >= 0.99:
            T_99 = T
            break
    print(f"  Step ablation: {step_ablation}", flush=True)
    print(f"  T_99 = {T_99}", flush=True)

    rho_standard = measure_spectral_radius(model, device)
    print(f"  rho (power iteration) = {rho_standard}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 3: Readout projection analysis", flush=True)
    print(f"{'='*60}", flush=True)

    P_r_static, rank_static, sv_readout = compute_readout_projection(model, device)
    print(f"  Static readout subspace rank: {rank_static} / {D_MODEL}", flush=True)
    print(f"  Top 5 readout SVs: {sv_readout[:5].round(4)}", flush=True)
    print(f"  (Per-sample state-dependent projection used in Phase 4)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 4: FTLE decomposition ({N_ANALYSIS_SAMPLES} samples)", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(9999)
    src, tgt = generate_batch("addition", N_ANALYSIS_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    all_sample_results = []
    for i in range(N_ANALYSIS_SAMPLES):
        print(f"\n  Sample {i+1}/{N_ANALYSIS_SAMPLES}:", flush=True)
        t0 = time.time()
        sample_result = ftle_analysis_single_sample(
            model, device, src[i], max(ANALYSIS_T_VALUES)
        )
        elapsed = time.time() - t0
        all_sample_results.append(sample_result)
        print(f"    Done in {elapsed:.1f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 5: Aggregation", flush=True)
    print(f"{'='*60}", flush=True)

    aggregated = {}
    for T in ANALYSIS_T_VALUES:
        lam_R_vals = [r[T]["lambda_R_max"] for r in all_sample_results
                      if T in r and not np.isnan(r[T]["lambda_R_max"])]
        lam_perp_vals = [r[T]["lambda_perp_max"] for r in all_sample_results
                         if T in r and not np.isnan(r[T]["lambda_perp_max"])]
        gap_vals = [r[T]["gap"] for r in all_sample_results
                    if T in r and r[T]["gap"] is not None]

        aggregated[T] = {
            "lambda_R": {
                "mean": round(float(np.mean(lam_R_vals)), 6) if lam_R_vals else None,
                "std": round(float(np.std(lam_R_vals)), 6) if lam_R_vals else None,
                "n": len(lam_R_vals),
            },
            "lambda_perp": {
                "mean": round(float(np.mean(lam_perp_vals)), 6) if lam_perp_vals else None,
                "std": round(float(np.std(lam_perp_vals)), 6) if lam_perp_vals else None,
                "n": len(lam_perp_vals),
            },
            "gap": {
                "mean": round(float(np.mean(gap_vals)), 6) if gap_vals else None,
                "std": round(float(np.std(gap_vals)), 6) if gap_vals else None,
                "n": len(gap_vals),
            },
        }

        if lam_R_vals and lam_perp_vals:
            print(f"  T={T:>2d}: lam_R={np.mean(lam_R_vals):.4f}+/-{np.std(lam_R_vals):.4f}, "
                  f"lam_perp={np.mean(lam_perp_vals):.4f}+/-{np.std(lam_perp_vals):.4f}, "
                  f"gap={np.mean(gap_vals):.4f}", flush=True)

    prop31_confirmed = False
    lam_R_5_val = aggregated.get(5, {}).get("lambda_R", {}).get("mean")
    lam_perp_5_val = aggregated.get(5, {}).get("lambda_perp", {}).get("mean")
    if lam_R_5_val is not None and lam_perp_5_val is not None:
        lam_R_5 = lam_R_5_val
        lam_perp_5 = lam_perp_5_val
        if lam_R_5 < 0 and lam_perp_5 > 0:
            prop31_confirmed = True
            print(f"\n  *** PROPOSITION 31 CONFIRMED ***", flush=True)
            print(f"  lambda_R = {lam_R_5:.4f} < 0 (readout contracts)", flush=True)
            print(f"  lambda_perp = {lam_perp_5:.4f} > 0 (orthogonal expands)", flush=True)
        else:
            print(f"\n  Proposition 31 status: MIXED", flush=True)
            print(f"  lambda_R = {lam_R_5:.4f} (need < 0)", flush=True)
            print(f"  lambda_perp = {lam_perp_5:.4f} (need > 0)", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D29: FTLE decomposition — direct test of Proposition 31",
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "seed": SEED,
            "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
            "variant": "variable_t", "fp_steps": FP_STEPS,
            "n_analysis_samples": N_ANALYSIS_SAMPLES,
        },
        "training": {
            "time_s": round(train_time, 1),
            "params": params,
        },
        "standard_diagnostics": {
            "step_ablation": step_ablation,
            "T_99": T_99,
            "spectral_radius": rho_standard,
        },
        "readout_projection": {
            "static_rank": rank_static,
            "dim_state": SEQ_LEN * D_MODEL,
            "dim_readout_subspace_static": rank_static * SEQ_LEN,
            "dim_ortho_subspace_static": (D_MODEL - rank_static) * SEQ_LEN,
            "top_singular_values": [round(float(x), 4) for x in sv_readout[:10]],
            "note": "Per-sample state-dependent projection used in FTLE analysis",
        },
        "ftle_analysis": {
            "per_sample": {str(i): r for i, r in enumerate(all_sample_results)},
            "aggregated": {str(k): v for k, v in aggregated.items()},
        },
        "prop31_verdict": {
            "confirmed": prop31_confirmed,
            "lambda_R_at_T5": aggregated.get(5, {}).get("lambda_R", {}).get("mean"),
            "lambda_perp_at_T5": aggregated.get(5, {}).get("lambda_perp", {}).get("mean"),
            "gap_at_T5": aggregated.get(5, {}).get("gap", {}).get("mean"),
            "rho_standard": rho_standard["mean"],
        },
    }

    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)

    print(f"\n  Results saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    run()
