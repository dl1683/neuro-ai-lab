"""
Experiment D6: Random-Matrix Null Model

Tests whether the observed Jacobian cancellation (718x conservatism in D3b)
is a LEARNED property of the dynamics or merely what happens when you
multiply non-identity matrices with similar spectra.

Approach:
1. Train CE-dynamics seed=42 (reuse D4 setup)
2. Extract FULL Jacobian spectra at convergence (all 1024 singular values per step)
3. Generate null-model Jacobians: same spectral profile, random orientations
4. Compute product-Jacobian sigma for: actual, matched-spectrum-random, isotropic-random
5. Compare conservatism under each null model

Three null models:
A. ISOTROPIC: J_t = sigma_max(t) * O_t (all SVs equal to sigma_max)
B. MATCHED SPECTRUM: J_t = U_t * diag(actual_spectrum_t) * V_t^T (actual SVs, random orientation)
C. MATCHED + ALIGNED: same as B but first singular vector alignment matches actual

If conservatism_actual ≈ conservatism_B, rotation is random (no learned structure).
If conservatism_actual > conservatism_B, dynamics have LESS cancellation than random.
If conservatism_actual < conservatism_B, dynamics have EXTRA cancellation (learned rotation).

Also computes: effective dimension of Jacobian spectrum (participation ratio).
"""
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

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


@torch.no_grad()
def _full_jacobian(model, s_single, c_single, eps=1e-4):
    L, d = s_single.shape
    n = L * d
    eye = torch.eye(n, device=s_single.device, dtype=s_single.dtype)
    E = eye.reshape(n, L, d)
    s_rep = s_single.unsqueeze(0).expand(n, -1, -1)
    c_rep = c_single.unsqueeze(0).expand(n, -1, -1)
    G_plus, _ = model.dynamics_step(s_rep + eps * E, c_rep)
    G_minus, _ = model.dynamics_step(s_rep - eps * E, c_rep)
    J = ((G_plus - G_minus) / (2 * eps)).reshape(n, n).t()
    return J


@torch.no_grad()
def extract_trajectory_jacobians(model, src_ids, T, n_samples=8):
    """Extract full Jacobian SVD at each trajectory step."""
    model.eval()
    context = model.encode(src_ids)
    B = src_ids.shape[0]
    s = model.init_state(B, src_ids.shape[1], src_ids.device)

    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    n_samples = min(n_samples, B)
    all_sample_data = []

    for idx in range(n_samples):
        c_i = context[idx]
        sample_jacobians = []
        sample_spectra = []
        sample_Us = []
        sample_Vhs = []

        for t in range(T):
            s_t = states[t][idx]
            J_t = _full_jacobian(model, s_t, c_i)
            U, S, Vh = torch.linalg.svd(J_t)

            sample_jacobians.append(J_t.cpu())
            sample_spectra.append(S.cpu().numpy())
            sample_Us.append(U.cpu())
            sample_Vhs.append(Vh.cpu())

        # Compute actual product Jacobian
        product_J = sample_jacobians[0].clone()
        for t in range(1, T):
            product_J = sample_jacobians[t] @ product_J
        actual_sigma = torch.linalg.svdvals(product_J)[0].item()

        # Singular vector alignments
        alignments = []
        for t in range(T - 1):
            cos_val = torch.dot(sample_Vhs[t][0], sample_Vhs[t+1][0]).abs().item()
            alignments.append(cos_val)

        all_sample_data.append({
            "spectra": sample_spectra,
            "actual_product_sigma": actual_sigma,
            "alignments": alignments,
            "jacobians_cpu": sample_jacobians,
        })

    return all_sample_data


def random_orthogonal(n, rng):
    """Generate random orthogonal matrix via QR of Gaussian."""
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    Q *= np.sign(d)[None, :]
    return Q


def null_model_A_isotropic(per_step_sigma_max, T, n, n_trials=100, rng=None):
    """Null A: J_t = sigma_max(t) * O_t (isotropic random orthogonal)."""
    if rng is None:
        rng = np.random.default_rng(0)

    results = []
    for _ in range(n_trials):
        product = np.eye(n)
        for t in range(T):
            O = random_orthogonal(n, rng)
            product = (per_step_sigma_max[t] * O) @ product
        s = np.linalg.svd(product, compute_uv=False)
        results.append(s[0])
    return results


def null_model_B_matched_spectrum(spectra_per_step, T, n_trials=100, rng=None):
    """Null B: J_t = U_t * diag(actual_spectrum_t) * V_t^T (matched spectrum, random orientation)."""
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(spectra_per_step[0])
    T_actual = len(spectra_per_step)

    results = []
    for _ in range(n_trials):
        product = np.eye(n)
        for t in range(T_actual):
            U = random_orthogonal(n, rng)
            V = random_orthogonal(n, rng)
            J_null = U @ np.diag(spectra_per_step[t]) @ V.T
            product = J_null @ product
        s = np.linalg.svd(product, compute_uv=False)
        results.append(s[0])
    return results


def participation_ratio(spectrum):
    """Effective dimension of a spectrum: (sum s_i)^2 / sum(s_i^2).
    For uniform spectrum = n, for single dominant SV = 1."""
    s = np.array(spectrum)
    s2 = s ** 2
    return (s2.sum()) ** 2 / (s2 ** 2).sum() if s2.sum() > 0 else 0


def _dynamics_ce_step(model, src, tgt, T):
    context = model.encode(src)
    B, L_out = src.shape
    half = L_out // 2
    s = model.init_state(B, L_out, src.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
    logits = model.readout_logits(s)
    logits_r = logits[:, :half, :]
    tgt_r = tgt[:, :half]
    loss = torch.nn.functional.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)), tgt_r.reshape(-1))
    return loss


def _e5_step(model, src, tgt, T, step, warmup_steps, lambda_1):
    context = model.encode(src)
    B, L_out = src.shape
    half = L_out // 2
    s = model.init_state(B, L_out, src.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
    logits = model.readout_logits(s)
    logits_r = logits[:, :half, :]
    tgt_r = tgt[:, :half]
    ce = torch.nn.functional.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)), tgt_r.reshape(-1))
    sc = (s - model.dynamics(s, context)).pow(2).mean()
    eff_lam = min(step / warmup_steps, 1.0) * lambda_1
    return ce + eff_lam * sc


def train_model(track, config, device):
    seed = config["seed"]
    set_seed(seed)
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        if track == "dynamics_ce":
            loss = _dynamics_ce_step(model, src, tgt, T)
        else:
            loss = _e5_step(model, src, tgt, T, step,
                            config["warmup_steps"], config.get("lambda_1", 1.0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{config['training_steps']} | Loss: {loss.item():.4f}",
                  flush=True)

    return model


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Random-matrix null model: is Jacobian cancellation learned or statistical?",
    }

    N_NULL_TRIALS = 200
    N_EXTRACT_SAMPLES = 8
    rng = np.random.default_rng(42)
    track_results = []

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'#' * 60}", flush=True)
        print(f"  Training {track} seed=42", flush=True)
        print(f"{'#' * 60}", flush=True)

        config = build_config(seed=42)
        if track == "e5":
            config["lambda_1"] = 1.0

        t0 = time.time()
        model = train_model(track, config, device)
        train_time = time.time() - t0
        print(f"  Training done in {train_time:.0f}s", flush=True)

        # Generate eval batch
        set_seed(999)
        eval_src, eval_tgt = generate_batch("addition", 512,
                                             config["seq_len"], config["vocab_size"])
        eval_src = eval_src.to(device)

        print(f"  Extracting full Jacobian spectra...", flush=True)
        t1 = time.time()
        sample_data = extract_trajectory_jacobians(
            model, eval_src, config["T"], n_samples=N_EXTRACT_SAMPLES)
        extract_time = time.time() - t1
        print(f"  Extraction done in {extract_time:.0f}s", flush=True)

        # Analyze each sample
        all_actual_sigmas = []
        all_null_A = []
        all_null_B = []
        all_spectra_stats = []

        for si, sd in enumerate(sample_data):
            actual_sigma = sd["actual_product_sigma"]
            all_actual_sigmas.append(actual_sigma)

            per_step_smax = [sp[0] for sp in sd["spectra"]]
            n = len(sd["spectra"][0])

            # Null A: isotropic
            null_a = null_model_A_isotropic(per_step_smax, config["T"], n,
                                            n_trials=N_NULL_TRIALS, rng=rng)
            all_null_A.append(null_a)

            # Null B: matched spectrum
            null_b = null_model_B_matched_spectrum(sd["spectra"], config["T"],
                                                   n_trials=N_NULL_TRIALS, rng=rng)
            all_null_B.append(null_b)

            # Spectral statistics
            pr_per_step = [participation_ratio(sp) for sp in sd["spectra"]]
            kappa_per_step = [sp[0] / sp[-1] if sp[-1] > 1e-10 else float('inf')
                              for sp in sd["spectra"]]

            all_spectra_stats.append({
                "participation_ratio": pr_per_step,
                "condition_number": kappa_per_step,
                "sigma_max_per_step": per_step_smax,
            })

            if si < 3:
                print(f"\n  Sample {si}:", flush=True)
                print(f"    Actual product sigma: {actual_sigma:.4f}", flush=True)
                print(f"    Null A (isotropic) mean: {np.mean(null_a):.4f} "
                      f"± {np.std(null_a):.4f}", flush=True)
                print(f"    Null B (matched)   mean: {np.mean(null_b):.4f} "
                      f"± {np.std(null_b):.4f}", flush=True)
                print(f"    Participation ratio per step: "
                      f"{', '.join(f'{p:.1f}' for p in pr_per_step)}", flush=True)
                print(f"    Condition number per step: "
                      f"{', '.join(f'{k:.1f}' for k in kappa_per_step)}", flush=True)
                print(f"    SV alignments: "
                      f"{', '.join(f'{a:.3f}' for a in sd['alignments'])}", flush=True)

        # Aggregate across samples
        mean_actual = np.mean(all_actual_sigmas)
        mean_null_a = np.mean([np.mean(x) for x in all_null_A])
        mean_null_b = np.mean([np.mean(x) for x in all_null_B])
        std_null_a = np.mean([np.std(x) for x in all_null_A])
        std_null_b = np.mean([np.std(x) for x in all_null_B])

        product_of_smax = 1.0
        for st in all_spectra_stats[0]["sigma_max_per_step"]:
            product_of_smax *= st

        conservatism_actual = product_of_smax / max(mean_actual, 1e-30)
        conservatism_null_a = product_of_smax / max(mean_null_a, 1e-30)
        conservatism_null_b = product_of_smax / max(mean_null_b, 1e-30)

        print(f"\n  === {track} AGGREGATE ===", flush=True)
        print(f"  Product of per-step sigma_max: {product_of_smax:.1f}", flush=True)
        print(f"  Actual product sigma:   {mean_actual:.4f} "
              f"(conservatism: {conservatism_actual:.1f}x)", flush=True)
        print(f"  Null A (isotropic):     {mean_null_a:.4f} ± {std_null_a:.4f} "
              f"(conservatism: {conservatism_null_a:.1f}x)", flush=True)
        print(f"  Null B (matched spec):  {mean_null_b:.4f} ± {std_null_b:.4f} "
              f"(conservatism: {conservatism_null_b:.1f}x)", flush=True)

        # Position of actual relative to null B distribution
        all_null_b_flat = [v for nb in all_null_B for v in nb]
        fraction_below_actual = sum(1 for v in all_null_b_flat if v < mean_actual) / len(all_null_b_flat)
        print(f"  Fraction of null-B trials below actual: {fraction_below_actual:.3f}", flush=True)

        if mean_actual < mean_null_b:
            print(f"  CONCLUSION: Actual has MORE cancellation than random rotation → "
                  f"dynamics learned EXTRA rotation structure", flush=True)
        elif mean_actual > mean_null_b * 1.5:
            print(f"  CONCLUSION: Actual has LESS cancellation than random rotation → "
                  f"dynamics have partial alignment (directed amplification)", flush=True)
        else:
            print(f"  CONCLUSION: Actual is comparable to random rotation → "
                  f"cancellation is statistical, not learned", flush=True)

        mean_pr = np.mean([np.mean(ss["participation_ratio"]) for ss in all_spectra_stats])
        mean_kappa = np.mean([np.mean([k for k in ss["condition_number"] if k < 1e10])
                              for ss in all_spectra_stats])

        track_result = {
            "track": track,
            "mean_actual_sigma": float(mean_actual),
            "mean_null_a_sigma": float(mean_null_a),
            "std_null_a_sigma": float(std_null_a),
            "mean_null_b_sigma": float(mean_null_b),
            "std_null_b_sigma": float(std_null_b),
            "product_of_smax": float(product_of_smax),
            "conservatism_actual": float(conservatism_actual),
            "conservatism_null_a": float(conservatism_null_a),
            "conservatism_null_b": float(conservatism_null_b),
            "fraction_null_b_below_actual": float(fraction_below_actual),
            "mean_participation_ratio": float(mean_pr),
            "mean_condition_number": float(mean_kappa),
            "train_time_s": train_time,
            "extract_time_s": extract_time,
        }
        track_results.append(track_result)

    results["track_results"] = track_results

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print("D6 RANDOM-MATRIX NULL MODEL SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n  {'':20s} {'CE-dynamics':>15s} {'E5':>15s}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for key in ["mean_actual_sigma", "mean_null_b_sigma", "conservatism_actual",
                "conservatism_null_b", "fraction_null_b_below_actual",
                "mean_participation_ratio", "mean_condition_number"]:
        ce = track_results[0][key]
        e5 = track_results[1][key]
        label = key.replace("_", " ").title()
        print(f"  {label:20s} {ce:15.4f} {e5:15.4f}", flush=True)

    print(f"\n  Interpretation:", flush=True)
    for tr in track_results:
        actual = tr["mean_actual_sigma"]
        null_b = tr["mean_null_b_sigma"]
        ratio = actual / null_b if null_b > 0 else float('inf')
        if ratio < 0.8:
            interp = "EXTRA learned rotation (actual << random)"
        elif ratio > 1.2:
            interp = "PARTIAL alignment (actual >> random)"
        else:
            interp = "STATISTICAL cancellation (actual ≈ random)"
        print(f"    {tr['track']}: actual/null_B = {ratio:.2f} → {interp}", flush=True)

    print(f"{'=' * 70}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d6_random_matrix_null.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
