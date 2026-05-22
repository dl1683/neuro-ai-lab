"""
Experiment D3b: Trajectory Lyapunov Validation

Codex mandated validation of D3 before further theorizing:
1. Eps sweep: does finite-difference step size affect results?
2. Autograd check: compare finite-diff Jacobian to torch.autograd
3. Shuffled-trajectory ablation: is temporal ordering important?
4. Corrected bounds: max_sigma^T, product-of-sigmas, actual amp

If alignment and amplification are stable across eps values,
and shuffled trajectories show different amplification than
ordered ones, the Jacobian rotation finding is validated.
"""
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import train, set_seed, count_params
from shared.diagnostics import sigma_max_ratio
from shared.data import generate_batch


def build_config(seed=42):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "n_dec_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "log_interval": 1000,
        "eval_samples": 512,
        "eval_batch_size": 512,
        "warmup_steps": 5000,
        "seed": seed,
    }


@torch.no_grad()
def _full_jacobian_fd(model, s_single, c_single, eps=1e-4):
    """Finite-difference Jacobian (same as diagnostics.py but with configurable eps)."""
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


def _autograd_jacobian(model, s_single, c_single):
    """Autograd Jacobian for validation."""
    s = s_single.clone().detach().requires_grad_(True)
    L, d = s.shape
    n = L * d
    c = c_single.unsqueeze(0)
    s_in = s.unsqueeze(0)
    out, _ = model.dynamics_step(s_in, c)
    out_flat = out.squeeze(0).reshape(n)

    J = torch.zeros(n, n, device=s.device, dtype=s.dtype)
    for i in range(n):
        if s.grad is not None:
            s.grad.zero_()
        out_flat[i].backward(retain_graph=True)
        J[i] = s.grad.reshape(n)
    return J


def trajectory_analysis(model, src_ids, T, n_samples, eps=1e-4):
    """Compute trajectory product-Jacobian with given eps."""
    model.eval()
    context = model.encode(src_ids)
    B = src_ids.shape[0]
    L_out = src_ids.shape[1]
    s = model.init_state(B, L_out, src_ids.device)
    d = s.shape[2]

    n_samples = min(n_samples, B)
    indices = torch.randperm(B, device=src_ids.device)[:n_samples]

    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    results_per_sample = []
    for idx in indices:
        c_i = context[idx]
        jacobians = []
        per_step_sigma = []
        sv_alignment = []
        prev_right_sv = None

        for t in range(T):
            s_t = states[t][idx]
            J_t = _full_jacobian_fd(model, s_t, c_i, eps=eps)
            jacobians.append(J_t)

            sm = torch.linalg.svdvals(J_t)[0].item()
            per_step_sigma.append(sm)

            _, _, Vh = torch.linalg.svd(J_t)
            right_sv = Vh[0]
            if prev_right_sv is not None:
                alignment = torch.dot(right_sv, prev_right_sv).abs().item()
                sv_alignment.append(alignment)
            prev_right_sv = right_sv

        product_J = jacobians[0].clone()
        for t in range(1, T):
            product_J = jacobians[t] @ product_J
        cum_sigma = torch.linalg.svdvals(product_J)[0].item()

        shuffled_indices = list(range(T))
        random.shuffle(shuffled_indices)
        shuffled_product = jacobians[shuffled_indices[0]].clone()
        for t in shuffled_indices[1:]:
            shuffled_product = jacobians[t] @ shuffled_product
        shuffled_sigma = torch.linalg.svdvals(shuffled_product)[0].item()

        product_of_sigmas = 1.0
        for sm in per_step_sigma:
            product_of_sigmas *= sm
        max_sigma_T = max(per_step_sigma) ** T

        results_per_sample.append({
            "per_step_sigma": per_step_sigma,
            "sv_alignment": sv_alignment,
            "ordered_cum_sigma": cum_sigma,
            "shuffled_cum_sigma": shuffled_sigma,
            "product_of_sigmas": product_of_sigmas,
            "max_sigma_T": max_sigma_T,
            "lyapunov": math.log(max(cum_sigma, 1e-30)) / T,
        })

    return results_per_sample


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D3b: Validate trajectory Lyapunov via eps sweep, autograd check, shuffled controls",
    }

    seed = 42
    config = build_config(seed=seed)
    set_seed(seed)

    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"params: {count_params(model)}", flush=True)

    print("\n=== TRAINING CE-dynamics seed=42 ===", flush=True)
    tr = train(model, "addition", "dynamics_ce", config, device)
    print(f"Training done in {tr['elapsed_s']:.0f}s", flush=True)

    model.eval()
    src, tgt = generate_batch("addition", config["eval_batch_size"],
                              config["seq_len"], config["vocab_size"])
    src, tgt = src.to(device), tgt.to(device)

    # --- 1. Autograd validation ---
    print("\n=== AUTOGRAD JACOBIAN CHECK ===", flush=True)
    context = model.encode(src)
    T = config["T"]
    s = model.init_state(src.shape[0], src.shape[1], device)
    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    autograd_checks = []
    for check_idx in range(4):
        s_t = states[0][check_idx]
        c_i = context[check_idx]

        J_fd = _full_jacobian_fd(model, s_t, c_i, eps=1e-4)

        model_copy = model
        J_ag = _autograd_jacobian(model_copy, s_t, c_i)

        diff = (J_fd - J_ag).norm() / J_ag.norm()
        sigma_fd = torch.linalg.svdvals(J_fd)[:5].tolist()
        sigma_ag = torch.linalg.svdvals(J_ag)[:5].tolist()

        _, _, Vh_fd = torch.linalg.svd(J_fd)
        _, _, Vh_ag = torch.linalg.svd(J_ag)
        sv_cos = torch.dot(Vh_fd[0], Vh_ag[0]).abs().item()

        check = {
            "example_idx": check_idx,
            "relative_frobenius_error": diff.item(),
            "top5_sigma_fd": sigma_fd,
            "top5_sigma_ag": sigma_ag,
            "sv1_alignment": sv_cos,
        }
        autograd_checks.append(check)
        print(f"  Example {check_idx}: rel_err={diff.item():.6f}, "
              f"sv1_cos={sv_cos:.6f}, "
              f"sigma1_fd={sigma_fd[0]:.4f} vs ag={sigma_ag[0]:.4f}", flush=True)

    results["autograd_checks"] = autograd_checks

    # --- 2. Eps sweep ---
    print("\n=== EPS SWEEP ===", flush=True)
    eps_values = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    eps_sweep = {}
    for eps in eps_values:
        print(f"  eps={eps:.0e}...", flush=True, end="")
        samples = trajectory_analysis(model, src, T, n_samples=8, eps=eps)

        avg_lyap = sum(s["lyapunov"] for s in samples) / len(samples)
        avg_ordered = sum(s["ordered_cum_sigma"] for s in samples) / len(samples)
        avg_shuffled = sum(s["shuffled_cum_sigma"] for s in samples) / len(samples)
        all_alignments = [a for s in samples for a in s["sv_alignment"]]
        avg_align = sum(all_alignments) / len(all_alignments) if all_alignments else 0

        avg_conserv_prod = sum(
            s["product_of_sigmas"] / max(s["ordered_cum_sigma"], 1e-30)
            for s in samples
        ) / len(samples)
        avg_conserv_max = sum(
            s["max_sigma_T"] / max(s["ordered_cum_sigma"], 1e-30)
            for s in samples
        ) / len(samples)

        entry = {
            "eps": eps,
            "avg_lyapunov": avg_lyap,
            "avg_ordered_cum_sigma": avg_ordered,
            "avg_shuffled_cum_sigma": avg_shuffled,
            "ordered_vs_shuffled_ratio": avg_ordered / max(avg_shuffled, 1e-30),
            "avg_sv_alignment": avg_align,
            "avg_conservatism_product": avg_conserv_prod,
            "avg_conservatism_max_sigma": avg_conserv_max,
        }
        eps_sweep[f"{eps:.0e}"] = entry
        print(f" lyap={avg_lyap:.4f}, ordered={avg_ordered:.2f}, "
              f"shuffled={avg_shuffled:.2f}, "
              f"ratio_o/s={avg_ordered/max(avg_shuffled,1e-30):.3f}, "
              f"align={avg_align:.4f}", flush=True)

    results["eps_sweep"] = eps_sweep

    # --- 3. Shuffled controls (larger sample at eps=1e-4) ---
    print("\n=== SHUFFLED TRAJECTORY CONTROLS (eps=1e-4, n=16) ===", flush=True)
    samples = trajectory_analysis(model, src, T, n_samples=16, eps=1e-4)

    ordered_sigmas = [s["ordered_cum_sigma"] for s in samples]
    shuffled_sigmas = [s["shuffled_cum_sigma"] for s in samples]

    multi_shuffle_per_sample = []
    context = model.encode(src)
    s = model.init_state(src.shape[0], src.shape[1], device)
    states_full = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states_full.append(s.clone())

    for sample_idx in range(min(8, src.shape[0])):
        c_i = context[sample_idx]
        jacobians = []
        for t in range(T):
            s_t = states_full[t][sample_idx]
            J_t = _full_jacobian_fd(model, s_t, c_i, eps=1e-4)
            jacobians.append(J_t)

        ordered_prod = jacobians[0].clone()
        for t in range(1, T):
            ordered_prod = jacobians[t] @ ordered_prod
        ordered_s = torch.linalg.svdvals(ordered_prod)[0].item()

        shuffle_results = []
        for _ in range(10):
            perm = list(range(T))
            random.shuffle(perm)
            prod = jacobians[perm[0]].clone()
            for t in perm[1:]:
                prod = jacobians[t] @ prod
            shuffle_results.append(torch.linalg.svdvals(prod)[0].item())

        multi_shuffle_per_sample.append({
            "ordered": ordered_s,
            "shuffled_mean": sum(shuffle_results) / len(shuffle_results),
            "shuffled_std": (sum((x - sum(shuffle_results)/len(shuffle_results))**2
                                for x in shuffle_results) / len(shuffle_results)) ** 0.5,
            "shuffled_min": min(shuffle_results),
            "shuffled_max": max(shuffle_results),
        })

        print(f"  Sample {sample_idx}: ordered={ordered_s:.4f}, "
              f"shuffled_mean={multi_shuffle_per_sample[-1]['shuffled_mean']:.4f} "
              f"(std={multi_shuffle_per_sample[-1]['shuffled_std']:.4f}, "
              f"range=[{multi_shuffle_per_sample[-1]['shuffled_min']:.4f}, "
              f"{multi_shuffle_per_sample[-1]['shuffled_max']:.4f}])", flush=True)

    results["shuffled_controls"] = multi_shuffle_per_sample

    # --- 4. Summary ---
    print(f"\n{'=' * 80}", flush=True)
    print("D3b VALIDATION SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)

    ag_errs = [c["relative_frobenius_error"] for c in autograd_checks]
    print(f"\nAutograd check: max relative error = {max(ag_errs):.6f} "
          f"(mean = {sum(ag_errs)/len(ag_errs):.6f})", flush=True)
    if max(ag_errs) < 0.01:
        print("  PASS: Finite-difference Jacobian matches autograd", flush=True)
    else:
        print("  WARNING: Large discrepancy between FD and autograd", flush=True)

    print("\nEps sweep stability:", flush=True)
    lyaps = [v["avg_lyapunov"] for v in eps_sweep.values()]
    lyap_range = max(lyaps) - min(lyaps)
    print(f"  Lyapunov range across eps: {lyap_range:.6f} "
          f"(values: {[f'{l:.4f}' for l in lyaps]})", flush=True)
    if lyap_range < 0.05:
        print("  PASS: Lyapunov exponent stable across eps values", flush=True)
    else:
        print("  WARNING: Lyapunov sensitive to eps choice", flush=True)

    aligns = [v["avg_sv_alignment"] for v in eps_sweep.values()]
    align_range = max(aligns) - min(aligns)
    print(f"  Alignment range across eps: {align_range:.6f} "
          f"(values: {[f'{a:.4f}' for a in aligns]})", flush=True)
    if align_range < 0.05:
        print("  PASS: SV alignment stable across eps values", flush=True)
    else:
        print("  WARNING: Alignment sensitive to eps choice", flush=True)

    print("\nShuffled trajectory test:", flush=True)
    ordered_mean = sum(s["ordered"] for s in multi_shuffle_per_sample) / len(multi_shuffle_per_sample)
    shuffled_mean = sum(s["shuffled_mean"] for s in multi_shuffle_per_sample) / len(multi_shuffle_per_sample)
    print(f"  Ordered mean cum_sigma: {ordered_mean:.4f}", flush=True)
    print(f"  Shuffled mean cum_sigma: {shuffled_mean:.4f}", flush=True)
    print(f"  Ratio ordered/shuffled: {ordered_mean/max(shuffled_mean, 1e-30):.4f}", flush=True)
    if abs(ordered_mean - shuffled_mean) / max(shuffled_mean, 1e-30) > 0.1:
        print("  SIGNIFICANT: Temporal ordering matters for trajectory stability", flush=True)
    else:
        print("  NOT SIGNIFICANT: Ordering doesn't strongly affect amplification", flush=True)

    print(f"\nCorrected conservatism (eps=1e-4):", flush=True)
    ref = eps_sweep["1e-04"]
    print(f"  Actual amplification: {ref['avg_ordered_cum_sigma']:.2f}x", flush=True)
    print(f"  Conservatism (product of sigmas / actual): {ref['avg_conservatism_product']:.1f}x", flush=True)
    print(f"  Conservatism (max_sigma^T / actual): {ref['avg_conservatism_max_sigma']:.1f}x", flush=True)

    print(f"{'=' * 80}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d3b_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
