"""
Experiment D18: Error Function Comparison (E3 vs E5)

Tests UESD Axiom A3 by comparing two fundamentally different error functions
on the SAME dynamics architecture. If both produce similar dynamical regimes,
the two-mechanism discovery (highway vs scattered) is a property of the
DYNAMICS, not the error function. If they differ, the error function shapes
the attractor landscape.

Error Functions:
- E5 (self-consistency): E(s,c) = ||F_theta(s,c)||^2
  "A state is good when the system has nothing left to change"

- E3 (denoising score): E(s,c) = ||s - D_theta(s + sigma*eps, c, sigma)||^2
  "A state is good when it can be recovered from corruption"
  This connects UESD to diffusion models: the forward process (noise)
  and reverse process (denoising) both live in state space S.

Protocol:
1. Train three tracks on addition (seed=42, 20K steps):
   a. CE-dynamics baseline (no extra loss)
   b. E5: CE + lambda * ||F(s_T,c)||^2
   c. E3: CE + lambda * denoising_loss(s_T, c)

2. E3 denoising training:
   - At each step, after computing s_T:
     - Sample noise eps ~ N(0,1), scale sigma from annealing schedule
     - Corrupt: s_noisy = s_T + sigma * eps
     - Denoise: s_denoised = D_theta(s_noisy, c, sigma)
     - Loss_denoise = ||s_T - s_denoised||^2
   - D_theta reuses the dynamics layer (F_theta) with sigma conditioning
   - Total loss = CE + lambda * Loss_denoise

3. At convergence, measure ALL the same diagnostics as D4/D5:
   - Lyapunov exponent, Jacobian alignment, amplification, overshoot
   - Self-consistency energy (even for E3 track)
   - Denoising error (even for E5 track)

4. Key comparisons:
   - Does E3 produce highway or scattered dynamics?
   - Does E3 achieve lower DENOISING error than E5?
   - Does E5 achieve lower SELF-CONSISTENCY energy than E3?
   - Are E3's Jacobians more similar to E5 or CE-dynamics?

PREDICTIONS:
1. E3 dynamics will be INTERMEDIATE between E5 and CE-dynamics
   (some regularization from denoising, but different geometry than SC)
2. E3 will have lower denoising error than E5, E5 will have lower SC energy
   (each optimizes its own error function, but some transfer)
3. E3's Jacobian alignment will be 0.65-0.75 (between E5's 0.81 and CE-dyn's 0.60)
4. E3 will show NOVEL stability patterns not seen in E5 or CE-dynamics
   -- this is the cutting-edge finding: different error functions create
   different attractors in the SAME dynamics architecture

If prediction 4 holds, it opens a whole new research direction: can we
DESIGN error functions to produce specific dynamical properties?
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        "sigma_min": 0.01,
        "sigma_max": 1.0,
    }


class SigmaConditioner(nn.Module):
    """Lightweight sigma conditioning: project scalar sigma to d_model."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, sigma):
        return self.proj(sigma.unsqueeze(-1))


def train_model(track, config, device):
    set_seed(config["seed"])
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)

    sigma_cond = None
    if track == "e3":
        sigma_cond = SigmaConditioner(config["d_model"]).to(device)

    all_params = list(model.parameters())
    if sigma_cond is not None:
        all_params += list(sigma_cond.parameters())

    model.train()
    optimizer = torch.optim.Adam(all_params, lr=config["lr"])
    T = config["T"]
    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        s = model.init_state(B, L, src.device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]
        ce = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                             tgt_r.reshape(-1))

        if track == "dynamics_ce":
            loss = ce

        elif track == "e5":
            s_next = model.dynamics(s, context)
            sc = (s_next - s).pow(2).mean()
            eff_lam = min(step / config["warmup_steps"], 1.0)
            loss = ce + eff_lam * sc

        elif track == "e3":
            # E3 denoising: corrupt s_T, use dynamics to denoise
            eff_lam = min(step / config["warmup_steps"], 1.0)

            # Sample noise level (log-uniform between sigma_min and sigma_max)
            log_sigma = (torch.rand(B, device=device)
                        * (np.log(config["sigma_max"]) - np.log(config["sigma_min"]))
                        + np.log(config["sigma_min"]))
            sigma = log_sigma.exp()  # [B]

            # Corrupt converged state
            eps = torch.randn_like(s)
            s_noisy = s.detach() + sigma.view(B, 1, 1) * eps

            # Condition on sigma: add sigma embedding to the noisy state
            sigma_emb = sigma_cond(sigma)  # [B, d_model]
            s_conditioned = s_noisy + sigma_emb.unsqueeze(1)

            # Denoise using dynamics (one step of F_theta)
            s_denoised = model.dynamics(s_conditioned, context)

            # Denoising loss: predict the CLEAN state from noisy
            denoise_loss = (s_denoised - s.detach()).pow(2).mean()

            loss = ce + eff_lam * denoise_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            extra = ""
            if track == "e5":
                with torch.no_grad():
                    s_next = model.dynamics(s.detach(), context)
                    sc_val = (s_next - s.detach()).pow(2).mean()
                extra = f" | SC: {sc_val.item():.4f}"
            elif track == "e3":
                extra = f" | Denoise: {denoise_loss.item():.4f}"
            print(f"    Step {step:>6d}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f} | CE: {ce.item():.4f}{extra}",
                  flush=True)

    model.eval()
    return model, sigma_cond


@torch.no_grad()
def compute_diagnostics(model, eval_src, eval_tgt, config, device):
    """Compute full trajectory diagnostics matching D4/D5."""
    T = config["T"]
    half = config["seq_len"] // 2

    context = model.encode(eval_src)
    B, L = eval_src.shape
    s = model.init_state(B, L, device)

    # Collect trajectory
    states = [s.clone()]
    for t in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    # Final accuracy
    logits = model.readout_logits(states[-1])
    preds = logits[:, :half, :].argmax(dim=-1)
    targets = eval_tgt[:, :half]
    token_acc = (preds == targets).float().mean().item()
    seq_acc = (preds == targets).all(dim=1).float().mean().item()

    # Self-consistency energy at final state
    s_next = model.dynamics(states[-1], context)
    sc_energy = (s_next - states[-1]).pow(2).mean().item()

    # Per-step update norms
    update_norms = []
    for t in range(T):
        norm = (states[t+1] - states[t]).norm(dim=-1).mean().item()
        update_norms.append(norm)

    # Jacobian-based diagnostics (on a small sample)
    n_diag = min(8, B)
    lyap_sum = 0.0
    align_sum = 0.0
    amp_list = []

    for idx in range(n_diag):
        s_single = states[0][idx]
        c_single = context[idx]
        L_s, d = s_single.shape
        n = L_s * d

        # Product Jacobian
        product_J = torch.eye(n, device=device)
        prev_v1 = None
        alignment_accum = 0.0
        n_align = 0

        for t in range(T):
            s_t = states[t][idx]
            eps = 1e-4
            eye = torch.eye(n, device=device).reshape(n, L_s, d)
            s_rep = s_t.unsqueeze(0).expand(n, -1, -1)
            c_rep = c_single.unsqueeze(0).expand(n, -1, -1)
            G_plus, _ = model.dynamics_step(s_rep + eps * eye, c_rep)
            G_minus, _ = model.dynamics_step(s_rep - eps * eye, c_rep)
            J_t = ((G_plus - G_minus) / (2 * eps)).reshape(n, n).t()

            product_J = J_t @ product_J

            # SV alignment
            _, _, Vh = torch.linalg.svd(J_t)
            v1 = Vh[0]
            if prev_v1 is not None:
                cos = torch.dot(v1, prev_v1).abs().item()
                alignment_accum += cos
                n_align += 1
            prev_v1 = v1

        # Product Jacobian spectral radius
        s_vals = torch.linalg.svdvals(product_J)
        lyap = torch.log(s_vals[0]).item() / T
        amp = s_vals[0].item()
        align = alignment_accum / max(n_align, 1)

        lyap_sum += lyap
        align_sum += align
        amp_list.append(amp)

    lyap_mean = lyap_sum / n_diag
    align_mean = align_sum / n_diag
    amp_mean = float(np.mean(amp_list))

    # Overshoot ratio (ordered vs shuffled)
    os_ratios = []
    for idx in range(min(4, B)):
        s_single = states[0][idx]
        c_single = context[idx]
        L_s, d = s_single.shape
        n = L_s * d

        # Ordered product
        prod_ord = torch.eye(n, device=device)
        jacobians = []
        for t in range(T):
            s_t = states[t][idx]
            eps = 1e-4
            eye = torch.eye(n, device=device).reshape(n, L_s, d)
            s_rep = s_t.unsqueeze(0).expand(n, -1, -1)
            c_rep = c_single.unsqueeze(0).expand(n, -1, -1)
            G_plus, _ = model.dynamics_step(s_rep + eps * eye, c_rep)
            G_minus, _ = model.dynamics_step(s_rep - eps * eye, c_rep)
            J_t = ((G_plus - G_minus) / (2 * eps)).reshape(n, n).t()
            prod_ord = J_t @ prod_ord
            jacobians.append(J_t)

        sigma_ord = torch.linalg.svdvals(prod_ord)[0].item()

        # Shuffled product (average over 3 permutations)
        sigma_shuf = 0.0
        for _ in range(3):
            perm = torch.randperm(T)
            prod_shuf = torch.eye(n, device=device)
            for t in perm:
                prod_shuf = jacobians[t] @ prod_shuf
            sigma_shuf += torch.linalg.svdvals(prod_shuf)[0].item()
        sigma_shuf /= 3

        os_ratios.append(sigma_ord / max(sigma_shuf, 1e-10))

    os_mean = float(np.mean(os_ratios))

    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "sc_energy": sc_energy,
        "lyapunov": lyap_mean,
        "alignment": align_mean,
        "amplification": amp_mean,
        "overshoot_ratio": os_mean,
        "update_norms": update_norms,
    }


@torch.no_grad()
def compute_denoising_error(model, sigma_cond, s_clean, context, config, device):
    """Measure denoising error across sigma levels."""
    sigmas = [0.01, 0.05, 0.1, 0.5, 1.0]
    B = s_clean.shape[0]
    results = {}

    for sigma_val in sigmas:
        eps = torch.randn_like(s_clean)
        s_noisy = s_clean + sigma_val * eps

        if sigma_cond is not None:
            sigma_t = torch.full((B,), sigma_val, device=device)
            sigma_emb = sigma_cond(sigma_t)
            s_conditioned = s_noisy + sigma_emb.unsqueeze(1)
        else:
            s_conditioned = s_noisy

        s_denoised = model.dynamics(s_conditioned, context)
        err = (s_denoised - s_clean).pow(2).sum(dim=-1).mean().item()

        results[f"sigma_{sigma_val}"] = {
            "denoise_error": err,
            "noise_norm": (sigma_val * eps).norm(dim=-1).mean().item(),
            "recovery_ratio": err / max((sigma_val * eps).pow(2).sum(dim=-1).mean().item(), 1e-10),
        }

    return results


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(seed=42)
    T = config["T"]
    V = config["vocab_size"]
    half = config["seq_len"] // 2

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Error function comparison: E3 denoising vs E5 self-consistency",
        "config": config,
    }

    # Generate eval data
    set_seed(999)
    eval_src, eval_tgt = generate_batch("addition", 4096, config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)

    for track in ["dynamics_ce", "e5", "e3"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  TRACK: {track}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()
        model, sigma_cond = train_model(track, config, device)
        train_time = time.time() - t0
        print(f"  Training: {train_time:.0f}s", flush=True)

        # Full diagnostics
        print(f"  Computing diagnostics...", flush=True)
        diag = compute_diagnostics(model, eval_src, eval_tgt, config, device)

        print(f"  Token acc: {diag['token_acc']:.4f}, "
              f"Seq acc: {diag['seq_acc']:.4f}", flush=True)
        print(f"  SC energy: {diag['sc_energy']:.6f}", flush=True)
        print(f"  Lyapunov: {diag['lyapunov']:.4f}", flush=True)
        print(f"  Alignment: {diag['alignment']:.4f}", flush=True)
        print(f"  Amplification: {diag['amplification']:.2f}x", flush=True)
        print(f"  Overshoot ratio: {diag['overshoot_ratio']:.4f}", flush=True)

        # Collect converged states for denoising test
        context = model.encode(eval_src)
        B = eval_src.shape[0]
        s = model.init_state(B, eval_src.shape[1], device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        s_clean = s

        # Denoising error (for ALL tracks, not just E3)
        print(f"  Denoising error across sigma:", flush=True)
        denoise_results = compute_denoising_error(
            model, sigma_cond if track == "e3" else None,
            s_clean, context, config, device
        )
        for key, val in denoise_results.items():
            print(f"    {key}: err={val['denoise_error']:.4f}, "
                  f"recovery={val['recovery_ratio']:.4f}", flush=True)

        all_results[track] = {
            "train_time_s": train_time,
            "diagnostics": diag,
            "denoising_errors": denoise_results,
        }

        del model, sigma_cond
        torch.cuda.empty_cache()

    # Cross-track comparison
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D18 ERROR FUNCTION COMPARISON", flush=True)
    print(f"{'=' * 60}", flush=True)

    header = f"  {'Metric':<20s} {'CE-dyn':>10s} {'E5':>10s} {'E3':>10s}"
    print(header, flush=True)
    print(f"  {'-' * 50}", flush=True)

    for metric in ["token_acc", "seq_acc", "sc_energy", "lyapunov",
                    "alignment", "amplification", "overshoot_ratio"]:
        vals = []
        for track in ["dynamics_ce", "e5", "e3"]:
            v = all_results[track]["diagnostics"][metric]
            vals.append(v)
        print(f"  {metric:<20s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}",
              flush=True)

    # Classification: which regime does E3 fall into?
    e5_lyap = all_results["e5"]["diagnostics"]["lyapunov"]
    ce_lyap = all_results["dynamics_ce"]["diagnostics"]["lyapunov"]
    e3_lyap = all_results["e3"]["diagnostics"]["lyapunov"]

    e5_align = all_results["e5"]["diagnostics"]["alignment"]
    ce_align = all_results["dynamics_ce"]["diagnostics"]["alignment"]
    e3_align = all_results["e3"]["diagnostics"]["alignment"]

    print(f"\n  E3 regime classification:", flush=True)

    # Distance to each regime
    dist_to_e5 = abs(e3_lyap - e5_lyap) + abs(e3_align - e5_align)
    dist_to_ce = abs(e3_lyap - ce_lyap) + abs(e3_align - ce_align)

    if dist_to_e5 < dist_to_ce * 0.5:
        print(f"    E3 is CLOSER to E5 highway regime", flush=True)
    elif dist_to_ce < dist_to_e5 * 0.5:
        print(f"    E3 is CLOSER to CE-dynamics scattered regime", flush=True)
    else:
        print(f"    E3 is INTERMEDIATE -- potentially a THIRD regime!", flush=True)

    e3_os = all_results["e3"]["diagnostics"]["overshoot_ratio"]
    if e3_os > 1.0:
        print(f"    E3 overshoot > 1.0 (like E5 highway)", flush=True)
    else:
        print(f"    E3 overshoot < 1.0 (like CE-dyn scattered)", flush=True)

    print(f"{'=' * 60}", flush=True)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d18_error_function_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
