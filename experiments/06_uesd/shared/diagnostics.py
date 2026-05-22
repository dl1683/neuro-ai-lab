"""Diagnostic measurements for UESD experiments.

Seven diagnostics (D1-D7) that evaluate equilibrium quality, decoder
confidence, attractor structure, basin stability, and spectral
properties of UESD models.  All functions take PyTorch tensors and
return Python floats or dicts of floats.
"""
import math
import torch


# ---------------------------------------------------------------------------
# D1: Token Accuracy
# ---------------------------------------------------------------------------


@torch.no_grad()
def token_accuracy(logits, target_ids):
    """Per-token accuracy and exact-sequence-match accuracy.

    Args:
        logits: (batch, seq_len, vocab_size) -- raw or scaled logits.
        target_ids: (batch, seq_len) -- ground-truth token ids.

    Returns:
        dict with 'token_acc' (float) and 'seq_acc' (float).
    """
    preds = logits.argmax(dim=-1)                        # (B, L)
    correct = preds == target_ids                        # (B, L)
    token_acc = correct.float().mean().item()
    seq_acc = correct.all(dim=-1).float().mean().item()  # all positions correct
    return {"token_acc": token_acc, "seq_acc": seq_acc}


# ---------------------------------------------------------------------------
# D2: Normalized Residual
# ---------------------------------------------------------------------------


@torch.no_grad()
def normalized_residual(model, state, context):
    """||s_{T+1} - s_T|| normalized by sqrt(L_out * d_model).

    Measures how far from equilibrium the state is: small values mean
    the dynamics have converged.

    Args:
        model: UESDModel instance (needs ``dynamics_step``).
        state: s_T tensor of shape (batch, L_out, d_model).
        context: encoded context of shape (batch, L_in, d_model).

    Returns:
        dict with 'mean' and 'std' of the per-example normalized residual.
    """
    s_next, _ = model.dynamics_step(state, context)
    diff = s_next - state                                # (B, L_out, d)
    # Per-example L2 norm of the update
    per_example_norm = diff.norm(dim=(-2, -1))           # (B,)
    _, L_out, d = state.shape
    normalizer = math.sqrt(L_out * d)
    normed = per_example_norm / normalizer
    return {"mean": normed.mean().item(), "std": normed.std().item()}


# ---------------------------------------------------------------------------
# D3: Decoder Margin
# ---------------------------------------------------------------------------


@torch.no_grad()
def decoder_margin(logits, target_ids):
    """Margin between the correct-class logit and the strongest wrong class.

    A positive margin means the model is predicting the correct token;
    negative means the strongest competitor wins.

    Args:
        logits: (batch, seq_len, vocab_size) -- can be cosine-sim / tau
                or any score tensor; only relative ordering matters.
        target_ids: (batch, seq_len).

    Returns:
        dict with 'mean_margin', 'min_margin', 'frac_positive'.
    """
    B, L, V = logits.shape

    # Gather the logit of the correct class at each position
    correct_logit = logits.gather(
        dim=-1, index=target_ids.unsqueeze(-1),
    ).squeeze(-1)                                        # (B, L)

    # Mask out the correct class before taking max over wrong classes
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=target_ids.unsqueeze(-1), value=True)
    wrong_logits = logits.masked_fill(mask, float("-inf"))
    max_wrong, _ = wrong_logits.max(dim=-1)              # (B, L)

    margin = correct_logit - max_wrong                   # (B, L)
    return {
        "mean_margin": margin.mean().item(),
        "min_margin": margin.min().item(),
        "frac_positive": (margin > 0).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# D4: Wrong-Attractor Rate
# ---------------------------------------------------------------------------


@torch.no_grad()
def wrong_attractor_rate(model, src_ids, target_ids, T,
                         residual_threshold=0.01):
    """Fraction of examples that converge but decode to wrong tokens.

    Args:
        model: UESDModel instance.
        src_ids: (batch, seq_len) source token ids.
        target_ids: (batch, seq_len) target token ids.
        T: number of dynamics steps.
        residual_threshold: normalized-residual cutoff for "converged".

    Returns:
        dict with 'wrong_attractor_rate', 'converged_frac',
        'converged_correct_frac'.
    """
    state, _ = model.unroll(src_ids, T)
    context = model.encode(src_ids)

    # Normalized residual per example
    s_next, _ = model.dynamics_step(state, context)
    diff_norm = (s_next - state).norm(dim=(-2, -1))      # (B,)
    _, L_out, d = state.shape
    normalizer = math.sqrt(L_out * d)
    normed_residual = diff_norm / normalizer              # (B,)

    # Readout accuracy per example (exact sequence match)
    logits = model.readout_logits(state)
    preds = logits.argmax(dim=-1)                        # (B, L)
    seq_correct = (preds == target_ids).all(dim=-1)       # (B,)

    converged = normed_residual < residual_threshold      # (B,)
    n_converged = converged.sum().item()
    B = src_ids.size(0)

    if n_converged == 0:
        return {
            "wrong_attractor_rate": 0.0,
            "converged_frac": 0.0,
            "converged_correct_frac": 0.0,
        }

    converged_correct = (converged & seq_correct).sum().item()
    converged_wrong = (converged & ~seq_correct).sum().item()

    return {
        "wrong_attractor_rate": converged_wrong / n_converged,
        "converged_frac": n_converged / B,
        "converged_correct_frac": converged_correct / n_converged,
    }


# ---------------------------------------------------------------------------
# D5: Basin Perturbation
# ---------------------------------------------------------------------------


@torch.no_grad()
def basin_perturbation(model, src_ids, T, sigma_frac=0.1, extra_steps=10):
    """Perturb s_T with Gaussian noise and check if readout is preserved.

    Args:
        model: UESDModel instance.
        src_ids: (batch, seq_len) source token ids.
        T: dynamics steps to reach s_T.
        sigma_frac: noise magnitude as a fraction of per-example state norm.
        extra_steps: additional dynamics steps from the perturbed state.

    Returns:
        dict with 'stability_frac' -- fraction of examples whose readout
        survives the perturbation.
    """
    state, _ = model.unroll(src_ids, T)
    context = model.encode(src_ids)

    # Original readout tokens
    orig_tokens = model.readout_logits(state).argmax(dim=-1)   # (B, L)

    # Per-example noise scaled so ||noise||_F = sigma_frac * ||s||_F
    state_norm = state.norm(dim=(-2, -1), keepdim=True)        # (B, 1, 1)
    n_elements = state.shape[-2] * state.shape[-1]
    noise = torch.randn_like(state) * (sigma_frac * state_norm / math.sqrt(n_elements))
    s_perturbed = state + noise

    # Run extra dynamics steps from perturbed state
    for _ in range(extra_steps):
        s_perturbed, _ = model.dynamics_step(s_perturbed, context)

    # Check readout after recovery
    recovered_tokens = model.readout_logits(s_perturbed).argmax(dim=-1)
    matches = (recovered_tokens == orig_tokens).all(dim=-1)    # (B,)
    stability_frac = matches.float().mean().item()
    return {"stability_frac": stability_frac}


# ---------------------------------------------------------------------------
# D6: Spectral Radius Estimate (Power Iteration via Finite Differences)
# ---------------------------------------------------------------------------


@torch.no_grad()
def spectral_radius(model, state, context, n_iterations=10):
    """Estimate spectral radius of the Jacobian dG/ds via power iteration.

    Uses central finite differences to approximate the Jacobian-vector
    product, which is robust and does not require autograd.

    Args:
        model: UESDModel instance.
        state: s tensor of shape (batch, L_out, d_model).
        context: encoded context of shape (batch, L_in, d_model).
        n_iterations: number of power-iteration steps.

    Returns:
        dict with 'mean_rho' and 'max_rho' across the batch.
    """
    eps = 1e-4

    # Initialize random direction, one per example
    v = torch.randn_like(state)
    # Normalize each example's direction to unit norm
    v = v / v.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    B = state.size(0)
    lam = torch.zeros(B, device=state.device)

    for _ in range(n_iterations):
        # Central finite-difference approximation of J @ v
        s_plus = state + eps * v
        s_minus = state - eps * v
        G_plus, _ = model.dynamics_step(s_plus, context)
        G_minus, _ = model.dynamics_step(s_minus, context)
        Jv = (G_plus - G_minus) / (2 * eps)              # (B, L, d)

        # Per-example norms
        Jv_norm = Jv.norm(dim=(-2, -1))                  # (B,)
        v_norm = v.norm(dim=(-2, -1))                     # (B,)
        lam = Jv_norm / v_norm.clamp(min=1e-12)

        # Update direction
        v = Jv / Jv_norm.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-12)

    return {
        "mean_rho": lam.mean().item(),
        "max_rho": lam.max().item(),
    }


# ---------------------------------------------------------------------------
# D7: Non-Normality Ratio (σ_max / ρ)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sigma_max_ratio(model, state, context, n_samples=4):
    """κ = σ_max(J) / ρ(J) via batched full-Jacobian finite differences.

    Computes the full Jacobian ∂G/∂s at the converged state, then
    extracts σ_max (largest singular value) and ρ (spectral radius)
    to quantify non-normality.  See finite_step_convergence.md Theorem 4.

    κ ≈ 1: normal — eigenvalue analysis reliable for finite-T bounds.
    κ ∈ (1, 1.5): mildly non-normal — practical for T=10.
    κ > 2: severely non-normal — transient growth even if ρ < 1.

    Args:
        model: UESDModel with dynamics_step(s, context).
        state: s_T of shape (B, L, d_model).
        context: encoded context of shape (B, L_in, d_model).
        n_samples: number of batch examples to analyze.

    Returns:
        dict with sigma_max_mean, rho_eig_mean, kappa_mean, kappa_max.
    """
    B, L, d = state.shape
    n = L * d

    n_samples = min(n_samples, B)
    indices = torch.randperm(B, device=state.device)[:n_samples]

    eps = 1e-4
    eye = torch.eye(n, device=state.device, dtype=state.dtype)
    E = eye.reshape(n, L, d)

    sigma_maxs = []
    rhos_eig = []
    kappas = []

    for idx in indices:
        s_i = state[idx]
        c_i = context[idx]

        s_rep = s_i.unsqueeze(0).expand(n, -1, -1)
        c_rep = c_i.unsqueeze(0).expand(n, -1, -1)

        G_plus, _ = model.dynamics_step(s_rep + eps * E, c_rep)
        G_minus, _ = model.dynamics_step(s_rep - eps * E, c_rep)

        J = ((G_plus - G_minus) / (2 * eps)).reshape(n, n).t()

        sigma_max = torch.linalg.svdvals(J)[0].item()

        eigvals = torch.linalg.eigvals(J)
        rho = eigvals.abs().max().item()

        kappa = sigma_max / max(rho, 1e-12)

        sigma_maxs.append(sigma_max)
        rhos_eig.append(rho)
        kappas.append(kappa)

    return {
        "sigma_max_mean": sum(sigma_maxs) / len(sigma_maxs),
        "sigma_max_max": max(sigma_maxs),
        "rho_eig_mean": sum(rhos_eig) / len(rhos_eig),
        "kappa_mean": sum(kappas) / len(kappas),
        "kappa_max": max(kappas),
    }


# ---------------------------------------------------------------------------
# D8: Trajectory Lyapunov Analysis (product Jacobian along dynamics path)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _full_jacobian(model, s_single, c_single, eps=1e-4):
    """Compute full Jacobian dG/ds for a single example via finite differences.

    Args:
        model: UESDModel with dynamics_step(s, context).
        s_single: state (L, d).
        c_single: context (L_in, d).
        eps: finite difference step.

    Returns:
        J: (n, n) Jacobian where n = L * d.
    """
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
def trajectory_lyapunov(model, src_ids, T, n_samples=8):
    """Compute product-Jacobian along the dynamics trajectory.

    Measures how perturbations actually propagate through all T steps,
    accounting for Jacobian rotation between steps. The Lyapunov exponent
    lambda_max = (1/T) * log(sigma_max(P_T)) determines true trajectory
    stability, which can differ dramatically from the per-step bound.

    Args:
        model: UESDModel instance.
        src_ids: (B, seq_len) source token ids.
        T: number of dynamics steps.
        n_samples: examples to analyze.

    Returns:
        dict with per-step and cumulative spectral properties.
    """
    model.eval()
    context = model.encode(src_ids)
    B, L_out = src_ids.shape
    s = model.init_state(B, L_out, src_ids.device)
    _, _, d = s.shape
    n = L_out * d

    n_samples = min(n_samples, B)
    indices = torch.randperm(B, device=src_ids.device)[:n_samples]

    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    per_step_sigma = []
    per_step_rho = []
    per_step_kappa = []
    cumulative_sigma = []
    cumulative_rho = []
    lyapunov_exponents = []
    sv_alignment = []

    for idx in indices:
        c_i = context[idx]
        product_J = None
        prev_right_sv = None
        step_data_sigma = []
        step_data_rho = []
        step_data_kappa = []
        cum_sigma = []
        cum_rho = []
        alignments = []

        for t in range(T):
            s_t = states[t][idx]
            J_t = _full_jacobian(model, s_t, c_i)

            sm = torch.linalg.svdvals(J_t)[0].item()
            eigvals = torch.linalg.eigvals(J_t)
            rh = eigvals.abs().max().item()
            kp = sm / max(rh, 1e-12)

            step_data_sigma.append(sm)
            step_data_rho.append(rh)
            step_data_kappa.append(kp)

            U, S, Vh = torch.linalg.svd(J_t)
            right_sv = Vh[0]
            if prev_right_sv is not None:
                alignment = torch.dot(right_sv, prev_right_sv).abs().item()
                alignments.append(alignment)
            prev_right_sv = right_sv

            if product_J is None:
                product_J = J_t.clone()
            else:
                product_J = J_t @ product_J

            prod_sm = torch.linalg.svdvals(product_J)[0].item()
            prod_eigvals = torch.linalg.eigvals(product_J)
            prod_rh = prod_eigvals.abs().max().item()
            cum_sigma.append(prod_sm)
            cum_rho.append(prod_rh)

        per_step_sigma.append(step_data_sigma)
        per_step_rho.append(step_data_rho)
        per_step_kappa.append(step_data_kappa)
        cumulative_sigma.append(cum_sigma)
        cumulative_rho.append(cum_rho)
        lyap = math.log(max(cum_sigma[-1], 1e-30)) / T
        lyapunov_exponents.append(lyap)
        sv_alignment.append(alignments)

    def avg_list_of_lists(lol):
        return [sum(x[i] for x in lol) / len(lol) for i in range(len(lol[0]))]

    avg_per_step_sigma = avg_list_of_lists(per_step_sigma)
    avg_per_step_kappa = avg_list_of_lists(per_step_kappa)
    avg_cum_sigma = avg_list_of_lists(cumulative_sigma)
    avg_cum_rho = avg_list_of_lists(cumulative_rho)
    avg_alignment = avg_list_of_lists(sv_alignment) if sv_alignment[0] else []

    mean_lyap = sum(lyapunov_exponents) / len(lyapunov_exponents)

    theorem4_bound = avg_per_step_sigma[-1] ** T
    actual_amplification = avg_cum_sigma[-1]
    conservatism_ratio = theorem4_bound / max(actual_amplification, 1e-30)

    n_expanding = sum(1 for s in avg_cum_sigma if s > 1.0)

    return {
        "lyapunov_max_mean": mean_lyap,
        "lyapunov_all": lyapunov_exponents,
        "theorem4_bound": theorem4_bound,
        "actual_amplification": actual_amplification,
        "conservatism_ratio": conservatism_ratio,
        "per_step_sigma_max": avg_per_step_sigma,
        "per_step_kappa": avg_per_step_kappa,
        "cumulative_sigma_max": avg_cum_sigma,
        "cumulative_rho": avg_cum_rho,
        "sv_alignment": avg_alignment,
        "n_expanding_steps": n_expanding,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Convenience: run all diagnostics
# ---------------------------------------------------------------------------


def run_all_diagnostics(model, src_ids, target_ids, T, config=None):
    """Run D1-D7 and return a combined dict.

    Args:
        model: UESDModel instance.
        src_ids: (batch, seq_len) source token ids.
        target_ids: (batch, seq_len) target token ids.
        T: number of dynamics steps.
        config: optional dict with overrides for diagnostic parameters
                (keys: 'residual_threshold', 'sigma_frac', 'extra_steps',
                 'n_power_iterations', 'compute_d7', 'd7_n_samples').

    Returns:
        dict mapping diagnostic names to their result dicts.
    """
    cfg = {
        "residual_threshold": 0.01,
        "sigma_frac": 0.1,
        "extra_steps": 10,
        "n_power_iterations": 10,
        "compute_d7": False,
        "d7_n_samples": 4,
    }
    if config:
        cfg.update(config)

    model.eval()

    # Shared forward pass: unroll once, reuse state and context
    with torch.no_grad():
        state, update_norms = model.unroll(src_ids, T)
        context = model.encode(src_ids)
        logits = model.readout_logits(state)

    results = {}

    # Per-step update norms (trajectory from unroll)
    results["update_trajectory"] = {
        f"step_{i+1}": n.item() for i, n in enumerate(update_norms)
    }

    # D1: Token Accuracy
    results["token_accuracy"] = token_accuracy(logits, target_ids)

    # D2: Normalized Residual
    results["normalized_residual"] = normalized_residual(
        model, state, context,
    )

    # D3: Decoder Margin
    results["decoder_margin"] = decoder_margin(logits, target_ids)

    # D4: Wrong-Attractor Rate
    results["wrong_attractor"] = wrong_attractor_rate(
        model, src_ids, target_ids, T,
        residual_threshold=cfg["residual_threshold"],
    )

    # D5: Basin Perturbation
    results["basin_perturbation"] = basin_perturbation(
        model, src_ids, T,
        sigma_frac=cfg["sigma_frac"],
        extra_steps=cfg["extra_steps"],
    )

    # D6: Spectral Radius
    results["spectral_radius"] = spectral_radius(
        model, state, context,
        n_iterations=cfg["n_power_iterations"],
    )

    # D7: Non-Normality Ratio (opt-in, computes full Jacobian)
    if cfg["compute_d7"]:
        results["sigma_max_ratio"] = sigma_max_ratio(
            model, state, context,
            n_samples=cfg["d7_n_samples"],
        )

    return results
