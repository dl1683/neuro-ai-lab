"""Diagnostic measurements for UESD experiments.

Six diagnostics (D1-D6) that evaluate equilibrium quality, decoder
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
# Convenience: run all diagnostics
# ---------------------------------------------------------------------------


def run_all_diagnostics(model, src_ids, target_ids, T, config=None):
    """Run D1-D6 and return a combined dict.

    Args:
        model: UESDModel instance.
        src_ids: (batch, seq_len) source token ids.
        target_ids: (batch, seq_len) target token ids.
        T: number of dynamics steps.
        config: optional dict with overrides for diagnostic parameters
                (keys: 'residual_threshold', 'sigma_frac', 'extra_steps',
                 'n_power_iterations').

    Returns:
        dict mapping diagnostic names to their result dicts.
    """
    cfg = {
        "residual_threshold": 0.01,
        "sigma_frac": 0.1,
        "extra_steps": 10,
        "n_power_iterations": 10,
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

    return results
