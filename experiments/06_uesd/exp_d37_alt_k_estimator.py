"""
Experiment D37: Alternate Contraction Rate Estimator Validation

PURPOSE: Address the "estimator artifact" threat — the last missing Codex 9/10
requirement. All prior k measurements use the same protocol (trajectory decay from
init_state toward fixed point). If that protocol has a systematic bias, the k≈0.988
setpoint could be an artifact of the MEASUREMENT, not the dynamics.

THREE INDEPENDENT ESTIMATORS:
  A. STANDARD (baseline): ||s_t - s*|| / ||s_{t-1} - s*||  (same as D33/D36)
  B. RANDOM-DIRECTION: Inject small random perturbations around s*, measure decay
     rate. Tests whether k depends on perturbation direction.
  C. PAIRWISE TRAJECTORY: Two independent random inits, measure how fast they
     converge to each other (no fixed-point reference needed).

OPTIONAL (d=64 only):
  D. FULL JACOBIAN: Compute actual d×d Jacobian at each token position, extract
     leading eigenvalue magnitude. Ground truth but O(d^2) per token.

MODELS TESTED:
  - Re-use trained models from D33 (baseline d=128) and D36 (small d=64)
  - Load saved checkpoints, measure k with all estimators
  - 4 models: baseline FT, baseline VT, small FT, small VT (from D36)

PREDICTIONS:
  If k≈0.988 is real: All estimators agree within noise (±0.002).
  If k≈0.988 is artifact: Alt estimators give different values, revealing
  the bias direction and magnitude.

FALSIFICATION: If any alt estimator gives k > 1.0 for a VT model (claiming
  expansion where the standard says contraction), the standard estimator has
  a critical bias.
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed
from shared.data import generate_batch

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30
N_PERTURBATION_DIRS = 20
PERTURBATION_SCALES = [0.001, 0.01, 0.03]
N_PAIRWISE_INITS = 10

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d37_alt_k_estimator.json"

MODEL_CONFIGS = {
    "baseline_ft_D8": {
        "source": "D33",
        "arch": {"d_model": 128, "n_heads": 4, "d_ff": 512},
        "carry_depth": 8, "seq_len": 16, "variant": "fixed_t", "seed": 42,
        "vocab_size": 64, "n_enc_layers": 2, "max_len": 64,
    },
    "baseline_vt_D8": {
        "source": "D33",
        "arch": {"d_model": 128, "n_heads": 4, "d_ff": 512},
        "carry_depth": 8, "seq_len": 16, "variant": "variable_t", "seed": 42,
        "vocab_size": 64, "n_enc_layers": 2, "max_len": 64,
    },
    "small_ft": {
        "source": "D36",
        "arch": {"d_model": 64, "n_heads": 2, "d_ff": 256},
        "carry_depth": 6, "seq_len": 12, "variant": "fixed_t", "seed": 42,
        "vocab_size": 64, "n_enc_layers": 2, "max_len": 64,
    },
    "small_vt": {
        "source": "D36",
        "arch": {"d_model": 64, "n_heads": 2, "d_ff": 256},
        "carry_depth": 6, "seq_len": 12, "variant": "variable_t", "seed": 42,
        "vocab_size": 64, "n_enc_layers": 2, "max_len": 64,
    },
}

TRAINING_STEPS = 20000
BATCH_SIZE = 256
LR = 3e-4
TRAIN_T = 10
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
T_MIN = min(VARIABLE_T_RANGE)


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(cfg, device):
    arch = cfg["arch"]
    model = UESDModel(
        cfg["vocab_size"], arch["d_model"], arch["n_heads"], arch["d_ff"],
        cfg["n_enc_layers"], cfg["max_len"],
    )
    return model.to(device)


def train_model(model, device, cfg):
    """Train a model from scratch matching D33/D36 protocol."""
    full_seed(cfg["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = cfg["seq_len"] // 2
    model.train()

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, cfg["seq_len"], cfg["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        if cfg["variant"] == "variable_t":
            t_steps = random.choice(VARIABLE_T_RANGE)
        else:
            t_steps = TRAIN_T

        logits = model(src, t_steps)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 4000 == 0 or step == 1:
            with torch.no_grad():
                eval_logits = model(src, TRAIN_T)
                preds = eval_logits[:, :half].argmax(dim=-1)
                acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
            print(f"        Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.4f}", flush=True)
            model.train()

    return model


# === ESTIMATOR A: Standard (baseline, same as D33/D36) ===

@torch.no_grad()
def estimator_standard(model, device, seq_len):
    """Standard k estimator: decay from init_state toward fixed point."""
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", EVAL_SAMPLES, seq_len, 64)
    src = src.to(device)
    context = model.encode(src)

    s_star = model.init_state(src.size(0), src.size(1), device)
    for _ in range(FP_T):
        s_star, _ = model.dynamics_step(s_star, context)

    s = model.init_state(src.size(0), src.size(1), device)
    prev_dist = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)
    k_values = []
    for t in range(TRAJECTORY_T):
        s, _ = model.dynamics_step(s, context)
        curr_dist = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)
        k_per = (curr_dist / prev_dist.clamp(min=1e-8)).mean().item()
        k_values.append(k_per)
        prev_dist = curr_dist

    stable_k = k_values[1:10]
    return {
        "method": "standard",
        "mean_k": round(float(np.mean(stable_k)), 6),
        "std_k": round(float(np.std(stable_k)), 6),
        "k_trajectory": [round(v, 6) for v in k_values],
    }


# === ESTIMATOR B: Random-direction perturbation around fixed point ===

@torch.no_grad()
def estimator_random_direction(model, device, seq_len):
    """Inject random perturbations around s*, measure per-direction decay rate.
    Sweeps multiple perturbation scales to check linearity regime."""
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", min(EVAL_SAMPLES, 512), seq_len, 64)
    src = src.to(device)
    context = model.encode(src)

    s_star = model.init_state(src.size(0), src.size(1), device)
    for _ in range(FP_T):
        s_star, _ = model.dynamics_step(s_star, context)

    s_star_orig = s_star.clone()
    per_scale_results = {}

    for scale in PERTURBATION_SCALES:
        direction_ks = []
        for d_idx in range(N_PERTURBATION_DIRS):
            torch.manual_seed(7777 + d_idx)
            delta = torch.randn_like(s_star_orig) * scale
            s_pert = s_star_orig.clone() + delta
            s_ref = s_star_orig.clone()

            prev_dist = delta.reshape(src.size(0), -1).norm(dim=-1)
            k_values = []
            for t in range(min(TRAJECTORY_T, 15)):
                s_pert, _ = model.dynamics_step(s_pert, context)
                s_ref, _ = model.dynamics_step(s_ref, context)
                curr_dist = (s_pert - s_ref).reshape(src.size(0), -1).norm(dim=-1)
                k_per = (curr_dist / prev_dist.clamp(min=1e-8)).mean().item()
                k_values.append(k_per)
                prev_dist = curr_dist

            stable_k = k_values[1:min(10, len(k_values))]
            if stable_k:
                direction_ks.append(float(np.mean(stable_k)))

        per_scale_results[str(scale)] = {
            "mean_k": round(float(np.mean(direction_ks)), 6),
            "std_k": round(float(np.std(direction_ks)), 6),
            "n_directions": len(direction_ks),
        }

    all_scale_ks = [v["mean_k"] for v in per_scale_results.values()]
    scale_range = max(all_scale_ks) - min(all_scale_ks)

    return {
        "method": "random_direction",
        "mean_k": round(float(np.mean(all_scale_ks)), 6),
        "std_k": round(float(np.std(all_scale_ks)), 6),
        "per_scale": per_scale_results,
        "scale_range": round(scale_range, 6),
        "scale_stable": scale_range < 0.002,
    }


# === ESTIMATOR C: Pairwise trajectory convergence ===

@torch.no_grad()
def estimator_pairwise(model, device, seq_len):
    """Perturbed init_state pairs with burn-in, then measure convergence rate."""
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", min(EVAL_SAMPLES, 512), seq_len, 64)
    src = src.to(device)
    context = model.encode(src)

    base_init = model.init_state(src.size(0), src.size(1), device)
    BURN_IN = 3

    pair_ks = []
    for p_idx in range(N_PAIRWISE_INITS):
        torch.manual_seed(5555 + p_idx * 2)
        noise1 = torch.randn_like(base_init) * 0.05
        s1 = base_init + noise1

        torch.manual_seed(5556 + p_idx * 2)
        noise2 = torch.randn_like(base_init) * 0.05
        s2 = base_init + noise2

        for _ in range(BURN_IN):
            s1, _ = model.dynamics_step(s1, context)
            s2, _ = model.dynamics_step(s2, context)

        prev_dist = (s1 - s2).reshape(src.size(0), -1).norm(dim=-1)
        k_values = []
        for t in range(TRAJECTORY_T):
            s1, _ = model.dynamics_step(s1, context)
            s2, _ = model.dynamics_step(s2, context)
            curr_dist = (s1 - s2).reshape(src.size(0), -1).norm(dim=-1)
            k_per = (curr_dist / prev_dist.clamp(min=1e-8)).mean().item()
            k_values.append(k_per)
            prev_dist = curr_dist

        stable_k = k_values[1:10]
        if stable_k:
            pair_ks.append(float(np.mean(stable_k)))

    return {
        "method": "pairwise",
        "mean_k": round(float(np.mean(pair_ks)), 6),
        "std_k": round(float(np.std(pair_ks)), 6),
        "per_pair_k": [round(v, 6) for v in pair_ks],
        "n_pairs": len(pair_ks),
        "burn_in_steps": BURN_IN,
    }


# === ESTIMATOR D: Full Jacobian eigenvalues (d=64 only) ===

def estimator_full_jacobian(model, device, seq_len, max_state_dim=1200):
    """Compute full system Jacobian eigenvalues (L*d × L*d). Feasible for small models."""
    state_dim = model.d_model * seq_len
    if state_dim > max_state_dim:
        return {
            "method": "full_jacobian",
            "skipped": True,
            "reason": f"state_dim={state_dim} (d={model.d_model}*L={seq_len}) > max={max_state_dim}",
        }

    model.eval()
    full_seed(9999)
    n_samples = 4
    src, _ = generate_batch("addition", n_samples, seq_len, 64)
    src = src.to(device)

    with torch.no_grad():
        context = model.encode(src)
        s_star = model.init_state(src.size(0), src.size(1), device)
        for _ in range(FP_T):
            s_star, _ = model.dynamics_step(s_star, context)

    d = model.d_model
    L = seq_len
    sample_rhos = []

    for b_idx in range(n_samples):
        s_single = s_star[b_idx:b_idx+1].detach().clone()
        ctx_single = context[b_idx:b_idx+1]

        jacobian_rows = []
        for flat_out in range(L * d):
            s_in = s_single.detach().requires_grad_(True)
            s_out, _ = model.dynamics_step(s_in, ctx_single)
            tok_idx = flat_out // d
            dim_idx = flat_out % d
            scalar = s_out[0, tok_idx, dim_idx]
            grad = torch.autograd.grad(scalar, s_in, retain_graph=False)[0]
            jacobian_rows.append(grad[0].reshape(-1).detach().cpu())

        J = torch.stack(jacobian_rows)  # (L*d, L*d)
        eigenvalues = torch.linalg.eigvals(J)
        spectral_radius = eigenvalues.abs().max().item()
        sample_rhos.append(spectral_radius)
        print(f"        sample {b_idx}: full Jacobian rho={spectral_radius:.4f} "
              f"(matrix {L*d}x{L*d})", flush=True)

    mean_spectral_radius = float(np.mean(sample_rhos))

    return {
        "method": "full_jacobian",
        "skipped": False,
        "mean_spectral_radius": round(mean_spectral_radius, 6),
        "std_spectral_radius": round(float(np.std(sample_rhos)), 6),
        "per_sample_rho": [round(v, 6) for v in sample_rhos],
        "state_dim": L * d,
        "n_samples": n_samples,
    }


def run_all_estimators(model, device, seq_len, include_jacobian=False):
    """Run all k estimators on a single model."""
    results = {}

    print("      Estimator A (standard)...", flush=True)
    t0 = time.time()
    results["standard"] = estimator_standard(model, device, seq_len)
    results["standard"]["time_s"] = round(time.time() - t0, 1)
    print(f"        k={results['standard']['mean_k']:.4f} ({results['standard']['time_s']:.1f}s)", flush=True)

    print("      Estimator B (random direction)...", flush=True)
    t0 = time.time()
    results["random_direction"] = estimator_random_direction(model, device, seq_len)
    results["random_direction"]["time_s"] = round(time.time() - t0, 1)
    print(f"        k={results['random_direction']['mean_k']:.4f} ({results['random_direction']['time_s']:.1f}s)", flush=True)

    print("      Estimator C (pairwise)...", flush=True)
    t0 = time.time()
    results["pairwise"] = estimator_pairwise(model, device, seq_len)
    results["pairwise"]["time_s"] = round(time.time() - t0, 1)
    print(f"        k={results['pairwise']['mean_k']:.4f} ({results['pairwise']['time_s']:.1f}s)", flush=True)

    if include_jacobian:
        print("      Estimator D (full Jacobian)...", flush=True)
        t0 = time.time()
        results["full_jacobian"] = estimator_full_jacobian(model, device, seq_len)
        results["full_jacobian"]["time_s"] = round(time.time() - t0, 1)
        if not results["full_jacobian"].get("skipped"):
            print(f"        spectral_radius={results['full_jacobian']['mean_spectral_radius']:.4f} "
                  f"({results['full_jacobian']['time_s']:.1f}s)", flush=True)
        else:
            print(f"        SKIPPED: {results['full_jacobian']['reason']}", flush=True)

    return results


def compute_analysis(all_results):
    """Cross-estimator agreement analysis."""
    analysis = {"models": {}, "verdict": None}

    max_disagreement = 0
    all_standard_k = []
    all_random_k = []
    all_pairwise_k = []

    for model_name, result in all_results.items():
        estimators = result["estimators"]
        k_std = estimators["standard"]["mean_k"]
        k_rand = estimators["random_direction"]["mean_k"]
        k_pair = estimators["pairwise"]["mean_k"]

        all_standard_k.append(k_std)
        all_random_k.append(k_rand)
        all_pairwise_k.append(k_pair)

        disagreement_rand = abs(k_std - k_rand)
        disagreement_pair = abs(k_std - k_pair)
        max_local = max(disagreement_rand, disagreement_pair)
        max_disagreement = max(max_disagreement, max_local)

        jac_note = None
        if "full_jacobian" in estimators and not estimators["full_jacobian"].get("skipped"):
            jac_note = estimators["full_jacobian"]["mean_spectral_radius"]

        analysis["models"][model_name] = {
            "k_standard": k_std,
            "k_random_direction": k_rand,
            "k_pairwise": k_pair,
            "k_jacobian_rho": jac_note,
            "max_disagreement": round(max_local, 6),
        }

    analysis["max_cross_model_disagreement"] = round(max_disagreement, 6)

    # Verdict
    if max_disagreement < 0.002:
        analysis["verdict"] = "ESTIMATOR_CONSISTENT"
        analysis["verdict_detail"] = (
            f"All estimators agree within {max_disagreement:.4f}. "
            "The k setpoint is NOT an artifact of the measurement protocol."
        )
    elif max_disagreement < 0.005:
        analysis["verdict"] = "WEAK_DISAGREEMENT"
        analysis["verdict_detail"] = (
            f"Max disagreement {max_disagreement:.4f} — small but non-negligible. "
            "Standard estimator may have mild bias but the setpoint direction is real."
        )
    else:
        analysis["verdict"] = "ESTIMATOR_ARTIFACT"
        analysis["verdict_detail"] = (
            f"Max disagreement {max_disagreement:.4f} — LARGE. "
            "The standard k estimator has a significant bias. "
            "The k≈0.988 setpoint may be an artifact."
        )

    # Check VT vs FT direction consistency
    vt_models = [n for n in all_results if "vt" in n]
    ft_models = [n for n in all_results if "ft" in n]

    for estimator_name in ["standard", "random_direction", "pairwise"]:
        vt_ks = [all_results[n]["estimators"][estimator_name]["mean_k"] for n in vt_models]
        ft_ks = [all_results[n]["estimators"][estimator_name]["mean_k"] for n in ft_models]
        if vt_ks and ft_ks:
            mean_dk = np.mean(vt_ks) - np.mean(ft_ks)
            analysis[f"dk_{estimator_name}"] = round(float(mean_dk), 6)

    return analysis


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D37: Alternate k Estimator Validation", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Models to test: {len(MODEL_CONFIGS)}", flush=True)

    existing = load_results()
    all_results = existing.get("models", {}) if existing else {}
    completed = set(all_results.keys())

    total = len(MODEL_CONFIGS)
    done = len(completed)
    print(f"\nD37: {total} total models ({done} done)\n", flush=True)

    for model_name, cfg in MODEL_CONFIGS.items():
        if model_name in completed:
            print(f"  [{model_name}] already done, skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  Model: {model_name}", flush=True)
        print(f"  Source: {cfg['source']}, arch: d={cfg['arch']['d_model']} h={cfg['arch']['n_heads']}", flush=True)
        print(f"  Variant: {cfg['variant']}, seed: {cfg['seed']}", flush=True)
        print(f"{'='*60}\n", flush=True)

        print("    Training model from scratch...", flush=True)
        t0 = time.time()
        full_seed(cfg["seed"])
        model = make_model(cfg, device)
        model = train_model(model, device, cfg)
        train_time = time.time() - t0
        print(f"    Training done in {train_time:.1f}s\n", flush=True)

        include_jac = cfg["arch"]["d_model"] <= 96
        print(f"    Running estimators (Jacobian: {include_jac})...", flush=True)
        estimator_results = run_all_estimators(model, device, cfg["seq_len"], include_jac)

        all_results[model_name] = {
            "config": cfg,
            "train_time_s": round(train_time, 1),
            "estimators": estimator_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        done += 1
        results_out = {
            "experiment": "D37_alt_k_estimator",
            "models": all_results,
            "n_complete": done,
            "n_total": total,
        }

        if done == total:
            results_out["analysis"] = compute_analysis(all_results)

        save_results(results_out)
        print(f"\n  [{done}/{total}] saved\n", flush=True)

        del model
        torch.cuda.empty_cache()

    if done == total:
        analysis = compute_analysis(all_results)
        results_out = {
            "experiment": "D37_alt_k_estimator",
            "models": all_results,
            "n_complete": done,
            "n_total": total,
            "analysis": analysis,
        }
        save_results(results_out)

        print("\n" + "="*60, flush=True)
        print("  ANALYSIS SUMMARY", flush=True)
        print("="*60, flush=True)
        for name, m in analysis["models"].items():
            print(f"  {name}:", flush=True)
            print(f"    standard={m['k_standard']:.4f}  random_dir={m['k_random_direction']:.4f}  "
                  f"pairwise={m['k_pairwise']:.4f}  disagreement={m['max_disagreement']:.4f}", flush=True)
            if m['k_jacobian_rho'] is not None:
                print(f"    jacobian_rho={m['k_jacobian_rho']:.4f}", flush=True)

        print(f"\n  VERDICT: {analysis['verdict']}", flush=True)
        print(f"  {analysis['verdict_detail']}", flush=True)

        for est_name in ["standard", "random_direction", "pairwise"]:
            dk_key = f"dk_{est_name}"
            if dk_key in analysis:
                print(f"  dk ({est_name}): {analysis[dk_key]:.4f}", flush=True)


if __name__ == "__main__":
    main()
