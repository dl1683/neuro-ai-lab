"""
Experiment D35: Non-Arithmetic Generalization (Prefix Sum)

Tests whether VT k-suppression generalizes beyond right-to-left carry
propagation (addition/subtraction) to left-to-right cumulative accumulation.

Prefix sum: output[i] = sum(input[0:i+1]) mod V.
Sequential depth is O(seq_len) — each output depends on ALL previous inputs.
Fundamentally different structure from addition's right-to-left carry:
  - Addition: carry propagates from LSB to MSB (right-to-left)
  - Prefix sum: accumulation propagates from position 0 to L-1 (left-to-right)
  - Same computational depth class, different dependency structure

This is a P5 test: if VT k-suppression appears here, it is a general property
of weight-tied dynamics, not an artifact of carry-based arithmetic.

Design:
  seq_len = {6, 8, 10, 12}  (4 depths, sequential depth = seq_len)
  4 seeds per depth (matched across FT/VT)
  2 variants (fixed_t, variable_t)
  Total: 4 x 4 x 2 = 32 runs

Key difference from addition experiments:
  Loss and accuracy computed over ALL seq_len positions, not just first half.

Predictions (from Prop 35):
  If VT k-suppression is a general property:
    VT k < FT k at ALL depths (matching D31 direction, p < 0.05)
    VT rho ~ 1.003 (VT ceiling, matching D32)
  If k-suppression is addition-specific:
    No significant FT/VT difference in k
    Prop 35 scope must be narrowed to carry-based tasks
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
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed
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
TRAIN_T = 10
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
T_MIN = min(VARIABLE_T_RANGE)

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30

SEEDS = [42, 137, 256, 512]

SEQ_LEN_CONFIGS = [
    {"seq_len": 6},
    {"seq_len": 8},
    {"seq_len": 10},
    {"seq_len": 12},
]

VARIANTS = ["fixed_t", "variable_t"]
Q_CHECKPOINT_INTERVAL = 2000

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d35_prefix_sum.json"


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


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


@torch.no_grad()
def measure_q_at_t_min(model, device, seq_len):
    """Measure T_MIN eval accuracy (full-sequence) as solvability proxy."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    model.eval()
    torch.manual_seed(12345)
    src, tgt = generate_batch("prefix_sum", 1024, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    logits = model(src, T_MIN)
    preds = logits.argmax(dim=-1)
    seq_acc = (preds == tgt).all(dim=1).float().mean().item()

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    random.setstate(py_state)
    np.random.set_state(np_state)
    return seq_acc


def train_model(model, device, seq_len, variant, seed):
    """Train on prefix sum with periodic q measurement at T_min."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    best_acc = 0.0
    q_trajectory = {}
    phase_transition_step = None

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("prefix_sum", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if variant == "variable_t":
            t_steps = random.choice(VARIABLE_T_RANGE)
        else:
            t_steps = TRAIN_T

        logits = model(src, t_steps)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % Q_CHECKPOINT_INTERVAL == 0 or step == 1:
            q = measure_q_at_t_min(model, device, seq_len)
            q_trajectory[step] = round(q, 4)

            with torch.no_grad():
                eval_logits = model(src, TRAIN_T)
                preds = eval_logits.argmax(dim=-1)
                seq_acc = (preds == tgt).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)

            if phase_transition_step is None and seq_acc > 0.5:
                phase_transition_step = step

            t_label = f"T={t_steps}" if variant == "variable_t" else f"T={TRAIN_T}"
            print(f"      Step {step:>6d}/{TRAINING_STEPS} | {t_label} | "
                  f"Loss: {loss.item():.4f} | Acc: {seq_acc:.4f} | "
                  f"q(T={T_MIN}): {q:.4f}", flush=True)
            model.train()

    elapsed = time.time() - t0

    mean_tmin_acc = sum(q_trajectory.values()) / max(len(q_trajectory), 1)

    print(f"      Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f} | "
          f"Cumulative q: {mean_tmin_acc:.4f} | Phase trans: step {phase_transition_step}",
          flush=True)
    return elapsed, best_acc, q_trajectory, mean_tmin_acc, phase_transition_step


def measure_spectral_radius(model, device, seq_len, n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("prefix_sum", 256, seq_len, VOCAB_SIZE)
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
    return {"mean": round(mean_rho, 4), "std": round(std_rho, 4)}


@torch.no_grad()
def measure_contraction_summary(model, device, seq_len):
    model.eval()
    full_seed(9999)
    src, tgt = generate_batch("prefix_sum", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

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
    mean_k = float(np.mean(stable_k))
    std_k = float(np.std(stable_k))

    s = model.init_state(src.size(0), src.size(1), device)
    actual_T99 = None
    for t in range(1, TRAJECTORY_T + 1):
        s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        preds = logits.argmax(dim=-1)
        seq_acc = (preds == tgt).all(dim=1).float().mean().item()
        if seq_acc >= 0.99 and actual_T99 is None:
            actual_T99 = t

    return {
        "mean_k": round(mean_k, 4),
        "std_k": round(std_k, 4),
        "T_99": actual_T99,
    }


def run_one_config(seq_cfg, variant, seed, device, completed_keys):
    seq_len = seq_cfg["seq_len"]
    key = f"L{seq_len}_{variant}_s{seed}"

    if key in completed_keys:
        return None

    print(f"\n    L={seq_len}, seed={seed}, variant={variant}", flush=True)

    full_seed(seed)
    model = make_model(device)

    train_time, best_acc, q_traj, mean_tmin_acc, phase_step = train_model(
        model, device, seq_len, variant, seed
    )

    print(f"      Measuring contraction...", flush=True)
    contraction = measure_contraction_summary(model, device, seq_len)

    print(f"      Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device, seq_len)

    print(f"      rho={spectral['mean']:.4f}+/-{spectral['std']:.4f}, "
          f"k={contraction['mean_k']:.4f}, T_99={contraction['T_99']}, "
          f"acc={best_acc:.4f}, cumQ={mean_tmin_acc:.4f}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "key": key,
        "seq_len": seq_len,
        "variant": variant,
        "seed": seed,
        "params": n_params,
        "accuracy": round(best_acc, 4),
        "spectral_radius": spectral,
        "contraction": contraction,
        "train_time_s": round(train_time, 1),
        "q_trajectory": q_traj,
        "mean_tmin_acc": round(mean_tmin_acc, 4),
        "phase_transition_step": phase_step,
        "run_signature": {
            "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "train_T": TRAIN_T,
            "variable_t_range": VARIABLE_T_RANGE,
        },
    }


def compute_analysis(results):
    """Paired FT vs VT comparison per seq_len, plus overall k/rho tests."""
    print("\n" + "=" * 60, flush=True)
    print("  D35 ANALYSIS: PREFIX SUM VT GENERALIZATION", flush=True)
    print("=" * 60, flush=True)

    by_len_seed = {}
    for run in results["runs"]:
        sl = run["seq_len"]
        s = run["seed"]
        v = run["variant"]
        key = (sl, s)
        if key not in by_len_seed:
            by_len_seed[key] = {}
        by_len_seed[key][v] = run

    print(f"\n  {'L':>3s} | {'k_FT':>8s} | {'k_VT':>8s} | {'dk':>8s} | "
          f"{'rho_FT':>8s} | {'rho_VT':>8s} | {'drho':>8s} | "
          f"{'T99_FT':>6s} | {'T99_VT':>6s} | {'n':>2s}", flush=True)
    print("  " + "-" * 85, flush=True)

    per_length = []
    all_ft_k = []
    all_vt_k = []
    all_ft_rho = []
    all_vt_rho = []
    all_paired_rows = []

    for sl in sorted(set(k[0] for k in by_len_seed)):
        ft_ks, vt_ks = [], []
        ft_rhos, vt_rhos = [], []
        ft_t99s, vt_t99s = [], []

        for seed in SEEDS:
            pair = by_len_seed.get((sl, seed), {})
            if "fixed_t" in pair and "variable_t" in pair:
                ft = pair["fixed_t"]
                vt = pair["variable_t"]
                ft_k = ft["contraction"]["mean_k"]
                vt_k = vt["contraction"]["mean_k"]
                ft_rho = ft["spectral_radius"]["mean"]
                vt_rho = vt["spectral_radius"]["mean"]
                ft_ks.append(ft_k)
                vt_ks.append(vt_k)
                ft_rhos.append(ft_rho)
                vt_rhos.append(vt_rho)
                ft_t99s.append(ft["contraction"]["T_99"])
                vt_t99s.append(vt["contraction"]["T_99"])
                all_ft_k.append(ft_k)
                all_vt_k.append(vt_k)
                all_ft_rho.append(ft_rho)
                all_vt_rho.append(vt_rho)
                all_paired_rows.append({
                    "L": sl, "seed": seed,
                    "k_ft": round(ft_k, 4), "k_vt": round(vt_k, 4),
                    "rho_ft": round(ft_rho, 4), "rho_vt": round(vt_rho, 4),
                    "T99_ft": ft["contraction"]["T_99"],
                    "T99_vt": vt["contraction"]["T_99"],
                    "acc_ft": ft["accuracy"], "acc_vt": vt["accuracy"],
                })

        if len(ft_ks) < 2:
            continue

        dk = np.mean(vt_ks) - np.mean(ft_ks)
        drho = np.mean(vt_rhos) - np.mean(ft_rhos)
        ft_t99_str = "/".join(str(t) if t else "X" for t in ft_t99s)
        vt_t99_str = "/".join(str(t) if t else "X" for t in vt_t99s)

        print(f"  {sl:3d} | {np.mean(ft_ks):8.4f} | {np.mean(vt_ks):8.4f} | "
              f"{dk:+8.4f} | {np.mean(ft_rhos):8.4f} | {np.mean(vt_rhos):8.4f} | "
              f"{drho:+8.4f} | {ft_t99_str:>6s} | {vt_t99_str:>6s} | "
              f"{len(ft_ks):2d}", flush=True)

        per_length.append({
            "seq_len": sl,
            "mean_ft_k": round(float(np.mean(ft_ks)), 4),
            "mean_vt_k": round(float(np.mean(vt_ks)), 4),
            "delta_k": round(float(dk), 4),
            "mean_ft_rho": round(float(np.mean(ft_rhos)), 4),
            "mean_vt_rho": round(float(np.mean(vt_rhos)), 4),
            "delta_rho": round(float(drho), 4),
            "n_seeds": len(ft_ks),
        })

    # Overall paired tests (all seeds pooled)
    if len(all_ft_k) >= 4:
        t_k, p_k = scipy_stats.ttest_rel(all_vt_k, all_ft_k)
        d_k = (np.mean(all_vt_k) - np.mean(all_ft_k)) / np.std(
            np.array(all_vt_k) - np.array(all_ft_k), ddof=1
        ) if np.std(np.array(all_vt_k) - np.array(all_ft_k), ddof=1) > 0 else 0.0

        t_rho, p_rho = scipy_stats.ttest_rel(all_vt_rho, all_ft_rho)
        d_rho = (np.mean(all_vt_rho) - np.mean(all_ft_rho)) / np.std(
            np.array(all_vt_rho) - np.array(all_ft_rho), ddof=1
        ) if np.std(np.array(all_vt_rho) - np.array(all_ft_rho), ddof=1) > 0 else 0.0

        vt_lower = sum(1 for vk, fk in zip(all_vt_k, all_ft_k) if vk < fk)

        print(f"\n  OVERALL PAIRED TESTS (n={len(all_ft_k)}):", flush=True)
        print(f"    k:   Dk={np.mean(all_vt_k)-np.mean(all_ft_k):+.4f}, "
              f"t={t_k:.3f}, p={p_k:.6f}, d={d_k:.2f}, "
              f"VT<FT: {vt_lower}/{len(all_ft_k)}", flush=True)
        print(f"    rho: Drho={np.mean(all_vt_rho)-np.mean(all_ft_rho):+.4f}, "
              f"t={t_rho:.3f}, p={p_rho:.6f}, d={d_rho:.2f}", flush=True)

        # Sign test for k
        try:
            from scipy.stats import binomtest
            p_sign = binomtest(vt_lower, len(all_ft_k), 0.5).pvalue
        except ImportError:
            from scipy.stats import binom_test
            p_sign = binom_test(vt_lower, len(all_ft_k), 0.5)

        print(f"    Sign test (k): {vt_lower}/{len(all_ft_k)} VT<FT, p={p_sign:.6f}",
              flush=True)

        # Verdict
        if p_k < 0.05 and vt_lower == len(all_ft_k):
            verdict = "VT K-SUPPRESSION GENERALIZES (unanimous, p<0.05)"
        elif p_k < 0.05:
            verdict = "VT K-SUPPRESSION SIGNIFICANT (non-unanimous)"
        elif vt_lower > len(all_ft_k) * 0.75:
            verdict = "DIRECTIONAL but not significant — need more seeds"
        else:
            verdict = "NO GENERALIZATION — k-suppression is task-specific"

        print(f"\n  VERDICT: {verdict}", flush=True)

        overall_stats = {
            "n_pairs": len(all_ft_k),
            "k_delta": round(float(np.mean(all_vt_k) - np.mean(all_ft_k)), 4),
            "k_t_stat": round(float(t_k), 3),
            "k_p_val": round(float(p_k), 6),
            "k_cohen_d": round(float(d_k), 2),
            "k_vt_lower_count": vt_lower,
            "k_sign_p": round(float(p_sign), 6),
            "rho_delta": round(float(np.mean(all_vt_rho) - np.mean(all_ft_rho)), 4),
            "rho_t_stat": round(float(t_rho), 3),
            "rho_p_val": round(float(p_rho), 6),
            "rho_cohen_d": round(float(d_rho), 2),
            "verdict": verdict,
        }
    else:
        overall_stats = {"error": "insufficient paired data"}
        verdict = "INSUFFICIENT DATA"

    # Comparison with D31 addition results (if available)
    print(f"\n  COMPARISON WITH D31 (ADDITION):", flush=True)
    print(f"    D31 k: Dk=-0.0023, p=0.000017, d=-3.66, 8/8 VT<FT", flush=True)
    if len(all_ft_k) >= 4:
        unanimity = "unanimous" if vt_lower == len(all_ft_k) else f"{vt_lower}/{len(all_ft_k)}"
        print(f"    D35 k: Dk={np.mean(all_vt_k)-np.mean(all_ft_k):+.4f}, "
              f"p={p_k:.6f}, d={d_k:.2f}, "
              f"{unanimity} VT<FT", flush=True)

    results["per_length_analysis"] = per_length
    results["paired_rows"] = all_paired_rows
    results["overall_stats"] = overall_stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    existing = load_checkpoint()
    completed_keys = set()
    if existing and "runs" in existing:
        for run in existing["runs"]:
            completed_keys.add(run["key"])
        print(f"Resuming: {len(completed_keys)} runs complete")
    else:
        existing = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D35: Non-arithmetic generalization test (prefix sum). "
                       "Tests whether VT k-suppression from D31 generalizes to "
                       "left-to-right cumulative accumulation (vs right-to-left "
                       "carry in addition). P5 falsification experiment.",
            "config": {
                "task": "prefix_sum",
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "batch_size": BATCH_SIZE, "lr": LR,
                "variable_t_range": VARIABLE_T_RANGE,
                "seeds": SEEDS, "seq_lens": [c["seq_len"] for c in SEQ_LEN_CONFIGS],
                "q_checkpoint_interval": Q_CHECKPOINT_INTERVAL,
                "fp_steps": FP_T, "trajectory_steps": TRAJECTORY_T,
                "loss_over": "full_sequence",
            },
            "runs": [],
        }

    total_runs = len(SEQ_LEN_CONFIGS) * len(SEEDS) * len(VARIANTS)
    done = len(completed_keys)
    print(f"\nD35: {total_runs} total runs ({done} done)\n", flush=True)

    for seq_cfg in SEQ_LEN_CONFIGS:
        sl = seq_cfg["seq_len"]
        print(f"\n{'=' * 60}", flush=True)
        print(f"  L={sl} (sequential depth = {sl})", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seed in SEEDS:
            for variant in VARIANTS:
                result = run_one_config(seq_cfg, variant, seed, device, completed_keys)
                if result is not None:
                    existing["runs"].append(result)
                    completed_keys.add(result["key"])
                    save_results(existing)
                    done += 1
                    print(f"\n  [{done}/{total_runs}] saved", flush=True)

    existing["completed"] = datetime.now(timezone.utc).isoformat()
    compute_analysis(existing)
    save_results(existing)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
