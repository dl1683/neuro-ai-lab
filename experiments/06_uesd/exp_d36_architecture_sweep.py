"""
Experiment D36: Architecture Sweep for VT k-Contraction Setpoint Robustness

Tests whether VT k≈0.988 is an architecture-specific attractor (d=128/heads=4)
or generalizes across different model dimensions. This is the #1 requirement
from Codex's 9/10 evidence bar (2026-05-25 review).

Design:
  5 architecture configs x 2 seeds x 2 variants = 20 runs
  Task: addition D=6 (L=12) — well-understood, fast, reliable baseline

  Configs:
    A: SMALL    — d_model=64,  heads=2, d_ff=256   (head_dim=32, ~175K params)
    B: BASELINE — d_model=128, heads=4, d_ff=512   (head_dim=32, ~694K params)
    C: LARGE    — d_model=256, heads=8, d_ff=1024  (head_dim=32, ~2.7M params)
    D: MANY_HEADS — d_model=128, heads=8, d_ff=512 (head_dim=16, ~694K params)
    E: FEW_HEADS  — d_model=128, heads=2, d_ff=512 (head_dim=64, ~694K params)

  Note: Configs A, B, C all share head_dim=32. D and E vary head_dim at fixed d_model.
  This separates d_model scaling from head_dim effects.

Predictions:
  If k≈0.988 is UNIVERSAL:
    VT k ≈ 0.988 across all 5 configs (within measurement noise ±0.001)
  If k≈0.988 is ARCHITECTURE-SPECIFIC:
    VT k varies significantly across configs (range > 0.003)
    Likely correlates with d_model, head_dim, or capacity
  Either way: dk < 0 should hold for all configs (VT suppresses k relative to FT)
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
from shared.training import set_seed
from shared.data import generate_batch

TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
N_ENC_LAYERS = 2
MAX_LEN = 64
TRAIN_T = 10
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
T_MIN = min(VARIABLE_T_RANGE)

SEQ_LEN = 12
CARRY_DEPTH = 6

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30

SEEDS = [42, 137]
VARIANTS = ["fixed_t", "variable_t"]
Q_CHECKPOINT_INTERVAL = 2000

ARCH_CONFIGS = [
    {"name": "small",      "d_model": 64,  "n_heads": 2, "d_ff": 256},
    {"name": "baseline",   "d_model": 128, "n_heads": 4, "d_ff": 512},
    {"name": "large",      "d_model": 256, "n_heads": 8, "d_ff": 1024},
    {"name": "many_heads", "d_model": 128, "n_heads": 8, "d_ff": 512},
    {"name": "few_heads",  "d_model": 128, "n_heads": 2, "d_ff": 512},
]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d36_architecture_sweep.json"


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


def make_model(arch_cfg, device):
    model = UESDModel(
        VOCAB_SIZE, arch_cfg["d_model"], arch_cfg["n_heads"],
        arch_cfg["d_ff"], N_ENC_LAYERS, MAX_LEN,
    )
    return model.to(device)


@torch.no_grad()
def measure_q_at_t_min(model, device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    model.eval()
    half = SEQ_LEN // 2
    torch.manual_seed(12345)
    src, tgt = generate_batch("addition", 1024, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    logits = model(src, T_MIN)
    preds = logits[:, :half].argmax(dim=-1)
    seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    random.setstate(py_state)
    np.random.set_state(np_state)
    return seq_acc


def train_model(model, device, variant, seed):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0
    q_trajectory = {}
    phase_transition_step = None

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if variant == "variable_t":
            t_steps = random.choice(VARIABLE_T_RANGE)
        else:
            t_steps = TRAIN_T

        logits = model(src, t_steps)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % Q_CHECKPOINT_INTERVAL == 0 or step == 1:
            q = measure_q_at_t_min(model, device)
            q_trajectory[step] = round(q, 4)

            with torch.no_grad():
                eval_logits = model(src, TRAIN_T)
                preds = eval_logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
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


def measure_spectral_radius(model, device, n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 256, SEQ_LEN, VOCAB_SIZE)
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
def measure_contraction_summary(model, device):
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
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
    tgt_result = tgt[:, :half]
    actual_T99 = None
    for t in range(1, TRAJECTORY_T + 1):
        s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        preds = logits[:, :half].argmax(dim=-1)
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        if seq_acc >= 0.99 and actual_T99 is None:
            actual_T99 = t

    return {
        "mean_k": round(mean_k, 4),
        "std_k": round(std_k, 4),
        "T_99": actual_T99,
    }


def run_one_config(arch_cfg, variant, seed, device, completed_keys):
    arch_name = arch_cfg["name"]
    key = f"{arch_name}_{variant}_s{seed}"

    if key in completed_keys:
        return None

    print(f"\n    arch={arch_name} (d={arch_cfg['d_model']}, h={arch_cfg['n_heads']}, "
          f"ff={arch_cfg['d_ff']}), seed={seed}, variant={variant}", flush=True)

    full_seed(seed)
    model = make_model(arch_cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params:,}", flush=True)

    train_time, best_acc, q_traj, mean_tmin_acc, phase_step = train_model(
        model, device, variant, seed
    )

    print(f"      Measuring contraction...", flush=True)
    contraction = measure_contraction_summary(model, device)

    print(f"      Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device)

    print(f"      rho={spectral['mean']:.4f}+/-{spectral['std']:.4f}, "
          f"k={contraction['mean_k']:.4f}, T_99={contraction['T_99']}, "
          f"acc={best_acc:.4f}, cumQ={mean_tmin_acc:.4f}", flush=True)

    return {
        "key": key,
        "arch_name": arch_name,
        "d_model": arch_cfg["d_model"],
        "n_heads": arch_cfg["n_heads"],
        "d_ff": arch_cfg["d_ff"],
        "head_dim": arch_cfg["d_model"] // arch_cfg["n_heads"],
        "carry_depth": CARRY_DEPTH,
        "seq_len": SEQ_LEN,
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


def compute_architecture_analysis(results):
    """Analyze VT k setpoint across architectures."""
    print("\n" + "=" * 70, flush=True)
    print("  ARCHITECTURE SWEEP ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    by_arch_seed = {}
    for run in results["runs"]:
        a = run["arch_name"]
        s = run["seed"]
        v = run["variant"]
        key = (a, s)
        if key not in by_arch_seed:
            by_arch_seed[key] = {}
        by_arch_seed[key][v] = run

    print(f"\n  {'Arch':>12s} | {'d':>4s} | {'h':>3s} | {'FT_k':>7s} | {'VT_k':>7s} | "
          f"{'dk':>8s} | {'FT_rho':>7s} | {'VT_rho':>7s} | {'drho':>8s} | "
          f"{'FT_T99':>6s} | {'VT_T99':>6s} | {'n':>2s}", flush=True)
    print("  " + "-" * 105, flush=True)

    arch_summary = []
    all_vt_k = []
    all_dk = []

    for arch_cfg in ARCH_CONFIGS:
        arch_name = arch_cfg["name"]
        ft_k_vals, vt_k_vals = [], []
        ft_rho_vals, vt_rho_vals = [], []
        ft_t99_vals, vt_t99_vals = [], []
        dk_vals, drho_vals = [], []

        for seed in SEEDS:
            pair = by_arch_seed.get((arch_name, seed), {})
            if "fixed_t" in pair and "variable_t" in pair:
                ft_k = pair["fixed_t"]["contraction"]["mean_k"]
                vt_k = pair["variable_t"]["contraction"]["mean_k"]
                ft_rho = pair["fixed_t"]["spectral_radius"]["mean"]
                vt_rho = pair["variable_t"]["spectral_radius"]["mean"]
                ft_t99 = pair["fixed_t"]["contraction"]["T_99"]
                vt_t99 = pair["variable_t"]["contraction"]["T_99"]

                ft_k_vals.append(ft_k)
                vt_k_vals.append(vt_k)
                ft_rho_vals.append(ft_rho)
                vt_rho_vals.append(vt_rho)
                ft_t99_vals.append(ft_t99 if ft_t99 is not None else TRAJECTORY_T + 1)
                vt_t99_vals.append(vt_t99 if vt_t99 is not None else TRAJECTORY_T + 1)
                dk_vals.append(vt_k - ft_k)
                drho_vals.append(vt_rho - ft_rho)
                all_vt_k.append(vt_k)
                all_dk.append(vt_k - ft_k)

        if not dk_vals:
            continue

        n = len(dk_vals)
        mean_ft_k = np.mean(ft_k_vals)
        mean_vt_k = np.mean(vt_k_vals)
        mean_dk = np.mean(dk_vals)
        mean_ft_rho = np.mean(ft_rho_vals)
        mean_vt_rho = np.mean(vt_rho_vals)
        mean_drho = np.mean(drho_vals)
        mean_ft_t99 = np.mean(ft_t99_vals)
        mean_vt_t99 = np.mean(vt_t99_vals)

        print(f"  {arch_name:>12s} | {arch_cfg['d_model']:4d} | {arch_cfg['n_heads']:3d} | "
              f"{mean_ft_k:7.4f} | {mean_vt_k:7.4f} | {mean_dk:+8.4f} | "
              f"{mean_ft_rho:7.4f} | {mean_vt_rho:7.4f} | {mean_drho:+8.4f} | "
              f"{mean_ft_t99:6.1f} | {mean_vt_t99:6.1f} | {n:2d}", flush=True)

        arch_summary.append({
            "arch_name": arch_name,
            "d_model": arch_cfg["d_model"],
            "n_heads": arch_cfg["n_heads"],
            "d_ff": arch_cfg["d_ff"],
            "head_dim": arch_cfg["d_model"] // arch_cfg["n_heads"],
            "mean_ft_k": round(float(mean_ft_k), 4),
            "mean_vt_k": round(float(mean_vt_k), 4),
            "mean_dk": round(float(mean_dk), 4),
            "mean_ft_rho": round(float(mean_ft_rho), 4),
            "mean_vt_rho": round(float(mean_vt_rho), 4),
            "mean_drho": round(float(mean_drho), 4),
            "n_seeds": n,
            "all_dk": [round(d, 4) for d in dk_vals],
            "all_vt_k": [round(v, 4) for v in vt_k_vals],
        })

    arch_mean_vt_k = [s["mean_vt_k"] for s in arch_summary]
    if len(arch_mean_vt_k) >= 3:
        arch_vt_k_range = max(arch_mean_vt_k) - min(arch_mean_vt_k)
        arch_vt_k_std = float(np.std(arch_mean_vt_k))
        seed_vt_k_range = max(all_vt_k) - min(all_vt_k) if all_vt_k else 0
        seed_vt_k_std = float(np.std(all_vt_k)) if all_vt_k else 0
        dk_neg_count = sum(1 for d in all_dk if d < 0)
        dk_total = len(all_dk)

        print(f"\n  ARCHITECTURE-LEVEL VT k: range={arch_vt_k_range:.4f}, "
              f"std={arch_vt_k_std:.4f}, mean={np.mean(arch_mean_vt_k):.4f}", flush=True)
        print(f"  SEED-LEVEL VT k: range={seed_vt_k_range:.4f}, "
              f"std={seed_vt_k_std:.4f} (includes within-arch seed noise)", flush=True)
        print(f"  dk sign: {dk_neg_count}/{dk_total} negative", flush=True)

        if arch_vt_k_range < 0.003:
            print("  VERDICT: VT k setpoint is ARCHITECTURE-ROBUST "
                  "(arch-mean range < 0.003)", flush=True)
        elif arch_vt_k_range < 0.010:
            print("  VERDICT: VT k setpoint WEAKLY VARIES with architecture "
                  "(arch-mean range 0.003-0.010)", flush=True)
        else:
            print("  VERDICT: VT k setpoint is ARCHITECTURE-SPECIFIC "
                  "(arch-mean range > 0.010)", flush=True)

        if dk_neg_count == dk_total:
            print(f"  dk UNANIMOUS: {dk_total}/{dk_total} negative across all architectures",
                  flush=True)

    results["architecture_analysis"] = arch_summary


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
            "purpose": "D36: Architecture sweep for VT k-contraction setpoint robustness. "
                       "Tests whether k~0.988 is specific to d=128/h=4 or generalizes "
                       "across architectures. Codex 9/10 bar requirement #1.",
            "config": {
                "task": "addition",
                "carry_depth": CARRY_DEPTH,
                "seq_len": SEQ_LEN,
                "vocab_size": VOCAB_SIZE,
                "n_enc_layers": N_ENC_LAYERS,
                "training_steps": TRAINING_STEPS,
                "train_T": TRAIN_T,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "variable_t_range": VARIABLE_T_RANGE,
                "seeds": SEEDS,
                "architectures": ARCH_CONFIGS,
                "q_checkpoint_interval": Q_CHECKPOINT_INTERVAL,
                "fp_steps": FP_T,
                "trajectory_steps": TRAJECTORY_T,
            },
            "runs": [],
        }

    total_runs = len(ARCH_CONFIGS) * len(SEEDS) * len(VARIANTS)
    done = len(completed_keys)
    print(f"\nD36: {total_runs} total runs ({done} done)\n", flush=True)

    for arch_cfg in ARCH_CONFIGS:
        arch_name = arch_cfg["name"]
        print(f"\n{'=' * 70}", flush=True)
        print(f"  Architecture: {arch_name} "
              f"(d={arch_cfg['d_model']}, h={arch_cfg['n_heads']}, ff={arch_cfg['d_ff']})",
              flush=True)
        print(f"{'=' * 70}", flush=True)

        for seed in SEEDS:
            for variant in VARIANTS:
                result = run_one_config(arch_cfg, variant, seed, device, completed_keys)
                if result is not None:
                    existing["runs"].append(result)
                    completed_keys.add(result["key"])
                    save_results(existing)
                    done += 1
                    print(f"\n  [{done}/{total_runs}] saved", flush=True)

    existing["completed"] = datetime.now(timezone.utc).isoformat()
    compute_architecture_analysis(existing)
    save_results(existing)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
