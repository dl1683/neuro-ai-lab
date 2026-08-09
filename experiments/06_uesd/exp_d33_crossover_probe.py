"""
Experiment D33: Paired-Depth Crossover Probe

Tests whether the D=8 VT rho anomaly is a sharp phase boundary or a smooth
crossover by measuring rho at ADJACENT depths D=7,8,9 with dense seeds.
Also directly measures training-time solvability q to test Prop 34.

Design (from Codex theory review recommendation):
  D = {6, 7, 8, 9, 10} (5 depths, D=6 and D=10 are controls from D28)
  4 seeds per depth (matched across FT/VT for paired comparison)
  2 variants (fixed_t, variable_t)
  Total: 5 depths x 4 seeds x 2 variants = 40 runs

KEY INNOVATION: Training-time q measurement
  Every 2000 steps during training, measure accuracy at T=T_min (=4).
  This gives the empirical solvability fraction q(D, T_min, step) as a
  function of training progress. If Prop 34 is correct, cumulative q
  should correlate with final delta_rho.

Predictions:
  If D=8 is a true phase boundary:
    delta_rho should be negative at D=7, cross zero between D=7-8, positive at D=8-9
  If D=8 is a seed fluke (as suggested by D=10 constant suppression):
    delta_rho ~ -0.0025 at ALL depths D=7,8,9
  If Prop 34 q-model is correct:
    training-time q should correlate with delta_rho across seeds AND depths
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

DEPTH_CONFIGS = [
    {"seq_len": 12, "carry_depth": 6},
    {"seq_len": 14, "carry_depth": 7},
    {"seq_len": 16, "carry_depth": 8},
    {"seq_len": 18, "carry_depth": 9},
    {"seq_len": 20, "carry_depth": 10},
]

VARIANTS = ["fixed_t", "variable_t"]
Q_CHECKPOINT_INTERVAL = 2000

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d33_crossover_probe.json"


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
    """Measure T_MIN eval accuracy as proxy for training-time solvability."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    model.eval()
    half = seq_len // 2
    torch.manual_seed(12345)
    src, tgt = generate_batch("addition", 1024, seq_len, VOCAB_SIZE)
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


def train_model(model, device, seq_len, variant, seed):
    """Train with periodic q measurement at T_min."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0
    q_trajectory = {}
    phase_transition_step = None

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
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
            q = measure_q_at_t_min(model, device, seq_len)
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


def measure_spectral_radius(model, device, seq_len, n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 256, seq_len, VOCAB_SIZE)
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
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
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


def run_one_config(depth_cfg, variant, seed, device, completed_keys):
    carry_depth = depth_cfg["carry_depth"]
    seq_len = depth_cfg["seq_len"]
    key = f"D{carry_depth}_{variant}_s{seed}"

    if key in completed_keys:
        return None

    print(f"\n    D={carry_depth} (L={seq_len}), seed={seed}, variant={variant}",
          flush=True)

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
        "carry_depth": carry_depth,
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


def compute_crossover_analysis(results):
    """Analyze delta_rho across D=6-10 using seed-matched paired tests."""
    print("\n" + "=" * 60, flush=True)
    print("  CROSSOVER ANALYSIS (PAIRED BY SEED)", flush=True)
    print("=" * 60, flush=True)

    by_depth_seed = {}
    for run in results["runs"]:
        d = run["carry_depth"]
        s = run["seed"]
        v = run["variant"]
        key = (d, s)
        if key not in by_depth_seed:
            by_depth_seed[key] = {}
        by_depth_seed[key][v] = run

    print(f"\n  {'D':>3s} | {'rho_FT':>10s} | {'rho_VT':>10s} | {'delta_rho':>10s} | "
          f"{'t_stat':>7s} | {'p_val':>7s} | {'q_VT':>8s} | {'n':>3s}", flush=True)
    print("  " + "-" * 75, flush=True)

    crossover_data = []
    all_paired_rows = []
    for d in sorted(set(k[0] for k in by_depth_seed)):
        paired_deltas = []
        ft_rhos = []
        vt_rhos = []
        ft_q_vals = []
        vt_q_vals = []

        for seed in SEEDS:
            pair = by_depth_seed.get((d, seed), {})
            if "fixed_t" in pair and "variable_t" in pair:
                ft_rho = pair["fixed_t"]["spectral_radius"]["mean"]
                vt_rho = pair["variable_t"]["spectral_radius"]["mean"]
                delta = vt_rho - ft_rho
                paired_deltas.append(delta)
                ft_rhos.append(ft_rho)
                vt_rhos.append(vt_rho)
                ft_q = pair["fixed_t"]["mean_tmin_acc"]
                vt_q = pair["variable_t"]["mean_tmin_acc"]
                ft_q_vals.append(ft_q)
                vt_q_vals.append(vt_q)
                all_paired_rows.append({
                    "D": d, "seed": seed,
                    "rho_ft": round(ft_rho, 4), "rho_vt": round(vt_rho, 4),
                    "delta_rho": round(delta, 4),
                    "q_ft": round(ft_q, 4), "q_vt": round(vt_q, 4),
                    "delta_q": round(vt_q - ft_q, 4),
                })

        if len(paired_deltas) < 2:
            continue

        t_stat, p_val = scipy_stats.ttest_rel(vt_rhos, ft_rhos)
        mean_delta = np.mean(paired_deltas)

        n = len(paired_deltas)
        print(f"  {d:3d} | {np.mean(ft_rhos):10.4f} | {np.mean(vt_rhos):10.4f} | "
              f"{mean_delta:+10.4f} | {t_stat:7.3f} | {p_val:7.4f} | "
              f"{np.mean(vt_q_vals):8.4f} | {n:3d}", flush=True)

        crossover_data.append({
            "D": d, "mean_ft_rho": round(float(np.mean(ft_rhos)), 4),
            "mean_vt_rho": round(float(np.mean(vt_rhos)), 4),
            "delta_rho": round(float(mean_delta), 4),
            "paired_t_stat": round(float(t_stat), 3),
            "paired_p_val": round(float(p_val), 4),
            "mean_ft_q": round(float(np.mean(ft_q_vals)), 4),
            "mean_vt_q": round(float(np.mean(vt_q_vals)), 4),
            "n_seeds": n,
        })

    if len(all_paired_rows) >= 5:
        deltas = [r["delta_rho"] for r in all_paired_rows]
        q_vts = [r["q_vt"] for r in all_paired_rows]
        r_pearson, p_pearson = scipy_stats.pearsonr(deltas, q_vts)
        r_spearman, p_spearman = scipy_stats.spearmanr(deltas, q_vts)
        print(f"\n  Per-seed correlation (n={len(all_paired_rows)}):", flush=True)
        print(f"    Pearson(delta_rho, q_VT):  r={r_pearson:.3f}, p={p_pearson:.4f}",
              flush=True)
        print(f"    Spearman(delta_rho, q_VT): r={r_spearman:.3f}, p={p_spearman:.4f}",
              flush=True)
        correlation_analysis = {
            "n_pairs": len(all_paired_rows),
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 4),
            "spearman_r": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 4),
        }
    else:
        correlation_analysis = None

    depth_deltas = [c["delta_rho"] for c in crossover_data]
    if len(depth_deltas) >= 3:
        neg_count = sum(1 for d in depth_deltas if d < -0.001)
        pos_count = sum(1 for d in depth_deltas if d > 0.001)
        zero_count = len(depth_deltas) - neg_count - pos_count
        print(f"\n  Regime counts: negative={neg_count}, zero={zero_count}, "
              f"positive={pos_count}", flush=True)

        if pos_count == 0:
            print("  VERDICT: D=8 anomaly is a SEED FLUKE (no positive delta at any D)",
                  flush=True)
        elif pos_count >= 2:
            print("  VERDICT: Phase BOUNDARY confirmed (multiple depths show positive delta)",
                  flush=True)
        else:
            print("  VERDICT: AMBIGUOUS (isolated positive delta)", flush=True)

    results["crossover_analysis"] = crossover_data
    results["paired_rows"] = all_paired_rows
    if correlation_analysis:
        results["correlation_analysis"] = correlation_analysis


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
            "purpose": "D33: Paired-depth crossover probe at D=6-10. Tests whether D=8 "
                       "VT rho anomaly is a sharp phase boundary or smooth crossover. "
                       "Includes training-time q measurement for direct Prop 34 test.",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "batch_size": BATCH_SIZE, "lr": LR,
                "variable_t_range": VARIABLE_T_RANGE,
                "seeds": SEEDS,
                "q_checkpoint_interval": Q_CHECKPOINT_INTERVAL,
                "fp_steps": FP_T, "trajectory_steps": TRAJECTORY_T,
            },
            "runs": [],
        }

    total_runs = len(DEPTH_CONFIGS) * len(SEEDS) * len(VARIANTS)
    done = len(completed_keys)
    print(f"\nD33: {total_runs} total runs ({done} done)\n", flush=True)

    for depth_cfg in DEPTH_CONFIGS:
        d = depth_cfg["carry_depth"]
        print(f"\n{'=' * 60}", flush=True)
        print(f"  D={d} (L={depth_cfg['seq_len']})", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seed in SEEDS:
            for variant in VARIANTS:
                result = run_one_config(depth_cfg, variant, seed, device, completed_keys)
                if result is not None:
                    existing["runs"].append(result)
                    completed_keys.add(result["key"])
                    save_results(existing)
                    done += 1
                    print(f"\n  [{done}/{total_runs}] saved", flush=True)

    existing["completed"] = datetime.now(timezone.utc).isoformat()
    compute_crossover_analysis(existing)
    save_results(existing)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
