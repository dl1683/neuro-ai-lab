"""
Experiment D31: D=8 Multi-Seed Replication

PRIORITY DIRECTIVE from Codex review: determine whether D=8 delta_rho>0 is genuine
or a single-seed anomaly.

D28 showed constant delta_rho ~ -0.0025 at 4/5 depths (D=2,4,6,10), but D=8
reversed to delta_rho = +0.0014 -- a 20sigma outlier. This experiment runs 8 seeds
each for fixed_t and variable_t at D=8, measuring rho per seed.

Adjudication:
  - If delta_rho > 0 for most seeds -> D=8 conditional phase boundary is real
  - If delta_rho re-centers near -0.0025 -> D=8 was seed-dependent anomaly

Also runs D=6 and D=10 as 3-seed controls to verify baseline delta_rho stability.
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

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30

PRIMARY_SEEDS = [42, 137, 256, 512, 1024, 1337, 2024, 7777]
CONTROL_SEEDS = [42, 137, 256]

CONFIGS = [
    {"seq_len": 16, "carry_depth": 8, "seeds": PRIMARY_SEEDS, "role": "primary"},
    {"seq_len": 12, "carry_depth": 6, "seeds": CONTROL_SEEDS, "role": "control"},
    {"seq_len": 20, "carry_depth": 10, "seeds": CONTROL_SEEDS, "role": "control"},
]

VARIANTS = ["fixed_t", "variable_t"]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d31_d8_multiseed.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_results(results):
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


def train_model(model, device, seq_len, variant, seed):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

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

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                eval_logits = model(src, TRAIN_T)
                preds = eval_logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            t_label = f"T={t_steps}" if variant == "variable_t" else f"T={TRAIN_T}"
            print(f"      Step {step:>6d}/{TRAINING_STEPS} | {t_label} | "
                  f"Loss: {loss.item():.4f} | Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"      Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


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

    s_check = s_star.clone()
    for _ in range(10):
        s_check, _ = model.dynamics_step(s_check, context)
    diff_flat = (s_check - s_star).reshape(src.size(0), -1)
    star_flat = s_star.reshape(src.size(0), -1)
    fp_residual = diff_flat.norm(dim=-1).mean().item()
    s_star_norm = star_flat.norm(dim=-1).mean().item()
    relative_residual = fp_residual / max(s_star_norm, 1e-8)

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
        "fp_relative_residual": round(relative_residual, 8),
    }


def run_one_seed(seq_len, device, variant, seed):
    carry_depth = seq_len // 2
    print(f"\n    seed={seed}, variant={variant}", flush=True)

    full_seed(seed)
    model = make_model(device)

    train_time, best_acc = train_model(model, device, seq_len, variant, seed)

    print(f"      Measuring contraction...", flush=True)
    contraction = measure_contraction_summary(model, device, seq_len)

    print(f"      Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device, seq_len)

    print(f"      rho={spectral['mean']:.4f}+/-{spectral['std']:.4f}, "
          f"k={contraction['mean_k']:.4f}, T_99={contraction['T_99']}, "
          f"acc={best_acc:.4f}", flush=True)

    return {
        "carry_depth": carry_depth,
        "seq_len": seq_len,
        "variant": variant,
        "seed": seed,
        "accuracy": round(best_acc, 4),
        "spectral_radius": spectral,
        "contraction": contraction,
        "train_time_s": round(train_time, 1),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    existing = load_checkpoint()
    completed_keys = set()
    if existing and "runs" in existing:
        for run in existing["runs"]:
            key = f"D{run['carry_depth']}_{run['variant']}_s{run['seed']}"
            completed_keys.add(key)
        print(f"Resuming: {len(completed_keys)} runs complete")
    else:
        existing = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D31: Multi-seed D=8 replication -- is delta_rho>0 at D=8 real or seed anomaly? "
                       "8 seeds x 2 variants at D=8, plus 3-seed controls at D=6,D=10.",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "variable_t_range": VARIABLE_T_RANGE,
                "batch_size": BATCH_SIZE, "lr": LR,
                "primary_seeds": PRIMARY_SEEDS,
                "control_seeds": CONTROL_SEEDS,
            },
            "runs": [],
        }

    total_configs = sum(
        len(c["seeds"]) * len(VARIANTS) for c in CONFIGS
    )
    print(f"\nD31: {total_configs} total runs "
          f"({len(completed_keys)} done)", flush=True)

    for cfg in CONFIGS:
        seq_len = cfg["seq_len"]
        carry_depth = cfg["carry_depth"]
        seeds = cfg["seeds"]
        role = cfg["role"]

        print(f"\n{'='*60}", flush=True)
        print(f"  D={carry_depth} ({role}) -- {len(seeds)} seeds x 2 variants", flush=True)
        print(f"{'='*60}", flush=True)

        for variant in VARIANTS:
            for seed in seeds:
                key = f"D{carry_depth}_{variant}_s{seed}"
                if key in completed_keys:
                    print(f"    {key} already complete, skipping", flush=True)
                    continue

                result = run_one_seed(seq_len, device, variant, seed)
                result["role"] = role
                existing["runs"].append(result)
                save_results(existing)

    print(f"\n{'='*60}", flush=True)
    print(f"  D31 MULTI-SEED SUMMARY", flush=True)
    print(f"{'='*60}\n", flush=True)

    for cfg in CONFIGS:
        d = cfg["carry_depth"]
        role = cfg["role"]
        runs_d = [r for r in existing["runs"] if r["carry_depth"] == d]

        ft_runs = [r for r in runs_d if r["variant"] == "fixed_t"]
        vt_runs = [r for r in runs_d if r["variant"] == "variable_t"]

        if not ft_runs or not vt_runs:
            continue

        ft_rhos = [r["spectral_radius"]["mean"] for r in ft_runs]
        vt_rhos = [r["spectral_radius"]["mean"] for r in vt_runs]

        ft_by_seed = {r["seed"]: r["spectral_radius"]["mean"] for r in ft_runs}
        vt_by_seed = {r["seed"]: r["spectral_radius"]["mean"] for r in vt_runs}

        common_seeds = sorted(set(ft_by_seed.keys()) & set(vt_by_seed.keys()))
        deltas = [vt_by_seed[s] - ft_by_seed[s] for s in common_seeds]

        print(f"  D={d} ({role}) -- {len(common_seeds)} paired seeds:")
        print(f"    {'seed':>6s}  {'rho_FT':>8s}  {'rho_VT':>8s}  {'delta_rho':>8s}")
        print(f"    {'----':>6s}  {'------':>8s}  {'------':>8s}  {'--':>8s}")
        for s in common_seeds:
            dr = vt_by_seed[s] - ft_by_seed[s]
            print(f"    {s:>6d}  {ft_by_seed[s]:>8.4f}  {vt_by_seed[s]:>8.4f}  {dr:>+8.4f}")

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        n = len(deltas)
        se = std_delta / np.sqrt(n)
        t_stat = mean_delta / max(se, 1e-12)
        p_two = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))

        print(f"    Mean delta_rho = {mean_delta:+.4f} +/- {std_delta:.4f} "
              f"(SE={se:.4f}, t={t_stat:.2f}, p={p_two:.4f}, n={n})")

        if p_two < 0.05 and mean_delta > 0:
            print(f"    >>> D={d} delta_rho > 0 SIGNIFICANT (p={p_two:.4f}) -- "
                  f"PHASE BOUNDARY REAL <<<")
        elif p_two < 0.05 and mean_delta < 0:
            print(f"    >>> D={d} delta_rho < 0 SIGNIFICANT (p={p_two:.4f}) -- "
                  f"CONSTANT delta_rho HOLDS <<<")
        else:
            print(f"    >>> D={d} NOT SIGNIFICANT (p={p_two:.4f}) -- "
                  f"INCONCLUSIVE <<<")
        print()

    d8_ft = [r for r in existing["runs"]
             if r["carry_depth"] == 8 and r["variant"] == "fixed_t"]
    d8_vt = [r for r in existing["runs"]
             if r["carry_depth"] == 8 and r["variant"] == "variable_t"]

    if d8_ft and d8_vt:
        ft_by_seed = {r["seed"]: r["spectral_radius"]["mean"] for r in d8_ft}
        vt_by_seed = {r["seed"]: r["spectral_radius"]["mean"] for r in d8_vt}
        common = sorted(set(ft_by_seed.keys()) & set(vt_by_seed.keys()))
        deltas = [vt_by_seed[s] - ft_by_seed[s] for s in common]
        pos_count = sum(1 for d in deltas if d > 0)
        neg_count = sum(1 for d in deltas if d <= 0)

        std_d = float(np.std(deltas, ddof=1))
        se_d = std_d / np.sqrt(len(common))
        t_d = float(np.mean(deltas)) / max(se_d, 1e-12)
        p_d = 2 * (1 - scipy_stats.t.cdf(abs(t_d), df=len(common) - 1))

        if p_d < 0.05 and float(np.mean(deltas)) > 0:
            verdict = "PHASE_BOUNDARY_REAL"
        elif p_d < 0.05 and float(np.mean(deltas)) < 0:
            verdict = "SEED_ANOMALY"
        else:
            verdict = "INCONCLUSIVE"

        existing["d8_adjudication"] = {
            "n_seeds": len(common),
            "mean_delta_rho": round(float(np.mean(deltas)), 6),
            "std_delta_rho": round(std_d, 6),
            "se_delta_rho": round(se_d, 6),
            "t_statistic": round(t_d, 3),
            "p_value": round(p_d, 6),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "verdict": verdict,
        }
        save_results(existing)

        print(f"  ADJUDICATION: {existing['d8_adjudication']['verdict']}")
        print(f"    {pos_count}/{len(common)} seeds have delta_rho > 0")
        print(f"    Mean delta_rho = {existing['d8_adjudication']['mean_delta_rho']:+.6f}")

    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
