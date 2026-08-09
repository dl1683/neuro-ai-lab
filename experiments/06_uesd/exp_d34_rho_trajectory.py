"""
Experiment D34: Rho Trajectory During Extended Multi-Task Training

Motivated by D32 finding: VT constrains rho to ~1.003 EVEN WHEN the model
doesn't learn the task. This experiment tracks how spectral radius EVOLVES
during training to test the "VT ceiling" hypothesis.

Design:
  Single depth: D=6 (seq_len=12) — easiest multi-task config from D32
  2 variants: fixed_t, variable_t
  60,000 training steps (3x D32's 20K — enough for multi-task phase transition)
  Rho + accuracy measured every 5000 steps

Predictions (VT ceiling model):
  1. VT rho ≈ 1.003 throughout training (constant ceiling)
  2. FT rho starts high (~1.004) and drops toward 1.002-1.003 as model learns
  3. Delta (VT-FT) starts large-negative, shrinks as FT drops
  4. Phase transition for multi-task occurs at ~step 30K-40K
  5. After phase transition: FT rho ≈ VT rho ≈ 1.003 (matching D28 learned state)

Alternative prediction (learning-dependent model):
  1. VT rho decreases as model learns (VT needs learning to constrain)
  2. Both FT and VT drop together
  3. Delta stays constant throughout training
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

TRAINING_STEPS = 60000
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

CARRY_DEPTH = 6
SEQ_LEN = 12
SEED = 42

CHECKPOINT_INTERVAL = 5000
FP_T = 100
EVAL_SAMPLES = 4096

VARIANTS = ["fixed_t", "variable_t"]
TASKS = ["addition", "subtraction"]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d34_rho_trajectory.json"


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


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


@torch.no_grad()
def measure_spectral_radius(model, device, n_vecs=10, n_iter=50):
    model.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(9999)

    src, _ = generate_batch("addition", 256, SEQ_LEN, VOCAB_SIZE)
    src = src.to(device)

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
                    s_next, s_pert, grad_outputs=v, create_graph=False
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)

    return round(float(np.mean(rhos)), 4), round(float(np.std(rhos)), 4)


@torch.no_grad()
def measure_task_accuracy(model, device, task):
    model.eval()
    half = SEQ_LEN // 2
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(8888)

    src, tgt = generate_batch(task, EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    logits = model(src, TRAIN_T)
    preds = logits[:, :half].argmax(dim=-1)
    seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    return round(seq_acc, 4)


def train_with_checkpoints(model, device, variant, seed, results):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    task_schedule = []
    rng = random.Random(seed + 999)
    for _ in range(TRAINING_STEPS):
        task_schedule.append(rng.choice(TASKS))

    t_schedule = []
    if variant == "variable_t":
        rng2 = random.Random(seed + 1000)
        for _ in range(TRAINING_STEPS):
            t_schedule.append(rng2.choice(VARIABLE_T_RANGE))
    else:
        t_schedule = [TRAIN_T] * TRAINING_STEPS

    checkpoints = []

    for step in range(1, TRAINING_STEPS + 1):
        task = task_schedule[step - 1]
        t_steps = t_schedule[step - 1]

        src, tgt = generate_batch(task, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, t_steps)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % CHECKPOINT_INTERVAL == 0:
            rho_mean, rho_std = measure_spectral_radius(model, device)
            add_acc = measure_task_accuracy(model, device, "addition")
            sub_acc = measure_task_accuracy(model, device, "subtraction")

            checkpoint = {
                "step": step,
                "rho_mean": rho_mean,
                "rho_std": rho_std,
                "addition_acc": add_acc,
                "subtraction_acc": sub_acc,
                "loss": round(loss.item(), 4),
            }
            checkpoints.append(checkpoint)
            save_results(results)

            print(f"    Step {step:>6d}/{TRAINING_STEPS} | "
                  f"rho={rho_mean:.4f}±{rho_std:.4f} | "
                  f"Add: {add_acc:.4f} | Sub: {sub_acc:.4f} | "
                  f"Loss: {loss.item():.4f}", flush=True)
            model.train()

    elapsed = time.time() - t0
    return elapsed, checkpoints


def run_variant(variant, device, results):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D34: D={CARRY_DEPTH}, variant={variant}, {TRAINING_STEPS} steps", flush=True)
    print(f"{'=' * 60}", flush=True)

    full_seed(SEED)
    model = make_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params}", flush=True)

    elapsed, checkpoints = train_with_checkpoints(model, device, variant, SEED, results)

    print(f"\n    Done in {elapsed:.1f}s", flush=True)

    run_result = {
        "variant": variant,
        "seed": SEED,
        "params": n_params,
        "train_time_s": round(elapsed, 1),
        "checkpoints": checkpoints,
    }
    results["runs"].append(run_result)
    save_results(results)
    print(f"    Saved.", flush=True)


def print_analysis(results):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  RHO TRAJECTORY ANALYSIS", flush=True)
    print(f"{'=' * 60}", flush=True)

    ft_run = next((r for r in results["runs"] if r["variant"] == "fixed_t"), None)
    vt_run = next((r for r in results["runs"] if r["variant"] == "variable_t"), None)

    if not ft_run or not vt_run:
        print("  Incomplete — need both variants", flush=True)
        return

    print(f"\n  {'Step':>6s} | {'FT rho':>8s} | {'VT rho':>8s} | {'Delta':>8s} | "
          f"{'FT Add':>7s} | {'VT Add':>7s} | {'FT Sub':>7s} | {'VT Sub':>7s}", flush=True)
    print(f"  {'-' * 80}", flush=True)

    for ft_cp, vt_cp in zip(ft_run["checkpoints"], vt_run["checkpoints"]):
        delta = vt_cp["rho_mean"] - ft_cp["rho_mean"]
        print(f"  {ft_cp['step']:6d} | {ft_cp['rho_mean']:8.4f} | {vt_cp['rho_mean']:8.4f} | "
              f"{delta:+8.4f} | {ft_cp['addition_acc']:7.4f} | {vt_cp['addition_acc']:7.4f} | "
              f"{ft_cp['subtraction_acc']:7.4f} | {vt_cp['subtraction_acc']:7.4f}", flush=True)

    ft_rhos = [c["rho_mean"] for c in ft_run["checkpoints"]]
    vt_rhos = [c["rho_mean"] for c in vt_run["checkpoints"]]
    vt_std = np.std(vt_rhos)
    ft_trend = ft_rhos[-1] - ft_rhos[0]

    print(f"\n  VT rho std across time: {vt_std:.4f} ({'CONSTANT' if vt_std < 0.001 else 'VARIES'})")
    print(f"  FT rho trend (last - first): {ft_trend:+.4f} ({'DROPS' if ft_trend < -0.001 else 'STABLE' if abs(ft_trend) < 0.001 else 'RISES'})")

    if vt_std < 0.001 and ft_trend < -0.001:
        print(f"\n  VERDICT: VT CEILING MODEL CONFIRMED")
        print(f"  VT rho is constant (~{np.mean(vt_rhos):.4f}), FT rho drops as model learns")
    elif vt_std < 0.001 and abs(ft_trend) < 0.001:
        print(f"\n  VERDICT: BOTH CONSTANT (learning insufficient at 60K)")
    else:
        print(f"\n  VERDICT: COMPLEX DYNAMICS (neither simple model fits)")

    results["analysis"] = {
        "vt_rho_std": round(float(vt_std), 4),
        "ft_rho_trend": round(float(ft_trend), 4),
        "vt_rho_mean": round(float(np.mean(vt_rhos)), 4),
        "ft_rho_first": ft_rhos[0],
        "ft_rho_last": ft_rhos[-1],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D34: Track rho trajectory during extended multi-task training. "
                   "Tests VT ceiling hypothesis: does VT constrain rho independently "
                   "of task learning?",
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
            "lr": LR, "train_T": TRAIN_T,
            "variable_t_range": VARIABLE_T_RANGE,
            "carry_depth": CARRY_DEPTH, "seq_len": SEQ_LEN,
            "seed": SEED, "tasks": TASKS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "runs": [],
    }

    for variant in VARIANTS:
        run_variant(variant, device, results)

    print_analysis(results)
    results["completed"] = datetime.now(timezone.utc).isoformat()
    save_results(results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
