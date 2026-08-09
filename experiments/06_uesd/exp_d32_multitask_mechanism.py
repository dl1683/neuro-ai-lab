"""
Experiment D32: Multi-Task Mechanism Test (D31-M2T from Codex synthesis)

PARADIGM-SHIFTING EXPERIMENT: Tests whether the readout-stable manifold +
non-normal orientation mechanism is causal and task-general, not an artifact
of addition-only single-task training.

Three arms:
  Arm A (baseline): Standard UESD with multi-task training (add + sub, 50/50)
  Arm B (no-layernorm): Remove all LayerNorm from dynamics block
  Arm C (iter-dropout): Randomly skip dynamics steps with p=0.2

Each arm trains at D={6,8,10,12} (seq_len={12,16,20,24}) with both FT and VT.
Total: 3 arms x 4 depths x 2 variants = 24 runs.

Predictions (from synthesis):
  - If mechanism is real and general:
    * Arm A preserves D28/D30 rho(D) profile across both tasks
    * VT shift remains ~-0.0025
    * T_99 = max(T_min, D_intrinsic)
  - If LayerNorm is causal stabilizer:
    * Arm B shows rho < 1 or unstable readout (higher WA errors)
  - If iterations are redundant error-correction:
    * Arm C degrades gracefully (not catastrophically)

Measurements per run: rho, k, T_99, accuracy on BOTH tasks, per-step
convergence curve, wrong-attractor rate.
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
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
SEED = 42

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30
ITER_DROPOUT_P = 0.2

DEPTH_CONFIGS = [
    {"seq_len": 12, "carry_depth": 6},
    {"seq_len": 16, "carry_depth": 8},
    {"seq_len": 20, "carry_depth": 10},
    {"seq_len": 24, "carry_depth": 12},
]

ARMS = ["baseline", "no_layernorm", "iter_dropout"]
VARIANTS = ["fixed_t", "variable_t"]
TASKS = ["addition", "subtraction"]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d32_multitask_mechanism.json"


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


def make_model(device, arm="baseline"):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)

    if arm == "no_layernorm":
        model.dynamics.norm1 = nn.Identity()
        model.dynamics.norm2 = nn.Identity()
        model.dynamics.norm3 = nn.Identity()

    return model.to(device)


def _predraw_schedules(seed, n_steps, variable_t_range, t_steps_fixed, dropout_p):
    """Pre-draw all stochastic schedules from a dedicated RNG so all arms
    see identical task/T sequences regardless of branch-specific RNG consumption."""
    rng = random.Random(seed)
    task_schedule = [rng.choice(TASKS) for _ in range(n_steps)]
    t_schedule = [rng.choice(variable_t_range) for _ in range(n_steps)]
    max_t = max(max(variable_t_range), t_steps_fixed)
    dropout_masks = []
    for _ in range(n_steps):
        dropout_masks.append([rng.random() < dropout_p for _ in range(max_t)])
    return task_schedule, t_schedule, dropout_masks


def train_multitask(model, device, seq_len, variant, arm):
    full_seed(SEED)
    task_schedule, vt_schedule, dropout_masks = _predraw_schedules(
        SEED + 7, TRAINING_STEPS, VARIABLE_T_RANGE, TRAIN_T, ITER_DROPOUT_P
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_add_acc = 0.0
    best_sub_acc = 0.0
    total_executed_steps = 0
    total_requested_steps = 0

    for step in range(1, TRAINING_STEPS + 1):
        task = task_schedule[step - 1]
        src, tgt = generate_batch(task, BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if variant == "variable_t":
            t_steps = vt_schedule[step - 1]
        else:
            t_steps = TRAIN_T

        total_requested_steps += t_steps

        if arm == "iter_dropout" and model.training:
            context = model.encode(src)
            B, L_out = src.shape
            s = model.init_state(B, L_out, device)
            mask = dropout_masks[step - 1]
            executed = 0
            for i in range(t_steps):
                if i < len(mask) and mask[i]:
                    continue
                s, _ = model.dynamics_step(s, context)
                executed += 1
            total_executed_steps += executed
            logits = model.readout_logits(s)
        else:
            logits = model(src, t_steps)
            total_executed_steps += t_steps

        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                add_src, add_tgt = generate_batch("addition", 512, seq_len, VOCAB_SIZE)
                add_src, add_tgt = add_src.to(device), add_tgt.to(device)
                add_logits = model(add_src, TRAIN_T)
                add_preds = add_logits[:, :half].argmax(dim=-1)
                add_acc = (add_preds == add_tgt[:, :half]).all(dim=1).float().mean().item()

                sub_src, sub_tgt = generate_batch("subtraction", 512, seq_len, VOCAB_SIZE)
                sub_src, sub_tgt = sub_src.to(device), sub_tgt.to(device)
                sub_logits = model(sub_src, TRAIN_T)
                sub_preds = sub_logits[:, :half].argmax(dim=-1)
                sub_acc = (sub_preds == sub_tgt[:, :half]).all(dim=1).float().mean().item()

                best_add_acc = max(best_add_acc, add_acc)
                best_sub_acc = max(best_sub_acc, sub_acc)

            t_label = f"T={t_steps}" if variant == "variable_t" else f"T={TRAIN_T}"
            print(f"      Step {step:>6d}/{TRAINING_STEPS} | {t_label} | "
                  f"Loss: {loss.item():.4f} | Add: {add_acc:.4f} | Sub: {sub_acc:.4f}",
                  flush=True)
            model.train()

    elapsed = time.time() - t0
    effective_t_ratio = total_executed_steps / max(total_requested_steps, 1)
    print(f"      Training done in {elapsed:.1f}s | Best add: {best_add_acc:.4f} | "
          f"Best sub: {best_sub_acc:.4f} | Effective T ratio: {effective_t_ratio:.3f}",
          flush=True)
    return elapsed, best_add_acc, best_sub_acc, effective_t_ratio


def measure_spectral_radius(model, device, seq_len, task="addition",
                            n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch(task, 256, seq_len, VOCAB_SIZE)
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
                    s_next, s_pert, grad_outputs=v, create_graph=False,
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    return {"mean": round(float(np.mean(rhos)), 4),
            "std": round(float(np.std(rhos)), 4)}


@torch.no_grad()
def measure_contraction_and_convergence(model, device, seq_len, task="addition"):
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch(task, EVAL_SAMPLES, seq_len, VOCAB_SIZE)
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
    step_accuracies = []
    tgt_result = tgt[:, :half]
    actual_T99 = None

    for t in range(1, TRAJECTORY_T + 1):
        s, _ = model.dynamics_step(s, context)
        curr_dist = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)
        k_per = (curr_dist / prev_dist.clamp(min=1e-8)).mean().item()
        k_values.append(k_per)
        prev_dist = curr_dist

        logits = model.readout_logits(s)
        preds = logits[:, :half].argmax(dim=-1)
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        step_accuracies.append(round(seq_acc, 4))
        if seq_acc >= 0.99 and actual_T99 is None:
            actual_T99 = t

    stable_k = k_values[1:10]
    wa_count = 0
    logits_final = model.readout_logits(s_star)
    preds_fp = logits_final[:, :half].argmax(dim=-1)
    correct_fp = (preds_fp == tgt_result).all(dim=1)
    wa_count = (~correct_fp).sum().item()

    return {
        "mean_k": round(float(np.mean(stable_k)), 4),
        "std_k": round(float(np.std(stable_k)), 4),
        "T_99": actual_T99,
        "fp_relative_residual": round(relative_residual, 8),
        "step_accuracies": step_accuracies[:15],
        "wrong_attractor_rate": round(wa_count / src.size(0), 4),
        "final_acc": step_accuracies[-1] if step_accuracies else 0.0,
    }


def run_one_config(depth_cfg, arm, variant, device, existing_runs):
    seq_len = depth_cfg["seq_len"]
    carry_depth = depth_cfg["carry_depth"]
    key = f"D{carry_depth}_{arm}_{variant}"

    if key in existing_runs:
        print(f"\n  {key} already complete, skipping", flush=True)
        return None

    print(f"\n{'='*60}", flush=True)
    print(f"  D={carry_depth} (L={seq_len}), arm={arm}, variant={variant}", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(SEED)
    model = make_model(device, arm=arm)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count}", flush=True)

    train_time, best_add, best_sub, eff_t_ratio = train_multitask(
        model, device, seq_len, variant, arm
    )

    results_per_task = {}
    for task in TASKS:
        print(f"    Measuring {task}...", flush=True)
        rho = measure_spectral_radius(model, device, seq_len, task=task)
        contraction = measure_contraction_and_convergence(
            model, device, seq_len, task=task
        )
        results_per_task[task] = {
            "spectral_radius": rho,
            **contraction,
        }
        print(f"      {task}: rho={rho['mean']:.4f}+/-{rho['std']:.4f}, "
              f"k={contraction['mean_k']:.4f}, T_99={contraction['T_99']}, "
              f"WA={contraction['wrong_attractor_rate']:.4f}", flush=True)

    run_result = {
        "key": key,
        "carry_depth": carry_depth,
        "seq_len": seq_len,
        "arm": arm,
        "variant": variant,
        "seed": SEED,
        "params": param_count,
        "train_time_s": round(train_time, 1),
        "best_addition_acc": round(best_add, 4),
        "best_subtraction_acc": round(best_sub, 4),
        "effective_t_ratio": round(eff_t_ratio, 4),
        "per_task": results_per_task,
        "run_signature": {
            "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE,
            "train_T": TRAIN_T,
            "variable_t_range": VARIABLE_T_RANGE,
            "iter_dropout_p": ITER_DROPOUT_P if arm == "iter_dropout" else 0.0,
            "tasks": TASKS,
        },
    }

    return run_result


def compute_d28_comparison(results):
    d28_ft_rho = {6: 1.0024, 8: 1.0016, 10: 1.0042, 12: 1.0039}
    d28_vt_rho = {6: 0.9996, 8: 1.0030, 10: 1.0017}

    print("\n" + "="*60, flush=True)
    print("  D28 COMPARISON (single-task addition vs multi-task)", flush=True)
    print("="*60, flush=True)

    for run in results["runs"]:
        if run["arm"] != "baseline":
            continue
        d = run["carry_depth"]
        v = run["variant"]
        add_rho = run["per_task"]["addition"]["spectral_radius"]["mean"]

        if v == "fixed_t" and d in d28_ft_rho:
            d28_val = d28_ft_rho[d]
            delta = add_rho - d28_val
            print(f"  D={d} FT: D32={add_rho:.4f} vs D28={d28_val:.4f} "
                  f"(delta={delta:+.4f})", flush=True)
        elif v == "variable_t" and d in d28_vt_rho:
            d28_val = d28_vt_rho[d]
            delta = add_rho - d28_val
            print(f"  D={d} VT: D32={add_rho:.4f} vs D28={d28_val:.4f} "
                  f"(delta={delta:+.4f})", flush=True)


def compute_arm_comparison(results):
    print("\n" + "="*60, flush=True)
    print("  ARM COMPARISON (mechanism causal test)", flush=True)
    print("="*60, flush=True)

    by_depth_variant = {}
    for run in results["runs"]:
        dv = f"D{run['carry_depth']}_{run['variant']}"
        if dv not in by_depth_variant:
            by_depth_variant[dv] = {}
        by_depth_variant[dv][run["arm"]] = run

    for dv in sorted(by_depth_variant.keys()):
        arms = by_depth_variant[dv]
        print(f"\n  {dv}:", flush=True)
        for arm_name in ARMS:
            if arm_name not in arms:
                continue
            r = arms[arm_name]
            add_rho = r["per_task"]["addition"]["spectral_radius"]["mean"]
            sub_rho = r["per_task"]["subtraction"]["spectral_radius"]["mean"]
            add_t99 = r["per_task"]["addition"]["T_99"]
            sub_t99 = r["per_task"]["subtraction"]["T_99"]
            add_acc = r["best_addition_acc"]
            sub_acc = r["best_subtraction_acc"]
            print(f"    {arm_name:15s}: add_rho={add_rho:.4f} sub_rho={sub_rho:.4f} "
                  f"add_T99={add_t99} sub_T99={sub_t99} "
                  f"add_acc={add_acc:.4f} sub_acc={sub_acc:.4f}", flush=True)


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
            "purpose": "D32: Multi-task mechanism test. Tests whether readout-stable manifold "
                       "mechanism generalizes beyond single-task addition. 3 arms "
                       "(baseline, no-layernorm, iter-dropout) x 4 depths x 2 variants.",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "variable_t_range": VARIABLE_T_RANGE, "seed": SEED,
                "iter_dropout_p": ITER_DROPOUT_P,
                "tasks": TASKS,
            },
            "runs": [],
        }

    total_runs = len(DEPTH_CONFIGS) * len(ARMS) * len(VARIANTS)
    done = len(completed_keys)
    print(f"\nD32: {total_runs} total runs ({done} done)\n", flush=True)

    for depth_cfg in DEPTH_CONFIGS:
        for arm in ARMS:
            for variant in VARIANTS:
                result = run_one_config(
                    depth_cfg, arm, variant, device, completed_keys
                )
                if result is not None:
                    existing["runs"].append(result)
                    completed_keys.add(result["key"])
                    save_results(existing)
                    done += 1
                    print(f"\n  [{done}/{total_runs}] saved", flush=True)

    existing["completed"] = datetime.now(timezone.utc).isoformat()

    compute_d28_comparison(existing)
    compute_arm_comparison(existing)

    save_results(existing)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
