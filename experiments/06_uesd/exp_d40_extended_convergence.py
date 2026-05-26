"""
Experiment D40: Extended Convergence Analysis (No Flow)

D38/D39 showed:
1. Flow correction fails fundamentally (cosine readout sensitivity + space
   misalignment: flow operates in pre-projection space, readout measures in
   post-projection normalized cosine space)
2. SC lambda=0.1 reduces residual (0.31->0.13) but no convergence at T=10
3. k~0.97 means convergence requires ~150 iterations, not 10

KEY INSIGHT: Evaluating convergence at T=10 is unfair for k=0.97.
Theory predicts convergence in O(ln(d0/eps)/(1-k)) iterations.
For d0=0.13, eps=0.01, k=0.97: T ~ 84 iterations needed.

THIS EXPERIMENT:
- Drop flow entirely (proven broken in D38 and D39)
- Multi-T evaluation: T=[10, 25, 50, 100, 200]
- Test whether UESD actually converges when given enough iterations
- Track accuracy at each T to verify answer-preserving convergence
- Log per-example residual distribution for convergence landscape analysis

TRAINING: 3-phase (no flow)
  Phase A: CE-only warmstart (15k steps, variable T in [4..16])
  Phase B: CE + margin-gated SC with warmup (15k steps)
  Phase C: CE + SC + recovery (10k steps)

SWEEP: lambda_sc in [0.0, 0.5, 1.0, 3.0]
  0.0 = CE-only ablation (does the system converge at high T without SC?)
SEEDS: [42, 137, 256, 512]

SUCCESS CRITERIA:
  converged_frac > 80% at some T <= 200
  accuracy maintained >= 99% at all evaluated T values
  WA rate <= 1% among converged examples
  OR: clear evidence convergence doesn't happen (plateauing residual)
"""
import json
import math
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
from shared.training import set_seed, count_params
from shared.data import generate_batch

# -- Architecture --------------------------------------------------------
VOCAB_SIZE = 64
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64

# -- Task -----------------------------------------------------------------
TASK = "addition"
SEQ_LEN = 16
HALF = SEQ_LEN // 2

# -- Training -------------------------------------------------------------
BATCH_SIZE = 256
LR = 3e-4
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
LOG_INTERVAL = 2000

PHASE_A_STEPS = 15000
PHASE_B_STEPS = 15000
PHASE_C_STEPS = 10000

# -- Loss weights ---------------------------------------------------------
LAMBDA_SC_VALUES = [0.0, 0.5, 1.0, 3.0]
LAMBDA_MARGIN = 0.1
LAMBDA_REC = 0.2
SC_WARMUP = 3000
GAMMA = 2.0
REC_SIGMA = 0.1
REC_EXTRA_STEPS = 3

# -- Evaluation -----------------------------------------------------------
EVAL_SAMPLES = 4096
FP_STEPS = 50
K_TRAJECTORY = 15
EVAL_T_VALUES = [10, 25, 50, 100, 200]
TRAJECTORY_SAMPLES = 256
TRAJECTORY_MAX_T = 200

# -- Seeds ----------------------------------------------------------------
SEEDS = [42, 137, 256, 512]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIR / "exp_d40_extended_convergence.json"


# -- Helpers --------------------------------------------------------------

def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_model(device):
    m = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return m.to(device)


def compute_margin(logits, target_ids):
    correct = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, target_ids.unsqueeze(-1), True)
    wrong = logits.masked_fill(mask, float("-inf"))
    max_wrong = wrong.max(dim=-1).values
    return correct - max_wrong


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


# -- Forward helpers ------------------------------------------------------

def forward_dynamics(model, src, device, t_steps=None):
    if t_steps is None:
        t_steps = random.choice(VARIABLE_T_RANGE)
    context = model.encode(src)
    B = src.size(0)
    s = model.init_state(B, SEQ_LEN, device)
    for _ in range(t_steps):
        s, _ = model.dynamics_step(s, context)
    return s, context, t_steps


def _ce_result_only(logits, tgt):
    lo = logits[:, :HALF].reshape(-1, logits.size(-1))
    ta = tgt[:, :HALF].reshape(-1)
    return F.cross_entropy(lo, ta)


def margin_gated_sc(model, h, context, logits, tgt):
    was_training = model.dynamics.training
    model.dynamics.eval()
    s_next, _ = model.dynamics_step(h, context)
    if was_training:
        model.dynamics.train()
    residual_sq = (s_next - h) ** 2
    residual_per_pos = residual_sq.mean(dim=-1)

    margin = compute_margin(logits[:, :HALF], tgt[:, :HALF])
    gate = (margin > 0).float()
    gate_frac = gate.mean().item()

    sc_result = residual_per_pos[:, :HALF]
    n_gated = gate.sum().clamp(min=1.0)
    sc_loss = (sc_result * gate).sum() / n_gated

    mg_loss = F.relu(GAMMA - margin).mean()
    return sc_loss, mg_loss, gate_frac


# -- Phase step functions -------------------------------------------------

def phase_a_step(model, src, tgt, device):
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)
    return ce, {"ce": ce.item(), "T": t_steps}


def phase_b_step(model, src, tgt, device, step_in_phase, lambda_sc):
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)

    sc, mg, gate_frac = margin_gated_sc(model, h, ctx, logits, tgt)

    eff_sc = lambda_sc * min(1.0, step_in_phase / SC_WARMUP)
    loss = ce + eff_sc * sc + LAMBDA_MARGIN * mg
    return loss, {
        "ce": ce.item(), "sc": sc.item(),
        "margin_loss": mg.item(), "eff_sc": eff_sc,
        "gate_frac": gate_frac, "T": t_steps,
    }


def phase_c_step(model, src, tgt, device, step_in_phase, lambda_sc):
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)

    sc, mg, gate_frac = margin_gated_sc(model, h, ctx, logits, tgt)

    noise = REC_SIGMA * torch.randn_like(h)
    s_rec = h.detach() + noise
    ctx_det = ctx.detach()
    was_training = model.dynamics.training
    model.dynamics.eval()
    for _ in range(REC_EXTRA_STEPS):
        s_rec, _ = model.dynamics_step(s_rec, ctx_det)
    if was_training:
        model.dynamics.train()
    logits_rec = model.readout_logits(s_rec)
    rec = _ce_result_only(logits_rec, tgt)

    loss = ce + lambda_sc * sc + LAMBDA_MARGIN * mg + LAMBDA_REC * rec
    return loss, {
        "ce": ce.item(), "sc": sc.item(),
        "margin_loss": mg.item(), "rec": rec.item(),
        "gate_frac": gate_frac, "T": t_steps,
    }


# -- Evaluation -----------------------------------------------------------

@torch.no_grad()
def measure_k(model, src, device):
    context = model.encode(src)
    B = src.size(0)
    s_star = model.init_state(B, SEQ_LEN, device)
    for _ in range(FP_STEPS):
        s_star, _ = model.dynamics_step(s_star, context)

    s = model.init_state(B, SEQ_LEN, device)
    prev_dist = (s - s_star).norm(dim=(-2, -1))
    k_vals = []
    for _ in range(K_TRAJECTORY):
        s, _ = model.dynamics_step(s, context)
        dist = (s - s_star).norm(dim=(-2, -1))
        ratios = dist / prev_dist.clamp(min=1e-10)
        k_vals.append(ratios.mean().item())
        prev_dist = dist
    return sum(k_vals[1:10]) / min(9, len(k_vals) - 1)


@torch.no_grad()
def evaluate_at_T(model, device, T_eval):
    """Evaluate model at a specific T value. Returns metrics dict."""
    model.eval()

    total_h_correct = 0
    total_h_tok_correct = 0
    total_converged = 0
    total_wa = 0
    total_margin_sum = 0.0
    total_margin_pos = 0
    total_residual_sum = 0.0
    total_positions = 0
    n_total = 0
    residual_list = []

    remaining = EVAL_SAMPLES
    while remaining > 0:
        bs = min(512, remaining)
        src, tgt = generate_batch(TASK, bs, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(bs, SEQ_LEN, device)
        for _ in range(T_eval):
            s, _ = model.dynamics_step(s, context)
        h = s

        logits_h = model.readout_logits(h)
        preds_h = logits_h[:, :HALF].argmax(dim=-1)
        seq_correct = (preds_h == tgt[:, :HALF]).all(dim=-1)
        total_h_correct += seq_correct.sum().item()
        total_h_tok_correct += (preds_h == tgt[:, :HALF]).sum().item()

        s_next, _ = model.dynamics_step(h, context)
        diff = s_next - h
        norm_r = diff.norm(dim=(-2, -1)) / math.sqrt(SEQ_LEN * D_MODEL)
        total_residual_sum += norm_r.sum().item()
        residual_list.extend(norm_r.cpu().tolist())

        margin = compute_margin(logits_h[:, :HALF], tgt[:, :HALF])
        total_margin_sum += margin.sum().item()
        total_margin_pos += (margin > 0).sum().item()
        total_positions += margin.numel()

        converged = norm_r < 0.01
        total_converged += converged.sum().item()
        total_wa += (converged & ~seq_correct).sum().item()

        n_total += bs
        remaining -= bs

    conv_frac = total_converged / n_total if n_total > 0 else 0.0
    wa_rate = total_wa / total_converged if total_converged > 0 else 0.0

    residuals = sorted(residual_list)
    n = len(residuals)

    return {
        "T": T_eval,
        "h_seq_acc": total_h_correct / n_total,
        "h_tok_acc": total_h_tok_correct / (n_total * HALF),
        "residual_mean": total_residual_sum / n_total,
        "residual_median": residuals[n // 2],
        "residual_p10": residuals[n // 10],
        "residual_p90": residuals[9 * n // 10],
        "residual_min": residuals[0],
        "residual_max": residuals[-1],
        "margin_mean": total_margin_sum / total_positions,
        "margin_frac_positive": total_margin_pos / total_positions,
        "converged_frac": conv_frac,
        "wrong_attractor_rate": wa_rate,
    }


@torch.no_grad()
def convergence_trajectory(model, device):
    """Track residual and accuracy step-by-step from T=1 to TRAJECTORY_MAX_T.

    Uses a smaller sample (TRAJECTORY_SAMPLES) for efficiency.
    Returns list of {T, residual_mean, residual_median, seq_acc, tok_acc}.
    """
    model.eval()
    src, tgt = generate_batch(TASK, TRAJECTORY_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    context = model.encode(src)
    s = model.init_state(TRAJECTORY_SAMPLES, SEQ_LEN, device)

    trajectory = []
    log_points = set([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50,
                      75, 100, 125, 150, 175, 200])

    for t_step in range(1, TRAJECTORY_MAX_T + 1):
        s, _ = model.dynamics_step(s, context)

        if t_step in log_points:
            s_next, _ = model.dynamics_step(s, context)
            diff = s_next - s
            norm_r = diff.norm(dim=(-2, -1)) / math.sqrt(SEQ_LEN * D_MODEL)

            logits = model.readout_logits(s)
            preds = logits[:, :HALF].argmax(dim=-1)
            seq_correct = (preds == tgt[:, :HALF]).all(dim=-1)
            tok_correct = (preds == tgt[:, :HALF])

            residuals = sorted(norm_r.cpu().tolist())
            n = len(residuals)

            trajectory.append({
                "T": t_step,
                "residual_mean": norm_r.mean().item(),
                "residual_median": residuals[n // 2],
                "residual_p10": residuals[n // 10],
                "residual_p90": residuals[9 * n // 10],
                "seq_acc": seq_correct.float().mean().item(),
                "tok_acc": tok_correct.float().mean().item(),
                "converged_frac": (norm_r < 0.01).float().mean().item(),
            })

    return trajectory


@torch.no_grad()
def full_evaluation(model, device):
    """Run multi-T evaluation + k measurement + convergence trajectory."""
    model.eval()

    multi_t_results = {}
    for T_eval in EVAL_T_VALUES:
        print(f"    Evaluating T={T_eval}...", end="", flush=True)
        t0 = time.time()
        result = evaluate_at_T(model, device, T_eval)
        elapsed = time.time() - t0
        multi_t_results[str(T_eval)] = result
        print(f" acc={result['h_seq_acc']:.4f} "
              f"res={result['residual_mean']:.6f} "
              f"conv={result['converged_frac']:.4f} "
              f"({elapsed:.1f}s)", flush=True)

    src_k, _ = generate_batch(TASK, 64, SEQ_LEN, VOCAB_SIZE)
    src_k = src_k.to(device)
    k_mean = measure_k(model, src_k, device)

    print(f"    Computing convergence trajectory...", end="", flush=True)
    t0 = time.time()
    trajectory = convergence_trajectory(model, device)
    print(f" ({time.time() - t0:.1f}s)", flush=True)

    model.train()
    return {
        "multi_T": multi_t_results,
        "k_mean": k_mean,
        "trajectory": trajectory,
    }


def print_eval_summary(phase, ev):
    print(f"\n  Phase {phase} evaluation:")
    print(f"    k: {ev['k_mean']:.4f}")
    print(f"    Multi-T results:")
    for T_str, r in ev["multi_T"].items():
        conv_str = f"conv={r['converged_frac']:.4f}"
        if r["converged_frac"] > 0 and r["wrong_attractor_rate"] > 0:
            conv_str += f" WA={r['wrong_attractor_rate']:.4f}"
        print(f"      T={T_str:>3s}: acc={r['h_seq_acc']:.4f} "
              f"res={r['residual_mean']:.6f} "
              f"(p10={r['residual_p10']:.6f} p90={r['residual_p90']:.6f}) "
              f"{conv_str}")
    traj = ev["trajectory"]
    print(f"    Trajectory (sample):")
    for pt in traj:
        if pt["T"] in [1, 10, 50, 100, 200]:
            print(f"      T={pt['T']:>3d}: res={pt['residual_mean']:.6f} "
                  f"acc={pt['seq_acc']:.4f} conv={pt['converged_frac']:.4f}")
    print(flush=True)


# -- Training orchestration -----------------------------------------------

def train_phase(model, optimizer, phase_fn, n_steps, phase_name, device,
                extra_kwargs=None):
    t0 = time.time()
    history = []
    for step in range(1, n_steps + 1):
        src, tgt = generate_batch(TASK, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        kwargs = extra_kwargs(step) if callable(extra_kwargs) else {}
        loss, info = phase_fn(model, src, tgt, device, **kwargs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0 or step == 1:
            parts = [f"  [{phase_name}] {step:>6d}/{n_steps} | L={loss.item():.4f}"]
            for k, v in info.items():
                if k != "T":
                    parts.append(f"{k}={v:.4f}")
            print(" | ".join(parts), flush=True)
            history.append({"step": step, "loss": loss.item(), **info})

    elapsed = time.time() - t0
    print(f"  {phase_name} done in {elapsed:.1f}s", flush=True)
    return {"history": history, "elapsed_s": elapsed}


def run_single(seed, lambda_sc, device):
    full_seed(seed)
    model = make_model(device)
    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"Seed {seed} | lambda_sc={lambda_sc} | {n_params:,} params | device={device}")
    print(f"{'='*60}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    run = {"seed": seed, "lambda_sc": lambda_sc, "n_params": n_params, "phases": {}}

    # Phase A: CE-only
    print("\n--- Phase A: CE-only ---", flush=True)
    run["phases"]["A"] = train_phase(
        model, optimizer, phase_a_step, PHASE_A_STEPS, "A", device,
    )
    print("\n  Phase A eval:", flush=True)
    run["phases"]["A"]["eval"] = full_evaluation(model, device)
    print_eval_summary("A", run["phases"]["A"]["eval"])

    # Phase B: CE + margin-gated SC
    print("\n--- Phase B: CE + Margin-Gated SC ---", flush=True)
    run["phases"]["B"] = train_phase(
        model, optimizer, phase_b_step, PHASE_B_STEPS, "B", device,
        extra_kwargs=lambda step: {"step_in_phase": step, "lambda_sc": lambda_sc},
    )
    print("\n  Phase B eval:", flush=True)
    run["phases"]["B"]["eval"] = full_evaluation(model, device)
    print_eval_summary("B", run["phases"]["B"]["eval"])

    # Phase C: CE + SC + recovery
    print("\n--- Phase C: CE + SC + Recovery ---", flush=True)
    run["phases"]["C"] = train_phase(
        model, optimizer, phase_c_step, PHASE_C_STEPS, "C", device,
        extra_kwargs=lambda step: {"step_in_phase": step, "lambda_sc": lambda_sc},
    )
    print("\n  Phase C eval:", flush=True)
    run["phases"]["C"]["eval"] = full_evaluation(model, device)
    print_eval_summary("C", run["phases"]["C"]["eval"])

    # Save checkpoint
    ckpt_dir = RESULTS_DIR / "d40_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"d40_s{seed}_lsc{lambda_sc}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "lambda_sc": lambda_sc,
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path.name}", flush=True)

    return run


# -- Main -----------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D40 Extended Convergence Analysis (No Flow)")
    print(f"Device: {device}")
    print(f"SC lambda sweep: {LAMBDA_SC_VALUES}")
    print(f"Eval T values: {EVAL_T_VALUES}")
    print(f"Seeds: {SEEDS}", flush=True)

    existing = load_checkpoint()
    completed = set()
    if existing and "runs" in existing:
        completed = {(r["seed"], r["lambda_sc"]) for r in existing["runs"]}

    results = existing or {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Extended convergence analysis: multi-T eval, no flow",
        "config": {
            "vocab_size": VOCAB_SIZE, "d_model": D_MODEL,
            "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "seq_len": SEQ_LEN,
            "phases": {
                "A": PHASE_A_STEPS, "B": PHASE_B_STEPS, "C": PHASE_C_STEPS,
            },
            "lambda_sc_values": LAMBDA_SC_VALUES,
            "lambda_margin": LAMBDA_MARGIN, "lambda_rec": LAMBDA_REC,
            "gamma": GAMMA, "variable_t_range": VARIABLE_T_RANGE,
            "eval_T_values": EVAL_T_VALUES,
            "trajectory_max_T": TRAJECTORY_MAX_T,
        },
        "runs": [],
    }

    for lambda_sc in LAMBDA_SC_VALUES:
        for seed in SEEDS:
            if (seed, lambda_sc) in completed:
                print(f"\nSeed {seed}, lambda_sc={lambda_sc} already complete, skipping.",
                      flush=True)
                continue
            run = run_single(seed, lambda_sc, device)
            results["runs"].append(run)
            save_results(results)

    results["completed"] = datetime.now(timezone.utc).isoformat()
    save_results(results)
    print(f"\nAll runs complete. Results: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
