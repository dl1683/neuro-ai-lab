"""
Experiment D38: Basin-Coupled UESD with Rectified-Flow Correction

PROBLEM: CE dynamics get 100% accuracy but no true fixed points.
E5 self-consistency gets true fixed points but 20% wrong attractors.
k<1 guarantees convergence to SOME fixed point, not the CORRECT one.

SOLUTION: Keep proven UESD contraction machinery. Add a rectified-flow
corrector as a manifold projector. Gate self-consistency by margin so
it only stabilizes correct basins. Mine wrong attractors as negatives.

ARCHITECTURE (~900K params):
  ContextEncoder: 2-layer TransformerEncoder, d=128, h=4, ff=512
  G: 1 tied TransformerDecoderLayer (existing UESD dynamics)
  FlowHead: MLP [z,h,c_pool,t] -> velocity, hidden=256
  Readout: tied embedding cosine-sim (existing)

TRAINING (4 phases, VT throughout):
  Phase A (15k): CE-only warm-start — establish correct basins
  Phase B (10k): CE + flow — add manifold projection
  Phase C (10k): CE + flow + margin-gated SC — add stabilization
  Phase D (10k): CE + flow + SC + margin + recovery — expand basins

SUCCESS CRITERIA:
  seq_accuracy >= 99.9%
  converged_frac >= 95%
  wrong_attractor_rate <= 1% (down from 20%)
  no seed collapse (seed 512 was E5's failure case)
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
from shared.model import BasinCoupledUESD
from shared.training import set_seed, count_params
from shared.data import generate_batch

# ── Architecture ──────────────────────────────────────────────────────
VOCAB_SIZE = 64
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
FLOW_HIDDEN = 256
FLOW_T_DIM = 16

# ── Task ──────────────────────────────────────────────────────────────
TASK = "addition"
SEQ_LEN = 16
HALF = SEQ_LEN // 2

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE = 256
LR = 3e-4
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
EVAL_T = 10
LOG_INTERVAL = 2000

PHASE_A_STEPS = 15000
PHASE_B_STEPS = 10000
PHASE_C_STEPS = 10000
PHASE_D_STEPS = 10000

# ── Loss weights ──────────────────────────────────────────────────────
LAMBDA_FLOW = 1.0
LAMBDA_SC = 0.02
LAMBDA_MARGIN = 0.1
LAMBDA_REC = 0.2
SC_WARMUP = 3000
GAMMA = 2.0
REC_SIGMA = 0.1
REC_EXTRA_STEPS = 3

# ── Evaluation ────────────────────────────────────────────────────────
EVAL_SAMPLES = 4096
FP_STEPS = 50
K_TRAJECTORY = 15
FLOW_K_EVAL = [4, 8]

# ── Seeds ─────────────────────────────────────────────────────────────
SEEDS = [42, 137, 256, 512]

RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "exp_d38_basin_coupled_flow.json"
)


# ── Helpers ───────────────────────────────────────────────────────────

def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_model(device):
    m = BasinCoupledUESD(
        VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN,
        flow_hidden=FLOW_HIDDEN, flow_t_dim=FLOW_T_DIM,
    )
    return m.to(device)


def compute_margin(logits, target_ids):
    """Differentiable per-position margin: logit_correct - max_wrong."""
    correct = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, target_ids.unsqueeze(-1), True)
    wrong = logits.masked_fill(mask, float("-inf"))
    max_wrong = wrong.max(dim=-1).values
    return correct - max_wrong


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Forward helpers ───────────────────────────────────────────────────

def forward_dynamics(model, src, device, t_steps=None):
    """Run encoder + dynamics, return (h, context).  Keeps grad."""
    if t_steps is None:
        t_steps = random.choice(VARIABLE_T_RANGE)
    context = model.encode(src)
    B = src.size(0)
    s = model.init_state(B, SEQ_LEN, device)
    for _ in range(t_steps):
        s, _ = model.dynamics_step(s, context)
    return s, context, t_steps


def flow_loss(model, h, context, tgt):
    """Rectified-flow training loss.  Detaches h and context."""
    B = h.size(0)
    y_embed = model.tok_emb(tgt).detach()
    eps = torch.randn_like(y_embed)
    t = torch.rand(B, 1, 1, device=h.device)
    z_t = (1 - t) * y_embed + t * eps
    target_v = y_embed - eps
    c_pool = context.mean(dim=1).detach()
    pred_v = model.flow_head(z_t, h.detach(), c_pool, t.view(B))
    return F.mse_loss(pred_v, target_v)


# ── Phase step functions ─────────────────────────────────────────────

def _ce_result_only(logits, tgt):
    """CE on result positions only (first HALF), ignoring zero-padded tail."""
    lo = logits[:, :HALF].reshape(-1, logits.size(-1))
    ta = tgt[:, :HALF].reshape(-1)
    return F.cross_entropy(lo, ta)


def phase_a_step(model, src, tgt, device):
    """CE-only with variable-T."""
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)
    return ce, {"ce": ce.item(), "T": t_steps}


def phase_b_step(model, src, tgt, device):
    """CE + rectified-flow."""
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)
    fl = flow_loss(model, h, ctx, tgt)
    loss = ce + LAMBDA_FLOW * fl
    return loss, {"ce": ce.item(), "flow": fl.item(), "T": t_steps}


def phase_c_step(model, src, tgt, device, step_in_phase):
    """CE + flow + margin-gated SC."""
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)

    # Self-consistency (residual)
    s_next, _ = model.dynamics_step(h, ctx)
    residual = s_next - h
    sc = (residual ** 2).mean()

    # Margin hinge loss on result positions only
    margin = compute_margin(logits[:, :HALF], tgt[:, :HALF])
    mg = F.relu(GAMMA - margin).mean()

    # Flow
    fl = flow_loss(model, h, ctx, tgt)

    eff_sc = LAMBDA_SC * min(1.0, step_in_phase / SC_WARMUP)
    loss = ce + LAMBDA_FLOW * fl + eff_sc * sc + LAMBDA_MARGIN * mg
    return loss, {
        "ce": ce.item(), "flow": fl.item(), "sc": sc.item(),
        "margin_loss": mg.item(), "eff_sc": eff_sc, "T": t_steps,
    }


def phase_d_step(model, src, tgt, device, step_in_phase):
    """Full: CE + flow + SC + margin + perturbation recovery."""
    h, ctx, t_steps = forward_dynamics(model, src, device)
    logits = model.readout_logits(h)
    ce = _ce_result_only(logits, tgt)

    s_next, _ = model.dynamics_step(h, ctx)
    residual = s_next - h
    sc = (residual ** 2).mean()

    margin = compute_margin(logits[:, :HALF], tgt[:, :HALF])
    mg = F.relu(GAMMA - margin).mean()

    fl = flow_loss(model, h, ctx, tgt)

    # Recovery: perturb h (detached), re-roll a few steps, train readout
    noise = REC_SIGMA * torch.randn_like(h)
    s_rec = h.detach() + noise
    ctx_det = ctx.detach()
    for _ in range(REC_EXTRA_STEPS):
        s_rec, _ = model.dynamics_step(s_rec, ctx_det)
    logits_rec = model.readout_logits(s_rec)
    rec = _ce_result_only(logits_rec, tgt)

    loss = (ce + LAMBDA_FLOW * fl + LAMBDA_SC * sc
            + LAMBDA_MARGIN * mg + LAMBDA_REC * rec)
    return loss, {
        "ce": ce.item(), "flow": fl.item(), "sc": sc.item(),
        "margin_loss": mg.item(), "rec": rec.item(), "T": t_steps,
    }


# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def measure_k(model, src, device):
    """Standard contraction-rate estimator from D33/D37."""
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
def evaluate_model(model, device):
    """Full evaluation: h-direct, flow-corrected, contraction, basins."""
    model.eval()

    total_h_correct = 0
    total_h_tok_correct = 0
    total_flow = {K: {"correct": 0, "tok_correct": 0, "margin_sum": 0.0}
                  for K in FLOW_K_EVAL}
    total_converged = 0
    total_wa = 0
    total_margin_sum = 0.0
    total_margin_pos = 0
    total_residual_sum = 0.0
    total_positions = 0
    n_total = 0

    remaining = EVAL_SAMPLES
    while remaining > 0:
        bs = min(512, remaining)
        src, tgt = generate_batch(TASK, bs, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(bs, SEQ_LEN, device)
        for _ in range(EVAL_T):
            s, _ = model.dynamics_step(s, context)
        h = s

        # h-direct
        logits_h = model.readout_logits(h)
        preds_h = logits_h[:, :HALF].argmax(dim=-1)
        seq_correct = (preds_h == tgt[:, :HALF]).all(dim=-1)
        total_h_correct += seq_correct.sum().item()
        total_h_tok_correct += (preds_h == tgt[:, :HALF]).sum().item()

        # Residual
        s_next, _ = model.dynamics_step(h, context)
        diff = s_next - h
        norm_r = diff.norm(dim=(-2, -1)) / math.sqrt(SEQ_LEN * D_MODEL)
        total_residual_sum += norm_r.sum().item()

        # Margin (result positions only)
        margin = compute_margin(logits_h[:, :HALF], tgt[:, :HALF])
        total_margin_sum += margin.sum().item()
        total_margin_pos += (margin > 0).sum().item()
        total_positions += margin.numel()

        # Wrong-attractor
        converged = norm_r < 0.01
        total_converged += converged.sum().item()
        total_wa += (converged & ~seq_correct).sum().item()

        # Flow-corrected
        for K in FLOW_K_EVAL:
            z = model.flow_correct(h, context, K=K)
            logits_z = model.readout_logits(z)
            preds_z = logits_z[:, :HALF].argmax(dim=-1)
            fc = (preds_z == tgt[:, :HALF]).all(dim=-1)
            total_flow[K]["correct"] += fc.sum().item()
            total_flow[K]["tok_correct"] += (preds_z == tgt[:, :HALF]).sum().item()
            fm = compute_margin(logits_z[:, :HALF], tgt[:, :HALF])
            total_flow[K]["margin_sum"] += fm.sum().item()

        n_total += bs
        remaining -= bs

    # Contraction rate
    src_k, _ = generate_batch(TASK, 64, SEQ_LEN, VOCAB_SIZE)
    src_k = src_k.to(device)
    k_mean = measure_k(model, src_k, device)

    conv_frac = total_converged / n_total if n_total > 0 else 0.0
    wa_rate = total_wa / total_converged if total_converged > 0 else 0.0

    result = {
        "h_seq_acc": total_h_correct / n_total,
        "h_tok_acc": total_h_tok_correct / (n_total * HALF),
        "residual_mean": total_residual_sum / n_total,
        "margin_mean": total_margin_sum / total_positions,
        "margin_frac_positive": total_margin_pos / total_positions,
        "converged_frac": conv_frac,
        "wrong_attractor_rate": wa_rate,
        "k_mean": k_mean,
    }
    for K in FLOW_K_EVAL:
        d = total_flow[K]
        result[f"flow_K{K}_seq_acc"] = d["correct"] / n_total
        result[f"flow_K{K}_tok_acc"] = d["tok_correct"] / (n_total * HALF)
        result[f"flow_K{K}_margin_mean"] = d["margin_sum"] / total_positions

    model.train()
    return result


# ── Training orchestration ────────────────────────────────────────────

def train_phase(model, optimizer, phase_fn, n_steps, phase_name, device,
                extra_kwargs=None):
    """Generic phase trainer with logging and timing."""
    t0 = time.time()
    history = []
    for step in range(1, n_steps + 1):
        src, tgt = generate_batch(TASK, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        kwargs = {"step_in_phase": step} if extra_kwargs else {}
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


def run_single(seed, device):
    """Full 4-phase training for one seed."""
    full_seed(seed)
    model = make_model(device)
    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"Seed {seed} | {n_params:,} params | device={device}")
    print(f"{'='*60}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    run = {"seed": seed, "n_params": n_params, "phases": {}}

    # Phase A
    print("\n--- Phase A: CE-only ---", flush=True)
    run["phases"]["A"] = train_phase(
        model, optimizer, phase_a_step, PHASE_A_STEPS, "A", device,
    )
    run["phases"]["A"]["eval"] = evaluate_model(model, device)
    print_eval_summary("A", run["phases"]["A"]["eval"])

    # Phase B
    print("\n--- Phase B: CE + Flow ---", flush=True)
    run["phases"]["B"] = train_phase(
        model, optimizer, phase_b_step, PHASE_B_STEPS, "B", device,
    )
    run["phases"]["B"]["eval"] = evaluate_model(model, device)
    print_eval_summary("B", run["phases"]["B"]["eval"])

    # Phase C
    print("\n--- Phase C: CE + Flow + Margin-gated SC ---", flush=True)
    run["phases"]["C"] = train_phase(
        model, optimizer, phase_c_step, PHASE_C_STEPS, "C", device,
        extra_kwargs=True,
    )
    run["phases"]["C"]["eval"] = evaluate_model(model, device)
    print_eval_summary("C", run["phases"]["C"]["eval"])

    # Phase D
    print("\n--- Phase D: Full (+ recovery) ---", flush=True)
    run["phases"]["D"] = train_phase(
        model, optimizer, phase_d_step, PHASE_D_STEPS, "D", device,
        extra_kwargs=True,
    )
    run["phases"]["D"]["eval"] = evaluate_model(model, device)
    print_eval_summary("D", run["phases"]["D"]["eval"])

    return run


def print_eval_summary(phase, ev):
    print(f"\n  Phase {phase} eval:")
    print(f"    h_seq_acc:     {ev['h_seq_acc']:.4f}")
    print(f"    converged:     {ev['converged_frac']:.4f}")
    print(f"    WA rate:       {ev['wrong_attractor_rate']:.4f}")
    print(f"    margin:        {ev['margin_mean']:.2f} ({ev['margin_frac_positive']:.2%} pos)")
    print(f"    residual:      {ev['residual_mean']:.6f}")
    print(f"    k:             {ev['k_mean']:.4f}")
    for K in FLOW_K_EVAL:
        print(f"    flow K={K}:      {ev[f'flow_K{K}_seq_acc']:.4f} "
              f"(margin {ev[f'flow_K{K}_margin_mean']:.2f})")
    print(flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D38 Basin-Coupled UESD + Rectified Flow")
    print(f"Device: {device}", flush=True)

    existing = load_checkpoint()
    completed_seeds = set()
    if existing and "runs" in existing:
        completed_seeds = {r["seed"] for r in existing["runs"]}

    results = existing or {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Basin-Coupled UESD with rectified-flow correction",
        "config": {
            "vocab_size": VOCAB_SIZE, "d_model": D_MODEL,
            "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "seq_len": SEQ_LEN,
            "flow_hidden": FLOW_HIDDEN, "flow_t_dim": FLOW_T_DIM,
            "phases": {
                "A": PHASE_A_STEPS, "B": PHASE_B_STEPS,
                "C": PHASE_C_STEPS, "D": PHASE_D_STEPS,
            },
            "lambda_flow": LAMBDA_FLOW, "lambda_sc": LAMBDA_SC,
            "lambda_margin": LAMBDA_MARGIN, "lambda_rec": LAMBDA_REC,
            "gamma": GAMMA, "variable_t_range": VARIABLE_T_RANGE,
        },
        "runs": [],
    }

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"\nSeed {seed} already complete, skipping.", flush=True)
            continue
        run = run_single(seed, device)
        results["runs"].append(run)
        save_results(results)

    results["completed"] = datetime.now(timezone.utc).isoformat()
    save_results(results)
    print(f"\nAll seeds complete. Results: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
