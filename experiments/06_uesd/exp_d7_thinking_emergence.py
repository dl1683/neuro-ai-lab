"""
Experiment D7: Thinking Emergence via Intermediate State Probing

Tests the core UESD thesis: do iterative dynamics perform PROGRESSIVE
COMPUTATION (i.e., "thinking"), or do all positions become correct
simultaneously?

Approach:
1. Train CE-dynamics and E5 on addition (seed=42)
2. At convergence, run dynamics s_0 -> s_1 -> ... -> s_T
3. At EACH intermediate step t, apply readout and measure:
   a. Per-position token accuracy (positions 0-3 of result half)
   b. Per-position readout margin (confidence)
   c. Per-position logit entropy (uncertainty)
4. Carry-chain analysis:
   a. For each eval example, compute exact carry pattern
   b. Group by carry-chain length at each position
   c. Test: do positions with longer carry chains become correct later?

Prediction if "thinking" occurs:
- Rightmost result positions (no carry dependency) correct first
- Leftmost result positions (max carry chain) correct later
- Wavefront of correctness moving right-to-left through dynamics steps
- Carry-chain length predicts the step at which a position becomes correct

Prediction if NO "thinking" (simultaneous transition):
- All positions become correct at the same step
- No correlation between carry-chain length and step-of-correctness

This is computationally cheap (evaluation only after training) and
scientifically very informative.
"""
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch


def build_config(seed=42):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "warmup_steps": 5000,
        "seed": seed,
    }


def train_model(track, config, device):
    seed = config["seed"]
    set_seed(seed)
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]

    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L_out = src.shape
        s = model.init_state(B, L_out, src.device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]

        if track == "dynamics_ce":
            loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                   tgt_r.reshape(-1))
        else:
            ce = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                 tgt_r.reshape(-1))
            s_next = model.dynamics(s, context)
            sc = (s_next - s).pow(2).sum(dim=-1).mean()
            eff_lam = min(step / config["warmup_steps"], 1.0)
            loss = ce + eff_lam * sc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    return model


def compute_carry_pattern(src, vocab_size):
    """Compute the carry-IN pattern for each example in the batch.

    Returns:
        carry_in: (B, half) tensor where carry_in[b, i] = 1 if position i
                  receives a carry from position i+1 (rightward neighbor)
        carry_dep_depth: (B, half) tensor where carry_dep_depth[b, i] =
                         depth of the incoming carry dependency chain.
                         0 = no carry-in (result is just (a_i + b_i) mod V)
                         1 = carry from i+1 (depends on one neighbor)
                         2 = carry chain from i+2 -> i+1 -> i (two deep)
                         etc.
    """
    B, L = src.shape
    half = L // 2
    a = src[:, 0::2][:, :half]
    b = src[:, 1::2][:, :half]

    # First compute carry_out at each position (right to left)
    carry_out = torch.zeros(B, half, dtype=torch.long, device=src.device)
    carry = torch.zeros(B, dtype=torch.long, device=src.device)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        carry = (s >= vocab_size).long()
        carry_out[:, i] = carry

    # carry_in[i] = carry_out[i+1] (carry from the right neighbor)
    # Position half-1 (rightmost) always has carry_in = 0
    carry_in = torch.zeros(B, half, dtype=torch.long, device=src.device)
    if half > 1:
        carry_in[:, :half-1] = carry_out[:, 1:half]

    # Carry dependency depth: how deep is the chain of consecutive
    # carry-generating positions flowing into position i?
    # Scan right to left: depth[i] = (1 + depth[i+1]) if carry_in[i]=1, else 0
    carry_dep_depth = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 2, -1, -1):
        has_carry = carry_in[:, i] == 1
        carry_dep_depth[has_carry, i] = 1 + carry_dep_depth[has_carry, i + 1]

    return carry_in, carry_dep_depth


@torch.no_grad()
def probe_intermediate_states(model, src, tgt, T, vocab_size):
    """Run dynamics and probe readout at each intermediate step.

    Returns dict with per-step, per-position accuracy, margin, and entropy.
    """
    model.eval()
    B, L = src.shape
    half = L // 2

    context = model.encode(src)
    s = model.init_state(B, L, src.device)

    step_results = []

    for t in range(T + 1):
        logits = model.readout_logits(s)
        preds = logits.argmax(dim=-1)

        correct = (preds == tgt).float()
        pos_acc = correct[:, :half].mean(dim=0).cpu().numpy()
        pad_acc = correct[:, half:].mean(dim=0).cpu().numpy()

        probs = F.softmax(logits, dim=-1)
        correct_probs = probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        pos_correct_prob = correct_probs[:, :half].mean(dim=0).cpu().numpy()

        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        pos_entropy = entropy[:, :half].mean(dim=0).cpu().numpy()

        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        margin_vals = sorted_logits[:, :, 0] - sorted_logits[:, :, 1]
        tgt_logit = logits.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        max_other = sorted_logits[:, :, 0].clone()
        is_correct_max = (preds == tgt)
        max_other[is_correct_max] = sorted_logits[:, :, 1][is_correct_max]
        signed_margin = tgt_logit - max_other
        pos_margin = signed_margin[:, :half].mean(dim=0).cpu().numpy()

        seq_acc_result = correct[:, :half].all(dim=1).float().mean().item()

        step_results.append({
            "step": t,
            "pos_accuracy": pos_acc.tolist(),
            "pos_correct_prob": pos_correct_prob.tolist(),
            "pos_entropy": pos_entropy.tolist(),
            "pos_margin": pos_margin.tolist(),
            "pad_accuracy": pad_acc.tolist(),
            "mean_accuracy_result": float(pos_acc.mean()),
            "seq_accuracy": seq_acc_result,
        })

        if t < T:
            s, _ = model.dynamics_step(s, context)

    return step_results


@torch.no_grad()
def carry_chain_analysis(model, src, tgt, T, vocab_size):
    """Analyze how carry-chain complexity affects step-of-correctness.

    For each example and each result position, find the first step
    where the readout becomes correct, then correlate with carry-chain length.
    """
    model.eval()
    B, L = src.shape
    half = L // 2

    carry_in, carry_dep_depth = compute_carry_pattern(src, vocab_size)

    context = model.encode(src)
    s = model.init_state(B, L, src.device)

    all_correct = torch.zeros(B, half, T + 1, dtype=torch.bool, device=src.device)

    for t in range(T + 1):
        logits = model.readout_logits(s)
        preds = logits.argmax(dim=-1)
        all_correct[:, :, t] = (preds[:, :half] == tgt[:, :half])
        if t < T:
            s, _ = model.dynamics_step(s, context)

    # First STABLE correct step: correct from t through T (Codex: report both)
    first_stable_step = torch.full((B, half), T + 1,
                                    dtype=torch.long, device=src.device)
    for t in range(T + 1):
        still_correct = all_correct[:, :, t:].all(dim=2)
        mask = (first_stable_step == T + 1) & still_correct
        first_stable_step[mask] = t

    # First HIT step: first time correct (may become wrong later)
    first_hit_step = torch.full((B, half), T + 1,
                                 dtype=torch.long, device=src.device)
    for t in range(T + 1):
        mask = (first_hit_step == T + 1) & all_correct[:, :, t]
        first_hit_step[mask] = t

    chain_to_first_correct = defaultdict(list)
    chain_to_first_hit = defaultdict(list)
    for bi in range(B):
        for pos in range(half):
            cl = carry_dep_depth[bi, pos].item()
            fs = first_stable_step[bi, pos].item()
            fh = first_hit_step[bi, pos].item()
            chain_to_first_correct[cl].append(fs)
            chain_to_first_hit[cl].append(fh)

    carry_in_to_first_correct = {0: [], 1: []}
    for bi in range(B):
        for pos in range(half):
            ci = carry_in[bi, pos].item()
            fc = first_stable_step[bi, pos].item()
            carry_in_to_first_correct[ci].append(fc)

    pos_to_first_correct = {}
    for pos in range(half):
        fcs = first_stable_step[:, pos].cpu().numpy()
        fch = first_hit_step[:, pos].cpu().numpy()
        pos_to_first_correct[pos] = {
            "mean_stable": float(np.mean(fcs)),
            "mean_hit": float(np.mean(fch)),
            "median_stable": float(np.median(fcs)),
            "std_stable": float(np.std(fcs)),
        }

    chain_stats = {}
    for cl in sorted(chain_to_first_correct.keys()):
        vals_s = chain_to_first_correct[cl]
        vals_h = chain_to_first_hit[cl]
        chain_stats[cl] = {
            "count": len(vals_s),
            "mean_first_stable": float(np.mean(vals_s)),
            "mean_first_hit": float(np.mean(vals_h)),
            "median_first_stable": float(np.median(vals_s)),
            "std_first_stable": float(np.std(vals_s)),
            "frac_never_correct": sum(1 for v in vals_s if v > T) / len(vals_s),
        }

    carry_in_stats = {}
    for ci in [0, 1]:
        vals = carry_in_to_first_correct[ci]
        if vals:
            carry_in_stats[ci] = {
                "count": len(vals),
                "mean_first_stable": float(np.mean(vals)),
                "median_first_stable": float(np.median(vals)),
            }

    # Position-stratified carry analysis: within each position, compare
    # first_correct_step for examples with carry_in=0 vs carry_in=1.
    # This isolates the carry effect from the position effect.
    pos_carry_stats = {}
    for pos in range(half):
        for ci in [0, 1]:
            mask = carry_in[:, pos] == ci
            if mask.sum() > 0:
                fcs = first_stable_step[mask, pos].cpu().numpy()
                fch = first_hit_step[mask, pos].cpu().numpy()
                key = f"pos{pos}_carry{ci}"
                pos_carry_stats[key] = {
                    "count": int(mask.sum()),
                    "mean_first_stable": float(np.mean(fcs)),
                    "mean_first_hit": float(np.mean(fch)),
                }

    # Difficulty-stratified: group by total number of carry-ins in the example
    total_carry_ins = carry_in.sum(dim=1)
    difficulty_stats = {}
    for nc in range(half + 1):
        mask = total_carry_ins == nc
        if mask.sum() > 0:
            fcs = first_stable_step[mask].cpu().numpy()
            difficulty_stats[nc] = {
                "count": int(mask.sum()),
                "mean_first_stable_all_pos": float(np.mean(fcs)),
                "per_pos_mean": [float(np.mean(first_stable_step[mask, p].cpu().numpy()))
                                 for p in range(half)],
            }

    return {
        "chain_length_stats": chain_stats,
        "carry_in_stats": carry_in_stats,
        "position_stats": pos_to_first_correct,
        "position_stratified_carry": pos_carry_stats,
        "difficulty_stats": {str(k): v for k, v in difficulty_stats.items()},
    }


@torch.no_grad()
def step_transition_analysis(model, src, tgt, T, vocab_size):
    """Detailed analysis of WHEN each position transitions to correct.

    Measures: at each step, how many new positions become correct?
    Is there a wavefront pattern?
    """
    model.eval()
    B, L = src.shape
    half = L // 2

    context = model.encode(src)
    s = model.init_state(B, L, src.device)

    prev_correct = torch.zeros(B, half, dtype=torch.bool, device=src.device)
    transitions = []

    for t in range(T + 1):
        logits = model.readout_logits(s)
        preds = logits.argmax(dim=-1)
        curr_correct = (preds[:, :half] == tgt[:, :half])

        newly_correct = curr_correct & ~prev_correct
        newly_wrong = ~curr_correct & prev_correct

        per_pos_new = newly_correct.float().mean(dim=0).cpu().numpy()
        per_pos_lost = newly_wrong.float().mean(dim=0).cpu().numpy()

        transitions.append({
            "step": t,
            "newly_correct_per_pos": per_pos_new.tolist(),
            "newly_wrong_per_pos": per_pos_lost.tolist(),
            "total_newly_correct": float(newly_correct.float().mean()),
            "total_newly_wrong": float(newly_wrong.float().mean()),
        })

        prev_correct = curr_correct.clone()

        if t < T:
            s, _ = model.dynamics_step(s, context)

    return transitions


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(seed=42)
    N_EVAL = 4096
    T = config["T"]
    V = config["vocab_size"]

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Thinking emergence: do dynamics progressively resolve carry chains?",
        "config": config,
        "n_eval": N_EVAL,
    }

    track_results = {}

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'#' * 60}", flush=True)
        print(f"  Training {track} seed=42", flush=True)
        print(f"{'#' * 60}", flush=True)

        t0 = time.time()
        model = train_model(track, config, device)
        train_time = time.time() - t0
        print(f"  Training done in {train_time:.0f}s", flush=True)

        model.eval()
        set_seed(999)
        eval_src, eval_tgt = generate_batch("addition", N_EVAL,
                                             config["seq_len"], V)
        eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)

        final_logits = model(eval_src, T)
        final_preds = final_logits.argmax(dim=-1)
        final_acc = (final_preds == eval_tgt).float().mean().item()
        final_seq_acc = (final_preds == eval_tgt).all(dim=1).float().mean().item()
        print(f"  Final accuracy: {final_acc:.4f} token, "
              f"{final_seq_acc:.4f} seq", flush=True)

        print(f"  Probing intermediate states...", flush=True)
        step_data = probe_intermediate_states(model, eval_src, eval_tgt, T, V)

        print(f"\n  === PER-STEP ACCURACY (result positions 0-3) ===", flush=True)
        print(f"  {'Step':>4s}  {'Pos0':>6s}  {'Pos1':>6s}  {'Pos2':>6s}  "
              f"{'Pos3':>6s}  {'Mean':>6s}  {'SeqAcc':>6s}", flush=True)
        for sd in step_data:
            pa = sd["pos_accuracy"]
            print(f"  {sd['step']:>4d}  {pa[0]:>6.3f}  {pa[1]:>6.3f}  "
                  f"{pa[2]:>6.3f}  {pa[3]:>6.3f}  "
                  f"{sd['mean_accuracy_result']:>6.3f}  "
                  f"{sd['seq_accuracy']:>6.3f}", flush=True)

        print(f"\n  === PER-STEP MARGIN (result positions 0-3) ===", flush=True)
        print(f"  {'Step':>4s}  {'Pos0':>7s}  {'Pos1':>7s}  {'Pos2':>7s}  "
              f"{'Pos3':>7s}", flush=True)
        for sd in step_data:
            pm = sd["pos_margin"]
            print(f"  {sd['step']:>4d}  {pm[0]:>7.2f}  {pm[1]:>7.2f}  "
                  f"{pm[2]:>7.2f}  {pm[3]:>7.2f}", flush=True)

        print(f"\n  === CARRY-CHAIN ANALYSIS ===", flush=True)
        carry_data = carry_chain_analysis(model, eval_src, eval_tgt, T, V)

        for cl, stats in carry_data["chain_length_stats"].items():
            print(f"    Depth {cl}: n={stats['count']:>5d}, "
                  f"stable={stats['mean_first_stable']:.2f}, "
                  f"hit={stats['mean_first_hit']:.2f}, "
                  f"never={stats['frac_never_correct']:.3f}", flush=True)

        for ci, stats in carry_data["carry_in_stats"].items():
            label = "no carry" if ci == 0 else "carry in"
            print(f"    {label}: n={stats['count']:>5d}, "
                  f"mean first stable={stats['mean_first_stable']:.2f}", flush=True)

        print(f"\n  === PER-POSITION FIRST-CORRECT STEP ===", flush=True)
        for pos, stats in carry_data["position_stats"].items():
            print(f"    Position {pos}: stable={stats['mean_stable']:.2f}, "
                  f"hit={stats['mean_hit']:.2f}, "
                  f"std={stats['std_stable']:.2f}", flush=True)

        print(f"\n  === POSITION-STRATIFIED CARRY EFFECT ===", flush=True)
        print(f"  (Within each position: carry_in=0 vs carry_in=1)", flush=True)
        for key, stats in carry_data["position_stratified_carry"].items():
            print(f"    {key}: n={stats['count']:>5d}, "
                  f"stable={stats['mean_first_stable']:.2f}, "
                  f"hit={stats['mean_first_hit']:.2f}", flush=True)
        half = config["seq_len"] // 2
        for pos in range(half):
            k0 = f"pos{pos}_carry0"
            k1 = f"pos{pos}_carry1"
            if k0 in carry_data["position_stratified_carry"] and k1 in carry_data["position_stratified_carry"]:
                m0 = carry_data["position_stratified_carry"][k0]["mean_first_stable"]
                m1 = carry_data["position_stratified_carry"][k1]["mean_first_stable"]
                delta = m1 - m0
                print(f"    Position {pos} carry effect: "
                      f"+{delta:.2f} steps (stable) when carry_in=1", flush=True)

        print(f"\n  === DIFFICULTY STRATIFICATION ===", flush=True)
        for nc, stats in carry_data["difficulty_stats"].items():
            print(f"    {nc} carry-ins: n={stats['count']:>5d}, "
                  f"mean first stable={stats['mean_first_stable_all_pos']:.2f}, "
                  f"per-pos={[f'{v:.2f}' for v in stats['per_pos_mean']]}",
                  flush=True)

        print(f"\n  === STEP TRANSITIONS ===", flush=True)
        transition_data = step_transition_analysis(
            model, eval_src, eval_tgt, T, V)
        for td in transition_data:
            nc = td["newly_correct_per_pos"]
            if sum(nc) > 0.001:
                print(f"    Step {td['step']}: newly correct "
                      f"[{nc[0]:.3f}, {nc[1]:.3f}, {nc[2]:.3f}, {nc[3]:.3f}]",
                      flush=True)

        # Test for thinking emergence: correlation between
        # carry dependency depth and first-correct step
        carry_in_data, dep_depth_data = compute_carry_pattern(eval_src, V)
        context = model.encode(eval_src)
        s = model.init_state(N_EVAL, config["seq_len"], eval_src.device)
        half = config["seq_len"] // 2
        all_correct_steps = torch.zeros(N_EVAL, half, T + 1,
                                         dtype=torch.bool, device=device)
        for t in range(T + 1):
            logits = model.readout_logits(s)
            preds = logits.argmax(dim=-1)
            all_correct_steps[:, :, t] = (preds[:, :half] == eval_tgt[:, :half])
            if t < T:
                s, _ = model.dynamics_step(s, context)

        first_correct = torch.full((N_EVAL, half), T + 1,
                                    dtype=torch.long, device=device)
        for t in range(T + 1):
            still_correct = all_correct_steps[:, :, t:].all(dim=2)
            mask = (first_correct == T + 1) & still_correct
            first_correct[mask] = t

        first_hit = torch.full((N_EVAL, half), T + 1,
                                dtype=torch.long, device=device)
        for t in range(T + 1):
            mask = (first_hit == T + 1) & all_correct_steps[:, :, t]
            first_hit[mask] = t

        dd_flat = dep_depth_data.reshape(-1).cpu().numpy()
        fs_flat = first_correct.reshape(-1).cpu().numpy()
        fh_flat = first_hit.reshape(-1).cpu().numpy()
        pos_flat = np.tile(np.arange(half), N_EVAL)
        valid = fs_flat <= T
        if valid.sum() > 10:
            corr_stable = np.corrcoef(dd_flat[valid], fs_flat[valid])[0, 1]
            corr_hit = np.corrcoef(dd_flat[valid], fh_flat[valid])[0, 1]
        else:
            corr_stable = float('nan')
            corr_hit = float('nan')

        # Position-controlled partial correlation (within-position)
        within_pos_corrs = []
        for pos in range(half):
            idx = pos_flat == pos
            idx_valid = idx & valid
            if idx_valid.sum() > 10:
                r = np.corrcoef(dd_flat[idx_valid], fs_flat[idx_valid])[0, 1]
                within_pos_corrs.append(r)
        mean_within_pos_corr = float(np.mean(within_pos_corrs)) if within_pos_corrs else float('nan')

        print(f"\n  === THINKING EMERGENCE TEST ===", flush=True)
        print(f"  Global corr(depth, first_stable) = {corr_stable:.4f}", flush=True)
        print(f"  Global corr(depth, first_hit)    = {corr_hit:.4f}", flush=True)
        print(f"  Within-position corr (mean)      = {mean_within_pos_corr:.4f}", flush=True)
        if mean_within_pos_corr > 0.05:
            print(f"  RESULT: WITHIN-POSITION CORRELATION — carry dependency "
                  f"predicts step-of-correctness CONTROLLING FOR POSITION → "
                  f"EVIDENCE OF PROGRESSIVE COMPUTATION", flush=True)
        elif corr_stable > 0.1:
            print(f"  RESULT: GLOBAL CORRELATION but weak within-position — "
                  f"may be position confound, not thinking", flush=True)
        else:
            print(f"  RESULT: NO CORRELATION — all positions become correct "
                  f"at roughly the same step", flush=True)

        pos_mean_fc = [carry_data["position_stats"][p]["mean_stable"]
                       for p in range(half)]
        is_right_to_left = all(pos_mean_fc[i] >= pos_mean_fc[i+1]
                                for i in range(half - 1))
        print(f"  Position mean first-stable steps: {pos_mean_fc}", flush=True)
        print(f"  Right-to-left wavefront: {is_right_to_left}", flush=True)

        track_results[track] = {
            "train_time_s": train_time,
            "final_token_acc": final_acc,
            "final_seq_acc": final_seq_acc,
            "step_data": step_data,
            "carry_data": {
                "chain_length_stats": {
                    str(k): v for k, v in carry_data["chain_length_stats"].items()
                },
                "carry_in_stats": {
                    str(k): v for k, v in carry_data["carry_in_stats"].items()
                },
                "position_stats": {
                    str(k): v for k, v in carry_data["position_stats"].items()
                },
                "position_stratified_carry": carry_data["position_stratified_carry"],
                "difficulty_stats": carry_data["difficulty_stats"],
            },
            "transitions": transition_data,
            "corr_depth_vs_stable": float(corr_stable) if not math.isnan(corr_stable) else None,
            "corr_depth_vs_hit": float(corr_hit) if not math.isnan(corr_hit) else None,
            "corr_within_position": float(mean_within_pos_corr) if not math.isnan(mean_within_pos_corr) else None,
            "position_mean_first_stable": pos_mean_fc,
            "right_to_left_wavefront": is_right_to_left,
        }

    # Final comparison
    print(f"\n{'=' * 70}", flush=True)
    print("D7 THINKING EMERGENCE SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    for track in ["dynamics_ce", "e5"]:
        tr = track_results[track]
        print(f"\n  {track}:", flush=True)
        print(f"    Final accuracy: {tr['final_token_acc']:.4f} token, "
              f"{tr['final_seq_acc']:.4f} seq", flush=True)
        print(f"    Corr(depth, stable): {tr['corr_depth_vs_stable']}, "
              f"  Corr(depth, hit): {tr['corr_depth_vs_hit']}", flush=True)
        print(f"    Within-position corr: {tr['corr_within_position']}", flush=True)
        print(f"    Position first-stable: {tr['position_mean_first_stable']}",
              flush=True)
        print(f"    Wavefront: {tr['right_to_left_wavefront']}", flush=True)

        print(f"    Step-by-step accuracy (result half):", flush=True)
        for sd in tr["step_data"]:
            pa = sd["pos_accuracy"]
            print(f"      t={sd['step']:>2d}: [{pa[0]:.3f}, {pa[1]:.3f}, "
                  f"{pa[2]:.3f}, {pa[3]:.3f}] mean={sd['mean_accuracy_result']:.3f}",
                  flush=True)

    ce_wpc = track_results["dynamics_ce"]["corr_within_position"]
    e5_wpc = track_results["e5"]["corr_within_position"]
    print(f"\n  INTERPRETATION:", flush=True)
    if ce_wpc is not None and ce_wpc > 0.05:
        print(f"    CE-dynamics: WITHIN-POSITION carry effect confirmed "
              f"(r={ce_wpc:.3f}) → progressive computation", flush=True)
    if e5_wpc is not None and e5_wpc > 0.05:
        print(f"    E5: WITHIN-POSITION carry effect confirmed "
              f"(r={e5_wpc:.3f}) → progressive computation", flush=True)
    if ce_wpc is not None and e5_wpc is not None:
        if abs(ce_wpc or 0) > abs(e5_wpc or 0):
            print(f"    CE-dynamics shows STRONGER progressive computation",
                  flush=True)
        elif abs(e5_wpc or 0) > abs(ce_wpc or 0):
            print(f"    E5 shows STRONGER progressive computation", flush=True)
        else:
            print(f"    Both tracks show similar progressive computation",
                  flush=True)

    print(f"{'=' * 70}", flush=True)

    results["track_results"] = track_results

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d7_thinking_emergence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
