"""
Experiment D8: Causal Carry Probing

Tests whether UESD dynamics ACTUALLY compute carry propagation,
not just produce correct final answers. Three progressively
stronger tests:

Phase 1 — Carry Probes (Observational):
  Train linear probes at each dynamics step to predict BOTH
  carry-in and carry-out at each result position.
  carry_in[k] = incoming carry into position k (from position k+1)
  carry_out[k] = outgoing carry from position k (to position k-1)
  Position 3 (LSB): carry_in always 0, carry_out depends on a_3+b_3.
  Position 0 (MSB): carry_out is overflow (discarded).
  Expected wavefront for carry_in: pos2 → pos1 → pos0 across steps.
  Expected wavefront for carry_out: pos3 → pos2 → pos1 across steps.

Phase 2 — Carry-Flip Perturbation (Correlational):
  Generate MATCHED pairs: (a) carry-flipping perturbation that changes
  carry_out at position k, (b) CONTROL perturbation that changes a_k
  by similar magnitude WITHOUT changing carry status. Divergence at
  position k-1 beyond control baseline = carry propagation signal.

Phase 3 — State Surgery (Causal):
  At intermediate step t, modify the state along the carry-out
  probe direction to flip carry_out at position k. Focus on boundary
  examples where carry propagates further leftward. Use SEPARATE
  held-out data for surgery (not used for probe training).

PREDICTIONS (per Codex review):
1. carry_in probes: pos2 crosses 80% at steps 2-4, pos1 at 4-6,
   pos0 at 5-8 for CE-dynamics. E5 smoother, similar range.
   pos3 is N/A (always 0). carry_out probes: pos3 first, pos0 last.
2. Surgery persistence: CE-dynamics retains, E5 may overwrite.
   Direct output change at k almost always; leftward change only
   for boundary examples (~1/64 of random inputs).
3. Perturbation divergence at k-1 should exceed matched control.
"""
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch


N_PROBE = 4096
N_SURGERY = 2048
N_PROBE_EPOCHS = 200
SEED = 42
T_SURGERY = [3, 5, 7]


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


def compute_carries(src, vocab_size):
    """Compute carry-in AND carry-out at each result position."""
    B, L = src.shape
    half = L // 2
    a = src[:, 0::2][:, :half]
    b = src[:, 1::2][:, :half]

    carry_out = torch.zeros(B, half, dtype=torch.long, device=src.device)
    carry = torch.zeros(B, dtype=torch.long, device=src.device)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        carry = (s >= vocab_size).long()
        carry_out[:, i] = carry

    carry_in = torch.zeros(B, half, dtype=torch.long, device=src.device)
    if half > 1:
        carry_in[:, :half - 1] = carry_out[:, 1:half]
    return carry_in, carry_out


@torch.no_grad()
def collect_all_states(model, src, T):
    """Run dynamics and return states at every step plus encoder context."""
    context = model.encode(src)
    B, L = src.shape
    s = model.init_state(B, L, src.device)
    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())
    return states, context


@torch.no_grad()
def run_from_state(model, s, context, remaining_steps):
    """Continue dynamics from a given state for remaining_steps."""
    for _ in range(remaining_steps):
        s, _ = model.dynamics_step(s, context)
    return s


# ========== TRAINING ==========

def train_model(track, config, device):
    """Train a UESD model with the given track (e5 or dynamics_ce)."""
    set_seed(config["seed"])
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]
    total = config["training_steps"]

    for step in range(1, total + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        half = L // 2
        s = model.init_state(B, L, device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]
        ce = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)), tgt_r.reshape(-1))

        if track == "e5":
            sc = (s - model.dynamics(s, context)).pow(2).mean()
            eff_lam = min(step / config.get("warmup_steps", 5000), 1.0)
            loss = ce + eff_lam * sc
        else:
            loss = ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{total} | Loss: {loss.item():.4f}"
                  + (f" | CE: {ce.item():.4f}" if track == "e5" else ""),
                  flush=True)

    model.eval()
    return model


# ========== PHASE 1: CARRY PROBES ==========

class CarryProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _train_single_probe(X, y, d_model, device):
    """Train one logistic regression probe. Returns (probe, val_acc, auc)."""
    n_pos = y.sum().item()
    n_neg = y.shape[0] - n_pos
    if n_pos < 10 or n_neg < 10:
        return None, float("nan"), float("nan"), int(n_pos), int(n_neg)

    probe = CarryProbe(d_model).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    n_train = int(0.8 * X.shape[0])
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    best_acc = 0.0
    best_state = None
    for epoch in range(N_PROBE_EPOCHS):
        logits = probe(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                val_preds = (probe(X_val) > 0).long()
                val_acc = (val_preds == y_val.long()).float().mean().item()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k_: v.clone() for k_, v in
                                  probe.state_dict().items()}

    if best_state:
        probe.load_state_dict(best_state)

    with torch.no_grad():
        val_logits = probe(X_val)
        val_preds = (val_logits > 0).long()
        val_acc = (val_preds == y_val.long()).float().mean().item()
        probs = torch.sigmoid(val_logits)
        sorted_probs, sorted_idx = probs.sort()
        sorted_labels = y_val.long()[sorted_idx]
        tp = sorted_labels.sum().item()
        fp = (1 - sorted_labels).sum().item()
        auc = 0.0
        if tp > 0 and fp > 0:
            auc_sum = 0.0
            tp_remaining = tp
            for i in range(len(sorted_labels)):
                if sorted_labels[i] == 0:
                    auc_sum += tp_remaining
                else:
                    tp_remaining -= 1
            auc = auc_sum / (tp * fp)

    return probe, val_acc, auc, int(n_pos), int(n_neg)


def train_carry_probes(states, carry_in, carry_out, T, half, d_model, device):
    """Train probes for BOTH carry-in and carry-out at each (step, position)."""
    results = {"carry_in": {}, "carry_out": {}}

    for label_name, labels in [("carry_in", carry_in), ("carry_out", carry_out)]:
        for t in range(T + 1):
            for k in range(half):
                X = states[t][:, k, :].detach().clone()
                y = labels[:, k].float()
                probe, acc, auc, n_pos, n_neg = _train_single_probe(
                    X, y, d_model, device
                )
                results[label_name][(t, k)] = {
                    "acc": acc, "auc": auc,
                    "n_pos": n_pos, "n_neg": n_neg,
                    "probe": probe,
                }

    return results


# ========== PHASE 2: CARRY-FLIP PERTURBATION ==========

def generate_carry_flip_pairs(n_examples, seq_len, vocab_size, flip_pos):
    """Generate MATCHED pairs: carry-flip AND control (no-carry-flip).

    Returns (src_base, src_carry_flip, src_control, valid_mask).
    carry_flip: changes a_k to flip carry_out status at position flip_pos.
    control: changes a_k by similar magnitude WITHOUT flipping carry_out.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (n_examples, half))
    b = torch.randint(0, vocab_size, (n_examples, half))

    carry = torch.zeros(n_examples, dtype=torch.long)
    for i in range(half - 1, flip_pos, -1):
        s = a[:, i] + b[:, i] + carry
        carry = (s >= vocab_size).long()
    carry_in_k = carry

    digit_sum = a[:, flip_pos] + b[:, flip_pos] + carry_in_k
    has_carry = (digit_sum >= vocab_size)

    a_flipped = a.clone()
    a_control = a.clone()
    valid = torch.ones(n_examples, dtype=torch.bool)

    for i in range(n_examples):
        orig_a = a[i, flip_pos].item()
        b_k = b[i, flip_pos].item()
        c_in = carry_in_k[i].item()
        threshold = vocab_size - b_k - c_in

        if has_carry[i]:
            # Flip: move a_k below threshold
            if threshold <= 0:
                valid[i] = False
                continue
            a_flipped[i, flip_pos] = torch.randint(0, threshold, (1,))
            # Control: keep a_k above threshold but change magnitude
            new_a = orig_a
            for _ in range(10):
                candidate = torch.randint(max(threshold, 0), vocab_size, (1,)).item()
                if candidate != orig_a:
                    new_a = candidate
                    break
            a_control[i, flip_pos] = new_a
        else:
            # Flip: move a_k above threshold
            if threshold >= vocab_size:
                valid[i] = False
                continue
            a_flipped[i, flip_pos] = torch.randint(max(threshold, 0), vocab_size, (1,))
            # Control: keep a_k below threshold but change magnitude
            new_a = orig_a
            for _ in range(10):
                candidate = torch.randint(0, max(threshold, 1), (1,)).item()
                if candidate != orig_a:
                    new_a = candidate
                    break
            a_control[i, flip_pos] = new_a

    def pack_input(a_vals, b_vals, seq_len):
        n = a_vals.shape[0]
        src = torch.zeros(n, seq_len, dtype=torch.long)
        src[:, 0::2] = a_vals[:, :seq_len // 2 + seq_len % 2]
        src[:, 1::2] = b_vals[:, :seq_len // 2]
        return src

    src_base = pack_input(a, b, seq_len)
    src_flip = pack_input(a_flipped, b, seq_len)
    src_ctrl = pack_input(a_control, b, seq_len)

    return src_base, src_flip, src_ctrl, valid


@torch.no_grad()
def measure_perturbation_divergence(model, src_base, src_flip, src_ctrl, T):
    """Measure per-position state divergence: carry-flip vs control."""
    states_base, _ = collect_all_states(model, src_base, T)
    states_flip, _ = collect_all_states(model, src_flip, T)
    states_ctrl, _ = collect_all_states(model, src_ctrl, T)

    half = src_base.shape[1] // 2
    div_flip = torch.zeros(T + 1, half)
    div_ctrl = torch.zeros(T + 1, half)

    for t in range(T + 1):
        diff_f = states_base[t] - states_flip[t]
        diff_c = states_base[t] - states_ctrl[t]
        for k in range(half):
            div_flip[t, k] = diff_f[:, k, :].norm(dim=-1).mean().item()
            div_ctrl[t, k] = diff_c[:, k, :].norm(dim=-1).mean().item()

    base_norm = states_base[-1][:, :half, :].norm(dim=-1).mean().item()
    return div_flip, div_ctrl, base_norm


# ========== PHASE 3: STATE SURGERY ==========

@torch.no_grad()
def state_surgery_experiment(model, src, tgt, carry_out, probe_results,
                             T, vocab_size, t_surgery_steps):
    """Flip carry_out at position k using carry_out probe direction.

    Uses carry_out (not carry_in) because carry_out at position k
    directly feeds carry_in at position k-1. Flipping carry_out tests
    whether the dynamics propagate the change leftward.
    """
    half = src.shape[1] // 2
    B = src.shape[0]
    results = []

    states, context = collect_all_states(model, src, T)

    logits_orig = model.readout_logits(states[T])
    preds_orig = logits_orig[:, :half, :].argmax(dim=-1)

    a = src[:, 0::2][:, :half]
    b = src[:, 1::2][:, :half]

    co_probes = probe_results["carry_out"]

    for t_surg in t_surgery_steps:
        for k in range(1, half):  # positions 1..3 have meaningful carry_out
            probe_key = (t_surg, k)
            if probe_key not in co_probes:
                continue
            pr = co_probes[probe_key]
            if pr.get("probe") is None or (isinstance(pr["acc"], float)
                                           and math.isnan(pr["acc"])):
                continue
            if pr["acc"] < 0.7:
                continue

            probe = pr["probe"]
            w = probe.linear.weight.data.squeeze()
            bias = probe.linear.bias.data.item()
            w_norm_sq = w.dot(w).item()

            s_modified = states[t_surg].clone()

            logits_k = probe(states[t_surg][:, k, :])
            pred_co = (logits_k > 0).long()
            actual_co = carry_out[:, k]
            correct_mask = pred_co == actual_co
            n_correct = correct_mask.sum().item()

            if n_correct < 50:
                results.append({
                    "t_surgery": t_surg, "position": k,
                    "status": "too_few_correct", "n_correct": n_correct,
                })
                continue

            # Reflect state across carry_out probe boundary
            state_k = s_modified[:, k, :]
            proj = (state_k @ w + bias) / w_norm_sq
            s_modified[:, k, :] = state_k - 2 * proj.unsqueeze(1) * w.unsqueeze(0)

            s_final = run_from_state(model, s_modified, context, T - t_surg)

            logits_surg = model.readout_logits(s_final)
            preds_surg = logits_surg[:, :half, :].argmax(dim=-1)

            pos_changed = (preds_surg != preds_orig)
            cm = correct_mask

            per_pos_change_rate = []
            for p in range(half):
                rate = pos_changed[cm, p].float().mean().item()
                per_pos_change_rate.append(rate)

            # Immediate flip success
            with torch.no_grad():
                new_logits_k = probe(s_modified[:, k, :])
                new_pred = (new_logits_k > 0).long()
                flip_success = (new_pred != pred_co)[cm].float().mean().item()

            # Persistence: use step-T probe for final state (per Codex review)
            final_probe_key = (T, k)
            persist_rate = float("nan")
            if final_probe_key in co_probes:
                fp = co_probes[final_probe_key]
                if fp.get("probe") is not None and fp["acc"] > 0.6:
                    final_logits_k = fp["probe"](s_final[:, k, :])
                    final_pred = (final_logits_k > 0).long()
                    orig_pred = fp["probe"](states[T][:, k, :])
                    orig_class = (orig_pred > 0).long()
                    persist_rate = (final_pred != orig_class)[cm].float().mean().item()

            # Identify boundary examples where carry propagates leftward
            if k > 0:
                ci_at_km1 = carry_out[:, k]  # carry_out[k] = carry_in[k-1]
                boundary = (ci_at_km1 == 1) & correct_mask
                n_boundary = boundary.sum().item()
                boundary_change = pos_changed[boundary, k - 1].float().mean().item() \
                    if n_boundary > 10 else float("nan")
            else:
                n_boundary = 0
                boundary_change = float("nan")

            results.append({
                "t_surgery": t_surg,
                "position": k,
                "n_correct": n_correct,
                "probe_acc": pr["acc"],
                "flip_success_immediate": flip_success,
                "flip_persistence_final": persist_rate,
                "per_pos_output_change_rate": per_pos_change_rate,
                "n_boundary_examples": n_boundary,
                "boundary_leftward_change": boundary_change,
            })

    return results


# ========== MAIN ==========

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(seed=SEED)
    T = config["T"]
    half = config["seq_len"] // 2
    V = config["vocab_size"]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Causal carry probing: does UESD compute carry propagation?",
        "config": config,
    }

    # Separate probe and surgery data (per Codex review)
    set_seed(SEED + 9999)
    probe_src, probe_tgt = generate_batch("addition", N_PROBE,
                                          config["seq_len"], V)
    probe_src = probe_src.to(device)
    probe_tgt = probe_tgt.to(device)
    probe_ci, probe_co = compute_carries(probe_src, V)

    set_seed(SEED + 8888)
    surg_src, surg_tgt = generate_batch("addition", N_SURGERY,
                                        config["seq_len"], V)
    surg_src = surg_src.to(device)
    surg_tgt = surg_tgt.to(device)
    _, surg_co = compute_carries(surg_src, V)

    print(f"\nCarry distribution (probe set):", flush=True)
    for k in range(half):
        n_ci = probe_ci[:, k].sum().item()
        n_co = probe_co[:, k].sum().item()
        print(f"  Position {k}: carry_in={n_ci}/{N_PROBE} ({n_ci/N_PROBE:.1%}), "
              f"carry_out={n_co}/{N_PROBE} ({n_co/N_PROBE:.1%})", flush=True)

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  TRACK: {track}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # --- Train ---
        t0 = time.time()
        print(f"\n  Training {track}...", flush=True)
        model = train_model(track, config, device)
        train_time = time.time() - t0
        print(f"  Training complete in {train_time:.0f}s", flush=True)

        # Quick accuracy check
        with torch.no_grad():
            logits = model(probe_src, T)
            preds = logits[:, :half, :].argmax(dim=-1)
            targets = probe_tgt[:, :half]
            token_acc = (preds == targets).float().mean().item()
            seq_acc = (preds == targets).all(dim=1).float().mean().item()
        print(f"  Token acc: {token_acc:.4f}, Seq acc: {seq_acc:.4f}", flush=True)

        # --- Collect states on probe data ---
        print(f"\n  Collecting states at all {T+1} steps...", flush=True)
        states, context = collect_all_states(model, probe_src, T)

        # ===== PHASE 1: CARRY PROBES (both carry_in and carry_out) =====
        print(f"\n  --- PHASE 1: Carry Probes ---", flush=True)
        probes = train_carry_probes(states, probe_ci, probe_co, T, half,
                                    config["d_model"], device)

        probe_grids = {}
        for label_name in ["carry_in", "carry_out"]:
            print(f"\n  {label_name} probe accuracy (step × position):", flush=True)
            print(f"  {'Step':>6s}", end="", flush=True)
            for k in range(half):
                print(f"  pos_{k:d}", end="", flush=True)
            print(flush=True)

            acc_grid = []
            for t in range(T + 1):
                row = []
                print(f"  {t:6d}", end="", flush=True)
                for k in range(half):
                    pr = probes[label_name][(t, k)]
                    acc = pr["acc"]
                    row.append(acc)
                    if math.isnan(acc):
                        print(f"    N/A", end="", flush=True)
                    else:
                        print(f"  {acc:.3f}", end="", flush=True)
                print(flush=True)
                acc_grid.append(row)
            probe_grids[label_name] = acc_grid

            # First step where probe >= 80%
            first_acc = []
            for k in range(half):
                found = -1
                for t in range(T + 1):
                    if not math.isnan(acc_grid[t][k]) and acc_grid[t][k] >= 0.80:
                        found = t
                        break
                first_acc.append(found)
            print(f"  First step >= 80%: {first_acc}", flush=True)
            if label_name == "carry_in":
                print(f"  (Expect wavefront: pos2 → pos1 → pos0; pos3 = N/A)",
                      flush=True)
            else:
                print(f"  (Expect wavefront: pos3 → pos2 → pos1)", flush=True)

        # ===== PHASE 2: CARRY-FLIP PERTURBATION WITH CONTROLS =====
        print(f"\n  --- PHASE 2: Carry-Flip Perturbation (with controls) ---",
              flush=True)

        perturbation_results = {}
        for flip_k in range(1, half):  # carry_out at positions 1..3
            print(f"\n  Flipping carry_out at position {flip_k}:", flush=True)
            src_base, src_flip, src_ctrl, valid = generate_carry_flip_pairs(
                N_PROBE // 2, config["seq_len"], V, flip_k
            )
            valid_mask = valid
            src_b = src_base[valid_mask].to(device)
            src_f = src_flip[valid_mask].to(device)
            src_c = src_ctrl[valid_mask].to(device)
            n_valid = src_b.shape[0]
            print(f"    Valid pairs: {n_valid}", flush=True)

            if n_valid < 100:
                print(f"    Too few valid pairs, skipping", flush=True)
                continue

            div_flip, div_ctrl, base_norm = measure_perturbation_divergence(
                model, src_b, src_f, src_c, T
            )

            print(f"    Carry-flip vs Control divergence (normalized):", flush=True)
            print(f"    {'Step':>6s}", end="", flush=True)
            for p in range(half):
                print(f"  flip_{p:d} ctrl_{p:d}", end="", flush=True)
            print(flush=True)

            dn_f = div_flip / max(base_norm, 1e-8)
            dn_c = div_ctrl / max(base_norm, 1e-8)
            for t in range(T + 1):
                print(f"    {t:6d}", end="", flush=True)
                for p in range(half):
                    print(f"  {dn_f[t, p]:.3f} {dn_c[t, p]:.3f}", end="", flush=True)
                print(flush=True)

            perturbation_results[flip_k] = {
                "n_valid": n_valid,
                "div_flip": div_flip.tolist(),
                "div_ctrl": div_ctrl.tolist(),
                "div_flip_norm": dn_f.tolist(),
                "div_ctrl_norm": dn_c.tolist(),
                "base_norm": base_norm,
            }

        # ===== PHASE 3: STATE SURGERY (on held-out data) =====
        print(f"\n  --- PHASE 3: State Surgery (held-out data) ---", flush=True)

        surgery_results = state_surgery_experiment(
            model, surg_src, surg_tgt, surg_co, probes,
            T, V, T_SURGERY,
        )

        for sr in surgery_results:
            if sr.get("status") == "too_few_correct":
                print(f"    t={sr['t_surgery']} pos={sr['position']}: "
                      f"too few correct ({sr['n_correct']})", flush=True)
                continue

            persist = sr['flip_persistence_final']
            persist_str = f"{persist:.3f}" if not math.isnan(persist) else "N/A"
            print(f"\n    t={sr['t_surgery']} pos={sr['position']} "
                  f"(probe_acc={sr['probe_acc']:.3f}, n={sr['n_correct']}):",
                  flush=True)
            print(f"      Flip success (immediate): {sr['flip_success_immediate']:.3f}",
                  flush=True)
            print(f"      Flip persistence (final):  {persist_str}", flush=True)
            print(f"      Output change per pos: {sr['per_pos_output_change_rate']}",
                  flush=True)
            if sr.get("n_boundary_examples", 0) > 10:
                print(f"      Boundary leftward change: "
                      f"{sr['boundary_leftward_change']:.3f} "
                      f"(n={sr['n_boundary_examples']})", flush=True)

        # ===== Store results =====
        track_result = {
            "track": track,
            "train_time_s": train_time,
            "token_acc": token_acc,
            "seq_acc": seq_acc,
            "probe_carry_in": probe_grids["carry_in"],
            "probe_carry_out": probe_grids["carry_out"],
            "perturbation": perturbation_results,
            "surgery": surgery_results,
        }
        all_results[track] = track_result

    # ===== SUMMARY =====
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D8 SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for track in ["dynamics_ce", "e5"]:
        tr = all_results[track]
        print(f"\n  {track}:", flush=True)
        print(f"    Accuracy: token={tr['token_acc']:.4f} seq={tr['seq_acc']:.4f}",
              flush=True)

        surg = tr["surgery"]
        valid_surg = [s for s in surg if "flip_success_immediate" in s]
        if valid_surg:
            avg_flip = sum(s["flip_success_immediate"] for s in valid_surg) / len(valid_surg)
            persist_vals = [s["flip_persistence_final"] for s in valid_surg
                           if not math.isnan(s["flip_persistence_final"])]
            avg_persist = sum(persist_vals) / len(persist_vals) if persist_vals else float("nan")
            persist_str = f"{avg_persist:.3f}" if not math.isnan(avg_persist) else "N/A"
            print(f"    Surgery: avg flip success={avg_flip:.3f}, "
                  f"avg persistence={persist_str}", flush=True)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d8_causal_carry_probing.json"

    # Clean up non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, (nn.Module, torch.Tensor))}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    with open(out_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
