"""
Experiment D17: Reconsideration Capacity

Direct test of UESD Axiom A3 (Single Error Principle): if the dynamics
minimize E(s) = ||F_theta(s,c)||^2, then injecting a WRONG intermediate
state should trigger self-correction -- the dynamics should "reconsider"
and converge back to the correct answer using the encoded context.

This is distinct from:
- D11 (basin radius): D11 perturbs INITIAL states; D17 corrupts CONVERGED states
- D8 (carry surgery): D8 flips one carry bit; D17 injects wholesale wrong answers
- D7 (thinking emergence): D7 measures progressive computation; D17 tests error recovery

Three phases:

Phase 1 -- Answer Injection:
  At the converged state s_T, replace position k's readout with a WRONG
  token's embedding (projected through readout_proj inverse). Run K additional
  dynamics steps. Measure recovery rate and speed.

Phase 2 -- Cross-Example Transplant:
  At step t_mid, swap position k's state vector between two examples that
  have DIFFERENT correct answers at position k. Run remaining dynamics.
  If dynamics recover -> computation is context-conditioned (genuine thinking).
  If dynamics follow transplanted state -> trajectory is memorized.

Phase 3 -- Error-Correcting Capacity:
  Simultaneously corrupt 1, 2, 3, all 4 result positions. Measure recovery
  as a function of corruption count. This quantifies the "error-correcting
  code" capacity of the dynamics -- analogous to Hamming distance in coding.

PREDICTIONS:
1. E5 recovers from single-position corruption at >80% rate (SC loss
   creates attractor basins around correct fixed points)
2. CE-dynamics recovers at lower rate (~50-60%) -- less structured basins
3. Recovery rate drops sharply with corruption count (error-correcting
   capacity is finite, probably ~1-2 positions)
4. Recovery is context-conditioned: cross-example transplant recovers to
   the CONTEXT's correct answer, not the transplanted state's answer
5. Recovery speed correlates with E5's self-consistency energy at the
   corrupted state -- higher energy = faster correction

If prediction 4 holds, it directly demonstrates that UESD dynamics perform
genuine input-conditioned computation, not trajectory memorization.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch


N_EVAL = 4096
K_EXTRA_STEPS = 20  # additional dynamics steps after corruption
SEED = 42


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
    set_seed(config["seed"])
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]
    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        s = model.init_state(B, L, src.device)
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

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def collect_states_and_context(model, src, T):
    context = model.encode(src)
    B, L = src.shape
    s = model.init_state(B, L, src.device)
    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())
    return states, context


@torch.no_grad()
def run_from_state(model, s, context, n_steps):
    for _ in range(n_steps):
        s, _ = model.dynamics_step(s, context)
    return s


@torch.no_grad()
def compute_energy(model, s, context):
    s_next = model.dynamics(s, context)
    residual = s_next - s
    return (residual ** 2).sum(dim=-1).mean(dim=-1)  # [B]


@torch.no_grad()
def get_predictions(model, s, half):
    logits = model.readout_logits(s)
    return logits[:, :half, :].argmax(dim=-1)


# ========== PHASE 1: Answer Injection ==========

@torch.no_grad()
def corrupt_state_at_position(model, s, pos, wrong_token_ids, device):
    """Replace state at position pos with embedding of wrong tokens.

    Uses tok_emb to get the token embedding, then projects through
    readout_proj (pseudo-inverse) to get the state-space representation.
    This is a principled corruption: we inject the embedding of a wrong
    answer into the dynamics state.
    """
    s_corrupted = s.clone()
    wrong_emb = model.tok_emb(wrong_token_ids)  # [B, d]
    s_corrupted[:, pos, :] = wrong_emb
    return s_corrupted


@torch.no_grad()
def phase1_answer_injection(model, eval_src, eval_tgt, config, device):
    """Inject wrong answers at converged state, test recovery."""
    T = config["T"]
    V = config["vocab_size"]
    half = config["seq_len"] // 2

    states, context = collect_states_and_context(model, eval_src, T)
    s_converged = states[T]
    preds_orig = get_predictions(model, s_converged, half)
    targets = eval_tgt[:, :half]
    correct_mask = (preds_orig == targets).all(dim=1)  # fully correct examples
    n_correct = correct_mask.sum().item()

    print(f"    Fully correct examples: {n_correct}/{eval_src.shape[0]}", flush=True)

    results = {}

    for corrupt_pos in range(half):
        print(f"\n    Corrupting position {corrupt_pos}:", flush=True)

        # Generate wrong tokens (shift by random offset 1..V-1)
        B = eval_src.shape[0]
        offset = torch.randint(1, V, (B,), device=device)
        correct_tokens = targets[:, corrupt_pos]
        wrong_tokens = (correct_tokens + offset) % V

        s_corrupted = corrupt_state_at_position(
            model, s_converged, corrupt_pos, wrong_tokens, device
        )

        # Check immediate prediction after corruption
        preds_corrupted = get_predictions(model, s_corrupted, half)
        immediate_wrong = (preds_corrupted[:, corrupt_pos] != targets[:, corrupt_pos])

        # Energy at corrupted state
        energy_clean = compute_energy(model, s_converged, context)
        energy_corrupted = compute_energy(model, s_corrupted, context)

        # Recovery: run additional dynamics steps
        recovery_by_step = []
        s_recovering = s_corrupted.clone()
        for k in range(1, K_EXTRA_STEPS + 1):
            s_recovering, _ = model.dynamics_step(s_recovering, context)
            preds_k = get_predictions(model, s_recovering, half)

            pos_recovered = (preds_k[:, corrupt_pos] == targets[:, corrupt_pos])
            all_recovered = (preds_k == targets).all(dim=1)

            pos_recovery_rate = pos_recovered[correct_mask].float().mean().item()
            full_recovery_rate = all_recovered[correct_mask].float().mean().item()

            # Also check OTHER positions haven't been disrupted
            other_intact = True
            for p in range(half):
                if p == corrupt_pos:
                    continue
                other_ok = (preds_k[:, p] == preds_orig[:, p])
                if other_ok[correct_mask].float().mean().item() < 0.95:
                    other_intact = False

            recovery_by_step.append({
                "extra_step": k,
                "pos_recovery": pos_recovery_rate,
                "full_recovery": full_recovery_rate,
                "other_intact": other_intact,
            })

            if k in [1, 2, 5, 10, 20]:
                print(f"      +{k} steps: pos_recovery={pos_recovery_rate:.3f}, "
                      f"full_recovery={full_recovery_rate:.3f}, "
                      f"other_intact={other_intact}", flush=True)

        # First step where position recovers (for >50% of examples)
        first_recovery = K_EXTRA_STEPS + 1
        for r in recovery_by_step:
            if r["pos_recovery"] > 0.50:
                first_recovery = r["extra_step"]
                break

        results[corrupt_pos] = {
            "immediate_wrong_rate": immediate_wrong[correct_mask].float().mean().item(),
            "energy_clean_mean": energy_clean[correct_mask].mean().item(),
            "energy_corrupted_mean": energy_corrupted[correct_mask].mean().item(),
            "energy_ratio": (energy_corrupted[correct_mask].mean() /
                           max(energy_clean[correct_mask].mean(), 1e-10)).item(),
            "first_recovery_step": first_recovery,
            "recovery_trajectory": recovery_by_step,
        }

    return results


# ========== PHASE 2: Cross-Example Transplant ==========

@torch.no_grad()
def phase2_cross_transplant(model, eval_src, eval_tgt, config, device):
    """Swap position k's state between two examples with different answers."""
    T = config["T"]
    half = config["seq_len"] // 2

    states, context = collect_states_and_context(model, eval_src, T)
    s_converged = states[T]
    preds_orig = get_predictions(model, s_converged, half)
    targets = eval_tgt[:, :half]
    correct_mask = (preds_orig == targets).all(dim=1)

    B = eval_src.shape[0]
    results = {}

    for transplant_pos in range(half):
        # Find pairs: examples i,j where both are correct but have
        # different answers at transplant_pos
        correct_idx = correct_mask.nonzero(as_tuple=True)[0]
        answers_at_pos = targets[correct_idx, transplant_pos]

        # Pair up: for each example, find one with different answer
        n_pairs = 0
        recovery_to_context = 0
        recovery_to_donor = 0
        recovery_to_neither = 0
        total_tested = 0

        # Efficient pairing: group by answer, cross-match
        max_pairs = min(1024, len(correct_idx) // 2)
        used = set()
        pairs = []
        for i in range(len(correct_idx)):
            if i in used or len(pairs) >= max_pairs:
                break
            for j in range(i + 1, len(correct_idx)):
                if j in used:
                    continue
                if answers_at_pos[i] != answers_at_pos[j]:
                    pairs.append((correct_idx[i].item(), correct_idx[j].item()))
                    used.add(i)
                    used.add(j)
                    break

        if len(pairs) < 50:
            print(f"    Position {transplant_pos}: too few pairs ({len(pairs)})",
                  flush=True)
            results[transplant_pos] = {"status": "too_few_pairs", "n_pairs": len(pairs)}
            continue

        # Test transplant for t_mid values
        for t_mid in [3, 5, 7, T]:
            t_label = f"t={t_mid}"
            s_at_t = states[t_mid] if t_mid < T else s_converged

            context_wins = 0
            donor_wins = 0
            neither_wins = 0
            pair_count = 0

            for idx_a, idx_b in pairs:
                # Transplant: inject B's state at pos into A's trajectory
                s_a = s_at_t[idx_a:idx_a+1].clone()
                s_b = s_at_t[idx_b:idx_b+1].clone()
                ctx_a = context[idx_a:idx_a+1]

                s_transplanted = s_a.clone()
                s_transplanted[:, transplant_pos, :] = s_b[:, transplant_pos, :]

                remaining = T - t_mid + K_EXTRA_STEPS
                s_final = run_from_state(model, s_transplanted, ctx_a, remaining)

                pred_final = get_predictions(model, s_final, half)
                context_answer = targets[idx_a, transplant_pos].item()
                donor_answer = targets[idx_b, transplant_pos].item()
                actual_pred = pred_final[0, transplant_pos].item()

                if actual_pred == context_answer:
                    context_wins += 1
                elif actual_pred == donor_answer:
                    donor_wins += 1
                else:
                    neither_wins += 1
                pair_count += 1

            total = pair_count
            results_key = f"pos{transplant_pos}_{t_label}"
            results[results_key] = {
                "n_pairs": pair_count,
                "context_wins": context_wins / total if total > 0 else 0,
                "donor_wins": donor_wins / total if total > 0 else 0,
                "neither_wins": neither_wins / total if total > 0 else 0,
            }

            print(f"    Position {transplant_pos} {t_label}: "
                  f"context={context_wins/total:.3f} "
                  f"donor={donor_wins/total:.3f} "
                  f"neither={neither_wins/total:.3f} "
                  f"(n={pair_count})", flush=True)

    return results


# ========== PHASE 3: Error-Correcting Capacity ==========

@torch.no_grad()
def phase3_error_correcting_capacity(model, eval_src, eval_tgt, config, device):
    """Simultaneously corrupt 1, 2, 3, 4 positions. Measure recovery."""
    T = config["T"]
    V = config["vocab_size"]
    half = config["seq_len"] // 2

    states, context = collect_states_and_context(model, eval_src, T)
    s_converged = states[T]
    preds_orig = get_predictions(model, s_converged, half)
    targets = eval_tgt[:, :half]
    correct_mask = (preds_orig == targets).all(dim=1)
    B = eval_src.shape[0]

    results = {}

    for n_corrupt in range(1, half + 1):
        print(f"\n    Corrupting {n_corrupt}/{half} positions:", flush=True)

        # Corrupt the first n_corrupt positions (0, 1, ..., n_corrupt-1)
        # This is a systematic sweep, not random
        corrupt_positions = list(range(n_corrupt))

        s_corrupted = s_converged.clone()
        for pos in corrupt_positions:
            offset = torch.randint(1, V, (B,), device=device)
            wrong_tokens = (targets[:, pos] + offset) % V
            s_corrupted = corrupt_state_at_position(
                model, s_corrupted, pos, wrong_tokens, device
            )

        # Energy after corruption
        energy_corrupted = compute_energy(model, s_corrupted, context)
        energy_clean = compute_energy(model, s_converged, context)

        # Recovery trajectory
        recovery_curve = []
        s_recovering = s_corrupted.clone()
        for k in range(1, K_EXTRA_STEPS + 1):
            s_recovering, _ = model.dynamics_step(s_recovering, context)
            preds_k = get_predictions(model, s_recovering, half)

            per_pos_recovery = []
            for p in range(half):
                pos_ok = (preds_k[:, p] == targets[:, p])
                per_pos_recovery.append(
                    pos_ok[correct_mask].float().mean().item()
                )

            full_recovery = (preds_k == targets).all(dim=1)
            full_rate = full_recovery[correct_mask].float().mean().item()

            recovery_curve.append({
                "extra_step": k,
                "per_pos_recovery": per_pos_recovery,
                "full_recovery": full_rate,
            })

            if k in [1, 2, 5, 10, 20]:
                print(f"      +{k}: per_pos={[f'{r:.3f}' for r in per_pos_recovery]}, "
                      f"full={full_rate:.3f}", flush=True)

        # Also try corrupting RANDOM subsets of n_corrupt positions
        # (average over 10 random subsets)
        if n_corrupt < half:
            random_subset_rates = []
            for trial in range(10):
                perm = torch.randperm(half)[:n_corrupt].tolist()
                s_rand_corrupt = s_converged.clone()
                for pos in perm:
                    offset = torch.randint(1, V, (B,), device=device)
                    wrong_tokens = (targets[:, pos] + offset) % V
                    s_rand_corrupt = corrupt_state_at_position(
                        model, s_rand_corrupt, pos, wrong_tokens, device
                    )

                s_final = run_from_state(model, s_rand_corrupt, context, K_EXTRA_STEPS)
                preds_final = get_predictions(model, s_final, half)
                full_ok = (preds_final == targets).all(dim=1)
                random_subset_rates.append(
                    full_ok[correct_mask].float().mean().item()
                )

            mean_random = float(np.mean(random_subset_rates))
            std_random = float(np.std(random_subset_rates))
        else:
            mean_random = recovery_curve[-1]["full_recovery"]
            std_random = 0.0

        results[n_corrupt] = {
            "n_corrupted": n_corrupt,
            "corrupt_positions": corrupt_positions,
            "energy_ratio": (energy_corrupted[correct_mask].mean() /
                           max(energy_clean[correct_mask].mean(), 1e-10)).item(),
            "recovery_curve": recovery_curve,
            "final_full_recovery": recovery_curve[-1]["full_recovery"],
            "random_subset_recovery_mean": mean_random,
            "random_subset_recovery_std": std_random,
        }

    return results


# ========== MAIN ==========

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(seed=SEED)
    T = config["T"]
    V = config["vocab_size"]
    half = config["seq_len"] // 2

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Reconsideration capacity: can UESD dynamics self-correct from wrong states?",
        "config": config,
        "K_extra_steps": K_EXTRA_STEPS,
    }

    # Generate eval data (same seed as D7/D16 for cross-comparison)
    set_seed(999)
    eval_src, eval_tgt = generate_batch("addition", N_EVAL,
                                        config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  TRACK: {track}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()
        model = train_model(track, config, device)
        train_time = time.time() - t0
        print(f"  Training: {train_time:.0f}s", flush=True)

        # Baseline accuracy
        with torch.no_grad():
            logits = model(eval_src, T)
            preds = logits[:, :half, :].argmax(dim=-1)
            targets = eval_tgt[:, :half]
            token_acc = (preds == targets).float().mean().item()
            seq_acc = (preds == targets).all(dim=1).float().mean().item()
        print(f"  Baseline: token={token_acc:.4f}, seq={seq_acc:.4f}", flush=True)

        # Phase 1: Answer Injection
        print(f"\n  --- PHASE 1: Answer Injection ---", flush=True)
        injection_results = phase1_answer_injection(
            model, eval_src, eval_tgt, config, device
        )

        # Phase 2: Cross-Example Transplant
        print(f"\n  --- PHASE 2: Cross-Example Transplant ---", flush=True)
        transplant_results = phase2_cross_transplant(
            model, eval_src, eval_tgt, config, device
        )

        # Phase 3: Error-Correcting Capacity
        print(f"\n  --- PHASE 3: Error-Correcting Capacity ---", flush=True)
        ecc_results = phase3_error_correcting_capacity(
            model, eval_src, eval_tgt, config, device
        )

        # Summary
        print(f"\n  === {track} SUMMARY ===", flush=True)

        # Phase 1 summary
        for pos in range(half):
            if pos in injection_results:
                r = injection_results[pos]
                print(f"    Injection pos {pos}: "
                      f"energy_ratio={r['energy_ratio']:.1f}x, "
                      f"first_recovery=+{r['first_recovery_step']} steps",
                      flush=True)

        # Phase 2 summary
        for key, r in transplant_results.items():
            if isinstance(r, dict) and "context_wins" in r:
                print(f"    Transplant {key}: "
                      f"context={r['context_wins']:.3f} "
                      f"donor={r['donor_wins']:.3f}",
                      flush=True)

        # Phase 3 summary
        print(f"    Error-correcting capacity:", flush=True)
        for nc in range(1, half + 1):
            if nc in ecc_results:
                r = ecc_results[nc]
                print(f"      {nc}/{half} corrupted: "
                      f"recovery={r['final_full_recovery']:.3f} "
                      f"(random subset: {r['random_subset_recovery_mean']:.3f}+/-"
                      f"{r['random_subset_recovery_std']:.3f})",
                      flush=True)

        all_results[track] = {
            "train_time_s": train_time,
            "token_acc": token_acc,
            "seq_acc": seq_acc,
            "phase1_injection": {str(k): v for k, v in injection_results.items()},
            "phase2_transplant": transplant_results,
            "phase3_ecc": {str(k): v for k, v in ecc_results.items()},
        }

        del model
        torch.cuda.empty_cache()

    # Cross-track comparison
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D17 RECONSIDERATION CAPACITY COMPARISON", flush=True)
    print(f"{'=' * 60}", flush=True)

    for key in ["phase1_injection", "phase3_ecc"]:
        print(f"\n  {key}:", flush=True)
        for track in ["dynamics_ce", "e5"]:
            tr = all_results[track][key]
            if key == "phase1_injection":
                recoveries = [v.get("first_recovery_step", 999)
                              for v in tr.values()
                              if isinstance(v, dict) and "first_recovery_step" in v]
                if recoveries:
                    print(f"    {track}: mean first recovery = "
                          f"+{np.mean(recoveries):.1f} steps", flush=True)
            elif key == "phase3_ecc":
                for nc_str, v in tr.items():
                    if isinstance(v, dict) and "final_full_recovery" in v:
                        print(f"    {track} {nc_str}-corrupt: "
                              f"{v['final_full_recovery']:.3f}", flush=True)

    # Phase 2: context vs donor wins comparison
    print(f"\n  Cross-example transplant (context-conditioned?):", flush=True)
    for track in ["dynamics_ce", "e5"]:
        tr = all_results[track]["phase2_transplant"]
        context_rates = [v["context_wins"] for v in tr.values()
                        if isinstance(v, dict) and "context_wins" in v]
        donor_rates = [v["donor_wins"] for v in tr.values()
                      if isinstance(v, dict) and "donor_wins" in v]
        if context_rates:
            print(f"    {track}: context={np.mean(context_rates):.3f}, "
                  f"donor={np.mean(donor_rates):.3f} "
                  f"-> {'CONTEXT-CONDITIONED' if np.mean(context_rates) > np.mean(donor_rates) else 'STATE-MEMORIZED'}",
                  flush=True)

    print(f"{'=' * 60}", flush=True)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d17_reconsideration.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
