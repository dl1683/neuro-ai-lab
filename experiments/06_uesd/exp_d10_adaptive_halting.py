"""
Experiment D10: Adaptive Halting — Learning When to Stop Thinking

The strongest test of the "thinking" hypothesis: if UESD dynamics
truly compute, the model should learn to ALLOCATE COMPUTATION
ADAPTIVELY. Easy inputs (no carry chains) should halt early;
hard inputs (long carry chains) should use more steps.

Architecture: UESD + PonderNet-style halting module
  - At each step t, a halting head predicts h_t = P(halt at step t)
  - Output is the expectation: E[readout(s_tau)]
  - Loss: CE(expected_output, target) + beta * KL(halt_dist || geometric_prior)

Three variants:
1. UESD-HALT: CE-dynamics + learned halting (PonderNet formulation)
2. UESD-THRESH: CE-dynamics + fixed threshold on ||s_{t+1} - s_t||
3. UESD-FIXED: CE-dynamics with fixed T=10 (baseline)

Key measurements:
- Average halting step per carry-chain-length group
- Per-position halting step (do rightmost positions decide first?)
- Accuracy vs. mean steps used (compute efficiency)
- Correlation between halting step and first-correct-step from D7

PREDICTIONS:
1. UESD-HALT learns to halt at T ≈ max_carry_chain_length + 1.
   No-carry examples halt at T=2-3. Max-chain examples halt at T=5-8.
2. The halting step distribution should be BIMODAL or have clear modes
   matching carry chain lengths {0, 1, 2, 3, 4}.
3. UESD-HALT should achieve the SAME accuracy as UESD-FIXED
   with 40-60% fewer average steps.
4. The beta parameter (KL penalty) controls speed-accuracy tradeoff:
   high beta = early halting (fast but less accurate),
   low beta = late halting (slow but more accurate).

If prediction 1 holds, it proves the dynamics are adaptive computation,
not just fixed-depth iteration.
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


SEED = 42
T_MAX = 15
T_BASELINE = 10  # fixed-T baseline uses standard T=10
LAMBDA_P = 0.1  # geometric prior parameter (0.01 was too flat)
BETA_VALUES = [0.01, 0.1, 1.0]


def build_config(seed=42):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": T_MAX,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 25000,
        "seed": seed,
    }


class HaltingHead(nn.Module):
    """Predicts halting probability from state."""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, s):
        # s: [B, L, d_model] -> [B] halting probability
        pooled = s.mean(dim=1)
        return torch.sigmoid(self.linear(pooled)).squeeze(-1)


def compute_carries(src, vocab_size):
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
    # Carry chain length at each position
    chain_len = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 2, -1, -1):
        has_carry = carry_in[:, i] == 1
        chain_len[has_carry, i] = 1 + chain_len[has_carry, i + 1]
    # Total carry-ins per example
    total_carries = carry_in.sum(dim=1)
    max_chain = chain_len.max(dim=1).values
    return carry_in, total_carries, max_chain


def geometric_prior(T, lam=LAMBDA_P):
    """Geometric distribution: p(t) = (1-lam)^(t-1) * lam, normalized over 1..T."""
    steps = torch.arange(1, T + 1, dtype=torch.float)
    log_probs = (steps - 1) * math.log(1 - lam) + math.log(lam)
    return F.softmax(log_probs, dim=0)


def train_adaptive(model, halting_head, config, device, beta=0.1):
    """Train UESD with PonderNet-style adaptive halting."""
    model.train()
    halting_head.train()

    all_params = list(model.parameters()) + list(halting_head.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config["lr"])
    T = config["T"]
    total = config["training_steps"]
    prior = geometric_prior(T, LAMBDA_P).to(device)

    for step in range(1, total + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        half = L // 2
        s = model.init_state(B, L, device)

        # Collect outputs and halting probs at each step
        halt_probs = []
        logits_list = []

        for t in range(T):
            s, _ = model.dynamics_step(s, context)
            # halting_head is evaluated at every step, including t = T-1 where the
            # PonderNet distribution below overrides it with running_prob (forced
            # halt). The redundant final call is retained so halt_probs traces are
            # uniform length T for the logged diagnostics.
            h_t = halting_head(s)  # [B]
            halt_probs.append(h_t)
            logits_t = model.readout_logits(s)
            logits_list.append(logits_t)

        # Compute halting distribution (PonderNet)
        # p(halt at t) = h_t * prod_{i<t}(1 - h_i)
        running_prob = torch.ones(B, device=device)
        halt_dist = []
        for t in range(T):
            if t == T - 1:
                p_t = running_prob  # must halt at last step
            else:
                p_t = halt_probs[t] * running_prob
                running_prob = running_prob * (1 - halt_probs[t])
            halt_dist.append(p_t)

        halt_dist = torch.stack(halt_dist, dim=1)  # [B, T]
        halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Weighted CE loss (only on result positions, not padding)
        ce_loss = torch.zeros(1, device=device)
        for t in range(T):
            logits_t = logits_list[t][:, :half, :]  # result positions only
            tgt_result = tgt[:, :half]
            ce_t = F.cross_entropy(
                logits_t.reshape(-1, logits_t.size(-1)),
                tgt_result.reshape(-1), reduction='none'
            ).reshape(B, half)
            ce_per_example = ce_t.mean(dim=1)  # [B]
            ce_loss = ce_loss + (halt_dist[:, t] * ce_per_example).mean()

        # Per-example KL(halt_dist || prior) — correct direction
        log_halt = halt_dist.clamp(min=1e-8).log()  # [B, T]
        log_prior = prior.log().unsqueeze(0).expand(B, -1)  # [B, T]
        kl_loss = (halt_dist * (log_halt - log_prior)).sum(dim=1).mean()

        loss = ce_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            mean_halt = (halt_dist * torch.arange(1, T + 1, device=device,
                         dtype=torch.float).unsqueeze(0)).sum(dim=1).mean().item()
            print(f"    Step {step:>6d}/{total} | Loss: {loss.item():.4f} "
                  f"| CE: {ce_loss.item():.4f} | KL: {kl_loss.item():.4f} "
                  f"| mean_halt: {mean_halt:.1f}", flush=True)

    model.eval()
    halting_head.eval()
    return model, halting_head


def train_fixed(model, config, device):
    """Train standard CE-dynamics with fixed T (baseline)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]
    total = config["training_steps"]

    for step in range(1, total + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)
        half = config["seq_len"] // 2
        logits = model(src, T)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]
        loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)), tgt_r.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{total} | Loss: {loss.item():.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def evaluate_adaptive(model, halting_head, src, tgt, T, vocab_size):
    """Evaluate adaptive model: accuracy + halting distribution."""
    model.eval()
    halting_head.eval()

    B, L = src.shape
    half = L // 2
    context = model.encode(src)
    s = model.init_state(B, L, src.device)

    halt_probs = []
    per_step_preds = []

    for t in range(T):
        s, _ = model.dynamics_step(s, context)
        # Final-step halting_head call is overridden by the forced halt below;
        # retained so halt_probs traces are uniform length T (see training loop).
        h_t = halting_head(s)
        halt_probs.append(h_t)
        logits_t = model.readout_logits(s)
        preds_t = logits_t[:, :half, :].argmax(dim=-1)
        per_step_preds.append(preds_t)

    # Compute expected halting step per example
    running_prob = torch.ones(B, device=src.device)
    halt_dist = []
    for t in range(T):
        if t == T - 1:
            p_t = running_prob
        else:
            p_t = halt_probs[t] * running_prob
            running_prob = running_prob * (1 - halt_probs[t])
        halt_dist.append(p_t)

    halt_dist = torch.stack(halt_dist, dim=1)
    halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)

    expected_halt = (halt_dist * torch.arange(1, T + 1, device=src.device,
                     dtype=torch.float).unsqueeze(0)).sum(dim=1)
    mode_halt = halt_dist.argmax(dim=1) + 1

    # Greedy accuracy: use readout at the modal halting step
    targets = tgt[:, :half]
    greedy_preds = torch.zeros_like(targets)
    for i in range(B):
        greedy_preds[i] = per_step_preds[mode_halt[i].item() - 1][i]

    tok_acc = (greedy_preds == targets).float().mean().item()
    seq_acc = (greedy_preds == targets).all(dim=1).float().mean().item()

    # Full T accuracy
    full_preds = per_step_preds[-1]
    full_tok_acc = (full_preds == targets).float().mean().item()

    return {
        "tok_acc_greedy": tok_acc,
        "seq_acc_greedy": seq_acc,
        "tok_acc_full_T": full_tok_acc,
        "mean_halt_step": expected_halt.mean().item(),
        "median_halt_step": expected_halt.median().item(),
        "halt_distribution": halt_dist.mean(dim=0).cpu().tolist(),
        "expected_halt_per_example": expected_halt.cpu(),
        "mode_halt_per_example": mode_halt.cpu(),
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(seed=SEED)
    half = config["seq_len"] // 2
    V = config["vocab_size"]
    T = config["T"]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Adaptive halting: does UESD learn to allocate computation?",
        "config": config,
    }

    # Generate eval data
    set_seed(SEED + 5555)
    eval_src, eval_tgt = generate_batch("addition", 4096,
                                        config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)
    _, total_carries, max_chain = compute_carries(eval_src, V)

    print(f"\nCarry chain distribution:", flush=True)
    for chain_len in range(half + 1):
        n = (max_chain == chain_len).sum().item()
        print(f"  max_chain={chain_len}: {n}/4096 ({n/4096:.1%})", flush=True)

    # === Baseline: Fixed T (uses standard T=10, NOT T_MAX) ===
    T_base = T_BASELINE
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Baseline: Fixed T={T_base} (adaptive uses T_MAX={T})", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(SEED)
    model_fixed = UESDModel(
        V, config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    print(f"  Params: {count_params(model_fixed)}", flush=True)

    # Train baseline with T=10 (standard config)
    baseline_config = {**config, "T": T_base}
    t0 = time.time()
    model_fixed = train_fixed(model_fixed, baseline_config, device)
    fixed_time = time.time() - t0

    with torch.no_grad():
        logits = model_fixed(eval_src, T_base)
        preds = logits[:, :half, :].argmax(dim=-1)
        targets = eval_tgt[:, :half]
        fixed_tok = (preds == targets).float().mean().item()
        fixed_seq = (preds == targets).all(dim=1).float().mean().item()

    print(f"  Fixed T={T_base}: tok={fixed_tok:.4f} seq={fixed_seq:.4f} "
          f"({fixed_time:.0f}s)", flush=True)

    all_results["baseline"] = {
        "tok_acc": fixed_tok, "seq_acc": fixed_seq,
        "train_time_s": fixed_time, "T": T_base,
    }

    # === Adaptive halting for each beta ===
    for beta in BETA_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Adaptive halting beta={beta}", flush=True)
        print(f"{'=' * 60}", flush=True)

        set_seed(SEED)
        model_adapt = UESDModel(
            V, config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        halt_head = HaltingHead(config["d_model"]).to(device)
        total_params = count_params(model_adapt) + count_params(halt_head)
        print(f"  Params: {total_params} (model={count_params(model_adapt)} "
              f"+ halt={count_params(halt_head)})", flush=True)

        t0 = time.time()
        model_adapt, halt_head = train_adaptive(
            model_adapt, halt_head, config, device, beta=beta
        )
        adapt_time = time.time() - t0

        eval_res = evaluate_adaptive(model_adapt, halt_head, eval_src, eval_tgt,
                                     T, V)

        print(f"\n  Results:", flush=True)
        print(f"    Greedy acc: tok={eval_res['tok_acc_greedy']:.4f} "
              f"seq={eval_res['seq_acc_greedy']:.4f}", flush=True)
        print(f"    Full-T acc: tok={eval_res['tok_acc_full_T']:.4f}", flush=True)
        print(f"    Mean halt step: {eval_res['mean_halt_step']:.2f}", flush=True)
        print(f"    Halt distribution: {[f'{p:.3f}' for p in eval_res['halt_distribution']]}",
              flush=True)

        # Per-chain-length analysis
        print(f"\n    Mean halt step by max carry chain:", flush=True)
        per_chain = {}
        for chain_len in range(half + 1):
            mask = max_chain == chain_len
            n = mask.sum().item()
            if n < 20:
                continue
            mean_halt = eval_res["expected_halt_per_example"][mask.cpu()].mean().item()
            per_chain[chain_len] = {"n": n, "mean_halt": mean_halt}
            print(f"      chain={chain_len}: mean_halt={mean_halt:.2f} (n={n})",
                  flush=True)

        all_results[f"beta_{beta}"] = {
            "tok_acc_greedy": eval_res["tok_acc_greedy"],
            "seq_acc_greedy": eval_res["seq_acc_greedy"],
            "tok_acc_full_T": eval_res["tok_acc_full_T"],
            "mean_halt_step": eval_res["mean_halt_step"],
            "halt_distribution": eval_res["halt_distribution"],
            "per_chain_length": per_chain,
            "train_time_s": adapt_time,
            "beta": beta,
        }

        del model_adapt, halt_head
        torch.cuda.empty_cache()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d10_adaptive_halting.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
