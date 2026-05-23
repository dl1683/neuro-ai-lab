"""
Experiment D9: Attention Flow Through Dynamics

Completely unexplored angle: what do the ATTENTION PATTERNS look like
inside the dynamics at each step? The TransformerDecoderLayer has:
  1. Self-attention: state positions attend to each other
  2. Cross-attention: state positions attend to encoder (input) positions

If carry propagation flows right-to-left through dynamics:
  - Cross-attention should shift from rightward input positions (early steps)
    to leftward input positions (late steps)
  - Self-attention should connect carry-dependent positions

If the dynamics process all positions simultaneously:
  - Attention patterns should be static across steps

This experiment also tests computational depth scaling by training on
8-digit addition (half=8, L=16). With T=10 and max carry chain=8,
the model has just enough steps. We can see if the attention wavefront
matches the longer carry chains.

PHASES:
1. Train CE-dynamics and E5 on 4-digit (L=8) and 8-digit (L=16) addition
2. Capture attention weights at each dynamics step
3. Analyze per-step, per-position attention patterns
4. Correlate with carry structure
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


N_EVAL = 2048
SEED = 42


def build_config(seed=42, seq_len=8):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "max_len": 32,
        "seq_len": seq_len,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "warmup_steps": 5000,
        "seed": seed,
    }


def compute_carry_in(src, vocab_size):
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
    return carry_in


def train_model(track, config, device):
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

        if step % 4000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{total} | Loss: {loss.item():.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def dynamics_step_with_attn(layer, s, context):
    """Replicate TransformerDecoderLayer forward (norm_first=True) with
    attention weight capture."""
    # Self-attention
    s_n1 = layer.norm1(s)
    sa_out, sa_w = layer.self_attn(
        s_n1, s_n1, s_n1, need_weights=True, average_attn_weights=True
    )
    s2 = s + sa_out

    # Cross-attention
    s_n2 = layer.norm2(s2)
    ca_out, ca_w = layer.multihead_attn(
        s_n2, context, context, need_weights=True, average_attn_weights=True
    )
    s3 = s2 + ca_out

    # FFN
    s_n3 = layer.norm3(s3)
    ff_out = layer.linear2(layer.activation(layer.linear1(s_n3)))
    s4 = s3 + ff_out

    return s4, sa_w, ca_w


@torch.no_grad()
def collect_attention_patterns(model, src, T):
    """Run dynamics and collect attention weights at each step."""
    model.eval()
    context = model.encode(src)
    B, L = src.shape
    s = model.init_state(B, L, src.device)

    self_attns = []
    cross_attns = []

    for t in range(T):
        s, sa_w, ca_w = dynamics_step_with_attn(model.dynamics, s, context)
        self_attns.append(sa_w.cpu())   # [B, L, L]
        cross_attns.append(ca_w.cpu())  # [B, L, L]

    return self_attns, cross_attns


def analyze_attention(self_attns, cross_attns, carry_in, half, T):
    """Analyze attention patterns for carry propagation structure."""
    B = self_attns[0].shape[0]
    results = {}

    # === Cross-attention analysis ===
    # For each result position k (0..half-1), compute which input positions
    # it attends to at each step. Input positions: even indices = a_i, odd = b_i
    cross_focus = torch.zeros(T, half)  # mean attended input position
    cross_right_mass = torch.zeros(T, half)  # mass on rightward input positions
    cross_per_step = []

    for t in range(T):
        ca = cross_attns[t]  # [B, L, L]
        step_data = {}
        for k in range(half):
            attn_k = ca[:, k, :]  # [B, L] — attention from result pos k
            # Weighted mean input position
            positions = torch.arange(ca.shape[2], dtype=torch.float)
            mean_pos = (attn_k * positions.unsqueeze(0)).sum(dim=1).mean().item()
            cross_focus[t, k] = mean_pos

            # Mass on positions right of k (higher index = more rightward = lower-order digits)
            right_start = min(2 * (k + 1), ca.shape[2])
            right_mass = attn_k[:, right_start:].sum(dim=1).mean().item()
            cross_right_mass[t, k] = right_mass

            step_data[k] = {
                "mean_input_pos": mean_pos,
                "right_mass": right_mass,
            }
        cross_per_step.append(step_data)

    results["cross_attention"] = {
        "focus_shift": cross_focus.tolist(),
        "right_mass": cross_right_mass.tolist(),
        "per_step": cross_per_step,
    }

    # === Self-attention analysis ===
    # For each result position k, check if it attends to position k+1 (carry source)
    carry_attn = torch.zeros(T, half)  # attention to carry-source position
    self_local = torch.zeros(T, half)  # self-attention (position attending to itself)

    for t in range(T):
        sa = self_attns[t]  # [B, L, L]
        for k in range(half):
            self_local[t, k] = sa[:, k, k].mean().item()
            if k < half - 1:
                carry_attn[t, k] = sa[:, k, k + 1].mean().item()

    results["self_attention"] = {
        "carry_source_attn": carry_attn.tolist(),
        "self_attn": self_local.tolist(),
    }

    # === Carry-conditional attention ===
    # Split examples by carry-in at each position, compare attention patterns
    carry_cond = {}
    for k in range(half - 1):
        has_carry = carry_in[:, k] == 1
        no_carry = carry_in[:, k] == 0
        n_carry = has_carry.sum().item()
        n_no = no_carry.sum().item()

        if n_carry < 50 or n_no < 50:
            continue

        step_diffs = []
        for t in range(T):
            ca = cross_attns[t]
            # Cross-attn from position k
            attn_carry = ca[has_carry, k, :].mean(dim=0)
            attn_no = ca[no_carry, k, :].mean(dim=0)
            diff = (attn_carry - attn_no).abs().sum().item()

            sa = self_attns[t]
            sa_carry = sa[has_carry, k, :].mean(dim=0)
            sa_no = sa[no_carry, k, :].mean(dim=0)
            sa_diff = (sa_carry - sa_no).abs().sum().item()

            step_diffs.append({
                "cross_attn_diff_L1": diff,
                "self_attn_diff_L1": sa_diff,
            })

        carry_cond[k] = {
            "n_carry": n_carry,
            "n_no_carry": n_no,
            "per_step": step_diffs,
        }

    results["carry_conditional"] = carry_cond

    # === Attention entropy per step ===
    cross_entropy_per_step = []
    self_entropy_per_step = []
    for t in range(T):
        ca = cross_attns[t][:, :half, :]
        ca_ent = -(ca * (ca + 1e-10).log()).sum(dim=-1).mean().item()
        cross_entropy_per_step.append(ca_ent)

        sa = self_attns[t][:, :half, :]
        sa_ent = -(sa * (sa + 1e-10).log()).sum(dim=-1).mean().item()
        self_entropy_per_step.append(sa_ent)

    results["entropy"] = {
        "cross_attn": cross_entropy_per_step,
        "self_attn": self_entropy_per_step,
    }

    return results


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Attention flow analysis through dynamics",
    }

    for seq_len, label in [(8, "4digit"), (16, "8digit")]:
        config = build_config(seed=SEED, seq_len=seq_len)
        half = seq_len // 2
        T = config["T"]
        V = config["vocab_size"]

        print(f"\n{'=' * 60}", flush=True)
        print(f"  {label} addition (L={seq_len}, half={half})", flush=True)
        print(f"{'=' * 60}", flush=True)

        set_seed(SEED + 7777)
        eval_src, eval_tgt = generate_batch("addition", N_EVAL, seq_len, V)
        eval_src = eval_src.to(device)
        eval_tgt = eval_tgt.to(device)
        carry_in = compute_carry_in(eval_src, V)

        print(f"  Carry-in rates:", flush=True)
        for k in range(half):
            rate = carry_in[:, k].float().mean().item()
            print(f"    Position {k}: {rate:.3f}", flush=True)

        seq_results = {}

        for track in ["dynamics_ce", "e5"]:
            print(f"\n  --- {track} ---", flush=True)

            print(f"  Training...", flush=True)
            t0 = time.time()
            model = train_model(track, config, device)
            train_time = time.time() - t0
            print(f"  Trained in {train_time:.0f}s", flush=True)

            # Accuracy
            with torch.no_grad():
                logits = model(eval_src, T)
                preds = logits[:, :half, :].argmax(dim=-1)
                targets = eval_tgt[:, :half]
                tok_acc = (preds == targets).float().mean().item()
                seq_acc = (preds == targets).all(dim=1).float().mean().item()
            print(f"  Accuracy: tok={tok_acc:.4f} seq={seq_acc:.4f}", flush=True)

            if tok_acc < 0.6:
                print(f"  Model did not learn — skipping attention analysis", flush=True)
                seq_results[track] = {
                    "token_acc": tok_acc, "seq_acc": seq_acc,
                    "status": "did_not_learn",
                }
                continue

            # Collect attention patterns
            print(f"  Collecting attention patterns...", flush=True)

            # Process in batches to avoid OOM
            batch_size = 512
            all_self_attns = [[] for _ in range(T)]
            all_cross_attns = [[] for _ in range(T)]

            for start in range(0, N_EVAL, batch_size):
                end = min(start + batch_size, N_EVAL)
                batch_src = eval_src[start:end]
                sa_list, ca_list = collect_attention_patterns(model, batch_src, T)
                for t in range(T):
                    all_self_attns[t].append(sa_list[t])
                    all_cross_attns[t].append(ca_list[t])

            self_attns = [torch.cat(sa, dim=0) for sa in all_self_attns]
            cross_attns = [torch.cat(ca, dim=0) for ca in all_cross_attns]

            # Analyze
            print(f"  Analyzing attention patterns...", flush=True)
            attn_results = analyze_attention(
                self_attns, cross_attns, carry_in.cpu(), half, T
            )

            # Print key findings
            print(f"\n  Cross-attention mean input position (result pos × step):",
                  flush=True)
            focus = attn_results["cross_attention"]["focus_shift"]
            print(f"  {'':>6s}", end="", flush=True)
            for k in range(half):
                print(f"  pos_{k:d}", end="", flush=True)
            print(flush=True)
            for t in range(T):
                print(f"  t={t:2d}  ", end="", flush=True)
                for k in range(half):
                    print(f"  {focus[t][k]:5.2f}", end="", flush=True)
                print(flush=True)

            print(f"\n  Self-attention to carry source (k→k+1):", flush=True)
            carry_sa = attn_results["self_attention"]["carry_source_attn"]
            print(f"  {'':>6s}", end="", flush=True)
            for k in range(half - 1):
                print(f"  pos_{k:d}", end="", flush=True)
            print(flush=True)
            for t in range(T):
                print(f"  t={t:2d}  ", end="", flush=True)
                for k in range(half - 1):
                    print(f"  {carry_sa[t][k]:5.3f}", end="", flush=True)
                print(flush=True)

            print(f"\n  Attention entropy per step:", flush=True)
            print(f"    Cross: {[f'{e:.3f}' for e in attn_results['entropy']['cross_attn']]}",
                  flush=True)
            print(f"    Self:  {[f'{e:.3f}' for e in attn_results['entropy']['self_attn']]}",
                  flush=True)

            # Carry-conditional analysis
            cc = attn_results["carry_conditional"]
            if cc:
                print(f"\n  Carry-conditional attention difference (L1):", flush=True)
                for k in sorted(cc.keys()):
                    diffs = cc[k]["per_step"]
                    cross_diffs = [d["cross_attn_diff_L1"] for d in diffs]
                    print(f"    Position {k} (n_carry={cc[k]['n_carry']}): "
                          f"cross_diff peak at step {cross_diffs.index(max(cross_diffs))} "
                          f"(max={max(cross_diffs):.4f})", flush=True)

            seq_results[track] = {
                "token_acc": tok_acc,
                "seq_acc": seq_acc,
                "train_time_s": train_time,
                "attention": attn_results,
            }

            del model
            torch.cuda.empty_cache()

        all_results[label] = seq_results

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d9_attention_flow.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
