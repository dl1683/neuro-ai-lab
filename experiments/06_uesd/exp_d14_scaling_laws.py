"""
Experiment D14: UESD Scaling Laws

Q4 from the formalization: "Does this scale?" Transformers scale predictably
with compute. Do iterative dynamics systems?

This experiment maps how UESD performance scales along three axes:
1. MODEL SIZE: d_model in {64, 128, 256, 512} (4x range)
2. ITERATION DEPTH: T in {2, 5, 10, 20} (10x range)
3. PROBLEM SIZE: seq_len in {8, 12, 16, 20} (carry chains 4-10 deep)

We measure: accuracy, compute (FLOPs), convergence speed, and final energy.

The key question: does UESD follow a POWER LAW like neural scaling laws?
If performance scales as acc ~ C^alpha where C is compute, what is alpha?
How does it compare to encoder-only baselines at matched compute?

Two specific hypotheses from the formalization:
H1: UESD is compute-efficient: weight-tied dynamics should achieve
    higher accuracy per parameter than encoder-only at the same FLOPs.
H2: There exists a critical T* below which the model cannot solve
    problems requiring carry chains longer than T*. This would be
    direct evidence that T provides "thinking depth."

PREDICTIONS:
1. Accuracy vs T follows a SIGMOID: near-zero below critical T*,
   rapid rise, then plateau. T* ~ max_carry_chain_length + 1.
2. Accuracy vs d_model follows a smooth power law (like standard
   neural scaling). Larger models converge faster and reach higher plateau.
3. UESD at d=128 T=10 outperforms encoder-only at d=256 (4L) in
   accuracy-per-FLOP on long carry chains.
4. For problem size scaling: accuracy degrades gracefully as seq_len
   increases, with degradation point shifting right for larger T.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, EncoderOnlyAblation
from shared.training import set_seed, count_params
from shared.data import generate_batch


SEED = 42

# Sweep dimensions
D_MODELS = [64, 128, 256]
T_VALUES = [2, 5, 10, 20]
SEQ_LENS = [8, 12, 16]
ENCODER_DEPTHS = [2, 4, 8]


def build_config(d_model=128, T=10, seq_len=8, steps=20000, seed=42):
    return {
        "vocab_size": 64,
        "d_model": d_model,
        "n_heads": max(2, d_model // 32),
        "d_ff": d_model * 4,
        "n_enc_layers": 2,
        "max_len": max(32, seq_len * 2),
        "seq_len": seq_len,
        "T": T,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": steps,
        "seed": seed,
    }


def estimate_flops_uesd(config):
    """Rough FLOP estimate for one forward pass of UESD."""
    d = config["d_model"]
    L = config["seq_len"]
    T = config["T"]
    # Encoder: ~2 layers of self-attention + FFN
    enc_flops = 2 * (4 * L * d * d + 2 * L * d * config["d_ff"])
    # Dynamics: T iterations of decoder layer (self-attn + cross-attn + FFN)
    dyn_flops = T * (6 * L * d * d + 2 * L * d * config["d_ff"])
    # Readout
    readout_flops = L * d * d + L * d * config["vocab_size"]
    return enc_flops + dyn_flops + readout_flops


def estimate_flops_encoder(config, n_layers):
    """Rough FLOP estimate for encoder-only."""
    d = config["d_model"]
    L = config["seq_len"]
    enc_flops = n_layers * (4 * L * d * d + 2 * L * d * config["d_ff"])
    readout_flops = L * d * d + L * d * config["vocab_size"]
    return enc_flops + readout_flops


def train_and_eval(model, config, device, T_override=None):
    """Train model with CE-dynamics and evaluate."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = T_override or config["T"]
    total = config["training_steps"]
    V = config["vocab_size"]
    seq_len = config["seq_len"]
    half = seq_len // 2

    history = []
    t0 = time.time()

    for step in range(1, total + 1):
        src, tgt = generate_batch("addition", config["batch_size"], seq_len, V)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits[:, :half, :].argmax(dim=-1)
                targets = tgt[:, :half]
                seq_acc = (preds == targets).all(dim=1).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "seq_acc": seq_acc})
            print(f"      Step {step:>6d}/{total} | CE: {loss.item():.4f} "
                  f"| seq: {seq_acc:.4f}", flush=True)

    train_time = time.time() - t0

    # Final evaluation
    model.eval()
    set_seed(SEED + 7777)
    eval_src, eval_tgt = generate_batch("addition", 4096, seq_len, V)
    eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)

    with torch.no_grad():
        logits = model(eval_src, T)
        preds = logits[:, :half, :].argmax(dim=-1)
        targets = eval_tgt[:, :half]
        tok_acc = (preds == targets).float().mean().item()
        seq_acc = (preds == targets).all(dim=1).float().mean().item()

    return {
        "tok_acc": tok_acc,
        "seq_acc": seq_acc,
        "train_time_s": train_time,
        "history": history,
    }


def train_and_eval_encoder(model, config, device, n_layers):
    """Train encoder-only and evaluate."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total = config["training_steps"]
    V = config["vocab_size"]
    seq_len = config["seq_len"]
    half = seq_len // 2

    history = []
    t0 = time.time()

    for step in range(1, total + 1):
        src, tgt = generate_batch("addition", config["batch_size"], seq_len, V)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits[:, :half, :].argmax(dim=-1)
                targets = tgt[:, :half]
                seq_acc = (preds == targets).all(dim=1).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "seq_acc": seq_acc})
            print(f"      Step {step:>6d}/{total} | CE: {loss.item():.4f} "
                  f"| seq: {seq_acc:.4f}", flush=True)

    train_time = time.time() - t0

    model.eval()
    set_seed(SEED + 7777)
    eval_src, eval_tgt = generate_batch("addition", 4096, seq_len, V)
    eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)

    with torch.no_grad():
        logits = model(eval_src)
        preds = logits[:, :half, :].argmax(dim=-1)
        targets = eval_tgt[:, :half]
        tok_acc = (preds == targets).float().mean().item()
        seq_acc = (preds == targets).all(dim=1).float().mean().item()

    return {
        "tok_acc": tok_acc,
        "seq_acc": seq_acc,
        "train_time_s": train_time,
        "history": history,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Scaling laws: accuracy vs model size, iteration depth, problem size",
    }

    # === Sweep 1: T scaling at fixed d_model=128, seq_len=8 ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sweep 1: T scaling (d=128, L=8)", flush=True)
    print(f"{'=' * 60}", flush=True)

    t_sweep = {}
    for T in T_VALUES:
        config = build_config(d_model=128, T=T, seq_len=8, steps=20000)
        flops = estimate_flops_uesd(config)
        print(f"\n  T={T} (FLOPs: {flops:.0f})", flush=True)

        set_seed(SEED)
        model = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        params = count_params(model)
        print(f"    Params: {params}", flush=True)

        result = train_and_eval(model, config, device, T_override=T)
        t_sweep[f"T_{T}"] = {
            "T": T, "params": params, "flops": flops,
            **result,
        }
        print(f"    Final: tok={result['tok_acc']:.4f} seq={result['seq_acc']:.4f}",
              flush=True)

        del model
        torch.cuda.empty_cache()

    all_results["t_sweep"] = t_sweep

    # === Sweep 2: d_model scaling at fixed T=10, seq_len=8 ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sweep 2: d_model scaling (T=10, L=8)", flush=True)
    print(f"{'=' * 60}", flush=True)

    d_sweep = {}
    for d in D_MODELS:
        config = build_config(d_model=d, T=10, seq_len=8, steps=20000)
        flops = estimate_flops_uesd(config)
        print(f"\n  d_model={d} (FLOPs: {flops:.0f})", flush=True)

        set_seed(SEED)
        model = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        params = count_params(model)
        print(f"    Params: {params}", flush=True)

        result = train_and_eval(model, config, device)
        d_sweep[f"d_{d}"] = {
            "d_model": d, "params": params, "flops": flops,
            **result,
        }
        print(f"    Final: tok={result['tok_acc']:.4f} seq={result['seq_acc']:.4f}",
              flush=True)

        del model
        torch.cuda.empty_cache()

    all_results["d_model_sweep"] = d_sweep

    # === Sweep 3: Problem size scaling at fixed d=128, T=10 ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sweep 3: Problem size scaling (d=128, T=10)", flush=True)
    print(f"{'=' * 60}", flush=True)

    l_sweep = {}
    for L in SEQ_LENS:
        config = build_config(d_model=128, T=10, seq_len=L, steps=25000)
        flops = estimate_flops_uesd(config)
        half = L // 2
        print(f"\n  seq_len={L} (half={half}, FLOPs: {flops:.0f})", flush=True)

        set_seed(SEED)
        model = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        params = count_params(model)
        print(f"    Params: {params}", flush=True)

        result = train_and_eval(model, config, device)
        l_sweep[f"L_{L}"] = {
            "seq_len": L, "half": half, "params": params, "flops": flops,
            **result,
        }
        print(f"    Final: tok={result['tok_acc']:.4f} seq={result['seq_acc']:.4f}",
              flush=True)

        del model
        torch.cuda.empty_cache()

    all_results["seq_len_sweep"] = l_sweep

    # === Sweep 4: Encoder-only baselines for compute comparison ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sweep 4: Encoder-only baselines (d=128, L=8)", flush=True)
    print(f"{'=' * 60}", flush=True)

    enc_sweep = {}
    for n_layers in ENCODER_DEPTHS:
        config = build_config(d_model=128, seq_len=8, steps=20000)
        flops = estimate_flops_encoder(config, n_layers)
        print(f"\n  Encoder {n_layers}L (FLOPs: {flops:.0f})", flush=True)

        set_seed(SEED)
        model = EncoderOnlyAblation(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], n_layers, config["max_len"],
        ).to(device)
        params = count_params(model)
        print(f"    Params: {params}", flush=True)

        result = train_and_eval_encoder(model, config, device, n_layers)
        enc_sweep[f"enc_{n_layers}L"] = {
            "n_layers": n_layers, "params": params, "flops": flops,
            **result,
        }
        print(f"    Final: tok={result['tok_acc']:.4f} seq={result['seq_acc']:.4f}",
              flush=True)

        del model
        torch.cuda.empty_cache()

    all_results["encoder_baselines"] = enc_sweep

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d14_scaling_laws.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
