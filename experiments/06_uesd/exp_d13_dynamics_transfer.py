"""
Experiment D13: Cross-Task Dynamics Transfer

The deepest test of the "thinking engine" hypothesis: if the UESD dynamics
module learns general-purpose iterative computation, it should TRANSFER
across tasks. If it only learns task-specific digit manipulation, transfer
will fail completely.

Protocol:
1. Train on ADDITION (the dynamics-requiring task) until convergence
2. Freeze the dynamics module (TransformerDecoderLayer weights)
3. Train ONLY the encoder + readout on a NEW task (multiplication,
   subtraction, or multi-digit comparison) — the dynamics must serve
   as a general iterative computation engine
4. Compare frozen-dynamics transfer vs full training vs encoder-only

Transfer tasks (all require sequential computation like addition):
- SUBTRACTION: A - B mod base^half (same carry structure, different operation)
- MULTIPLICATION: (A * B) mod base^half (harder, requires partial products)
- COMPARISON: output 1 at position of first differing digit, 0 elsewhere
  (requires left-to-right sequential scan)

Key measurements:
- Transfer accuracy vs full training accuracy (transfer gap)
- Learning curve comparison: does transfer learn faster?
- Per-position accuracy: which positions benefit from transfer?
- Dynamics step utilization: does the transferred model use the same
  number of effective steps?

PREDICTIONS:
1. SUBTRACTION transfer works well (>90% seq_acc) because carry/borrow
   propagation has the same computational structure as addition.
2. MULTIPLICATION transfer is partial (<60% seq_acc) because partial
   product accumulation is a fundamentally different operation.
3. COMPARISON transfer works moderately (70-85%) because sequential
   scan shares the "propagate information rightward" structure.
4. Transfer models learn FASTER (fewer steps to converge) because the
   dynamics module already knows how to propagate information.
5. If predictions 1-4 hold, the dynamics module encodes general
   "iterative information propagation" — not just addition-specific rules.

This is the strongest possible evidence for UESD as a universal
computation substrate: the same dynamics serve multiple tasks.
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
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch


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
        "seed": seed,
    }


# === New task generators ===

def generate_subtraction_batch(batch_size, seq_len, vocab_size):
    """A - B mod base^half. Borrow propagation mirrors carry propagation."""
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    borrow = torch.zeros(batch_size, dtype=torch.long)
    result = torch.zeros(batch_size, half, dtype=torch.long)
    for i in range(half - 1, -1, -1):
        diff = a[:, i].long() - b[:, i].long() - borrow
        borrow = (diff < 0).long()
        result[:, i] = diff % vocab_size

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


def generate_comparison_batch(batch_size, seq_len, vocab_size):
    """Compare A vs B digit by digit, MSB first.
    Output: 0=equal, 1=A>B, 2=A<B at the FIRST differing position.
    All subsequent positions get 0. This requires left-to-right sequential scan.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a
    input_ids[:, 1::2] = b

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(batch_size):
        found = False
        for j in range(half):
            if not found and a[i, j] != b[i, j]:
                target_ids[i, j] = 1 if a[i, j] > b[i, j] else 2
                found = True
            # remaining positions stay 0
    return input_ids, target_ids


def generate_multiply_mod_batch(batch_size, seq_len, vocab_size):
    """Multiply A * B mod base^half. Uses the same interleaved input format.
    Much harder than addition — requires accumulating partial products.
    We use smaller base to keep product manageable.
    """
    half = seq_len // 2
    # Use smaller effective base to keep numbers manageable
    eff_base = min(vocab_size, 16)
    a = torch.randint(0, eff_base, (batch_size, half))
    b = torch.randint(0, eff_base, (batch_size, half))

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a
    input_ids[:, 1::2] = b

    # Compute A * B as multi-digit numbers in base eff_base
    result = torch.zeros(batch_size, half, dtype=torch.long)
    for i in range(batch_size):
        val_a = 0
        val_b = 0
        for j in range(half):
            val_a = val_a * eff_base + a[i, j].item()
            val_b = val_b * eff_base + b[i, j].item()
        product = (val_a * val_b) % (eff_base ** half)
        for j in range(half - 1, -1, -1):
            result[i, j] = product % eff_base
            product //= eff_base

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


TRANSFER_TASKS = {
    "subtraction": generate_subtraction_batch,
    "comparison": generate_comparison_batch,
    "multiply_mod": generate_multiply_mod_batch,
}


def train_ce_dynamics(model, task_gen, config, device, freeze_dynamics=False):
    """Train with CE-dynamics loss. Optionally freeze the dynamics module."""
    if freeze_dynamics:
        for param in model.dynamics.parameters():
            param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())

    model.train()
    optimizer = torch.optim.Adam(trainable, lr=config["lr"])
    T = config["T"]
    total = config["training_steps"]
    V = config["vocab_size"]
    seq_len = config["seq_len"]
    half = seq_len // 2

    history = {"ce": [], "seq_acc": []}

    for step in range(1, total + 1):
        src, tgt = task_gen(config["batch_size"], seq_len, V)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits[:, :half, :].argmax(dim=-1)
                targets = tgt[:, :half]
                seq_acc = (preds == targets).all(dim=1).float().mean().item()
            history["ce"].append(loss.item())
            history["seq_acc"].append(seq_acc)
            print(f"    Step {step:>6d}/{total} | CE: {loss.item():.4f} "
                  f"| seq_acc: {seq_acc:.4f}", flush=True)

    model.eval()
    if freeze_dynamics:
        for param in model.dynamics.parameters():
            param.requires_grad = True

    return history


def evaluate_task(model, task_gen, config, device, n_eval=4096):
    """Evaluate model on a task."""
    V = config["vocab_size"]
    T = config["T"]
    half = config["seq_len"] // 2

    model.eval()
    with torch.no_grad():
        src, tgt = task_gen(n_eval, config["seq_len"], V)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
        preds = logits[:, :half, :].argmax(dim=-1)
        targets = tgt[:, :half]

        tok_acc = (preds == targets).float().mean().item()
        seq_acc = (preds == targets).all(dim=1).float().mean().item()

        # Per-position accuracy
        per_pos = []
        for p in range(half):
            pos_acc = (preds[:, p] == targets[:, p]).float().mean().item()
            per_pos.append(pos_acc)

    return {"tok_acc": tok_acc, "seq_acc": seq_acc, "per_position": per_pos}


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config(SEED)
    V = config["vocab_size"]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Cross-task dynamics transfer: does the thinking engine generalize?",
        "config": config,
    }

    # === Step 1: Train source model on addition ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Step 1: Train source model on ADDITION (CE-dynamics)", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(SEED)
    source_model = UESDModel(
        V, config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    print(f"  Params: {count_params(source_model)}", flush=True)

    from shared.data import generate_addition_batch
    t0 = time.time()
    source_history = train_ce_dynamics(
        source_model, generate_addition_batch, config, device
    )
    source_time = time.time() - t0

    # Evaluate source on addition
    set_seed(SEED + 8888)
    source_eval = evaluate_task(
        source_model, generate_addition_batch, config, device
    )
    print(f"  Source on addition: tok={source_eval['tok_acc']:.4f} "
          f"seq={source_eval['seq_acc']:.4f} ({source_time:.0f}s)", flush=True)

    all_results["source"] = {
        "task": "addition",
        "train_time_s": source_time,
        "eval": source_eval,
        "history": source_history,
    }

    # Save source dynamics state for transfer
    source_dynamics_state = {
        k: v.clone() for k, v in source_model.dynamics.state_dict().items()
    }

    # === Step 2: Transfer to each target task ===
    for task_name, task_gen in TRANSFER_TASKS.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Transfer task: {task_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        task_results = {}

        # --- Condition A: Full training from scratch ---
        print(f"\n  Condition A: Full training from scratch", flush=True)
        set_seed(SEED)
        model_full = UESDModel(
            V, config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)

        t0 = time.time()
        hist_full = train_ce_dynamics(model_full, task_gen, config, device)
        time_full = time.time() - t0

        set_seed(SEED + 8888)
        eval_full = evaluate_task(model_full, task_gen, config, device)
        print(f"    Full: tok={eval_full['tok_acc']:.4f} "
              f"seq={eval_full['seq_acc']:.4f} ({time_full:.0f}s)", flush=True)

        task_results["full_training"] = {
            "train_time_s": time_full,
            "eval": eval_full,
            "history": hist_full,
        }
        del model_full
        torch.cuda.empty_cache()

        # --- Condition B: Transfer (frozen dynamics) ---
        print(f"\n  Condition B: Transfer (frozen dynamics from addition)", flush=True)
        set_seed(SEED)
        model_transfer = UESDModel(
            V, config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        # Load source dynamics
        model_transfer.dynamics.load_state_dict(source_dynamics_state)

        t0 = time.time()
        hist_transfer = train_ce_dynamics(
            model_transfer, task_gen, config, device, freeze_dynamics=True
        )
        time_transfer = time.time() - t0

        set_seed(SEED + 8888)
        eval_transfer = evaluate_task(model_transfer, task_gen, config, device)
        print(f"    Transfer: tok={eval_transfer['tok_acc']:.4f} "
              f"seq={eval_transfer['seq_acc']:.4f} ({time_transfer:.0f}s)", flush=True)

        # Check if source still works on addition with transferred model
        set_seed(SEED + 8888)
        addition_with_transfer = evaluate_task(
            model_transfer, generate_addition_batch, config, device
        )
        print(f"    Transfer model on addition: "
              f"seq={addition_with_transfer['seq_acc']:.4f}", flush=True)

        task_results["frozen_transfer"] = {
            "train_time_s": time_transfer,
            "eval": eval_transfer,
            "addition_eval": addition_with_transfer,
            "history": hist_transfer,
        }
        del model_transfer
        torch.cuda.empty_cache()

        # --- Condition C: Transfer (fine-tuned dynamics) ---
        print(f"\n  Condition C: Transfer (fine-tuned dynamics)", flush=True)
        set_seed(SEED)
        model_ft = UESDModel(
            V, config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        ).to(device)
        model_ft.dynamics.load_state_dict(source_dynamics_state)

        t0 = time.time()
        hist_ft = train_ce_dynamics(model_ft, task_gen, config, device)
        time_ft = time.time() - t0

        set_seed(SEED + 8888)
        eval_ft = evaluate_task(model_ft, task_gen, config, device)
        print(f"    Fine-tune: tok={eval_ft['tok_acc']:.4f} "
              f"seq={eval_ft['seq_acc']:.4f} ({time_ft:.0f}s)", flush=True)

        task_results["finetuned_transfer"] = {
            "train_time_s": time_ft,
            "eval": eval_ft,
            "history": hist_ft,
        }
        del model_ft
        torch.cuda.empty_cache()

        # Transfer gap analysis
        full_acc = eval_full["seq_acc"]
        frozen_acc = eval_transfer["seq_acc"]
        ft_acc = eval_ft["seq_acc"]
        task_results["transfer_gap"] = {
            "full_vs_frozen": full_acc - frozen_acc,
            "full_vs_finetuned": full_acc - ft_acc,
            "frozen_vs_finetuned": ft_acc - frozen_acc,
        }
        print(f"\n    Transfer gaps:", flush=True)
        print(f"      Full vs Frozen:    {full_acc - frozen_acc:+.4f}", flush=True)
        print(f"      Full vs Finetuned: {full_acc - ft_acc:+.4f}", flush=True)

        all_results[task_name] = task_results

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d13_dynamics_transfer.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
