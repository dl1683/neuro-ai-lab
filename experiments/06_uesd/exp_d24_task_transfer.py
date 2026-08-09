"""
Experiment D24: Task Transfer — Do Dynamics Learn Algebraic Structure?

The most ambitious test of the UESD thesis: do dynamics learn generalizable
computation, or just task-specific mappings?

Protocol:
1. Train on base-64 ADDITION (L=8, standard setup)
2. WITHOUT retraining, evaluate on:
   a) Subtraction (A - B mod base)
   b) Bitwise XOR (A XOR B)
   c) Element-wise max (max(a_i, b_i))
3. Measure zero-shot transfer accuracy at T=10

If any transfer occurs, it means the dynamics learned something about
modular arithmetic beyond just carry propagation. Even partial transfer
(e.g., subtraction works on easy digits but fails on borrows) is interesting.

Control: also test transfer of an encoder-only model (no dynamics) and
a fresh untrained model, to ensure any transfer signal is from the
dynamics, not the encoder or embeddings.

PREDICTIONS (bold):
1. Subtraction: 10-30% seq_acc (partial transfer — model recognizes
   digit-pair structure but gets borrow logic wrong)
2. XOR: 0-5% (no transfer — XOR has no carry/borrow structure)
3. Element-wise max: 20-50% (partial — position-independence helps)
4. If denoising-trained D22 variant transfers better than baseline,
   that implies denoising learns more robust representations

This is cutting-edge research. If subtraction transfers at >50%,
it's a breakthrough — dynamics learned algebraic inverse structure.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, EncoderOnlyAblation
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEED = 42
TRAINING_STEPS = 20000
BATCH_SIZE = 256
SEQ_LEN = 8
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 32
TRAIN_T = 10

EVAL_SAMPLES = 4096
EVAL_T_VALUES = [1, 3, 5, 8, 10, 15, 20]


def generate_subtraction_batch(batch_size, seq_len, vocab_size):
    """Multi-digit subtraction: (A - B) mod base^half.

    Same input format as addition: interleaved [a0,b0,a1,b1,...].
    Borrow propagation goes right-to-left, analogous to carry in addition.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    borrow = torch.zeros(batch_size, dtype=torch.long)
    result = torch.zeros(batch_size, half, dtype=torch.long)
    for i in range(half - 1, -1, -1):
        diff = a[:, i] - b[:, i] - borrow
        result[:, i] = diff % vocab_size
        borrow = (diff < 0).long()

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


def generate_xor_batch(batch_size, seq_len, vocab_size):
    """Bitwise XOR of each digit pair: result_i = a_i XOR b_i (mod vocab_size).

    No carry/borrow — purely position-wise operation.
    XOR is defined as (a + b) mod 2 extended to base-64 via bitwise XOR
    of the integer representations.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    result = a ^ b
    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


def generate_max_batch(batch_size, seq_len, vocab_size):
    """Element-wise max: result_i = max(a_i, b_i).

    No carry/borrow — purely position-wise operation.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    result = torch.maximum(a, b)
    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


TRANSFER_TASKS = {
    "subtraction": generate_subtraction_batch,
    "xor": generate_xor_batch,
    "element_max": generate_max_batch,
}


def train_addition(model, device):
    """Standard CE-dynamics training on addition."""
    set_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, TRAIN_T)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s", flush=True)
    return elapsed


@torch.no_grad()
def evaluate_transfer(model, device, task_name, gen_fn, T):
    """Evaluate a trained model on a transfer task at given T."""
    model.eval()
    half = SEQ_LEN // 2
    set_seed(9999)
    src, tgt = gen_fn(EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    logits = model(src, T)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()

    per_pos = {}
    for p in range(half):
        per_pos[f"pos{p}"] = (preds[:, p] == tgt_result[:, p]).float().mean().item()

    return {
        "tok_acc": round(tok_acc, 4),
        "seq_acc": round(seq_acc, 4),
        **{k: round(v, 3) for k, v in per_pos.items()},
    }


@torch.no_grad()
def evaluate_encoder_transfer(model, device, task_name, gen_fn):
    """Evaluate encoder-only on transfer task."""
    model.eval()
    half = SEQ_LEN // 2
    set_seed(9999)
    src, tgt = gen_fn(EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    logits = model(src)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()

    return {
        "tok_acc": round(tok_acc, 4),
        "seq_acc": round(seq_acc, 4),
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "D24: Task transfer — do dynamics learn generalizable algebraic structure?",
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
            "train_T": TRAIN_T,
        },
    }

    # === Phase 1: Train on addition ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 1: TRAINING ON ADDITION", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(SEED)
    uesd_model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    uesd_model = uesd_model.to(device)
    print(f"    UESD params: {count_params(uesd_model)}", flush=True)
    train_time = train_addition(uesd_model, device)

    # Verify addition performance
    print(f"\n  Verifying addition performance...", flush=True)
    set_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    half = SEQ_LEN // 2
    uesd_model.eval()
    with torch.no_grad():
        logits = uesd_model(src, TRAIN_T)
        preds = logits[:, :half].argmax(dim=-1)
        add_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
    print(f"    Addition seq_acc at T=10: {add_acc:.4f}", flush=True)
    all_results["addition_accuracy"] = round(add_acc, 4)
    all_results["train_time_s"] = round(train_time, 1)

    # === Phase 2: Train encoder-only on addition (control) ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 2: ENCODER-ONLY CONTROL", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(SEED)
    enc_model = EncoderOnlyAblation(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    enc_model = enc_model.to(device)
    optimizer = torch.optim.Adam(enc_model.parameters(), lr=LR)
    enc_model.train()

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)
        logits = enc_model(src)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_model.parameters(), 1.0)
        optimizer.step()
        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f}", flush=True)

    enc_model.eval()
    with torch.no_grad():
        set_seed(9999)
        src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)
        logits = enc_model(src)
        preds = logits[:, :half].argmax(dim=-1)
        enc_add_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
    print(f"    Encoder-only addition seq_acc: {enc_add_acc:.4f}", flush=True)
    all_results["encoder_addition_accuracy"] = round(enc_add_acc, 4)

    # === Phase 3: Untrained baseline ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 3: UNTRAINED BASELINE", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(SEED)
    untrained_model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    untrained_model = untrained_model.to(device)
    untrained_model.eval()

    # === Phase 4: Zero-shot transfer evaluation ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 4: ZERO-SHOT TRANSFER", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_results["transfer"] = {}

    for task_name, gen_fn in TRANSFER_TASKS.items():
        print(f"\n  Task: {task_name}", flush=True)
        all_results["transfer"][task_name] = {}

        # Trained UESD at various T
        print(f"    UESD (trained on addition):", flush=True)
        uesd_transfer = {}
        for T in EVAL_T_VALUES:
            result = evaluate_transfer(uesd_model, device, task_name, gen_fn, T)
            uesd_transfer[f"T={T}"] = result
            print(f"      T={T:>2d}: tok={result['tok_acc']:.4f} seq={result['seq_acc']:.4f}",
                  flush=True)
        all_results["transfer"][task_name]["uesd_trained"] = uesd_transfer

        # Untrained UESD at T=10
        print(f"    UESD (untrained):", flush=True)
        untrained_result = evaluate_transfer(untrained_model, device, task_name, gen_fn, TRAIN_T)
        all_results["transfer"][task_name]["uesd_untrained"] = untrained_result
        print(f"      T=10: tok={untrained_result['tok_acc']:.4f} seq={untrained_result['seq_acc']:.4f}",
              flush=True)

        # Encoder-only (trained on addition)
        print(f"    Encoder-only (trained on addition):", flush=True)
        enc_result = evaluate_encoder_transfer(enc_model, device, task_name, gen_fn)
        all_results["transfer"][task_name]["encoder_trained"] = enc_result
        print(f"      tok={enc_result['tok_acc']:.4f} seq={enc_result['seq_acc']:.4f}",
              flush=True)

    del untrained_model, enc_model
    torch.cuda.empty_cache()

    # === Phase 5: Fine-tune transfer (how fast does each task learn?) ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 5: FINE-TUNING TRANSFER SPEED", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_results["finetune"] = {}

    for task_name, gen_fn in TRANSFER_TASKS.items():
        print(f"\n  Fine-tuning on {task_name}:", flush=True)

        # Start from addition-trained checkpoint
        set_seed(SEED + 1)
        ft_model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
        ft_model = ft_model.to(device)
        ft_model.load_state_dict(uesd_model.state_dict())
        ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=LR * 0.1)

        ft_model.train()
        milestones = {}
        for step in range(1, 5001):
            src, tgt = gen_fn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
            src, tgt = src.to(device), tgt.to(device)
            logits = ft_model(src, TRAIN_T)
            loss = F.cross_entropy(
                logits[:, :half].reshape(-1, logits.size(-1)),
                tgt[:, :half].reshape(-1),
            )
            ft_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            ft_optimizer.step()

            if step in [10, 50, 100, 500, 1000, 2000, 5000]:
                ft_model.eval()
                with torch.no_grad():
                    set_seed(9999)
                    eval_src, eval_tgt = gen_fn(EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
                    eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)
                    eval_logits = ft_model(eval_src, TRAIN_T)
                    eval_preds = eval_logits[:, :half].argmax(dim=-1)
                    ft_acc = (eval_preds == eval_tgt[:, :half]).all(dim=1).float().mean().item()
                milestones[f"step_{step}"] = round(ft_acc, 4)
                print(f"    Step {step:>5d}: seq_acc={ft_acc:.4f}", flush=True)
                ft_model.train()

        # Train from scratch comparison
        print(f"  Training {task_name} from scratch:", flush=True)
        set_seed(SEED)
        scratch_model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
        scratch_model = scratch_model.to(device)
        scratch_optimizer = torch.optim.Adam(scratch_model.parameters(), lr=LR)
        scratch_model.train()
        scratch_milestones = {}

        for step in range(1, 5001):
            src, tgt = gen_fn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
            src, tgt = src.to(device), tgt.to(device)
            logits = scratch_model(src, TRAIN_T)
            loss = F.cross_entropy(
                logits[:, :half].reshape(-1, logits.size(-1)),
                tgt[:, :half].reshape(-1),
            )
            scratch_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scratch_model.parameters(), 1.0)
            scratch_optimizer.step()

            if step in [10, 50, 100, 500, 1000, 2000, 5000]:
                scratch_model.eval()
                with torch.no_grad():
                    set_seed(9999)
                    eval_src, eval_tgt = gen_fn(EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
                    eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)
                    eval_logits = scratch_model(eval_src, TRAIN_T)
                    eval_preds = eval_logits[:, :half].argmax(dim=-1)
                    scratch_acc = (eval_preds == eval_tgt[:, :half]).all(dim=1).float().mean().item()
                scratch_milestones[f"step_{step}"] = round(scratch_acc, 4)
                print(f"    Scratch step {step:>5d}: seq_acc={scratch_acc:.4f}", flush=True)
                scratch_model.train()

        del ft_model, scratch_model
        torch.cuda.empty_cache()

        all_results["finetune"][task_name] = {
            "from_addition": milestones,
            "from_scratch": scratch_milestones,
        }

    # === Final summary ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D24 TRANSFER SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    print(f"\n  Zero-shot transfer (T=10):", flush=True)
    print(f"  {'Task':<15} {'UESD':>8} {'Untrained':>10} {'Enc-only':>10}", flush=True)
    for task_name in TRANSFER_TASKS:
        t = all_results["transfer"][task_name]
        uesd_acc = t["uesd_trained"]["T=10"]["seq_acc"]
        untr_acc = t["uesd_untrained"]["seq_acc"]
        enc_acc = t["encoder_trained"]["seq_acc"]
        transfer = uesd_acc - untr_acc
        print(f"  {task_name:<15} {uesd_acc:>8.4f} {untr_acc:>10.4f} {enc_acc:>10.4f} "
              f"(transfer: {transfer:+.4f})", flush=True)

    print(f"\n  Fine-tuning speed (seq_acc at step 500):", flush=True)
    print(f"  {'Task':<15} {'From add':>10} {'Scratch':>10} {'Speedup':>10}", flush=True)
    for task_name in TRANSFER_TASKS:
        ft = all_results["finetune"][task_name]
        from_add = ft["from_addition"].get("step_500", 0)
        scratch = ft["from_scratch"].get("step_500", 0)
        speedup = from_add / max(scratch, 1e-6)
        print(f"  {task_name:<15} {from_add:>10.4f} {scratch:>10.4f} {speedup:>10.2f}x",
              flush=True)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d24_task_transfer.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
