"""D19: Step Ablation Falsification Test

FALSIFICATION TARGET: "Parallel computation engine" thesis.
If seq_acc(T=1) is within 0.02 of seq_acc(T=10), the model is NOT
gaining computation from iterative dynamics -- thesis weakened/falsified.

Design:
  1. Train CE-dynamics and E5 models (standard config, 20K steps)
  2. At test time, evaluate each at T = 1,2,3,4,5,6,7,8,10,15,20,32
  3. For each T, measure:
     - Token/sequence accuracy on clean inputs
     - Per-carry-chain accuracy (chain depth 0,1,2,3)
     - Recovery after single-position corruption (D17-style)
  4. Compute falsification metrics:
     - ratio = seq_acc(T=1) / seq_acc(T=10)
     - If ratio >= 0.98 for CE-dynamics, thesis is weakened

Source: Codex meta-analysis falsification test #1 (highest priority)
"""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from shared.model import UESDModel
from shared.training import train, count_params, set_seed
from shared.data import generate_addition_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

CFG = {
    "vocab_size": 64, "d_model": 128, "n_heads": 4, "d_ff": 512,
    "n_enc_layers": 2, "max_len": 32, "T": 10,
    "training_steps": 20000, "batch_size": 256, "seq_len": 8,
    "lr": 3e-4, "log_interval": 5000, "seed": 42,
}
T_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 32]
N_EVAL = 4096


def compute_carry_chains(a, b, base):
    """Compute max carry chain depth for each example."""
    half = a.shape[1]
    carry = torch.zeros(a.shape[0], dtype=torch.long)
    chain_lengths = torch.zeros(a.shape[0], dtype=torch.long)
    current_chain = torch.zeros(a.shape[0], dtype=torch.long)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        new_carry = s // base
        has_carry = (new_carry > 0).long()
        current_chain = has_carry * (current_chain + 1)
        chain_lengths = torch.max(chain_lengths, current_chain)
        carry = new_carry
    return chain_lengths


@torch.no_grad()
def evaluate_at_T(model, T_eval, n_samples=N_EVAL, batch_size=512):
    """Evaluate model at a specific T value."""
    model.eval()
    total_tok_correct = 0
    total_tok = 0
    total_seq_correct = 0
    total_seq = 0
    chain_correct = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}

    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_addition_batch(bs, CFG["seq_len"], CFG["vocab_size"])

        half = CFG["seq_len"] // 2
        a = src[:, 0::2][:, :half]
        b = src[:, 1::2][:, :half]
        chains = compute_carry_chains(a, b, CFG["vocab_size"])

        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        logits = model(src, T_eval)
        preds = logits.argmax(dim=-1)
        correct = preds == tgt
        tok_correct = correct[:, :half].sum().item()
        tok_total = correct[:, :half].numel()
        seq_correct = correct[:, :half].all(dim=-1)

        total_tok_correct += tok_correct
        total_tok += tok_total
        total_seq_correct += seq_correct.sum().item()
        total_seq += bs

        for c in range(4):
            mask = chains == c
            if mask.any():
                chain_correct[c][0] += seq_correct[mask].sum().item()
                chain_correct[c][1] += mask.sum().item()

        remaining -= bs

    tok_acc = total_tok_correct / total_tok
    seq_acc = total_seq_correct / total_seq
    chain_accs = {}
    for c in range(4):
        if chain_correct[c][1] > 0:
            chain_accs[c] = chain_correct[c][0] / chain_correct[c][1]
        else:
            chain_accs[c] = None

    return {
        "token_acc": tok_acc,
        "seq_acc": seq_acc,
        "chain_accs": chain_accs,
        "n_samples": total_seq,
    }


@torch.no_grad()
def corruption_recovery_at_T(model, T_eval, n_samples=N_EVAL, batch_size=512):
    """D17-style single-position corruption test at a given T."""
    model.eval()
    half = CFG["seq_len"] // 2
    recovery_rates = []

    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_addition_batch(bs, CFG["seq_len"], CFG["vocab_size"])
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        context = model.encode(src)
        s = model.init_state(bs, CFG["seq_len"], DEVICE)
        for _ in range(CFG["T"]):
            s, _ = model.dynamics_step(s, context)

        clean_logits = model.readout_logits(s)
        clean_preds = clean_logits.argmax(dim=-1)
        clean_correct = (clean_preds[:, :half] == tgt[:, :half]).all(dim=-1)

        if clean_correct.sum() < 10:
            remaining -= bs
            continue

        s_clean = s[clean_correct]
        ctx_clean = context[clean_correct]
        tgt_clean = tgt[clean_correct]
        n_good = s_clean.shape[0]

        pos = 0
        s_corrupt = s_clean.clone()
        s_corrupt[:, pos, :] = torch.randn_like(s_corrupt[:, pos, :])

        s_recov = s_corrupt
        for _ in range(T_eval):
            s_recov, _ = model.dynamics_step(s_recov, ctx_clean)

        recov_logits = model.readout_logits(s_recov)
        recov_preds = recov_logits.argmax(dim=-1)
        pos_recovered = (recov_preds[:, pos] == tgt_clean[:, pos]).float().mean().item()
        recovery_rates.append((n_good, pos_recovered))

        remaining -= bs

    if not recovery_rates:
        return {"pos0_recovery": 0.0}
    total = sum(n for n, _ in recovery_rates)
    wavg = sum(n * r for n, r in recovery_rates) / total
    return {"pos0_recovery": wavg}


def run_track(track_name, track_type):
    """Train and evaluate one track at all T values."""
    print(f"\n{'='*60}")
    print(f"  TRACK: {track_name}")
    print(f"{'='*60}")

    set_seed(CFG["seed"])
    model = UESDModel(
        CFG["vocab_size"], CFG["d_model"], CFG["n_heads"],
        CFG["d_ff"], CFG["n_enc_layers"], CFG["max_len"],
    ).to(DEVICE)
    print(f"  Params: {count_params(model)}")

    train_result = train(model, "addition", track_type, CFG, device=DEVICE)

    base_eval = evaluate_at_T(model, CFG["T"])
    print(f"  Baseline T={CFG['T']}: tok={base_eval['token_acc']:.4f} seq={base_eval['seq_acc']:.4f}")

    if base_eval["seq_acc"] < 0.95:
        print(f"  WARNING: Baseline accuracy too low ({base_eval['seq_acc']:.4f}), results may not be meaningful")

    results = {"baseline": base_eval, "step_ablation": {}, "corruption_recovery": {}}

    print(f"\n  --- Step Ablation (T=1..32) ---")
    for T_val in T_VALUES:
        eval_r = evaluate_at_T(model, T_val)
        results["step_ablation"][T_val] = eval_r
        chain_str = ", ".join(
            f"c{c}={'N/A' if v is None else f'{v:.3f}'}"
            for c, v in eval_r["chain_accs"].items()
        )
        print(f"    T={T_val:>2d}: tok={eval_r['token_acc']:.4f} seq={eval_r['seq_acc']:.4f} | {chain_str}")

    print(f"\n  --- Corruption Recovery at Various T ---")
    for T_recov in [1, 2, 5, 10, 20]:
        cr = corruption_recovery_at_T(model, T_recov)
        results["corruption_recovery"][T_recov] = cr
        print(f"    +{T_recov:>2d} recovery steps: pos0_recovery={cr['pos0_recovery']:.4f}")

    ratio = results["step_ablation"][1]["seq_acc"] / max(results["step_ablation"][10]["seq_acc"], 1e-9)
    results["falsification_ratio"] = ratio
    results["training_time"] = train_result["elapsed_s"]

    print(f"\n  FALSIFICATION METRIC: seq_acc(T=1)/seq_acc(T=10) = {ratio:.4f}")
    if ratio >= 0.98:
        print(f"  >>> THESIS WEAKENED: T=1 nearly matches T=10 <<<")
    elif ratio >= 0.80:
        print(f"  >>> PARTIAL: T=1 captures {ratio*100:.1f}% -- dynamics help but modestly <<<")
    else:
        print(f"  >>> THESIS SUPPORTED: T=1 captures only {ratio*100:.1f}% -- dynamics essential <<<")

    return results


def main():
    results = {"config": CFG, "T_values": T_VALUES, "tracks": {}}

    results["tracks"]["dynamics_ce"] = run_track("dynamics_ce", "dynamics_ce")
    results["tracks"]["e5"] = run_track("e5", "e5")

    print(f"\n{'='*60}")
    print(f"  D19 STEP ABLATION SUMMARY")
    print(f"{'='*60}")
    for name, track in results["tracks"].items():
        ratio = track["falsification_ratio"]
        t1 = track["step_ablation"][1]["seq_acc"]
        t10 = track["step_ablation"][10]["seq_acc"]
        print(f"  {name}:")
        print(f"    seq_acc(T=1)={t1:.4f}, seq_acc(T=10)={t10:.4f}, ratio={ratio:.4f}")
        if ratio >= 0.98:
            print(f"    VERDICT: THESIS WEAKENED")
        elif ratio >= 0.80:
            print(f"    VERDICT: PARTIAL SUPPORT (dynamics modestly helpful)")
        else:
            print(f"    VERDICT: THESIS SUPPORTED (dynamics essential)")

        print(f"    Step-accuracy curve:")
        for T_val in T_VALUES:
            sa = track["step_ablation"][T_val]["seq_acc"]
            bar = "#" * int(sa * 40)
            print(f"      T={T_val:>2d}: {sa:.4f} |{bar}")

    out_path = "experiments/06_uesd/results/exp_d19_step_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
