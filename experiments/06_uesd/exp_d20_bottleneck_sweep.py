"""D20: Vocabulary/Bottleneck Sweep Falsification Test

FALSIFICATION TARGET: "Bandwidth bottleneck requires iterative bypass" thesis.
If accuracy and recovery metrics are flat while vocab_size (hence log2(V)
channel capacity) varies by 4x+, the bottleneck story is unsupported.

Design:
  1. Sweep V in {16, 32, 64, 128, 256} with fixed d=128, T=10
  2. For each V, train CE-dynamics (20K steps, 3 seeds)
  3. Measure: accuracy, step-curve (T=1 vs T=10), Lyapunov, recovery
  4. Key metric: does accuracy or step-dependence scale with log2(V)?

Source: Codex meta-analysis falsification test #6 + experiment #3
"""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from shared.model import UESDModel
from shared.training import train, count_params, set_seed
from shared.data import generate_addition_batch
from shared.diagnostics import run_all_diagnostics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

V_VALUES = [16, 32, 64, 128, 256]
SEEDS = [42, 1337, 2024]
BASE_CFG = {
    "d_model": 128, "n_heads": 4, "d_ff": 512,
    "n_enc_layers": 2, "max_len": 32, "T": 10,
    "training_steps": 20000, "batch_size": 256, "seq_len": 8,
    "lr": 3e-4, "log_interval": 5000,
}
N_EVAL = 4096


@torch.no_grad()
def evaluate_at_T(model, vocab_size, T_eval, n_samples=N_EVAL, batch_size=512):
    model.eval()
    total_tok_correct = 0
    total_tok = 0
    total_seq_correct = 0
    total_seq = 0
    half = BASE_CFG["seq_len"] // 2

    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_addition_batch(bs, BASE_CFG["seq_len"], vocab_size)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        logits = model(src, T_eval)
        preds = logits.argmax(dim=-1)
        correct = preds == tgt
        total_tok_correct += correct[:, :half].sum().item()
        total_tok += correct[:, :half].numel()
        total_seq_correct += correct[:, :half].all(dim=-1).sum().item()
        total_seq += bs
        remaining -= bs

    return {
        "token_acc": total_tok_correct / total_tok,
        "seq_acc": total_seq_correct / total_seq,
    }


@torch.no_grad()
def measure_dynamics_metrics(model, vocab_size, n_samples=2048, batch_size=256):
    """Measure Lyapunov-like and recovery metrics."""
    model.eval()
    half = BASE_CFG["seq_len"] // 2
    T = BASE_CFG["T"]

    src, tgt = generate_addition_batch(n_samples, BASE_CFG["seq_len"], vocab_size)
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)

    context = model.encode(src)
    s = model.init_state(n_samples, BASE_CFG["seq_len"], DEVICE)
    norms = []
    for _ in range(T):
        s_new, norm = model.dynamics_step(s, context)
        norms.append(norm.item())
        s = s_new

    update_norms = norms
    contraction = norms[-1] / max(norms[0], 1e-9)

    s_clean = s.clone()
    s_corrupt = s.clone()
    s_corrupt[:, 0, :] = torch.randn_like(s_corrupt[:, 0, :])

    s_recov = s_corrupt
    for _ in range(10):
        s_recov, _ = model.dynamics_step(s_recov, context)

    clean_logits = model.readout_logits(s_clean)
    recov_logits = model.readout_logits(s_recov)
    clean_preds = clean_logits.argmax(dim=-1)
    recov_preds = recov_logits.argmax(dim=-1)
    clean_correct = (clean_preds[:, :half] == tgt[:, :half]).all(dim=-1)

    if clean_correct.sum() > 0:
        pos0_match = (recov_preds[clean_correct, 0] == tgt[clean_correct, 0]).float().mean().item()
    else:
        pos0_match = 0.0

    return {
        "update_norms": [float(n) for n in update_norms],
        "contraction_ratio": contraction,
        "recovery_10step": pos0_match,
    }


def run_vocab_size(V, seed):
    """Train and evaluate one (V, seed) configuration."""
    print(f"\n  V={V}, seed={seed}")
    set_seed(seed)
    cfg = {**BASE_CFG, "vocab_size": V, "seed": seed}

    model = UESDModel(V, cfg["d_model"], cfg["n_heads"], cfg["d_ff"],
                      cfg["n_enc_layers"], cfg["max_len"]).to(DEVICE)
    print(f"    Params: {count_params(model)}")

    train_result = train(model, "addition", "dynamics_ce", cfg, device=DEVICE)

    result = {
        "V": V, "log2_V": float(torch.tensor(V).float().log2()),
        "seed": seed, "training_time": train_result["elapsed_s"],
    }

    eval_T10 = evaluate_at_T(model, V, 10)
    eval_T1 = evaluate_at_T(model, V, 1)
    eval_T3 = evaluate_at_T(model, V, 3)
    result["acc_T10"] = eval_T10
    result["acc_T1"] = eval_T1
    result["acc_T3"] = eval_T3
    result["step_dependence"] = eval_T10["seq_acc"] - eval_T1["seq_acc"]

    dynamics = measure_dynamics_metrics(model, V)
    result["dynamics"] = dynamics

    print(f"    T=10: tok={eval_T10['token_acc']:.4f} seq={eval_T10['seq_acc']:.4f}")
    print(f"    T=1:  tok={eval_T1['token_acc']:.4f} seq={eval_T1['seq_acc']:.4f}")
    print(f"    T=3:  tok={eval_T3['token_acc']:.4f} seq={eval_T3['seq_acc']:.4f}")
    print(f"    Step dependence (T10-T1): {result['step_dependence']:.4f}")
    print(f"    Contraction: {dynamics['contraction_ratio']:.4f}")
    print(f"    Recovery +10: {dynamics['recovery_10step']:.4f}")

    return result


def main():
    results = {"config": BASE_CFG, "V_values": V_VALUES, "seeds": SEEDS, "runs": []}

    for V in V_VALUES:
        print(f"\n{'='*60}")
        print(f"  VOCAB SIZE V={V} (log2={torch.tensor(V).float().log2():.1f} bits)")
        print(f"{'='*60}")
        for seed in SEEDS:
            r = run_vocab_size(V, seed)
            results["runs"].append(r)

    print(f"\n{'='*60}")
    print(f"  D20 BOTTLENECK SWEEP SUMMARY")
    print(f"{'='*60}")

    for V in V_VALUES:
        v_runs = [r for r in results["runs"] if r["V"] == V]
        mean_seq = sum(r["acc_T10"]["seq_acc"] for r in v_runs) / len(v_runs)
        mean_dep = sum(r["step_dependence"] for r in v_runs) / len(v_runs)
        mean_recov = sum(r["dynamics"]["recovery_10step"] for r in v_runs) / len(v_runs)
        log2v = v_runs[0]["log2_V"]
        print(f"  V={V:>3d} (log2={log2v:.1f}): seq_acc={mean_seq:.4f}, "
              f"step_dep={mean_dep:.4f}, recovery={mean_recov:.4f}")

    v_accs = []
    v_deps = []
    for V in V_VALUES:
        v_runs = [r for r in results["runs"] if r["V"] == V]
        v_accs.append(sum(r["acc_T10"]["seq_acc"] for r in v_runs) / len(v_runs))
        v_deps.append(sum(r["step_dependence"] for r in v_runs) / len(v_runs))

    acc_range = max(v_accs) - min(v_accs)
    dep_range = max(v_deps) - min(v_deps)
    print(f"\n  Accuracy range across V: {acc_range:.4f}")
    print(f"  Step-dependence range across V: {dep_range:.4f}")

    if acc_range < 0.05 and dep_range < 0.05:
        print(f"  >>> THESIS WEAKENED: metrics flat across 4x V range <<<")
    else:
        print(f"  >>> THESIS SUPPORTED: metrics vary with bottleneck capacity <<<")

    out_path = "experiments/06_uesd/results/exp_d20_bottleneck_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
