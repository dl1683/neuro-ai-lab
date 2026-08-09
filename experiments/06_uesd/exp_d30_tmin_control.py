"""
Experiment D30: T_min Control — Direct Test of Proposition 32

Prop 32 predicts that T_99 <= T_min for variable-T trained models, because the
CE gradient at T=T_min provides the binding constraint on readout convergence.

This experiment systematically varies T_min while keeping:
- L=8 (carry depth D=4, well-characterized from D23/D28)
- T_max scaled to maintain a fixed window width (T_max = T_min + 12)
- Architecture, optimizer, seed all identical

Configurations:
  A: T_range = {2, 4, 6, 8, 10, 12, 14}     (T_min=2)
  B: T_range = {4, 6, 8, 10, 12, 14, 16}     (T_min=4, D23 standard)
  C: T_range = {6, 8, 10, 12, 14, 16, 18}    (T_min=6)
  D: T_range = {8, 10, 12, 14, 16, 18, 20}   (T_min=8)
  E: Fixed T=10                                (baseline, no VT)

PREDICTIONS (Prop 32):
1. T_99(A) <= 2  (most aggressive contraction)
2. T_99(B) <= 4  (should match D23 result of T_99=3)
3. T_99(C) <= 6  (relaxed contraction)
4. T_99(D) <= 8  (most relaxed VT)
5. T_99(E) = 5   (fixed-T baseline, matches D23 L=8 baseline)
6. MONOTONIC: T_99(A) <= T_99(B) <= T_99(C) <= T_99(D)
7. Spectral radius rho should be ~independent of T_min (architectural, not training)

Falsification criteria:
- If T_99 does NOT increase monotonically with T_min → Prop 32 fails
- If T_99 >> T_min for any config → binding constraint mechanism wrong
- If rho varies significantly across configs → dynamics are T_min-dependent
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEED = 42
TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
SEQ_LEN = 8

CONFIGS = {
    "A_tmin2": {"t_range": list(range(2, 15, 2)), "label": "T_min=2"},
    "B_tmin4": {"t_range": list(range(4, 17, 2)), "label": "T_min=4 (standard)"},
    "C_tmin6": {"t_range": list(range(6, 19, 2)), "label": "T_min=6"},
    "D_tmin8": {"t_range": list(range(8, 21, 2)), "label": "T_min=8"},
    "E_fixed": {"t_range": None, "label": "Fixed T=10 (baseline)"},
}
FIXED_T = 10

EVAL_T_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
EVAL_SAMPLES = 4096
FP_T = 100
POWER_ITER_VECS = 10
POWER_ITER_STEPS = 50

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d30_tmin_control.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_results(results):
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


def train_model(model, device, config_name):
    cfg = CONFIGS[config_name]
    t_range = cfg["t_range"]
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = SEQ_LEN // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if t_range is not None:
            t_steps = random.choice(t_range)
        else:
            t_steps = FIXED_T

        logits = model(src, t_steps)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                eval_logits = model(src, FIXED_T)
                preds = eval_logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={t_steps:>2d} | "
                  f"Loss: {loss.item():.4f} | Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


@torch.no_grad()
def evaluate_step_ablation(model, device):
    model.eval()
    half = SEQ_LEN // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    results = {}
    for T in EVAL_T_VALUES:
        logits = model(src, T)
        preds = logits[:, :half].argmax(dim=-1)
        tok_acc = (preds == tgt_result).float().mean().item()
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
        results[f"T={T}"] = {
            "tok_acc": round(tok_acc, 6),
            "seq_acc": round(seq_acc, 6),
        }
    return results


def measure_spectral_radius(model, device):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 256, SEQ_LEN, VOCAB_SIZE)
    src = src.to(device)

    with torch.no_grad():
        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)
        for _ in range(FP_T):
            s, _ = model.dynamics_step(s, context)
        s_star = s.detach().clone()

    rhos = []
    for _ in range(POWER_ITER_VECS):
        v = torch.randn_like(s_star[:32])
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for _ in range(POWER_ITER_STEPS):
            with torch.enable_grad():
                s_pert = s_star[:32].detach().requires_grad_(True)
                s_next, _ = model.dynamics_step(s_pert, context[:32])
                jvp = torch.autograd.grad(
                    s_next, s_pert, grad_outputs=v, create_graph=False
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    std_rho = float(np.std(rhos))
    print(f"    Spectral radius: rho={mean_rho:.4f}+/-{std_rho:.4f}", flush=True)
    return {"mean": round(mean_rho, 4), "std": round(std_rho, 4)}


def compute_t99(step_ablation):
    for T in sorted(EVAL_T_VALUES):
        key = f"T={T}"
        if key in step_ablation and step_ablation[key]["seq_acc"] >= 0.99:
            return T
    return None


def run_one_config(config_name, device):
    cfg = CONFIGS[config_name]
    print(f"\n{'='*60}", flush=True)
    print(f"  Config {config_name}: {cfg['label']}", flush=True)
    print(f"  T_range: {cfg['t_range'] or f'fixed T={FIXED_T}'}", flush=True)
    print(f"{'='*60}", flush=True)

    full_seed(SEED)
    model = make_model(device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    train_time, best_acc = train_model(model, device, config_name)

    print("    Evaluating step ablation...", flush=True)
    step_abl = evaluate_step_ablation(model, device)
    t99 = compute_t99(step_abl)
    print(f"    T_99 = {t99}", flush=True)
    for k, v in step_abl.items():
        print(f"      {k}: seq={v['seq_acc']:.4f}", flush=True)

    print("    Measuring spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device)

    del model
    torch.cuda.empty_cache()

    t_min = cfg["t_range"][0] if cfg["t_range"] else FIXED_T
    return {
        "config": config_name,
        "label": cfg["label"],
        "t_range": cfg["t_range"],
        "t_min": t_min,
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_acc, 4),
        "step_ablation": step_abl,
        "T_99": t99,
        "spectral_radius": spectral,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed = {r["config"] for r in all_results["runs"]}
        print(f"\n  RESUMING: {len(completed)} configs already done", flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D30: T_min control experiment - tests Proposition 32",
            "seq_len": SEQ_LEN,
            "carry_depth": SEQ_LEN // 2,
            "architecture": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
            },
            "training": {
                "steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                "lr": LR, "seed": SEED,
            },
            "runs": [],
        }
        completed = set()

    for config_name in CONFIGS:
        if config_name in completed:
            print(f"\n  {config_name} already done, skipping", flush=True)
            continue
        result = run_one_config(config_name, device)
        all_results["runs"].append(result)
        save_results(all_results)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  D30 T_MIN CONTROL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n  {'Config':<12} {'T_min':>5} {'T_99':>4} {'T_99<=T_min?':>12} "
          f"{'rho':>6} {'train_acc':>9}", flush=True)
    print(f"  {'-'*52}", flush=True)

    vt_entries = []
    none_count = 0
    for r in all_results["runs"]:
        t_min = r["t_min"]
        t99 = r["T_99"]
        is_vt = "fixed" not in r["config"]
        if is_vt and t99 is None:
            none_count += 1
        satisfied = "YES" if t99 is not None and t99 <= t_min else ("DNF" if t99 is None else "NO")
        rho = r["spectral_radius"]["mean"]
        print(f"  {r['config']:<12} {t_min:>5} {str(t99) if t99 is not None else 'DNF':>4} "
              f"{satisfied:>12} {rho:>6.4f} {r['best_train_acc']:>9.4f}", flush=True)
        if is_vt:
            vt_entries.append((t_min, t99, r["config"]))

    # Prop 32 verdict
    print(f"\n  PROPOSITION 32 TESTS:", flush=True)

    if none_count > 0:
        print(f"    WARNING: {none_count} VT config(s) did not converge (T_99=DNF)", flush=True)

    # Test 1: T_99 <= T_min for all VT configs (None = FAIL)
    vt_pass = (len(vt_entries) > 0
               and none_count == 0
               and all(t99 <= t_min for t_min, t99, cfg in vt_entries))
    print(f"    T_99 <= T_min (all VT): {'PASS' if vt_pass else 'FAIL'}", flush=True)

    # Test 2: Monotonicity (requires all VT configs to have finite T_99)
    if none_count == 0 and len(vt_entries) >= 2:
        vt_sorted = sorted(vt_entries, key=lambda x: x[0])
        monotonic = all(vt_sorted[i][1] <= vt_sorted[i+1][1] for i in range(len(vt_sorted)-1))
    else:
        monotonic = False
    print(f"    Monotonic T_99 vs T_min: {'PASS' if monotonic else 'FAIL'}", flush=True)

    # Test 3: rho independence
    rhos = [r["spectral_radius"]["mean"] for r in all_results["runs"]]
    rho_range = max(rhos) - min(rhos)
    rho_independent = rho_range < 0.01
    print(f"    rho independence (range<0.01): {'PASS' if rho_independent else 'FAIL'} "
          f"(range={rho_range:.4f})", flush=True)

    # Overall
    all_pass = vt_pass and monotonic and rho_independent
    verdict = "CONFIRMED" if all_pass else "PARTIAL" if (vt_pass or monotonic) else "REFUTED"
    print(f"\n  PROP 32 VERDICT: {verdict}", flush=True)

    all_results["prop32_tests"] = {
        "t99_leq_tmin": vt_pass,
        "monotonic": monotonic,
        "rho_independent": rho_independent,
        "rho_range": round(rho_range, 4),
        "none_count": none_count,
        "verdict": verdict,
    }

    save_results(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    return all_results


if __name__ == "__main__":
    run()
