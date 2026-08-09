"""
Experiment D23: Carry-Depth Phase Diagram

Tests how the compute window scales with problem difficulty (carry-chain length).

Key question: does the minimum T needed for >99% accuracy increase with L?
- If yes → dynamics provide genuine depth-dependent computation
- If no (flat) → computation is purely parallel, T buys precision not depth
- If the window NARROWS at larger L → finite-horizon limitation worsens with scale

Design:
- L = {4, 8, 12, 16, 20, 24} (carry depths 2, 4, 6, 8, 10, 12)
- Training: CE-dynamics at T=10, 30K steps
- Plus: best D22 variant (if available) at each L
- Eval: T = {1, 2, 3, 5, 8, 10, 15, 20, 32, 48}
- 1 seed per configuration (expand if interesting)

PREDICTIONS:
1. Minimum T for 99% scales sublinearly with L (parallel computation hypothesis)
2. Compute window width is roughly constant (5-15 useful steps)
3. L=24 (carry depth 12) may require T>10 — dynamics necessity scales with difficulty
4. Larger L amplifies D22-style recovery deficits (if they exist)
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

SEED = 42
TRAINING_STEPS = 30000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
TRAIN_T = 10

SEQ_LENS = [4, 8, 12, 16, 20, 24]
EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 32, 48]
EVAL_SAMPLES = 4096
RECOVERY_SIGMAS = [0.01, 0.1, 0.5, 1.0]  # fraction of state norm
RECOVERY_EXTRA_STEPS = [1, 5, 10, 20]

D22_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d22_robust_dynamics.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d23_carry_depth_scaling.json"


def full_seed(seed):
    """Seed torch + Python random for determinism."""
    set_seed(seed)
    random.seed(seed)


def save_checkpoint(results):
    """Atomically save results to disk after each config."""
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    """Load existing results for resume support."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


def load_d22_best_variant():
    """Check D22 results and return the best training variant name."""
    if not D22_RESULTS_PATH.exists():
        return None
    with open(D22_RESULTS_PATH) as f:
        d22 = json.load(f)
    best_variant = None
    best_acc = 0.0
    for variant in ["variable_t", "denoising", "combined"]:
        if variant in d22 and "summary" in d22[variant]:
            acc = d22[variant]["summary"].get("mean_T10_seq", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_variant = variant
    return best_variant if best_acc > 0.5 else None


def train_baseline(model, device, seq_len):
    """Standard CE-dynamics training at fixed T=10."""
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
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
            with torch.no_grad():
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


def train_denoising(model, device, seq_len):
    """D22-style denoising-robust training."""
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    denoise_steps = [3, 4, 5, 6, 7]
    sigma_frac = 0.3
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        inject_step = random.choice(denoise_steps)
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(1, TRAIN_T + 1):
            s, _ = model.dynamics_step(s, context)
            if t == inject_step:
                noise_scale = sigma_frac * s.detach().norm(dim=-1, keepdim=True)
                noise = torch.randn_like(s) * noise_scale
                s = s + noise

        logits = model.readout_logits(s)
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
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


def get_t_range(seq_len):
    """Scale variable-T range with carry-depth so larger L sees higher T during training."""
    base = [4, 6, 8, 10, 12, 14, 16]
    carry_depth = seq_len // 2
    if carry_depth > 8:
        extra = list(range(18, min(carry_depth * 3, 48) + 1, 2))
        return base + extra
    return base


def train_variable_t(model, device, seq_len):
    """D22-style variable-T curriculum."""
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    t_range = get_t_range(seq_len)
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(t_range)
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
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
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


def train_combined(model, device, seq_len):
    """D22-style combined (variable-T + denoising)."""
    full_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    t_range = get_t_range(seq_len)
    denoise_steps = [3, 4, 5, 6, 7]
    sigma_frac = 0.3
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(t_range)
        inject_step = random.choice([t for t in denoise_steps if t < T])
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)

        for t in range(1, T + 1):
            s, _ = model.dynamics_step(s, context)
            if t == inject_step:
                noise_scale = sigma_frac * s.detach().norm(dim=-1, keepdim=True)
                noise = torch.randn_like(s) * noise_scale
                s = s + noise

        logits = model.readout_logits(s)
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
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


TRAIN_FNS = {
    "baseline": train_baseline,
    "denoising": train_denoising,
    "variable_t": train_variable_t,
    "combined": train_combined,
}


@torch.no_grad()
def evaluate_step_ablation(model, device, seq_len):
    """Accuracy at each T value, with per-carry-position breakdown."""
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    results = {}
    for T in EVAL_T_VALUES:
        logits = model(src, T)
        preds = logits[:, :half].argmax(dim=-1)
        tok_acc = (preds == tgt_result).float().mean().item()
        seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()

        per_pos = {}
        for p in range(half):
            per_pos[f"c{p}"] = (preds[:, p] == tgt_result[:, p]).float().mean().item()

        results[f"T={T}"] = {
            "tok_acc": round(tok_acc, 4),
            "seq_acc": round(seq_acc, 4),
            **{k: round(v, 3) for k, v in per_pos.items()},
        }

    return results


@torch.no_grad()
def evaluate_recovery(model, device, seq_len):
    """Perturb at T=10, run extra steps, measure recovery."""
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)
    for _ in range(10):
        s, _ = model.dynamics_step(s, context)

    state_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    results = {}
    for sigma in RECOVERY_SIGMAS:
        noise = torch.randn_like(s) * (sigma * state_norm)
        s_noisy = s + noise

        preds_noisy = model.readout_logits(s_noisy)[:, :half].argmax(dim=-1)
        wa_at_0 = 1.0 - (preds_noisy == tgt_result).all(dim=1).float().mean().item()

        s_extra = s_noisy.clone()
        extra_results = {}
        prev = 0
        for extra in RECOVERY_EXTRA_STEPS:
            for _ in range(extra - prev):
                s_extra, _ = model.dynamics_step(s_extra, context)
            prev = extra
            preds_extra = model.readout_logits(s_extra)[:, :half].argmax(dim=-1)
            wa_after = 1.0 - (preds_extra == tgt_result).all(dim=1).float().mean().item()
            recovery = wa_at_0 - wa_after
            extra_results[f"+{extra}"] = {
                "WA": round(wa_after, 4),
                "recovery": round(recovery, 4),
            }

        results[f"sigma={sigma}"] = {
            "WA_at_0": round(wa_at_0, 4),
            **extra_results,
        }

    return results


def evaluate_encoder_only_baseline(device, seq_len):
    """Quick encoder-only test at this L for reference."""
    from shared.model import EncoderOnlyAblation
    full_seed(SEED)
    model = EncoderOnlyAblation(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2

    model.train()
    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    logits = model(src)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt[:, :half]).float().mean().item()
    seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
    del model
    torch.cuda.empty_cache()
    return {"tok_acc": round(tok_acc, 4), "seq_acc": round(seq_acc, 4)}


def run_seq_len(seq_len, variant_name, train_fn, device):
    """Train and evaluate one (seq_len, variant) configuration."""
    print(f"\n  L={seq_len}, variant={variant_name}, seed={SEED}", flush=True)

    full_seed(SEED)
    model = make_model(device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    train_time, best_train_acc = train_fn(model, device, seq_len)

    print("    Evaluating step ablation...", flush=True)
    step_abl = evaluate_step_ablation(model, device, seq_len)
    for k, v in step_abl.items():
        print(f"      {k}: seq={v['seq_acc']:.4f}", flush=True)

    print("    Evaluating recovery...", flush=True)
    recovery = evaluate_recovery(model, device, seq_len)
    for sigma_key, sigma_data in recovery.items():
        best_rec = max(v["recovery"] for k, v in sigma_data.items() if k != "WA_at_0")
        print(f"      {sigma_key}: WA@0={sigma_data['WA_at_0']:.4f} best_recovery={best_rec:+.4f}",
              flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "carry_depth": seq_len // 2,
        "variant": variant_name,
        "seed": SEED,
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_train_acc, 4),
        "step_ablation": step_abl,
        "recovery": recovery,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    d22_best = load_d22_best_variant()
    variants_to_run = ["baseline"]
    if d22_best and d22_best in TRAIN_FNS:
        variants_to_run.append(d22_best)
        print(f"\nD22 best variant: {d22_best} — will train this alongside baseline",
              flush=True)
    else:
        print(f"\nNo D22 results found — running baseline only", flush=True)

    # Resume from checkpoint if available
    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed_keys = {(r["seq_len"], r["variant"]) for r in all_results["runs"]}
        completed_enc = set(all_results.get("encoder_baselines", {}).keys())
        print(f"\n  RESUMING: {len(all_results['runs'])} runs + {len(completed_enc)} encoder baselines already done",
              flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D23: Carry-depth phase diagram — how compute window scales with L",
            "d22_best_variant": d22_best,
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "train_T": TRAIN_T,
                "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
                "seq_lens": SEQ_LENS,
            },
            "runs": [],
            "encoder_baselines": {},
        }
        completed_keys = set()
        completed_enc = set()

    # Train encoder-only at each L for reference
    print(f"\n{'=' * 60}", flush=True)
    print(f"  ENCODER-ONLY BASELINES", flush=True)
    print(f"{'=' * 60}", flush=True)
    for seq_len in SEQ_LENS:
        key = f"L={seq_len}"
        if key in completed_enc:
            print(f"  L={seq_len}... SKIPPED (already done)", flush=True)
            continue
        print(f"  L={seq_len}...", flush=True)
        enc_result = evaluate_encoder_only_baseline(device, seq_len)
        all_results["encoder_baselines"][key] = enc_result
        save_checkpoint(all_results)
        print(f"    Encoder-only: tok={enc_result['tok_acc']:.4f} seq={enc_result['seq_acc']:.4f}",
              flush=True)

    # Main runs
    for variant_name in variants_to_run:
        train_fn = TRAIN_FNS[variant_name]
        print(f"\n{'=' * 60}", flush=True)
        print(f"  VARIANT: {variant_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seq_len in SEQ_LENS:
            if (seq_len, variant_name) in completed_keys:
                print(f"\n  L={seq_len}, variant={variant_name} — SKIPPED (already done)",
                      flush=True)
                continue
            result = run_seq_len(seq_len, variant_name, train_fn, device)
            all_results["runs"].append(result)
            save_checkpoint(all_results)

    # Phase diagram summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D23 PHASE DIAGRAM", flush=True)
    print(f"{'=' * 60}", flush=True)

    for variant_name in variants_to_run:
        print(f"\n  Variant: {variant_name}", flush=True)
        header = f"  {'L':>3} {'depth':>5} {'train':>6}"
        for T in EVAL_T_VALUES:
            header += f" {'T='+str(T):>6}"
        header += f" {'enc_only':>8}"
        print(header, flush=True)
        print("  " + "-" * len(header), flush=True)

        for seq_len in SEQ_LENS:
            run = next((r for r in all_results["runs"]
                       if r["seq_len"] == seq_len and r["variant"] == variant_name), None)
            if run is None:
                continue

            enc = all_results["encoder_baselines"].get(f"L={seq_len}", {})
            line = f"  {seq_len:>3} {seq_len//2:>5} {run['best_train_acc']:>6.3f}"
            for T in EVAL_T_VALUES:
                acc = run["step_ablation"].get(f"T={T}", {}).get("seq_acc", 0)
                line += f" {acc:>6.3f}"
            line += f" {enc.get('seq_acc', 0):>8.3f}"
            print(line, flush=True)

    # Compute window analysis
    print(f"\n  COMPUTE WINDOW ANALYSIS:", flush=True)
    for variant_name in variants_to_run:
        print(f"\n  Variant: {variant_name}", flush=True)
        for seq_len in SEQ_LENS:
            run = next((r for r in all_results["runs"]
                       if r["seq_len"] == seq_len and r["variant"] == variant_name), None)
            if run is None:
                continue

            accs = [(T, run["step_ablation"].get(f"T={T}", {}).get("seq_acc", 0))
                    for T in EVAL_T_VALUES]
            min_t_99 = next((T for T, a in accs if a >= 0.99), None)
            max_t_99 = None
            for T, a in reversed(accs):
                if a >= 0.99:
                    max_t_99 = T
                    break
            peak_t = max(accs, key=lambda x: x[1])

            window_str = f"[{min_t_99}-{max_t_99}]" if min_t_99 and max_t_99 else "none"
            print(f"    L={seq_len}: min_T_99={min_t_99} max_T_99={max_t_99} "
                  f"window={window_str} peak=T={peak_t[0]}({peak_t[1]:.4f})", flush=True)

    # Recovery scaling
    print(f"\n  RECOVERY SCALING:", flush=True)
    for variant_name in variants_to_run:
        print(f"\n  Variant: {variant_name}", flush=True)
        for seq_len in SEQ_LENS:
            run = next((r for r in all_results["runs"]
                       if r["seq_len"] == seq_len and r["variant"] == variant_name), None)
            if run is None:
                continue

            rec_vals = []
            for sigma_key, sigma_data in run["recovery"].items():
                for k, v in sigma_data.items():
                    if k != "WA_at_0":
                        rec_vals.append(v["recovery"])
            mean_rec = sum(rec_vals) / len(rec_vals) if rec_vals else 0
            print(f"    L={seq_len}: mean_recovery={mean_rec:+.4f}", flush=True)

    # Final save
    save_checkpoint(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
