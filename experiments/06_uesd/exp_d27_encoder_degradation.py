"""
Experiment D27: Encoder Degradation — Testing the Channel-Coding Interpretation

Proposition 22 claims dynamics act as an iterative decoder in a channel-coding
framework. This experiment tests the prediction: degrading encoder quality should
increase T_min proportionally to the information deficit, with a minimum quality
threshold below which dynamics cannot compensate.

Two approaches to encoder degradation:
1. Noise injection: add AWGN to encoder context c at inference time (controlled SNR)
2. Layer removal: train with 2 encoder layers, evaluate with 0/1 layers

Additionally tests the "When Errors Are Features" prediction: moderate encoder
degradation at L=8/L=12 (strong encoder) may IMPROVE dynamics accuracy by
preventing co-adaptation (analogous to dropout).

Cross-attention ablation (Codex recommendation): remove cross-attention re-reading
at inference to test whether the dynamics need iterative channel access (BP-like)
or can work from a single encoder snapshot.

PREDICTIONS:
1. T_min increases with noise level (more noise → more iterations needed)
2. There exists a noise threshold where dynamics fail (channel capacity limit)
3. At L=8, moderate noise (sigma=0.1-0.3) may IMPROVE accuracy (anti-co-adaptation)
4. Removing cross-attention re-reading collapses performance for high-noise regime
   but has minimal effect for low-noise regime (already decoded in first few steps)
5. Step-accuracy "waterfall" steepens with L (longer codes → sharper waterfall)

SEEDS: [42, 1337, 2024] for statistical robustness
"""
import json
import math
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

SEEDS = [42, 1337, 2024]
TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 32

T_RANGE = [4, 6, 8, 10, 12, 14, 16]

EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20]
EVAL_SAMPLES = 4096
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

L_VALUES = [8, 12, 24]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d27_encoder_degradation.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)


def save_checkpoint(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(seq_len, device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


def train_variable_t(model, device, seed, seq_len):
    """Variable-T training (D22 protocol, standard for all D27 models)."""
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(T_RANGE)
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
    return elapsed, best_acc


@torch.no_grad()
def evaluate_with_noise(model, device, seq_len, noise_sigma, T):
    """Evaluate with Gaussian noise injected into encoder context.
    Noise is scaled per-position by ||c_i|| so sigma controls relative SNR.
    """
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)

    if noise_sigma > 0:
        ctx_norm = context.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(context) * (noise_sigma * ctx_norm)
        context = context + noise

    s = model.init_state(src.size(0), src.size(1), device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)

    logits = model.readout_logits(s)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
    return tok_acc, seq_acc


@torch.no_grad()
def evaluate_no_crossattn_reread(model, device, seq_len, noise_sigma, T):
    """Evaluate with cross-attention sublayer fully disabled after step 1.

    Monkey-patches the dynamics layer's _mha_block to return zeros,
    completely removing the cross-attention residual contribution.
    This is a strict architectural ablation, not just zeroed input.
    """
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)

    if noise_sigma > 0:
        ctx_norm = context.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(context) * (noise_sigma * ctx_norm)
        context = context + noise

    s = model.init_state(src.size(0), src.size(1), device)

    # First step: normal (with cross-attention)
    s, _ = model.dynamics_step(s, context)

    # Disable cross-attention sublayer entirely via monkey-patch
    original_mha_block = model.dynamics._mha_block
    model.dynamics._mha_block = lambda x, *args, **kwargs: torch.zeros_like(x)
    try:
        for _ in range(T - 1):
            s, _ = model.dynamics_step(s, context)
    finally:
        model.dynamics._mha_block = original_mha_block

    logits = model.readout_logits(s)
    preds = logits[:, :half].argmax(dim=-1)
    tok_acc = (preds == tgt_result).float().mean().item()
    seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
    return tok_acc, seq_acc


@torch.no_grad()
def evaluate_step_profile(model, device, seq_len, noise_sigma):
    """Get step-by-step accuracy profile (the "waterfall curve")."""
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_result = tgt[:, :half]

    context = model.encode(src)

    if noise_sigma > 0:
        ctx_norm = context.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(context) * (noise_sigma * ctx_norm)
        context = context + noise

    s = model.init_state(src.size(0), src.size(1), device)
    profile = {}

    for t in range(1, 21):
        s, _ = model.dynamics_step(s, context)
        if t in EVAL_T_VALUES:
            logits = model.readout_logits(s)
            preds = logits[:, :half].argmax(dim=-1)
            seq_acc = (preds == tgt_result).all(dim=1).float().mean().item()
            profile[f"T={t}"] = round(seq_acc, 4)

    return profile


@torch.no_grad()
def measure_inter_position_correlation(model, device, seq_len):
    """Measure average correlation between adjacent position states at each step.

    Channel-coding prediction: correlation should approach tanh(1/2)=0.462
    at the optimal step (where accuracy first saturates).
    """
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", min(EVAL_SAMPLES, 1024), seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    context = model.encode(src)
    s = model.init_state(src.size(0), src.size(1), device)

    correlations = {}
    for t in range(1, 16):
        s, _ = model.dynamics_step(s, context)

        # Compute mean correlation between adjacent output positions
        s_out = s[:, :half]  # only output positions
        if s_out.size(1) < 2:
            continue

        # Normalize per-position
        s_norm = F.normalize(s_out, dim=-1)
        cos_sims = []
        for i in range(s_norm.size(1) - 1):
            cos = (s_norm[:, i] * s_norm[:, i+1]).sum(dim=-1)
            cos_sims.append(cos.mean().item())

        mean_corr = sum(cos_sims) / len(cos_sims)
        correlations[f"t={t}"] = round(mean_corr, 4)

    return correlations


def run_one_config(seq_len, seed, device):
    """Train model for one (L, seed) and run all evaluations."""
    print(f"\n  L={seq_len}, seed={seed}", flush=True)

    full_seed(seed)
    model = make_model(seq_len, device)
    params = count_params(model)
    print(f"    Params: {params}", flush=True)

    train_time, best_acc = train_variable_t(model, device, seed, seq_len)
    print(f"    Training done in {train_time:.1f}s | Best acc: {best_acc:.4f}", flush=True)

    # Phase 1: Noise sweep at T=10
    print("    Phase 1: Noise sweep at T=10...", flush=True)
    noise_results = {}
    for sigma in NOISE_LEVELS:
        tok, seq = evaluate_with_noise(model, device, seq_len, sigma, T=10)
        noise_results[f"sigma={sigma}"] = {"tok_acc": round(tok, 4), "seq_acc": round(seq, 4)}
        print(f"      sigma={sigma}: seq={seq:.4f}", flush=True)

    # Phase 2: Step profiles at different noise levels
    print("    Phase 2: Step profiles (waterfall curves)...", flush=True)
    waterfall = {}
    for sigma in [0.0, 0.1, 0.3, 1.0]:
        profile = evaluate_step_profile(model, device, seq_len, sigma)
        waterfall[f"sigma={sigma}"] = profile
        print(f"      sigma={sigma}: {profile}", flush=True)

    # Phase 3: Cross-attention ablation
    print("    Phase 3: Cross-attention re-read ablation...", flush=True)
    crossattn_results = {}
    for sigma in [0.0, 0.1, 0.3]:
        tok_normal, seq_normal = evaluate_with_noise(model, device, seq_len, sigma, T=10)
        tok_no_reread, seq_no_reread = evaluate_no_crossattn_reread(
            model, device, seq_len, sigma, T=10
        )
        crossattn_results[f"sigma={sigma}"] = {
            "normal": round(seq_normal, 4),
            "no_reread": round(seq_no_reread, 4),
            "delta": round(seq_normal - seq_no_reread, 4),
        }
        print(f"      sigma={sigma}: normal={seq_normal:.4f} no_reread={seq_no_reread:.4f} "
              f"delta={seq_normal - seq_no_reread:+.4f}", flush=True)

    # Phase 4: Inter-position correlation (Nishimori test)
    print("    Phase 4: Inter-position correlation...", flush=True)
    correlations = measure_inter_position_correlation(model, device, seq_len)
    nishimori_target = 0.4622
    closest_step = min(correlations.keys(),
                       key=lambda k: abs(correlations[k] - nishimori_target))
    print(f"      Correlations: {correlations}", flush=True)
    print(f"      Closest to tanh(1/2)={nishimori_target}: {closest_step} "
          f"= {correlations[closest_step]:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "L": seq_len,
        "D": seq_len // 2,
        "seed": seed,
        "params": params,
        "train_time_s": round(train_time, 1),
        "best_train_acc": round(best_acc, 4),
        "noise_sweep": noise_results,
        "waterfall_curves": waterfall,
        "crossattn_ablation": crossattn_results,
        "inter_position_correlation": correlations,
        "nishimori_closest_step": closest_step,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed_keys = {(r["L"], r["seed"]) for r in all_results["runs"]}
        print(f"\n  RESUMING: {len(all_results['runs'])} runs already done", flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D27: Encoder degradation — testing channel-coding interpretation (Prop 22)",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                "lr": LR, "t_range": T_RANGE, "noise_levels": NOISE_LEVELS,
                "l_values": L_VALUES, "seeds": SEEDS,
            },
            "predictions": {
                "1": "T_min increases with noise level",
                "2": "Noise threshold exists where dynamics fail (channel capacity limit)",
                "3": "At L=8, moderate noise (sigma=0.1-0.3) may IMPROVE accuracy",
                "4": "Removing cross-attn re-reading collapses high-noise but not low-noise",
                "5": "Waterfall steepens with L (longer code = sharper transition)",
            },
            "runs": [],
        }
        completed_keys = set()

    for seq_len in L_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  SEQUENCE LENGTH L={seq_len} (carry depth D={seq_len // 2})", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seed in SEEDS:
            if (seq_len, seed) in completed_keys:
                print(f"\n  L={seq_len} seed={seed} — SKIPPED (already done)", flush=True)
                continue
            result = run_one_config(seq_len, seed, device)
            all_results["runs"].append(result)
            save_checkpoint(all_results)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"  D27 ENCODER DEGRADATION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for seq_len in L_VALUES:
        runs = [r for r in all_results["runs"] if r["L"] == seq_len]
        if not runs:
            continue

        print(f"\n  L={seq_len} (D={seq_len // 2}):", flush=True)

        # Noise sweep summary
        for sigma in [0.0, 0.1, 0.3, 1.0]:
            accs = [r["noise_sweep"].get(f"sigma={sigma}", {}).get("seq_acc", 0) for r in runs]
            mean_acc = sum(accs) / len(accs) if accs else 0
            print(f"    sigma={sigma}: mean_seq={mean_acc:.4f}", flush=True)

        # Cross-attention ablation summary
        for sigma in [0.0, 0.1, 0.3]:
            deltas = [r["crossattn_ablation"].get(f"sigma={sigma}", {}).get("delta", 0)
                      for r in runs]
            mean_delta = sum(deltas) / len(deltas) if deltas else 0
            print(f"    crossattn delta at sigma={sigma}: {mean_delta:+.4f}", flush=True)

    # Prediction assessment
    print(f"\n  PREDICTION ASSESSMENT:", flush=True)

    # Pred 3: Does moderate noise improve L=8?
    l8_runs = [r for r in all_results["runs"] if r["L"] == 8]
    if l8_runs:
        base_accs = [r["noise_sweep"]["sigma=0.0"]["seq_acc"] for r in l8_runs]
        for sigma in [0.1, 0.2, 0.3]:
            noisy_accs = [r["noise_sweep"].get(f"sigma={sigma}", {}).get("seq_acc", 0)
                          for r in l8_runs]
            base_mean = sum(base_accs) / len(base_accs)
            noisy_mean = sum(noisy_accs) / len(noisy_accs)
            improvement = noisy_mean - base_mean
            status = "IMPROVED" if improvement > 0.001 else "NO CHANGE" if abs(improvement) < 0.001 else "DEGRADED"
            print(f"    L=8 sigma={sigma}: {status} ({improvement:+.4f})", flush=True)

    save_checkpoint(all_results)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    return all_results


if __name__ == "__main__":
    run()
