"""
Experiment D15: Nishimori Calibration Test

The UESD formalization (Section 8.3) makes the most specific quantitative
prediction in the entire framework: UESD dynamics should naturally approach
the Nishimori critical point, where:

  average confidence = average accuracy
  rho = tanh(1/2) ≈ 0.462

This is grounded in prior research from this lab showing that 7 independent
substrates (neural networks, spin glasses, biological circuits, etc.) all
converge to rho = tanh(1/2) with CV = 1% at their critical point.

The prediction for UESD: as dynamics iterate from random initialization
toward convergence, there exists a critical step t* where:
1. The average per-position readout confidence ≈ 0.462
2. At that step, the model is perfectly calibrated (ECE ≈ 0)
3. This holds across different training tracks (E5, CE-dynamics)

The test: at each dynamics step t, compute readout probabilities and measure
calibration. If the dynamics naturally pass through a Nishimori-like critical
point, that connects a 694K-param addition model to universal statistical
mechanics.

PREDICTIONS:
1. There exists t* where avg confidence ≈ 0.462 ± 0.05
2. ECE at t* is lower than at any other step (minimal calibration error)
3. This holds for both E5 and CE-dynamics
4. The critical step t* correlates with carry-chain length (harder
   problems take longer to reach criticality)
5. Encoder-only models do NOT show a clean Nishimori-like transition
   (no iterative dynamics to pass through criticality)
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, EncoderOnlyAblation
from shared.training import set_seed, count_params
from shared.data import generate_batch


NISHIMORI_RHO = np.tanh(0.5)  # ≈ 0.4621
N_CALIBRATION_BINS = 20
TAU_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]
MODEL_TAU = 0.1  # model's trained readout temperature — primary analysis uses this


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
        "warmup_steps": 5000,
        "seed": seed,
    }


def train_model(track, config, device):
    set_seed(config["seed"])
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    T = config["T"]
    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        context = model.encode(src)
        B, L = src.shape
        s = model.init_state(B, L, src.device)
        for _ in range(T):
            s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]

        if track == "dynamics_ce":
            loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                   tgt_r.reshape(-1))
        else:
            ce = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                 tgt_r.reshape(-1))
            s_next = model.dynamics(s, context)
            sc = (s_next - s).pow(2).sum(dim=-1).mean()
            eff_lam = min(step / config["warmup_steps"], 1.0)
            loss = ce + eff_lam * sc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            print(f"    Step {step:>6d}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    model.eval()
    return model


def compute_carries(src, vocab_size):
    B, L = src.shape
    half = L // 2
    a = src[:, 0::2][:, :half]
    b = src[:, 1::2][:, :half]
    carry = torch.zeros(B, dtype=torch.long, device=src.device)
    carry_in = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        carry = (s >= vocab_size).long()
        if i > 0:
            carry_in[:, i - 1] = carry
    max_chain = torch.zeros(B, dtype=torch.long, device=src.device)
    chain_len = torch.zeros(B, half, dtype=torch.long, device=src.device)
    for i in range(half - 2, -1, -1):
        has_carry = carry_in[:, i] == 1
        chain_len[has_carry, i] = 1 + chain_len[has_carry, i + 1]
    max_chain = chain_len.max(dim=1).values
    return max_chain


def _readout_probs(model, s, half, readout_tau=None):
    """Compute readout probabilities using the model's actual readout pipeline.

    Uses readout_proj + normalize + cosine sim (matching model.readout_logits),
    optionally overriding the temperature for sensitivity analysis.
    """
    h = model.readout_proj(s[:, :half, :])
    W = model.tok_emb.weight
    h = F.normalize(h, dim=-1)
    W = F.normalize(W, dim=-1)
    tau = readout_tau if readout_tau is not None else model.tau
    logits = torch.matmul(h, W.t()) / tau  # [B, half, V]
    return F.softmax(logits, dim=-1)


def _compute_ece_mce(conf_flat, corr_flat):
    """Standard ECE/MCE: bin by max predicted-class probability."""
    bin_edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for b_idx in range(N_CALIBRATION_BINS):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        mask = (conf_flat >= lo) & (conf_flat < hi)
        n = mask.sum()
        if n > 0:
            bin_accs.append(float(corr_flat[mask].mean()))
            bin_confs.append(float(conf_flat[mask].mean()))
            bin_counts.append(int(n))
        else:
            bin_accs.append(float('nan'))
            bin_confs.append(float('nan'))
            bin_counts.append(0)

    total = len(conf_flat)
    ece = sum(
        (bc / total) * abs(ba - bconf)
        for ba, bconf, bc in zip(bin_accs, bin_confs, bin_counts)
        if not np.isnan(ba)
    )
    mce = max(
        (abs(ba - bconf) for ba, bconf in zip(bin_accs, bin_confs)
         if not np.isnan(ba)),
        default=0.0,
    )
    return ece, mce, bin_accs, bin_confs, bin_counts


@torch.no_grad()
def calibration_analysis(model, eval_src, eval_tgt, config, device,
                         readout_tau=None):
    """Compute per-step calibration metrics for UESD dynamics.

    Uses the model's actual readout (readout_proj + cosine sim / tau).
    Confidence = max predicted-class probability (standard ECE definition).
    """
    T = config["T"]
    half = config["seq_len"] // 2

    context = model.encode(eval_src)
    B, L = eval_src.shape
    s = model.init_state(B, L, device)
    targets = eval_tgt[:, :half]

    step_results = []

    for t in range(T + 1):
        probs = _readout_probs(model, s, half, readout_tau)  # [B, half, V]

        # Confidence = max predicted-class probability (standard ECE)
        max_probs, preds = probs.max(dim=-1)  # [B, half]
        correct = (preds == targets).float()

        conf_flat = max_probs.reshape(-1).cpu().numpy()
        corr_flat = correct.reshape(-1).cpu().numpy()

        ece, mce, bin_accs, bin_confs, bin_counts = _compute_ece_mce(
            conf_flat, corr_flat
        )

        avg_confidence = float(conf_flat.mean())
        avg_accuracy = float(corr_flat.mean())
        rho_distance = abs(avg_confidence - NISHIMORI_RHO)

        step_results.append({
            "step": t,
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "ece": ece,
            "mce": mce,
            "rho_distance": rho_distance,
            "nishimori_rho": float(NISHIMORI_RHO),
            "confidence_equals_accuracy": abs(avg_confidence - avg_accuracy),
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
        })

        if t < T:
            s, _ = model.dynamics_step(s, context)

    rho_dists = [r["rho_distance"] for r in step_results]
    t_star = int(np.argmin(rho_dists))
    calib_at_tstar = step_results[t_star]["ece"]

    eces = [r["ece"] for r in step_results]
    t_ece_min = int(np.argmin(eces))

    tau_used = readout_tau if readout_tau is not None else model.tau
    return {
        "per_step": step_results,
        "t_star_nishimori": t_star,
        "rho_at_tstar": step_results[t_star]["avg_confidence"],
        "ece_at_tstar": calib_at_tstar,
        "t_ece_min": t_ece_min,
        "ece_min": eces[t_ece_min],
        "rho_at_ece_min": step_results[t_ece_min]["avg_confidence"],
        "nishimori_test": {
            "rho_nearest_step": t_star,
            "ece_min_step": t_ece_min,
            "steps_coincide": t_star == t_ece_min,
            "rho_at_ece_min_distance": abs(
                step_results[t_ece_min]["avg_confidence"] - NISHIMORI_RHO
            ),
        },
        "readout_tau": tau_used,
    }


@torch.no_grad()
def per_chain_calibration(model, eval_src, eval_tgt, max_chain, config,
                          device):
    """Check if Nishimori transition step depends on carry-chain length.

    Uses model's trained readout (readout_proj + cosine / tau).
    Confidence = max predicted-class probability.
    """
    T = config["T"]
    half = config["seq_len"] // 2

    context = model.encode(eval_src)
    B, L = eval_src.shape
    s = model.init_state(B, L, device)
    targets = eval_tgt[:, :half]

    chain_results = {}
    for chain_len in range(half):
        mask = max_chain == chain_len
        n = mask.sum().item()
        if n < 50:
            continue
        chain_results[int(chain_len)] = {
            "n": n,
            "avg_confidence_per_step": [],
            "avg_accuracy_per_step": [],
        }

    for t in range(T + 1):
        probs = _readout_probs(model, s, half)
        max_probs, preds = probs.max(dim=-1)
        correct = (preds == targets).float()

        for chain_len in chain_results:
            mask = max_chain == chain_len
            chain_conf = max_probs[mask].mean().item()
            chain_acc = correct[mask].mean().item()
            chain_results[chain_len]["avg_confidence_per_step"].append(chain_conf)
            chain_results[chain_len]["avg_accuracy_per_step"].append(chain_acc)

        if t < T:
            s, _ = model.dynamics_step(s, context)

    for chain_len, data in chain_results.items():
        confs = data["avg_confidence_per_step"]
        dists = [abs(c - NISHIMORI_RHO) for c in confs]
        data["t_star"] = int(np.argmin(dists))
        data["rho_at_tstar"] = confs[data["t_star"]]

    return chain_results


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config()
    V = config["vocab_size"]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Nishimori calibration test: does UESD approach rho=tanh(1/2)?",
        "nishimori_rho": float(NISHIMORI_RHO),
        "config": config,
    }

    # Generate eval data
    set_seed(5555)
    eval_src, eval_tgt = generate_batch("addition", 4096, config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)
    max_chain = compute_carries(eval_src, V)

    for track in ["dynamics_ce", "e5"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Training: {track}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model = train_model(track, config, device)
        print(f"  Params: {count_params(model)}", flush=True)

        track_results = {}

        # Primary analysis: model's trained tau (pre-registered)
        print(f"\n  Primary calibration (tau={MODEL_TAU})...", flush=True)
        cal = calibration_analysis(model, eval_src, eval_tgt, config, device)
        track_results["primary"] = cal
        tstar = cal["t_star_nishimori"]
        rho = cal["rho_at_tstar"]
        ece = cal["ece_at_tstar"]
        nt = cal["nishimori_test"]
        print(f"    rho-nearest step: t*={tstar}, rho={rho:.4f} "
              f"(target={NISHIMORI_RHO:.4f}), ECE={ece:.4f}", flush=True)
        print(f"    ECE-min step: t={nt['ece_min_step']}, "
              f"rho={cal['rho_at_ece_min']:.4f}, ECE={cal['ece_min']:.4f}, "
              f"coincide={nt['steps_coincide']}", flush=True)
        for sr in cal["per_step"]:
            if sr["step"] in [0, 1, 3, 5, 7, 10]:
                print(f"    t={sr['step']}: conf={sr['avg_confidence']:.4f} "
                      f"acc={sr['avg_accuracy']:.4f} "
                      f"ECE={sr['ece']:.4f} "
                      f"|conf-rho|={sr['rho_distance']:.4f}", flush=True)

        # Shuffled-label control: same computation, but targets are permuted
        # If calibration looks good with shuffled labels, the metric is broken
        print(f"\n  Shuffled-label control...", flush=True)
        perm = torch.randperm(eval_tgt.shape[0], device=device)
        shuffled_tgt = eval_tgt[perm]
        cal_shuffled = calibration_analysis(
            model, eval_src, shuffled_tgt, config, device
        )
        track_results["shuffled_label_control"] = cal_shuffled
        print(f"    Shuffled t*={cal_shuffled['t_star_nishimori']}, "
              f"rho={cal_shuffled['rho_at_tstar']:.4f}, "
              f"ECE={cal_shuffled['ece_at_tstar']:.4f}", flush=True)

        # Secondary: tau sweep (sensitivity analysis, not primary evidence)
        print(f"\n  Tau sensitivity sweep...", flush=True)
        tau_sweep = {}
        for tau in TAU_VALUES:
            cal_tau = calibration_analysis(
                model, eval_src, eval_tgt, config, device, readout_tau=tau
            )
            tau_sweep[f"tau_{tau}"] = {
                "t_star": cal_tau["t_star_nishimori"],
                "rho_at_tstar": cal_tau["rho_at_tstar"],
                "ece_at_tstar": cal_tau["ece_at_tstar"],
            }
            print(f"    tau={tau}: t*={cal_tau['t_star_nishimori']}, "
                  f"rho={cal_tau['rho_at_tstar']:.4f}, "
                  f"ECE={cal_tau['ece_at_tstar']:.4f}", flush=True)
        track_results["tau_sensitivity"] = tau_sweep

        # Per-chain calibration at model's tau
        print(f"\n  Per-chain calibration...", flush=True)
        chain_cal = per_chain_calibration(
            model, eval_src, eval_tgt, max_chain, config, device
        )
        for cl, data in chain_cal.items():
            print(f"    chain={cl}: t*={data['t_star']}, "
                  f"rho={data['rho_at_tstar']:.4f}, n={data['n']}",
                  flush=True)

        track_results["per_chain_calibration"] = {
            str(k): v for k, v in chain_cal.items()
        }

        all_results[track] = track_results

        del model
        torch.cuda.empty_cache()

    # Encoder-only control (no iterative dynamics)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  CONTROL: Encoder-only (4L)", flush=True)
    print(f"{'=' * 60}", flush=True)

    set_seed(42)
    enc_model = EncoderOnlyAblation(
        V, config["d_model"], config["n_heads"],
        config["d_ff"], 4, config["max_len"],
    ).to(device)
    enc_model.train()
    optimizer = torch.optim.Adam(enc_model.parameters(), lr=config["lr"])
    half = config["seq_len"] // 2

    for step in range(1, config["training_steps"] + 1):
        src, tgt = generate_batch("addition", config["batch_size"],
                                  config["seq_len"], V)
        src, tgt = src.to(device), tgt.to(device)
        logits = enc_model(src)
        logits_r = logits[:, :half, :]
        tgt_r = tgt[:, :half]
        loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                               tgt_r.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_model.parameters(), 1.0)
        optimizer.step()
        if step % 5000 == 0:
            print(f"    Step {step}/{config['training_steps']} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    enc_model.eval()

    # Encoder calibration (single-shot, no dynamics)
    # Uses max predicted-class probability as confidence (standard ECE)
    with torch.no_grad():
        logits = enc_model(eval_src)
        probs = F.softmax(logits[:, :half, :], dim=-1)
        targets = eval_tgt[:, :half]
        max_probs, preds = probs.max(dim=-1)
        correct = (preds == targets).float()

        conf_flat = max_probs.reshape(-1).cpu().numpy()
        corr_flat = correct.reshape(-1).cpu().numpy()
        ece, mce, bin_accs, bin_confs, bin_counts = _compute_ece_mce(
            conf_flat, corr_flat
        )

        enc_conf = float(conf_flat.mean())
        enc_acc = float(corr_flat.mean())
        enc_seq_acc = (preds == targets).all(dim=1).float().mean().item()

    print(f"  Encoder: conf={enc_conf:.4f}, acc={enc_acc:.4f}, "
          f"ECE={ece:.4f}, seq_acc={enc_seq_acc:.4f}", flush=True)
    print(f"  |conf - rho| = {abs(enc_conf - NISHIMORI_RHO):.4f}", flush=True)

    all_results["encoder_control"] = {
        "avg_confidence": enc_conf,
        "avg_accuracy": enc_acc,
        "ece": ece,
        "mce": mce,
        "seq_accuracy": enc_seq_acc,
        "rho_distance": abs(enc_conf - NISHIMORI_RHO),
    }

    del enc_model
    torch.cuda.empty_cache()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d15_nishimori_calibration.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
