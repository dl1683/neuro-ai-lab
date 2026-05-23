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


@torch.no_grad()
def calibration_analysis(model, eval_src, eval_tgt, config, device,
                         readout_tau=0.1):
    """Compute per-step calibration metrics for UESD dynamics."""
    T = config["T"]
    half = config["seq_len"] // 2
    V = config["vocab_size"]

    context = model.encode(eval_src)
    B, L = eval_src.shape
    s = model.init_state(B, L, device)
    targets = eval_tgt[:, :half]

    step_results = []

    for t in range(T + 1):
        # Readout probabilities at current state
        logits = model.readout_logits(s)  # [B, L, V] using cosine sim / tau
        # Override with specified tau for this analysis
        emb = model.tok_emb.weight  # [V, d]
        cos_sim = F.cosine_similarity(
            s[:, :half, :].unsqueeze(2),  # [B, half, 1, d]
            emb.unsqueeze(0).unsqueeze(0),  # [1, 1, V, d]
            dim=-1,
        )  # [B, half, V]
        probs = F.softmax(cos_sim / readout_tau, dim=-1)  # [B, half, V]

        # Confidence: probability assigned to the correct token
        correct_probs = probs.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)  # [B, half]

        # Correctness: does argmax match target?
        preds = probs.argmax(dim=-1)  # [B, half]
        correct = (preds == targets).float()  # [B, half]

        # Flatten for calibration analysis
        conf_flat = correct_probs.reshape(-1).cpu().numpy()
        corr_flat = correct.reshape(-1).cpu().numpy()

        # Calibration: bin by confidence, measure accuracy per bin
        bin_edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []

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

        # Expected Calibration Error
        total = len(conf_flat)
        ece = sum(
            (bc / total) * abs(ba - bconf)
            for ba, bconf, bc in zip(bin_accs, bin_confs, bin_counts)
            if not np.isnan(ba)
        )

        # Maximum Calibration Error
        mce = max(
            (abs(ba - bconf) for ba, bconf in zip(bin_accs, bin_confs)
             if not np.isnan(ba)),
            default=0.0,
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

        # Advance dynamics (except after last step)
        if t < T:
            s, _ = model.dynamics_step(s, context)

    # Find the step closest to Nishimori rho
    rho_dists = [r["rho_distance"] for r in step_results]
    t_star = int(np.argmin(rho_dists))
    calib_at_tstar = step_results[t_star]["ece"]

    return {
        "per_step": step_results,
        "t_star_nishimori": t_star,
        "rho_at_tstar": step_results[t_star]["avg_confidence"],
        "ece_at_tstar": calib_at_tstar,
        "readout_tau": readout_tau,
    }


@torch.no_grad()
def per_chain_calibration(model, eval_src, eval_tgt, max_chain, config,
                          device, readout_tau=0.1):
    """Check if Nishimori transition step depends on carry-chain length."""
    T = config["T"]
    half = config["seq_len"] // 2
    V = config["vocab_size"]

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

    s = model.init_state(B, L, device)
    for t in range(T + 1):
        emb = model.tok_emb.weight
        cos_sim = F.cosine_similarity(
            s[:, :half, :].unsqueeze(2),
            emb.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        probs = F.softmax(cos_sim / readout_tau, dim=-1)
        correct_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        preds = probs.argmax(dim=-1)
        correct = (preds == targets).float()

        for chain_len in chain_results:
            mask = max_chain == chain_len
            chain_conf = correct_probs[mask].mean().item()
            chain_acc = correct[mask].mean().item()
            chain_results[chain_len]["avg_confidence_per_step"].append(chain_conf)
            chain_results[chain_len]["avg_accuracy_per_step"].append(chain_acc)

        if t < T:
            s, _ = model.dynamics_step(s, context)

    # Find t* per chain length
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

        # Sweep readout temperatures
        for tau in TAU_VALUES:
            print(f"\n  Calibration analysis (tau={tau})...", flush=True)
            cal = calibration_analysis(
                model, eval_src, eval_tgt, config, device, readout_tau=tau
            )
            track_results[f"tau_{tau}"] = cal

            tstar = cal["t_star_nishimori"]
            rho = cal["rho_at_tstar"]
            ece = cal["ece_at_tstar"]
            print(f"    t*={tstar}, rho={rho:.4f} (target={NISHIMORI_RHO:.4f}), "
                  f"ECE={ece:.4f}", flush=True)

            # Per-step summary
            for sr in cal["per_step"]:
                if sr["step"] in [0, 1, 3, 5, 7, 10]:
                    print(f"    t={sr['step']}: conf={sr['avg_confidence']:.4f} "
                          f"acc={sr['avg_accuracy']:.4f} "
                          f"ECE={sr['ece']:.4f} "
                          f"|conf-rho|={sr['rho_distance']:.4f}", flush=True)

        # Per-chain calibration at default tau
        print(f"\n  Per-chain calibration (tau=0.1)...", flush=True)
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
    with torch.no_grad():
        logits = enc_model(eval_src)
        probs = F.softmax(logits[:, :half, :], dim=-1)
        targets = eval_tgt[:, :half]
        correct_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        preds = probs.argmax(dim=-1)
        correct = (preds == targets).float()

        enc_conf = correct_probs.mean().item()
        enc_acc = correct.float().mean().item()
        enc_preds_correct = (preds == targets).all(dim=1).float().mean().item()

    print(f"  Encoder: conf={enc_conf:.4f}, acc={enc_acc:.4f}, "
          f"seq_acc={enc_preds_correct:.4f}", flush=True)
    print(f"  |conf - rho| = {abs(enc_conf - NISHIMORI_RHO):.4f}", flush=True)

    all_results["encoder_control"] = {
        "avg_confidence": enc_conf,
        "avg_accuracy": enc_acc,
        "seq_accuracy": enc_preds_correct,
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
