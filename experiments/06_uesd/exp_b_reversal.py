"""
Experiment B: Reversal Main Test

Core test: can dynamics solve non-trivial transformations?
E5 vs E1 comparison. Lambda sweep.

Protocol (from design_revision_r3.md — deleted during the 2026-08-09 consolidation; recover via `git show 80fc8b4:experiments/06_uesd/design_revision_r3.md`):
1. Train Track A (E1) on reversal. 20K steps.
2. Train Track B (E5) on reversal with lambda_1 in {0, 0.1, 1.0, 10.0}. 20K steps each.
3. Train AR baseline on reversal. 20K steps.
4. Train encoder-only ablation on reversal. 20K steps.
5. Evaluate all on 10K test examples.
6. Report D1-D6 for each.

Gates:
- Track A reversal accuracy >= 90%: PASS (UESD can transform)
- Track B wrong-attractor rate < 5%: E5 VIABLE
- Track B wrong-attractor rate > 20%: E5 DEAD
- Track A/B within 5% of AR baseline: COMPETITIVE
- Encoder-only ablation >> 80%: CONCERN (encoder doing the work)
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, ARBaseline, EncoderOnlyAblation, default_config
from shared.training import (
    train, evaluate_uesd, evaluate_ar, evaluate_encoder_only, count_params,
)


LAMBDA_SWEEP = [0.0, 0.1, 1.0, 10.0]


def build_config():
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "n_dec_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "log_interval": 1000,
        "eval_samples": 10000,
        "eval_batch_size": 512,
        "warmup_steps": 5000,
    }


def check_gates(results):
    """Apply Experiment B gate logic."""
    gates = {}

    # Track A reversal accuracy
    e1_acc = results["track_a_e1"]["eval"]["token_accuracy"]["token_acc"]
    if e1_acc >= 0.90:
        gates["track_a_reversal"] = "PASS"
        gates["track_a_verdict"] = f"UESD can transform. Token acc = {e1_acc:.4f} >= 0.90"
    elif e1_acc < 0.70:
        gates["track_a_reversal"] = "FAIL"
        gates["track_a_verdict"] = f"Dynamics can't transform. Token acc = {e1_acc:.4f} < 0.70"
    else:
        gates["track_a_reversal"] = "INVESTIGATE"
        gates["track_a_verdict"] = f"Weak performance. Token acc = {e1_acc:.4f} in [0.70, 0.90)"

    # Track B lambda sweep: find best by (accuracy, converged_frac, -rho)
    best_lam, best_acc, best_wa = None, -1.0, 1.0
    best_score = (-1.0, -1.0, 1.0)
    for lam in LAMBDA_SWEEP:
        key = f"track_b_e5_lam{lam}"
        if key not in results:
            continue
        ev = results[key]["eval"]
        acc = ev["token_accuracy"]["token_acc"]
        wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 1.0)
        conv_frac = ev.get("wrong_attractor", {}).get("converged_frac", 0.0)
        rho = ev.get("spectral_radius", {}).get("mean_rho", 1.0)
        gates[f"e5_lam{lam}_acc"] = f"{acc:.4f}"
        gates[f"e5_lam{lam}_wrong_attractor"] = f"{wa:.4f}"
        score = (acc, conv_frac, -rho)
        if score > best_score:
            best_acc, best_lam, best_wa = acc, lam, wa
            best_score = score

    gates["best_lambda"] = str(best_lam)
    gates["best_e5_accuracy"] = f"{best_acc:.4f}"

    if best_wa < 0.05:
        gates["e5_viability"] = f"VIABLE (wrong-attractor rate = {best_wa:.4f} < 0.05)"
    elif best_wa > 0.20:
        gates["e5_viability"] = f"DEAD (wrong-attractor rate = {best_wa:.4f} > 0.20)"
    else:
        gates["e5_viability"] = f"UNCERTAIN (wrong-attractor rate = {best_wa:.4f})"

    # AR comparison
    ar_acc = results["ar_baseline"]["eval"]["token_accuracy"]["token_acc"]
    gates["ar_accuracy"] = f"{ar_acc:.4f}"
    gap = abs(best_acc - ar_acc)
    if gap < 0.05:
        gates["competitive"] = f"COMPETITIVE (gap = {gap:.4f} < 0.05)"
    elif best_acc > ar_acc:
        gates["competitive"] = f"UESD WINS (gap = {gap:.4f})"
    else:
        gates["competitive"] = f"AR WINS (gap = {gap:.4f})"

    # Spectral radius gate (best lambda model)
    if best_lam is not None:
        best_key = f"track_b_e5_lam{best_lam}"
        if best_key in results:
            sr = results[best_key]["eval"].get("spectral_radius", {})
            mean_rho = sr.get("mean_rho")
            max_rho = sr.get("max_rho")
            if max_rho is not None:
                gates["spectral_radius_gate"] = (
                    f"PASS (mean={mean_rho:.4f}, max={max_rho:.4f})"
                    if max_rho < 1.05
                    else f"ABOVE (mean={mean_rho:.4f}, max={max_rho:.4f})"
                )

    # Encoder-only confound
    enc_acc = results["encoder_only"]["eval"]["token_accuracy"]["token_acc"]
    gates["encoder_only_accuracy"] = f"{enc_acc:.4f}"
    if enc_acc > 0.80:
        gates["encoder_confound"] = "CONCERN (encoder solving > 80%)"
    else:
        gates["encoder_confound"] = "OK (dynamics doing meaningful work)"

    return gates


def run():
    config = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = {"config": config, "device": device, "timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Track A (E1): Embedding Regression ---
    print("=" * 60)
    print("TRACK A (E1): Embedding Regression on Reversal")
    print("=" * 60)
    model_e1 = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model_e1):,}")
    train_result = train(model_e1, "reversal", "e1", config, device)
    eval_result = evaluate_uesd(model_e1, "reversal", config, device)
    results["track_a_e1"] = {
        "params": count_params(model_e1),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}")
    print()

    # --- Track B (E5): Lambda Sweep ---
    for lam in LAMBDA_SWEEP:
        lam_str = f"lam{lam}"
        print("=" * 60)
        print(f"TRACK B (E5): Self-Consistency on Reversal (lambda_1={lam})")
        print("=" * 60)
        model_e5 = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        )
        e5_config = {**config, "lambda_1": lam}
        if lam == 0.0:
            e5_config["warmup_steps"] = 0
        print(f"  Params: {count_params(model_e5):,}")
        train_result = train(model_e5, "reversal", "e5", e5_config, device)
        eval_result = evaluate_uesd(model_e5, "reversal", config, device)
        results[f"track_b_e5_{lam_str}"] = {
            "params": count_params(model_e5),
            "lambda_1": lam,
            "train": train_result["history"],
            "elapsed_s": train_result["elapsed_s"],
            "eval": eval_result,
        }
        print(f"  Token Acc:       {eval_result['token_accuracy']['token_acc']:.4f}")
        print(f"  Seq Acc:         {eval_result['token_accuracy']['seq_acc']:.4f}")
        print(f"  Wrong Attractor: {eval_result['wrong_attractor']['wrong_attractor_rate']:.4f}")
        print(f"  Decoder Margin:  {eval_result['decoder_margin']['mean_margin']:.4f}")
        print(f"  Spectral Radius: {eval_result['spectral_radius']['mean_rho']:.4f}")
        print(f"  Basin Stability: {eval_result['basin_perturbation']['stability_frac']:.4f}")
        print()

    # --- AR Baseline ---
    print("=" * 60)
    print("AR BASELINE on Reversal")
    print("=" * 60)
    model_ar = ARBaseline(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["n_dec_layers"],
        config["max_len"],
    )
    print(f"  Params: {count_params(model_ar):,}")
    train_result = train(model_ar, "reversal", "ar", config, device)
    eval_result = evaluate_ar(model_ar, "reversal", config, device)
    results["ar_baseline"] = {
        "params": count_params(model_ar),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}")
    print()

    # --- Encoder-Only Ablation ---
    print("=" * 60)
    print("ENCODER-ONLY ABLATION on Reversal")
    print("=" * 60)
    model_enc = EncoderOnlyAblation(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model_enc):,}")
    train_result = train(model_enc, "reversal", "encoder_only", config, device)
    eval_result = evaluate_encoder_only(model_enc, "reversal", config, device)
    results["encoder_only"] = {
        "params": count_params(model_enc),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}")
    print()

    # --- Gates ---
    print("=" * 60)
    print("GATE RESULTS")
    print("=" * 60)
    gates = check_gates(results)
    results["gates"] = gates
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print()

    # --- Key Comparisons ---
    print("=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)
    e1_acc = results["track_a_e1"]["eval"]["token_accuracy"]["token_acc"]
    ar_acc = results["ar_baseline"]["eval"]["token_accuracy"]["token_acc"]
    enc_acc = results["encoder_only"]["eval"]["token_accuracy"]["token_acc"]
    print(f"  E1 vs AR:           {e1_acc:.4f} vs {ar_acc:.4f} (gap: {abs(e1_acc - ar_acc):.4f})")
    print(f"  E1 vs Encoder-Only: {e1_acc:.4f} vs {enc_acc:.4f} (dynamics value: {e1_acc - enc_acc:+.4f})")
    print()

    for lam in LAMBDA_SWEEP:
        key = f"track_b_e5_lam{lam}"
        if key in results:
            ev = results[key]["eval"]
            acc = ev["token_accuracy"]["token_acc"]
            wa = ev["wrong_attractor"]["wrong_attractor_rate"]
            rho = ev["spectral_radius"]["mean_rho"]
            margin = ev["decoder_margin"]["mean_margin"]
            basin = ev["basin_perturbation"]["stability_frac"]
            print(f"  E5(lam={lam:>4}): acc={acc:.4f} | WA={wa:.4f} | rho={rho:.4f} | margin={margin:.4f} | basin={basin:.4f}")
    print()

    # --- Save ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_b_reversal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run()
