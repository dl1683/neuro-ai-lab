"""
Experiment C: Sort — Dynamics Necessity Test

Tests whether UESD dynamics add value on a task that requires
data-dependent reordering, not a fixed permutation.

Sorting (ascending) of length-8 sequences from V=64 requires
computing element ranks via global comparison — fundamentally
different from reversal (fixed permutation solvable by positional
embedding alone).

This experiment directly addresses the Codex-flagged encoder-only
confound from Exp A/B: if encoder-only fails on sorting but UESD
succeeds, dynamics are proven necessary.

Protocol:
1. Train E1 on sort. 20K steps.
2. Train E5 on sort with lambda_1 in {0.1, 1.0}. 20K steps each.
   (Narrowed from Exp B — lambda=0.1 was the sweet spot, lambda=0
   doesn't converge, lambda=10 hurts.)
3. Train AR baseline on sort. 20K steps.
4. Train encoder-only ablation on sort. 20K steps.
5. Evaluate all on 10K test examples.

Gates:
- E1 sort accuracy >= 80%: PASS (UESD can sort)
- E5 wrong-attractor rate < 5%: E5 VIABLE
- Encoder-only accuracy < 80%: DYNAMICS NECESSARY
- UESD within 5% of AR: COMPETITIVE
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, ARBaseline, EncoderOnlyAblation, default_config
from shared.training import (
    train, evaluate_uesd, evaluate_ar, evaluate_encoder_only, count_params,
)


LAMBDA_SWEEP = [0.1, 1.0]


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
    gates = {}

    # E1 sort accuracy
    e1_acc = results["track_a_e1"]["eval"]["token_accuracy"]["token_acc"]
    if e1_acc >= 0.80:
        gates["track_a_sort"] = "PASS"
        gates["track_a_verdict"] = f"UESD can sort. Token acc = {e1_acc:.4f} >= 0.80"
    elif e1_acc < 0.50:
        gates["track_a_sort"] = "FAIL"
        gates["track_a_verdict"] = f"Dynamics can't sort. Token acc = {e1_acc:.4f} < 0.50"
    else:
        gates["track_a_sort"] = "INVESTIGATE"
        gates["track_a_verdict"] = f"Partial sorting. Token acc = {e1_acc:.4f} in [0.50, 0.80)"

    # E5 lambda sweep
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
        gates[f"e5_lam{lam}_converged_frac"] = f"{conv_frac:.4f}"
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

    # Encoder-only — the key gate
    enc_acc = results["encoder_only"]["eval"]["token_accuracy"]["token_acc"]
    gates["encoder_only_accuracy"] = f"{enc_acc:.4f}"
    if enc_acc > 0.80:
        gates["encoder_confound"] = "CONCERN (encoder solving > 80%)"
    else:
        gates["dynamics_necessity"] = f"CONFIRMED (encoder at {enc_acc:.4f}, UESD at {e1_acc:.4f})"

    return gates


def run():
    config = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print(flush=True)

    results = {"config": config, "device": device, "timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Track A (E1): Embedding Regression ---
    print("=" * 60, flush=True)
    print("TRACK A (E1): Embedding Regression on Sort", flush=True)
    print("=" * 60, flush=True)
    model_e1 = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model_e1):,}", flush=True)
    train_result = train(model_e1, "sort", "e1", config, device)
    eval_result = evaluate_uesd(model_e1, "sort", config, device)
    results["track_a_e1"] = {
        "params": count_params(model_e1),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- Track B (E5): Lambda Sweep ---
    for lam in LAMBDA_SWEEP:
        lam_str = f"lam{lam}"
        print("=" * 60, flush=True)
        print(f"TRACK B (E5): Self-Consistency on Sort (lambda_1={lam})", flush=True)
        print("=" * 60, flush=True)
        model_e5 = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        )
        e5_config = {**config, "lambda_1": lam}
        print(f"  Params: {count_params(model_e5):,}", flush=True)
        train_result = train(model_e5, "sort", "e5", e5_config, device)
        eval_result = evaluate_uesd(model_e5, "sort", config, device)
        results[f"track_b_e5_{lam_str}"] = {
            "params": count_params(model_e5),
            "lambda_1": lam,
            "train": train_result["history"],
            "elapsed_s": train_result["elapsed_s"],
            "eval": eval_result,
        }
        print(f"  Token Acc:       {eval_result['token_accuracy']['token_acc']:.4f}", flush=True)
        print(f"  Seq Acc:         {eval_result['token_accuracy']['seq_acc']:.4f}", flush=True)
        print(f"  Wrong Attractor: {eval_result['wrong_attractor']['wrong_attractor_rate']:.4f}", flush=True)
        print(f"  Decoder Margin:  {eval_result['decoder_margin']['mean_margin']:.4f}", flush=True)
        print(f"  Spectral Radius: {eval_result['spectral_radius']['mean_rho']:.4f}", flush=True)
        print(f"  Basin Stability: {eval_result['basin_perturbation']['stability_frac']:.4f}", flush=True)
        print(flush=True)

    # --- AR Baseline ---
    print("=" * 60, flush=True)
    print("AR BASELINE on Sort", flush=True)
    print("=" * 60, flush=True)
    model_ar = ARBaseline(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["n_dec_layers"],
        config["max_len"],
    )
    print(f"  Params: {count_params(model_ar):,}", flush=True)
    train_result = train(model_ar, "sort", "ar", config, device)
    eval_result = evaluate_ar(model_ar, "sort", config, device)
    results["ar_baseline"] = {
        "params": count_params(model_ar),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- Encoder-Only Ablation ---
    print("=" * 60, flush=True)
    print("ENCODER-ONLY ABLATION on Sort", flush=True)
    print("=" * 60, flush=True)
    model_enc = EncoderOnlyAblation(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model_enc):,}", flush=True)
    train_result = train(model_enc, "sort", "encoder_only", config, device)
    eval_result = evaluate_encoder_only(model_enc, "sort", config, device)
    results["encoder_only"] = {
        "params": count_params(model_enc),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- Gates ---
    print("=" * 60, flush=True)
    print("GATE RESULTS", flush=True)
    print("=" * 60, flush=True)
    gates = check_gates(results)
    results["gates"] = gates
    for k, v in gates.items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    # --- Save ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_c_sort.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
