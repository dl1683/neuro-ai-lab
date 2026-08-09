"""
Experiment A: Copy Smoke Test

Gate: do dynamics converge to correct embeddings at all?

Protocol (from design_revision_r3.md — deleted during the 2026-08-09 consolidation; recover via `git show 80fc8b4:experiments/06_uesd/design_revision_r3.md`):
1. Train Track A (E1) on copy task. 20K steps.
2. Train Track B (E5, lambda_1=1.0) on copy task. 20K steps.
3. Train AR baseline on copy task. 20K steps.
4. Train encoder-only ablation on copy task. 20K steps.
5. Evaluate all on 10K test examples.
6. Report D1-D6 for each.

Gates:
- Track A copy accuracy >= 99%: PASS (dynamics converge)
- Track A copy accuracy < 90%: FAIL (dynamics fundamentally broken, stop)
- Track A between 90-99%: INVESTIGATE (readout or convergence issue)
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
    """Apply Experiment A gate logic."""
    gates = {}

    # Primary gate: Track A (E1) copy accuracy
    e1_acc = results["track_a_e1"]["eval"]["token_accuracy"]["token_acc"]
    if e1_acc >= 0.99:
        gates["track_a_copy"] = "PASS"
        gates["track_a_verdict"] = f"Dynamics converge. Token acc = {e1_acc:.4f} >= 0.99"
    elif e1_acc < 0.90:
        gates["track_a_copy"] = "FAIL"
        gates["track_a_verdict"] = f"Dynamics broken. Token acc = {e1_acc:.4f} < 0.90"
    else:
        gates["track_a_copy"] = "INVESTIGATE"
        gates["track_a_verdict"] = f"Partial convergence. Token acc = {e1_acc:.4f} in [0.90, 0.99)"

    # Track B (E5) diagnostics
    if "track_b_e5_lam1.0" in results:
        e5_eval = results["track_b_e5_lam1.0"]["eval"]
        e5_acc = e5_eval["token_accuracy"]["token_acc"]
        wa_rate = e5_eval.get("wrong_attractor", {}).get("wrong_attractor_rate", None)
        margin = e5_eval.get("decoder_margin", {}).get("mean_margin", None)
        mean_rho = e5_eval.get("spectral_radius", {}).get("mean_rho", None)
        max_rho = e5_eval.get("spectral_radius", {}).get("max_rho", None)

        gates["track_b_accuracy"] = f"{e5_acc:.4f}"
        if wa_rate is not None:
            if wa_rate < 0.05:
                gates["track_b_wrong_attractor"] = f"VIABLE ({wa_rate:.4f} < 0.05)"
            elif wa_rate > 0.20:
                gates["track_b_wrong_attractor"] = f"DEAD ({wa_rate:.4f} > 0.20)"
            else:
                gates["track_b_wrong_attractor"] = f"UNCERTAIN ({wa_rate:.4f})"
        if margin is not None:
            gates["decoder_margin_gate"] = "PASS" if margin > 0.1 else f"BELOW ({margin:.4f})"
        if max_rho is not None:
            gates["spectral_radius_gate"] = (
                f"PASS (mean={mean_rho:.4f}, max={max_rho:.4f})"
                if max_rho < 1.05
                else f"ABOVE (mean={mean_rho:.4f}, max={max_rho:.4f})"
            )

    # Encoder-only comparison
    if "encoder_only" in results:
        enc_acc = results["encoder_only"]["eval"]["token_accuracy"]["token_acc"]
        gates["encoder_only_accuracy"] = f"{enc_acc:.4f}"
        if enc_acc > 0.80:
            gates["encoder_confound"] = "CONCERN (encoder solving most of the task)"
        else:
            gates["encoder_confound"] = "OK (dynamics doing meaningful work)"

    return gates


def run():
    config = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")
    print()

    results = {"config": config, "device": device, "timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Track A (E1): Embedding Regression ---
    print("=" * 60)
    print("TRACK A (E1): Embedding Regression on Copy")
    print("=" * 60)
    model_e1 = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    n_params = count_params(model_e1)
    print(f"  Params: {n_params:,}")
    train_result = train(model_e1, "copy", "e1", config, device)
    eval_result = evaluate_uesd(model_e1, "copy", config, device)
    results["track_a_e1"] = {
        "params": n_params,
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}")
    print()

    # --- Track B (E5): Self-Consistency + Readout, lambda_1 = 1.0 ---
    print("=" * 60)
    print("TRACK B (E5): Self-Consistency + Readout on Copy (lambda_1=1.0)")
    print("=" * 60)
    model_e5 = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    e5_config = {**config, "lambda_1": 1.0}
    print(f"  Params: {count_params(model_e5):,}")
    train_result = train(model_e5, "copy", "e5", e5_config, device)
    eval_result = evaluate_uesd(model_e5, "copy", config, device)
    results["track_b_e5_lam1.0"] = {
        "params": count_params(model_e5),
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc:       {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:         {eval_result['token_accuracy']['seq_acc']:.4f}")
    print(f"  Wrong Attractor: {eval_result['wrong_attractor']['wrong_attractor_rate']:.4f}")
    print(f"  Decoder Margin:  {eval_result['decoder_margin']['mean_margin']:.4f}")
    print(f"  Spectral Radius: {eval_result['spectral_radius']['mean_rho']:.4f}")
    print()

    # --- AR Baseline ---
    print("=" * 60)
    print("AR BASELINE on Copy")
    print("=" * 60)
    model_ar = ARBaseline(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["n_dec_layers"],
        config["max_len"],
    )
    n_params_ar = count_params(model_ar)
    print(f"  Params: {n_params_ar:,}")
    train_result = train(model_ar, "copy", "ar", config, device)
    eval_result = evaluate_ar(model_ar, "copy", config, device)
    results["ar_baseline"] = {
        "params": n_params_ar,
        "train": train_result["history"],
        "elapsed_s": train_result["elapsed_s"],
        "eval": eval_result,
    }
    print(f"  Token Acc: {eval_result['token_accuracy']['token_acc']:.4f}")
    print(f"  Seq Acc:   {eval_result['token_accuracy']['seq_acc']:.4f}")
    print()

    # --- Encoder-Only Ablation ---
    print("=" * 60)
    print("ENCODER-ONLY ABLATION on Copy")
    print("=" * 60)
    model_enc = EncoderOnlyAblation(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    n_params_enc = count_params(model_enc)
    print(f"  Params: {n_params_enc:,}")
    train_result = train(model_enc, "copy", "encoder_only", config, device)
    eval_result = evaluate_encoder_only(model_enc, "copy", config, device)
    results["encoder_only"] = {
        "params": n_params_enc,
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

    # --- Save ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_a_copy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run()
