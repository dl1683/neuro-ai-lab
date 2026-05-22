"""
Experiment D: Compositional Tasks — Dynamics Necessity (Hard)

Tests whether UESD dynamics are necessary on tasks that require
compositional/sequential computation, not just data-dependent routing.

Two tasks:
1. Addition (base-64, multi-digit): Input interleaves digit pairs [a0,b0,a1,b1,...],
   output is A+B mod base^half. Carry propagation goes right-to-left — requires
   O(L) sequential computation. Encoder with 2 layers has depth O(1), so carry
   chains longer than 2 should break encoder-only.

2. Dedup (deduplicate + sort): Input has repeated elements, output is unique sorted
   values + zero padding. Non-bijective: many inputs → same output. Requires
   counting/grouping operations, not position-wise routing.

Addresses encoder-only confound from Exp A/B/C: if encoder-only fails on addition
or dedup but UESD succeeds, dynamics are proven necessary.

Protocol per task:
1. Train E1 on task. 20K steps.
2. Train E5 with lambda_1 in {0.1, 1.0}. 20K steps each.
3. Train AR baseline. 20K steps.
4. Train encoder-only ablation. 20K steps.
5. Evaluate all on 10K test examples.

Gates:
- UESD acc >= 80%: PASS (dynamics can solve it)
- E5 wrong-attractor < 5%: E5 VIABLE
- Encoder-only acc < 80%: DYNAMICS NECESSARY
- UESD within 5% of AR: COMPETITIVE

Note: Addition output is half-length + zero padding. Token accuracy includes
trivial zeros. Use seq_acc as the harder metric (all positions must be correct).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, ARBaseline, EncoderOnlyAblation
from shared.training import (
    train, evaluate_uesd, evaluate_ar, evaluate_encoder_only, count_params,
)


LAMBDA_SWEEP = [0.1, 1.0]
TASKS = ["addition", "dedup"]


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


def check_gates(results, task):
    gates = {}

    e1_key = f"{task}_e1"
    e1_eval = results[e1_key]["eval"]
    e1_tok = e1_eval["token_accuracy"]["token_acc"]
    e1_seq = e1_eval["token_accuracy"]["seq_acc"]

    if e1_tok >= 0.80:
        gates["track_a"] = "PASS"
        gates["track_a_verdict"] = f"UESD can do {task}. Token acc = {e1_tok:.4f}, Seq acc = {e1_seq:.4f}"
    elif e1_tok < 0.50:
        gates["track_a"] = "FAIL"
        gates["track_a_verdict"] = f"Dynamics can't do {task}. Token acc = {e1_tok:.4f}"
    else:
        gates["track_a"] = "INVESTIGATE"
        gates["track_a_verdict"] = f"Partial {task}. Token acc = {e1_tok:.4f}"

    best_lam, best_acc, best_wa = None, -1.0, 1.0
    best_score = (-1.0, -1.0, 1.0)
    for lam in LAMBDA_SWEEP:
        key = f"{task}_e5_lam{lam}"
        if key not in results:
            continue
        ev = results[key]["eval"]
        acc = ev["token_accuracy"]["token_acc"]
        wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 1.0)
        conv_frac = ev.get("wrong_attractor", {}).get("converged_frac", 0.0)
        rho = ev.get("spectral_radius", {}).get("mean_rho", 1.0)
        gates[f"e5_lam{lam}_acc"] = f"{acc:.4f}"
        gates[f"e5_lam{lam}_wa"] = f"{wa:.4f}"
        gates[f"e5_lam{lam}_conv"] = f"{conv_frac:.4f}"
        score = (acc, conv_frac, -rho)
        if score > best_score:
            best_acc, best_lam, best_wa = acc, lam, wa
            best_score = score

    gates["best_lambda"] = str(best_lam)
    gates["best_e5_accuracy"] = f"{best_acc:.4f}"

    if best_wa < 0.05:
        gates["e5_viability"] = f"VIABLE (WA = {best_wa:.4f} < 0.05)"
    elif best_wa > 0.20:
        gates["e5_viability"] = f"DEAD (WA = {best_wa:.4f} > 0.20)"
    else:
        gates["e5_viability"] = f"UNCERTAIN (WA = {best_wa:.4f})"

    ar_key = f"{task}_ar"
    ar_acc = results[ar_key]["eval"]["token_accuracy"]["token_acc"]
    gates["ar_accuracy"] = f"{ar_acc:.4f}"
    gap = abs(best_acc - ar_acc)
    if gap < 0.05:
        gates["competitive"] = f"COMPETITIVE (gap = {gap:.4f})"
    elif best_acc > ar_acc:
        gates["competitive"] = f"UESD WINS (gap = {gap:.4f})"
    else:
        gates["competitive"] = f"AR WINS (gap = {gap:.4f})"

    enc_key = f"{task}_enc"
    enc_acc = results[enc_key]["eval"]["token_accuracy"]["token_acc"]
    enc_seq = results[enc_key]["eval"]["token_accuracy"]["seq_acc"]
    gates["encoder_only_token_acc"] = f"{enc_acc:.4f}"
    gates["encoder_only_seq_acc"] = f"{enc_seq:.4f}"
    if enc_acc > 0.80:
        gates["encoder_confound"] = f"CONCERN (encoder token_acc={enc_acc:.4f}, seq_acc={enc_seq:.4f})"
    else:
        gates["dynamics_necessity"] = f"CONFIRMED (encoder={enc_acc:.4f}, best_UESD={best_acc:.4f})"

    return gates


def run_task(task, config, device, results):
    """Run all 5 models on a single task."""
    print(flush=True)
    print("#" * 70, flush=True)
    print(f"  TASK: {task.upper()}", flush=True)
    print("#" * 70, flush=True)

    # --- E1 ---
    print("=" * 60, flush=True)
    print(f"TRACK A (E1): Embedding Regression on {task}", flush=True)
    print("=" * 60, flush=True)
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model):,}", flush=True)
    tr = train(model, task, "e1", config, device)
    ev = evaluate_uesd(model, task, config, device)
    results[f"{task}_e1"] = {
        "params": count_params(model),
        "train": tr["history"],
        "elapsed_s": tr["elapsed_s"],
        "eval": ev,
    }
    print(f"  Token Acc: {ev['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {ev['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- E5 lambda sweep ---
    for lam in LAMBDA_SWEEP:
        print("=" * 60, flush=True)
        print(f"TRACK B (E5): Self-Consistency on {task} (lambda_1={lam})", flush=True)
        print("=" * 60, flush=True)
        model = UESDModel(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], config["n_enc_layers"], config["max_len"],
        )
        e5_config = {**config, "lambda_1": lam}
        print(f"  Params: {count_params(model):,}", flush=True)
        tr = train(model, task, "e5", e5_config, device)
        ev = evaluate_uesd(model, task, config, device)
        results[f"{task}_e5_lam{lam}"] = {
            "params": count_params(model),
            "lambda_1": lam,
            "train": tr["history"],
            "elapsed_s": tr["elapsed_s"],
            "eval": ev,
        }
        print(f"  Token Acc:       {ev['token_accuracy']['token_acc']:.4f}", flush=True)
        print(f"  Seq Acc:         {ev['token_accuracy']['seq_acc']:.4f}", flush=True)
        print(f"  Wrong Attractor: {ev['wrong_attractor']['wrong_attractor_rate']:.4f}", flush=True)
        print(f"  Decoder Margin:  {ev['decoder_margin']['mean_margin']:.4f}", flush=True)
        print(f"  Spectral Radius: {ev['spectral_radius']['mean_rho']:.4f}", flush=True)
        print(f"  Basin Stability: {ev['basin_perturbation']['stability_frac']:.4f}", flush=True)
        print(flush=True)

    # --- AR Baseline ---
    print("=" * 60, flush=True)
    print(f"AR BASELINE on {task}", flush=True)
    print("=" * 60, flush=True)
    model = ARBaseline(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["n_dec_layers"],
        config["max_len"],
    )
    print(f"  Params: {count_params(model):,}", flush=True)
    tr = train(model, task, "ar", config, device)
    ev = evaluate_ar(model, task, config, device)
    results[f"{task}_ar"] = {
        "params": count_params(model),
        "train": tr["history"],
        "elapsed_s": tr["elapsed_s"],
        "eval": ev,
    }
    print(f"  Token Acc: {ev['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {ev['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- Encoder-Only ---
    print("=" * 60, flush=True)
    print(f"ENCODER-ONLY ABLATION on {task}", flush=True)
    print("=" * 60, flush=True)
    model = EncoderOnlyAblation(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model):,}", flush=True)
    tr = train(model, task, "encoder_only", config, device)
    ev = evaluate_encoder_only(model, task, config, device)
    results[f"{task}_enc"] = {
        "params": count_params(model),
        "train": tr["history"],
        "elapsed_s": tr["elapsed_s"],
        "eval": ev,
    }
    print(f"  Token Acc: {ev['token_accuracy']['token_acc']:.4f}", flush=True)
    print(f"  Seq Acc:   {ev['token_accuracy']['seq_acc']:.4f}", flush=True)
    print(flush=True)

    # --- Gates ---
    print("=" * 60, flush=True)
    print(f"GATE RESULTS ({task.upper()})", flush=True)
    print("=" * 60, flush=True)
    gates = check_gates(results, task)
    results[f"{task}_gates"] = gates
    for k, v in gates.items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)


def run():
    config = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {"config": config, "device": device, "timestamp": datetime.now(timezone.utc).isoformat()}

    for task in TASKS:
        run_task(task, config, device, results)

    # --- Summary ---
    print("\n" + "=" * 70, flush=True)
    print("DYNAMICS NECESSITY SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for task in TASKS:
        gates = results.get(f"{task}_gates", {})
        enc_tok = gates.get("encoder_only_token_acc", "?")
        enc_seq = gates.get("encoder_only_seq_acc", "?")
        e1_eval = results.get(f"{task}_e1", {}).get("eval", {})
        e1_tok = e1_eval.get("token_accuracy", {}).get("token_acc", 0)
        e1_seq = e1_eval.get("token_accuracy", {}).get("seq_acc", 0)
        necessity = gates.get("dynamics_necessity", gates.get("encoder_confound", "?"))
        print(f"\n  {task.upper()}:", flush=True)
        print(f"    UESD:         token={e1_tok:.4f}, seq={e1_seq:.4f}", flush=True)
        print(f"    Encoder-only: token={enc_tok}, seq={enc_seq}", flush=True)
        print(f"    Verdict:      {necessity}", flush=True)

    # --- Save ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d_compositional.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
