"""
Experiment D2d: Depth-Matched Encoder Multi-Seed Sweep

D2 tested 4L and 8L encoders with single runs only. Codex review
flagged this as insufficient: "single-run 4L/8L numbers are not stable
enough to support strong relative-efficiency conclusions."

This runs 5 seeds each for 4L and 8L encoders with proper seeding to
establish reliable baselines for the parameter-efficiency comparison.
"""
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import EncoderOnlyAblation
from shared.training import train, evaluate_encoder_only, count_params, set_seed


SEEDS = [42, 137, 256, 512, 1024]


def build_config(seed=None):
    cfg = {
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
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def run_depth_sweep(n_layers, device, results):
    seed_results = []
    for seed in SEEDS:
        config = build_config(seed=seed)
        print("  enc_%dL seed=%d..." % (n_layers, seed), end=" ", flush=True)

        set_seed(seed)
        model = EncoderOnlyAblation(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], n_layers, config["max_len"],
        )
        if seed == SEEDS[0]:
            print("[%d params]" % count_params(model), end=" ", flush=True)

        tr = train(model, "addition", "encoder_only", config, device)
        ev = evaluate_encoder_only(model, "addition", config, device)

        tok = ev["token_accuracy"]["token_acc"]
        seq = ev["token_accuracy"]["seq_acc"]
        print("tok=%.4f seq=%.4f" % (tok, seq), flush=True)

        seed_results.append({
            "seed": seed,
            "token_acc": tok,
            "seq_acc": seq,
            "train_final_loss": tr["history"][-1]["loss"] if tr["history"] else None,
            "elapsed_s": tr["elapsed_s"],
        })

    tok_accs = [r["token_acc"] for r in seed_results]
    seq_accs = [r["seq_acc"] for r in seed_results]
    summary = {
        "n_layers": n_layers,
        "params": count_params(model),
        "seeds": SEEDS,
        "runs": seed_results,
        "token_acc_mean": statistics.mean(tok_accs),
        "token_acc_std": statistics.stdev(tok_accs) if len(tok_accs) > 1 else 0,
        "seq_acc_mean": statistics.mean(seq_accs),
        "seq_acc_std": statistics.stdev(seq_accs) if len(seq_accs) > 1 else 0,
        "success_rate": sum(1 for s in seq_accs if s > 0.9) / len(seq_accs),
    }
    results["enc_%dL" % n_layers] = summary

    print("  enc_%dL: mean tok=%.4f (+/-%.4f), seq=%.4f (+/-%.4f), success=%d/5" % (
        n_layers, summary["token_acc_mean"], summary["token_acc_std"],
        summary["seq_acc_mean"], summary["seq_acc_std"],
        int(summary["success_rate"] * 5),
    ), flush=True)
    return summary


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: %s" % device, flush=True)
    if device == "cuda":
        print("GPU: %s" % torch.cuda.get_device_name(), flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Multi-seed depth-matched encoder sweep (4L/8L) for parameter-efficiency comparison",
    }

    for n_layers in [4, 8]:
        print("\n" + "#" * 70, flush=True)
        print("  5-SEED SWEEP: Encoder-Only %dL" % n_layers, flush=True)
        print("#" * 70, flush=True)
        run_depth_sweep(n_layers, device, results)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("EXP D2d: DEPTH-MATCHED ENCODER SWEEP RESULTS", flush=True)
    print("=" * 70, flush=True)

    for n_layers in [4, 8]:
        key = "enc_%dL" % n_layers
        s = results[key]
        print("  enc_%dL (%d params): tok=%.4f+/-%.4f, seq=%.4f+/-%.4f, success=%d/5" % (
            n_layers, s["params"],
            s["token_acc_mean"], s["token_acc_std"],
            s["seq_acc_mean"], s["seq_acc_std"],
            int(s["success_rate"] * 5),
        ), flush=True)

    # Parameter efficiency comparison (placeholder for D2b results)
    print("\n  Parameter efficiency comparison:", flush=True)
    for n_layers in [4, 8]:
        s = results["enc_%dL" % n_layers]
        print("    enc_%dL: %d params, seq=%.4f+/-%.4f" % (
            n_layers, s["params"], s["seq_acc_mean"], s["seq_acc_std"],
        ), flush=True)
    print("    (Compare with UESD CE-dynamics: 694K params from D2b)", flush=True)
    print("=" * 70, flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d2d_depth_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to %s" % out_path, flush=True)

    return results


if __name__ == "__main__":
    run()
