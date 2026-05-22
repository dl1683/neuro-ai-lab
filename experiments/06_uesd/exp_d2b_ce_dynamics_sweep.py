"""
Experiment D2b: CE-Dynamics Seed Sweep

Fills the most critical gap from D2: CE-dynamics (UESD + pure CE, no SC)
was only tested once. This runs 5 seeds with proper seeding (set_seed
BEFORE model creation) to confirm robustness.

Also re-runs E5 and encoder-only 2L sweeps with fixed seeding for
comparison.
"""
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, EncoderOnlyAblation
from shared.training import (
    train, evaluate_uesd, evaluate_encoder_only, count_params, set_seed,
)


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


def run_sweep(model_type, device, results):
    seed_results = []
    for seed in SEEDS:
        config = build_config(seed=seed)
        print(f"  {model_type} seed={seed}...", end=" ", flush=True)

        set_seed(seed)
        if model_type == "ce_dynamics":
            model = UESDModel(
                config["vocab_size"], config["d_model"], config["n_heads"],
                config["d_ff"], config["n_enc_layers"], config["max_len"],
            )
            tr = train(model, "addition", "dynamics_ce", config, device)
            ev = evaluate_uesd(model, "addition", config, device)
        elif model_type == "e5":
            model = UESDModel(
                config["vocab_size"], config["d_model"], config["n_heads"],
                config["d_ff"], config["n_enc_layers"], config["max_len"],
            )
            e5_config = {**config, "lambda_1": 1.0}
            tr = train(model, "addition", "e5", e5_config, device)
            ev = evaluate_uesd(model, "addition", config, device)
        else:
            model = EncoderOnlyAblation(
                config["vocab_size"], config["d_model"], config["n_heads"],
                config["d_ff"], config["n_enc_layers"], config["max_len"],
            )
            tr = train(model, "addition", "encoder_only", config, device)
            ev = evaluate_encoder_only(model, "addition", config, device)

        tok = ev["token_accuracy"]["token_acc"]
        seq = ev["token_accuracy"]["seq_acc"]
        print(f"tok={tok:.4f} seq={seq:.4f}", flush=True)

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
        "seeds": SEEDS,
        "runs": seed_results,
        "token_acc_mean": statistics.mean(tok_accs),
        "token_acc_std": statistics.stdev(tok_accs) if len(tok_accs) > 1 else 0,
        "seq_acc_mean": statistics.mean(seq_accs),
        "seq_acc_std": statistics.stdev(seq_accs) if len(seq_accs) > 1 else 0,
    }
    results[f"sweep_{model_type}"] = summary
    print(f"  {model_type} mean tok={summary['token_acc_mean']:.4f} "
          f"(+/-{summary['token_acc_std']:.4f}), "
          f"seq={summary['seq_acc_mean']:.4f} "
          f"(+/-{summary['seq_acc_std']:.4f})", flush=True)
    print(flush=True)
    return summary


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Properly-seeded sweep: CE-dynamics vs E5 vs encoder-only 2L on addition",
        "seeding": "set_seed called BEFORE model creation (fixed from D2)",
    }

    for model_type in ["ce_dynamics", "e5", "enc_2L"]:
        print("#" * 70, flush=True)
        print(f"  5-SEED SWEEP: {model_type}", flush=True)
        print("#" * 70, flush=True)
        run_sweep(model_type, device, results)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("EXP D2b: PROPERLY-SEEDED SWEEP RESULTS", flush=True)
    print("=" * 70, flush=True)

    for model_type in ["ce_dynamics", "e5", "enc_2L"]:
        key = f"sweep_{model_type}"
        s = results[key]
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        print(f"  {model_type:15s}: tok={s['token_acc_mean']:.4f}+/-{s['token_acc_std']:.4f}, "
              f"seq={s['seq_acc_mean']:.4f}+/-{s['seq_acc_std']:.4f}, "
              f"success={successes}/5", flush=True)

    print("=" * 70, flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d2b_ce_dynamics_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
