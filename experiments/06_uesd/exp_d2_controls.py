"""
Experiment D2: Additional Controls for Dynamics Necessity Claim

Addresses Codex Evidence Gate findings on Exp D:

1. CE-matched dynamics ablation: UESD architecture with pure CE loss (no MSE,
   no SC). Isolates whether iterative dynamics alone help, independent of loss.
2. Depth-matched encoder-only: 4-layer and 8-layer encoder-only models. Tests
   whether simply adding depth (without weight-tied iteration) solves addition.
3. Seed sweep: Run key models across 5 seeds for statistical robustness.

If CE-dynamics succeeds AND depth-matched encoder fails, the claim strengthens
to: "weight-tied iterative dynamics are necessary, not just more depth."

If CE-dynamics fails, the SC term is essential, and the finding is about E5's
loss design, not dynamics per se.

If depth-matched encoder succeeds, the claim weakens to: "2-layer encoder
can't do it" (trivially true, uninteresting).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel, EncoderOnlyAblation, _make_encoder
from shared.training import (
    train, evaluate_uesd, evaluate_encoder_only, count_params, set_seed,
)

# NOTE: set_seed must be called BEFORE model creation, not just before
# training. The train() function calls set_seed internally, but that
# only controls data generation and optimizer — not model initialization.
# All model creation below is preceded by an explicit set_seed call.


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


class DeepEncoderOnly(EncoderOnlyAblation):
    """Encoder-only with configurable depth."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_enc_layers, max_len):
        super(EncoderOnlyAblation, self).__init__()
        self.d_model = d_model
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_enc = torch.nn.Embedding(max_len, d_model)
        self.encoder = _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.tau = 0.1


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Codex Evidence Gate controls for Exp D dynamics necessity claim",
    }

    # ====================================================================
    # CONTROL 1: CE-matched dynamics ablation (single seed first)
    # ====================================================================
    print("\n" + "#" * 70, flush=True)
    print("  CONTROL 1: UESD + Pure CE (no MSE, no SC)", flush=True)
    print("#" * 70, flush=True)
    config = build_config(seed=42)
    set_seed(42)
    model = UESDModel(
        config["vocab_size"], config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    )
    print(f"  Params: {count_params(model):,}", flush=True)
    tr = train(model, "addition", "dynamics_ce", config, device)
    ev = evaluate_uesd(model, "addition", config, device)
    results["dynamics_ce"] = {
        "params": count_params(model),
        "seed": 42,
        "train": tr["history"],
        "elapsed_s": tr["elapsed_s"],
        "eval": ev,
    }
    dce_tok = ev["token_accuracy"]["token_acc"]
    dce_seq = ev["token_accuracy"]["seq_acc"]
    print(f"  Token Acc: {dce_tok:.4f}", flush=True)
    print(f"  Seq Acc:   {dce_seq:.4f}", flush=True)
    print(flush=True)

    # ====================================================================
    # CONTROL 2: Depth-matched encoder-only (4-layer and 8-layer)
    # ====================================================================
    for n_layers in [4, 8]:
        print("#" * 70, flush=True)
        print(f"  CONTROL 2: Encoder-Only {n_layers}L", flush=True)
        print("#" * 70, flush=True)
        config = build_config(seed=42)
        set_seed(42)
        model = DeepEncoderOnly(
            config["vocab_size"], config["d_model"], config["n_heads"],
            config["d_ff"], n_layers, config["max_len"],
        )
        print(f"  Params: {count_params(model):,}", flush=True)
        tr = train(model, "addition", "encoder_only", config, device)
        ev = evaluate_encoder_only(model, "addition", config, device)
        results[f"enc_{n_layers}L"] = {
            "params": count_params(model),
            "n_enc_layers": n_layers,
            "seed": 42,
            "train": tr["history"],
            "elapsed_s": tr["elapsed_s"],
            "eval": ev,
        }
        tok = ev["token_accuracy"]["token_acc"]
        seq = ev["token_accuracy"]["seq_acc"]
        print(f"  Token Acc: {tok:.4f}", flush=True)
        print(f"  Seq Acc:   {seq:.4f}", flush=True)
        print(flush=True)

    # ====================================================================
    # CONTROL 3: Seed sweep for key models on addition
    # ====================================================================
    print("#" * 70, flush=True)
    print("  CONTROL 3: 5-SEED SWEEP (E5 lam=1.0 + Encoder-Only 2L)", flush=True)
    print("#" * 70, flush=True)

    for model_type in ["e5", "enc_2L"]:
        seed_results = []
        for seed in SEEDS:
            config = build_config(seed=seed)
            print(f"  {model_type} seed={seed}...", end=" ", flush=True)

            set_seed(seed)
            if model_type == "e5":
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

        import statistics
        tok_accs = [r["token_acc"] for r in seed_results]
        seq_accs = [r["seq_acc"] for r in seed_results]
        results[f"seed_sweep_{model_type}"] = {
            "seeds": SEEDS,
            "runs": seed_results,
            "token_acc_mean": statistics.mean(tok_accs),
            "token_acc_std": statistics.stdev(tok_accs) if len(tok_accs) > 1 else 0,
            "seq_acc_mean": statistics.mean(seq_accs),
            "seq_acc_std": statistics.stdev(seq_accs) if len(seq_accs) > 1 else 0,
        }
        print(f"  {model_type} mean tok={statistics.mean(tok_accs):.4f} "
              f"(+/-{statistics.stdev(tok_accs):.4f}), "
              f"seq={statistics.mean(seq_accs):.4f} "
              f"(+/-{statistics.stdev(seq_accs):.4f})", flush=True)
        print(flush=True)

    # ====================================================================
    # SUMMARY & GATES
    # ====================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXP D2 CONTROL RESULTS", flush=True)
    print("=" * 70, flush=True)

    # CE-dynamics result
    dce = results["dynamics_ce"]["eval"]["token_accuracy"]
    print(f"\n  CE-dynamics:       tok={dce['token_acc']:.4f}, seq={dce['seq_acc']:.4f}", flush=True)
    if dce["token_acc"] >= 0.80:
        print("    -> Dynamics + CE alone SUCCEEDS. SC not required.", flush=True)
        results["ce_dynamics_verdict"] = "DYNAMICS_SUFFICIENT"
    else:
        print("    -> Dynamics + CE alone FAILS. SC term essential.", flush=True)
        results["ce_dynamics_verdict"] = "SC_REQUIRED"

    # Depth-matched encoders
    for n_layers in [4, 8]:
        key = f"enc_{n_layers}L"
        if key in results:
            ev = results[key]["eval"]["token_accuracy"]
            print(f"  Encoder-{n_layers}L:       tok={ev['token_acc']:.4f}, seq={ev['seq_acc']:.4f}", flush=True)
            if ev["token_acc"] >= 0.80:
                print(f"    -> {n_layers}-layer encoder SUCCEEDS. Depth alone sufficient.", flush=True)
            else:
                print(f"    -> {n_layers}-layer encoder FAILS. Depth alone insufficient.", flush=True)

    # Seed sweep summary
    for model_type in ["e5", "enc_2L"]:
        key = f"seed_sweep_{model_type}"
        if key in results:
            ss = results[key]
            print(f"  {model_type} 5-seed:    tok={ss['token_acc_mean']:.4f}+/-{ss['token_acc_std']:.4f}, "
                  f"seq={ss['seq_acc_mean']:.4f}+/-{ss['seq_acc_std']:.4f}", flush=True)

    # Overall verdict
    print("\n" + "=" * 70, flush=True)
    dce_ok = dce["token_acc"] >= 0.80
    enc4_ok = results.get("enc_4L", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0) >= 0.80
    enc8_ok = results.get("enc_8L", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0) >= 0.80

    if dce_ok and not enc4_ok and not enc8_ok:
        verdict = "STRONG: Dynamics + CE succeeds, deep encoders fail. Iterative dynamics necessary."
        results["overall_verdict"] = "DYNAMICS_NECESSARY_CONFIRMED"
    elif dce_ok and (enc4_ok or enc8_ok):
        verdict = "WEAK: Both dynamics and deep encoder succeed. Depth suffices — dynamics convenient but not necessary."
        results["overall_verdict"] = "DEPTH_SUFFICIENT"
    elif not dce_ok:
        verdict = "SC_ESSENTIAL: Dynamics alone (without SC) insufficient. E5 success is about SC loss design."
        results["overall_verdict"] = "SC_LOSS_DESIGN"
    else:
        verdict = "INCONCLUSIVE"
        results["overall_verdict"] = "INCONCLUSIVE"

    print(f"  VERDICT: {verdict}", flush=True)
    print("=" * 70, flush=True)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d2_controls.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
