"""
Experiment D2c: Full Stability Analysis with D7 (σ_max / ρ)

Tests the Theorem 4 prediction: non-normality ratio κ = σ_max/ρ
determines whether finite-T convergence is reliable. Trains CE-dynamics
and E5 models, then runs complete D1-D7 diagnostics on converged states.

Key questions:
- Is κ < 1.5 for all trained models (mild non-normality)?
- Does κ differ between CE-dynamics (robust) and E5 (bimodal)?
- Does κ correlate with wrong-attractor rate?
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import train, set_seed, count_params
from shared.diagnostics import run_all_diagnostics, sigma_max_ratio
from shared.data import generate_batch


SEEDS = [42, 137]


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
        "eval_samples": 2048,
        "eval_batch_size": 512,
        "warmup_steps": 5000,
    }
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def evaluate_full(model, config, device):
    """Run D1-D7 on fresh evaluation data."""
    model.eval()
    T = config["T"]
    n_batches = config["eval_samples"] // config["eval_batch_size"]

    all_d1_d6 = []

    for _ in range(n_batches):
        src, tgt = generate_batch(
            "addition", config["eval_batch_size"],
            config["seq_len"], config["vocab_size"],
        )
        src, tgt = src.to(device), tgt.to(device)

        diag_cfg = {"compute_d7": False}
        result = run_all_diagnostics(model, src, tgt, T, config=diag_cfg)
        all_d1_d6.append(result)

    # D7 on a single batch (expensive per-example Jacobian)
    src, tgt = generate_batch(
        "addition", config["eval_batch_size"],
        config["seq_len"], config["vocab_size"],
    )
    src, tgt = src.to(device), tgt.to(device)

    with torch.no_grad():
        state, _ = model.unroll(src, T)
        context = model.encode(src)

    d7 = sigma_max_ratio(model, state, context, n_samples=8)

    # Average D1-D6 across batches
    averaged = {}
    keys_to_avg = ["token_accuracy", "normalized_residual", "decoder_margin",
                   "wrong_attractor", "basin_perturbation", "spectral_radius"]
    for key in keys_to_avg:
        sub_keys = all_d1_d6[0][key].keys()
        averaged[key] = {}
        for sk in sub_keys:
            vals = [r[key][sk] for r in all_d1_d6]
            averaged[key][sk] = sum(vals) / len(vals)

    averaged["sigma_max_ratio"] = d7
    return averaged


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Full D1-D7 stability analysis: σ_max/ρ non-normality ratio on CE-dynamics vs E5",
    }

    runs = []

    for track in ["dynamics_ce", "e5"]:
        for seed in SEEDS:
            print(f"\n{'#' * 60}", flush=True)
            print(f"  {track} seed={seed}", flush=True)
            print(f"{'#' * 60}", flush=True)

            config = build_config(seed=seed)
            set_seed(seed)

            model = UESDModel(
                config["vocab_size"], config["d_model"], config["n_heads"],
                config["d_ff"], config["n_enc_layers"], config["max_len"],
            )
            print(f"  params: {count_params(model)}", flush=True)

            if track == "e5":
                config["lambda_1"] = 1.0

            tr = train(model, "addition", track, config, device)
            final_loss = tr["history"][-1]["loss"] if tr["history"] else None
            print(f"  training done in {tr['elapsed_s']:.0f}s, final loss={final_loss}", flush=True)

            print("  running D1-D7 diagnostics...", flush=True)
            diag = evaluate_full(model, config, device)

            tok = diag["token_accuracy"]["token_acc"]
            seq = diag["token_accuracy"]["seq_acc"]
            rho = diag["spectral_radius"]["mean_rho"]
            sigma = diag["sigma_max_ratio"]["sigma_max_mean"]
            kappa = diag["sigma_max_ratio"]["kappa_mean"]
            wa = diag["wrong_attractor"]["wrong_attractor_rate"]
            basin = diag["basin_perturbation"]["stability_frac"]

            print(f"  tok={tok:.4f} seq={seq:.4f}", flush=True)
            print(f"  ρ={rho:.4f} σ_max={sigma:.4f} κ={kappa:.4f}", flush=True)
            print(f"  WA={wa:.4f} basin={basin:.4f}", flush=True)

            run_data = {
                "track": track,
                "seed": seed,
                "train_final_loss": final_loss,
                "elapsed_s": tr["elapsed_s"],
                "diagnostics": diag,
            }
            runs.append(run_data)

    results["runs"] = runs

    # Summary table
    print(f"\n{'=' * 70}", flush=True)
    print("D2c STABILITY ANALYSIS RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'track':15s} {'seed':>5s} {'tok':>6s} {'seq':>6s} {'ρ':>6s} {'σ_max':>6s} {'κ':>6s} {'WA':>6s} {'basin':>6s}", flush=True)
    print("-" * 70, flush=True)
    for r in runs:
        d = r["diagnostics"]
        print(f"{r['track']:15s} {r['seed']:5d} "
              f"{d['token_accuracy']['token_acc']:6.4f} "
              f"{d['token_accuracy']['seq_acc']:6.4f} "
              f"{d['spectral_radius']['mean_rho']:6.4f} "
              f"{d['sigma_max_ratio']['sigma_max_mean']:6.4f} "
              f"{d['sigma_max_ratio']['kappa_mean']:6.4f} "
              f"{d['wrong_attractor']['wrong_attractor_rate']:6.4f} "
              f"{d['basin_perturbation']['stability_frac']:6.4f}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Theorem 4 check
    print("\nTheorem 4 prediction check:", flush=True)
    for r in runs:
        d = r["diagnostics"]
        kappa = d["sigma_max_ratio"]["kappa_mean"]
        rho = d["spectral_radius"]["mean_rho"]
        sigma = d["sigma_max_ratio"]["sigma_max_mean"]
        seq = d["token_accuracy"]["seq_acc"]
        if kappa < 1.5:
            status = "MILD (eigenvalue analysis reliable)"
        elif kappa < 2.0:
            status = "MODERATE (finite-T may have transient growth)"
        else:
            status = "SEVERE (σ_max bound needed, not ρ)"
        print(f"  {r['track']} s={r['seed']}: κ={kappa:.3f} → {status} | seq={seq:.4f}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d2c_stability_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
