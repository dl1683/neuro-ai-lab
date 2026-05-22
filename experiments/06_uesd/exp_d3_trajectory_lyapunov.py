"""
Experiment D3: Trajectory Lyapunov Exponents

Why does UESD work when sigma_max > 1? D2c showed the Theorem 4 bound
(sigma_max^T) overestimates actual instability by 3+ orders of magnitude.

This experiment measures the PRODUCT of Jacobians along the dynamics
trajectory, not just the single-point Jacobian at convergence. The key
quantity is the maximum Lyapunov exponent:

    lambda_max = (1/T) * log(sigma_max(J_T * J_{T-1} * ... * J_1))

If lambda_max < 0, the trajectory is asymptotically stable even when
individual per-step sigma_max > 1. This happens when the dominant
singular value directions ROTATE between steps (don't align), creating
effective damping despite per-step amplification.

Additional measurements:
- Per-step Jacobian properties: sigma_max_t, kappa_t at each step t
- Cumulative product: sigma_max(P_t) vs step t
- Singular vector alignment: cos(v_max(J_t), v_max(J_{t+1}))
- Theorem 4 conservatism ratio: sigma_max^T / sigma_max(product)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import train, set_seed, count_params
from shared.diagnostics import trajectory_lyapunov, sigma_max_ratio
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
        "eval_samples": 512,
        "eval_batch_size": 512,
        "warmup_steps": 5000,
    }
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def run_trajectory_analysis(model, config, device, n_samples=16):
    """Run trajectory Lyapunov analysis on trained model."""
    model.eval()
    T = config["T"]

    src, tgt = generate_batch(
        "addition", config["eval_batch_size"],
        config["seq_len"], config["vocab_size"],
    )
    src, tgt = src.to(device), tgt.to(device)

    print("    computing trajectory Jacobians (this takes a few minutes)...", flush=True)
    traj = trajectory_lyapunov(model, src, T, n_samples=n_samples)

    with torch.no_grad():
        state, _ = model.unroll(src, T)
        context = model.encode(src)
    d7 = sigma_max_ratio(model, state, context, n_samples=n_samples)

    return traj, d7


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Trajectory Lyapunov exponents: why sigma_max > 1 but basin stability > 99%",
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

            print("  running trajectory Lyapunov analysis...", flush=True)
            traj, d7 = run_trajectory_analysis(model, config, device, n_samples=16)

            lyap = traj["lyapunov_max_mean"]
            actual_amp = traj["actual_amplification"]
            thm4_bound = traj["theorem4_bound"]
            conservatism = traj["conservatism_ratio"]
            d7_kappa = d7["kappa_mean"]
            d7_sigma = d7["sigma_max_mean"]

            print(f"  D7 (single-point): sigma_max={d7_sigma:.4f}, kappa={d7_kappa:.4f}", flush=True)
            print(f"  Lyapunov max: {lyap:.6f} ({'STABLE' if lyap < 0 else 'UNSTABLE'})", flush=True)
            print(f"  Theorem 4 bound: {thm4_bound:.2f}x", flush=True)
            print(f"  Actual amplification: {actual_amp:.4f}x", flush=True)
            print(f"  Conservatism ratio: {conservatism:.1f}x", flush=True)

            print("  Per-step sigma_max:", flush=True)
            for t, sm in enumerate(traj["per_step_sigma_max"]):
                print(f"    step {t+1}: sigma_max={sm:.4f}", flush=True)

            print("  Cumulative sigma_max(product):", flush=True)
            for t, csm in enumerate(traj["cumulative_sigma_max"]):
                print(f"    steps 1..{t+1}: sigma_max(prod)={csm:.4f}", flush=True)

            if traj["sv_alignment"]:
                print("  Singular vector alignment:", flush=True)
                for t, al in enumerate(traj["sv_alignment"]):
                    print(f"    step {t+1}->{t+2}: cos={al:.4f}", flush=True)

            run_data = {
                "track": track,
                "seed": seed,
                "train_final_loss": final_loss,
                "elapsed_s": tr["elapsed_s"],
                "trajectory": traj,
                "d7_single_point": d7,
            }
            runs.append(run_data)

    results["runs"] = runs

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("D3 TRAJECTORY LYAPUNOV SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'track':15s} {'seed':>5s} {'lyap':>8s} {'status':>8s} {'thm4':>10s} {'actual':>8s} {'ratio':>8s} {'kappa':>6s}", flush=True)
    print("-" * 70, flush=True)
    for r in runs:
        t = r["trajectory"]
        d = r["d7_single_point"]
        status = "STABLE" if t["lyapunov_max_mean"] < 0 else "UNSTABLE"
        print(f"{r['track']:15s} {r['seed']:5d} "
              f"{t['lyapunov_max_mean']:8.5f} {status:>8s} "
              f"{t['theorem4_bound']:10.1f} {t['actual_amplification']:8.4f} "
              f"{t['conservatism_ratio']:8.1f} {d['kappa_mean']:6.3f}", flush=True)
    print(f"{'=' * 70}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d3_trajectory_lyapunov.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
