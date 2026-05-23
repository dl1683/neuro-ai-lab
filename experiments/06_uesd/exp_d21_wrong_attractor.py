"""D21: Wrong-Attractor Rate Under Latent Noise

FALSIFICATION TARGET: "Stable iterative solver" thesis.
If small latent noise at converged states causes >5% wrong-attractor rate
that doesn't recover with extra steps, the system is not a robust solver.

Design:
  1. Train CE-dynamics and E5 models (standard config)
  2. Run to convergence (T=10), verify correct output
  3. Inject Gaussian noise N(0, sigma) at converged state s_T
  4. Run additional K steps and measure:
     - Wrong-attractor rate: fraction landing on WRONG answer
     - Recovery rate: fraction that returns to CORRECT answer
     - Basin escape threshold: smallest sigma causing >5% WA
  5. Sweep sigma in {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0}
  6. Extra steps K in {0, 1, 2, 5, 10, 20}

Source: Codex meta-analysis falsification test #5
"""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from shared.model import UESDModel
from shared.training import train, count_params, set_seed
from shared.data import generate_addition_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

CFG = {
    "vocab_size": 64, "d_model": 128, "n_heads": 4, "d_ff": 512,
    "n_enc_layers": 2, "max_len": 32, "T": 10,
    "training_steps": 20000, "batch_size": 256, "seq_len": 8,
    "lr": 3e-4, "log_interval": 5000, "seed": 42,
}
SIGMA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
EXTRA_STEPS = [0, 1, 2, 5, 10, 20]
N_EVAL = 4096


@torch.no_grad()
def wrong_attractor_analysis(model, sigma_values, extra_steps,
                              n_samples=N_EVAL, batch_size=512):
    """Measure wrong-attractor rate and recovery under noise."""
    model.eval()
    half = CFG["seq_len"] // 2
    T = CFG["T"]

    all_src, all_tgt = [], []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_addition_batch(bs, CFG["seq_len"], CFG["vocab_size"])
        all_src.append(src)
        all_tgt.append(tgt)
        remaining -= bs
    src = torch.cat(all_src).to(DEVICE)
    tgt = torch.cat(all_tgt).to(DEVICE)
    n = src.shape[0]

    context = model.encode(src)
    s = model.init_state(n, CFG["seq_len"], DEVICE)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)

    clean_logits = model.readout_logits(s)
    clean_preds = clean_logits.argmax(dim=-1)
    clean_correct = (clean_preds[:, :half] == tgt[:, :half]).all(dim=-1)
    n_correct = clean_correct.sum().item()
    print(f"    Clean baseline: {n_correct}/{n} correct ({n_correct/n:.4f})")

    if n_correct < 100:
        print(f"    Too few correct examples, skipping")
        return {}

    s_converged = s[clean_correct].clone()
    ctx_good = context[clean_correct]
    tgt_good = tgt[clean_correct]
    preds_good = clean_preds[clean_correct]

    state_norm = s_converged.norm(dim=-1).mean().item()
    print(f"    Converged state norm: {state_norm:.4f}")

    results = {}
    for sigma in sigma_values:
        results[sigma] = {}
        noise = torch.randn_like(s_converged) * sigma
        s_noisy = s_converged + noise

        for K in extra_steps:
            s_curr = s_noisy.clone()
            for _ in range(K):
                s_curr, _ = model.dynamics_step(s_curr, ctx_good)

            logits = model.readout_logits(s_curr)
            preds = logits.argmax(dim=-1)

            matches_clean = (preds[:, :half] == preds_good[:, :half]).all(dim=-1)
            matches_target = (preds[:, :half] == tgt_good[:, :half]).all(dim=-1)

            wrong_attractor = (~matches_clean & ~matches_target)
            wa_but_correct = (~matches_clean & matches_target)

            wa_rate = (~matches_target).float().mean().item()
            recovery_rate = matches_target.float().mean().item()
            true_wa_rate = wrong_attractor.float().mean().item()

            per_pos_correct = []
            for p in range(half):
                pc = (preds[:, p] == tgt_good[:, p]).float().mean().item()
                per_pos_correct.append(pc)

            results[sigma][K] = {
                "wrong_answer_rate": wa_rate,
                "correct_rate": recovery_rate,
                "true_wrong_attractor_rate": true_wa_rate,
                "per_position_accuracy": per_pos_correct,
            }

        wa_at_0 = results[sigma][0]["wrong_answer_rate"]
        wa_at_20 = results[sigma][20]["wrong_answer_rate"]
        recovery = wa_at_0 - wa_at_20
        print(f"    sigma={sigma:.2f}: WA@0={wa_at_0:.4f}, WA@20={wa_at_20:.4f}, "
              f"recovery={recovery:.4f}")

    basin_threshold = None
    for sigma in sigma_values:
        if results[sigma][0]["wrong_answer_rate"] > 0.05:
            basin_threshold = sigma
            break

    recovery_threshold = None
    for sigma in sigma_values:
        if results[sigma][20]["wrong_answer_rate"] > 0.05:
            recovery_threshold = sigma
            break

    summary = {
        "n_correct_baseline": n_correct,
        "state_norm": state_norm,
        "basin_escape_threshold": basin_threshold,
        "unrecoverable_threshold": recovery_threshold,
        "noise_results": {str(s): {str(k): v for k, v in kv.items()}
                         for s, kv in results.items()},
    }
    return summary


def run_track(track_name, track_type):
    print(f"\n{'='*60}")
    print(f"  TRACK: {track_name}")
    print(f"{'='*60}")

    set_seed(CFG["seed"])
    model = UESDModel(
        CFG["vocab_size"], CFG["d_model"], CFG["n_heads"],
        CFG["d_ff"], CFG["n_enc_layers"], CFG["max_len"],
    ).to(DEVICE)
    print(f"  Params: {count_params(model)}")

    train_result = train(model, "addition", track_type, CFG, device=DEVICE)

    print(f"\n  --- Wrong-Attractor Analysis ---")
    wa_results = wrong_attractor_analysis(model, SIGMA_VALUES, EXTRA_STEPS)
    wa_results["training_time"] = train_result["elapsed_s"]

    if wa_results.get("basin_escape_threshold") is not None:
        thresh = wa_results["basin_escape_threshold"]
        print(f"\n  Basin escape threshold: sigma={thresh}")
        if wa_results.get("unrecoverable_threshold") is not None:
            recov_thresh = wa_results["unrecoverable_threshold"]
            print(f"  Unrecoverable threshold: sigma={recov_thresh}")
            if recov_thresh <= thresh:
                print(f"  >>> THESIS WEAKENED: noise that escapes basin is never recovered <<<")
            else:
                print(f"  >>> PARTIAL: basin escapes at sigma={thresh}, but recovers up to sigma={recov_thresh} <<<")
        else:
            print(f"  >>> THESIS SUPPORTED: model recovers from all tested noise levels <<<")
    else:
        print(f"  >>> THESIS SUPPORTED: no basin escape even at sigma=2.0 <<<")

    return wa_results


def main():
    results = {"config": CFG, "sigma_values": SIGMA_VALUES,
               "extra_steps": EXTRA_STEPS, "tracks": {}}

    results["tracks"]["dynamics_ce"] = run_track("dynamics_ce", "dynamics_ce")
    results["tracks"]["e5"] = run_track("e5", "e5")

    print(f"\n{'='*60}")
    print(f"  D21 WRONG-ATTRACTOR SUMMARY")
    print(f"{'='*60}")
    for name, track in results["tracks"].items():
        if not track:
            continue
        print(f"\n  {name}:")
        print(f"    State norm: {track.get('state_norm', 'N/A')}")
        print(f"    Basin escape threshold: sigma={track.get('basin_escape_threshold', 'never')}")
        print(f"    Unrecoverable threshold: sigma={track.get('unrecoverable_threshold', 'never')}")

    out_path = "experiments/06_uesd/results/exp_d21_wrong_attractor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
