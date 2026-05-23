"""
Experiment D5: Multi-Seed Stability + Failure Connection

Combines two goals:
1. MULTI-SEED D4 VALIDATION (Codex priority 10/8/9): 5 seeds per track
   to validate whether the CE-dynamics/E5 stability mechanism difference
   is robust or a seed=42 artifact.
2. FAILURE CONNECTION: E5 seed=512 fails in D2b (stuck CE=2.08). Does its
   stability trajectory predict the failure?

METHODOLOGY FIXES (from Codex D4 review):
- n_samples: 4 → 8 (2x more diagnostic examples)
- Shuffles: 1 → 5 per trajectory (proper null distribution)
- Randomize diagnostic example selection
- Report per-seed summary metrics with cross-seed statistics

PREDICTIONS:
1. CE-dynamics: all 5 seeds show three-phase regime (untrained/exploring/settled).
   Alignment dip below 0.15 in all seeds.
2. E5 successful seeds: high alignment throughout (>0.7), continuous lyap reduction.
3. E5 failing seed (512): alignment stays high BUT lyapunov/amplification stay
   elevated. No CE transition. "Highway to wrong attractor."
4. Cross-seed variance: CE-dynamics alignment variance peaks during exploring
   phase, E5 alignment variance stays low.

Seeds: [42, 137, 256, 512, 1024] (matching D2b for direct comparison)
"""
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

N_SHUFFLES = 5
N_DIAG_SAMPLES = 8
SEEDS = [42, 137, 256, 512, 1024]


def build_config(seed=42):
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "max_len": 32,
        "seq_len": 8,
        "T": 10,
        "batch_size": 256,
        "lr": 3e-4,
        "training_steps": 20000,
        "warmup_steps": 5000,
        "seed": seed,
    }


@torch.no_grad()
def _full_jacobian(model, s_single, c_single, eps=1e-4):
    L, d = s_single.shape
    n = L * d
    eye = torch.eye(n, device=s_single.device, dtype=s_single.dtype)
    E = eye.reshape(n, L, d)
    s_rep = s_single.unsqueeze(0).expand(n, -1, -1)
    c_rep = c_single.unsqueeze(0).expand(n, -1, -1)
    G_plus, _ = model.dynamics_step(s_rep + eps * E, c_rep)
    G_minus, _ = model.dynamics_step(s_rep - eps * E, c_rep)
    J = ((G_plus - G_minus) / (2 * eps)).reshape(n, n).t()
    return J


@torch.no_grad()
def snapshot_trajectory(model, src_ids, T, n_samples=N_DIAG_SAMPLES,
                        n_shuffles=N_SHUFFLES):
    model.eval()
    context = model.encode(src_ids)
    B = src_ids.shape[0]
    s = model.init_state(B, src_ids.shape[1], src_ids.device)

    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    n_samples = min(n_samples, B)
    indices = random.sample(range(B), n_samples)

    lyapunov_exponents = []
    per_step_sigmas_all = []
    alignments_all = []
    ordered_sigs = []
    shuffled_sigs_all = []

    for idx in indices:
        c_i = context[idx]
        jacobians = []
        per_step_sigma = []
        alignments = []
        prev_right_sv = None

        for t in range(T):
            s_t = states[t][idx]
            J_t = _full_jacobian(model, s_t, c_i)
            jacobians.append(J_t)

            sm = torch.linalg.svdvals(J_t)[0].item()
            per_step_sigma.append(sm)

            _, _, Vh = torch.linalg.svd(J_t)
            right_sv = Vh[0]
            if prev_right_sv is not None:
                al = torch.dot(right_sv, prev_right_sv).abs().item()
                alignments.append(al)
            prev_right_sv = right_sv

        product_J = jacobians[0].clone()
        for t in range(1, T):
            product_J = jacobians[t] @ product_J
        cum_sigma = torch.linalg.svdvals(product_J)[0].item()

        sample_shuffled = []
        for _ in range(n_shuffles):
            perm = list(range(T))
            random.shuffle(perm)
            shuffled_prod = jacobians[perm[0]].clone()
            for t in perm[1:]:
                shuffled_prod = jacobians[t] @ shuffled_prod
            sample_shuffled.append(torch.linalg.svdvals(shuffled_prod)[0].item())

        lyap = math.log(max(cum_sigma, 1e-30)) / T
        lyapunov_exponents.append(lyap)
        per_step_sigmas_all.append(per_step_sigma)
        alignments_all.append(alignments)
        ordered_sigs.append(cum_sigma)
        shuffled_sigs_all.append(sample_shuffled)

    def avg_lists(lol):
        return [sum(x[i] for x in lol) / len(lol) for i in range(len(lol[0]))]

    avg_sigma = avg_lists(per_step_sigmas_all)
    avg_align = avg_lists(alignments_all) if alignments_all[0] else []
    mean_lyap = sum(lyapunov_exponents) / len(lyapunov_exponents)
    mean_ordered = sum(ordered_sigs) / len(ordered_sigs)

    all_shuffled_flat = [v for sl in shuffled_sigs_all for v in sl]
    mean_shuffled = sum(all_shuffled_flat) / len(all_shuffled_flat)

    product_of_sigmas = 1.0
    for sv in avg_sigma:
        product_of_sigmas *= sv
    max_sigma_T = max(avg_sigma) ** T

    os_ratios = []
    for i in range(n_samples):
        shuf_mean = sum(shuffled_sigs_all[i]) / len(shuffled_sigs_all[i])
        os_ratios.append(ordered_sigs[i] / max(shuf_mean, 1e-30))
    mean_os = sum(os_ratios) / len(os_ratios)

    model.train()

    return {
        "lyapunov_mean": mean_lyap,
        "lyapunov_std": (sum((l - mean_lyap)**2 for l in lyapunov_exponents)
                         / len(lyapunov_exponents)) ** 0.5,
        "actual_amplification": mean_ordered,
        "shuffled_amplification": mean_shuffled,
        "ordered_shuffled_ratio": mean_os,
        "per_step_sigma": avg_sigma,
        "sv_alignment": avg_align,
        "mean_alignment": sum(avg_align) / len(avg_align) if avg_align else 0,
        "alignment_std": (sum((a - sum(avg_align)/len(avg_align))**2
                              for a in avg_align) / len(avg_align)) ** 0.5
                         if avg_align else 0,
        "product_of_sigmas": product_of_sigmas,
        "max_sigma_T": max_sigma_T,
        "conservatism_product": product_of_sigmas / max(mean_ordered, 1e-30),
        "conservatism_max": max_sigma_T / max(mean_ordered, 1e-30),
        "n_samples": n_samples,
        "n_shuffles": n_shuffles,
    }


def _e5_step(model, src, tgt, T, step, warmup_steps, lambda_1):
    context = model.encode(src)
    B, L_out = src.shape
    s = model.init_state(B, L_out, src.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
    logits = model.readout_logits(s)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    sc = (s - model.dynamics(s, context)).pow(2).mean()
    eff_lam = min(step / warmup_steps, 1.0) * lambda_1
    loss = ce + eff_lam * sc
    return loss, {"ce_loss": ce.item(), "sc_loss": sc.item(), "eff_lambda": eff_lam}


def _dynamics_ce_step(model, src, tgt, T, **kwargs):
    context = model.encode(src)
    B, L_out = src.shape
    s = model.init_state(B, L_out, src.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
    logits = model.readout_logits(s)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    return loss, {}


def train_with_snapshots(model, task, track, config, device,
                         diag_interval=1000, diag_samples=N_DIAG_SAMPLES):
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 3e-4))
    total_steps = config["training_steps"]
    T = config["T"]

    eval_src, eval_tgt = generate_batch(
        task, config.get("eval_batch_size", 512),
        config["seq_len"], config["vocab_size"],
    )
    eval_src, eval_tgt = eval_src.to(device), eval_tgt.to(device)

    snapshots = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        src, tgt = generate_batch(task, config["batch_size"], config["seq_len"], config["vocab_size"])
        src, tgt = src.to(device), tgt.to(device)

        if track == "dynamics_ce":
            loss, info = _dynamics_ce_step(model, src, tgt, T)
        elif track == "e5":
            loss, info = _e5_step(model, src, tgt, T, step,
                                  config.get("warmup_steps", 5000),
                                  config.get("lambda_1", 1.0))
        else:
            raise ValueError(f"Unknown track: {track}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step == 1:
            print(f"  Step {step:>6d}/{total_steps} | Loss: {loss.item():.4f}"
                  + (f" | CE: {info.get('ce_loss',0):.4f} | SC: {info.get('sc_loss',0):.4f}"
                     if track == "e5" else ""),
                  flush=True)

        if step % diag_interval == 0 or step == 1:
            snap = snapshot_trajectory(model, eval_src, T,
                                       n_samples=diag_samples)
            snap["step"] = step
            snap["loss"] = loss.item()
            if track == "e5":
                snap["ce_loss"] = info.get("ce_loss", 0)
                snap["sc_loss"] = info.get("sc_loss", 0)
            snapshots.append(snap)

            status = "STABLE" if snap["lyapunov_mean"] < 0 else "UNSTABLE"
            print(f"    [DIAG step={step}] lyap={snap['lyapunov_mean']:.4f} ({status}) "
                  f"amp={snap['actual_amplification']:.2f}x "
                  f"align={snap['mean_alignment']:.3f} "
                  f"o/s={snap['ordered_shuffled_ratio']:.3f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s", flush=True)

    return {"snapshots": snapshots, "elapsed_s": elapsed}


def classify_outcome(snapshots):
    final = snapshots[-1]
    loss = final.get("ce_loss", final["loss"])
    return "FAIL" if loss > 1.0 else "SUCCESS"


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Multi-seed stability validation + failure connection hypothesis",
        "methodology": {
            "n_diagnostic_samples": N_DIAG_SAMPLES,
            "n_shuffles": N_SHUFFLES,
            "diagnostic_interval": 1000,
            "seeds": SEEDS,
        },
    }

    runs = []

    for track in ["e5", "dynamics_ce"]:
        for seed in SEEDS:
            print(f"\n{'#' * 60}", flush=True)
            print(f"  {track} seed={seed}", flush=True)
            print(f"{'#' * 60}", flush=True)

            config = build_config(seed=seed)
            if track == "e5":
                config["lambda_1"] = 1.0
            set_seed(seed)

            model = UESDModel(
                config["vocab_size"], config["d_model"], config["n_heads"],
                config["d_ff"], config["n_enc_layers"], config["max_len"],
            )
            print(f"  params: {count_params(model)}", flush=True)

            tr = train_with_snapshots(
                model, "addition", track, config, device,
                diag_interval=1000, diag_samples=N_DIAG_SAMPLES,
            )

            outcome = classify_outcome(tr["snapshots"])
            run_data = {
                "track": track,
                "seed": seed,
                "outcome": outcome,
                "snapshots": tr["snapshots"],
                "elapsed_s": tr["elapsed_s"],
            }
            runs.append(run_data)

            final = tr["snapshots"][-1]
            loss_val = final.get("ce_loss", final["loss"])
            print(f"\n  OUTCOME: {outcome} (final CE={loss_val:.4f})", flush=True)
            print(f"  Final: lyap={final['lyapunov_mean']:.4f} "
                  f"amp={final['actual_amplification']:.2f}x "
                  f"align={final['mean_alignment']:.3f} "
                  f"o/s={final['ordered_shuffled_ratio']:.3f}", flush=True)

    results["runs"] = runs

    # ========== ANALYSIS ==========
    print(f"\n{'=' * 80}", flush=True)
    print("D5 MULTI-SEED STABILITY + FAILURE CONNECTION", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Per-run summary table
    print(f"\n{'track':12s} {'seed':>5s} {'out':>5s} {'lyap_i':>7s} {'lyap_f':>7s} "
          f"{'ali_i':>6s} {'ali_min':>7s} {'ali_f':>6s} "
          f"{'amp_f':>6s} {'o/s_f':>6s} {'CE_f':>7s}", flush=True)
    print("-" * 95, flush=True)

    for r in runs:
        snaps = r["snapshots"]
        first = snaps[0]
        last = snaps[-1]
        align_vals = [s["mean_alignment"] for s in snaps]
        align_min = min(align_vals)
        final_ce = last.get("ce_loss", last["loss"])

        print(f"{r['track']:12s} {r['seed']:5d} {r['outcome']:>5s} "
              f"{first['lyapunov_mean']:7.4f} {last['lyapunov_mean']:7.4f} "
              f"{first['mean_alignment']:6.3f} {align_min:7.3f} {last['mean_alignment']:6.3f} "
              f"{last['actual_amplification']:6.2f} {last['ordered_shuffled_ratio']:6.3f} "
              f"{final_ce:7.4f}", flush=True)

    # Cross-seed statistics per track
    for track in ["e5", "dynamics_ce"]:
        track_runs = [r for r in runs if r["track"] == track]
        success_runs = [r for r in track_runs if r["outcome"] == "SUCCESS"]
        fail_runs = [r for r in track_runs if r["outcome"] == "FAIL"]

        print(f"\n  --- {track} cross-seed stats ---", flush=True)
        print(f"  Success: {len(success_runs)}/{len(track_runs)} "
              f"(seeds: {[r['seed'] for r in success_runs]})", flush=True)
        if fail_runs:
            print(f"  Fail: {len(fail_runs)}/{len(track_runs)} "
                  f"(seeds: {[r['seed'] for r in fail_runs]})", flush=True)

        if success_runs:
            final_lyaps = [r["snapshots"][-1]["lyapunov_mean"] for r in success_runs]
            final_amps = [r["snapshots"][-1]["actual_amplification"] for r in success_runs]
            final_aligns = [r["snapshots"][-1]["mean_alignment"] for r in success_runs]
            final_os = [r["snapshots"][-1]["ordered_shuffled_ratio"] for r in success_runs]

            def mean_std(vals):
                m = sum(vals) / len(vals)
                s = (sum((v-m)**2 for v in vals) / len(vals)) ** 0.5
                return m, s

            lm, ls = mean_std(final_lyaps)
            am, asm = mean_std(final_amps)
            alm, als = mean_std(final_aligns)
            osm, oss = mean_std(final_os)

            print(f"  SUCCESS final metrics (mean +/- std):", flush=True)
            print(f"    Lyapunov:      {lm:.4f} +/- {ls:.4f}", flush=True)
            print(f"    Amplification: {am:.2f} +/- {asm:.2f}", flush=True)
            print(f"    Alignment:     {alm:.3f} +/- {als:.3f}", flush=True)
            print(f"    O/S ratio:     {osm:.3f} +/- {oss:.3f}", flush=True)

        if fail_runs:
            fail_lyaps = [r["snapshots"][-1]["lyapunov_mean"] for r in fail_runs]
            fail_amps = [r["snapshots"][-1]["actual_amplification"] for r in fail_runs]
            fail_aligns = [r["snapshots"][-1]["mean_alignment"] for r in fail_runs]
            print(f"  FAIL final metrics:", flush=True)
            for r in fail_runs:
                f = r["snapshots"][-1]
                print(f"    seed={r['seed']}: lyap={f['lyapunov_mean']:.4f} "
                      f"amp={f['actual_amplification']:.2f} "
                      f"align={f['mean_alignment']:.3f}", flush=True)

    # Alignment trajectory comparison (key hypothesis test)
    e5_runs = [r for r in runs if r["track"] == "e5"]
    e5_success = [r for r in e5_runs if r["outcome"] == "SUCCESS"]
    e5_fail = [r for r in e5_runs if r["outcome"] == "FAIL"]
    ce_runs = [r for r in runs if r["track"] == "dynamics_ce"]

    if e5_success and e5_fail:
        print(f"\n  --- FAILURE HYPOTHESIS TEST ---", flush=True)
        steps_to_check = [s["step"] for s in e5_success[0]["snapshots"]]
        for step in steps_to_check:
            s_aligns = []
            f_aligns = []
            s_lyaps = []
            f_lyaps = []
            for r in e5_success:
                for sn in r["snapshots"]:
                    if sn["step"] == step:
                        s_aligns.append(sn["mean_alignment"])
                        s_lyaps.append(sn["lyapunov_mean"])
            for r in e5_fail:
                for sn in r["snapshots"]:
                    if sn["step"] == step:
                        f_aligns.append(sn["mean_alignment"])
                        f_lyaps.append(sn["lyapunov_mean"])

            if s_aligns and f_aligns:
                sm = sum(s_aligns) / len(s_aligns)
                fm = sum(f_aligns) / len(f_aligns)
                sl = sum(s_lyaps) / len(s_lyaps)
                fl = sum(f_lyaps) / len(f_lyaps)
                print(f"  Step {step:>5d}: succ_align={sm:.3f} fail_align={fm:.3f} "
                      f"diff={fm-sm:+.3f} | succ_lyap={sl:.4f} fail_lyap={fl:.4f}", flush=True)

    # CE-dynamics alignment dip analysis
    if ce_runs:
        print(f"\n  --- CE-DYNAMICS ALIGNMENT DIP ANALYSIS ---", flush=True)
        for r in ce_runs:
            snaps = r["snapshots"]
            align_vals = [s["mean_alignment"] for s in snaps]
            align_min = min(align_vals)
            align_min_idx = align_vals.index(align_min)
            align_min_step = snaps[align_min_idx]["step"]
            align_recovery = snaps[min(align_min_idx + 3, len(snaps)-1)]["mean_alignment"]
            print(f"  seed={r['seed']}: align_min={align_min:.3f} at step {align_min_step} "
                  f"-> recovery={align_recovery:.3f} | "
                  f"outcome={r['outcome']}", flush=True)

    print(f"\n{'=' * 80}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d5_failure_stability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
