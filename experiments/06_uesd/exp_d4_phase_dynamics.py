"""
Experiment D4: Phase Transition Dynamics

Track how trajectory stability evolves DURING training, not just after.
Key questions from Codex D3 review:
1. Does edge-of-chaos regime emerge suddenly (phase transition) or gradually?
2. Does the onset of Jacobian rotation correlate with the CE loss phase transition?
3. How do per-step sigma_max profiles change as training progresses?
4. Does the shuffled vs ordered trajectory gap widen during training?

Diagnostic snapshots every 500 steps for CE-dynamics and E5.
Each snapshot: lightweight trajectory analysis (4 samples, fast).
Full analysis at steps 1, 1000, 2000, ..., 20000.
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
def snapshot_trajectory(model, src_ids, T, n_samples=4):
    """Lightweight trajectory analysis for mid-training snapshots."""
    model.eval()
    context = model.encode(src_ids)
    B = src_ids.shape[0]
    s = model.init_state(B, src_ids.shape[1], src_ids.device)

    states = [s.clone()]
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    n_samples = min(n_samples, B)
    indices = list(range(n_samples))

    lyapunov_exponents = []
    per_step_sigmas_all = []
    alignments_all = []
    ordered_vs_shuffled = []

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

        perm = list(range(T))
        random.shuffle(perm)
        shuffled_prod = jacobians[perm[0]].clone()
        for t in perm[1:]:
            shuffled_prod = jacobians[t] @ shuffled_prod
        shuffled_sigma = torch.linalg.svdvals(shuffled_prod)[0].item()

        lyap = math.log(max(cum_sigma, 1e-30)) / T
        lyapunov_exponents.append(lyap)
        per_step_sigmas_all.append(per_step_sigma)
        alignments_all.append(alignments)
        ordered_vs_shuffled.append({
            "ordered": cum_sigma,
            "shuffled": shuffled_sigma,
        })

    def avg_lists(lol):
        return [sum(x[i] for x in lol) / len(lol) for i in range(len(lol[0]))]

    avg_sigma = avg_lists(per_step_sigmas_all)
    avg_align = avg_lists(alignments_all) if alignments_all[0] else []
    mean_lyap = sum(lyapunov_exponents) / len(lyapunov_exponents)
    mean_ordered = sum(d["ordered"] for d in ordered_vs_shuffled) / len(ordered_vs_shuffled)
    mean_shuffled = sum(d["shuffled"] for d in ordered_vs_shuffled) / len(ordered_vs_shuffled)

    product_of_sigmas = 1.0
    for s in avg_sigma:
        product_of_sigmas *= s
    max_sigma_T = max(avg_sigma) ** T

    model.train()

    return {
        "lyapunov_mean": mean_lyap,
        "actual_amplification": mean_ordered,
        "shuffled_amplification": mean_shuffled,
        "ordered_shuffled_ratio": mean_ordered / max(mean_shuffled, 1e-30),
        "per_step_sigma": avg_sigma,
        "sv_alignment": avg_align,
        "mean_alignment": sum(avg_align) / len(avg_align) if avg_align else 0,
        "product_of_sigmas": product_of_sigmas,
        "max_sigma_T": max_sigma_T,
        "conservatism_product": product_of_sigmas / max(mean_ordered, 1e-30),
        "conservatism_max": max_sigma_T / max(mean_ordered, 1e-30),
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


def train_with_snapshots(model, task, track, config, device, diag_interval=500, diag_samples=4):
    """Training loop with periodic trajectory diagnostics."""
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

    history = []
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
            snap = snapshot_trajectory(model, eval_src, T, n_samples=diag_samples)
            snap["step"] = step
            snap["loss"] = loss.item()
            if track == "e5":
                snap["ce_loss"] = info.get("ce_loss", 0)
                snap["sc_loss"] = info.get("sc_loss", 0)
            snapshots.append(snap)

            status = "STABLE" if snap["lyapunov_mean"] < 0 else "UNSTABLE"
            print(f"    [DIAG step={step}] lyap={snap['lyapunov_mean']:.4f} ({status}) "
                  f"amp={snap['actual_amplification']:.2f}x "
                  f"shuf={snap['shuffled_amplification']:.2f}x "
                  f"align={snap['mean_alignment']:.3f} "
                  f"c_prod={snap['conservatism_product']:.1f}x",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s", flush=True)

    return {
        "history": history,
        "snapshots": snapshots,
        "elapsed_s": elapsed,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Phase transition dynamics: how trajectory stability evolves during training",
    }

    runs = []

    for track in ["dynamics_ce", "e5"]:
        seed = 42
        print(f"\n{'#' * 60}", flush=True)
        print(f"  {track} seed={seed} — phase dynamics", flush=True)
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
            diag_interval=500, diag_samples=4,
        )

        run_data = {
            "track": track,
            "seed": seed,
            "snapshots": tr["snapshots"],
            "elapsed_s": tr["elapsed_s"],
        }
        runs.append(run_data)

        print(f"\n  --- Phase Dynamics Timeline ({track}) ---", flush=True)
        print(f"  {'step':>6s} {'loss':>8s} {'lyap':>8s} {'amp':>8s} {'shuf':>8s} "
              f"{'o/s':>6s} {'align':>6s} {'c_prod':>8s}", flush=True)
        print(f"  {'-'*66}", flush=True)
        for s in tr["snapshots"]:
            print(f"  {s['step']:6d} {s['loss']:8.4f} {s['lyapunov_mean']:8.4f} "
                  f"{s['actual_amplification']:8.2f} {s['shuffled_amplification']:8.2f} "
                  f"{s['ordered_shuffled_ratio']:6.3f} {s['mean_alignment']:6.3f} "
                  f"{s['conservatism_product']:8.1f}", flush=True)

    results["runs"] = runs

    print(f"\n{'=' * 70}", flush=True)
    print("D4 PHASE DYNAMICS SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    for run_data in runs:
        track = run_data["track"]
        snaps = run_data["snapshots"]
        first = snaps[0]
        last = snaps[-1]

        phase_step = None
        for i, s in enumerate(snaps):
            if i > 0 and s["loss"] < 0.5 and snaps[i-1]["loss"] > 1.0:
                phase_step = s["step"]
                break

        print(f"\n  {track}:", flush=True)
        print(f"    Lyapunov: {first['lyapunov_mean']:.4f} (step 1) -> "
              f"{last['lyapunov_mean']:.4f} (step {last['step']})", flush=True)
        print(f"    Amplification: {first['actual_amplification']:.2f}x -> "
              f"{last['actual_amplification']:.2f}x", flush=True)
        print(f"    Mean alignment: {first['mean_alignment']:.3f} -> "
              f"{last['mean_alignment']:.3f}", flush=True)
        if phase_step:
            print(f"    CE phase transition at step ~{phase_step}", flush=True)

    print(f"{'=' * 70}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d4_phase_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
