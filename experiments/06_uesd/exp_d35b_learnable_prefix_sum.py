"""
Experiment D35b: Learnable Prefix Sum (V=8) — Task Generalization Test

D35 showed prefix sum with V=64 is unlearnable at this scale, so k-suppression
could not be tested on a non-addition task. D35b uses V=8 (learnable) to
test whether VT k-suppression generalizes beyond addition.

Key changes from D35:
  1. VOCAB_SIZE = 8 (learnable in 20K steps)
  2. Intermediate k/rho measurements every 5K steps (matched-loss analysis)
  3. Per-position and token accuracy (not just sequence accuracy)
  4. Loss-matched comparison: compare k at equal loss, not equal step

If k-suppression appears on learned prefix sum, the mechanism generalizes
beyond addition. If it vanishes, k-suppression is addition-specific.

Design:
  seq_len = {6, 8}  (2 depths — sufficient for generalization test)
  4 seeds per depth (matched across FT/VT)
  2 variants (fixed_t, variable_t)
  Total: 2 x 4 x 2 = 16 runs

Codex synthesis recommendation: "Launch D35b with V=8. Needed to separate
'addition artifact' from 'learned finite-time dynamics mechanism.'"
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed
from shared.data import generate_batch

TRAINING_STEPS = 20000
BATCH_SIZE = 256
VOCAB_SIZE = 8
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64
TRAIN_T = 10
VARIABLE_T_RANGE = [4, 6, 8, 10, 12, 14, 16]
T_MIN = min(VARIABLE_T_RANGE)

FP_T = 100
EVAL_SAMPLES = 4096
TRAJECTORY_T = 30

SEEDS = [42, 137, 256, 512]

SEQ_LEN_CONFIGS = [
    {"seq_len": 6},
    {"seq_len": 8},
]

VARIANTS = ["fixed_t", "variable_t"]
Q_CHECKPOINT_INTERVAL = 2000
DYNAMICS_CHECKPOINT_INTERVAL = 5000

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d35b_learnable_prefix_sum.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def make_model(device):
    model = UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN)
    return model.to(device)


@torch.no_grad()
def measure_q_at_t_min(model, device, seq_len):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    model.eval()
    torch.manual_seed(12345)
    src, tgt = generate_batch("prefix_sum", 1024, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    logits = model(src, T_MIN)
    preds = logits.argmax(dim=-1)
    seq_acc = (preds == tgt).all(dim=1).float().mean().item()
    token_acc = (preds == tgt).float().mean().item()
    per_pos = [(preds[:, i] == tgt[:, i]).float().mean().item() for i in range(seq_len)]

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    random.setstate(py_state)
    np.random.set_state(np_state)
    return seq_acc, token_acc, per_pos


def _save_rng_state(device):
    state = {
        "cpu": torch.random.get_rng_state(),
        "py": random.getstate(),
        "np": np.random.get_state(),
    }
    if device == "cuda":
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state, device):
    torch.random.set_rng_state(state["cpu"])
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    if device == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


@torch.no_grad()
def measure_spectral_radius(model, device, seq_len, n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("prefix_sum", 256, seq_len, VOCAB_SIZE)
    src = src.to(device)

    with torch.no_grad():
        context = model.encode(src)
        s = model.init_state(src.size(0), src.size(1), device)
        for _ in range(FP_T):
            s, _ = model.dynamics_step(s, context)
        s_star = s.detach().clone()

    rhos = []
    for _ in range(n_vecs):
        v = torch.randn_like(s_star[:32])
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for _ in range(n_iter):
            with torch.enable_grad():
                s_pert = s_star[:32].detach().requires_grad_(True)
                s_next, _ = model.dynamics_step(s_pert, context[:32])
                jvp = torch.autograd.grad(
                    s_next, s_pert,
                    grad_outputs=v,
                    create_graph=False,
                )[0]
            new_norm = jvp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = jvp.detach() / new_norm

        rho = new_norm.squeeze(-1).mean().item()
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    std_rho = float(np.std(rhos))
    return {"mean": round(mean_rho, 4), "std": round(std_rho, 4)}


@torch.no_grad()
def measure_contraction_summary(model, device, seq_len):
    model.eval()
    full_seed(9999)
    src, tgt = generate_batch("prefix_sum", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    context = model.encode(src)

    s_star = model.init_state(src.size(0), src.size(1), device)
    for _ in range(FP_T):
        s_star, _ = model.dynamics_step(s_star, context)

    s = model.init_state(src.size(0), src.size(1), device)
    prev_dist = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)
    k_values = []
    for t in range(TRAJECTORY_T):
        s, _ = model.dynamics_step(s, context)
        curr_dist = (s - s_star).reshape(src.size(0), -1).norm(dim=-1)
        k_per = (curr_dist / prev_dist.clamp(min=1e-8)).mean().item()
        k_values.append(k_per)
        prev_dist = curr_dist

    stable_k = k_values[1:10]
    mean_k = float(np.mean(stable_k))
    std_k = float(np.std(stable_k))

    s = model.init_state(src.size(0), src.size(1), device)
    actual_T99 = None
    for t in range(1, TRAJECTORY_T + 1):
        s, _ = model.dynamics_step(s, context)
        logits = model.readout_logits(s)
        preds = logits.argmax(dim=-1)
        seq_acc = (preds == tgt).all(dim=1).float().mean().item()
        if seq_acc >= 0.99 and actual_T99 is None:
            actual_T99 = t

    return {
        "mean_k": round(mean_k, 4),
        "std_k": round(std_k, 4),
        "T_99": actual_T99,
    }


@torch.no_grad()
def measure_eval_loss(model, device, seq_len):
    """Standardized eval loss on fixed batch at TRAIN_T (common across FT/VT)."""
    model.eval()
    torch.manual_seed(54321)
    src, tgt = generate_batch("prefix_sum", 1024, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    logits = model(src, TRAIN_T)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    preds = logits.argmax(dim=-1)
    seq_acc = (preds == tgt).all(dim=1).float().mean().item()
    token_acc = (preds == tgt).float().mean().item()
    per_pos = [(preds[:, i] == tgt[:, i]).float().mean().item() for i in range(seq_len)]
    return loss.item(), seq_acc, token_acc, per_pos


def train_model(model, device, seq_len, variant, seed):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    best_acc = 0.0
    q_trajectory = {}
    eval_loss_trajectory = {}
    dynamics_checkpoints = []
    phase_transition_step = None

    for step in range(1, TRAINING_STEPS + 1):
        src, tgt = generate_batch("prefix_sum", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        if variant == "variable_t":
            t_steps = random.choice(VARIABLE_T_RANGE)
        else:
            t_steps = TRAIN_T

        logits = model(src, t_steps)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % Q_CHECKPOINT_INTERVAL == 0 or step == 1:
            seq_q, token_q, per_pos_q = measure_q_at_t_min(model, device, seq_len)
            q_trajectory[step] = round(seq_q, 4)

            rng_state = _save_rng_state(device)
            eval_l, eval_seq_acc, eval_tok_acc, _ = measure_eval_loss(model, device, seq_len)
            _restore_rng_state(rng_state, device)
            eval_loss_trajectory[step] = round(eval_l, 4)
            best_acc = max(best_acc, eval_seq_acc)

            if phase_transition_step is None and eval_seq_acc > 0.5:
                phase_transition_step = step

            t_label = f"T={t_steps}" if variant == "variable_t" else f"T={TRAIN_T}"
            print(f"      Step {step:>6d}/{TRAINING_STEPS} | {t_label} | "
                  f"EvalLoss: {eval_l:.4f} | SeqAcc: {eval_seq_acc:.4f} | TokAcc: {eval_tok_acc:.4f} | "
                  f"q(T={T_MIN}): {seq_q:.4f}", flush=True)
            model.train()

        if step % DYNAMICS_CHECKPOINT_INTERVAL == 0:
            rng_state = _save_rng_state(device)
            print(f"      [Dynamics checkpoint at step {step}]", flush=True)
            contraction = measure_contraction_summary(model, device, seq_len)
            spectral = measure_spectral_radius(model, device, seq_len)
            seq_q_ckpt, token_q_ckpt, per_pos_ckpt = measure_q_at_t_min(model, device, seq_len)
            eval_l_ckpt, eval_seq_ckpt, eval_tok_ckpt, eval_pos_ckpt = measure_eval_loss(model, device, seq_len)
            _restore_rng_state(rng_state, device)
            dynamics_checkpoints.append({
                "step": step,
                "k": contraction["mean_k"],
                "k_std": contraction["std_k"],
                "rho": spectral["mean"],
                "rho_std": spectral["std"],
                "T_99": contraction["T_99"],
                "eval_loss": round(eval_l_ckpt, 4),
                "eval_seq_acc": round(eval_seq_ckpt, 4),
                "eval_token_acc": round(eval_tok_ckpt, 4),
                "eval_per_pos_acc": [round(p, 4) for p in eval_pos_ckpt],
                "q_seq": round(seq_q_ckpt, 4),
                "q_token": round(token_q_ckpt, 4),
                "q_per_pos": [round(p, 4) for p in per_pos_ckpt],
            })
            print(f"      k={contraction['mean_k']:.4f}, rho={spectral['mean']:.4f}, "
                  f"T_99={contraction['T_99']}, eval_loss={eval_l_ckpt:.4f}, q={seq_q_ckpt:.4f}", flush=True)
            model.train()

    elapsed = time.time() - t0

    mean_tmin_acc = sum(q_trajectory.values()) / max(len(q_trajectory), 1)

    print(f"      Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f} | "
          f"Cumulative q: {mean_tmin_acc:.4f} | Phase trans: step {phase_transition_step}",
          flush=True)
    return elapsed, best_acc, q_trajectory, eval_loss_trajectory, dynamics_checkpoints, mean_tmin_acc, phase_transition_step


def run_one_config(seq_cfg, variant, seed, device, completed_keys):
    seq_len = seq_cfg["seq_len"]
    key = f"L{seq_len}_{variant}_s{seed}"

    if key in completed_keys:
        return None

    print(f"\n    L={seq_len}, seed={seed}, variant={variant}", flush=True)

    full_seed(seed)
    model = make_model(device)

    train_time, best_acc, q_traj, loss_traj, dyn_ckpts, mean_tmin_acc, phase_step = train_model(
        model, device, seq_len, variant, seed
    )

    print(f"      Measuring final contraction...", flush=True)
    contraction = measure_contraction_summary(model, device, seq_len)

    print(f"      Measuring final spectral radius...", flush=True)
    spectral = measure_spectral_radius(model, device, seq_len)

    seq_q_final, token_q_final, per_pos_final = measure_q_at_t_min(model, device, seq_len)

    final_eval_loss, final_seq_acc, final_tok_acc, final_per_pos = measure_eval_loss(model, device, seq_len)

    print(f"      rho={spectral['mean']:.4f}+/-{spectral['std']:.4f}, "
          f"k={contraction['mean_k']:.4f}, T_99={contraction['T_99']}, "
          f"finalEval={final_seq_acc:.4f}, cumQ={mean_tmin_acc:.4f}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "key": key,
        "seq_len": seq_len,
        "variant": variant,
        "seed": seed,
        "params": n_params,
        "accuracy": round(best_acc, 4),
        "final_eval_loss": round(final_eval_loss, 4),
        "final_eval_seq_acc": round(final_seq_acc, 4),
        "final_eval_token_acc": round(final_tok_acc, 4),
        "final_eval_per_pos_acc": [round(p, 4) for p in final_per_pos],
        "q_token_tmin": round(token_q_final, 4),
        "q_per_pos_tmin": [round(p, 4) for p in per_pos_final],
        "spectral_radius": spectral,
        "contraction": contraction,
        "train_time_s": round(train_time, 1),
        "q_trajectory": q_traj,
        "eval_loss_trajectory": loss_traj,
        "dynamics_checkpoints": dyn_ckpts,
        "mean_tmin_acc": round(mean_tmin_acc, 4),
        "phase_transition_step": phase_step,
        "run_signature": {
            "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "train_T": TRAIN_T,
            "vocab_size": VOCAB_SIZE,
            "variable_t_range": VARIABLE_T_RANGE,
        },
    }


def compute_analysis(results):
    print("\n" + "=" * 60, flush=True)
    print("  D35b ANALYSIS: LEARNABLE PREFIX SUM (V=8)", flush=True)
    print("=" * 60, flush=True)

    by_len_seed = {}
    for run in results["runs"]:
        sl = run["seq_len"]
        s = run["seed"]
        v = run["variant"]
        key = (sl, s)
        if key not in by_len_seed:
            by_len_seed[key] = {}
        by_len_seed[key][v] = run

    print(f"\n  {'L':>3s} | {'k_FT':>8s} | {'k_VT':>8s} | {'dk':>8s} | "
          f"{'rho_FT':>8s} | {'rho_VT':>8s} | {'drho':>8s} | "
          f"{'T99_FT':>6s} | {'T99_VT':>6s} | {'n':>2s}", flush=True)
    print("  " + "-" * 85, flush=True)

    all_ft_k = []
    all_vt_k = []
    all_ft_rho = []
    all_vt_rho = []

    for sl in sorted(set(k[0] for k in by_len_seed)):
        ft_ks, vt_ks = [], []
        ft_rhos, vt_rhos = [], []
        ft_t99s, vt_t99s = [], []

        for seed in SEEDS:
            pair = by_len_seed.get((sl, seed), {})
            if "fixed_t" in pair and "variable_t" in pair:
                ft = pair["fixed_t"]
                vt = pair["variable_t"]
                ft_k = ft["contraction"]["mean_k"]
                vt_k = vt["contraction"]["mean_k"]
                ft_rho = ft["spectral_radius"]["mean"]
                vt_rho = vt["spectral_radius"]["mean"]
                ft_ks.append(ft_k)
                vt_ks.append(vt_k)
                ft_rhos.append(ft_rho)
                vt_rhos.append(vt_rho)
                ft_t99s.append(ft["contraction"]["T_99"])
                vt_t99s.append(vt["contraction"]["T_99"])
                all_ft_k.append(ft_k)
                all_vt_k.append(vt_k)
                all_ft_rho.append(ft_rho)
                all_vt_rho.append(vt_rho)

        if len(ft_ks) >= 2:
            dk = np.mean(vt_ks) - np.mean(ft_ks)
            drho = np.mean(vt_rhos) - np.mean(ft_rhos)
            t99_ft = [x for x in ft_t99s if x is not None]
            t99_vt = [x for x in vt_t99s if x is not None]
            print(f"  {sl:>3d} | {np.mean(ft_ks):>8.4f} | {np.mean(vt_ks):>8.4f} | {dk:>+8.4f} | "
                  f"{np.mean(ft_rhos):>8.4f} | {np.mean(vt_rhos):>8.4f} | {drho:>+8.4f} | "
                  f"{np.mean(t99_ft) if t99_ft else 'None':>6} | {np.mean(t99_vt) if t99_vt else 'None':>6} | "
                  f"{len(ft_ks):>2d}", flush=True)

    if len(all_ft_k) >= 4:
        t_k, p_k = scipy_stats.ttest_rel(all_vt_k, all_ft_k)
        t_rho, p_rho = scipy_stats.ttest_rel(all_vt_rho, all_ft_rho)
        dk_vals = np.array(all_vt_k) - np.array(all_ft_k)
        unanimous = all(dk_vals < 0)
        print(f"\n  OVERALL k: dk={dk_vals.mean():+.4f}, t={t_k:.2f}, p={p_k:.6f}, "
              f"unanimous={unanimous} ({sum(dk_vals<0)}/{len(dk_vals)})", flush=True)
        print(f"  OVERALL rho: drho={np.mean(np.array(all_vt_rho)-np.array(all_ft_rho)):+.4f}, "
              f"t={t_rho:.2f}, p={p_rho:.6f}", flush=True)

    # Matched-loss analysis
    print("\n  --- MATCHED-LOSS ANALYSIS ---", flush=True)
    for sl in sorted(set(k[0] for k in by_len_seed)):
        for seed in SEEDS:
            pair = by_len_seed.get((sl, seed), {})
            if "fixed_t" not in pair or "variable_t" not in pair:
                continue
            ft_ckpts = pair["fixed_t"].get("dynamics_checkpoints", [])
            vt_ckpts = pair["variable_t"].get("dynamics_checkpoints", [])
            if not ft_ckpts or not vt_ckpts:
                continue
            print(f"\n  L={sl}, seed={seed}:", flush=True)
            for vc in vt_ckpts:
                best_match = min(ft_ckpts, key=lambda fc: abs(fc["eval_loss"] - vc["eval_loss"]))
                dk = vc["k"] - best_match["k"]
                print(f"    VT step {vc['step']}: eval_loss={vc['eval_loss']:.4f}, k={vc['k']:.4f} | "
                      f"FT matched step {best_match['step']}: eval_loss={best_match['eval_loss']:.4f}, "
                      f"k={best_match['k']:.4f} | dk={dk:+.4f}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D35b: Learnable Prefix Sum (V={VOCAB_SIZE})", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Configs: {len(SEQ_LEN_CONFIGS)} depths x {len(SEEDS)} seeds x {len(VARIANTS)} variants "
          f"= {len(SEQ_LEN_CONFIGS)*len(SEEDS)*len(VARIANTS)} runs", flush=True)

    existing = load_checkpoint()
    if existing:
        results = existing
        completed_keys = {r["key"] for r in results["runs"]}
        print(f"Resuming from {len(completed_keys)} completed runs", flush=True)
    else:
        results = {
            "experiment": "D35b_learnable_prefix_sum",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": "prefix_sum",
            "vocab_size": VOCAB_SIZE,
            "config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS,
                "train_T": TRAIN_T,
                "variable_T_range": VARIABLE_T_RANGE,
                "training_steps": TRAINING_STEPS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
            },
            "runs": [],
        }
        completed_keys = set()

    total = len(SEQ_LEN_CONFIGS) * len(SEEDS) * len(VARIANTS)

    for seq_cfg in SEQ_LEN_CONFIGS:
        sl = seq_cfg["seq_len"]
        print(f"\n{'='*60}", flush=True)
        print(f"D={sl // 2} (L={sl})", flush=True)
        print(f"{'='*60}", flush=True)

        for seed in SEEDS:
            for variant in VARIANTS:
                result = run_one_config(seq_cfg, variant, seed, device, completed_keys)
                if result is not None:
                    results["runs"].append(result)
                    completed_keys.add(result["key"])
                    save_results(results)
                    print(f"\n  [{len(results['runs'])}/{total}] saved\n", flush=True)

    compute_analysis(results)
    save_results(results)
    print("\nD35b COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
