"""
D23 Multi-Seed Extension: Variable-T at L=20,24 with 3 seeds + rho measurement.

Addresses Codex combined review priority:
- T_99=3 universality at L=20,24
- rho scaling with complexity (within same experiment)
- Seed variance at high L

Runs variable_t only, 3 seeds x 2 L values = 6 configs.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params
from shared.data import generate_batch

TRAINING_STEPS = 30000
BATCH_SIZE = 256
VOCAB_SIZE = 64
LR = 3e-4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENC_LAYERS = 2
MAX_LEN = 64

SEQ_LENS = [20, 24]
SEEDS = [42, 1337, 2024]
EVAL_T_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 32, 48]
EVAL_SAMPLES = 4096
FP_T = 100

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "exp_d23_multiseed.json"


def full_seed(seed):
    set_seed(seed)
    random.seed(seed)


def make_model(device):
    return UESDModel(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_ENC_LAYERS, MAX_LEN).to(device)


def save_checkpoint(results):
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


def load_checkpoint():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def get_t_range(seq_len):
    base = [4, 6, 8, 10, 12, 14, 16]
    carry_depth = seq_len // 2
    if carry_depth > 8:
        extra = list(range(18, min(carry_depth * 3, 32) + 1, 2))
        return base + extra
    return base


def train_variable_t(model, device, seq_len, seed):
    full_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    half = seq_len // 2
    t_range = get_t_range(seq_len)
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step in range(1, TRAINING_STEPS + 1):
        T = random.choice(t_range)
        src, tgt = generate_batch("addition", BATCH_SIZE, seq_len, VOCAB_SIZE)
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, T)
        loss = F.cross_entropy(
            logits[:, :half].reshape(-1, logits.size(-1)),
            tgt[:, :half].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                preds = logits[:, :half].argmax(dim=-1)
                seq_acc = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
                best_acc = max(best_acc, seq_acc)
            print(f"    Step {step:>6d}/{TRAINING_STEPS} | T={T:>2d} | Loss: {loss.item():.4f} | "
                  f"Seq Acc: {seq_acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s | Best acc: {best_acc:.4f}", flush=True)
    return elapsed, best_acc


def evaluate_step_ablation(model, device, seq_len):
    model.eval()
    half = seq_len // 2
    full_seed(9999)
    src, tgt = generate_batch("addition", EVAL_SAMPLES, seq_len, VOCAB_SIZE)
    src, tgt = src.to(device), tgt.to(device)

    results = {}
    with torch.no_grad():
        context = model.encode(src)
        for T in EVAL_T_VALUES:
            s = model.init_state(src.size(0), src.size(1), device)
            for t in range(T):
                s, _ = model.dynamics_step(s, context)
            logits = model.readout(s)
            preds = logits[:, :half].argmax(dim=-1)
            tok_correct = (preds == tgt[:, :half]).float().mean().item()
            seq_correct = (preds == tgt[:, :half]).all(dim=1).float().mean().item()
            results[f"T={T}"] = {
                "tok_acc": round(tok_correct, 4),
                "seq_acc": round(seq_correct, 4),
            }
    return results


def measure_spectral_radius(model, device, seq_len, n_vecs=10, n_iter=50):
    model.eval()
    full_seed(9999)
    src, _ = generate_batch("addition", 256, seq_len, VOCAB_SIZE)
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
    print(f"    Spectral radius: rho={mean_rho:.4f}+/-{std_rho:.4f}", flush=True)
    return {"mean": round(mean_rho, 4), "std": round(std_rho, 4)}


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("runs"):
        all_results = checkpoint
        completed_keys = {(r["seq_len"], r["seed"]) for r in all_results["runs"]}
        print(f"\n  RESUMING: {len(all_results['runs'])} runs already done", flush=True)
    else:
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "purpose": "D23 multi-seed: T_99 + rho at L=20,24 with 3 seeds (Codex priority)",
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
                "n_enc_layers": N_ENC_LAYERS, "vocab_size": VOCAB_SIZE,
                "training_steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                "lr": LR, "seq_lens": SEQ_LENS, "seeds": SEEDS,
                "variant": "variable_t",
            },
            "runs": [],
        }
        completed_keys = set()

    for seq_len in SEQ_LENS:
        for seed in SEEDS:
            if (seq_len, seed) in completed_keys:
                print(f"\n  L={seq_len}, seed={seed}... SKIPPED (already done)", flush=True)
                continue

            print(f"\n{'=' * 60}", flush=True)
            print(f"  L={seq_len} (D={seq_len//2}), seed={seed}, variant=variable_t", flush=True)
            print(f"{'=' * 60}", flush=True)

            full_seed(seed)
            model = make_model(device)
            params = count_params(model)
            print(f"    Params: {params}", flush=True)

            train_time, best_acc = train_variable_t(model, device, seq_len, seed)

            print("    Evaluating step ablation...", flush=True)
            step_abl = evaluate_step_ablation(model, device, seq_len)
            for k, v in step_abl.items():
                print(f"      {k}: seq={v['seq_acc']:.4f}", flush=True)

            print("    Measuring spectral radius...", flush=True)
            spectral = measure_spectral_radius(model, device, seq_len)

            del model
            torch.cuda.empty_cache()

            run_result = {
                "seq_len": seq_len,
                "carry_depth": seq_len // 2,
                "variant": "variable_t",
                "seed": seed,
                "params": params,
                "train_time_s": round(train_time, 1),
                "best_train_acc": round(best_acc, 4),
                "step_ablation": step_abl,
                "spectral_radius": spectral,
            }

            all_results["runs"].append(run_result)
            save_checkpoint(all_results)
            print(f"\n  Saved ({len(all_results['runs'])} runs total)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  ALL DONE — {len(all_results['runs'])} runs", flush=True)
    print(f"{'=' * 60}", flush=True)

    for r in all_results["runs"]:
        t99 = "?"
        for t_key in ["T=2", "T=3", "T=5", "T=8"]:
            if r["step_ablation"].get(t_key, {}).get("seq_acc", 0) >= 0.99:
                t99 = t_key.replace("T=", "")
                break
        rho = r.get("spectral_radius", {}).get("mean", "?")
        print(f"  L={r['seq_len']} seed={r['seed']}: T_99={t99}, rho={rho}, acc={r['best_train_acc']}")


if __name__ == "__main__":
    run()
