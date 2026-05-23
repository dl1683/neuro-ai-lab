"""
Experiment D11: Energy Landscape Cartography

The UESD framework claims that "thinking" is trajectory through high-energy
regions and "generation" is convergence into basins. This experiment maps
the actual energy landscape E(s) = ||F_theta(s,c)||^2 to test that claim.

Three phases:
1. BASIN STRUCTURE: For trained models (both E5 and CE-dynamics), sample
   many random initial states and trace where trajectories converge. Do
   distinct inputs map to distinct basins? Are basins geometrically clean?

2. LANDSCAPE SLICING: Take 2D slices through state space (PCA of trajectory
   endpoints) and evaluate the energy function on a grid. Visualize the
   energy landscape and overlay actual trajectories.

3. WRONG ATTRACTOR GEOMETRY: For E5, compare the state-space geometry of
   successful vs failed seeds. Do failed seeds converge to a geometrically
   distinct basin? Can we identify the "wrong attractor" basin?

Key measurements:
- Number of distinct basins (clustered by final-state cosine similarity)
- Basin radius (max perturbation that still converges to same answer)
- Energy barrier height between basins
- Trajectory path length ratio (actual / geodesic)
- Wrong-attractor basin overlap with correct basin

PREDICTIONS:
1. CE-dynamics has FEWER, LARGER basins than E5 (scattered dynamics =
   wider attraction, rotation-based stability = less structured landscape)
2. E5 has SHARPER basins with HIGHER barriers (compression dynamics =
   tight convergence, but vulnerable to wrong-attractor trapping)
3. Wrong attractor basin for failed E5 seeds is geometrically CLOSE to
   correct basin (small energy barrier) - explaining stochastic failure
4. Trajectory path length ratio > 2x for CE-dynamics (circuitous paths
   due to rotation), ~1.2-1.5x for E5 (more direct convergence)

If prediction 3 holds, it suggests wrong-attractor failure is a near-miss
phenomenon, not a fundamentally different computation path - and could
potentially be fixed by modest noise injection or basin-aware initialization.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.model import UESDModel
from shared.training import set_seed, count_params, train
from shared.data import generate_batch


SEEDS = [42, 512]  # 42 = reliable success, 512 = stochastic failure in D2b
N_BASIN_PROBES = 512
N_GRID_POINTS = 50
PERTURBATION_SCALES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


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
        "seed": seed,
    }


@torch.no_grad()
def compute_energy(model, s, context):
    """E(s,c) = ||F_theta(s,c)||^2 where F is the residual update."""
    s_new = model.dynamics(s, context)
    residual = s_new - s
    return (residual ** 2).sum(dim=-1).mean(dim=-1)  # [B]


@torch.no_grad()
def trace_trajectory(model, src, T, return_energies=True):
    """Unroll dynamics and return all states + energies."""
    context = model.encode(src)
    B, L = src.shape
    s = model.init_state(B, L, src.device)

    states = [s.clone()]
    energies = []

    for t in range(T):
        if return_energies:
            e = compute_energy(model, s, context)
            energies.append(e)
        s, _ = model.dynamics_step(s, context)
        states.append(s.clone())

    if return_energies:
        e_final = compute_energy(model, s, context)
        energies.append(e_final)

    return states, energies, context


@torch.no_grad()
def phase1_basin_structure(model, eval_src, eval_tgt, config, device):
    """Map basin structure: how many distinct attractors exist?"""
    T = config["T"]
    V = config["vocab_size"]
    half = config["seq_len"] // 2

    # Get final states and predictions for all eval examples
    states, energies, context = trace_trajectory(model, eval_src, T)
    s_final = states[-1]

    # Predictions
    logits = model.readout_logits(s_final)
    preds = logits[:, :half, :].argmax(dim=-1)
    targets = eval_tgt[:, :half].to(device)
    correct = (preds == targets).all(dim=1)

    # Flatten final states for clustering
    B = s_final.shape[0]
    s_flat = s_final.reshape(B, -1)  # [B, L*d]

    # Cosine similarity matrix (sample to keep tractable)
    n_sample = min(B, N_BASIN_PROBES)
    idx = torch.randperm(B)[:n_sample]
    s_sample = F.normalize(s_flat[idx], dim=-1)
    sim_matrix = torch.mm(s_sample, s_sample.t())

    # Cluster by thresholding cosine similarity — sweep thresholds
    THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
    threshold_results = {}
    basins_default = None
    for threshold in THRESHOLDS:
        visited = set()
        basins = []
        for i in range(n_sample):
            if i in visited:
                continue
            cluster = (sim_matrix[i] > threshold).nonzero(as_tuple=True)[0].tolist()
            visited.update(cluster)
            basins.append(cluster)
        threshold_results[f"thresh_{threshold}"] = {
            "n_basins": len(basins),
            "basin_sizes": sorted([len(b) for b in basins], reverse=True)[:20],
        }
        if threshold == 0.95:
            basins_default = basins

    basins = basins_default

    # Per-basin accuracy
    basin_stats = []
    for i, basin in enumerate(basins[:20]):
        basin_idx = idx[basin]
        basin_correct = correct[basin_idx].float().mean().item()
        basin_energy = energies[-1][basin_idx].mean().item()
        basin_stats.append({
            "basin_id": i,
            "size": len(basin),
            "accuracy": basin_correct,
            "mean_energy": basin_energy,
        })

    # Energy statistics
    energy_correct = energies[-1][correct].mean().item() if correct.any() else float('nan')
    energy_wrong = energies[-1][~correct].mean().item() if (~correct).any() else float('nan')

    # Trajectory energy profile (mean over examples)
    energy_profile = [e.mean().item() for e in energies]

    return {
        "n_basins_found": len(basins),
        "basin_sizes": [len(b) for b in basins],
        "top_basin_stats": basin_stats,
        "threshold_sensitivity": threshold_results,
        "energy_correct_mean": energy_correct,
        "energy_wrong_mean": energy_wrong,
        "energy_profile": energy_profile,
        "seq_accuracy": correct.float().mean().item(),
        "sim_matrix_mean": sim_matrix.mean().item(),
        "sim_matrix_std": sim_matrix.std().item(),
    }


@torch.no_grad()
def phase2_basin_radius(model, eval_src, eval_tgt, config, device):
    """Test basin radius: how much perturbation before answer changes?"""
    T = config["T"]
    half = config["seq_len"] // 2

    # Get baseline predictions
    states_base, _, context = trace_trajectory(model, eval_src, T, return_energies=False)
    s_final_base = states_base[-1]
    logits_base = model.readout_logits(s_final_base)
    preds_base = logits_base[:, :half, :].argmax(dim=-1)

    B = eval_src.shape[0]
    results = {}

    for scale in PERTURBATION_SCALES:
        # Perturb initial state
        s0 = model.init_state(B, eval_src.shape[1], device)
        noise = torch.randn_like(s0) * scale
        s_perturbed = s0 + noise

        # Run dynamics from perturbed initial state
        s = s_perturbed
        for t in range(T):
            s, _ = model.dynamics_step(s, context)

        logits_pert = model.readout_logits(s)
        preds_pert = logits_pert[:, :half, :].argmax(dim=-1)

        # Measure agreement with baseline
        tok_agreement = (preds_pert == preds_base).float().mean().item()
        seq_agreement = (preds_pert == preds_base).all(dim=1).float().mean().item()

        # State distance from baseline final state
        state_dist = (s - s_final_base).norm(dim=-1).mean().item()

        results[f"scale_{scale}"] = {
            "perturbation_scale": scale,
            "tok_agreement": tok_agreement,
            "seq_agreement": seq_agreement,
            "final_state_distance": state_dist,
        }

    return results


@torch.no_grad()
def phase3_landscape_slice(model, eval_src, config, device):
    """2D energy landscape slice for a FIXED context using perturbed initial states.

    Uses one input's context throughout: perturbs initial decoder state N times,
    runs dynamics, PCA's the final states, then evaluates energy grid under
    the same context. This gives a true context-conditional landscape slice.
    """
    T = config["T"]
    N_PERTURBATIONS = 256

    # Use a single input's context
    src_single = eval_src[:1]
    context = model.encode(src_single)  # [1, L, d]
    L_out = src_single.shape[1]
    d_model = config["d_model"]

    # Generate perturbed trajectories from same context
    s0_base = model.init_state(1, L_out, device)
    final_states = []
    all_traj_states = []

    for _ in range(N_PERTURBATIONS):
        noise = torch.randn_like(s0_base) * 0.5
        s = s0_base + noise
        traj = [s.clone()]
        ctx = context.expand(1, -1, -1)
        for t in range(T):
            s, _ = model.dynamics_step(s, ctx)
            traj.append(s.clone())
        final_states.append(s.squeeze(0))
        all_traj_states.append(traj)

    s_final = torch.stack(final_states, dim=0)  # [N, L, d]
    B = N_PERTURBATIONS

    # PCA of final states (all from same context)
    s_flat = s_final.reshape(B, -1).cpu().numpy()
    mean = s_flat.mean(axis=0)
    centered = s_flat - mean

    U, S_vals, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    total_var = (S_vals ** 2).sum()
    var_explained = (S_vals[:10] ** 2) / total_var

    coords1 = np.linspace(-3, 3, N_GRID_POINTS)
    coords2 = np.linspace(-3, 3, N_GRID_POINTS)

    scale1 = S_vals[0] / np.sqrt(B)
    scale2 = S_vals[1] / np.sqrt(B)

    energy_grid = np.zeros((N_GRID_POINTS, N_GRID_POINTS))

    for i, c1 in enumerate(coords1):
        points = []
        for j, c2 in enumerate(coords2):
            point = mean + c1 * scale1 * pc1 + c2 * scale2 * pc2
            points.append(point)

        points_tensor = torch.tensor(
            np.array(points), dtype=torch.float32, device=device
        ).reshape(N_GRID_POINTS, L_out, d_model)

        ctx_expanded = context.expand(N_GRID_POINTS, -1, -1)
        e = compute_energy(model, points_tensor, ctx_expanded)
        energy_grid[i] = e.cpu().numpy()

    # Project perturbed trajectories onto PCA (use first 8 trajectories)
    trajectory_projections = []
    for t_idx in range(T + 1):
        projs_1, projs_2 = [], []
        for traj in all_traj_states[:8]:
            s_t = traj[t_idx].squeeze(0).reshape(-1).cpu().numpy()
            projs_1.append(((s_t - mean) @ pc1) / scale1)
            projs_2.append(((s_t - mean) @ pc2) / scale2)
        trajectory_projections.append({
            "step": t_idx,
            "pc1_mean": float(np.mean(projs_1)),
            "pc1_std": float(np.std(projs_1)),
            "pc2_mean": float(np.mean(projs_2)),
            "pc2_std": float(np.std(projs_2)),
        })

    return {
        "variance_explained_top10": var_explained.tolist(),
        "energy_grid_min": float(energy_grid.min()),
        "energy_grid_max": float(energy_grid.max()),
        "energy_grid_mean": float(energy_grid.mean()),
        "energy_grid_shape": list(energy_grid.shape),
        "trajectory_projections": trajectory_projections,
        "grid_range": {"c1": [-3, 3], "c2": [-3, 3]},
        "scale1": float(scale1),
        "scale2": float(scale2),
        "n_perturbations": N_PERTURBATIONS,
    }


@torch.no_grad()
def phase4_path_efficiency(model, eval_src, config, device):
    """Measure trajectory path efficiency: actual / geodesic distance."""
    T = config["T"]
    B = min(eval_src.shape[0], 512)
    src = eval_src[:B]

    states, _, _ = trace_trajectory(model, src, T, return_energies=False)

    # Path length: sum of step-to-step distances
    path_lengths = torch.zeros(B, device=device)
    for t in range(T):
        step_dist = (states[t + 1] - states[t]).norm(dim=-1).mean(dim=-1)
        path_lengths += step_dist

    # Geodesic: direct distance from s_0 to s_T
    geodesic = (states[-1] - states[0]).norm(dim=-1).mean(dim=-1)

    # Path efficiency ratio
    ratio = path_lengths / geodesic.clamp(min=1e-8)

    # Per-step displacement magnitudes
    step_displacements = []
    for t in range(T):
        d = (states[t + 1] - states[t]).norm(dim=-1).mean().item()
        step_displacements.append(d)

    return {
        "path_length_mean": path_lengths.mean().item(),
        "path_length_std": path_lengths.std().item(),
        "geodesic_mean": geodesic.mean().item(),
        "geodesic_std": geodesic.std().item(),
        "efficiency_ratio_mean": ratio.mean().item(),
        "efficiency_ratio_std": ratio.std().item(),
        "step_displacements": step_displacements,
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    config = build_config()
    V = config["vocab_size"]
    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "purpose": "Energy landscape cartography: basin structure, radius, slicing, path efficiency",
        "config": config,
    }

    # Generate eval data (shared across all runs)
    set_seed(9999)
    eval_src, eval_tgt = generate_batch("addition", 1024, config["seq_len"], V)
    eval_src = eval_src.to(device)
    eval_tgt = eval_tgt.to(device)

    # === Control: Untrained model ===
    print(f"\n{'=' * 60}", flush=True)
    print(f"  CONTROL: Untrained model", flush=True)
    print(f"{'=' * 60}", flush=True)
    set_seed(42)
    ctrl_model = UESDModel(
        V, config["d_model"], config["n_heads"],
        config["d_ff"], config["n_enc_layers"], config["max_len"],
    ).to(device)
    ctrl_model.eval()
    ctrl_basin = phase1_basin_structure(ctrl_model, eval_src, eval_tgt, config, device)
    ctrl_path = phase4_path_efficiency(ctrl_model, eval_src, config, device)
    print(f"  Basins: {ctrl_basin['n_basins_found']}, "
          f"Energy correct: {ctrl_basin['energy_correct_mean']:.4f}", flush=True)
    print(f"  Path ratio: {ctrl_path['efficiency_ratio_mean']:.3f}", flush=True)
    all_results["control_untrained"] = {
        "basin_structure": ctrl_basin,
        "path_efficiency": ctrl_path,
    }
    del ctrl_model
    torch.cuda.empty_cache()

    for track in ["e5", "dynamics_ce"]:
        for seed in SEEDS:
            label = f"{track}_seed{seed}"
            print(f"\n{'=' * 60}", flush=True)
            print(f"  {label}", flush=True)
            print(f"{'=' * 60}", flush=True)

            cfg = build_config(seed)
            set_seed(seed)
            model = UESDModel(
                V, cfg["d_model"], cfg["n_heads"],
                cfg["d_ff"], cfg["n_enc_layers"], cfg["max_len"],
            ).to(device)
            print(f"  params: {count_params(model)}", flush=True)

            t0 = time.time()
            tr = train(model, "addition", track, cfg, device)
            train_time = time.time() - t0
            final_loss = tr["history"][-1]["loss"]
            print(f"  Final loss: {final_loss:.4f}", flush=True)

            model.eval()

            # Phase 1: Basin structure
            print(f"\n  Phase 1: Basin structure...", flush=True)
            basin_results = phase1_basin_structure(model, eval_src, eval_tgt, cfg, device)
            print(f"    Basins found: {basin_results['n_basins_found']}", flush=True)
            print(f"    Seq accuracy: {basin_results['seq_accuracy']:.4f}", flush=True)
            print(f"    Energy (correct): {basin_results['energy_correct_mean']:.4f}", flush=True)
            print(f"    Energy (wrong): {basin_results['energy_wrong_mean']:.4f}", flush=True)

            # Phase 2: Basin radius
            print(f"\n  Phase 2: Basin radius...", flush=True)
            radius_results = phase2_basin_radius(model, eval_src, eval_tgt, cfg, device)
            for k, v in radius_results.items():
                print(f"    {k}: seq_agree={v['seq_agreement']:.3f} "
                      f"state_dist={v['final_state_distance']:.3f}", flush=True)

            # Phase 3: Landscape slice
            print(f"\n  Phase 3: Landscape slice (PCA)...", flush=True)
            slice_results = phase3_landscape_slice(model, eval_src, cfg, device)
            print(f"    Variance explained (PC1,2): "
                  f"{slice_results['variance_explained_top10'][0]:.3f}, "
                  f"{slice_results['variance_explained_top10'][1]:.3f}", flush=True)
            print(f"    Energy grid: min={slice_results['energy_grid_min']:.4f} "
                  f"max={slice_results['energy_grid_max']:.4f}", flush=True)

            # Phase 4: Path efficiency
            print(f"\n  Phase 4: Path efficiency...", flush=True)
            path_results = phase4_path_efficiency(model, eval_src, cfg, device)
            print(f"    Path/geodesic ratio: {path_results['efficiency_ratio_mean']:.3f} "
                  f"+/- {path_results['efficiency_ratio_std']:.3f}", flush=True)

            all_results[label] = {
                "train_time_s": train_time,
                "final_loss": final_loss,
                "basin_structure": basin_results,
                "basin_radius": radius_results,
                "landscape_slice": slice_results,
                "path_efficiency": path_results,
            }

            del model
            torch.cuda.empty_cache()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_d11_energy_landscape.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run()
