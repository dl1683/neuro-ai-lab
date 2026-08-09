**BUGS FOUND**

None blocking.

Key checks passed:
- RNG isolation: D34 saves/restores Torch CPU/CUDA RNG around rho and accuracy measurement at [exp_d34_rho_trajectory.py:90](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:90) and [exp_d34_rho_trajectory.py:132](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:132). `generate_batch` uses Torch RNG only, so missing Python/NumPy state restore is not a current issue.
- Task schedule: pre-drawn from a dedicated `random.Random(seed + 999)` and randomly interleaves add/sub at [exp_d34_rho_trajectory.py:156](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:156). This matches D32’s random 50/50 interleaving pattern, though not the same exact seed stream.
- Spectral radius: despite the outer `@torch.no_grad()`, the inner `torch.enable_grad()` at [exp_d34_rho_trajectory.py:109](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:109) correctly enables autograd for the fixed-point Jacobian probe.
- Loss: one task per step, interleaved by schedule, same target slice/cross-entropy pattern as D32.
- Atomic result write: `tmp.replace(RESULTS_PATH)` at [exp_d34_rho_trajectory.py:76](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:76) is the right same-directory atomic write pattern.

**WARNINGS**

- Checkpoints are only in memory until each 60K-step variant finishes. `save_results()` is called after `train_with_checkpoints()` returns at [exp_d34_rho_trajectory.py:224](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d34_rho_trajectory.py:224), not after each 5K rho checkpoint. A crash at step 55K loses that run’s trajectory.
- No resume support. Unlike D32/D33, D34 does not load existing results and skip completed runs. Restarting overwrites/restarts the experiment.
- Rho is measured only on addition inputs. D32 measured rho per task for addition and subtraction. For a multi-task trajectory, addition-only rho is acceptable for a fixed probe, but it can miss task-conditioned subtraction dynamics.
- Only one seed. This is fine for “trajectory case study,” not enough for a robust claim about phase-transition timing.

**PERFORMANCE**

- Expected training time: about 6000s per run from the D32 reference, so roughly 3.3 hours total for 2 runs if D32’s D=6 20K timing was ~2000s.
- Memory: same `UESDModel` architecture and D=6/seq_len=12/batch=256 as D32 baseline, so should be safe on a 32GB RTX 5090.
- Rho overhead: every 5K over 60K gives 12 rho measurements per run, 24 across both variants. Each rho measurement does 100 no-grad fixed-point steps plus 10 x 50 gradient Jacobian-vector iterations on batch 32. Expect low single-digit percent overhead relative to 60K training steps, plus small accuracy-eval overhead.

**COMPARISON WITH D32/D33**

- Same model architecture and core hyperparameters as D32/D33: `D_MODEL=128`, `N_HEADS=4`, `D_FF=512`, `N_ENC_LAYERS=2`, `TRAIN_T=10`, same VT range.
- Same spectral-radius method as D32/D33: fixed-point after `FP_T=100`, then power iteration with `torch.autograd.grad`.
- RNG isolation is better than D32/D33 spectral measurement. D32/D33 call `full_seed(9999)` inside final rho measurement; D34 restores Torch RNG because measurement happens during training.

**SCIENTIFIC VALIDITY**

D34 tests the stated mechanism well as a two-run longitudinal probe: it asks whether VT rho stays capped while training progresses, and whether FT rho moves toward it.

Limits:
- 60K steps plausibly allows a phase transition if D32’s 20K was undertrained, but it does not guarantee one.
- 12 checkpoints per run is enough to see coarse monotonic/non-monotonic trend and a broad transition window, not precise transition timing.
- Main confounders are one seed, addition-only rho probe, and no per-task rho split.

**VERDICT: CONDITIONAL PASS**

Run it if you accept this as a descriptive D34 trajectory probe. Before an expensive run, I would strongly consider saving after each 5K checkpoint and adding resume/skip logic. Syntax check passed with `python -m py_compile` for D34/D32/D33.

