**Verdict**

Do not run these as-is. D11 and D12 have hard runtime bugs, and all three currently overclaim relative to what their measurements can identify. The ideas are strong, but the designs need tighter controls before they can bear theoretical weight.

**Cross-Cutting Blockers**

1. D11 and D12 will crash after training. Both call `tr["ce_history"]`, but `shared.training.train()` returns `{"history": ..., "elapsed_s": ...}`. See [training.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py:173), versus [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:390) and [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:316).

2. D11/D12 analyze models in train mode after `train()`. `TransformerDecoderLayer` has dropout by default, so trajectories, energies, deterministic baselines, and Langevin noise are contaminated by dropout unless `model.eval()` is called before analysis.

3. Losses train on all `seq_len` positions, but most evaluations only score the first half. For addition-style tasks, the second half is padded zeros, so half the CE is an easy padding objective. This matters badly for D13.

---

**D11: Energy Landscape Cartography**

Correctness:
- `compute_energy()` is mechanically consistent with the code’s residual field: `model.dynamics(s, context) - s`, then squared norm averaged over positions [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:83). It is not numerically unstable, but dropout must be disabled.
- `trace_trajectory()` is fine structurally, but again stochastic in train mode.
- `phase1_basin_structure()` does not sample many random initial states as promised. It uses one deterministic `pos_dec` initialization per input and clusters final states across different contexts [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:122). That measures answer/input-state geometry, not basins for a fixed landscape.
- Basin clustering is order-dependent and threshold-artifact prone. It uses leader-style cosine thresholding, not connected components or clustering stability [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:142). `basins[:20]` are first-found, not largest.
- `phase2_basin_radius()` measures agreement with baseline predictions, not correctness, and perturbs only the initial decoder state, not basin radius around an attractor [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:192). `eval_tgt` is unused.
- `phase3_landscape_slice()` computes PCA over final states from many contexts, then evaluates the grid under a single context [D11](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d11_energy_landscape.py:267). That mixes cross-example variation with one-example energy geometry.
- It does not actually estimate energy barriers, despite claiming that measurement.
- `phase4_path_efficiency()` is usable as a descriptive metric, but the norm averages over positions rather than full-state distance, so ratios are not literal geodesic ratios.

Statistical plan:
- `1024` eval examples and `512` basin probes are enough for coarse descriptive plots, not enough for basin-count claims under arbitrary `0.95` threshold.
- Need threshold sweep, bootstrap CIs over examples, and 3-5 training seeds per track.
- For basin claims, hold context fixed, sample many initial perturbations, and compute connected components or HDBSCAN-like cluster stability.
- Compare correct vs wrong states with permutation tests and bootstrap CIs on energy, distance, and margin.

Alternative explanations:
- Basin clusters can be answer-token clusters, not attractors.
- PCA slices can mostly reflect context/target identity.
- E5 lower energy can simply be the self-consistency loss, not deeper basin structure.
- Path inefficiency can be decoder-layer geometry/normalization, not “thinking through high-energy regions.”

Missing controls:
- Random untrained model.
- Encoder-only or `T=1` baseline.
- Same checkpoints as D7/D8/D10.
- Fixed-context basin maps.
- Threshold sensitivity and random projection controls.
- Explicit D11-to-D12 barrier estimates.

Priority fix:
- Redesign D11 as context-conditional basin analysis: `model.eval()`, fixed examples, many initial perturbations per example, threshold-sensitivity/connected-component clustering, and correct the `ce_history` crash.

Prediction:
- As written, it crashes. After minimal fixes, I expect E5 to show lower final energy than CE-dynamics mostly because E5 directly optimizes residual self-consistency. Confidence: high.
- Basin counts will be highly threshold-sensitive. Confidence: high.
- PCA grids will be hard to interpret as landscapes because PCs are across contexts. Confidence: high.

Parsimony:
- Drop PCA grid/barrier claims for first run. Keep final energy, perturbation stability, and path length on fixed contexts.

---

**D12: Langevin Escape**

Correctness:
- Schedule functions are simple and fine, though linear/cosine never reach exactly zero at the last noisy candidate step; the code skips noise at `t == T-1` [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:149).
- `compute_carries()` is basically correct for incoming carry-chain length, but max chain is `0..3` for `seq_len=8`, not `0..4` [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:102).
- Noise injection is mechanically consistent with `s_{t+1}=s+F(s,c)+noise`, since `dynamics_step()` returns `s_new` [model.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:41).
- But the model is still in train mode, so deterministic and Langevin paths include dropout.
- Energy is tracked before adding the current step’s noise [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:144). That is a clean drift-energy measurement at the current noisy state, but it does not measure post-noise exploration or convergence.
- Majority voting is a major confound. It turns Langevin into an 8-sample ensemble [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:179), so gains need not mean basin escape.
- `torch.bincount(..., minlength=64)` hard-codes vocab size.
- Rescue analysis selects the best tau on the same eval set [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:230).
- It only tests seed `42`, despite the wrong-attractor hypothesis depending on failed E5 seed `512` [D12](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d12_langevin_escape.py:299).

Statistical plan:
- `2048` examples is adequate for aggregate accuracy and carry-chain bins, but not enough for reliable “rescued wrong examples” if deterministic failure is rare.
- Use paired McNemar tests for deterministic vs noisy correctness.
- Use bootstrap CIs for rescued, broken, and net-gain rates.
- Separate tau-selection and final-test sets.
- Correct for multiple comparisons across `7 tau * 3 schedules * 2 tracks`.
- Report single-sample noisy accuracy separately from majority-vote accuracy.

Alternative explanations:
- Improvement could be ensembling, not basin escape.
- Noise could act like readout margin smoothing or dropout augmentation.
- If tau hurts easy examples and helps hard examples, that could be variance/regularization rather than escape.
- Annealing can outperform constant simply because it injects less total noise late.

Missing controls:
- `tau=0` through the exact Langevin path.
- Single noisy sample versus majority vote.
- Same total noise budget across schedules.
- Noise at initial state only, final state only, and random direction controls.
- Failed E5 seed `512`.
- D11-conditioned analysis: examples classified by basin/barrier before noise.

Priority fix:
- Rebuild D12 around paired eval-mode single-trajectory rescue on failed E5 seed `512`, with majority vote treated as a separate ensemble baseline.

Prediction:
- As written, it crashes. After fixing that, moderate noise will mostly hurt or be neutral; majority voting may create small apparent gains. Confidence: medium-high.
- Clean rescue of wrong-attractor failures is unlikely without selecting for failed examples from seed `512`. Confidence: high.
- Annealing may beat constant at high tau because it is less destructive near readout, not because it proves Langevin escape. Confidence: medium-high.

Parsimony:
- First run only `tau in {0, 0.005, 0.01, 0.05}`, cosine schedule, deterministic paired baseline, seed `512`, single-sample plus optional ensemble.

---

**D13: Cross-Task Dynamics Transfer**

Correctness:
- Subtraction generator is correct for even `seq_len=8`: MSB-first `A-B mod base^half` with right-to-left borrow [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:83). It will break for odd `seq_len`, same pattern as addition, but current config is even.
- Comparison generator is formally correct, but statistically broken. With base 64 and four digits, the first digit differs about `63/64` of the time, so almost all labels are at position 0 and positions 1-3 are nearly always zero [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:105). This is not a sequential scan task.
- Multiplication is internally correct for `eff_base=16`, but the doc says vocab/base 64 while the task only uses digits `0..15` [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:136). That changes difficulty and makes comparison to addition unfair.
- `freeze_dynamics=True` freezes `model.dynamics.parameters()` [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:171), which should include the TransformerDecoderLayer and spectral-norm parametrizations. But `tok_emb`, `pos_dec`, encoder, and readout remain trainable, so the system can reprogram the interface around frozen dynamics.
- Transfer loads only the source dynamics, not the source encoder/readout [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:340). That is a valid dynamics-pretraining test, but not full model transfer.
- No hash/parameter-diff check proves frozen dynamics stayed unchanged.
- The claimed encoder-only baseline is not implemented.
- Training CE includes padded second-half zeros [D13](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d13_dynamics_transfer.py:192), while evaluation scores only first half. This makes comparison especially weak for comparison.

Statistical plan:
- `4096` eval examples is fine for accuracy CIs, but one seed is not enough.
- Use at least 5 seeds and report mean/CI over seeds.
- Compare learning curves on held-out eval batches, not training-batch seq accuracy.
- Use mixed-effects regression or paired bootstrap over examples plus seed-level tests.
- Measure time-to-threshold and area under eval learning curve.
- For transfer, compare frozen-addition dynamics against frozen-random dynamics and frozen-dynamics trained on an unrelated task.

Alternative explanations:
- Subtraction success could come from encoder/readout learning around a generic nonlinear block, not transferable computation.
- Comparison success will mostly mean the task is trivial under the current distribution.
- Multiplication failure could reflect base/domain mismatch or padded loss, not lack of transferable dynamics.
- Fine-tuned transfer success could simply be initialization benefit, not frozen algorithm reuse.

Missing controls:
- Random frozen dynamics.
- Frozen dynamics from copy/reversal/sort.
- Encoder-only/deep-encoder baseline.
- Full-source checkpoint transfer.
- Interface-limited transfer where `pos_dec` or encoder is also frozen.
- Balanced comparison data with forced equal prefixes.
- Same-base multiplication or explicitly separate low-base addition source.

Priority fix:
- Fix the target task distributions and baselines before running: balanced comparison, fair multiplication base, masked loss over result positions, and random-frozen dynamics control.

Prediction:
- Subtraction: frozen dynamics may show some benefit, but I would not expect >90% seq accuracy reliably with only dynamics loaded. Confidence: medium.
- Comparison: current task will look deceptively strong because it is mostly first-digit comparison plus zeros. Confidence: high.
- Multiplication: full training may struggle; frozen transfer likely underperforms full/fine-tuned. Confidence: medium.
- Any “general iterative computation” claim will remain unsupported without random-frozen and encoder-only controls. Confidence: high.

Parsimony:
- Drop multiplication initially. Run addition-to-subtraction plus balanced comparison, with random-frozen dynamics and encoder-only baselines. That gives much cleaner signal.