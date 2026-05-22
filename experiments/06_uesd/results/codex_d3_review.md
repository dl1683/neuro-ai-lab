**Verdict**

D3 is a strong pilot diagnostic, but not yet a research-grade mechanism proof. The core observation is real: the trajectory product Jacobian is far smaller than a naive per-step singular-value story would predict. But the current writeup overclaims the exact conservatism ratios, the “rotation is the mechanism” certainty, and the “edge of chaos” framing.

I could not find `CLAUDE.md` in this repo scope; the root has `.claude/` but no `CLAUDE.md`. I reviewed the other requested files directly.

**Evidence Gate**

The biggest issue is in [diagnostics.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/diagnostics.py:476): `theorem4_bound = avg_per_step_sigma[-1] ** T`. That is the last averaged per-step sigma raised to `T`, not the product of the per-step sigmas, not `max_t sigma_t^T`, and not the fixed-point `sigma_max(J*)^T` bound. So the headline “1027x” is not the theorem bound as stated. Recomputed from the saved averages:

| Run | Stored Ratio | Product-of-Avg-Sigmas Ratio |
|---|---:|---:|
| CE seed 42 | 26.6x | ~5001x |
| CE seed 137 | 6.5x | ~508x |
| E5 seed 42 | 1027x | ~372x |
| E5 seed 137 | 331x | ~1363x |

So “Theorem 4 is conservative” survives. “1027x is the central number” does not.

Finite-difference Jacobians at `eps=1e-4` are plausible but not validated. Central differences on a float32 CUDA Transformer with layer norm, attention softmax, dropout disabled, and ReLU-like FFN are not obviously wrong, but singular-vector alignment is sensitive to numerical noise and top-singular-value degeneracy. You need an eps sweep: `1e-3, 3e-4, 1e-4, 3e-5, 1e-5`, plus an autograd Jacobian check on 2-4 examples. If alignment and cumulative sigma are stable, the artifact concern mostly clears.

Sixteen trajectories are enough to say “this phenomenon exists in these trained models.” They are not enough for population-level claims about CE vs E5 mechanisms. The within-run Lyapunov values are tight for E5 seed 42, but seed count is still only two per track.

The product-Jacobian quantity is the right finite-time measure. But call it **finite-time Lyapunov exponent / tangent amplification**, not asymptotic Lyapunov exponent. For larger `T`, higher dimension, or full spectrum, use QR/SVD reorthogonalization. Direct products are fine for `T=10`, but QR is the correct method if this becomes a serious Lyapunov-spectrum diagnostic.

The alignment metric is suggestive but incomplete. Cosine between dominant right singular vectors only tests one direction, and it can be meaningless if `sigma_1 ≈ sigma_2`. Add singular gaps, principal angles between top-k right subspaces, and the overlap `||V_{t+1,k}^T U_{P_t,k}||` between the next Jacobian’s input-amplifying subspace and the current product’s output-amplified subspace. That is closer to the actual cancellation geometry.

**Theory**

The non-normal connection is currently hand-wavy but promising. A rigorous theorem would be a finite-time bound for non-autonomous products:

`||J_T ... J_1|| <= product_t sigma_t * product_t alpha_t`

where `alpha_t` captures alignment between the right singular subspace of `J_t` and the amplified left/output subspace of `P_{t-1}`. In rank-1 intuition, growth is approximately `sigma_t * sigma(P_{t-1}) * |<v_t, u_{P_{t-1}}> |`. Shrinkage happens when later Jacobians have small gain on the dominant output direction of the previous product, even if their own `sigma_max > 1`.

That is the geometric criterion you want: not “consecutive right singular vectors rotate,” but “the next map’s high-gain input subspace is misaligned with the current product’s high-gain output subspace.”

Edge of chaos is oversold. `lambda_max > 0` over 10 steps means finite-time perturbation amplification, not established chaos. Values `0.045-0.199` are “near-critical finite-time amplification,” not chaos unless you show asymptotic positive exponents, invariant dynamics, sensitivity, and bounded recurrent behavior. Use “near marginal tangent stability” for now.

**Alternative Explanations**

Yes, the states themselves may be the stability mechanism. D3 measures `J_t` along learned trajectories, so stability may come from the trajectory visiting regions where Jacobians cancel or where the readout margin is robust. That is not a flaw, but it changes the claim: the mechanism is state-conditioned trajectory geometry, not a global Jacobian property.

CE-dynamics has `converged_frac=0.0`, so fixed-point stability theory does not apply cleanly. The Lyapunov analysis still applies as a finite-horizon input-output sensitivity diagnostic. Do not use CE-dynamics D3 to support fixed-point convergence claims.

The finite-difference artifact test is mandatory before publishing the rotation claim. Also test random held-out batches, not one generated evaluation batch.

**Cross-Domain Imports**

Use non-normal fluid dynamics for vocabulary and tools: transient growth, pseudospectrum, resolvent norm, Kreiss constant. But avoid implying the same physics.

Use Oseledets/MET only after switching to long trajectories and QR Lyapunov spectra. Current D3 is finite-time, input-conditioned, length-10 tangent analysis.

Reservoir computing is directly relevant: echo state property, contractivity, spectral radius vs singular values, and task performance near marginal stability. This is probably the cleanest external framing.

Random matrix products give the right null model: compare trained `J_t` products against shuffled-time Jacobians, random orthogonal rotations preserving singular spectra, and random matrices with matched spectra. If trained ordering beats/shows structured cancellation beyond nulls, the “learned rotation” claim gets teeth.

**Next Directions**

| Direction | Impact | Feasibility | Novelty |
|---|---:|---:|---:|
| Rotation-corrected Theorem 4.1 | 10 | 7 | 9 |
| Eps/autograd validation + full spectrum | 9 | 9 | 5 |
| Phase transition dynamics during training | 9 | 7 | 8 |
| Length scaling `L=16,32` | 9 | 6 | 7 |
| Compare to DEQ/COCONUT/RNN dynamics | 8 | 5 | 8 |
| Random-matrix/null product controls | 8 | 8 | 8 |
| Adaptive `T` via trajectory FTLE | 7 | 8 | 7 |
| Rotation-aware loss | 7 | 5 | 8 |
| Information flow per step | 6 | 5 | 6 |
| Pseudospectrum/Kreiss diagnostic | 8 | 6 | 7 |
| Margin-conditioned tangent analysis | 8 | 8 | 7 |

Two additional high-value directions: shuffled-trajectory ablation, where you multiply the same Jacobians in random order to test whether temporal ordering matters; and task-error sensitivity, where you measure whether high-gain tangent directions actually affect readout logits or lie mostly in null/readout-irrelevant subspaces.

**Priority Directive**

Fix and validate D3 before theorizing further: implement a corrected trajectory diagnostic with product-of-per-step bound, eps sweep, autograd spot-checks, singular gaps, top-k subspace angles, QR Lyapunov spectrum, and shuffled/random null controls. This is the single next move because it determines whether D3 is a real discovery or a suggestive but fragile measurement artifact.

**Anti-Overconfidence**

The “1027x” number is not reliable as stated because the bound computation is inconsistent with the theorem. Two seeds per track are not enough to claim a mechanism difference between CE and E5. `lambda_max=0.2` is not evidence of chaos. “Jacobian rotation is the stability mechanism” should become: “finite-time tangent amplification is much smaller than naive per-step singular-value bounds, and subspace misalignment is the leading candidate mechanism.”

