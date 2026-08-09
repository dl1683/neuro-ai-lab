I read [UNIFIED_ERROR_SPACE.md](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/docs/UNIFIED_ERROR_SPACE.md) and scanned all requested directories. The key conclusion: UESD’s problem is no longer “make it converge.” It is “make the correct basin the easiest basin.” A contraction proof alone can make wrong answers more reliable.

**Survey Result**

| Lineage | Convergence Mechanism | Wrong-Attractor Handling | Cost | UESD Fit |
|---|---|---|---|---|
| DEQ, Bai/Kolter/Koltun | Solve `z = fθ(z, x)` with root finding; train through equilibrium by implicit differentiation. Jacobian regularization stabilizes forward/backward solves. | No reported “wrong-attractor rate.” DEQ guarantees/encourages reaching an equilibrium, not semantic correctness. Correctness comes from supervised endpoint loss. | Iterative solve; O(1) memory, extra NFEs. Jacobian reg is cheap-ish via stochastic estimates. | Strong fit. Use DEQ machinery, but not DEQ alone. |
| Monotone operators, Ryu/Boyd; monDEQ | Cast equilibrium as zero of monotone operator; resolvents/operator splitting give stable convergence. Strong monotonicity gives uniqueness. | Can eliminate multiple fixed points if the conditional operator is globally/strongly monotone. Cannot guarantee the unique fixed point is the right answer. | Architectural constraints, spectral parameterization, splitting iterations; may reduce expressivity. | Use selectively around the final correction step, not the whole model. |
| EBMs / Hopfield | Shape energy so positives are low energy and negatives/spurious states are high energy. Hopfield-style Lyapunov energy ensures descent. | Directly relevant: basin engineering, negative samples, contrastive divergence, spurious-minima suppression. | Negative mining/sampling cost; contrastive batches. | Best answer to wrong attractors. |
| Predictive coding / Friston / Rao-Ballard | Iterative hierarchical prediction-error minimization. Precision/gain weights decide which errors dominate. | Handles ambiguity by priors, precision weighting, top-down constraints, and recurrent error correction. Not a hard uniqueness guarantee. | Extra recurrent inference steps; local errors cheap at small scale. | Good as auxiliary “multi-level error” supervision. |
| Contractive autoencoders | Penalize encoder Jacobian norm to enforce local invariance/contraction near data manifold. | Makes local basins smoother and robust, but can deepen wrong basins if applied indiscriminately. | Very cheap relative to full implicit solves. | Use near target manifold only; avoid global k-suppression. |
| Neural ODEs / ANODEs | Continuous dynamics with adaptive solvers; adjoint or checkpointed gradients. ANODE adds dimensions to avoid topology bottlenecks. | Adaptive time helps hard cases, but ODE uniqueness can preserve topology and prevent basin crossing unless augmented/noisy. | Solver overhead; stiffness can be expensive. | Use variable-T/adaptive stopping, not full ODE training first. |
| General implicit layers | Constrain architecture so equilibrium exists, is unique, or is nonexpansive/contractive. | Same issue: convergence proof is orthogonal to target correctness unless coupled to endpoint/energy shaping. | Depends on constraint strength. | Useful scaffolding, not sufficient objective. |

**Most Promising UESD Fix**

Adopt a **target-conditioned contractive EBM-DEQ hybrid**:

1. Keep the DEQ-style fixed-point dynamics.
   Use root finding or fixed unroll during training, implicit differentiation when stable enough. Add Bai-style Jacobian regularization, but target `J_F` near the endpoint and sampled neighborhood, not global over-suppression.

2. Replace pure self-consistency with a shaped energy:
   `L = λ_fix ||F(s_T,c)||² + λ_pos d(s_T, s*_y) + λ_neg InfoNCE(s_T, positives, hard_negatives) + λ_readout CE(R(s_T), y) + λ_jac ||J_F(s_T)||²`.
   The missing term is hard-negative basin repulsion. Self-consistency says “stop moving”; contrastive energy says “stop moving at the right attractor.”

3. Add basin audits as a training metric:
   For each prompt, start from 8-16 perturbed initial states. Measure: convergence rate, target hit rate, wrong-attractor clustering, basin margin `E(wrong)-E(correct)`, and readout accuracy. Optimize the basin margin, not just `k`.

4. Use monotonicity locally:
   Make the final correction/readout-prox step strongly monotone or averaged, but do not force the whole expressive dynamics to be globally monotone yet. Global monDEQ is attractive theoretically but risks underfitting a 700K-param model.

5. Add predictive-coding style multiscale errors:
   Intermediate states should retain target constraints: semantic anchor, syntax/readout anchor, and self-consistency anchor. This matches the LSR “anchor retention” lesson: useful intermediate anchors must preserve constraints while exposing repairable degrees of freedom.

**Skip For Now**

Skip full Neural ODE training, full global monDEQ, and pure contractive penalties as the main fix. They improve convergence diagnostics but do not directly attack the 20% wrong-attractor failure. Also skip “stronger contraction everywhere”; with `k=0.988`, stronger contraction may just make bad basins stickier.

Primary sources used: [DEQ](https://arxiv.org/abs/1909.01377), [Jacobian regularized DEQ](https://proceedings.mlr.press/v139/bai21b.html), [monDEQ](https://arxiv.org/abs/2006.08591), [Ryu/Boyd monotone primer](https://web.stanford.edu/~boyd/papers/monotone_primer.html), [Contractive Autoencoders](https://icml.cc/2011/papers/455_icmlpaper.pdf), [Neural ODE](https://arxiv.org/abs/1806.07366), [Augmented Neural ODE](https://arxiv.org/abs/1904.01681), [Rao-Ballard predictive coding](https://www.nature.com/articles/nn0199_79), [Friston predictive coding/free energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/).

