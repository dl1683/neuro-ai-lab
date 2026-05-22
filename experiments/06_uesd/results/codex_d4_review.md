**Verdict**

D4 is a useful pilot, but the current writeup overclaims. The raw run supports: “in seed 42, CE-dynamics and E5 show very different trajectory-Jacobian profiles during training.” It does not yet support “fundamentally different stability mechanisms discovered” as a statistical or mechanistic conclusion.

**Methodology Critique**

4 samples per snapshot is not sufficient for statistical claims. Worse, `snapshot_trajectory` always uses the first 4 eval examples, not a random sample. That makes the time-series internally comparable, but not population-representative.

`snapshot_trajectory` is mostly computing the intended finite-time product Jacobian correctly: central finite differences, per-step Jacobians, ordered product `J_t @ ... @ J_0`, SVD amplification, and finite-time Lyapunov `log(sigma(product))/T`. But there are caveats:

- `product_of_sigmas` multiplies averaged per-step sigmas, not the mean per-sample product. That makes `conservatism_product` an aggregate diagnostic, not a clean theorem-bound ratio.
- Alignment is only cosine between consecutive top right singular vectors. It is fragile if top singular values are close, and it is not the actual cancellation geometry identified in the D3 review.
- The shuffled control uses one random permutation per sample. O/S ratios need many shuffles per trajectory to estimate a null distribution.
- Finite difference `eps=1e-4` is unvalidated. D3 already flagged the need for eps sweeps and autograd spot-checks.

The fixed eval batch is a double-edged choice. It is good for tracking training dynamics over time on identical inputs. It is bad for general claims because all phase stories are conditioned on the same 4 examples. The eval batch is generated once and reused, but not used in the loss, so this is not direct train/eval leakage. Still, random training batches can include similar or identical addition cases, and the diagnostic is not held-out in a strong sense.

**Claim Validation**

1. **Two fundamentally different stability mechanisms.**  
Evidence: suggestive for this seed. CE alignment drops to 0.068 then recovers; E5 stays above 0.77 and final amplification is lower.  
Alternative: seed artifact, fixed-four-example artifact, SC directly shaping the measured Jacobian, or singular-vector instability.  
Strengthen: 5+ seeds, 16-32 held-out examples, top-k subspace metrics, lambda sweep.

2. **CE “scattered dynamics.”**  
Evidence: alignment dip is real in saved numbers.  
Alternative: early-training singular vector degeneracy, not meaningful rotation; loss plateau dynamics unrelated to task exploration.  
Strengthen: singular gaps, top-k principal angles, readout-sensitivity of rotated directions.

3. **E5 “highway dynamics.”**  
Evidence: E5 maintains high alignment and reduces amplification to ~1.8x.  
Alternative: SC loss forces proximity to fixed-point-like regions, so “highway” is just the regularizer’s direct footprint.  
Strengthen: compare CE + weak SC, lambda sweep, and matched-CE checkpoints.

4. **Conservatism paradox explained.**  
Evidence: E5 has higher reported conservatism while lower actual amplification.  
Concern: ratio uses product of averaged sigmas, and explanation depends on a weak alignment metric.  
Strengthen: sample-wise conservatism distributions and rotation-corrected bound using product-subspace alignment.

5. **O/S diagnostic.**  
Evidence: E5 O/S >1, CE late O/S often <1.  
Concern: 0.93 vs 1.10 is small with 4 samples and one shuffle. Not meaningful yet.  
Strengthen: 100+ shuffles per trajectory, CI against shuffled null, matched-spectrum random controls.

6. **Three-phase regime is CE-specific.**  
Evidence: observed in seed 42.  
Alternative: CE’s transition timing in this seed; E5 could show it in failed or different seeds.  
Strengthen: multi-seed phase alignment by CE-loss threshold, not wall-clock step.

7. **SC accelerates CE transition.**  
Evidence: E5 reaches low CE around step 2000 vs CE-dynamics around 3000 in this run.  
Alternative: seed/data-order artifact; E5 also has known failure modes from D2.  
Strengthen: survival/time-to-threshold analysis across seeds.

**Statistical Rigor**

No CE-vs-E5 difference is statistically significant as currently run. There is N=1 seed per track, and the 41 snapshots are not independent replicates. The apparent differences are large enough to justify follow-up, not publication-grade inference.

A proper comparison needs multiple seeds per track, random held-out diagnostic batches, per-seed summary metrics defined in advance, and confidence intervals over seeds. Example metrics: minimum pre-transition alignment, final Lyapunov, final amplification, final O/S, time-to-CE-threshold, and post-transition mean alignment.

**Mechanistic Depth**

“Scattered dynamics” vs “highway dynamics” is a useful metaphor, not yet a demonstrated mechanism. The data show different alignment/amplification traces. They do not yet prove why those traces occur.

Yes, the alignment difference could be an artifact of the SC loss. E5’s SC term directly penalizes residual dynamics at `s_T`, which can bias trajectories toward fixed-point neighborhoods where Jacobians are more consistent and lower gain. That is still scientifically interesting, but it weakens the claim that E5 discovered an independent mechanism.

The O/S interpretation is too strong. Final CE O/S = 0.93 and E5 O/S = 1.10 are small differences without uncertainty. Early E5 O/S = 1.84 is more interesting, but needs a many-shuffle null.

**Connection To D3**

D4 extends D3 in the right direction: it tracks the product-Jacobian phenomenon during training rather than only after convergence. It also gives a plausible loss-dependent hypothesis: CE-dynamics may pass through a low-alignment exploration phase while E5 avoids it.

It does not change the D3 review’s main cautions. The D3 review said rotation was a leading candidate, not proven; D4 still has the same unresolved issues: finite-difference validation, top-k subspace geometry, random-matrix nulls, held-out batches, and readout relevance of high-gain directions.

Also, CE-dynamics still should not be used as fixed-point stability evidence if it lacks convergence in earlier diagnostics. Its D4 result is a finite-horizon tangent-sensitivity result.

**Next Experiments**

| Direction | Impact | Feasibility | Risk, 10 = lowest |
|---|---:|---:|---:|
| Multi-seed D4, 5 seeds per track | 10 | 8 | 9 |
| Length scaling, L=16 and L=32 | 8 | 6 | 6 |
| Rotation-corrected Theorem 4.1 | 9 | 6 | 5 |
| Random-matrix null model | 9 | 8 | 8 |
| Explicit rotation loss | 6 | 5 | 4 |
| Task-error sensitivity | 8 | 7 | 7 |

**Priority Directive**

Do multi-seed D4 next, but fix the diagnostic first: use 5 seeds per track, at least 16 held-out diagnostic examples, randomize diagnostic examples, run many shuffles per trajectory, and report per-seed summary metrics with CIs. Until the single-seed result survives that, the mechanism language should stay provisional.