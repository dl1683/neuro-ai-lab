I read the UESD doc and mined the requested repos. The strongest conclusion is simple: `||F(s)|| -> 0` is only a stopping/stability condition. UESD needs a separate **attractor promotion and basin editing layer** that answers: “is this fixed point the right one, should it be promoted, and how do we escape/reshape it if not?”

I did not find an `Attractors and Fixed Points` folder in `Open Exploration`, but I did find the relevant fixed-point material under `Recursion and Self-Reference`.

**Most Actionable Findings**

1. **Self-consistency gives existence, not correctness**
Source: `Market Reports\Open Exploration\Recursion and Self-Reference\fixed_points_and_recursion.md`

Insight: The file distinguishes Brouwer-style fixed-point existence from Banach-style contraction with uniqueness and convergence. It also discusses Hopfield networks, where memories are attractor basins and noisy cues settle into the nearest stored pattern.

UESD application: Your current result proves “the dynamics settle.” It does not prove the correct answer is the unique or preferred attractor. The 20% wrong fixed points are exactly the Hopfield failure mode: stable recall of the wrong memory.

Mechanism/experiment: For correct and wrong fixed points, estimate basin volume, Jacobian spectral radius, Hessian/curvature, and local contraction. Add a contrastive basin loss: make correct endpoints contractive and low energy, while wrong fixed points are either higher energy or locally unstable.

2. **Candidate promotion is the missing layer**
Sources: `Latent-Space-Reasoning\DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md`, `Latent-Space-Reasoning\DIFFUSION_ERROR_FUNCTION_GEOMETRY.md`, `Latent-Space-Reasoning\DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md`

Insight: LSR separates “a repair candidate exists” from “this candidate should be promoted.” In the v7 diffusion notes, promotion is candidate-aware: a candidate is positive only if it improves over the selected trajectory. The audit also frames strict correction as detect, diagnose, repair, verify.

UESD application: A fixed point should not automatically become the output. It should become a candidate endpoint that must pass a promotion criterion.

Mechanism/experiment: Train a UESD endpoint promotion head using trajectory features: final readout confidence, verifier score, residual structure, path smoothness, perturbation recovery, and agreement across sampled trajectories. Compare:
`self-consistency only` vs `self-consistency + promotion gate`.

3. **Use wrong attractors as explicit negatives**
Sources: `knowledge-surgeon\KNOWLEDGE_AUGMENTATION_FINDINGS.md`, `knowledge-surgeon\CAPABILITY_MAPPING_PROGRESS.md`

Insight: Knowledge Surgeon found that negative examples prevent leakage better than positive augmentation alone. It also cautions that naive global/vector edits fail; useful edits should stay on local tangent/geodesic directions and target early/control regions.

UESD application: The 20% wrong fixed points are valuable supervised negatives. Do not just train toward correct endpoints; train away from known wrong endpoints.

Mechanism/experiment: Build fixed-point pairs per prompt: `{correct attractor, wrong attractor}`. Add a contrastive energy term:
- lower energy / stronger contraction around correct fixed point
- higher energy / weaker contraction around wrong fixed point
- tangent-constrained correction vector from wrong to correct state

4. **Bioelectric morphogenesis is the clearest analogy**
Sources: `Market Reports\Open Exploration\Developmental Computation\bioelectric_computation.md`, `morphogenetic_programming.md`

Insight: Planarian regeneration can settle into stable but wrong morphology, such as two-headed attractors. The lesson is that the target pattern must be encoded in the landscape/control hierarchy, not just in local update rules. Neural cellular automata achieve robustness by training from damaged states back to the target.

UESD application: Wrong UESD fixed points are “wrong morphology” states. The solution is not more convergence pressure; it is target-conditioned landscape shaping and repair training.

Mechanism/experiment: Damage-training for UESD:
- start from correct terminal states plus perturbations
- start from known wrong fixed points
- train `F` to recover to the correct endpoint
- measure whether the correct basin expands and wrong basin volume shrinks

5. **Noise helps only when coupled to selection**
Sources: `Market Reports\Open Exploration\Noise and Randomness\noise_as_computation.md`, `Error Correction Across Scales\when_errors_are_features.md`, `Latent-Space-Reasoning\src\latent_reasoning\core\autopoietic\homeostasis.py`

Insight: Noise enables basin escape, stochastic resonance, simulated annealing, and sampling, but useful error requires selection. LSR also has a homeostatic temperature controller that raises temperature when diversity is too low and lowers it when diversity is too high.

UESD application: Langevin noise should not be generic exploration. It should be adaptive: increase noise when the system is prematurely collapsing into suspicious attractors, then cool after a candidate passes verification.

Mechanism/experiment: Run ensemble UESD trajectories with adaptive temperature:
- maintain target diversity across states
- sample candidate fixed points
- promote only verified endpoints
- test parallel tempering against fixed-temperature Langevin

6. **Nishimori suggests calibrated noise/basin selection**
Source: `_meta\research\nishimori-cross-domain.md`

Insight: The Nishimori condition links optimal inference to calibration: the assumed noise/prior matches the true corruption process. The `_meta` note frames this as confidence matching accuracy, with a proposed correlation law `rho(tau)=tanh(1/(2tau))`.

UESD application: Wrong attractors may be an off-Nishimori failure: the system is highly self-consistent but miscalibrated. Confidence/self-consistency no longer tracks correctness.

Mechanism/experiment: Sweep Langevin temperature `tau` and measure:
- correct basin hit rate
- wrong attractor rate
- confidence-vs-accuracy calibration
- pairwise state correlation
- whether optimal correctness occurs near the predicted Nishimori operating point

7. **Graceful degradation says add redundancy, detection, repair**
Sources: `Market Reports\Open Exploration\Error Correction Across Scales\graceful_degradation.md`, `_meta\insights\cross-domain-mechanisms.md`

Insight: Robust systems combine redundancy, detection, correction, plasticity, and cascade prevention. The `_meta` mechanism summary gives a four-step universal correction loop: redundancy, detection, correction, cascade.

UESD application: A single trajectory to a single fixed point is brittle. Correctness should be redundant and cross-checked.

Mechanism/experiment:
- run multiple perturbed trajectories
- cluster endpoints
- require independent agreement before promotion
- if endpoints disagree, re-enter dynamics with targeted perturbation or verifier gradient
- track degradation curves under state/prompt perturbation

8. **Criticality should be search phase, not endpoint phase**
Sources: `Market Reports\Open Exploration\Edge of Chaos\criticality_as_computation.md`, `phase_transitions_in_learning.md`

Insight: Critical systems maximize sensitivity, memory, and dynamic range. Learning phase transitions can move systems between memorization-like and generalization-like basins.

UESD application: If dynamics are too contractive too early, they may lock into wrong attractors. UESD likely needs near-critical exploration followed by contraction only after correctness evidence appears.

Mechanism/experiment: Use a two-phase spectral schedule:
- exploration: largest Lyapunov/Jacobian eigenvalue near zero
- commitment: make dynamics contractive only after verifier/promotion passes
Compare wrong-attractor rate against always-contractive training.

9. **Geometry matters locally, but global alignment claims are risky**
Sources: `Market Reports\Open Exploration\Information Geometry\the_geometry_of_learning.md`, `_meta\research\platonic-priors.md`, `llm-platonic-geometry\RESEARCH_FINDINGS.md`, `llm-rosetta-stone\docs\reports\ROUNDTABLE_OVERFITTING_CRISIS_FEB5_2026.md`

Insight: Information geometry gives useful instruments: flatness, Hessian spectra, saddles, mode connectivity, low-dimensional manifolds. But `_meta` and Rosetta both warn that high-dimensional global alignment can overfit badly; local neighborhood structure is more credible than global CKA/vector claims.

UESD application: Basin boundaries should be studied with local geometry, not raw Euclidean distance or global spectral claims.

Mechanism/experiment:
- build mutual-kNN graphs of endpoint neighborhoods
- compare correct vs wrong fixed points by local intrinsic dimension, curvature, Hessian outliers, Lyapunov estimate
- use permutation/null tests before claiming separability
- train basin classifiers with PCA/regularized low-dimensional features to avoid Rosetta-style overfit

10. **Readout/interface geometry can dominate attractor correctness**
Sources: `LLM Genome Project\DELETE_ME.md`, successor repo `AI Moonshots\moonshot-llm-genome\CLAUDE.md` and archived cycle notes

Insight: The requested `LLM Genome Project` folder is effectively empty and points to deletion/migration. The successor Genome work says broad internal transfer claims failed, while interface/tokenizer/embed/lm_head priors survived more strongly. Cross-tokenizer and cross-architecture geometry often breaks.

UESD application: UESD’s readout `R` should not be treated as harmless post-processing. A continuous fixed point is only “correct” relative to the output interface. Wrong attractors may be readout/interface failures as much as dynamics failures.

Mechanism/experiment:
- freeze `F`, vary or perturb `R`, and measure wrong-attractor rate
- co-train readout-aware energy, not just `F(s)=0`
- add an interface compatibility loss so fixed points are separated in output-relevant directions

11. **Rosetta says relations transfer better than vectors**
Source: `llm-rosetta-stone\docs\reports\ROUNDTABLE_COMPREHENSIVE_JAN8_2026.md`

Insight: Rosetta’s core result is “Rosetta exists in relations, not vectors”: RSA similarity can be high while cosine/vector transfer fails. Within-model steering works; cross-model steering fails.

UESD application: Correctness may be a relational property among state components, not a coordinate location. Basin selection should preserve relational constraints, not just move states toward a vector centroid.

Mechanism/experiment:
- train a relational verifier over state neighborhoods or feature interactions
- compare vector-distance-to-target vs relation-preserving endpoint score
- test whether wrong attractors satisfy local vector criteria but violate relational constraints

**Recommended UESD Experiment Suite**

1. **Fixed-point census**
For each prompt, sample many trajectories, cluster endpoints, label correctness, and estimate basin volume, energy, Jacobian spectral radius, Hessian/curvature, local intrinsic dimension, and readout margin.

2. **Promotion gate**
Treat every fixed point as a candidate. Train a promotion model to accept correct endpoints and reject wrong stable endpoints. Report strict accuracy, not only convergence.

3. **Contrastive basin shaping**
Use known wrong fixed points as negatives. Lower energy and increase contraction around correct endpoints; raise energy or reduce contraction around wrong endpoints.

4. **Damage-repair training**
Perturb correct endpoints and wrong endpoints, then train the dynamics to recover to the correct endpoint. Measure expansion of correct basin volume.

5. **Adaptive noise and parallel tempering**
Use ensemble diversity and verifier disagreement to control temperature. Compare fixed `tau`, annealing, adaptive homeostasis, and parallel tempering.

6. **Nishimori calibration sweep**
Sweep temperature/noise assumptions and measure correctness, calibration, and state correlation. Look for the point where confidence matches accuracy, not where convergence is fastest.

7. **Readout canary**
Hold dynamics constant and perturb/swap/readout-train `R`. If wrong-attractor rate changes sharply, endpoint correctness is interface-shaped and the loss must be readout-aware.

8. **Local geometry audit**
Use mutual-kNN, PCA/null-calibrated classifiers, curvature, and Lyapunov estimates to distinguish correct from wrong basins. Avoid high-dimensional global separator claims unless they survive permutation tests.

Bottom line: the best import from the sibling repos is not a single theory; it is an architecture change. UESD needs **search dynamics + candidate promotion + calibrated noise + explicit wrong-attractor negatives + repair training**. That directly targets the failure mode: stable but incorrect fixed points.

