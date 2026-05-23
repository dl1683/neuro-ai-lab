You are reviewing an experiment design as a senior architectural authority.

Read and analyze these files:
- experiments/06_uesd/exp_d10_adaptive_halting.py (the experiment under review)
- experiments/06_uesd/shared/model.py (UESD model architecture)
- experiments/06_uesd/shared/data.py (data generation)
- experiments/06_uesd/shared/training.py (shared training utilities)
- experiments/06_uesd/exp_d7_thinking_emergence.py (related experiment: wavefront timing)
- experiments/06_uesd/results/codex_d8_design_review.md (prior review for context)

CONTEXT:
D10 tests adaptive halting. The hypothesis is that UESD dynamics learn to allocate computation based on input complexity. It uses PonderNet-style learned halting: at each dynamics step t, a halting head predicts h_t = P(halt at step t). The output is the expectation over readouts weighted by the halting distribution.

Key claims being tested:
1. Easy inputs (no carry chains) should halt early, hard inputs (long chains) should halt late
2. Halting distribution should be multimodal, matching carry chain complexity groups
3. Adaptive halting should match fixed-T accuracy with 40-60 percent fewer mean steps
4. Beta parameter (KL penalty) controls speed-accuracy tradeoff

This is the strongest test of the thinking hypothesis. If UESD truly computes, computation should be adaptive.

Apply ALL of the following, in order:

1. CORRECTNESS AUDIT
   For every function, trace the data flow:
   - Is compute_carries() correct? Does it handle MSB-first convention properly?
   - Is the PonderNet halting distribution computed correctly? (running product of 1-h_i, forced halt at T)
   - Is the loss correct? (weighted CE + beta * KL to geometric prior)
   - Is the KL computation correct? (direction, log-space, batch handling)
   - Is the greedy evaluation correct? (using mode of halt distribution)
   - Are there any numerical stability issues?

2. STATISTICAL PLAN
   - Is N=4096 eval sufficient for per-chain-length analysis?
   - Are the chain length groups balanced enough for meaningful comparisons?
   - Is 25000 training steps enough for the halting head to converge?
   - What statistical tests should be applied to the per-chain-length halting results?

3. ALTERNATIVE EXPLANATIONS
   For each prediction, what alternative mechanism could produce the same result?
   - Could the halting head learn to halt based on input token patterns (not computation)?
   - Could the correlation between chain length and halting step be epiphenomenal?
   - Could the geometric prior dominate and mask the adaptive signal?
   - What controls are missing?

4. DESIGN GAPS
   - The halting head pools over positions. Should it be per-position?
   - Should there be a threshold-based baseline (halt when update norm is below epsilon)?
   - How does this interact with the E5 vs CE-dynamics distinction from prior experiments?
   - Should both training regimes be tested? (Currently only CE-dynamics baseline)
   - The fixed-T baseline uses T_MAX=15 not T=10. Is this a fair comparison?

5. CONNECTION TO D7
   D7 measures when computation arrives at each position. D10 measures when the model chooses to halt. These should be deeply connected:
   - Can we correlate D7 first-correct-step with D10 halting step?
   - Should they share checkpoints?
   - What does it mean if halting step is less than first-correct-step?

6. PRIORITY DIRECTIVE
   What is the single most important thing to fix or add before running D10?

7. PREDICTION
   Your own predictions for what D10 will find, with confidence levels.

8. ANTI-OVERCONFIDENCE PROTOCOL
   For each confidence point, cite specific evidence, not general plausibility.

9. PARSIMONY MANDATE
   Is the experiment design minimal? What can be removed without losing signal?
