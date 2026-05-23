You are reviewing two new experiment designs as a senior architectural authority.

Read and analyze these files:
- experiments/06_uesd/exp_d15_nishimori_calibration.py (Nishimori calibration test)
- experiments/06_uesd/exp_d16_information_trajectory.py (information accumulation via probes)
- experiments/06_uesd/shared/model.py (UESD architecture)
- experiments/06_uesd/shared/data.py (data generators)
- experiments/06_uesd/results/codex_d11_d13_design_review.md (prior review for patterns)
- experiments/06_uesd/results/codex_d14_design_review.md (prior review for patterns)
- docs/UNIFIED_ERROR_SPACE.md (sections 8.3 and 2.2 especially)

CONTEXT:
These are the most theoretically ambitious experiments in the UESD series.

D15 tests whether UESD dynamics approach the Nishimori critical point where rho = tanh(1/2) approximately 0.462 — a specific quantitative prediction from prior lab research on 7 independent substrates. The test: at each dynamics step, compute readout probabilities and measure calibration. If there exists a step t* where average confidence approximately equals 0.462 and the model is well-calibrated, that connects a 694K-param model to universal statistical mechanics.

D16 tests Axiom A4 (thinking-generating continuum) by training linear probes from intermediate states s_t to targets y* at each step. The prediction: information about the answer increases monotonically through dynamics steps, with rate depending on carry-chain difficulty.

Apply ALL of the following for EACH experiment:

1. CORRECTNESS AUDIT
   - D15: Is the confidence computation correct? Does sweeping readout tau make sense? Is the calibration (ECE) computed correctly?
   - D16: Is the linear probe methodology sound? Is the train/test split correct? Could the probe be overfitting?

2. STATISTICAL PLAN
   - Sample sizes, seed counts, confidence intervals
   - Is 4096 eval examples enough? Is one seed enough?

3. ALTERNATIVE EXPLANATIONS
   - D15: Could any readout temperature tau produce a confidence near 0.462? Would that invalidate the Nishimori interpretation?
   - D15: Is the Nishimori prediction even applicable here (addition task, not a spin glass)?
   - D16: Could probe accuracy increase simply because the state becomes more structured, not because it contains target information?
   - D16: Is a linear probe sufficient, or could nonlinear information be missed?

4. DESIGN GAPS
   - Missing controls or baselines
   - How do these connect to D7-D14?

5. PRIORITY DIRECTIVE
   For EACH experiment, what is the single most important fix?

6. PREDICTION
   Your predictions with confidence levels.

7. PARSIMONY MANDATE
   Can either experiment be simplified?
