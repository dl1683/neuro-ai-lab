**Verdict**

Do not treat D15 as a Nishimori test as written. D16 is a useful probe pilot, but its current split and controls are too weak for an A4/information-accumulation claim.

**D15: Nishimori Calibration**

Top correctness issue: the confidence metric is wrong for calibration. It uses probability assigned to the true token as “confidence” [exp_d15_nishimori_calibration.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d15_nishimori_calibration.py:171), then bins that against argmax correctness. Standard ECE should bin `max_softmax_prob` against whether `argmax == target`. Current ECE can make a bad model look calibrated because low true-label probability plus wrong prediction gives a small bin error.

Second correctness issue: D15 bypasses the trained readout. The model’s readout applies `readout_proj`, normalization, and `tau` [model.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:56), but D15 directly compares raw `s` to token embeddings [exp_d15_nishimori_calibration.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d15_nishimori_calibration.py:163). That is not the model’s actual decoder.

Sweeping `tau` is a major interpretability problem. Since `tau` directly controls probability sharpness [exp_d15_nishimori_calibration.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d15_nishimori_calibration.py:53), a value near `0.462` can be produced by readout-temperature tuning. That weakens the Nishimori interpretation unless `tau` is fixed before evaluation or selected on a separate calibration set and treated as a nuisance parameter.

The theory bridge is also under-specified. The doc says the Nishimori identity holds in continuous error space and that softmax moves the system off the Nishimori manifold [UNIFIED_ERROR_SPACE.md](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/docs/UNIFIED_ERROR_SPACE.md:326). D15 tests token-softmax probabilities on an addition task. That may be a proxy, but it is not yet the continuous-space quantity the theory names.

Statistical plan: `4096` examples, `4` result positions, so about `16k` token observations is enough for aggregate pilot CIs. It is not enough for long carry-chain bins or a physics claim. One model seed is not enough. Use at least `5` training seeds for design validation, ideally `7+` if invoking the “seven substrates” framing. Bootstrap CIs over examples, seed-level CIs over checkpoints, and separate tau-selection from final testing.

Missing controls: fixed `tau=model.tau`, proper max-prob ECE, untrained model, random-readout control, shuffled-label control, encoder-only with the same calibration computation, D7 first-stable-step alignment, D11 energy/update-norm alignment, and D14-style carry-depth stratification.

Priority fix: replace the metric stack first: use the actual `model.readout_logits`, compute confidence as predicted-class probability, report ECE/MCE/reliability diagrams, and pre-register `tau` rather than sweeping it as evidence.

Prediction: with proper calibration, I expect some step/tau combinations to hit average confidence near `0.462`, but mostly because temperature and accuracy trajectories make that easy. Confidence: high. I do not expect a clean “ECE minimum exactly at rho” result across seeds. Confidence: medium-high.

Parsimony: simplify D15 to one model track, fixed tau, proper ECE, 5 seeds, and a single falsifiable claim: “the ECE-minimizing step has confidence near 0.462 more often than controls.”

**D16: Information Trajectory**

The probe concept is sound as a descriptive lower-bound test: frozen `s_t` states are collected across dynamics [exp_d16_information_trajectory.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d16_information_trajectory.py:140), then linear probes estimate decodable target information. But probe accuracy is not mutual information, and monotone probe accuracy is not by itself proof of “thinking.”

The train/test split is the biggest flaw. It flattens examples and positions, then randomly splits tokens [exp_d16_information_trajectory.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d16_information_trajectory.py:159). Positions from the same addition problem can land in both train and test. Split by example first, then flatten positions.

Overfitting risk is moderate, not catastrophic: the probe has about `8k` parameters and around `13k` token-level train samples. But because examples leak across splits, this sample count is misleading. Add validation curves, L2/logistic regression, shuffled-label probes, and fixed train/test indices shared across all `t`.

Alternative explanations: increasing probe accuracy could mean states become more linearly organized, not that total target information increases. Since addition targets are deterministic functions of the source [data.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/data.py:22), a probe may decode source/carry features rather than answer commitment. Linear probes can also miss nonlinear information at early steps.

Statistical plan: `4096` examples is enough for a pilot if split by example. Use `5` model seeds, fixed probe splits, bootstrap CIs over held-out examples, and seed-level CIs for monotonicity/backtracking. For carry-chain bins, ensure each bin has enough held-out examples or aggregate across larger eval sets.

Missing controls: probes on `s_0`, encoder context, raw embeddings/input-derived features, shuffled targets, random model states, nonlinear MLP probe as an upper-bound check, and direct readout accuracy paired with D7 first-stable-correct steps. Connect to D11/D12 by comparing probe jumps to update norm/energy/noise sensitivity; connect to D14 by stratifying by T and carry depth.

Priority fix: change to example-level train/test splits with identical held-out examples across all steps and controls.

Prediction: probe accuracy will generally rise with `t`, but not strictly monotonically across every seed and chain bin. Confidence: high. Carry-dependent positions will likely show delayed gains or jumps. Confidence: medium. E5 may look smoother than CE-dynamics, but I would not trust that claim until split/control issues are fixed. Confidence: medium-low.

Parsimony: run D16 first as a clean descriptive experiment: one track, 5 seeds, example-level split, linear probe plus shuffled-label and encoder-context controls. Add nonlinear probes only after the linear result is stable.