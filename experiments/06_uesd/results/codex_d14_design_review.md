**Verdict**

Do not run D14 as the final scaling-law experiment yet. It is a reasonable pilot, but it cannot support the stated scaling, critical-depth, or accuracy-per-FLOP claims as written.

**Top Findings**

1. **Training objective is wrong for addition.**  
   D14 trains CE over all `seq_len` positions, but addition targets only the first half and pads the second half with zeros. See [exp_d14_scaling_laws.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d14_scaling_laws.py:118) and [data.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/data.py:44). Evaluation scores only the first half. This lets half the loss be an easy padding objective and contaminates every scaling comparison.

2. **FLOP estimates are not accurate enough for accuracy-per-FLOP claims.**  
   Decoder dynamics are undercounted: a decoder layer has self-attn, cross-attn, and FFN, but D14 uses `6 * L * d^2 + 2 * L * d * d_ff` per iteration [exp_d14_scaling_laws.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d14_scaling_laws.py:85). It should be closer to `8 * L * d^2 + 2 * L * d * d_ff` before attention `L^2 d`, layernorm, spectral-norm overhead, and backward pass. Encoder estimates are rough but less wrong. Current FLOPs are okay as labels, not as evidence.

3. **The T sweep confounds inference depth with total training compute.**  
   Every T gets 20K optimizer steps [exp_d14_scaling_laws.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d14_scaling_laws.py:230), so higher-T models receive more forward/backward compute. A rising T curve could be “more training compute per update,” not “more thinking depth.” Need both fixed-step and fixed-training-FLOP comparisons, plus train-to-plateau checks.

4. **One seed cannot support scaling-law claims.**  
   `SEED = 42` is used throughout [exp_d14_scaling_laws.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d14_scaling_laws.py:51). Use at least 5 seeds for final claims; 3 is acceptable only for a pilot. Report seed-level CIs, not just eval-binomial CIs.

5. **D14 does not measure the key D7 mechanism.**  
   The prediction is about critical T versus carry-chain length, but D14 only reports aggregate token/sequence accuracy. D7 already has carry-depth and first-stable-correct machinery [exp_d7_thinking_emergence.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d7_thinking_emergence.py:116), [exp_d7_thinking_emergence.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d7_thinking_emergence.py:244). D14 should stratify accuracy and first-stable step by carry depth on the same eval batches.

6. **Encoder-only baselines are not compute-matched or architecture-matched.**  
   Encoder baselines are only `{2,4,8}` layers at `d=128,L=8` [exp_d14_scaling_laws.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d14_scaling_laws.py:321). They are not matched to UESD FLOPs, params, training compute, or long-carry `L=12/16`. Add compute-matched encoder grids, untied decoder-depth baselines, and ideally AR/Universal Transformer-style baselines.

**Statistical Plan**

Use 5 seeds minimum. For expensive configs, run a 3-seed pilot, prune, then 5 seeds on the final grid.

20K steps is not established. Train until validation curves plateau, or run fixed budgets like `{5K, 20K, 80K}` for representative configs.

For T scaling: fit sigmoid/threshold models with seed bootstrap CIs for `T*`, and compare against monotone/log-linear alternatives by AIC/BIC or held-out likelihood.

For d scaling: three widths are not enough for a power law. Add more widths or downgrade the claim. Fit loss/error, not raw accuracy near ceiling.

For pairwise UESD vs encoder claims: use paired eval batches and McNemar or mixed-effects logistic regression over per-example correctness, with seed as a random effect.

**Alternative Explanations**

T scaling could reflect more training/inference compute, not iterative reasoning.

d scaling could be small-width capacity failure plus fixed-step undertraining, not a clean power law.

Encoder-only may be disadvantaged by missing decoder state/cross-attn, or advantaged by full self-attention. Current comparison cannot identify either.

**Priority Directive**

Fix the loss first: train CE only on result positions. No amount of seeds or plotting rescues a scaling-law experiment trained half on padded zeros.

**Predictions**

As written, T will probably improve monotonically, but the sigmoid/critical-T interpretation will be weak. Confidence: high.

`d=256` will likely beat `d=128` and `d=64`, but not cleanly enough to claim a power law. Confidence: medium-high.

UESD may beat encoder-only at some L=8 points, but the accuracy-per-FLOP claim will not survive stricter compute matching without more controls. Confidence: medium.

L=16 will expose degradation, especially at low T. Confidence: medium-high.

**Parsimony**

Simplify D14 to: masked loss, `d=128`, `T={2,5,10,20}`, `L={8,16}`, 5 seeds, D7 carry-depth stratification, and one compute-matched encoder/untied baseline grid. Drop the power-law claim until there are enough widths and training budgets to fit it.