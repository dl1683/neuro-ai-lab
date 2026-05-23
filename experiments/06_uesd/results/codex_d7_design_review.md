**Verdict**

D7 is a good experiment idea, but the current implementation has one central validity bug: the carry-chain variable does not measure the carry dependency of the result position it is correlated against. Fix that before running.

**Major Findings**

1. **Carry pattern is off by one.**  
In [exp_d7_thinking_emergence.py](<C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d7_thinking_emergence.py:116>), `carries[:, i] = carry` stores the carry *out* of position `i`, not the carry *in* to position `i`. But the hypothesis is about whether result digit `i` depends on carries from the right. The rightmost digit should always have carry-in length 0; the current code can label it as chain length 1 if it generates a carry leftward.

2. **The chain-length statistic is therefore not measuring what D7 claims.**  
The current chain loop at [exp_d7_thinking_emergence.py](<C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d7_thinking_emergence.py:139>) counts consecutive carry-outs ending at the current position. For result-position difficulty, you want incoming carry-chain length into that position: position 2 should get length 1 when position 3 emits a carry; currently it often gets 0.

3. **“First correct step” is actually “first stable-correct step.”**  
At [exp_d7_thinking_emergence.py](<C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d7_thinking_emergence.py:237>), `all_correct[:, :, t:].all(dim=2)` requires correctness from `t` through `T`. That is useful, but it is not first time correct. Report both `first_hit_step` and `first_stable_step`; otherwise transition dynamics and transient readout behavior get collapsed.

4. **Global correlation is confounded by position.**  
Carry-chain length is structurally tied to position: rightmost position cannot have incoming carry, leftmost can have the longest chain. A positive `corr(chain_length, first_correct_step)` can mostly reflect “left positions are harder,” not carry-chain computation. Use within-position tests or a regression controlling for position.

5. **Single seed is not enough for a mechanism claim.**  
Prior D-series results showed strong seed dependence, especially for E5. N=4096 eval examples gives tight estimates for one trained model, but it does not establish that CE-dynamics or E5 generally exhibit a wavefront. Use at least the D2b seed set if this becomes a claim.

**Statistical Plan**

N=4096 is enough for per-model/per-position estimates, including rare length-3 chains, but the unit of inference should be examples clustered within trained seed.

Use:
- Example-level bootstrap CIs for per-position accuracy curves.
- Position-stratified permutation test: shuffle carry-chain labels within each position.
- Mixed/ordinal model: `first_stable_step ~ chain_length + position + track + chain_length:track`, clustered by example and seed.
- Survival-style treatment for never-correct positions instead of dropping them from the correlation at [exp_d7_thinking_emergence.py](<C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d7_thinking_emergence.py:462>).

**Alternative Explanations**

A correlation would not automatically prove “thinking.” It could be:

- Position-specific readout calibration.
- Final readout head becoming meaningful only near late states.
- Local digit difficulty, target digit distribution, or modulo wrap difficulty.
- The model solving carry globally early while the shared readout exposes it late.
- E5 fixed-point pressure creating monotonic margins without stepwise algorithmic computation.

To rule these out, add no-carry/all-carry controlled slices, position-stratified carry tests, per-step trained linear probes, causal input interventions that flip only lower-order carry status, and a one-step/deep-encoder baseline probed the same way.

**Prediction**

CE-dynamics will likely show more visible intermediate movement and more transient correct-then-wrong behavior, because prior D3/D4 results suggest finite-horizon, non-convergent “scattered” dynamics. E5 should show smoother, more stable margins once it succeeds, but possibly a weaker wavefront if SC pulls the trajectory toward a whole-answer fixed point rather than exposing clean digitwise stages.

I would expect both to show some right-to-left structure, but I would trust it only if it survives position-stratified carry-in analysis. E5 may look cleaner in `first_stable_step`; CE-dynamics may look stronger in `first_hit_step`.

**Priority Directive**

Fix `compute_carry_pattern` so it returns incoming carry and incoming carry-chain length per result position. Until that is corrected, D7’s central carry-chain correlation is not interpretable.