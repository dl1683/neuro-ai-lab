Findings (correctness issues that can change conclusions):

1. **High — seed handling does not control model initialization**  
   In `run_one_config`, the model is created before `full_seed(seed)` is called, so `seed` affects training data/`T` sequence but not `UESDModel` weights.  
   - The initialization call happens at `[exp_d27_encoder_degradation.py:274]` (model creation)  
   - Seed reset happens later at `[exp_d27_encoder_degradation.py:96]` (inside `train_variable_t`)  
   This breaks the intended 3-seed protocol for exact reproducibility and makes runs for different seeds incomparable by initialization effects.

2. **Medium — cross-attention ablation is not a true “hard off”**  
   Zeroing context in `evaluate_no_crossattn_reread` does suppress content rereads, but `TransformerDecoderLayer` cross-attention can still contribute non-zero output because zero keys/values still pass through linear projections with biases.  
   - Zeroing occurs at `[exp_d27_encoder_degradation.py:190]`  
   - This tests “low-content reread,” not a true architectural ablation of cross-attn.  
   You can still see a meaningful drop, but if you want strict removal, this is not airtight.

3. **Low/medium — noise scaling is per-position, not fixed-per-element/global**  
   `noise = randn_like(context) * (sigma * context.norm(dim=-1, keepdim=True))` scales noise by each vector’s norm, so each token position gets a different effective sigma.  
   - [exp_d27_encoder_degradation.py:144], [exp_d27_encoder_degradation.py:180], [exp_d27_encoder_degradation.py:215]  
   If the intended protocol is a fixed AWGN SNR, this is a protocol-choice issue and can change the interpretation of “sigma.”

4. **Medium — Nishimori-style metric likely not aligned with the theory target**  
   `measure_inter_position_correlation` computes cosine on L2-normalized hidden states `[B, half, D]` and reports adjacent position cosine in latent space, not directly on decoded probabilities or marginals.  
   - [exp_d27_encoder_degradation.py:214], [exp_d27_encoder_degradation.py:221], [exp_d27_encoder_degradation.py:228]  
   If Nishimori prediction is meant for posterior/codeword-overlap quantities, this may be a proxy rather than the target statistic.

5. **Low — checkpoint resume keying is coarse**  
   Resume skips by `(L, seed)` only. If you change any config field (e.g., eval set, noise grid, model setup) while reusing the same result file, stale runs can be treated as complete.  
   - [exp_d27_encoder_degradation.py:381–386]

What is correct/good:

- `encode`, `init_state`, `dynamics_step`, and `readout_logits` usage is consistent with `UESDModel` API.  
  - `[shared/model.py:37]`, `[shared/model.py:41]`, `[shared/model.py:44]`, `[shared/model.py:50]`, `[shared/model.py:67]`
- `generate_batch` receives explicit `seq_len` in all callsites and handles L=8,12,24 correctly for addition’s even-length case (first `half` outputs are the target).  
  - `[exp_d27_encoder_degradation.py:105]`, `[exp_d27_encoder_degradation.py:137]`, `[shared/data.py:22]`, `[shared/data.py:34]`, `[shared/data.py:39]`

`dynamics_step` with zero context:
- It should not error on zero context; shape/broadcasting are valid.  
- It does not remove cross-attention logic, only removes informative memory content (partially).

Memory and runtime on 24GB GPU:

- Model is small: 694,016 params for `D_MODEL=128`, `N_ENC_LAYERS=2` with this architecture.  
  - Measured one training-like forward/backward memory peak on this machine: roughly ~1GB max at L=24, batch 256.
- With batch 256 and avg `T≈10`, measured local training step time is about **~86 ms/step**, so one 20k-step run is ~**29 min**.  
- 9 runs total (L={8,12,24} × 3 seeds): ~**4.3 hours** of training, plus evaluations (low single-digit minutes).  
  So overall roughly **~4.5–5h** on this hardware class.

Summary of blocking issues to fix before launch:
1. Move `full_seed(seed)` to before `make_model(...)` in each run.
2. Clarify/replace cross-attn ablation to truly disable cross-attention block if that is the hypothesis.
3. Decide whether noise is intended per-position AWGN-scaled-by-norm or fixed global/per-element AWGN and document it explicitly in code.