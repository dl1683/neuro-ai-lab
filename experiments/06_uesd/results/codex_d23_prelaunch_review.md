### CRITICAL

1. **No durable checkpointing; crash loses all completed work**
- Location: [exp_d23_carry_depth_scaling.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d23_carry_depth_scaling.py)
- Why it is critical: `json.dump(...)` is done only once at the end (near the final save block). If the job dies at L=20 (as happened), all encoder baselines and all completed `(L, variant)` runs are lost.
- Exact fix needed:
  - Write results after each config completes.
  - Use atomic writes (`tempfile.NamedTemporaryFile` + `os.replace`) after each `encoder_baselines[L]` write and each `runs.append(result)`.
  - On startup, support resume by loading existing `results` file and skipping already-finished `seq_len`/`variant` entries.

---

### WARNING

2. **Variable-T schedules are likely too narrow for L=20/24**
- Location: [exp_d23_carry_depth_scaling.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d23_carry_depth_scaling.py)
- `train_variable_t` and `train_combined` use `t_range=[4,6,8,10,12,14,16]` for all L, while evaluation probes up to `48`.
- If depth-compute scaling is the hypothesis, this can under-train large-L configurations and bias the phase-diagram result.
- Fix: scale `t_range` with `seq_len`/carry-depth, and ensure it includes values near/equal to evaluation max for large L (or at least includes 20+ for L≥20).

3. **`train_combined` small-T injection behavior is valid now but brittle**
- Location: [exp_d23_carry_depth_scaling.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments/06_uesd/exp_d23_carry_depth_scaling.py)
- `inject_step = random.choice([t for t in denoise_steps if t < T])` is fine for current `denoise_steps` + `t_range`, but this silently relies on a non-empty filtered list.
- Add an explicit guard (`assert`) or fallback for future changes to t-lists/values.

4. **Reproducibility gap for combined/denoising Variable-T randomness**
- Same file.
- `random.choice` uses Python `random` but only Torch RNG is seeded (`set_seed`), so runs are not fully deterministic across process restarts.
- Fix: seed `random` alongside torch (`random.seed(SEED)`).

5. **Metric precision is intentionally reduced before post-analysis**
- Same file.
- `evaluate_step_ablation` and `evaluate_recovery` round values before storing, then post-hoc analysis computes windows on rounded values. This can shift thresholded metrics (e.g., `>=0.99`) at coarse precision.
- Fix: store full float metrics in `all_results`, and round only for display/logging.

---

### NOTE

6. **Gradient flow is okay**
- In all 4 training paths (`train_baseline`, `train_denoising`, `train_variable_t`, `train_combined`) `loss.backward()` is valid:
  - Baseline: `model(src, T)` path is differentiable.
  - Denoising/combined: noise injection uses detached scale + stochastic tensor; state `s` still carries graph history.
- `UESDModel`/`EncoderOnlyAblation` in [model.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments/06_uesd/shared/model.py) do not detach logits/state before loss.

7. **Config/data sanity is mostly okay**
- `MAX_LEN=64` is above all `SEQ_LENS` (max 24), so positional embeddings are safe.
- `TRAINING_STEPS=30000` is uniform as intended by script design.
- `EVAL_T_VALUES` up to 48 is valid for `model(src, T)`.

8. **Resource assessment**
- `BATCH_SIZE=256` with `D_MODEL=128`, `N_HEADS=4`, `L<=24` is likely feasible on 32GB 5090 for 10–16 step training; `EVAL_SAMPLES=4096` + `T=48` is the heavier part and the most likely runtime/memory stress point.
- If memory spikes appear, chunk evaluation (`EVAL_SAMPLES` looped in smaller batches) to reduce OOM risk without changing results.