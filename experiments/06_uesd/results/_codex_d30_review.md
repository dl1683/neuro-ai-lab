High-level: the script is structurally aligned with D23/D28, but there are 2 correctness issues that can make the final Prop-32 verdict wrong.

- **HIGH** — `T_99` failures can be silently ignored in Prop-32 tests.  
  - `compute_t99` returns `None` when no ablation point reaches 0.99 (`[207-212](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L207)`), but those `None` entries are dropped before checks (`[304-313](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L304)`).  
  - `vt_pass` and monotonicity use `t99_values` after filtering out `None` (`[319-325](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L319)`), so a non-converging config can be excluded and the verdict can incorrectly pass (`all([]) == True`).  
  - **Impact**: false `PASS`/`CONFIRMED` possible.

- **HIGH** — `VERDICT` can become `CONFIRMED` without any finite `T_99` evidence for variable-T configs.  
  - If all VT configs have `None`, both `vt_pass` and monotonic checks are vacuously true due filtering. This is the same root cause above but more directly affects overall verdict (`[319-337](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L319)`).

- **MEDIUM** — `EVAL_T_VALUES` grid is too coarse for robust detection of `T_99` transitions and can miss threshold crossings.  
  - Only `[1,2,3,4,5,6,7,8,10,12,15,20]` are tested (`[70,158-166](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L70)`).  
  - Any true transition at gaps (e.g., 9, 11, 13, 14, 16–19) is unobservable; monotonicity claims across `T_min` are then under-resolved.

- **MEDIUM** — `compute_t99` uses rounded accuracy, which can fabricate a threshold hit.  
  - `evaluate_step_ablation` stores rounded values (`[164-166](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L164)`), then `compute_t99` compares against `0.99` (`[209-211](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L209)`).  
  - A true `0.98995` can round to `0.99` and be counted as pass.

- **LOW** — fixed-T baseline is handled fine, but not explicitly tested in Prop-32 logic beyond exclusion.  
  - Baseline is configured and trained correctly with `t_range=None -> FIXED_T=10` (`[66-73,118-121,242-243](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L66)`), yet `vt_pass` excludes `"fixed"` and does not validate the scripted baseline expectation (`[319](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L319)`).

- **LOW** — Numerical/metric note on spectral radius: `D30` matches `D28` exactly, so no new implementation bug there, but it is using a sampled per-token norm update rather than a flattened-state Jacobian norm.  
  - `measure_spectral_radius` is the same pattern as D28 (`[170-204](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d30_tmin_control.py#L170)` vs `[294-331](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d28_contraction_ratio.py#L294)`).  
  - If the intent is a strict global spectral norm estimate, this is a methodological limitation to flag.

Everything else in training-loop mechanics looks aligned with D23/D28 patterns:
- variable-T sampling and fixed-T branch are correct,
- checkpoint resume load/append flow is coherent,
- and no immediate crash paths are obvious from static inspection.