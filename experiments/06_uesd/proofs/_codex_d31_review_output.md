I reviewed both files. I did a line-by-line parity check on D31 vs D28 plus execution/logic flow checks.

1. **Primary correctness issue — statistical test is not an actual paired t-test**
   - In `experiments/06_uesd/exp_d31_d8_multiseed.py` lines `346-350`, you compute:
     - `mean_delta = mean(deltas)`
     - `std_delta = np.std(deltas)` (population std, `ddof=0`)
     - `t_stat = mean_delta / (std_delta / sqrt(n))`
   - This is a t-statistic-like quantity, but it is **not the standard paired t-test implementation** (should use sample std `ddof=1` for small `n`, and/or `scipy.stats.ttest_rel` and compare p-value/significance).
   - Severity: **medium** (decision rule impact).
   - Also note line `355` vs `358-360` uses ad-hoc one-sided thresholds (`>2` or `<-2` plus fixed magnitude gate), not a calibrated two-sided significance framework.

2. **Adjudication rule is heuristic and can conflict with t-test output**
   - `experiments/06_uesd/exp_d31_d8_multiseed.py` lines `386-388`:
     - Verdict is `PHASE_BOUNDARY_REAL` if `pos_count >= 6`, `SEED_ANOMALY` if `neg_count >= 6`.
   - This is a hard majority rule on signs, independent of effect size/SE. It can override statistical ambiguity from the paired delta distribution.
   - Severity: **medium** (may overstate/discard weak evidence).

3. **Seed argument in spectral measurement is unused**
   - `measure_spectral_radius(..., seed)` takes `seed` at line `131` and `238`, but never uses it.
   - Not a runtime bug (still matches D28 behavior using fixed 9999), but it is dead API surface and potentially misleading.
   - Severity: **low**.

4. **Potential duplication in seeding calls (not incorrect, but redundant)**
   - `run_one_seed` seeds at line `229`, then `train_model` reseeds at `89`.
   - This still gives independent runs by seed, but it is redundant and can confuse intent (“seed before init and before train” is true, but the second reseed is unnecessary once model is already initialized).
   - Severity: **low**.

5. **Unused import**
   - `count_params` imported at line `30` but never used in D31.
   - Severity: **low**.

What looks correct:

6. **Seed isolation**
   - Per-run seeding happens before both model init and training call path (`229` then `89`), and eval measurements also use fixed `9999` (`133` and `173`).
   - This satisfies “independent per-seed training” and “fixed eval seed.”

7. **Measurement consistency with D28**
   - `full_seed(9999)` is used in contraction and spectral routines.
   - Contraction/spectral formulas and hyperparameters appear aligned to D28 (`FP_T`, trajectory loop, power-iteration dimensions), including mean-k windowing over `k_values[1:10]` (same stable-step intent as D28).
   - Key lines:
     - D31: `measure_contraction_summary` `173`, `184-199`, `202-203`, `208-214`.
     - D28: `measure_contraction` analogous blocks at `142-151`, `176-204`, `249-258`, `260-264`.
     - D31: `measure_spectral_radius` `133-166` and D28: `294-331`.
   - No major shape mismatch or off-by-one regression found here.

8. **Resume logic**
   - Resume keying is coherent: `D{carry_depth}_{variant}_s{seed}` at lines `266-268`; completed runs are skipped with exact matching keys at `305-308`.
   - This is correct for the current config set (no carry-depth collisions across intended experiments).
   - Minor robustness caveat: malformed/legacy JSON missing `carry_depth`/`seed`/`variant` would crash on resume, but that would be a stale-checkpoint compatibility issue.

9. **Resource/time**
   - The run loop is exactly 28 runs by construction (`287-289`, with CONFIGS `[8+3+3] * 2 = 28`).
   - No runtime guard/ETA check is present, so 3 min/run implies ~84 min minimum plus measurement overhead and startup I/O; expected in code.

If you want, I can propose a minimal patch for the stats block only (proper paired t-test + clear one-sided/two-sided decision criteria) without touching training/measurement code.