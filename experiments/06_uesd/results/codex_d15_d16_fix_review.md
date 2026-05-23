**D15 Verdict: NEEDS MORE FIXES**

The critical correctness fixes were made correctly.

Original issue: D15 bypassed the trained readout. Fixed. `_readout_probs` now matches `UESDModel.readout_logits`:

```python
h = model.readout_proj(s[:, :half, :])
W = model.tok_emb.weight
h = F.normalize(h, dim=-1)
W = F.normalize(W, dim=-1)
logits = torch.matmul(h, W.t()) / tau
```

This matches [model.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:56):

```python
h = self.readout_proj(s)
...
return torch.matmul(h, W.t()) / self.tau
```

Original issue: confidence used true-label probability instead of predicted-class confidence. Fixed. D15 now uses:

```python
max_probs, preds = probs.max(dim=-1)
correct = (preds == targets).float()
conf_flat = max_probs.reshape(-1).cpu().numpy()
```

Original issue: tau sweep could manufacture `rho ~= 0.462`. Partly fixed. Primary analysis uses model tau, and sweep is explicitly secondary:

```python
cal = calibration_analysis(model, eval_src, eval_tgt, config, device)
...
# Secondary: tau sweep (sensitivity analysis, not primary evidence)
```

Remaining problem: the experiment still defines `t*` only by closeness to Nishimori rho:

```python
rho_dists = [r["rho_distance"] for r in step_results]
t_star = int(np.argmin(rho_dists))
calib_at_tstar = step_results[t_star]["ece"]
```

That does not test the stronger claim from the review: whether the ECE-minimizing step also has confidence near `0.462`. It reports ECE at the rho-nearest step, but does not compare to `argmin(ECE)` or controls.

Other remaining concerns:
- Still one training seed: `build_config(seed=42)`.
- No bootstrap CIs or seed-level CIs.
- Shuffled-label and encoder controls were added, good, but untrained/random-readout controls are still missing.
- The theory gap remains: this is still token-softmax calibration on addition, not a direct continuous-error-space Nishimori observable.

Single required fix before I would call it ready: add `t_ece_min`, `rho_at_ece_min`, and the same statistic for shuffled/encoder/untrained controls, with multi-seed CIs.

**D16 Verdict: NEEDS MORE FIXES**

The prior critical split bug was fixed correctly.

Original issue: token-level split leaked positions from the same example across train/test. Fixed. The split is now example-level:

```python
perm = torch.randperm(B, generator=rng)
n_train = int(train_frac * B)
return perm[:n_train], perm[n_train:]
```

and probes flatten only after indexing examples:

```python
train_feat = states_t[train_idx, :half, :].reshape(-1, d_model)
train_labels = targets[train_idx, :half].reshape(-1)
test_feat = states_t[test_idx, :half, :].reshape(-1, d_model)
test_labels = targets[test_idx, :half].reshape(-1)
```

Good fix: the same split is shared across all steps:

```python
train_idx, test_idx = make_example_split(eval_src.shape[0])
...
for t in range(T + 1):
    probe_acc, probe_per_pos = train_linear_probe(...)
```

Good fix: direct readout trajectory was added:

```python
logits = model.readout_logits(s_t)
preds = logits[:, :half, :].argmax(dim=-1)
```

Good fix: shuffled-label controls were added, but only at `t=0` and `t=T`:

```python
for t in [0, T]:
    sh_acc, sh_pos = train_shuffled_probe(...)
```

Remaining issues:
- Still one model seed: `seed=42`.
- No validation curve, regularization, early stopping, or probe-seed repeats, so probe training variance/overfit is still unquantified.
- Shuffled controls should run at every `t`, or at least the same reported step subset, because the claim is trajectory shape.
- Missing controls from the original review remain: encoder context probe, raw input-derived features, random model states, and optional nonlinear probe upper bound.
- The monotonicity check is exact and brittle:

```python
monotonic = all(probe_accs[i] <= probe_accs[i + 1] ...)
```

It should use CIs or a tolerance; otherwise tiny probe noise can flip the headline result.

No major new correctness bug introduced. The implementation is materially better, but still not complete enough for an A4/information-accumulation claim.

Single required fix before ready: add multi-seed probe evaluation with held-out CIs and shuffled controls across the full trajectory.