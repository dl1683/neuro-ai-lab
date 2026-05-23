**Verdict**

Do not run D8 yet. The core idea is strong, but the current implementation mixes up three different targets: `carry_in[k]`, `carry_out[k]`, and “leftward propagation from k.” That makes the expected wavefront and surgery interpretation invalid as written.

**Correctness Audit**

1. `compute_carry_in()` is mostly correct for incoming carry labels. It computes carry-out right-to-left, then shifts `carry_out[:, 1:]` into `carry_in[:, :-1]` at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:80>). With MSB=0, LSB=3, `carry_in[3]` is always zero.

2. The Phase 1 prediction is wrong for `carry_in`. The docstring expects “position 3 first” at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:26>), but position 3 has no carry-in and will be `N/A`. If probing carry-in, expected wavefront is `pos2 -> pos1 -> pos0`, not `pos3 -> pos0`. If probing carry-out, use a separate `carry_out` label.

3. Phase 2 flips carry-out at `flip_pos`, not carry-in at `flip_pos`. The generator computes rightward carry into `flip_pos`, then changes `a[flip_pos]` so `a_k+b_k+c_in` crosses the base threshold at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:269>). That is a carry-out intervention. The loop then uses `range(half - 1)` and skips position 3 at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:561>), but position 3 is exactly the first carry source. It should likely test carry-out flips at `k=3,2,1`; `k=0` only affects discarded overflow.

4. Phase 2 is heavily confounded because the paired inputs differ in the local digit `a_k`. Any divergence at state position `k` can be ordinary sensitivity to the changed input token, not carry propagation. You need matched controls that change `a_k` by comparable magnitude without changing carry status.

5. The reflection formula itself is correct. For boundary `w·x+b=0`, the code at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:385>) computes `x' = x - 2(w·x+b)/||w||^2 w`.

6. Surgery interpretation is not correct yet. Flipping `carry_in[k]` changes result digit `k` almost always, but it changes leftward positions only when the local sum is exactly on a carry boundary. In base 64, that is roughly 1/64 of examples. So “output should change at k and leftward” at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:19>) is too strong for random examples.

7. `per_pos_correct_change_rate` is misnamed. It measures `preds_surg == result_f` at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:416>), not whether the output changed correctly. Unaffected positions can score high just because they stayed correct.

8. Final persistence uses the probe trained at `t_surg` on the final state at [exp_d8_causal_carry_probing.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d8_causal_carry_probing.py:437>). That can confuse “carry overwritten” with representational rotation. Use the probe for `(T, k)` or train step-specific persistence probes.

**Statistical Plan**

`N=4096` is enough for balanced carry-in probes at positions 0-2: validation has about 819 examples, so accuracy SE is around 1.5-2%. It is not enough for rare leftward propagation from a carry-in flip. Boundary events are only about 1/64, and multi-position propagation is much rarer.

The train/val split is acceptable for quick probe accuracy, but it should be shuffled/stratified and separated from surgery. Right now the same 4096 examples train probes and evaluate causal surgery, which leaks probe fitting into the causal phase.

Use targeted balanced datasets:
- carry-in labels for `pos0..2`;
- carry-out labels for `pos1..3` or `pos0..3` with overflow handled separately;
- boundary cases where flipping carry-in actually changes carry-out;
- held-out surgery examples never used for probe training.

**Alternative Explanations**

A positive wavefront or surgery result could still be non-algorithmic:

- probes decode target digit, local sum, or position-specific readout margin rather than carry;
- the model computes the whole answer globally early, while probes only expose it gradually;
- perturbation divergence reflects changed input token identity, not carry propagation;
- reflection moves states off-manifold and directly damages the readout;
- E5 fixed-point pressure creates smooth monotone margins that look like staged computation;
- decoder self-attention spreads any local state edit to all positions in one step.

Missing controls: random-label probes, random matched-norm surgery directions, non-carry-changing digit perturbations, carry-changing vs non-carry-changing matched pairs, encoder-only/deep-encoder baselines, and carry-out probes distinct from carry-in probes.

**Design Gaps**

D7 and D8 should share checkpoints. Otherwise D7’s timing results and D8’s causal results may describe different learned algorithms. At minimum, save/load the same CE-dynamics and E5 checkpoints by seed, and run D7/D8 probes on those exact models.

Also add baselines: encoder-only 2L, encoder-only 8L, AR, `T=1`, and a no-state/digit-feature probe baseline.

**Prediction**

For correctly specified carry-in probes, I expect:

- CE-dynamics: `pos2` crosses 0.8 around steps 2-4, `pos1` around 4-6, `pos0` around 5-8. More nonmonotonicity and stronger transient state movement.
- E5, successful seed: smoother probes, likely later or more compressed wavefront; `pos2` around 2-4, `pos1` around 4-7, `pos0` around 6-9. Failed E5 seed: probes may remain near chance or reflect wrong attractor structure.
- `pos3` carry-in should be `N/A`, not high accuracy.

For surgery, after fixing the target:
- carry-in surgery should mostly change only digit `k`; leftward changes should be near 1-2% unless boundary-enriched.
- carry-out surgery at `k=3,2,1` should change `k-1` much more reliably.
- CE-dynamics should show more persistence but also more off-manifold damage.
- E5 should overwrite more often if the fixed point is strongly context-derived.

**Priority Directive**

Fix the carry target semantics before running: explicitly separate `carry_in` and `carry_out`, correct the expected wavefront/index loops, and design surgery/perturbation around the carry variable you actually want to test. Everything else depends on that.