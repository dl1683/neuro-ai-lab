I found no `CLAUDE.md` or `.claude/CLAUDE.md` in the workspace, so this is based on the experiment files only.

**Verdict:** the small-model result supports the attractor story, but it does not “confirm” it in the strong sense yet. It is exactly the right kind of null-test evidence: FT starts near the proposed VT band, and `dk` collapses to ~0 instead of staying negative. But this is one seed, one architecture, same estimator, same task, and the observed `+0.0004` is tiny relative to reported within-run `k` stds (`FT std_k=0.0038`, `VT std_k=0.0013`). Treat it as a successful first null-test probe, not a completed confirmation.

Alternative explanations remain live:

- estimator floor/bias around `k≈0.989`;
- seed noise;
- small architecture undercapacity changing the learned dynamics;
- different convergence/optimization regime rather than a true VT setpoint;
- `k` being insensitive to the same directions where `rho` changes massively.

The `rho` result is the more surprising part. `FT rho=1.0364` says the small FT model learned a much more locally expansive fixed-point map than the d=128 baseline. VT cuts that to `1.0142`, which is a large stabilizing move, but note: `1.0142` is still far above the previous d=128 “VT ceiling” around `1.002`. So the mechanism is not “VT always pins rho near 1.002.” Better wording:

> VT regularizes spectral radius strongly when the FT solution becomes highly expansive, but the achieved rho level is architecture/regime dependent.

The `k/rho` dissociation is real here: `k` barely moves while `rho` drops by `0.022`. That suggests VT is not simply lowering one global contraction knob. It may be suppressing worst-case local Jacobian directions while leaving average trajectory contraction nearly unchanged. On the small architecture, FT likely uses sharper or more strained dynamics because capacity is tighter; VT’s multi-T objective penalizes those expansive local modes more heavily.

On the VT setpoint: `VT k=0.9894` being 2.6 sigma above the d=128 setpoint is suggestive, not decisive. Comparing one small-model seed against a d=128 cluster overstates precision. The setpoint is probably determined by the whole stack: architecture width, head dimension, residual/LN geometry, train-T distribution, optimizer, task difficulty, training horizon, and the k estimator itself. D36 can distinguish “weak architecture dependence” from noise only after architecture-level means are available.

Predictions for remaining D36 configs:

- **baseline d=128/h=4:** should reproduce prior D=6 behavior: VT `k≈0.988`, negative `dk≈-0.002` to `-0.003`, small rho suppression, `T_99≈3`.
- **large d=256/h=8:** if capacity reduces FT strain, FT `rho` should be much lower than small; VT rho suppression should shrink. VT k may land near or slightly below d=128 if the setpoint scales with width.
- **many_heads d=128/h=8/head_dim=16:** watch for head-dim effects. Smaller head_dim could increase optimization strain; possible higher FT rho and a VT k shift upward.
- **few_heads d=128/h=2/head_dim=64:** may be closer to baseline or more stable if wider heads help carry structure. Prediction: VT k near d=128 band unless head count itself matters.
- **small seed 137:** most important immediate check. If VT k stays near `0.989` and `dk≈0` again, architecture dependence becomes much more credible. If it returns to `0.988`, seed noise/estimator becomes the simpler explanation.

Confidence for **T5: VT k-contraction** should **not increase** yet under the anti-overconfidence protocol. This is specific new evidence for the attractor/null-test version of the claim, but it is one pair and the positive `dk` is within noise scale. I would keep T5 at **8.8/10**, with revised wording:

> VT k-contraction is strongly supported under the current metric stack; the D36 small pair supports a setpoint/attractor interpretation, but architecture dependence and estimator artifacts remain unresolved.