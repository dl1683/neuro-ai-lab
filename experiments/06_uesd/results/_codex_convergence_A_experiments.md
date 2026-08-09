**Bottom Line**

The experiments say convergence is not the hard part anymore. **Attractor selection is.** VT reliably shapes the global flow so trajectories reach a readout-stable basin faster, but neither `k<1`, low residual, nor `rho≈1` identifies whether that basin decodes correctly.

The most promising fix is: **train CE/readout correctness first, then apply self-consistency only inside already-correct/high-margin basins.** E5 failed because SC competed too early and made wrong basins stable.

**Key Clues**

1. **CE dynamics solve correctness but not fixed points.**
   In D2b, CE-dynamics solved addition 5/5 seeds with mean seq acc `0.9999`, std `0.0002`. E5 solved 4/5, with seed 512 stuck at wrong attractor: token acc `0.5074`, seq acc `0.0`, final loss `2.081`. D2c then showed CE has `converged_frac=0.0` but `99.9-100%` accuracy, while E5 converges but retains wrong-attractor risk. So CE learns the right basin trajectory; SC learns stopping.

2. **SC creates premature basin freezing.**
   D2/D2b/D2c are the cleanest warning. The only difference between CE-dynamics and E5 is the SC term, and that term introduces the 20% failure mode. D2c also shows E5 has higher non-normality: kappa `1.85-2.12` vs CE `1.45-1.57`. This suggests SC does not just “stabilize”; it distorts the flow and can stabilize the wrong manifold.

3. **`k` is global basin-channeling, not local stability.**
   D37 is decisive here. For baseline d=128 D8:
   - standard `k`: FT `0.9910`, VT `0.9882`, `dk=-0.0029`
   - random-dir `k`: FT `0.9947`, VT `0.9950`, basically neutral
   - pairwise `k`: FT `1.0144`, VT `1.0266`, `dk=+0.0122`

   That means VT improves convergence from the actual initialization toward the task basin while making inter-trajectory separation larger. It is not simple local contraction.

4. **The k/rho dissociation says there are two independent knobs.**
   D35b prefix sum has strong `dk<0` while `drho>0`: overall `dk≈-0.0053`, but `drho=+0.0018`, `p=0.001137`. D10 addition similarly has some `dk<0` with weak or positive `drho`. So `k` tracks solver-path convergence; `rho` tracks worst-direction local spectral behavior. Correctness lives in neither scalar alone.

5. **VT has a setpoint, not a universal push-down law.**
   D36 small d=64 is the important null. When FT `k` was already at/below the VT setpoint, VT did not push lower: small architecture mean FT `k=0.9884`, VT `k=0.9892`, `dk=+0.0008`, while `rho` dropped hard from `1.0304` to `1.0118`. This supports a VT attractor/setpoint around `0.988-0.989`, with weak architecture dependence. It also means VT is a regularizer, not a correctness guarantee.

6. **Readout directions can be stable while the full state is not.**
   D29b found margin-critical/readout-direct FTLE negative while null directions are positive: at T=5, `lambda_R_direct=-0.004306`, `lambda_null_direct=+0.04031`. That explains why `rho>1` and local expansion can coexist with correct outputs. The model only needs the readout-critical subspace to settle.

7. **Noise after convergence is not a rescue mechanism.**
   D12 Langevin helped little because the tested E5 runs were already almost correct. At `tau=0.05`, single-sample E5 dropped badly (`0.7368` or `0.8701` seq acc), while majority vote recovered near-correct (`0.9912`, `0.9956`). D21 noise injection after convergence often worsened WA: E5 at sigma `1.0` had `WA_at_0=0.4780`, `WA_at_20=0.8645`; CE had `15.5%` WA after 20 extra steps even at sigma `0.01`. Use noise before commitment or as multi-start basin sampling, not after fixation.

**Implication For Solving Convergence**

The solution should not be “make `k` smaller.” It should be **two-stage basin training**:

- Stage 1: train CE/contrastive readout to make correct basins large and high-margin.
- Stage 2: add SC only when the current trajectory is already correct/high-margin, or ramp SC based on margin.
- Penalize low-residual wrong states directly: `SC * wrong_margin` should be expensive, not rewarded.
- Use VT as basin-shaping, since it reliably lowers operational `k` and `T_99`.
- Add a verifier/judge or contrastive basin selector for multi-start trajectories; LSR’s basin-sampling work points in this direction.
- Measure success with readout-critical FTLE, margin, basin identity, and wrong-attractor rate, not residual alone.

External scan reinforced this: `_meta` and Open Exploration emphasize criticality/calibration, not blind damping; LSR repeatedly found perturbations work by switching basins; Rosetta/platonic work warns that global geometry metrics can be misleading without behavioral validation. The convergence fix is therefore **calibrated basin selection plus delayed stabilization**, not stronger contraction.

