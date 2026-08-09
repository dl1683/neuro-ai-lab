**UESD Convergence Blueprint**

**1. Diagnosis**

The root cause is not insufficient convergence. It is **uncoupled attractor selection**.

Self-consistency loss rewards `||G(s,c)-s|| -> 0` regardless of what the fixed point means. That creates stable wrong basins. CE training solves correctness because it directly shapes the readout boundary, but it does not require the endpoint to become a fixed point. So the gap is:

```text
CE: correct endpoints, weak fixed points
SC/E5: strong fixed points, weak semantic selection
```

All seven analyses converge on the same diagnosis:

- `k < 1` only proves convergence inside the basin already selected.
- Low residual is not evidence of correctness.
- Wrong attractors are low-residual, negative-margin states.
- Stronger contraction can make wrong basins stickier.
- Correctness must be coupled through margin, contrastive basin shaping, perturbation recovery, and readout-aware validation.
- Noise helps only before commitment and only with selection.
- The readout/interface is not harmless; correctness is defined relative to it.

The winning formulation is:

```text
Correct convergence = low residual + positive readout margin + recovery basin + wrong-attractor repulsion
```

**2. The Solution**

Build a **Basin-Coupled UESD with Rectified-Flow Correction**.

This is the single best path because it preserves the proven UESD convergence machinery while adding the missing correctness mechanism.

Architecture:

```text
prompt
 -> ContextEncoder
 -> learned output slots s0
 -> tied UESD compute map G, applied 3-5 times
 -> h = G^T(s0,c)
 -> small conditional rectified-flow corrector H_phi(z,h,c,t)
 -> final latent z0_hat
 -> readout R(z0_hat)
```

Interpretation:

- `G` is the latent computation engine.
- `H_phi` is the manifold/correctness projector.
- `R` is readout only, but its margin enters training and acceptance.
- Self-consistency is delayed and margin-gated, not used as the main objective from step zero.

Why this wins over alternatives:

- Pure E5 loses because it stabilizes wrong states.
- Pure CE loses because it does not produce fixed points.
- Full monotone/convex DEQ is clean but likely underexpressive.
- Full diffusion is heavier than needed.
- Patch/byte hierarchy is right for scaling, but not the first fix.
- Promotion-only gating detects wrong attractors but does not reshape them.

The core loss should be:

```text
L =
  L_CE
+ lambda_flow   L_flow
+ lambda_sc     ||G(h,c)-h||^2
+ lambda_fp     softplus(K*r/(1-k_hat) + gamma - margin)
+ lambda_rec    L_recovery
+ lambda_bad    L_wrong_attractor
+ lambda_contract L_local_contract
```

Where:

```text
r = ||G(h,c)-h||
margin = logit_y - max_{z != y} logit_z
```

The fixed-point term is only good when residual is small relative to margin. Wrong fixed points are no longer rewarded.

**3. Implementation Plan**

1. **Warm-start with CE dynamics**

Train the existing UESD dynamics using endpoint CE only:

```text
h = G^T(s0,c)
L = CE(R(h), y)
```

Goal: recover CE behavior: near-100% accuracy, even if `converged_frac=0`.

2. **Add margin-gated self-consistency**

Continue training from the CE checkpoint. Add:

```text
L_sc = ||G(h,c)-h||^2
L_fp = softplus(K*r/(1-k_hat) + gamma - margin)
```

Ramp `lambda_sc` slowly. Do not let SC dominate while margins are weak.

3. **Add rectified-flow corrector**

Train a small flow head over target output embeddings.

```text
eps ~ N(0,I)
z_t = (1-t) * y_embed + t * eps
target_velocity = y_embed - eps
L_flow = ||H_phi(z_t,h,c,t) - target_velocity||^2
```

Inference:

```text
h = G^T(s0,c)
z = h or h + sigma*eps
for k in K..1:
    t = k/K
    z = z - (1/K) * H_phi(z,h,c,t)
decode R(z)
```

Use `K=4` initially, `K=8` for low-margin cases.

4. **Mine wrong attractors**

During training, roll out from perturbed states and collect endpoints with:

```text
residual low
margin negative or wrong decoded sequence
```

Use them as hard negatives:

```text
L_bad = softplus(gamma - margin(s_bad)) * exp(-r_bad^2 / sigma^2)
```

This directly attacks the 20% failure mode.

5. **Damage-repair training**

Perturb correct endpoints and known wrong endpoints, then train recovery:

```text
L_rec = CE(R(G^T(s_correct + noise,c)), y)
      + CE(R(G^T(s_wrong + noise,c)), y)
```

This expands the correct basin and shrinks wrong basin volume.

6. **Acceptance rule**

At inference, never accept low residual alone. Accept only if:

```text
residual < eps_r
margin > m_accept
readout stable under one extra G step
readout stable under small latent perturbation
```

If not accepted, run extra flow steps or 2-4 perturbed restarts and choose the highest-margin accepted endpoint.

**4. First Experiment**

Run this on the current synthetic addition task, because it already exposes the CE-vs-E5 split.

Target size: about 1M params.

Concrete model:

```text
V = 64
L_out = current task length
d = 128
ContextEncoder: 2-layer TransformerEncoder, 4 heads, ff=512
G: 1 tied TransformerDecoderLayer, 4 heads, ff=512
Flow head: MLP over [z, h, c_pool, t_embed] -> velocity
Readout: LayerNorm + Linear(d,V)
T_G = 5
K_flow = 4 train/eval, also test K=8
```

Training schedule:

```text
Phase A: 10k-20k steps CE-only
Phase B: 10k steps CE + flow
Phase C: 10k steps CE + flow + margin-gated SC
Phase D: 5k-10k steps with wrong-attractor mining + recovery
```

Initial weights:

```text
lambda_CE = 1.0
lambda_flow = 1.0
lambda_sc = ramp 0 -> 0.02
lambda_fp = 0.1
lambda_rec = 0.2
lambda_bad = 0.2
lambda_contract = 0.02
margin gamma = 2.0
```

Success criteria:

```text
seq accuracy >= CE baseline, ideally >= 99.9%
converged_frac >= 95%
wrong-attractor rate <= 1%
median residual below E5 threshold
positive median readout margin at accepted endpoints
no seed with E5-style collapse to 0% seq accuracy
```

Report separately:

```text
accuracy
residual
margin
wrong-attractor count
basin hit rate under 8-16 perturbations per prompt
readout stability under extra step
```

**5. Risks**

The flow head may bypass UESD, turning `G` into a weak conditioner. Control this by ablating `h`, measuring flow performance without `G`, and requiring `G` residual/margin improvement.

SC may still freeze bad basins if ramped too early. Gate it by margin and stop ramping when wrong-attractor rate rises.

Wrong-attractor mining may overfit known failures. Keep fresh mined negatives each epoch and include perturbation neighborhoods, not just exact endpoints.

Flow matching may average multimodal targets. On deterministic addition this is fine; for language later, use multi-sample flow or contrastive target sets.

Acceptance rules may hide failure by rejecting too much. Track rejection rate as a first-class metric.

Local contractivity may reduce useful search. Contract readout-critical directions near endpoints; do not globally suppress all null/search directions.

**6. What To Skip**

Skip pure E5/self-consistency as the main objective. It is the source of the wrong-attractor failure.

Skip “make `k` smaller” as the solution. `k=0.988` is already enough for convergence; lower `k` does not identify the correct basin.

Skip global monotone/strongly convex UESD for the first build. It is a theorem benchmark, not the best practical architecture.

Skip full diffusion/DDPM. Rectified flow gives the needed data-manifold correction with fewer steps.

Skip post-convergence Langevin as a rescue. Noise after fixation mostly destabilizes; use perturbations before commitment with selection.

Skip byte/patch hierarchy for this immediate experiment. It is the right scaling direction, but the current failure is basin correctness, not sequence granularity.