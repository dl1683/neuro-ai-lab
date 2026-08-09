I read `docs/UNIFIED_ERROR_SPACE.md`, the two completed LCM/byte-transformer analyses, current UESD experiment outputs, and scanned the requested sibling directories. The design conclusion is: do not try to make self-consistency alone imply correctness. It cannot. You need convergence of the latent dynamics plus a correctness-carrying projection/readout mechanism with an explicit margin.

Below are 3 implementable architectures.

**Architecture 1: Contractive Compute + Rectified-Flow Corrector**

This is the best first build.

**Layer Structure**

Reuse current UESD backbone, but reinterpret it:

```text
prompt tokens x
 -> ContextEncoder: 2-layer TransformerEncoder, d=128, h=4, ff=512
 -> initial state s0: learned target-state slots, shape [L_out, 128]
 -> tied compute map Gc, applied T_compute=3..5 times
 -> rectified-flow corrector H_phi, applied K=4..8 ODE steps
 -> readout head R
```

Concrete modules:

```text
Token embedding: V x 128                          ~8K for V=64
Positional embeddings: 2L x 128                   ~3K

ContextEncoder:
  2 x TransformerEncoderLayer(d=128,h=4,ff=512)  ~396K

G_compute:
  1 x TransformerDecoderLayer(d=128,h=4,ff=512)
  weight tied across iterations                   ~263K
  residual gate alpha(c) in [0.2,0.8]             ~33K
  spectral norm on q/k/v/o and FFN linear layers  no new params

FlowCorrector H_phi:
  input per slot = concat(z_t, h, c_pool, t)       128+128+128+16
  MLP 400 -> 256 -> 256 -> 128, SiLU              ~202K
  FiLM from c_pool,t for both hidden layers        ~132K

Readout:
  LayerNorm + Linear(128,V) per slot               ~8K
```

Total: about `1.04M` params for V=64, L=8-12. For larger vocab, only embedding/readout scale.

**Training Loss**

Let `y* in R^{L x d}` be the target embedding sequence from the learned output embedding table or a small target encoder. Let:

```text
h = G_c^T(s0)
epsilon ~ N(0,I)
z0 = y*
z1 = epsilon
z_t = (1-t) z0 + t z1       where t ~ Uniform(0,1)
u*(z_t,t,c,h) = z1 - z0
```

Train rectified flow from noise to data in reverse, or equivalently learn velocity toward data:

```text
L_flow = E || H_phi(z_t, h, c, t) - (z0 - z1) ||^2
```

Endpoint/readout losses:

```text
L_CE = CE(R(z_hat_0), y_tokens)
L_contract = sum_t max(0, ||G(s_t,c)-G(s'_t,c)|| / ||s_t-s'_t|| - k0)^2
L_stable = ||G(h,c)-h||^2
L_margin = max(0, m - logit_y + max_{v != y} logit_v)
```

Full loss:

```text
L = L_CE
  + 1.0 L_flow
  + 0.05 L_contract
  + 0.02 L_stable
  + 0.1 L_margin
```

Use `k0=0.98`, margin `m=2.0`. Warm-start from the existing CE-trained dynamics for 5k-10k steps, then add flow.

**Inference**

1. Encode prompt: `c = Enc(x)`.
2. Initialize learned state slots `s0`.
3. Compute `h = G^3..G^5(s0,c)`.
4. Initialize `z_1 = h + sigma * eps`, or for deterministic tasks simply `z_1 = h`.
5. Euler integrate flow from `t=1` to `t=0`:

```text
for k in K..1:
    t = k / K
    z = z - (1/K) * H_phi(z, h, c, t)
```

6. Decode `R(z)`.
7. Accept only if both hold:
   `||G(h,c)-h|| < eps_latent` and readout margin > `m_accept`.
   Otherwise run `K=8` instead of `K=4`, or restart flow with 2 small noise seeds and choose lowest CE-free energy.

**Convergence Proof Sketch**

`G` is not the correctness guarantee. It is a bounded compute operator. With spectral normalization and residual gate, local perturbation growth is bounded, and current VT evidence says this is already directionally contractive near trajectories.

The flow stage gives the correctness guarantee under standard learned-vector-field assumptions. If the target manifold has readout margin `m`, and the learned flow velocity has integrated error less than `m / ||W_R||`, then the final latent remains inside the correct readout cell. Formally, Gronwall bounds endpoint error:

```text
||z_hat_0 - y*|| <= (e^{L_H}-1)/L_H * sup_t ||H_phi - u*||
```

If this is below the decoder margin radius, readout is correct.

**Wrong-Attractor Mitigation**

Self-consistency no longer defines the final answer. Wrong stable `h` can still occur, but the flow head is trained to map the conditional computation summary onto the supervised output manifold. Add a rejection rule: low residual is insufficient; require readout margin and stability under one extra flow step.

**RTX 5090 Cost**

Current 700K UESD runs took roughly 1.1k-2.0k seconds per 20k-step run on the logged 5090 laptop. This model is ~1.5x params and adds flow training, so expect:

```text
training: ~35-60 min per 20k-step synthetic run on 5090 laptop,
          likely ~20-40 min on desktop RTX 5090
inference: 3-5 G steps + 4-8 MLP flow steps
           well under 2 ms per batch item for L<=12, batch-friendly
```

**Architecture 2: Strongly-Convex Monotone Energy UESD**

This is the cleanest mathematical convergence guarantee for deterministic tasks.

**Layer Structure**

```text
ContextEncoder: same 2-layer transformer, d=128       ~396K
Target slots s: [L,128]
Energy E_theta(s,c): strongly convex ICNN
Gradient dynamics: s_{t+1} = s_t - eta * grad_s E(s_t,c)
Readout R(s)
```

Use an input-convex neural network:

```text
q0 = Linear_s(s) + Linear_c(c_pool)
q1 = softplus(W1_pos q0 + A1 s + B1 c_pool + b1)
q2 = softplus(W2_pos q1 + A2 s + B2 c_pool + b2)
E  = sum_slots [ softplus(w3_pos q2 + a3 s + b3 c_pool) ]
   + (mu/2) ||s||^2
```

Implementation details:

- `W*_pos = softplus(raw_W*)` to enforce nonnegative convex paths.
- Hidden width 256.
- Add `mu=0.05` strong convexity.
- Spectral-normalize all `A_i`, `B_i`, and cap positive weights.
- Estimate global smoothness `L_E`; choose `eta < 2 / L_E`.

Parameter count:

```text
ContextEncoder                           ~396K
ICNN energy, slot-shared                  ~260K-340K
Readout                                  ~8K
Embeddings/positions                     ~12K
Total                                    ~680K-760K
```

**Training Loss**

Let `s* = Embed(y*)`.

```text
L_score = || grad_s E_theta(s*, c) ||^2
L_energy_margin = E(s*,c) + log sum_j exp((gamma - E(s_j^-,c))/tau)
L_CE = CE(R(s_T), y*)
L_denoise = E_{sigma,eps} || ProxSteps(s* + sigma eps, c) - s* ||^2
L_L = max(0, L_hat_E - L_max)^2
```

Full:

```text
L = L_CE
  + 1.0 L_score
  + 0.5 L_energy_margin
  + 0.5 L_denoise
  + 0.1 L_L
```

Negatives `s_j^-` should include wrong-attractor states from prior E5 runs, not just random embeddings.

**Inference**

1. Encode context `c`.
2. Initialize `s0` from learned slots plus context projection.
3. Iterate:

```text
for t in 1..T_max:
    s = s - eta * grad_s E(s,c)
    if ||grad_s E(s,c)|| < eps and readout_margin(s) > m:
        break
```

4. Decode with `R(s)`.

Use `T_max=20`, usually stop in 6-12 steps.

**Convergence Proof Sketch**

Because `E(s,c)` is `mu`-strongly convex and `L`-smooth in `s`, it has exactly one minimizer for each context. Gradient descent with `0 < eta < 2/L` converges linearly:

```text
||s_t - s_c*|| <= q^t ||s_0 - s_c*||
q = max(|1 - eta mu|, |1 - eta L|)
```

If training enforces `grad E(s*,c)=0` and an energy/readout margin around `s*`, then the unique fixed point is the correct target embedding up to the margin radius.

**Wrong-Attractor Mitigation**

There are no extra attractors in a strongly convex energy. This directly attacks the 20% wrong-attractor issue. The tradeoff is expressivity: globally convex energy is best for deterministic synthetic tasks. For multimodal natural language, use a mixture of convex energies with a gated mode variable, but then the hard guarantee becomes per-selected-mode.

**RTX 5090 Cost**

Training is close to current UESD, but each step needs gradient through energy during unroll.

```text
training: ~30-55 min per 20k steps on 5090 laptop class hardware
inference: 6-20 gradient steps; likely 1-4 ms/sample for L<=12
```

This is slower than Architecture 1 at inference, but has the cleanest proof.

**Architecture 3: Patch-Level UESD With Dual Stability Gate**

This is the architecture to scale beyond toy token slots.

**Layer Structure**

```text
Byte/token input
 -> LocalPatchEncoder
 -> patch states P0: [N_patch, 128]
 -> Global UESD dynamics over patches
 -> Conditional denoising/readout per patch
 -> LocalPatchDecoder
```

Concrete for current synthetic tasks:

```text
Patch size: 2 tokens, stride 2
N_patch = ceil(L/2)

LocalPatchEncoder:
  token emb 64x128
  1-layer local transformer or MLP over patch tokens      ~80K

GlobalPatchUESD:
  tied TransformerDecoderLayer d=128,h=4,ff=512           ~263K
  applied T_i adaptively per patch/group

EntropyDepthController:
  MLP over [residual, decoder_entropy, margin, delta]      ~20K

PatchFlowHead:
  same as Architecture 1 but shared over patches           ~250K-330K

LocalPatchDecoder:
  MLP 128 -> 256 -> patch_size*V logits                   ~65K
```

Total: about `700K-900K`.

**Training Loss**

For patch target embeddings `p_i*`:

```text
L_patch_CE = sum_i CE(LocalDecoder(z_i), y_patch_i)

L_flow = sum_i E || H(z_{i,t}, h_i, c, t) - (p_i* - eps_i) ||^2

L_latent_stability =
  sum_i 1[active_i] ||G(p_i,c)-p_i||^2

L_readout_stability =
  KL(R(z_i^T) || R(z_i^{T+1}))
  + KL(R(z_i^T) || R(z_i^T + delta)), delta ~ N(0, sigma^2 I)

L_depth =
  beta * mean(T_i)
```

Full:

```text
L = L_patch_CE
  + 1.0 L_flow
  + 0.05 L_latent_stability
  + 0.2 L_readout_stability
  + 0.01 L_depth
```

Depth controller target:

```text
T_i = clamp(
  T_base
  + a * entropy_i
  + b * residual_i
  + g * readout_instability_i,
  2, 16
)
```

For differentiable training, sample `T_i` from buckets `{2,4,8,16}` with straight-through Gumbel or just train all buckets and use controller only at inference.

**Inference**

1. Encode prompt into patches.
2. For each patch, compute initial entropy and residual.
3. Run global patch dynamics:

```text
for t in 1..16:
    update only active patches
    recompute residual, decoder entropy, readout KL
    deactivate patch if:
       residual < eps
       and decoder entropy < H_max
       and readout KL over last step < delta
```

4. Run 2-4 flow steps on stable patches; 8 steps on unstable patches.
5. Decode local patches.
6. If a patch has low latent residual but high readout entropy, do not accept it as converged.

**Convergence Proof Sketch**

If each active patch update is contractive with `k_i < 1` under fixed neighboring context, and the block-coordinate schedule visits every active patch, then patch states converge to a block fixed point. The flow/readout layer then gives the same margin-based correctness guarantee as Architecture 1. The key improvement is that convergence is checked at two levels:

```text
latent fixed point: ||G(s)-s|| small
readout fixed point: R(s_t) stable under extra step and perturbation
```

A wrong attractor usually passes the first test but fails the second.

**Wrong-Attractor Mitigation**

This architecture attacks the specific failure mode from two angles:

1. Patch latents separate semantic/global computation from local realization, reducing basin entanglement.
2. Low residual alone is not accepted. Readout entropy, margin, and perturbation stability must agree.

This incorporates the byte-transformer scan: uncertainty buys compute, and correctness must be evaluated at both latent and readout levels.

**RTX 5090 Cost**

```text
training: ~40-70 min per 20k-step synthetic run
inference: adaptive, usually 2-6 global steps for easy patches,
           8-16 for hard patches, plus 2-8 flow steps
           still cheap at L<=32; much better scaling than token-level fixed T
```

**Recommendation**

Build in this order:

1. **Architecture 1 first.** It is the highest probability fix for the current 20% wrong-attractor problem because it keeps the working UESD computation and adds a correctness-trained manifold projector.
2. **Architecture 2 second.** Use it as the theorem benchmark: if the convex-energy model solves the task but underperforms flow, you have a clean proof/expressivity tradeoff.
3. **Architecture 3 third.** Use when moving beyond fixed small token sequences.

The important design rule: convergence must mean `stable + correct-margin`, not just `stable`. Current E5 optimizes “has stopped moving.” These architectures optimize “has stopped moving inside the correct readout basin.”