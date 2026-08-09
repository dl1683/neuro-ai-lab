**Bottom line:** BLT and MegaByte strongly validate the idea that most useful computation can happen in continuous latent space before a discrete byte readout. But they do **not** provide fixed-point convergence mechanisms. Their lesson for UESD is architectural: move the dynamical state from token/byte positions to **patch latents**, allocate compute by **uncertainty/entropy**, and make continuous-to-discrete readout a lightweight local decoder rather than the central operation.

**Sources used:** BLT arXiv/HTML paper, Hugging Face BLT docs, MegaByte arXiv/NeurIPS paper. Key references: BLT paper abstract and architecture sections state that bytes are encoded into dynamically sized patches, processed by a latent transformer, and decoded by local modules, with entropy patching used to allocate compute where data complexity is higher [BLT arXiv](https://arxiv.org/abs/2412.09871), [BLT HTML](https://ar5iv.labs.arxiv.org/html/2412.09871v1), [HF BLT docs](https://huggingface.co/docs/transformers/v5.0.0rc1/model_doc/blt). MegaByte segments byte sequences into fixed-size patches, uses a global model between patches and local model within patches [MegaByte arXiv](https://arxiv.org/abs/2305.07185), [NeurIPS page](https://papers.nips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html).

**1. BLT: byte -> patch -> byte**

BLT’s core move is to remove fixed-vocabulary tokenization and replace it with dynamically sized byte patches. The pipeline is:

1. Raw UTF-8 bytes are segmented into patches.
2. A lightweight local encoder maps bytes inside each patch into a continuous patch representation.
3. A large latent/global transformer operates autoregressively over the sequence of patch embeddings.
4. A lightweight local decoder maps global patch representations back into byte-level predictions.

The BLT paper describes this directly: a lightweight local encoder encodes bytes into patch representations, a large latent transformer operates over patch representations, and a lightweight local decoder decodes the next patch of bytes. The important detail for UESD is that the “main” model never needs to operate at the raw byte granularity. The expensive transformer is invoked once per patch, not once per byte.

This is closely analogous to UESD’s continuous dynamics, but not identical. BLT’s patch embeddings are continuous latent states, and intermediate computation is performed over those states. However, BLT is still a standard finite-depth feedforward/autoregressive model. It computes:

`patch_latents -> transformer layers -> contextualized_patch_latents -> byte logits`

It does not repeatedly apply a shared map until convergence. There is no explicit `s_{t+1} = G(s_t, c)` loop, no contraction proof, and no stopping criterion based on residual norm. Its “continuous space” is representational, not dynamical.

For UESD, this matters because BLT shows that continuous patch states can be a stable interface between local discrete structure and global semantic computation. But BLT does not solve wrong-attractor failures. If anything, it suggests a cleaner state space in which attractor basins might be better behaved: patch latents are less noisy than byte/token latents because they summarize local context.

**2. BLT entropy patching and variable compute**

BLT’s most relevant mechanism is entropy-based patching. It trains or uses a small byte-level language model to estimate next-byte entropy. High-entropy positions are treated as likely patch boundaries. The result: difficult regions get shorter patches, so the expensive latent transformer is invoked more often; predictable regions get longer patches, so fewer global steps are used.

This is conceptually very close to UESD’s variable iteration depth, but not mathematically identical.

BLT variable compute:

`hard region -> shorter patches -> more global transformer calls per byte`

UESD variable compute:

`hard state/token/patch -> more fixed-point iterations G`

Both allocate more computation where uncertainty is high. But BLT changes the **spatial discretization** of the sequence, while UESD changes the **temporal depth** of inference. BLT says: “make the unit smaller when prediction is hard.” UESD says: “iterate longer when convergence is hard.”

The hybrid version is natural:

`entropy high -> shorter patch and/or larger T`
`entropy low -> longer patch and/or smaller T`

That gives two independent compute knobs: patch granularity and iteration count.

**3. MegaByte: fixed patch hierarchy**

MegaByte is the simpler predecessor. It uses fixed-size byte patches rather than entropy-adaptive patches. The architecture is:

1. Patch embedder: embeds each byte and concatenates/group bytes into fixed-size patch representations.
2. Global model: large autoregressive transformer over patch embeddings.
3. Local model: smaller autoregressive transformer that generates bytes inside each patch, conditioned on the global patch representation.

MegaByte’s global-to-local interface is a continuous embedding. That is useful for UESD because it establishes a clean division of labor:

- global model handles long-range semantics at patch level
- local model handles exact byte realization
- the interface between them is continuous

But again, MegaByte does not contain convergence dynamics. Its global patch embedding is produced by a finite transformer pass. The local model then autoregressively reads out bytes. There is no guarantee that the patch representation is a fixed point or even close to one.

The convergence insight is negative but valuable: hierarchical byte models get much of their efficiency by **reducing the number of global latent decisions**, not by iterating those decisions to equilibrium. UESD could add what MegaByte lacks: a convergence criterion over the global patch representation before byte readout.

**4. Byte-level readout: continuous -> discrete**

Both BLT and MegaByte handle the readout problem conventionally: continuous hidden states are projected to byte logits, and generation proceeds autoregressively at byte level. The model does not need to discretize intermediate states. It only discretizes at the boundary where it emits the next byte.

This maps well to UESD. UESD should avoid forcing intermediate fixed-point states to correspond directly to tokens. Instead:

`continuous state s* -> local decoder -> byte/token distribution`

This decouples convergence from discrete output. The fixed point should be judged by residual stability and decoder confidence/calibration, not by exact equality to a token embedding.

For wrong-attractor failures, the readout layer creates a diagnostic opportunity. A bad attractor may have low residual but high entropy, low margin, poor local decoder consistency, or disagreement under perturbation. BLT/MegaByte imply that continuous convergence alone is insufficient; the continuous-to-discrete interface must be monitored.

Useful UESD criteria:

- residual norm: `||G(s_t, c) - s_t||`
- decoder entropy: `H(p(bytes | s_t))`
- margin: `logit_top1 - logit_top2`
- stability under extra iterations
- stability under small perturbations of `s_t`
- local/global consistency between patch decoder and surrounding context

A true fixed point should be both dynamically stable and readout-stable.

**5. Is there a natural convergence mechanism in patch models?**

Not explicitly. BLT and MegaByte do not converge in the fixed-point sense. But patching may create better **implicit basins**.

Why? Byte/token-level dynamics are extremely local and jagged. A single token embedding may correspond to many syntactic/semantic continuations. Patch embeddings aggregate several bytes and local context, so they may represent a more coherent semantic object: a word fragment, word, whitespace-delimited unit, code fragment, or local byte pattern.

That can help UESD in three ways:

1. Lower effective sequence length: fewer states to stabilize.
2. Smoother latent manifold: patch latents average or attend over local bytes.
3. Better basin separation: high-entropy boundaries isolate hard decisions from easy completions.

The key hypothesis: wrong attractors are more likely when the state space forces global semantic decisions and local spelling/token decisions into the same latent point. Patch hierarchy separates them. UESD dynamics can converge on “what patch should mean” before the local decoder decides “which bytes express it.”

**6. Are BLT variable patches and UESD variable T the same mechanism?**

They are the same **principle** but different **implementation levels**.

Shared principle:

`uncertainty should buy compute`

BLT implements this by varying patch length. High entropy creates more patch boundaries, increasing global compute density. UESD implements it by varying iteration depth. High uncertainty should cause more applications of `G`.

The stronger hybrid is to use entropy twice:

1. Use entropy to decide patch boundaries.
2. Use entropy/residual to decide iteration depth per patch.

For example:

```text
for patch i:
    h_i = local_encoder(bytes_i, context)
    s_i^0 = h_i
    T_i = schedule(entropy_i, residual_i, decoder_uncertainty_i)
    for t in 0..T_i:
        s_i^{t+1} = G(s_i^t, global_context)
    bytes_i = local_decoder(s_i^{T_i})
```

The schedule should not be entropy-only. Entropy predicts difficulty, but convergence failure is a dynamical fact. A better rule:

`T_i = min(T_max, T_base + alpha * entropy_i + beta * residual_slope_i + gamma * decoder_entropy_i)`

Stop early when:

`||s_{t+1} - s_t|| < eps`
and decoder entropy is low
and output distribution is stable over the last few iterations.

This is more robust than simply running more steps everywhere. With contraction `k=0.988`, convergence can be slow near hard regions. Adaptive T is almost mandatory if you want fixed-point quality without wasting compute.

**7. Concrete hybrid proposals**

**Proposal A: UESD over BLT-style patch latents**

Use a BLT-like local encoder to create patch embeddings. Replace or augment the BLT global transformer with UESD dynamics:

`S_{t+1} = G(S_t, C)`

where `S_t` is the sequence of patch states, not token states. `G` can be a transformer block, recurrent transformer, or learned denoising/refinement operator. The local decoder emits bytes from the converged patch states.

This is the cleanest hybrid. It makes UESD operate at the same abstraction level where BLT spends its expensive computation.

**Proposal B: entropy-adaptive iteration depth**

Use BLT’s entropy model or the model’s own decoder entropy to allocate iterations:

- low entropy: 1-3 iterations
- medium entropy: 4-8 iterations
- high entropy: iterate until residual/readout stability or `T_max`

This directly targets wrong-attractor risk: hard regions get more opportunities to settle, while easy continuations avoid unnecessary refinement.

**Proposal C: patch-level fixed points with local byte refinement**

Each patch has its own fixed point:

`s_i* = G_i(s_i*, context, neighboring_patch_states)`

Patches can converge semi-independently, with global context updated between rounds. This resembles block-coordinate fixed-point iteration:

1. encode all patches
2. update hard patches
3. refresh global context
4. decode stable patches
5. continue only unstable patches

This is attractive because wrong-attractor failures are often local. A globally fixed `T` wastes compute on easy regions and underserves hard ones.

**Proposal D: attractor validation via local decoder**

After convergence, decode bytes multiple ways:

- decode from `s_T`
- decode from `s_{T-1}`
- decode after one extra iteration
- decode after small latent perturbation

If the byte distribution changes sharply, the state is not readout-stable even if the residual is small. This is a practical wrong-attractor detector.

**8. What byte transformers teach UESD**

They teach four main lessons.

First, continuous intermediate computation is viable. BLT and MegaByte both show that the main model can operate over learned continuous patch representations without softmax/discrete choices at every intermediate step.

Second, patching is a better unit for expensive reasoning than bytes or tokens. Tokens are brittle because the vocabulary is fixed. Bytes are too granular. Patches give a learned, flexible middle layer.

Third, variable compute should be uncertainty-driven. BLT’s entropy patching is the strongest evidence here. UESD should not use fixed iteration depth unless the goal is simplicity. Entropy, residual decay, and decoder uncertainty should control T.

Fourth, convergence must be evaluated at two levels: latent stability and discrete readout stability. A contraction proof gives uniqueness for a given map/context under assumptions, but it does not guarantee that the resulting attractor corresponds to the desired discrete output. BLT/MegaByte remind us that the final byte decoder is its own source of error.

**Recommended direction:** build UESD at patch level, not token level. Use BLT-style local encoding/decoding as the byte interface, run fixed-point dynamics only over patch latents, and allocate iteration depth using entropy plus residual/readout-stability checks. This keeps UESD’s convergence machinery focused on the semantic latent space where it is most likely to help, while leaving exact byte emission to a specialized local decoder.