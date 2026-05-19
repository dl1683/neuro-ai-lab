# Experiment 05: Drift-Diffusion Models as Transformer Depth

## Biological Parallel

When humans make perceptual decisions (e.g., "are these dots moving left or right?"), the brain accumulates evidence over time until a threshold is reached. This is captured by the **drift-diffusion model (DDM)**:

- Evidence variable `x(t)` starts at 0 and drifts with rate `v` (signal strength) plus noise `σ·dW`
- Decision occurs when `x(t)` hits threshold `+a` (choice A) or `-a` (choice B)
- Accuracy: `P(correct) = 1 / (1 + exp(-2va/σ²))` — a **logistic function** of signal-to-noise
- Reaction time: increases with difficulty (lower `v`), decreases with urgency (lower `a`)
- Neural correlate: ramping activity in LIP, FEF, and superior colliculus

The DDM is one of the most successful models in cognitive science, explaining both accuracy AND reaction time with the same mechanism. The key insight: **decisions emerge from sequential evidence accumulation, not single-shot computation.**

## AI Parallel: Residual Streams as Evidence Accumulation

In a transformer, the residual stream carries information across layers:

`h_L = h_0 + Σ_{l=1}^{L} f_l(h_{l-1})`

Each layer adds a **residual update** to the running representation. This is structurally identical to:

`x(t) = x(0) + Σ_{t=1}^{T} (v·dt + σ·dW_t)`

The hypothesis: **transformer layers implement drift-diffusion dynamics.** Each layer accumulates evidence (drift) with some noise, and the final classification emerges when accumulated evidence crosses a threshold.

If true, this means:
- The residual stream magnitude along the correct-class direction should **ramp linearly** across layers (drift)
- Layer-to-layer variability should be roughly constant (diffusion)
- The "decision" should emerge at a specific layer depth (threshold crossing)
- Harder examples should require more layers (more accumulation time)
- This connects transformer depth to reaction time in biological decision-making

## Connection to CTI

The DDM accuracy formula `P = 1/(1 + exp(-2va/σ²))` is exactly the CTI universal law `P = σ(α·κ)` for binary discrimination where:
- `α·κ = 2va/σ²` — the effective signal-to-noise ratio
- `v` = drift rate = discriminability (κ in CTI)
- `a/σ²` = threshold/noise = the system's operating point (α in CTI)

This gives us a principled bridge: if transformers implement DDM dynamics, they should obey CTI scaling laws as a consequence.

## Hypothesis

1. The projection of residual stream activations onto the correct-class direction increases approximately linearly across layers (drift)
2. Layer-to-layer fluctuations of this projection have approximately constant variance (diffusion)
3. Classification confidence emerges at a specific layer (threshold crossing), and this layer comes later for harder inputs
4. Early exit at the threshold-crossing layer achieves comparable accuracy to running all layers
5. The relationship between depth and accuracy follows DDM predictions: `P = 1/(1 + exp(-2vL/σ²))` where L = depth

## Protocol

### Phase 1: Measuring DDM Dynamics in Pretrained Transformers

**Models:** GPT-2 (small, medium, large), ViT-B/16, ResNet-50 (as residual baseline)

For each model and a dataset of inputs:

1. **Extract residual stream at every layer**: `h_0, h_1, ..., h_L`
2. **Define the "decision axis"**: for classification, use the weight vector of the correct class from the final linear layer. For language models, use the embedding of the correct next token.
3. **Project residual stream onto decision axis**: `d_l = h_l · w_correct` for each layer l
4. **Measure drift**: fit linear regression `d_l = v·l + b`. The slope `v` is the drift rate.
5. **Measure diffusion**: compute `σ² = Var(d_l - d_{l-1})` across samples. Should be approximately constant across layers.
6. **Find threshold crossing**: the layer where `d_l` first exceeds the decision threshold (defined as the point where the model would classify correctly with >90% confidence)

### Phase 2: DDM Predictions

Test whether the DDM framework makes accurate predictions:

1. **Accuracy vs depth**: for each input, classify using only the first L layers. Plot accuracy vs L. Compare to DDM prediction: `P(L) = 1/(1 + exp(-2vL/σ²))`
2. **Difficulty modulates drift**: partition inputs by difficulty (e.g., model confidence). Harder inputs should have lower drift rate `v`.
3. **Threshold crossing predicts confidence**: inputs that cross the threshold earlier should have higher final confidence.
4. **Noise predicts errors**: inputs with higher diffusion noise `σ²` should be more likely to be misclassified.

### Phase 3: DDM-Guided Early Exit

If transformers implement DDM dynamics, we can build a principled early-exit mechanism:

1. At each layer, compute the projection `d_l` onto the decision axis
2. If `|d_l|` exceeds a threshold `a`, exit early and classify
3. The threshold `a` can be set from the DDM to achieve a target accuracy
4. Compare to: fixed early exit (exit at layer L/2), confidence-based early exit (exit when softmax > threshold)

### Experimental Setup

**Classification tasks:**
- CIFAR-10/100 with ViT-B/16 (pretrained and fine-tuned)
- ImageNet subset (1000 images, 10 classes) with ViT and ResNet-50

**Language modeling:**
- WikiText-103 with GPT-2 (small, medium, large)
- Next-token prediction accuracy at each layer depth

**Architectures to probe:**
- ViT-B/16 (12 layers) — primary
- GPT-2 small (12 layers) — language comparison
- GPT-2 medium (24 layers) — depth scaling
- ResNet-50 (via residual blocks) — non-transformer residual baseline

### Measurements

1. **Drift rate per input**: slope of `d_l` vs layer index (should be positive for correct, negative for incorrect)
2. **Diffusion coefficient per input**: variance of layer-to-layer changes in `d_l`
3. **Threshold crossing layer**: first layer where classification would be correct
4. **DDM fit quality**: R² of DDM accuracy prediction vs actual layer-wise accuracy
5. **Early exit efficiency**: accuracy vs compute (FLOPs) for DDM exit vs baselines
6. **Drift rate vs difficulty**: correlation between drift rate and human difficulty ratings (if available) or model confidence

### Analysis

1. **Is the drift linear?** Plot `d_l` vs layer for individual inputs (should be approximately linear with noise). Compute linearity R² across the dataset.
2. **Is the diffusion constant?** Plot `Var(Δd_l)` vs layer. Should be approximately flat.
3. **Does the DDM predict accuracy vs depth?** Fit DDM parameters (v, σ², a) from early layers, predict accuracy at deeper layers.
4. **Does difficulty modulate drift?** Group inputs by difficulty, compare drift rates.
5. **Does DDM early exit outperform heuristic early exit?** Compare accuracy-compute Pareto frontiers.
6. **Cross-architecture consistency**: do ViT, GPT-2, and ResNet all show DDM dynamics?

### Success Criteria

- Drift linearity: R² > 0.8 for >70% of correctly classified inputs
- Diffusion constancy: coefficient of variation of per-layer diffusion < 0.3
- DDM accuracy prediction: R² > 0.9 when predicting accuracy-vs-depth curves
- Drift-difficulty correlation: Spearman ρ > 0.5 between drift rate and input difficulty
- DDM early exit achieves >95% of full-model accuracy with <60% of compute (FLOPs)

## Expected Output

- Figure 1: Residual stream projection `d_l` vs layer for example inputs (easy, medium, hard)
- Figure 2: Distribution of drift rates across inputs, colored by correctness
- Figure 3: Accuracy vs depth — actual vs DDM prediction (should overlay closely)
- Figure 4: Threshold crossing layer vs final confidence (scatter)
- Figure 5: Early exit Pareto frontier — DDM vs confidence-based vs fixed
- Figure 6: Diffusion coefficient across layers (should be flat)
- Table 1: DDM fit parameters (v, σ², a) across architectures and tasks

## Key References

- Ratcliff (1978): A theory of memory retrieval, Psychological Review
- Gold & Shadlen (2007): The neural basis of decision making, Annual Review of Neuroscience
- Bogacz et al. (2006): The physics of optimal decision making, Journal of Mathematical Psychology
- Teerapittayanon et al. (2016): BranchyNet: Fast inference via early exiting
- Elbayad et al. (2020): Depth-adaptive transformer
- Schuster et al. (2022): Confident adaptive language modeling (CALM)
- Vul et al. (2014): One and done? Optimal decisions from very few samples

## Estimated Time

- Setup: 3 hours (extract residual streams, implement DDM analysis pipeline)
- Probing pretrained models: 4-6 hours (multiple architectures, datasets)
- Early exit experiments: 2-3 hours
- Analysis: 2 hours
- Total: ~2 days
