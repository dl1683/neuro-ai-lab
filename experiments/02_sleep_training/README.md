# Experiment 02: Sleep-Cycle Training Schedules

## Biological Parallel

The brain doesn't learn continuously. It cycles through distinct phases every ~90 minutes during sleep:

1. **NREM slow-wave sleep**: High-amplitude, low-frequency oscillations. The brain compresses and consolidates what was learned during the day. Synaptic homeostasis hypothesis (Tononi & Cirelli): synapses are globally downscaled, keeping the strongest connections and pruning the weak ones. This is literally information bottleneck compression.

2. **REM sleep**: Desynchronized, wake-like activity. The brain replays experiences in distorted form (dreams). Function: testing whether the compressed representation still works. Bizarre dream content = probing edge cases of the compressed manifold.

3. **Repeat 4-5 times per night**, with NREM decreasing and REM increasing across cycles. Early cycles: heavy compression. Late cycles: heavy testing.

Sleep deprivation degrades generalization far more than memorization — consistent with sleep being essential for the compression step that enables generalization.

## Hypothesis

A training schedule that alternates between compression phases (aggressive pruning/distillation, analogous to NREM) and testing phases (generative replay with augmented data, analogous to REM) will:
1. Reach the same final accuracy as standard training
2. With better generalization (lower generalization gap)
3. And a sparser final network (fewer active parameters)

## Protocol

### Baseline
- Dataset: CIFAR-10 (standard), also test on Tiny-ImageNet
- Architecture: ResNet-18
- Standard training: SGD with cosine LR schedule, 200 epochs, standard augmentation
- Measure: train acc, val acc, generalization gap, model sparsity, representation rho

### Sleep-Cycle Training

One "sleep cycle" = NREM phase + REM phase. Run 5 sleep cycles per "night" (matching biology).

**NREM phase (compression, 10 epochs per cycle):**
- High weight decay (10x baseline) — global synaptic downscaling
- Knowledge distillation from the model's own previous checkpoint (self-distillation)
- Magnitude pruning: zero out bottom 10% of weights per cycle
- Low learning rate (0.1x baseline)
- Purpose: compress, consolidate, remove redundancy

**REM phase (testing, 5 epochs per cycle):**
- Standard weight decay
- Aggressive data augmentation (CutMix, MixUp, random erasing) — "bizarre dream content"
- Optionally: generative replay using a small VAE trained on the dataset
- Normal learning rate
- Purpose: test compressed representation against edge cases, fine-tune

**Night structure:**
- 5 cycles per "night", total = 5 * (10 + 5) = 75 epochs
- 3 "nights" = 225 epochs (comparable to 200-epoch baseline)
- Ratio shifts across nights: early nights = more NREM, late nights = more REM (matching biology)

### Measurements
1. Val accuracy at end of each phase (does REM catch NREM damage?)
2. Generalization gap: train_acc - val_acc (should be smaller than baseline)
3. Active parameter count after each NREM (should decrease)
4. Representation rho after each cycle (should converge faster than baseline)
5. DFA exponent of loss series within each phase

### Ablations
- NREM only (compression without testing) — expect: good sparsity, poor accuracy
- REM only (augmentation without compression) — expect: similar to standard augmented training
- Random cycling (arbitrary LR/WD changes) — control for "any cycling helps"
- Different cycle lengths (30, 60, 120 epoch "nights")
- Different NREM:REM ratios

### Success Criteria
- Sleep-cycle training achieves <= 1% accuracy loss vs baseline with >= 30% parameter reduction
- Generalization gap is measurably smaller (>= 2 percentage points)
- The effect is NOT explained by simple LR cycling (ablation: random cycling control doesn't match)

## Expected Output
- Figure 1: Accuracy curves for sleep-cycle vs baseline vs ablations
- Figure 2: Generalization gap over training (sleep should stay tighter)
- Figure 3: Sparsity progression (parameters pruned per NREM phase)
- Figure 4: rho trajectory comparison (sleep vs baseline — does sleep converge rho faster?)
- Table 1: Final metrics across all conditions

## Key References
- Tononi & Cirelli (2003): Sleep and synaptic homeostasis hypothesis
- Hinton, Vinyals, Dean (2015): Knowledge distillation
- Frankle & Carlin (2019): Lottery ticket hypothesis
- Yun et al. (2020): CutMix regularization
- Walker (2017): Why We Sleep (pop-sci but great for intuition)

## Estimated Time
- Setup: 3 hours (training loop modifications)
- Training runs: 6-8 hours (baseline + sleep + ablations, GPU)
- Analysis: 2 hours
- Total: ~1.5 days
