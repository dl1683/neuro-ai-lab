# Experiment 03: Reconsolidation-Inspired Continual Learning

## Biological Parallel

When you recall a memory, something remarkable happens: the memory becomes temporarily **unstable** (labile). During this window (~6 hours in humans), the memory can be updated, strengthened, or even erased. Then it re-stabilizes (reconsolidates) in its modified form. This was discovered by Nader, Schafe & LeDoux (2000) — one of the most important findings in modern neuroscience.

This is why memories change over time. They're not static recordings — they're **living representations** that get updated every time they're accessed. The brain SOLVES the stability-plasticity dilemma not by freezing old knowledge, but by making it selectively plastic when relevant.

Key biological details:
- Retrieval triggers protein synthesis-dependent reconsolidation (~6 hour window)
- Only ACTIVATED memories become labile (inactive ones stay stable)
- The memory is integrated with current context during reconsolidation
- Stronger/older memories require stronger reactivation to destabilize
- Blocking protein synthesis during the labile window causes amnesia for that specific memory

## The AI Problem: Catastrophic Forgetting

When a neural network learns Task B after Task A, it overwrites the weights needed for Task A. Standard solutions:
- **EWC (Elastic Weight Consolidation)**: penalize changes to "important" weights — but importance is static
- **PackNet**: freeze weights per task — but capacity runs out
- **Replay**: store old examples — but expensive and privacy-concerning
- **Progressive Networks**: add new capacity per task — doesn't scale

None of these capture the biological insight: **make old knowledge plastic ONLY when it's relevant to what you're currently learning, then re-fix it.**

## Hypothesis

A reconsolidation-inspired mechanism will reduce catastrophic forgetting better than EWC while using less memory than replay:

1. When a new training sample activates old representations strongly (= "retrieval"), temporarily increase plasticity for those specific weights
2. Allow the activated weights to update (= "labilization + reconsolidation")
3. Re-stabilize by restoring low learning rate for those weights
4. Inactive old representations stay frozen (unactivated = stable)

This selectively updates ONLY the old knowledge that is relevant to the new task, leaving irrelevant old knowledge untouched.

## Protocol

### The Reconsolidation Mechanism

For each training batch on the current task:

1. **Forward pass**: compute activations in all layers
2. **Retrieval detection**: for each neuron, measure activation relative to its historical mean. If activation > threshold (e.g., 2 sigma above mean from previous tasks), flag it as "retrieved"
3. **Labilization**: for flagged neurons, multiply their learning rate by a "labilization factor" (e.g., 5-10x) for this batch
4. **Update**: standard backprop, but labilized weights move more
5. **Reconsolidation**: after the batch, restore normal learning rate. Optionally, apply a small "consolidation" L2 penalty toward the updated value (anchoring the new state)

### Benchmarks
- **Split MNIST**: 5 tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9)
- **Split CIFAR-10**: 5 tasks (2 classes each)
- **Permuted MNIST**: 10 tasks (different pixel permutations)
- **Split Tiny-ImageNet**: 10 tasks (20 classes each)

### Baselines
- **Naive SGD**: train sequentially, no protection (lower bound)
- **EWC**: Fisher information penalty on important weights
- **Replay**: store 200 examples per task, mix into training
- **PackNet**: progressive pruning and freezing
- **Reconsolidation** (ours): the mechanism above

### Architecture
- MLP (2 hidden layers, 400 units) for MNIST tasks
- ResNet-18 for CIFAR/Tiny-ImageNet tasks

### Hyperparameters to Sweep
- Retrieval threshold: 1, 2, 3 sigma above historical mean
- Labilization factor: 2x, 5x, 10x, 20x learning rate multiplier
- Labilization duration: 1 batch, 5 batches, 1 epoch
- Consolidation penalty strength: 0, 0.01, 0.1

### Measurements
1. Average accuracy across all tasks after learning the final task
2. Backward transfer: how much does learning new tasks hurt old ones?
3. Forward transfer: does reconsolidation help with new tasks?
4. Per-task accuracy trajectories over time
5. Percentage of weights labilized per task (should decrease for unrelated tasks)

### Success Criteria
- Average accuracy >= EWC with zero stored examples (no replay buffer needed)
- Backward transfer loss < 5% per task (vs 20-40% for naive SGD)
- The mechanism is selective: related tasks trigger more reconsolidation than unrelated ones (measured by % weights labilized)

## Expected Output
- Figure 1: Average accuracy across tasks (all methods compared)
- Figure 2: Per-task accuracy over time (showing selective forgetting/retention)
- Figure 3: Heatmap of "labilization rate" across layers and tasks (which weights reconsolidate?)
- Figure 4: Ablation over threshold and labilization factor
- Table 1: Final metrics across benchmarks and methods

## Key References
- Nader, Schafe & LeDoux (2000): Fear memories require protein synthesis for reconsolidation, Nature
- Kirkpatrick et al. (2017): Overcoming catastrophic forgetting (EWC), PNAS
- Mallya & Lazebnik (2018): PackNet
- Shin et al. (2017): Continual learning with deep generative replay
- Hupbach et al. (2007): Reconsolidation of episodic memories

## Estimated Time
- Setup: 3 hours (implement reconsolidation mechanism + baselines)
- Training: 4-6 hours (multiple benchmarks, multiple methods)
- Analysis: 2 hours
- Total: ~1.5 days
