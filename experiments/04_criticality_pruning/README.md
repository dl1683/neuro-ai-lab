# Experiment 04: Criticality-Preserving Pruning

## Biological Parallel

Despite the brain having ~86 billion neurons and ~100 trillion synapses, the **effective computational degree** is only ~2-3 (eLife 2023). A "sparse backbone" of critical connections carries almost all the information flow. This backbone is maintained at a specific dynamical regime: the **critical point** where neural avalanches follow power-law distributions with branching ratio sigma = 1.

Key biological facts:
- Cortical networks have effective degree ~2-3 despite anatomical degree ~10,000
- Beggs & Plenz (2003): neuronal avalanches follow power-law with exponent -3/2 (signature of criticality)
- At criticality (sigma = 1): maximum dynamic range, maximum information transmission, maximum sensitivity
- Departure from criticality (sigma != 1): degraded computation, regardless of network size
- The brain actively maintains criticality through homeostatic plasticity

The implication: **what matters is not how many connections you have, but whether the surviving connections maintain critical dynamics.**

## AI Parallel: Lottery Tickets

- Frankle & Carlin (2019): ~5% of weights form a "winning ticket" that achieves full accuracy
- Standard pruning: remove smallest-magnitude weights → works but is ad hoc
- No principled reason why magnitude should be the right criterion
- What if the right criterion is: **preserve criticality**?

## Hypothesis

Pruning a neural network while maintaining critical avalanche dynamics (branching ratio sigma ≈ 1) will:
1. Achieve higher accuracy at the same sparsity level compared to magnitude pruning
2. Produce a network whose remaining connections form a "sparse backbone" with effective degree ~3
3. The resulting network will be more robust to further perturbation

## Protocol

### Defining "Criticality" in a Neural Network

We adapt the branching ratio concept to artificial networks:

**Activation avalanches:**
1. Feed input through the network
2. Define "active" neurons as those with activation > threshold (e.g., > 0 for ReLU)
3. For each layer transition: count how many neurons in layer L+1 are activated per active neuron in layer L
4. Branching ratio sigma = mean(activated_L+1 / activated_L) across layers and samples
5. At sigma = 1: each active neuron activates exactly 1 in the next layer (critical)
6. sigma > 1: expanding (supercritical); sigma < 1: dying (subcritical)

**Alternative: Jacobian spectral radius**
1. Compute the Jacobian of each layer's output w.r.t. input
2. Spectral radius rho(J) = largest eigenvalue magnitude
3. At criticality: rho(J) = 1 for each layer
4. This is equivalent to: gradients neither explode nor vanish = critical propagation

### Pruning Methods to Compare

**1. Magnitude Pruning (baseline)**
- Remove weights with smallest absolute value
- Standard iterative: prune 20% → retrain → repeat

**2. Criticality-Preserving Pruning (ours)**
- At each pruning step, candidate each weight for removal
- For each candidate: estimate the change in branching ratio (delta_sigma)
- Remove the weight that changes sigma the LEAST (keeps sigma closest to 1)
- Retrain briefly after each pruning step
- Approximation for speed: use the Jacobian spectral radius as proxy (cheaper than full avalanche analysis)

**3. Random Pruning (lower bound)**
- Remove random weights

**4. SNIP / GraSP (structured baselines)**
- SNIP: prune at initialization based on connection sensitivity
- GraSP: prune based on gradient signal preservation

### Experimental Setup
- Architectures: MLP (784-300-100-10), ResNet-18, small ViT
- Datasets: MNIST, CIFAR-10, CIFAR-100
- Sparsity levels: 50%, 70%, 90%, 95%, 99%
- Pruning schedule: iterative (10 rounds of pruning + retraining)

### Measurements
1. **Accuracy vs sparsity** curves for all methods
2. **Branching ratio** of pruned networks (does our method actually maintain sigma ≈ 1?)
3. **Effective degree** of the pruned network (is it close to ~3?)
4. **Robustness**: accuracy under input perturbation (noise, adversarial) at each sparsity level
5. **Avalanche size distribution**: does the pruned network show power-law avalanches?
6. **Jacobian spectral radius** per layer before/after pruning

### Analysis
1. At what sparsity does each method break? Criticality pruning should survive to higher sparsity.
2. What is the effective degree of the surviving backbone? Prediction: ~3 regardless of original architecture.
3. Does maintaining sigma = 1 correlate with maintaining accuracy? (Plot sigma vs accuracy across pruning steps)
4. Does the surviving network exhibit lottery-ticket-like properties? (Re-initialize remaining weights to original values and retrain from scratch)

### Success Criteria
- Criticality pruning maintains >90% accuracy at 95% sparsity where magnitude pruning drops below 85%
- The effective degree of the surviving backbone converges to ~3 (±1)
- Branching ratio of criticality-pruned networks stays within [0.9, 1.1] at all sparsity levels
- The accuracy-sparsity Pareto frontier of criticality pruning dominates magnitude pruning

## Expected Output
- Figure 1: Accuracy vs sparsity (all methods, all architectures)
- Figure 2: Branching ratio vs sparsity (criticality method should be flat at 1.0)
- Figure 3: Effective degree distribution of surviving backbone
- Figure 4: Avalanche size distributions (log-log) at different sparsity levels
- Table 1: Summary metrics at 90% and 95% sparsity across architectures

## Key References
- Frankle & Carlin (2019): The Lottery Ticket Hypothesis, ICLR
- Beggs & Plenz (2003): Neuronal avalanches in neocortical circuits, J Neuroscience
- Priesemann et al. (2014): Spike avalanches in vivo, Frontiers in Systems Neuroscience
- Lee, Ajanthan, Torr (2019): SNIP: Single-shot network pruning
- Wang, Zhang, Grosse (2020): GraSP: Picking the gradient signal

## Estimated Time
- Setup: 4 hours (implement branching ratio computation + pruning loop)
- Training: 6-8 hours (multiple architectures × methods × sparsity levels)
- Analysis: 3 hours
- Total: ~2 days
