# Experiment 01: Predicting Grokking from Representation Dynamics

## Biological Parallel

In human cognition, understanding arrives suddenly after extended plateaus ("aha!" moments). Neuroscience shows this is a phase transition: internal representations reorganize silently during the plateau, then cross a critical threshold causing discontinuous improvement. EEG studies show gamma bursts (phase transitions) at the moment of insight, preceded by alpha increase (internal search).

## AI Phenomenon

Grokking (Power et al., 2022): neural networks trained on algorithmic tasks (modular arithmetic, permutation groups) first memorize the training set (100% train accuracy, chance validation), then after many more epochs suddenly generalize (validation accuracy jumps to 100%). The delay between memorization and generalization can be 10-100x the memorization time.

## Hypothesis

Grokking occurs when the mean pairwise correlation between class representations (rho) crosses a critical threshold. Specifically:
- During memorization: representations are high-dimensional and uncorrelated (rho << 0.46)
- During the plateau: rho increases monotonically as representations reorganize
- At grokking: rho crosses ~0.46 (tanh(1/2)) — the neural collapse equicorrelation
- After grokking: rho stabilizes at the equicorrelation value

If true, we can PREDICT grokking onset from rho trajectory without seeing validation accuracy.

## Protocol

### Setup
- Task: modular arithmetic (a op b mod p) for op in {+, -, *, /}, p in {97, 113}
- Architecture: 1-layer transformer (standard grokking setup from Power et al.)
- Also test: 2-layer MLP, small CNN on CIFAR-10 subsets (to check generality)
- Weight decay: 0.01 (required for grokking)
- Training: full batch, AdamW, lr=1e-3

### Measurements (every epoch)
1. **rho**: mean pairwise cosine similarity between class centroids in final hidden layer
2. **rho_std**: standard deviation of pairwise cosines (should decrease toward 0 = equicorrelation)
3. **DFA exponent**: of the loss time series (should approach 1.0 before grokking)
4. **Effective rank**: of the representation matrix (should decrease = compression)
5. **Train/val accuracy**: ground truth for when grokking occurs

### Analysis
1. Plot rho trajectory overlaid with validation accuracy. Does rho cross 0.46 at grokking?
2. Fit sigmoid to rho(t). Does the inflection point predict grokking epoch?
3. Test across different moduli, operations, architectures. Is the threshold universal?
4. Compute "early warning" metric: can we predict grokking 10+ epochs before it happens?

### Success Criteria
- rho crosses a consistent threshold (within 10% of 0.46) at grokking across >= 3 different settings
- The threshold crossing predicts grokking at least 5 epochs before validation accuracy jumps
- The phenomenon is not trivially explained by weight norm or loss alone

## Expected Output
- Figure 1: rho vs epoch with grokking onset marked (multiple settings overlaid)
- Figure 2: Threshold crossing time vs actual grokking time (scatter, should be y=x)
- Figure 3: Representation geometry visualization (PCA) at pre-grokking, grokking, post-grokking
- Table 1: Threshold values across all settings (is it consistently ~0.46?)

## Key References
- Power et al. (2022): Grokking: Generalization beyond overfitting on small algorithmic datasets
- Nanda et al. (2023): Progress measures for grokking via mechanistic interpretability
- Papyan, Han, Donoho (2020): Neural collapse (ETF convergence)
- Zhong et al. (2024): The clock and the pizza (grokking representations)

## Estimated Time
- Setup: 2 hours (standard grokking setup is well-documented)
- Training runs: 4-6 hours (multiple settings, GPU)
- Analysis: 2 hours
- Total: ~1 day
