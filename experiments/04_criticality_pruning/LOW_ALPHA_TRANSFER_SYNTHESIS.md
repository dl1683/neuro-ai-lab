# Low-Alpha Transfer Synthesis

This document summarizes the current transfer evidence for adaptive dense-tail path correction at `98%` sparsity.

## Core result

The method is not an aggressive path-flow replacement for magnitude pruning. The transferable rule is a very small path-aware modulation of magnitude scores near the pruning cliff.

## Fashion-MNIST CNN

Same-run baseline and alpha sweep:

| Alpha | One-shot accuracy | After masked FT | Readout |
|---:|---:|---:|---|
| `0.00` | `44.55%` | `83.34%` | magnitude baseline |
| `0.05` | `45.58%` | `83.94%` | best recovery |
| `0.08` | `49.38%` | `83.65%` | best balanced |
| `0.15` | `50.01%` | `82.93%` | best one-shot, recovery cost |
| `0.50` | `27.74%` | `80.34%` | too aggressive |

## CIFAR-10 CNN transfer

Real CIFAR-10 images, 20k train subset, 5k test subset, two seeds, small CNN, `98%` global sparsity:

| Alpha | One-shot accuracy | After masked FT | Readout |
|---:|---:|---:|---|
| `0.00` | `16.65%` | `45.28%` | magnitude baseline |
| `0.03` | `18.14%` | `45.79%` | best one-shot / balanced |
| `0.05` | `16.50%` | `46.40%` | best recovery |
| `0.08` | `15.45%` | `43.35%` | too much modulation for CIFAR |
| `0.15` | `12.50%` | `24.24%` | collapse |
| `0.20` | `11.29%` | `14.91%` | near-collapse |

## Current practical rule

Use `alpha=0.03-0.05` at the severe sparsity cliff unless there is a model-specific sweep proving that stronger correction helps. The reusable default in `shared/adaptive_path_pruning.py` is now `0.05` for balanced/recovery severe-sparsity use.

## What transferred

- The weak correction idea transferred from Fashion-MNIST CNN to CIFAR-10 CNN.
- The exact best alpha did not transfer.
- Strong path weighting failed badly on CIFAR.

## What did not transfer

The larger one-shot gains seen at `alpha=0.08-0.15` on Fashion-MNIST did not transfer to CIFAR. CIFAR requires a smaller correction.

## Claim update

The result is now more credible but narrower: adaptive path correction is a low-alpha severe-sparsity guardrail for dense tails, not a universal replacement for magnitude pruning or SynFlow.

Artifacts:

- `results/04_criticality_pruning/cifar10_cnn_98pct_adaptive_alpha_ft_sweep.json`
- `experiments/04_criticality_pruning/CIFAR10_CNN_98PCT_ADAPTIVE_ALPHA_FT_SWEEP.md`
- `results/04_criticality_pruning/cnn_98pct_adaptive_alpha_ft_sweep.json`
- `experiments/04_criticality_pruning/CNN_98PCT_ADAPTIVE_ALPHA_FT_SWEEP.md`

## CIFAR-10 sparsity-cliff correction

A second CIFAR-10 CNN run swept low alphas across `95%`, `98%`, and `99%` sparsity. This partially supports the low-alpha idea but shows the transfer story has variance and should not be overstated.

| Sparsity | Magnitude before/after | Best low-alpha before delta | Best low-alpha after delta | Interpretation |
|---:|---:|---:|---:|---|
| `95%` | `33.97%` / `55.89%` | `+2.71` pts at `alpha=0.05` | `+0.50` pts at `alpha=0.03` | low-alpha helps |
| `98%` | `19.46%` / `47.61%` | `0.00` pts | `+0.33` pts at `alpha=0.05` | mostly tied, tiny recovery edge |
| `99%` | `14.51%` / `34.30%` | `+1.46` pts at `alpha=0.05` | `0.00` pts | one-shot edge, recovery hurt |

Takeaway: the CIFAR transfer is real enough to keep pursuing but not yet strong enough to claim as stable. The next gate is a larger paired GPU replicate with more seeds.

Artifact: `results/04_criticality_pruning/cifar10_cnn_low_alpha_sparsity_cliff.json`.

## Six-seed GPU replicate update

A six-seed CIFAR-10 GPU replicate resolved the earlier ambiguity. Device was `cuda` on `NVIDIA GeForce RTX 5090 Laptop GPU`.

Setup: real CIFAR-10 images, 20k train subset, 5k test subset, six seeds, small CNN, paired magnitude vs low-alpha dense-tail path correction at `95/98/99%` sparsity.

Paired deltas vs magnitude:

| Sparsity | Alpha | One-shot delta | One-shot wins | After-FT delta | After-FT wins |
|---:|---:|---:|---:|---:|---:|
| `95%` | `0.03` | `+1.49` pts | `5/6` | `-0.44` pts | `1/6` |
| `95%` | `0.05` | `+1.47` pts | `6/6` | `-0.66` pts | `2/6` |
| `98%` | `0.03` | `+1.13` pts | `4/6` | `-0.29` pts | `3/6` |
| `98%` | `0.05` | `+0.85` pts | `3/6` | `-1.85` pts | `0/6` |
| `99%` | `0.03` | `-0.19` pts | `3/6` | `-1.42` pts | `1/6` |
| `99%` | `0.05` | `+0.46` pts | `3/6` | `-5.64` pts | `0/6` |

Conclusion: low-alpha path correction is a real one-shot preservation signal at `95-98%` on CIFAR, but magnitude remains the safer fine-tuning initializer. The reusable implementation now defaults to `alpha=0.03` for balanced/one-shot and `alpha=0.0` for recovery.

Artifact: `results/04_criticality_pruning/cifar10_cnn_low_alpha_gpu_replicate.json`.

## Bridge-floor negative result

A structural bridge-floor repair was tested on CIFAR-10 CNN at `98%` and `99%` sparsity. The repair starts from global magnitude pruning, forces each `fc1` hidden unit to keep at least 1 or 2 incoming weights, then removes low-score unprotected weights elsewhere to preserve the same total parameter budget.

Result: dead `fc1` hidden units were eliminated, but accuracy did not materially improve.

| Sparsity | Method | Before delta vs magnitude | After-FT delta vs magnitude | Interpretation |
|---:|---|---:|---:|---|
| `98%` | `bridge_floor1` | `-0.18` pts | `+0.10` pts | neutral |
| `98%` | `bridge_floor2` | `-0.68` pts | `+0.03` pts | neutral/slightly worse |
| `99%` | `bridge_floor1` | `+0.07` pts | `+0.07` pts | neutral |
| `99%` | `bridge_floor2` | `+0.03` pts | `-0.88` pts | worse after FT |

Conclusion: bridge liveness is necessary for avoiding total collapse, but it is not sufficient for a better mask. The useful mechanism must preserve a good bridge, not merely any bridge.

Artifact: `results/04_criticality_pruning/cifar10_cnn_bridge_floor.json`.
