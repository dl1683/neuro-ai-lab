# Criticality Pruning Experiment Log

This folder contains the active real-experiment thread for severe-sparsity pruning.

## Strongest finding

Global SynFlow can catastrophically starve dense classifier bridges in CNNs at severe global sparsity.

The clearest evidence is aggregated in:

- `SYNFLOW_PATHOLOGY_SYNTHESIS.md`
- `results/04_criticality_pruning/synflow_pathology_synthesis.json`

Cross-dataset synthesis:

- Severe-sparsity CNN cases synthesized: `3`.
- Global SynFlow zero-`fc1` allocation cases: `3/3`.
- Mean global SynFlow after-fine-tuning delta vs magnitude: `-42.80` points.
- Mean layerwise SynFlow after-fine-tuning delta vs magnitude: `-22.21` points.

Mechanism: global SynFlow allocates zero weights to the first dense classifier bridge (`fc1`) in the tested CNNs, so masked fine-tuning cannot recover. Layerwise SynFlow restores a nominal per-layer budget but still trails magnitude, implying the dense-bridge ranking is also poor.

## Secondary finding

Low-alpha dense-tail path correction can improve one-shot severe pruning, but it is not a robust fine-tuning initializer.

Best synthesis:

- `LOW_ALPHA_TRANSFER_SYNTHESIS.md`
- `CIFAR10_CNN_LOW_ALPHA_GPU_REPLICATE.md`
- `CNN_98PCT_ADAPTIVE_ALPHA_FT_SWEEP.md`

Current rule encoded in `shared/adaptive_path_pruning.py`:

- `balanced` / `one_shot`: use tiny `alpha=0.03` near the severe sparsity cliff.
- `recovery`: use `alpha=0.0` unless a domain-specific sweep proves otherwise.

## Negative result

A simple structural bridge-floor repair is not enough.

- `CIFAR10_CNN_BRIDGE_FLOOR.md`
- `results/04_criticality_pruning/cifar10_cnn_bridge_floor.json`

The repair eliminated dead `fc1` hidden units but produced near-zero accuracy deltas, so useful bridge quality matters more than mere bridge liveness.

## Reusable code

- `shared/adaptive_path_pruning.py`: tiny dense-tail path correction and mask utilities.
- `shared/pruning_diagnostics.py`: layer keep-rate and dense-bridge collapse diagnostics.

## Reproduction commands

Run the strongest pathology synthesis:

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
```

Re-run the CIFAR SynFlow pathology experiment:

```powershell
python experiments\04_criticality_pruning\cifar10_cnn_synflow_pathology.py
```

Re-run the six-seed low-alpha CIFAR replicate:

```powershell
python experiments\04_criticality_pruning\cifar10_cnn_low_alpha_gpu_replicate.py
```

Re-run the Fashion-MNIST CNN SynFlow mechanism checks:

```powershell
python experiments\04_criticality_pruning\synflow_cnn_mask_forensics_98pct.py
python experiments\04_criticality_pruning\synflow_cnn_layerwise_rescue_98pct.py
```

## Current claim boundary

Supported:

- Global SynFlow can fail structurally at severe sparsity by allocating zero weights to dense classifier bridges.
- This failure replicated on Fashion-MNIST CNN and CIFAR-10 CNN.
- Layerwise SynFlow is not sufficient in the tested CNNs.
- Tiny path correction can help one-shot pruning at `95-98%` on CIFAR, but the effect is modest and not a recovery win.

Not supported:

- Adaptive path correction is a universal replacement for magnitude pruning.
- Low-alpha path correction is the best fine-tuning initializer.
- Merely preventing dead hidden units is enough to improve severe pruning.
