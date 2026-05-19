# Neuro-AI Lab

Real experiments at the boundary of neuroscience-inspired mechanisms and neural network behavior.

The current standout result is not a toy demo: **global SynFlow can catastrophically fail at severe CNN sparsity by allocating zero weights to the dense classifier bridge.** The failure replicated on Fashion-MNIST and CIFAR-10 CNNs, survived masked fine-tuning checks, and is packaged with diagnostics so future pruning runs can catch the pathology directly.

## Headline finding

At `98-99%` global sparsity, global SynFlow can assign `0` surviving weights to `fc1`, the first dense classifier bridge in CNNs. Once that bridge is gone, masked fine-tuning cannot recover the model.

Cross-dataset synthesis:

| Case | Magnitude after FT | Global SynFlow after FT | Delta | Global `fc1` keep | Dead bridge units |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST CNN, `98%` | `80.86%` | `10.28%` | `-70.59` pts | `0.0000` | `128/128` |
| CIFAR-10 CNN, `98%` | `44.08%` | `9.76%` | `-34.32` pts | `0.0000` | `192/192` |
| CIFAR-10 CNN, `99%` | `33.24%` | `9.76%` | `-23.48` pts | `0.0000` | `192/192` |

Aggregate:

- Global SynFlow zero-bridge cases: `3/3`.
- Mean after-fine-tuning delta vs magnitude: `-42.80` points.
- Layerwise SynFlow partially repairs allocation, but still averages `-22.21` points vs magnitude after fine-tuning.

Primary artifact:

- `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md`

## Why this matters

SynFlow is often treated as a label-free pruning baseline. These experiments show a concrete severe-sparsity failure mode: a global saliency allocation can preserve early convolutional weights while deleting the dense bridge required for classification. The scalar pruning score looks valid, but the mask is structurally unrecoverable.

The practical guardrail is simple: severe global pruning methods should report per-layer keep rates, dead bridge units, and classifier reachability before their scores are trusted.

Reusable diagnostic code lives in:

- `shared/pruning_diagnostics.py`

## Secondary result

A tiny dense-tail path correction can improve one-shot severe pruning, but it is not a universal replacement for magnitude pruning.

Six-seed CIFAR-10 GPU replicate:

| Sparsity | Alpha | One-shot delta vs magnitude | One-shot wins | After-FT delta vs magnitude | After-FT wins |
|---:|---:|---:|---:|---:|---:|
| `95%` | `0.03` | `+1.49` pts | `5/6` | `-0.44` pts | `1/6` |
| `98%` | `0.03` | `+1.13` pts | `4/6` | `-0.29` pts | `3/6` |
| `99%` | `0.03` | `-0.19` pts | `3/6` | `-1.42` pts | `1/6` |

Current rule encoded in `shared/adaptive_path_pruning.py`:

- `balanced` / `one_shot`: tiny `alpha=0.03` near the severe sparsity cliff.
- `recovery`: `alpha=0.0` unless a domain-specific sweep proves otherwise.

## Repository map

| Path | Purpose |
|---|---|
| `experiments/04_criticality_pruning/README.md` | Main experiment navigation page. |
| `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md` | Strongest cross-dataset result. |
| `experiments/04_criticality_pruning/LOW_ALPHA_TRANSFER_SYNTHESIS.md` | Secondary low-alpha pruning result and limitations. |
| `experiments/04_criticality_pruning/CIFAR10_CNN_SYNFLOW_PATHOLOGY.md` | CIFAR-10 CUDA replication of the SynFlow failure. |
| `shared/pruning_diagnostics.py` | Structural mask diagnostics for dense-bridge collapse. |
| `shared/adaptive_path_pruning.py` | Reusable low-alpha dense-tail path correction utilities. |
| `results/04_criticality_pruning/` | JSON result artifacts used by synthesis/audit scripts. |

## Reproduce the strongest claim

The synthesis and audit are lightweight because they read checked-in result artifacts:

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
```

To rerun the full CIFAR pathology experiment on GPU:

```powershell
python experiments\04_criticality_pruning\cifar10_cnn_synflow_pathology.py
```

That command trains small CIFAR-10 CNNs, compares magnitude/global SynFlow/layerwise SynFlow at `98%` and `99%` sparsity, and writes:

- `results/04_criticality_pruning/cifar10_cnn_synflow_pathology.json`
- `experiments/04_criticality_pruning/CIFAR10_CNN_SYNFLOW_PATHOLOGY.md`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments use PyTorch and torchvision. GPU is optional for synthesis/audits, but recommended for rerunning CIFAR training.

## Claim boundaries

Supported by current artifacts:

- Global SynFlow can structurally fail at severe CNN sparsity by allocating zero weights to dense classifier bridges.
- The failure replicated on Fashion-MNIST CNN and CIFAR-10 CNN.
- Layerwise SynFlow is not sufficient in the tested CNNs.
- Tiny path correction can modestly improve one-shot severe pruning around `95-98%` on CIFAR.

Not supported:

- Adaptive path correction is a universal replacement for magnitude pruning.
- Low-alpha path correction is the best fine-tuning initializer.
- Merely forcing every hidden unit to stay live is enough to improve masks.

## Status

The repo has moved beyond the initial pilot phase. The strongest current contribution is the SynFlow dense-bridge collapse finding plus reusable diagnostics for severe-sparsity pruning audits.
