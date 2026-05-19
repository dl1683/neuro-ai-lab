# neuro-ai-lab

Five experiments exploring neuroscience-AI crossovers. Each tests a specific hypothesis about how biological brains and artificial neural networks share computational principles.

## Vision

Brains and neural networks converge on similar solutions for similar reasons. This repo investigates five concrete instances where a known biological mechanism suggests a novel AI technique (or vice versa). Each experiment is designed to produce preliminary results in 1-2 days and, if successful, can spin off into a standalone research project.

## Experiments

| # | Name | Brain Mechanism | AI Application | Key Question |
|---|------|----------------|----------------|--------------|
| 01 | Grokking Prediction | Sudden insight (phase transition) | Predicting generalization onset | Can we detect grokking before it happens from internal representation metrics? |
| 02 | Sleep Training | NREM compression + REM testing cycles | Training schedule design | Does cycling between compression and testing phases outperform monotonic training? |
| 03 | Reconsolidation | Memory labilization on retrieval | Continual learning | Does making activated weights temporarily plastic reduce catastrophic forgetting? |
| 04 | Criticality Pruning | Sparse backbone (effective degree ~3) | Network pruning | Does preserving critical dynamics (branching ratio = 1) during pruning outperform magnitude pruning? |
| 05 | DDM as Depth | Drift-diffusion evidence accumulation | Understanding transformer depth | Do residual streams follow drift-diffusion dynamics across layers? |

## Shared Infrastructure

`shared/` contains utilities used across experiments:
- **criticality.py** — DFA exponents, branching ratio, avalanche size distributions, power-law fitting
- **representations.py** — Mean pairwise cosine (rho), neural collapse metrics (ETF distance), class centroid tracking
- **training.py** — Standard training loops with per-epoch representation checkpointing
- **visualization.py** — Standardized plots for criticality measures, training curves, representation dynamics

## Quick Start

```bash
pip install -r requirements.txt
cd experiments/01_grokking_prediction
python run.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- powerlaw (for power-law fitting)
- scikit-learn (for metrics)

## Status

All experiments are in **design phase**. See individual experiment READMEs for detailed protocols.
