# Path-Flow Post-Prune Fine-Tuning

This experiment checks whether path-flow masks remain good after masked fine-tuning, not just immediately after one-shot pruning.

Setup: trained sklearn-digits MLPs, `90%` and `95%` sparsity, five seeds, 55 masked fine-tuning epochs.

| Method | Before FT | After FT | Recovery gain | Retention after FT |
|---|---:|---:|---:|---:|
| `magnitude` | `0.6146` | `0.7610` | `0.1464` | `0.7939` |
| `gradient_saliency` | `0.7814` | `0.8874` | `0.1060` | `0.9257` |
| `path_flow` | `0.7890` | `0.8903` | `0.1013` | `0.9287` |

Path-flow after-FT gain vs magnitude: `0.1293`.
Path-flow after-FT gain vs gradient saliency: `0.0029`.
