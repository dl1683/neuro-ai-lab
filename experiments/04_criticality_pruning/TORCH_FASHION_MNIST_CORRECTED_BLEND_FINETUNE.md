# Torch Fashion-MNIST Corrected Blend Fine-Tuning

Masked fine-tuning after pruning Fashion-MNIST MLPs with magnitude vs corrected no-balance path blend at 95% and 98% sparsity.

| Sparsity | Method | Before FT | After FT | Retention after FT |
|---:|---|---:|---:|---:|
| `0.95` | `magnitude` | `0.6860` | `0.8310` | `0.9815` |
| `0.95` | `corrected_path_blend` | `0.7133` | `0.8340` | `0.9851` |
| `0.98` | `magnitude` | `0.5353` | `0.7965` | `0.9408` |
| `0.98` | `corrected_path_blend` | `0.6055` | `0.8033` | `0.9488` |
