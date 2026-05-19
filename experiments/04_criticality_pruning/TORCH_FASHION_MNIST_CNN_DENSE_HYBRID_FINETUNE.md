# Torch Fashion-MNIST CNN Dense-Hybrid Fine-Tuning

Masked fine-tuning after pruning Fashion-MNIST CNNs. Conv layers are magnitude-pruned; dense layers are either magnitude or dense-hybrid path-flow alpha=0.20.

| Sparsity | Method | Before FT | After FT | Retention after FT |
|---:|---|---:|---:|---:|
| `0.95` | `magnitude` | `0.6241` | `0.8469` | `1.0012` |
| `0.95` | `dense_hybrid` | `0.6354` | `0.8500` | `1.0049` |
| `0.98` | `magnitude` | `0.4391` | `0.8070` | `0.9541` |
| `0.98` | `dense_hybrid` | `0.4959` | `0.7955` | `0.9405` |
