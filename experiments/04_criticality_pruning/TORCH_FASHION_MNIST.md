# Torch Fashion-MNIST Path-Flow Check

Torch MLP trained on 16k Fashion-MNIST examples, evaluated on 3k examples. One-shot pruning of fc1/fc2 at high sparsity using magnitude, gradient saliency, and label-free path-flow.

Device: `cuda`

| Sparsity | Method | Mean accuracy | Std | Mean retention |
|---:|---|---:|---:|---:|
| `0.90` | `magnitude` | `0.8032` | `0.0215` | `0.9485` |
| `0.90` | `gradient_saliency` | `0.7100` | `0.0417` | `0.8382` |
| `0.90` | `path_flow` | `0.5768` | `0.0545` | `0.6808` |
| `0.95` | `magnitude` | `0.6860` | `0.0340` | `0.8100` |
| `0.95` | `gradient_saliency` | `0.5982` | `0.0415` | `0.7061` |
| `0.95` | `path_flow` | `0.4833` | `0.0167` | `0.5707` |
| `0.98` | `magnitude` | `0.5353` | `0.0573` | `0.6318` |
| `0.98` | `gradient_saliency` | `0.3478` | `0.0258` | `0.4106` |
| `0.98` | `path_flow` | `0.3470` | `0.1187` | `0.4110` |
