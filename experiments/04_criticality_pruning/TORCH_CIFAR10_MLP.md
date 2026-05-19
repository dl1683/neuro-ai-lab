# Torch CIFAR-10 MLP Path-Flow Exploration

Exploratory CIFAR-10 MLP test. Train on 20k examples, evaluate on 5k, compare magnitude/path blends alpha=0..1 and gradient saliency at 90/95/98% sparsity.

Device: `cuda`

| Alpha | Mean accuracy over 90/95/98% |
|---:|---:|
| `0.00` | `0.3399` |
| `0.05` | `0.3395` |
| `0.10` | `0.3418` |
| `0.25` | `0.3441` |
| `0.50` | `0.3442` |
| `1.00` | `0.3415` |

## Per-sparsity summary

| Sparsity | Method | Accuracy | Retention |
|---:|---|---:|---:|
| `0.90` | `gradient_saliency` | `0.3307` | `0.6927` |
| `0.90` | `path_alpha_0.0` | `0.4101` | `0.8583` |
| `0.90` | `path_alpha_0.05` | `0.4114` | `0.8610` |
| `0.90` | `path_alpha_0.1` | `0.4096` | `0.8572` |
| `0.90` | `path_alpha_0.25` | `0.4123` | `0.8629` |
| `0.90` | `path_alpha_0.5` | `0.4034` | `0.8443` |
| `0.90` | `path_alpha_1.0` | `0.3960` | `0.8290` |
| `0.95` | `gradient_saliency` | `0.2723` | `0.5703` |
| `0.95` | `path_alpha_0.0` | `0.3461` | `0.7237` |
| `0.95` | `path_alpha_0.05` | `0.3430` | `0.7173` |
| `0.95` | `path_alpha_0.1` | `0.3469` | `0.7254` |
| `0.95` | `path_alpha_0.25` | `0.3488` | `0.7294` |
| `0.95` | `path_alpha_0.5` | `0.3460` | `0.7237` |
| `0.95` | `path_alpha_1.0` | `0.3447` | `0.7214` |
| `0.98` | `gradient_saliency` | `0.2249` | `0.4710` |
| `0.98` | `path_alpha_0.0` | `0.2634` | `0.5508` |
| `0.98` | `path_alpha_0.05` | `0.2642` | `0.5526` |
| `0.98` | `path_alpha_0.1` | `0.2690` | `0.5627` |
| `0.98` | `path_alpha_0.25` | `0.2713` | `0.5674` |
| `0.98` | `path_alpha_0.5` | `0.2832` | `0.5925` |
| `0.98` | `path_alpha_1.0` | `0.2839` | `0.5941` |
