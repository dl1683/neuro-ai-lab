# Fashion-MNIST MLP SynFlow Comparison

Fashion-MNIST MLP comparison against SynFlow. Adaptive path uses alpha 0 at 90%, 0.25 at 95%, 0.50 at 98%.

| Sparsity | Method | Mean accuracy | Std | Retention |
|---:|---|---:|---:|---:|
| `0.90` | `magnitude` | `0.8032` | `0.0215` | `0.9485` |
| `0.90` | `gradient_saliency` | `0.7100` | `0.0417` | `0.8382` |
| `0.90` | `synflow` | `0.7905` | `0.0055` | `0.9337` |
| `0.90` | `adaptive_path` | `0.8032` | `0.0215` | `0.9485` |
| `0.95` | `magnitude` | `0.6860` | `0.0340` | `0.8100` |
| `0.95` | `gradient_saliency` | `0.5982` | `0.0415` | `0.7061` |
| `0.95` | `synflow` | `0.6868` | `0.0022` | `0.8113` |
| `0.95` | `adaptive_path` | `0.7133` | `0.0060` | `0.8425` |
| `0.98` | `magnitude` | `0.5353` | `0.0573` | `0.6318` |
| `0.98` | `gradient_saliency` | `0.3478` | `0.0258` | `0.4106` |
| `0.98` | `synflow` | `0.4425` | `0.0165` | `0.5225` |
| `0.98` | `adaptive_path` | `0.5530` | `0.0063` | `0.6533` |

Mean adaptive minus SynFlow: `0.0499`.
Mean adaptive minus magnitude: `0.0150`.
