# Fashion-MNIST CNN SynFlow Comparison

Fashion-MNIST CNN comparison against SynFlow. Adaptive dense-hybrid uses magnitude conv layers and path-corrected dense layers at 95/98% sparsity.

| Sparsity | Method | Mean accuracy | Std | Retention |
|---:|---|---:|---:|---:|
| `0.90` | `magnitude` | `0.7444` | `0.0166` | `0.8791` |
| `0.90` | `synflow` | `0.7650` | `0.0267` | `0.9038` |
| `0.90` | `adaptive_dense_hybrid` | `0.7444` | `0.0166` | `0.8791` |
| `0.95` | `magnitude` | `0.6122` | `0.0067` | `0.7231` |
| `0.95` | `synflow` | `0.6438` | `0.0110` | `0.7605` |
| `0.95` | `adaptive_dense_hybrid` | `0.6339` | `0.0324` | `0.7485` |
| `0.98` | `magnitude` | `0.4351` | `0.0384` | `0.5142` |
| `0.98` | `synflow` | `0.1008` | `0.0015` | `0.1190` |
| `0.98` | `adaptive_dense_hybrid` | `0.4946` | `0.0609` | `0.5839` |

Mean adaptive minus SynFlow: `0.1211`.
Mean adaptive minus magnitude: `0.0270`.
