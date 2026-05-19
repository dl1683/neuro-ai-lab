# SynFlow CNN Mask Forensics at 98% Sparsity

Fashion-MNIST CNN 98% global pruning mask forensics. Measures structural mask damage for magnitude, SynFlow, and adaptive dense-hybrid pruning.

| Method | Accuracy | Retention | Dead conv1 | Dead conv2 | Dead fc1 hidden | Dead classes | Min class fan-in |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4675` | `0.5530` | `0.0` | `0.0` | `36.0` | `0.0` | `38.0` |
| `synflow` | `0.1008` | `0.1192` | `0.0` | `0.0` | `128.0` | `0.0` | `112.0` |
| `adaptive_dense_hybrid` | `0.4775` | `0.5645` | `0.0` | `0.0` | `39.0` | `0.0` | `29.5` |

## Mean layer keep rates

| Method | conv1 | conv2 | fc1 | fc2 |
|---|---:|---:|---:|---:|
| `magnitude` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `synflow` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `adaptive_dense_hybrid` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
