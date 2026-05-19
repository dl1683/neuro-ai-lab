# SynFlow CNN Layerwise Rescue at 98% Sparsity

Fashion-MNIST CNN 98% pruning rescue test. Layerwise SynFlow keeps 2% inside each layer instead of allowing global SynFlow to starve fc1.

| Method | Before FT | After FT | After retention | fc1 keep rate | Dead fc1 hidden |
|---|---:|---:|---:|---:|---:|
| `magnitude` | `0.4439` | `0.8341` | `0.9864` | `0.0099` | `36.0` |
| `global_synflow` | `0.1008` | `0.1028` | `0.1215` | `0.0000` | `128.0` |
| `layerwise_synflow` | `0.1472` | `0.5021` | `0.5937` | `0.0200` | `37.0` |
| `adaptive_dense_hybrid` | `0.4861` | `0.8244` | `0.9749` | `0.0055` | `38.0` |

Interpretation: this isolates global allocation from the SynFlow score itself. If layerwise SynFlow recovers while global SynFlow stays at chance, the failure is not simply `SynFlow bad`; it is global score allocation starving the dense bridge.
