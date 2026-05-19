# Torch Fashion-MNIST Layerwise Pruning Check

Check whether Fashion-MNIST path-flow failure is caused by global pruning allocation by comparing global vs layerwise masks.

| Sparsity | Method | Mean accuracy | Std | Retention |
|---:|---|---:|---:|---:|
| `0.90` | `magnitude_global` | `0.8032` | `0.0215` | `0.9485` |
| `0.90` | `path_flow_global` | `0.5768` | `0.0545` | `0.6808` |
| `0.90` | `magnitude_layerwise` | `0.4665` | `0.0698` | `0.5503` |
| `0.90` | `gradient_layerwise` | `0.3453` | `0.0970` | `0.4070` |
| `0.90` | `path_flow_layerwise` | `0.4385` | `0.0088` | `0.5180` |
| `0.95` | `magnitude_global` | `0.6860` | `0.0340` | `0.8100` |
| `0.95` | `path_flow_global` | `0.4833` | `0.0167` | `0.5707` |
| `0.95` | `magnitude_layerwise` | `0.4063` | `0.0817` | `0.4792` |
| `0.95` | `gradient_layerwise` | `0.2010` | `0.0847` | `0.2366` |
| `0.95` | `path_flow_layerwise` | `0.3572` | `0.1632` | `0.4235` |
| `0.98` | `magnitude_global` | `0.5353` | `0.0573` | `0.6318` |
| `0.98` | `path_flow_global` | `0.3470` | `0.1187` | `0.4110` |
| `0.98` | `magnitude_layerwise` | `0.2513` | `0.0277` | `0.2966` |
| `0.98` | `gradient_layerwise` | `0.1565` | `0.0578` | `0.1843` |
| `0.98` | `path_flow_layerwise` | `0.1947` | `0.0463` | `0.2304` |
