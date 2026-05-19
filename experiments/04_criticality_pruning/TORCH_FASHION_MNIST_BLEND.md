# Torch Fashion-MNIST Magnitude/Path-Flow Blend

Search conservative blends between magnitude and path-flow on Fashion-MNIST MLP. alpha=0 is magnitude; alpha=1 is full path-flow modulation.

Best alpha: `0.0`

| Alpha | Mean accuracy over 90/95/98% |
|---:|---:|
| `0.00` | `0.6748` |
| `0.05` | `0.6656` |
| `0.10` | `0.6483` |
| `0.20` | `0.6114` |
| `0.35` | `0.5709` |
| `0.50` | `0.5242` |
| `0.75` | `0.4955` |
| `1.00` | `0.4691` |
