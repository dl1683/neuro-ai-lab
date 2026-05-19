# SynFlow Pathology Synthesis

Global SynFlow can catastrophically starve dense classifier bridges in CNNs at severe global sparsity; layerwise SynFlow partially repairs allocation but remains far below magnitude after masked fine-tuning in the tested CNNs.

## Cross-dataset cases

| Dataset/model | Sparsity | Magnitude after FT | Global SynFlow after FT | Global delta | Global fc1 keep | Global dead fc1 | Layerwise after FT | Layerwise delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST CNN | `0.98` | `0.8086` | `0.1028` | `-0.7059` | `0.0000` | `128.0/128` | `0.5021` | `-0.3065` |
| CIFAR-10 CNN | `0.98` | `0.4408` | `0.0976` | `-0.3432` | `0.0000` | `192.0/192` | `0.2375` | `-0.2032` |
| CIFAR-10 CNN | `0.99` | `0.3324` | `0.0976` | `-0.2348` | `0.0000` | `192.0/192` | `0.1758` | `-0.1566` |

## Aggregate

- Cases: `3`.
- Global SynFlow zero-fc1 cases: `3/3`.
- Mean global SynFlow after-FT delta vs magnitude: `-0.4280`.
- Mean layerwise SynFlow after-FT delta vs magnitude: `-0.2221`.

## Interpretation

The failure is structural, not just a modest score-quality difference. In every synthesized severe-sparsity CNN case, global SynFlow assigns zero parameters to `fc1`, so the dense classifier bridge is absent and masked fine-tuning cannot recover. Layerwise SynFlow restores a nominal per-layer budget, but still trails magnitude badly, which means the SynFlow ranking inside the dense bridge is also weak in these settings.

Practical guardrail: severe global pruning methods should emit dense-bridge diagnostics before their scores are trusted: per-layer keep rate, dead hidden bridge units, and after-prune reachability.
