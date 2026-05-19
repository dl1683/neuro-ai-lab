# TinyViT Strong Selector Boundary Synthesis

Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.

Seeds synthesized: `7`
V3 positive vs magnitude: `5/7`
V3 matched best evaluated candidate: `5/7`
Mean V3 delta vs magnitude: `+0.0350`
Mean V3 gap to best candidate: `0.0007`
V4 positive vs magnitude: `5/7`
V4 matched best evaluated candidate: `7/7`
Mean V4 delta vs magnitude: `+0.0357`
Mean V4 gap to best candidate: `0.0000`

| Seed | V3 projected | V4 projected | Best evaluated | Magnitude after | V4 after | Best after | V4 delta | V4 gap |
|---:|---|---|---|---:|---:|---:|---:|---:|
| `298` | `all_route_liveness_floor` | `magnitude` | `magnitude` | `0.1670` | `0.1670` | `0.1670` | `+0.0000` | `0.0000` |
| `299` | `all_route_liveness_floor` | `magnitude` | `magnitude` | `0.1218` | `0.1218` | `0.1218` | `+0.0000` | `0.0000` |
| `300` | `all_route_liveness_floor` | `all_route_liveness_floor` | `all_route_liveness_floor` | `0.1382` | `0.1484` | `0.1484` | `+0.0102` | `0.0000` |
| `301` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1139` | `0.1452` | `0.1452` | `+0.0313` | `0.0000` |
| `302` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1199` | `0.1601` | `0.1601` | `+0.0402` | `0.0000` |
| `303` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0414` | `0.1408` | `0.1408` | `+0.0994` | `0.0000` |
| `304` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1022` | `0.1709` | `0.1709` | `+0.0687` | `0.0000` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, V3 chooses SynFlow and the choice wins. When the margin is small and route death is high, V3 routes to a trainability or liveness guardrail. V4 adds masked pre-finetune accuracy to that ambiguous branch. On the completed strong TinyViT seeds, that prospective diagnostic removes the two V3 guardrail misses and matches the best evaluated candidate on every seed. This is still a projection over completed candidate evaluations, but it defines the next prospective validation target precisely.
