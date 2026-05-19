# TinyViT Strong Selector Boundary Synthesis

Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.

Seeds synthesized: `8`
V3 positive vs magnitude: `6/8`
V3 matched best evaluated candidate: `5/8`
Mean V3 delta vs magnitude: `+0.0314`
Mean V3 gap to best candidate: `0.0028`
V4 positive vs magnitude: `6/8`
V4 matched best evaluated candidate: `7/8`
Mean V4 delta vs magnitude: `+0.0320`
Mean V4 gap to best candidate: `0.0021`

| Seed | V3 projected | V4 projected | Best evaluated | Magnitude after | V4 after | Best after | V4 delta | V4 gap |
|---:|---|---|---|---:|---:|---:|---:|---:|
| `298` | `all_route_liveness_floor` | `magnitude` | `magnitude` | `0.1670` | `0.1670` | `0.1670` | `+0.0000` | `0.0000` |
| `299` | `all_route_liveness_floor` | `magnitude` | `magnitude` | `0.1218` | `0.1218` | `0.1218` | `+0.0000` | `0.0000` |
| `300` | `all_route_liveness_floor` | `all_route_liveness_floor` | `all_route_liveness_floor` | `0.1382` | `0.1484` | `0.1484` | `+0.0102` | `0.0000` |
| `301` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1139` | `0.1452` | `0.1452` | `+0.0313` | `0.0000` |
| `302` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1199` | `0.1601` | `0.1601` | `+0.0402` | `0.0000` |
| `303` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0414` | `0.1408` | `0.1408` | `+0.0994` | `0.0000` |
| `304` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1022` | `0.1709` | `0.1709` | `+0.0687` | `0.0000` |
| `306` | `attn_mlp_readout_repair` | `attn_mlp_readout_repair` | `global_synflow` | `0.1099` | `0.1158` | `0.1329` | `+0.0059` | `0.0171` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, the selector chooses SynFlow and the choice wins. When feature alignment favors liveness repair, the rule can beat magnitude but can still miss SynFlow. Seed 306 is the important correction: V4 selects attention+MLP repair from pre-finetune feature alignment and masked behavior; that selected repair beats magnitude but trails SynFlow. The next selector therefore needs a SynFlow recovery-prior term, not only feature alignment and row liveness.
