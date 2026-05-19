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
V5 positive vs magnitude: `6/8`
V5 matched best evaluated candidate: `8/8`
Mean V5 delta vs magnitude: `+0.0341`
Mean V5 gap to best candidate: `0.0000`

| Seed | V4 projected | V5 projected | Best evaluated | Magnitude after | V5 after | Best after | V5 delta | V5 gap |
|---:|---|---|---|---:|---:|---:|---:|---:|
| `298` | `magnitude` | `magnitude` | `magnitude` | `0.1670` | `0.1670` | `0.1670` | `+0.0000` | `0.0000` |
| `299` | `magnitude` | `magnitude` | `magnitude` | `0.1218` | `0.1218` | `0.1218` | `+0.0000` | `0.0000` |
| `300` | `all_route_liveness_floor` | `all_route_liveness_floor` | `all_route_liveness_floor` | `0.1382` | `0.1484` | `0.1484` | `+0.0102` | `0.0000` |
| `301` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1139` | `0.1452` | `0.1452` | `+0.0313` | `0.0000` |
| `302` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1199` | `0.1601` | `0.1601` | `+0.0402` | `0.0000` |
| `303` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0414` | `0.1408` | `0.1408` | `+0.0994` | `0.0000` |
| `304` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1022` | `0.1709` | `0.1709` | `+0.0687` | `0.0000` |
| `306` | `attn_mlp_readout_repair` | `global_synflow` | `global_synflow` | `0.1099` | `0.1329` | `0.1329` | `+0.0230` | `0.0000` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, the selector chooses SynFlow and the choice wins. When feature alignment favors liveness repair, the rule can beat magnitude but can still miss SynFlow. Seed 306 is the important correction: V4 selects attention+MLP repair from pre-finetune feature alignment and masked behavior; that selected repair beats magnitude but trails SynFlow. V5 adds a simple SynFlow masked-recovery prior: if SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow. On the completed boundary set, this fixes seed 306 without breaking the earlier guardrail cases. This is still a projection and needs a fresh prospective run.
