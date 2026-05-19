# TinyViT Strong Selector Boundary Synthesis

Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.

Seeds synthesized: `14`
V3 positive vs magnitude: `8/14`
V3 matched best evaluated candidate: `8/14`
Mean V3 delta vs magnitude: `+0.0243`
Mean V3 gap to best candidate: `0.0084`
V4 positive vs magnitude: `8/14`
V4 matched best evaluated candidate: `10/14`
Mean V4 delta vs magnitude: `+0.0247`
Mean V4 gap to best candidate: `0.0081`
V5 positive vs magnitude: `9/14`
V5 matched best evaluated candidate: `12/14`
Mean V5 delta vs magnitude: `+0.0325`
Mean V5 gap to best candidate: `0.0002`
V6 positive vs magnitude: `10/14`
V6 matched best evaluated candidate: `13/14`
Mean V6 delta vs magnitude: `+0.0327`
Mean V6 gap to best candidate: `0.0001`

| Seed | V5 projected | V6 projected | Best evaluated | Magnitude after | V5 after | V6 after | Best after | V6 delta | V6 gap |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| `298` | `magnitude` | `magnitude` | `magnitude` | `0.1670` | `0.1670` | `0.1670` | `0.1670` | `+0.0000` | `0.0000` |
| `299` | `magnitude` | `magnitude` | `magnitude` | `0.1218` | `0.1218` | `0.1218` | `0.1218` | `+0.0000` | `0.0000` |
| `300` | `all_route_liveness_floor` | `all_route_liveness_floor` | `all_route_liveness_floor` | `0.1382` | `0.1484` | `0.1484` | `0.1484` | `+0.0102` | `0.0000` |
| `301` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1139` | `0.1452` | `0.1452` | `0.1452` | `+0.0313` | `0.0000` |
| `302` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1199` | `0.1601` | `0.1601` | `0.1601` | `+0.0402` | `0.0000` |
| `303` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0414` | `0.1408` | `0.1408` | `0.1408` | `+0.0994` | `0.0000` |
| `304` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1022` | `0.1709` | `0.1709` | `0.1709` | `+0.0687` | `0.0000` |
| `306` | `global_synflow` | `global_synflow` | `global_synflow` | `0.1099` | `0.1329` | `0.1329` | `0.1329` | `+0.0230` | `0.0000` |
| `308` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0725` | `0.1018` | `0.1018` | `0.1018` | `+0.0293` | `0.0000` |
| `310` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0586` | `0.1516` | `0.1516` | `0.1516` | `+0.0930` | `0.0000` |
| `311` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0979` | `0.1591` | `0.1591` | `0.1591` | `+0.0612` | `0.0000` |
| `312` | `attn_mlp_readout_repair` | `minimal_liveness_repair` | `minimal_liveness_repair` | `0.1189` | `0.1187` | `0.1210` | `0.1210` | `+0.0021` | `0.0000` |
| `315` | `all_route_liveness_floor` | `all_route_liveness_floor` | `magnitude` | `0.1242` | `0.1234` | `0.1234` | `0.1242` | `-0.0008` | `0.0008` |
| `313` | `magnitude` | `magnitude` | `magnitude` | `0.1644` | `0.1644` | `0.1644` | `0.1644` | `+0.0000` | `0.0000` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, the selector chooses SynFlow and the choice usually wins. When feature alignment favors liveness repair, the current rule is still weak. Seed 306 motivated the V5 SynFlow masked-recovery prior. Seed 310 prospectively validates that branch: V4-style liveness selection stays at the magnitude floor, while V5 selects SynFlow and matches the best evaluated candidate. Seed 312 motivated V6's live-repair masked-before tie-breaker, which fixes that case in projection. But seed 315 is the new limitation: V6 selects all-route liveness from pre-finetune diagnostics, yet magnitude is slightly better after fine-tuning. Seed 313 prospectively validates that V6 can keep magnitude when magnitude is clearly strongest, but the live-repair branch still needs a better magnitude-vs-repair guardrail.
