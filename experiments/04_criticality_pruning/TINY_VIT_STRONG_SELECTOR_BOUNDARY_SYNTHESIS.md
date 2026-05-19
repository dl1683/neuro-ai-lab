# TinyViT Strong Selector Boundary Synthesis

Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.

Seeds synthesized: `15`
V3 positive vs magnitude: `9/15`
V3 matched best evaluated candidate: `9/15`
Mean V3 delta vs magnitude: `+0.0246`
Mean V3 gap to best candidate: `0.0079`
V4 positive vs magnitude: `9/15`
V4 matched best evaluated candidate: `11/15`
Mean V4 delta vs magnitude: `+0.0250`
Mean V4 gap to best candidate: `0.0075`
V5 positive vs magnitude: `10/15`
V5 matched best evaluated candidate: `13/15`
Mean V5 delta vs magnitude: `+0.0323`
Mean V5 gap to best candidate: `0.0002`
V6 positive vs magnitude: `11/15`
V6 matched best evaluated candidate: `14/15`
Mean V6 delta vs magnitude: `+0.0325`
Mean V6 gap to best candidate: `0.0001`
V7 positive vs magnitude: `11/15`
V7 matched best evaluated candidate: `15/15`
Mean V7 delta vs magnitude: `+0.0325`
Mean V7 gap to best candidate: `0.0000`

| Seed | V6 projected | V7 projected | Best evaluated | Magnitude after | V6 after | V7 after | Best after | V7 delta | V7 gap |
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
| `312` | `minimal_liveness_repair` | `minimal_liveness_repair` | `minimal_liveness_repair` | `0.1189` | `0.1210` | `0.1210` | `0.1210` | `+0.0021` | `0.0000` |
| `315` | `all_route_liveness_floor` | `magnitude` | `magnitude` | `0.1242` | `0.1234` | `0.1242` | `0.1242` | `+0.0000` | `0.0000` |
| `313` | `magnitude` | `magnitude` | `magnitude` | `0.1644` | `0.1644` | `0.1644` | `0.1644` | `+0.0000` | `0.0000` |
| `320` | `global_synflow` | `global_synflow` | `global_synflow` | `0.0901` | `0.1193` | `0.1193` | `0.1193` | `+0.0292` | `0.0000` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, the selector chooses SynFlow and the choice usually wins. When feature alignment favors liveness repair, the current rule is still weak. Seed 306 motivated the V5 SynFlow masked-recovery prior. Seed 310 prospectively validates that branch: V4-style liveness selection stays at the magnitude floor, while V5 selects SynFlow and matches the best evaluated candidate. Seed 312 motivated V6's live-repair masked-before tie-breaker, which fixes that case in projection. Seed 315 motivates V7: when a direct live-repair feature win over magnitude is tiny, keep magnitude. In projection, V7 fixes seed 315 without breaking the SynFlow branches or the seed-312 tie-breaker. Seed 320 prospectively validates that V7 still preserves the SynFlow feature branch.
