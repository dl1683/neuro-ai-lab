# TinyViT Strong Selector Boundary Synthesis

Post-hoc synthesis across all completed strong TinyViT 90% sparsity runs. Applies the same V3 selector rule to every evaluated seed and compares the projected choice with the best evaluated candidate.

Seeds synthesized: `13`
V3 positive vs magnitude: `8/13`
V3 matched best evaluated candidate: `8/13`
Mean V3 delta vs magnitude: `+0.0262`
Mean V3 gap to best candidate: `0.0090`
V4 positive vs magnitude: `8/13`
V4 matched best evaluated candidate: `10/13`
Mean V4 delta vs magnitude: `+0.0266`
Mean V4 gap to best candidate: `0.0086`
V5 positive vs magnitude: `9/13`
V5 matched best evaluated candidate: `12/13`
Mean V5 delta vs magnitude: `+0.0351`
Mean V5 gap to best candidate: `0.0002`
V6 positive vs magnitude: `10/13`
V6 matched best evaluated candidate: `13/13`
Mean V6 delta vs magnitude: `+0.0353`
Mean V6 gap to best candidate: `0.0000`

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
| `313` | `magnitude` | `magnitude` | `magnitude` | `0.1644` | `0.1644` | `0.1644` | `0.1644` | `+0.0000` | `0.0000` |

## Interpretation

This is not a new training run; it is a rule projection over all completed strong TinyViT candidate evaluations. The result is useful because the selector is applied before seeing fine-tune recovery, while the scorecard compares that choice against the evaluated recovery.

The boundary is now concrete. When SynFlow's centered CLS/residual-stream feature margin is large, the selector chooses SynFlow and the choice usually wins. When feature alignment favors liveness repair, V5 can still miss the exact repair family. Seed 306 motivated the V5 SynFlow masked-recovery prior. Seed 310 prospectively validates that branch: V4-style liveness selection stays at the magnitude floor, while V5 selects SynFlow and matches the best evaluated candidate. Seed 312 is the live-repair limitation: V5 chooses attention+MLP repair by a small feature margin, but minimal liveness is best and the selected policy slightly trails magnitude. V6 adds the narrow tie-breaker this failure suggests: when live-repair feature margins are tiny, choose the live repair with better masked-before trainability. In projection, V6 fixes seed 312 and reaches the evaluated oracle on the current boundary set. Seed 313 prospectively validates that V6 does not override magnitude when magnitude is the most trainable sparse template.
