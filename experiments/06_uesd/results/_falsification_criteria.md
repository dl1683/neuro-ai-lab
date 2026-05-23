# UESD Falsification Criteria

These criteria define the conditions under which each thesis component would be
considered weakened or falsified. All future Codex reviews should evaluate results
against these criteria.

## Thesis Components

### T1: "Iterative dynamics provide essential computation"
- **Test**: D19 step ablation
- **WEAKENED if**: seq_acc(T=1) / seq_acc(T=10) >= 0.98
- **FALSIFIED if**: seq_acc(T=1) == seq_acc(T=10) for both CE-dynamics and E5
- **SUPPORTED if**: ratio < 0.50 (dynamics provide >50% of accuracy)

### T2: "Softmax bottleneck necessitates iterative bypass"
- **Test**: D20 bottleneck sweep
- **WEAKENED if**: accuracy range < 0.05 across 4x V variation AND step-dependence range < 0.05
- **FALSIFIED if**: accuracy AND step-dependence AND recovery are all flat across V
- **SUPPORTED if**: clear monotonic relationship between log2(V) and step-dependence

### T3: "Dynamics create stable iterative solver"
- **Test**: D21 wrong-attractor rate
- **WEAKENED if**: WA > 5% at sigma=0.1 (relative to state norm) AND doesn't recover with extra steps
- **FALSIFIED if**: WA > 20% at sigma=0.05 with no recovery at +20 steps
- **SUPPORTED if**: basin escape threshold > 0.5 * state_norm AND recovery at +20 reduces WA to < 2%

### T4: "Self-consistency loss creates meaningful energy landscape"
- **Test**: Compare E5 vs CE-dynamics across D19/D20/D21
- **WEAKENED if**: E5 shows no advantage in step-dependence, bottleneck sensitivity, or basin stability
- **FALSIFIED if**: E5 performs worse than CE-dynamics on all three metrics
- **SUPPORTED if**: E5 shows stronger step-dependence, tighter basins, or better recovery

### T5: "Parallel computation engine (not sequential reasoning)"
- **Test**: D7 carry-chain depth correlation, D10 adaptive halting
- **WEAKENED if**: strong positive correlation (r > 0.5) between carry depth and convergence steps
- **FALSIFIED if**: convergence steps scale linearly with carry depth (r > 0.8)
- **SUPPORTED if**: r < 0.2 and all positions stabilize at same step

### T6: "Carry representations are decorative, not causal"
- **Test**: D8 causal surgery
- **WEAKENED if**: output change persists for > 5 steps after carry flip
- **FALSIFIED if**: carry flip causes permanent output change proportional to intervention
- **SUPPORTED if**: zero output change at all positions/steps despite 100% carry flip success

## Confirmation Bias Guards

When reviewing results, Codex MUST check:
1. Are we only testing conditions where the thesis can be confirmed?
2. Are null results being explained away rather than taken at face value?
3. Are we using per-seed outliers to support claims while treating failures as "noise"?
4. Is the same data being used to both suggest AND test a hypothesis?
5. Are effect sizes meaningful or just statistically significant?

## Scoring Rubric

For each thesis component, rate:
- **PASS**: Evidence strongly supports; falsification criteria not triggered
- **INCONCLUSIVE**: Mixed evidence; cannot confirm or deny
- **WEAKENED**: Some falsification criteria triggered
- **FALSIFIED**: Core falsification criteria triggered

Overall thesis health = minimum score across all components.
