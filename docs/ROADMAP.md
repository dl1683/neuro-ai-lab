# Roadmap: From Circuit Viability to a Stronger Pruning Theory

This roadmap starts from the current evidence in `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`.

## Current scientific status

The repo has moved from a pathology to a mechanism:

1. Global SynFlow can create hidden cutsets by deleting dense classifier bridges.
2. Capacity constraints rescue the collapse in CNN settings.
3. Residual networks require a hierarchy of constraints:
   - liveness/homeostasis first;
   - route-family targets second;
   - degeneracy/overconcentration control third.
4. Deep residual transfer is positive, but the simplest reserve rule is strongest there.

## Immediate priorities

### 1. Standard residual benchmark

Goal: show the mechanism is not tied to custom TinyResNet variants.

Candidate:

- CIFAR ResNet-18 or small ResNet-20.

Minimum experiment:

- magnitude;
- global SynFlow;
- plain capacity reserve;
- tuned route split;
- diversity route optimizer.

Success bar:

- capacity method beats magnitude on mean or clearly explains when magnitude wins;
- dead-output and route-quality diagnostics predict the result.

### 2. Derive diversity penalty weights

Current diversity optimizer works in TinyResNet but uses hand-set penalties.

Needed:

- measure route-family sensitivity by perturbing protected capacity;
- set concentration penalties from sensitivity;
- avoid hand-picked projection-overuse thresholds.

Success bar:

- derived penalties match or beat fixed penalties on fresh seeds.

### 3. Route-quality predictor audit

Goal: predict recoverability before fine-tuning.

Metrics:

- dead outputs;
- route min;
- projection min;
- readout score;
- main-path min;
- route-family concentration;
- deviation from magnitude viability template.

Success bar:

- predictor ranks masks in the same order as after-FT accuracy across methods and seeds.

### 4. Transformer analogue

Do not jump straight to LLM scale. First define route families.

Candidate analogues:

- MLP up/down projection routes;
- attention head output routes;
- residual-stream route capacity;
- layernorm-mediated bottlenecks.

Minimum experiment:

- small transformer or ViT-like model;
- severe unstructured pruning;
- route diagnostics for MLP/readout/residual families.

Success bar:

- identify a nontrivial route-collapse failure or show the hierarchy does not apply.

### 5. Claim-audit automation

Goal: make the repo hard to dismiss.

Implemented:

- `experiments/04_criticality_pruning/audit_circuit_viability_claims.py`
- `docs/CLAIM_AUDIT.md`

The script checks:

- every README headline number appears in a result JSON;
- every claim in `docs/CLAIM_EVIDENCE_LEDGER.md` has an artifact;
- no unsupported claim says more than the evidence.

Current status:

- `230/230` headline checks pass.

Next improvement:

- expand the audit to parse more claims from docs automatically instead of using explicit checks.

## Public claim to protect

Use this claim until stronger evidence exists:

> Severe pruning can create hidden circuit cutsets. Capacity constraints inspired by circuit viability prevent these failures and can improve extreme-sparsity recovery across CNN and residual settings. The evidence supports a hierarchy: preserve liveness first, then route-family balance and degeneracy.

Do not claim:

- pruning is solved;
- route-capacity always beats magnitude;
- current penalty weights are theoretically derived;
- transformer transfer has been shown.
