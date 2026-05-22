## Evidence-Gate Review (Experiment D)

### 1) Loss-function confound (E1 vs E5 vs encoder-only) — **High**

- `exp_d_compositional.py` explicitly reports different objectives: E1 uses `loss = MSE + 0.1*CE`, E5 uses `loss = λ₁*SC + CE`, encoder-only uses pure CE.  
  - Code path: [_e1_step](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py), [_e5_step](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py), [Encoder-only section](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d_compositional.py).
- E1 is an acknowledged design compromise (README note in `_e1_step`) and its CE is explicitly downweighted to just stabilize readout; this is an architectural confound when interpreting E1 failures.
- The claim “dynamics necessity confirmed” should not be inferred from E1’s 0.5077 token / 0.0 seq as evidence against alternatives, because E1 is not a pure dynamics ablation of the same objective.  
- This is **strong evidence of confound at the objective level**, and it weakens any mechanistic claim about dynamics unless a CE-matched ablation is added.

### 2) Single-seed validity / uncertainty — **High**

- `run()` and `train()` do not set seed per model run (no `seed` in config, no per-model seed loop), so this is effectively one run per condition.  
  - [exp_d_compositional.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d_compositional.py), [train()](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py).
- For `eval_samples=10000`, seq-level binomial CIs (95%) are broad enough that close margins are not meaningful:
  - Addition encoder-only `seq_acc=0.0011` (≈11/10k) => roughly Wilson ≈ `[0.0006, 0.0020]`.
  - Addition E5 `seq_acc=1.0000` => lower 95% bound ≈ `0.9996` for n=10k.
  - AR `seq_acc=0.9998` => lower 95% bound ≈ `0.99946`.
  - E5 vs AR gap of `0.0002` on seq is within overlap.
- Verdict: large gap (encoder-only vs E5) survives single-run noise in magnitude, but **statistical robustness is not established**.

### 3) Encoder-only capacity confound — **Medium**

- Encoder-only has 2 encoder layers and ~425k params; UESD variants have 694k params.  
  - [model sizes in results](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d_compositional.json).
- Claim about “encoder cannot do it” is actually “this 2-layer encoder with this width/params/dataset does poorly on this setting.”  
- A 4-layer encoder (or param-matched comparison) is a missing control; without it, this is not a clean necessity proof for recurrence at fixed compute capacity.

### 4) Training sufficiency — **Medium**

- Only one 20k-step training run per condition; no learning-curve variance over seeds.  
- Addition-E5 runs show non-monotonic CE/SC behavior in late phase (e.g., steps 14–20k), suggesting partial oscillation not clear asymptotic convergence.  
- The JSON contains full histories, but the protocol does not test continuation beyond 20k; sufficiency for convergence/stability is unproven.

### 5) Task design (base-64 4-digit addition) — **Medium**

- Task is framed as O(L) carry chains (at most 4 steps for 4-digit).  
  - [script rationale](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d_compositional.py).
- A 2-layer full-attention encoder has nontrivial global receptive path in one forward pass; it is therefore possible the limitation is partly representational/optimization at this scale rather than a hard asymptotic impossibility.
- No length sweep is present in this exp; no evidence that failure persists at longer addition sequences where sequential depth pressure increases further.

### 6) Dedup encoder confound acknowledged? — **Low**

- Yes, but only partially in the “necessity” framing.  
  - Dedup results show encoder-only at 0.9923 token / 0.9570 seq, and gate marks this as `CONCERN`.
- This supports the view that dynamics-necessity in D is **task-specific to addition as implemented**, not universal.
- The current narrative could overstate generality if read as “confirmed dynamics necessity generally.”

### 7) AR comparison fairness — **Medium**

- AR baseline is close (`seq=0.9998`) and may be “essentially tied” with E5 (`seq=1.0000`) given CI overlap.  
  - [AR eval](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d_compositional.json).
- But comparison is not capacity-matched: AR has 950k params vs 694k for UESD and different compute path (autoregressive decoding vs parallel updates).  
- Also note training/eval settings are same steps/samples, but architecture depth/parameters differ.

### 8) Statistical methodology (token vs seq; padding) — **Medium**

- Script explicitly warns token accuracy is inflated by trivial padding in addition (`Token accuracy includes trivial zeros. Use seq_acc as primary metric`).  
  - [exp_d_compositional.py note](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d_compositional.py).
- Gates still use token-accuracy thresholds for some decisions (e.g., dynamics-necessity gating), which is a design tension: seq should drive core claims for compositional correctness.
- If token_accuracy includes all positions, padding can dominate easy wins; seq_acc here is the correct primary metric.

## Verdict

- **Not publishable as-is for a “dynamics necessity” claim.**  
- **Status: Needs additional experiments and explicit caveats.**

Recommended minimum additions:
1. Seed sweep (≥5 seeds) with seed-averaged seq/token and confidence intervals.
2. Add a CE-matched recurrent/ablation control (e.g., CE-only UESD same architecture, no SC term).
3. Capacity-matched and depth-matched encoder-only baselines (same layers/params).
4. Addition length sweep (carry-chain depth >4) to stress sequential capacity.
5. Keep seq-accuracy as principal metric in claims; report token metrics masked by structural-padding positions.