You are reviewing three new experiment designs as a senior architectural authority.

Read and analyze these files:
- experiments/06_uesd/exp_d11_energy_landscape.py (energy landscape cartography)
- experiments/06_uesd/exp_d12_langevin_escape.py (Langevin noise injection)
- experiments/06_uesd/exp_d13_dynamics_transfer.py (cross-task dynamics transfer)
- experiments/06_uesd/shared/model.py (UESD architecture)
- experiments/06_uesd/shared/data.py (data generators)
- experiments/06_uesd/shared/training.py (training loops)
- experiments/06_uesd/results/codex_d8_design_review.md (prior review for context)
- experiments/06_uesd/results/codex_d10_design_review.md (prior review for context)

CONTEXT:
These are the most ambitious experiments in the UESD series. They go beyond mechanistic analysis into testing deep theoretical claims:

D11 maps the energy landscape E(s,c) = ||F_theta(s,c)||^2 to test whether thinking is trajectory through high-energy regions. It measures basin structure, radius, PCA landscape slices, and path efficiency.

D12 tests Langevin dynamics (noise injection at inference time) on trained models. The hypothesis is that noise can rescue wrong-attractor failures by helping escape bad basins. This tests Section 4.2 of the UESD formalization.

D13 tests cross-task transfer: train dynamics on addition, freeze them, and train only encoder+readout on subtraction/comparison/multiplication. If the dynamics transfer, they encode general iterative computation, not task-specific rules.

Apply ALL of the following, for EACH experiment:

1. CORRECTNESS AUDIT
   For every function, check data flow, index handling, numerical stability.
   Special attention to:
   - D11: Is compute_energy correct? PCA computation? Grid evaluation?
   - D12: Is noise injection correct? Energy tracking before noise? Schedule functions?
   - D13: Are subtraction/comparison/multiplication generators correct?
   - D13: Does freezing dynamics actually freeze all the right parameters?

2. STATISTICAL PLAN
   - Are sample sizes adequate?
   - Are the right comparisons being made?
   - What statistical tests are needed?

3. ALTERNATIVE EXPLANATIONS
   For each prediction, what else could produce the same result?
   - D11: Could basin clustering be an artifact of cosine similarity threshold?
   - D12: Could noise just be regularizing, not escaping basins?
   - D13: Could transfer success just mean the task is easy, not that dynamics transfer?

4. DESIGN GAPS
   - What controls are missing?
   - What baselines are needed?
   - How do these experiments connect to each other and to D7-D10?

5. PRIORITY DIRECTIVE
   For EACH experiment, what is the single most important fix before running?

6. PREDICTION
   Your predictions for each experiment, with confidence levels and evidence.

7. PARSIMONY MANDATE
   Can any experiment be simplified without losing signal?
