You are reviewing experiment code for correctness before launch.

Read and review: experiments/06_uesd/exp_d31_d8_multiseed.py

Also read the reference implementation it is based on: experiments/06_uesd/exp_d28_contraction_ratio.py

CONTEXT:
D31 is a multi-seed replication of D=8 contraction ratio measurement.
D28 showed Δρ (rho_VT - rho_FT) ≈ -0.0025 at 4/5 depths, but D=8 had Δρ = +0.0014 (20σ outlier).
D31 runs 8 seeds × 2 variants at D=8 to determine if this is real or a seed anomaly.
Also includes 3-seed controls at D=6 and D=10.

Total runs: (8 seeds × 2 variants × 1 depth) + (3 seeds × 2 variants × 2 depths) = 28 runs.
Each run: 20K training steps + spectral radius + contraction measurement.

CHECK FOR:
1. BUGS: Off-by-one errors, wrong variable references, tensor shape mismatches
2. SEED ISOLATION: Does each seed get properly independent training? Is the seed set before model init AND training?
3. MEASUREMENT CONSISTENCY: Does the spectral radius measurement match D28 exactly? Same eval seed (9999)?
4. RESUME LOGIC: Does checkpoint resume correctly identify completed runs?
5. RESOURCE: Each run takes ~3 min. 28 runs = ~84 min total. Is this reasonable?
6. STATISTICAL ANALYSIS: Is the paired t-test at the end correct? Is the adjudication threshold reasonable?
7. Any issues with the shared model/data imports?

Be specific about line numbers and exact issues found.
