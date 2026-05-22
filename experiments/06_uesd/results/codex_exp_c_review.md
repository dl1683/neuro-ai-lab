Saved: [experiments/06_uesd/results/exp_c_sort_integrity_review.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_c_sort_integrity_review.md)

Audit verdict:
- Exp C does not establish dynamics necessity on sort at `L=8, V=64`.
- Encoder confound label is appropriate (encoder is still near parity), but “dynamics validated” is not.
- Largest risk is methodological: single-run, near-ceiling metrics with no seed/CI reporting, plus a notable doc-vs-JSON metric mismatch.
- `max_rho > 1` (E1, `max_rho=1.19865`) is a substantive stability red flag for the current UESD theory assumptions, despite high token accuracy.

If you want, I can now draft a stricter rewrite for Exp C’s interpretation section in `EXPERIMENTS.md` to make the limitations explicit.