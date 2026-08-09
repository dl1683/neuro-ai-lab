I found 1 critical issue that should block launch, plus several moderate risks.

1. **CRITICAL: `torch.no_grad` breaks spectral-radius computation**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L270), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L292)
- Problem: `measure_spectral_radius` is decorated with `@torch.no_grad()` but calls `torch.autograd.grad` at line 295.
- Impact: This context typically prevents building a grad graph for `s_next`, so VJP/JVP can be invalid or fail. Since this is called unconditionally in `run_one_config`, the experiment can fail on the spectral-radius step.
- Fix: remove `@torch.no_grad()` or wrap only non-differentiated sections in `torch.no_grad()` and keep the Jacobian loop inside `with torch.enable_grad():`.

2. **MODERATE: Contraction-ratio definition is not the Banach norm as documented**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L163), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L174)
- Problem: `dist_t = (s - s_star).norm(dim=-1)` computes per-position norms, then `mean()` is taken before ratioing. This is different from `||s_t - s*||` of the full state tensor (or a fixed vector norm choice).
- Impact: Reported `k_t` can differ from the intended contraction factor, especially when norms vary across positions.

3. **MODERATE: Fixed-point approximation has no convergence gate**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L142), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L147)
- Problem: `FP_T=100` is hard-coded and the measured residual is only logged.
- Impact: If `fp_residual` is not small, downstream contraction and `predicted_T99` become unreliable.
- Fix: make `FP_T` adaptive (iterate until residual threshold or cap max T), and skip/badge metrics as invalid when residual is high.

4. **MODERATE: Edge-case handling for unstable maps is weak**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L229), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L234), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L257)
- Problem: When mean `k >= 1`, `predicted_T99` is forced to `inf`, but the run still proceeds with readout comparisons and summary merging.
- Impact: You lose explicit “non-contractive / divergent” status, and downstream interpretation may treat it as incomplete success with noisy metrics.

5. **MODERATE: Numerical stability is only partially guarded**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L190), [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L206)
- What’s good: per-sample ratio uses clamp(1e-8), preventing exact divide-by-zero.
- Risk: mean-ratio path still uses scalar denom check and can be sensitive when distances collapse; no warning/logging if unstable collapse occurs.

6. **LOW: Seed ordering is correct, but reproducibility is partial**
- Location: [exp_d28_contraction_ratio.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\exp_d28_contraction_ratio.py#L317), [training.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\shared/training.py#L110)
- Good: model is created after `full_seed(SEED)` in `run_one_config`, so D27-style seed ordering issue is not present.
- Note: seed helper sets torch/cuda and Python `random` only; if any future path uses NumPy randomness, include `np.random.seed(seed)` for full reproducibility.

7. **LOW: Data generator has an odd-length edge bug**
- Location: [data.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\shared/data.py#L34), [data.py](/C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\shared/data.py#L39)
- For odd `seq_len`, `a[:, :seq_len // 2 + seq_len % 2]` can over-index `a` and break assignment (the current experiment uses even lengths, but this is latent).

MEMORY (4096×L=24×d=128×T=100)

- Feature-map size per tensor: `B*L*d*4B = 4096*24*128*4 ≈ 50 MB` for fp32.
- Encoder/decoder state + context implies multiple such tensors (~150–200 MB) plus attention score buffers per layer (`B*heads*L*L*4` ≈ 38 MB per decoder/encoder layer at 4 heads).
- With 2 encoder layers + 1 dynamics layer in fp32 and temporary buffers, peak should be on the order of a few hundred MB to about ~1 GB; 100-step fixed-point loop is memory-reused, not multiplicative in T.
- So `4096, L=24, d=128, T=100` should fit comfortably on typical modern GPUs (even 8 GB), barring other large concurrent jobs.

SPECTRAL RADIUS: correctness
- Given the current goal and implementation, it is an eigenvalue-style power iteration using VJP form, not a full two-sided singular-value power iteration.
- If you intend Lipschitz/spectral norm of Jacobian, switch to a true JVP/VJP alternating setup or `torch.autograd.functional.jvp`-based iteration; if intent is dominant eigenvalue magnitude, keep VJP but run with explicit `torch.enable_grad()` and consider convergence diagnostics.