The core theorem is negative:

`k < 1` gives convergence only inside the basin of the fixed point already selected by the dynamics. It says nothing about whether that fixed point has positive readout margin. So UESD needs a **convergence-correctness coupling**, not more contraction alone.

A useful sufficient condition is:

Let `G_c` be locally `k`-contractive, `r(s)=||G_c(s)-s||`, and decoder margin

`m_y(s)=logit_y(s)-max_{z != y} logit_z(s)`.

If readout margin is `K`-Lipschitz and `G_c` is contractive in the current basin, then any approximate fixed point `s` is within

`||s-s*|| <= r(s)/(1-k)`

of the true basin fixed point. Therefore the converged fixed point is guaranteed correct if

`m_y(s) > K r(s)/(1-k)`.

This gives the first concrete correction to E5:

`L_margin_fp = softplus(K r(s_T)/(1-k_hat) + gamma - m_y(s_T))`

Train not merely for `r -> 0`, but for **residual small relative to margin buffer**. Wrong attractors are exactly states with `r≈0` and `m_y<0`.

The stronger mechanism is to replace E5 as “the energy” with a semantic Lyapunov function:

`E_c(s) = E_task(s,c,y) + alpha E_sc(s,c) + beta E_barrier(s,c)`

and use

`s_{t+1}=s_t - eta M(s,c) grad_s E_c(s) + A(s,c) grad_s E_c(s)`

where `M` is positive definite dissipative flow and `A=-A^T` is optional solenoidal exploration. If `E_c` satisfies a local PL inequality around the correct attractor,

`1/2 ||grad E_c||^2 >= mu (E_c - E*)`

and is `L`-smooth, gradient dynamics with `eta <= 1/L` gives

`E(s_t)-E* <= (1-eta mu)^t (E(s_0)-E*)`.

Correctness follows if the only low-energy basin in the context-conditioned reachable region has positive decoder margin. This is the missing structural assumption.

Concrete losses:

```text
L = CE(R(s_T), y)
  + lambda_sc ||G(s_T,c)-s_T||^2
  + lambda_fp softplus(K r_T/(1-k_hat)+gamma-m_y(s_T))
  + lambda_rec E_{xi}[CE(R(G^T(s*_y+xi,c)), y)]
  + lambda_bad E_{s_bad}[softplus(gamma-m_y(s_bad)) * exp(-r(s_bad)^2/sigma^2)]
  + lambda_bif softplus(epsilon - sigma_min(I - dG/ds))
```

Interpretation:

`L_rec` expands the correct basin by training recovery from perturbations around correct states.

`L_bad` raises or destabilizes wrong fixed points: if a wrong state has low residual, make it expensive.

`L_bif` prevents continuation training from crossing a saddle-node/pitchfork bifurcation where the correct fixed-point branch disappears or splits.

Use CE-trained dynamics as a warm start, then continue with increasing `lambda_sc`. This is homotopy continuation: start from a model whose endpoint is correct but not fixed, then gradually force fixed-point behavior while constraining the fixed-point branch to keep positive margin. Monitor `m_y(s*)`, `r`, `k`, and `sigma_min(I-J)`. If `sigma_min(I-J)->0`, a bifurcation is near; stop or reduce SC pressure.

For basin engineering, estimate basin volume directly:

`B_y = P_{xi ~ N(0,sigma^2 I)}[R(G^T(s*_y+xi,c))=y]`.

Optimize

`L_basin = -log B_y + eta log sum_{z != y} B_z`

using soft margins instead of hard indicators. This directly makes correct basins larger than wrong basins. It is more aligned with the failure than raw residual loss.

The k/rho dissociation is useful if treated as anisotropic dynamics:

`S = U_readout ⊕ V_search`.

Enforce contraction in `U_readout`:

`||P_U J P_U|| <= k_U < 1`

but allow controlled expansion or neutral flow in `V_search` while uncertainty is high:

`rho(P_V J P_V) >= 1` early, then anneal below 1 near convergence.

So the dynamics become:

```text
explore:   contract readout-relevant error, allow rho expansion in search/null directions
select:    couple V_search into U_readout through semantic energy decrease
settle:    anneal noise/solenoidal flow, contract both U and V
```

This turns `rho>1` from a bug into a search reservoir, provided expansion is gated by energy decrease and shut off before fixation.

My recommended UESD next design:

1. Keep variable-T training for `k` suppression.
2. Add fixed-point margin loss `softplus(Kr/(1-k)+gamma-m)`.
3. Add perturbation recovery around correct endpoints.
4. Add wrong-attractor mining: roll out from noisy states, collect `r≈0,m<0`, train escape/repulsion.
5. Warm-start from CE, then anneal in SC with bifurcation monitors.
6. Treat `rho` as controlled search-space expansion, not as the convergence metric.

Bottom line: correct convergence is not a Banach theorem. It becomes defensible only when UESD trains **semantic attractor basins**: low residual, positive margin, large recovery basin, and destabilized wrong fixed points.

