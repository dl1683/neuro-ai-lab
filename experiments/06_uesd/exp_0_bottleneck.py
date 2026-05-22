"""
Experiment 0: Information Bottleneck Analysis of Softmax Collapse

Mathematical derivation + numerical verification.

Claim: The softmax + sampling step at each AR generation step limits
information flow to at most log2(V) bits per step, while continuous
dynamics in R^d can carry O(d * bits_per_dim) bits.

This is a MATHEMATICAL experiment -- no model training, just derivations
and numerical computations to verify bounds.
"""

import numpy as np
import json
from pathlib import Path


def mutual_information_softmax_bound(V: int) -> float:
    """
    Upper bound on mutual information through softmax + sampling.

    For any random variable h_t and x_t = sample(softmax(W*h_t)):
    I(h_t; x_t) <= H(x_t) <= log2(V)

    This follows from the data processing inequality:
    h_t -> softmax(W*h_t) -> sample -> x_t

    Each step can only decrease MI. The final x_t in {1,...,V} has
    entropy at most log2(V).
    """
    return np.log2(V)


def state_capacity_continuous(d: int, bits_per_dim: float = 32.0) -> float:
    """
    Information capacity of a continuous state vector in R^d.

    At float32 precision: d x 32 bits.
    In practice, effective capacity is lower due to:
    - Correlations between dimensions
    - Precision not fully utilized (most values near 0)
    - Embedding geometry constraints

    But even a conservative estimate of ~1 effective bit per dimension
    gives d bits, which is >> log2(V) for any reasonable V and d.
    """
    return d * bits_per_dim


def compression_ratio(d: int, V: int, bits_per_dim: float = 32.0) -> float:
    """How much information is lost in the softmax collapse."""
    continuous = state_capacity_continuous(d, bits_per_dim)
    discrete = mutual_information_softmax_bound(V)
    return continuous / discrete


def effective_bandwidth_ar(V: int, L: int) -> float:
    """
    Total information bandwidth of AR generation for L tokens.

    Each token carries at most log2(V) bits. L tokens carry at most
    L x log2(V) bits. But through the sequential bottleneck, each
    step conditions on previous tokens, so the total is bounded by:

    I(h_0; x_1, ..., x_L) <= Sum_t I(h_0; x_t | x_{<t})
                           <= Sum_t H(x_t | x_{<t})
                           <= L x log2(V)

    In practice, much less -- conditional entropy decreases with context.
    """
    return L * np.log2(V)


def effective_bandwidth_uesd(d: int, L: int, conservative_bits_per_dim: float = 1.0) -> float:
    """
    Information bandwidth of UESD generation.

    The state s_T in R^{Lxd} carries Lxd continuous values.
    At even 1 effective bit per dimension: L x d bits.

    No sequential bottleneck -- all positions refined simultaneously.
    """
    return L * d * conservative_bits_per_dim


def bridge_capacity_analysis(V: int, d: int, L: int):
    """
    Formalize the softmax bottleneck as a circuit viability failure.

    In circuit viability terms:
    - The "bridge" is the softmax + sampling step
    - Bridge capacity = log2(V) bits per step
    - Required capacity = d bits per step (to preserve state)
    - Capacity deficit = d - log2(V) bits

    Analogy to SynFlow bridge collapse:
    - SynFlow starves fc1 -> zero capacity through the classifier bridge
    - Softmax starves the state -> log2(V) capacity through the token bridge
    - Both are structural bottlenecks, not weight-quality problems
    """
    bridge_cap = np.log2(V)
    required_cap = d  # conservative: 1 bit per dim
    deficit = required_cap - bridge_cap
    deficit_ratio = deficit / required_cap

    return {
        "bridge_capacity_bits": bridge_cap,
        "required_capacity_bits": required_cap,
        "capacity_deficit_bits": deficit,
        "deficit_ratio": deficit_ratio,
        "analogy": "SynFlow allocates 0 weights to fc1 (100% deficit). "
                   f"Softmax allocates {bridge_cap:.1f} bits to the token bridge "
                   f"({deficit_ratio*100:.1f}% deficit)."
    }


def data_processing_inequality_chain():
    """
    Formal derivation of the information bottleneck.

    At each AR step, the chain is:
    h_t -> z_t = W*h_t -> p_t = softmax(z_t) -> x_t ~ Cat(p_t) -> e_t = Embed(x_t)

    By the data processing inequality (DPI):
    I(h_t; e_t) <= I(h_t; x_t) <= I(h_t; p_t) <= I(h_t; z_t) <= I(h_t; h_t) = H(h_t)

    The tightest bound is at the sampling step:
    I(h_t; x_t) <= H(x_t) <= log2(V)

    This means the re-embedded state e_t = Embed(x_t) carries at most
    log2(V) bits of information about h_t, regardless of d, model size,
    or training procedure.

    For UESD: s_{t+1} = s_t + F_theta(s_t, c)
    I(s_t; s_{t+1}) is NOT constrained by any discrete bottleneck.
    The mutual information is bounded only by the information in s_t itself.
    """
    return {
        "chain": "h_t -> z_t -> p_t -> x_t -> e_t",
        "dpi_bounds": [
            "I(h_t; e_t) <= I(h_t; x_t)  [DPI: sampling then embedding]",
            "I(h_t; x_t) <= H(x_t)        [MI bounded by marginal entropy]",
            "H(x_t) <= log2(V)            [V-ary categorical]",
        ],
        "conclusion": "I(h_t; e_t) <= log2(V) bits -- hard ceiling, model-independent",
        "uesd_alternative": "I(s_t; s_{t+1}) -- no discrete bottleneck, bounded only by H(s_t)"
    }


def run_analysis():
    """Run the full bottleneck analysis for standard LLM configurations."""

    results = {
        "experiment": "0_information_bottleneck",
        "purpose": "Quantify the information bottleneck created by softmax collapse",
        "derivation": data_processing_inequality_chain(),
        "configurations": [],
    }

    configs = [
        {"name": "UESD_POC",     "d": 128,  "V": 64,    "L": 8},
        {"name": "Small_LM",     "d": 768,  "V": 32000, "L": 512},
        {"name": "GPT2",         "d": 1024, "V": 50257, "L": 1024},
        {"name": "LLaMA_7B",     "d": 4096, "V": 32000, "L": 4096},
        {"name": "GPT4_class",   "d": 8192, "V": 128000,"L": 8192},
    ]

    for cfg in configs:
        d, V, L = cfg["d"], cfg["V"], cfg["L"]

        ar_bandwidth = effective_bandwidth_ar(V, L)
        uesd_bandwidth = effective_bandwidth_uesd(d, L)
        bridge = bridge_capacity_analysis(V, d, L)

        analysis = {
            **cfg,
            "softmax_bound_bits_per_step": mutual_information_softmax_bound(V),
            "state_capacity_bits_float32": state_capacity_continuous(d),
            "state_capacity_bits_conservative": state_capacity_continuous(d, 1.0),
            "compression_ratio_float32": compression_ratio(d, V),
            "compression_ratio_conservative": compression_ratio(d, V, 1.0),
            "ar_total_bandwidth_bits": ar_bandwidth,
            "uesd_total_bandwidth_bits": uesd_bandwidth,
            "bandwidth_ratio": uesd_bandwidth / ar_bandwidth,
            "bridge_analysis": bridge,
        }
        results["configurations"].append(analysis)

    # Summary
    print("=" * 70)
    print("EXPERIMENT 0: INFORMATION BOTTLENECK ANALYSIS")
    print("=" * 70)
    print()
    print("DERIVATION (Data Processing Inequality):")
    print("  h_t -> z_t=W*h_t -> p_t=softmax(z_t) -> x_t~Cat(p_t) -> e_t=Embed(x_t)")
    print("  By DPI: I(h_t; e_t) <= I(h_t; x_t) <= H(x_t) <= log2(V)")
    print("  This bound is HARD -- independent of model size, training, architecture.")
    print()

    print(f"{'Config':<15} {'d':>6} {'V':>8} {'Softmax bits':>14} {'State bits':>12} {'Ratio':>8} {'Deficit':>8}")
    print("-" * 70)
    for cfg in results["configurations"]:
        print(f"{cfg['name']:<15} {cfg['d']:>6} {cfg['V']:>8} "
              f"{cfg['softmax_bound_bits_per_step']:>14.1f} "
              f"{cfg['state_capacity_bits_conservative']:>12.0f} "
              f"{cfg['compression_ratio_conservative']:>8.1f}x "
              f"{cfg['bridge_analysis']['deficit_ratio']*100:>7.1f}%")

    print()
    print("CIRCUIT VIABILITY ANALOGY:")
    print("  SynFlow: allocates 0 weights to fc1 -> 100% bridge deficit -> collapse")
    print("  Softmax: allocates log2(V) bits to token bridge -> >90% deficit -> bottleneck")
    print("  Both are STRUCTURAL failures, not weight-quality failures.")
    print()
    print("UESD RESOLUTION:")
    print("  s_{t+1} = s_t + F_theta(s_t, c) -- no discrete bottleneck")
    print("  I(s_t; s_{t+1}) bounded only by H(s_t) -- full state preserved")

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_0_bottleneck.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_analysis()
