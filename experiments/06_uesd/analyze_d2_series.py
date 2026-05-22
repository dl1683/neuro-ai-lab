"""
Analysis: Combined D2 Series Results

Processes results from D2, D2b, D2c, D2d into a unified comparison
with proper statistical analysis including Wilson score confidence
intervals for success rates.
"""
import json
import math
import sys
from pathlib import Path


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for binomial proportion (95% CI by default)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return p_hat, lo, hi


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def analyze_d2b(data):
    """Analyze D2b properly-seeded sweep results."""
    if not data:
        print("  [D2b not found]")
        return

    print("=" * 70)
    print("D2b: PROPERLY-SEEDED SWEEP (CE-dynamics vs E5 vs Encoder-2L)")
    print("=" * 70)

    for model_type in ["ce_dynamics", "e5", "enc_2L"]:
        key = "sweep_%s" % model_type
        if key not in data:
            continue
        s = data[key]
        runs = s["runs"]
        n = len(runs)
        successes = sum(1 for r in runs if r["seq_acc"] > 0.9)
        p, lo, hi = wilson_ci(successes, n)

        print("\n  %s (%d seeds):" % (model_type, n))
        print("    Token acc: %.4f +/- %.4f" % (s["token_acc_mean"], s["token_acc_std"]))
        print("    Seq acc:   %.4f +/- %.4f" % (s["seq_acc_mean"], s["seq_acc_std"]))
        print("    Success:   %d/%d = %.0f%% [95%% CI: %.0f%%--%.0f%%]" % (
            successes, n, p * 100, lo * 100, hi * 100))

        for r in runs:
            status = "OK" if r["seq_acc"] > 0.9 else "FAIL"
            print("      seed=%d: tok=%.4f seq=%.4f [%s]" % (
                r["seed"], r["token_acc"], r["seq_acc"], status))


def analyze_d2d(data):
    """Analyze D2d depth-matched encoder sweep."""
    if not data:
        print("  [D2d not found]")
        return

    print("\n" + "=" * 70)
    print("D2d: DEPTH-MATCHED ENCODER SWEEP (4L/8L)")
    print("=" * 70)

    for key in ["enc_4L", "enc_8L"]:
        if key not in data:
            continue
        s = data[key]
        runs = s["runs"]
        n = len(runs)
        successes = sum(1 for r in runs if r["seq_acc"] > 0.9)
        p, lo, hi = wilson_ci(successes, n)

        print("\n  %s (%d params, %d seeds):" % (key, s["params"], n))
        print("    Token acc: %.4f +/- %.4f" % (s["token_acc_mean"], s["token_acc_std"]))
        print("    Seq acc:   %.4f +/- %.4f" % (s["seq_acc_mean"], s["seq_acc_std"]))
        print("    Success:   %d/%d = %.0f%% [95%% CI: %.0f%%--%.0f%%]" % (
            successes, n, p * 100, lo * 100, hi * 100))


def analyze_d2c(data):
    """Analyze D2c stability analysis results."""
    if not data:
        print("  [D2c not found]")
        return

    print("\n" + "=" * 70)
    print("D2c: STABILITY ANALYSIS (D7: sigma_max / rho)")
    print("=" * 70)

    header = "%15s %5s %6s %6s %6s %6s %6s %6s %6s" % (
        "track", "seed", "tok", "seq", "rho", "s_max", "kappa", "WA", "basin")
    print("\n  " + header)
    print("  " + "-" * len(header))

    for r in data.get("runs", []):
        d = r["diagnostics"]
        print("  %15s %5d %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f" % (
            r["track"], r["seed"],
            d["token_accuracy"]["token_acc"],
            d["token_accuracy"]["seq_acc"],
            d["spectral_radius"]["mean_rho"],
            d["sigma_max_ratio"]["sigma_max_mean"],
            d["sigma_max_ratio"]["kappa_mean"],
            d["wrong_attractor"]["wrong_attractor_rate"],
            d["basin_perturbation"]["stability_frac"],
        ))


def combined_comparison(d2b, d2d):
    """Cross-model parameter efficiency comparison."""
    if not d2b:
        return

    print("\n" + "=" * 70)
    print("PARAMETER EFFICIENCY COMPARISON")
    print("=" * 70)

    rows = []

    # UESD CE-dynamics from D2b
    if "sweep_ce_dynamics" in d2b:
        s = d2b["sweep_ce_dynamics"]
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        _, lo, hi = wilson_ci(successes, len(s["runs"]))
        rows.append(("UESD CE-dyn", 694016, s["seq_acc_mean"], s["seq_acc_std"],
                      successes, len(s["runs"]), lo, hi))

    # E5 from D2b
    if "sweep_e5" in d2b:
        s = d2b["sweep_e5"]
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        _, lo, hi = wilson_ci(successes, len(s["runs"]))
        rows.append(("UESD E5", 694016, s["seq_acc_mean"], s["seq_acc_std"],
                      successes, len(s["runs"]), lo, hi))

    # Encoder-2L from D2b
    if "sweep_enc_2L" in d2b:
        s = d2b["sweep_enc_2L"]
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        _, lo, hi = wilson_ci(successes, len(s["runs"]))
        rows.append(("Enc-2L", 425344, s["seq_acc_mean"], s["seq_acc_std"],
                      successes, len(s["runs"]), lo, hi))

    # Depth-matched from D2d
    if d2d:
        for key, name in [("enc_4L", "Enc-4L"), ("enc_8L", "Enc-8L")]:
            if key in d2d:
                s = d2d[key]
                successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
                _, lo, hi = wilson_ci(successes, len(s["runs"]))
                rows.append((name, s["params"], s["seq_acc_mean"], s["seq_acc_std"],
                              successes, len(s["runs"]), lo, hi))

    print("\n  %-15s %8s %10s %10s %12s" % (
        "Model", "Params", "Seq Acc", "Std", "Success [CI]"))
    print("  " + "-" * 60)
    for name, params, seq_mean, seq_std, succ, n, lo, hi in rows:
        print("  %-15s %8d %10.4f %10.4f %4d/%d [%.0f--%.0f%%]" % (
            name, params, seq_mean, seq_std, succ, n, lo * 100, hi * 100))


def main():
    results_dir = Path(__file__).parent / "results"

    d2b = load_json(results_dir / "exp_d2b_ce_dynamics_sweep.json")
    d2c = load_json(results_dir / "exp_d2c_stability_analysis.json")
    d2d = load_json(results_dir / "exp_d2d_depth_sweep.json")

    analyze_d2b(d2b)
    analyze_d2d(d2d)
    analyze_d2c(d2c)
    combined_comparison(d2b, d2d)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if d2b and "sweep_ce_dynamics" in d2b:
        s = d2b["sweep_ce_dynamics"]
        n = len(s["runs"])
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        _, lo, hi = wilson_ci(successes, n)
        if successes == n:
            print("  CE-dynamics: %d/%d success [CI: %.0f--100%%]" % (successes, n, lo * 100))
            print("  -> ROBUST: all seeds succeed with proper seeding")
        elif successes >= n * 0.8:
            print("  CE-dynamics: %d/%d success [CI: %.0f--%.0f%%]" % (
                successes, n, lo * 100, hi * 100))
            print("  -> MOSTLY ROBUST but not 100%% reliable")
        else:
            print("  CE-dynamics: %d/%d success" % (successes, n))
            print("  -> NOT ROBUST")

    if d2b and "sweep_e5" in d2b:
        s = d2b["sweep_e5"]
        n = len(s["runs"])
        successes = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
        if successes < n:
            print("  E5 (SC+CE): %d/%d success -> BIMODAL (wrong-attractor trap)" % (successes, n))

    missing = []
    if not d2b:
        missing.append("D2b")
    if not d2d:
        missing.append("D2d")
    if not d2c:
        missing.append("D2c")
    if missing:
        print("\n  Missing results: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
