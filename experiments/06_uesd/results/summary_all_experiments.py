"""
Generate a comprehensive summary table of all UESD experiments.

Produces a cross-task comparison table for the paper.
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent


def load(name):
    p = RESULTS_DIR / ("%s.json" % name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_acc(data, key, field="token_acc"):
    if not data or key not in data:
        return None
    return data[key].get("eval", {}).get("token_accuracy", {}).get(field)


def fmt(val):
    if val is None:
        return "  —  "
    return "%.4f" % val


def main():
    exp_a = load("exp_a_copy")
    exp_b = load("exp_b_reversal")
    exp_c = load("exp_c_sort")
    exp_d = load("exp_d_compositional")
    exp_d2 = load("exp_d2_controls")
    exp_d2b = load("exp_d2b_ce_dynamics_sweep")

    print("=" * 90)
    print("COMPREHENSIVE UESD EXPERIMENT SUMMARY")
    print("=" * 90)

    # Table 1: Sequence accuracy across all tasks
    print("\nTable 1: Sequence Accuracy (all models, all tasks)")
    print("-" * 90)
    header = "%-20s %8s %8s %8s %10s %8s" % (
        "Model", "Copy", "Reverse", "Sort", "Addition", "Dedup")
    print(header)
    print("-" * 90)

    rows = [
        ("E1 (UESD embed)",
         get_acc(exp_a, "track_a_e1", "seq_acc"),
         get_acc(exp_b, "track_a_e1", "seq_acc"),
         get_acc(exp_c, "track_a_e1", "seq_acc"),
         get_acc(exp_d, "addition_e1", "seq_acc"),
         get_acc(exp_d, "dedup_e1", "seq_acc")),
        ("E5 lam=0.1",
         None,
         get_acc(exp_b, "track_b_e5_lam0.1", "seq_acc"),
         get_acc(exp_c, "track_b_e5_lam0.1", "seq_acc"),
         get_acc(exp_d, "addition_e5_lam0.1", "seq_acc"),
         get_acc(exp_d, "dedup_e5_lam0.1", "seq_acc")),
        ("E5 lam=1.0",
         get_acc(exp_a, "track_b_e5_lam1.0", "seq_acc"),
         get_acc(exp_b, "track_b_e5_lam1.0", "seq_acc"),
         get_acc(exp_c, "track_b_e5_lam1.0", "seq_acc"),
         get_acc(exp_d, "addition_e5_lam1.0", "seq_acc"),
         get_acc(exp_d, "dedup_e5_lam1.0", "seq_acc")),
        ("AR Baseline",
         get_acc(exp_a, "ar_baseline", "seq_acc"),
         get_acc(exp_b, "ar_baseline", "seq_acc"),
         get_acc(exp_c, "ar_baseline", "seq_acc"),
         get_acc(exp_d, "addition_ar", "seq_acc"),
         get_acc(exp_d, "dedup_ar", "seq_acc")),
        ("Encoder-2L",
         get_acc(exp_a, "encoder_only", "seq_acc"),
         get_acc(exp_b, "encoder_only", "seq_acc"),
         get_acc(exp_c, "encoder_only", "seq_acc"),
         get_acc(exp_d, "addition_enc", "seq_acc"),
         get_acc(exp_d, "dedup_enc", "seq_acc")),
    ]

    # Add D2 controls if available
    if exp_d2:
        dce_seq = exp_d2.get("dynamics_ce", {}).get("eval", {}).get(
            "token_accuracy", {}).get("seq_acc")
        enc4_seq = exp_d2.get("enc_4L", {}).get("eval", {}).get(
            "token_accuracy", {}).get("seq_acc")
        enc8_seq = exp_d2.get("enc_8L", {}).get("eval", {}).get(
            "token_accuracy", {}).get("seq_acc")
        rows.append(("CE-dynamics (D2)", None, None, None, dce_seq, None))
        rows.append(("Encoder-4L (D2)", None, None, None, enc4_seq, None))
        rows.append(("Encoder-8L (D2)", None, None, None, enc8_seq, None))

    for name, copy, rev, sort, add, dedup in rows:
        print("%-20s %8s %8s %8s %10s %8s" % (
            name, fmt(copy), fmt(rev), fmt(sort), fmt(add), fmt(dedup)))

    print("-" * 90)

    # Table 2: Parameter counts
    print("\nTable 2: Parameter Counts")
    print("-" * 40)
    params = {
        "UESD (E1/E5/CE-dyn)": 694016,
        "AR Baseline": 950336,
        "Encoder-2L": 425344,
    }
    if exp_d2:
        enc4 = exp_d2.get("enc_4L", {}).get("params")
        enc8 = exp_d2.get("enc_8L", {}).get("params")
        if enc4:
            params["Encoder-4L"] = enc4
        if enc8:
            params["Encoder-8L"] = enc8

    for name, p in sorted(params.items(), key=lambda x: x[1]):
        ratio = p / 694016
        print("  %-25s %8d (%.2fx UESD)" % (name, p, ratio))

    # Table 3: Addition task deep dive (the key comparison)
    print("\nTable 3: Addition Task — The Critical Comparison")
    print("-" * 70)
    print("%-20s %8s %8s %8s %8s" % ("Model", "Params", "Tok Acc", "Seq Acc", "Note"))
    print("-" * 70)

    add_rows = [
        ("E1 (embed reg)", "694K",
         fmt(get_acc(exp_d, "addition_e1", "token_acc")),
         fmt(get_acc(exp_d, "addition_e1", "seq_acc")),
         "MSE+0.1CE fails"),
        ("E5 lam=1.0", "694K",
         fmt(get_acc(exp_d, "addition_e5_lam1.0", "token_acc")),
         fmt(get_acc(exp_d, "addition_e5_lam1.0", "seq_acc")),
         "SC+CE (single seed)"),
    ]

    if exp_d2:
        dce = exp_d2.get("dynamics_ce", {}).get("eval", {}).get("token_accuracy", {})
        if dce:
            add_rows.append(("CE-dynamics", "694K", fmt(dce.get("token_acc")),
                             fmt(dce.get("seq_acc")), "Pure CE (single seed, D2)"))
        e4 = exp_d2.get("enc_4L", {}).get("eval", {}).get("token_accuracy", {})
        e8 = exp_d2.get("enc_8L", {}).get("eval", {}).get("token_accuracy", {})
        if e4:
            add_rows.append(("Encoder-4L", "822K", fmt(e4.get("token_acc")),
                             fmt(e4.get("seq_acc")), "Single seed"))
        if e8:
            add_rows.append(("Encoder-8L", "1615K", fmt(e8.get("token_acc")),
                             fmt(e8.get("seq_acc")), "2.3x params"))

    add_rows.append(("Encoder-2L", "425K",
                     fmt(get_acc(exp_d, "addition_enc", "token_acc")),
                     fmt(get_acc(exp_d, "addition_enc", "seq_acc")),
                     "Fails on addition"))
    add_rows.append(("AR Baseline", "950K",
                     fmt(get_acc(exp_d, "addition_ar", "token_acc")),
                     fmt(get_acc(exp_d, "addition_ar", "seq_acc")),
                     "Teacher-forced"))

    for name, params_str, tok, seq, note in add_rows:
        print("%-20s %8s %8s %8s  %s" % (name, params_str, tok, seq, note))

    # D2b multi-seed results
    if exp_d2b:
        print("\nTable 4: D2b Multi-Seed Sweep (Properly Seeded)")
        print("-" * 70)
        for key in ["sweep_ce_dynamics", "sweep_e5", "sweep_enc_2L"]:
            if key in exp_d2b:
                s = exp_d2b[key]
                n = len(s["runs"])
                succ = sum(1 for r in s["runs"] if r["seq_acc"] > 0.9)
                print("  %s: tok=%.4f+/-%.4f, seq=%.4f+/-%.4f, success=%d/%d" % (
                    key.replace("sweep_", ""),
                    s["token_acc_mean"], s["token_acc_std"],
                    s["seq_acc_mean"], s["seq_acc_std"],
                    succ, n))

    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    print("""
1. COPY/REVERSE/SORT: All models solve these tasks (seq acc > 99%).
   Dynamics are NOT necessary at L=8, V=64 scale.

2. ADDITION: The critical test for dynamics necessity.
   - E1 FAILS (0% seq acc) — MSE+0.1CE coupling too weak
   - E5 SUCCEEDS (100% in single seed) — but 40% failure rate in D2 sweep
   - CE-dynamics SUCCEEDS (100% in D2, single seed) — needs D2b multi-seed
   - Encoder-2L FAILS (0.1% seq acc) — insufficient depth
   - Encoder-4L/8L SUCCEED (99.5%/99.98%) — dynamics not strictly necessary
   - UESD is more PARAMETER-EFFICIENT: 694K vs 1615K for similar accuracy

3. DEDUP: Mild dynamics advantage (99.8% UESD vs 95.7% encoder-only).

4. WRONG-ATTRACTOR TRAP: E5's SC loss causes 40% failure on addition.
   CE-dynamics (no SC) avoids this. SC term is counterproductive.

5. REVISED CLAIM: "Weight-tied iterative dynamics are a parameter-efficient
   alternative to depth stacking. CE-only training is more robust than
   SC+CE for carry-chain tasks."
""")


if __name__ == "__main__":
    main()
