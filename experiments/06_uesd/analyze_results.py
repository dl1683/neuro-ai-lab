"""Analyze and summarize UESD experiment results."""
import json
from pathlib import Path


def load_results(name):
    path = Path(__file__).parent / "results" / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def summarize_exp_a():
    r = load_results("exp_a_copy")
    if not r:
        print("Experiment A results not found.")
        return

    print_header("EXPERIMENT A: COPY SMOKE TEST")

    print(f"\nDevice: {r.get('device', 'unknown')}")
    print(f"Timestamp: {r.get('timestamp', 'unknown')}")

    # Param counts
    print("\n--- Parameter Counts ---")
    for key in ["track_a_e1", "track_b_e5_lam1.0", "ar_baseline", "encoder_only"]:
        if key in r:
            print(f"  {key}: {r[key].get('params', '?'):,}")

    # Accuracy comparison
    print("\n--- Token Accuracy ---")
    for key, label in [("track_a_e1", "E1 (embed reg)"),
                       ("track_b_e5_lam1.0", "E5 (SC + CE)"),
                       ("ar_baseline", "AR baseline"),
                       ("encoder_only", "Encoder-only")]:
        if key in r:
            ev = r[key].get("eval", {})
            ta = ev.get("token_accuracy", {})
            tok = ta.get("token_acc", "?")
            seq = ta.get("seq_acc", "?")
            if isinstance(tok, float):
                print(f"  {label:20s}: token={tok:.4f}, seq={seq:.4f}")
            else:
                print(f"  {label:20s}: token={tok}, seq={seq}")

    # UESD-specific diagnostics
    print("\n--- UESD Diagnostics ---")
    for key, label in [("track_a_e1", "E1"), ("track_b_e5_lam1.0", "E5")]:
        if key in r:
            ev = r[key].get("eval", {})
            nr = ev.get("normalized_residual", {})
            dm = ev.get("decoder_margin", {})
            wa = ev.get("wrong_attractor", {})
            bp = ev.get("basin_perturbation", {})
            sr = ev.get("spectral_radius", {})
            print(f"\n  {label}:")
            if nr:
                print(f"    Residual (norm): mean={nr.get('mean', '?'):.6f}, std={nr.get('std', '?'):.6f}")
            if dm:
                print(f"    Decoder margin:  mean={dm.get('mean_margin', '?'):.4f}, frac_pos={dm.get('frac_positive', '?'):.4f}")
            if wa:
                print(f"    Wrong attractor: rate={wa.get('wrong_attractor_rate', '?'):.4f}, converged={wa.get('converged_frac', '?'):.4f}")
            if bp:
                print(f"    Basin stability: {bp.get('stability_frac', '?'):.4f}")
            if sr:
                print(f"    Spectral radius: mean={sr.get('mean_rho', '?'):.4f}, max={sr.get('max_rho', '?'):.4f}")

    # Gates
    if "gates" in r:
        print("\n--- Gates ---")
        for k, v in r["gates"].items():
            print(f"  {k}: {v}")

    # Training time
    print("\n--- Training Time ---")
    for key, label in [("track_a_e1", "E1"), ("track_b_e5_lam1.0", "E5"),
                       ("ar_baseline", "AR"), ("encoder_only", "Enc-only")]:
        if key in r:
            elapsed = r[key].get("elapsed_s", "?")
            if isinstance(elapsed, (int, float)):
                print(f"  {label:10s}: {elapsed:.1f}s")


def summarize_exp_b():
    r = load_results("exp_b_reversal")
    if not r:
        print("\nExperiment B results not found.")
        return

    print_header("EXPERIMENT B: REVERSAL MAIN TEST")

    print(f"\nDevice: {r.get('device', 'unknown')}")

    # Accuracy comparison table
    print("\n--- Accuracy Comparison ---")
    print(f"  {'Model':25s} {'Token Acc':>10s} {'Seq Acc':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    for key, label in [("track_a_e1", "E1 (embed reg)")]:
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    for lam in [0.0, 0.1, 1.0, 10.0]:
        key = f"track_b_e5_lam{lam}"
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {'E5 (lam=' + str(lam) + ')':25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    for key, label in [("ar_baseline", "AR baseline"), ("encoder_only", "Encoder-only")]:
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    # E5 diagnostics table
    print("\n--- E5 Lambda Sweep Diagnostics ---")
    print(f"  {'Lambda':>6s} {'Acc':>7s} {'WA Rate':>8s} {'Margin':>8s} {'Rho':>7s} {'Basin':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

    for lam in [0.0, 0.1, 1.0, 10.0]:
        key = f"track_b_e5_lam{lam}"
        if key in r:
            ev = r[key]["eval"]
            acc = ev["token_accuracy"]["token_acc"]
            wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 0)
            margin = ev.get("decoder_margin", {}).get("mean_margin", 0)
            rho = ev.get("spectral_radius", {}).get("mean_rho", 0)
            basin = ev.get("basin_perturbation", {}).get("stability_frac", 0)
            print(f"  {lam:>6.1f} {acc:>7.4f} {wa:>8.4f} {margin:>8.4f} {rho:>7.4f} {basin:>7.4f}")

    # Gates
    if "gates" in r:
        print("\n--- Gates ---")
        for k, v in r["gates"].items():
            print(f"  {k}: {v}")

    # Key comparisons
    print("\n--- Key Comparisons ---")
    e1_acc = r.get("track_a_e1", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    ar_acc = r.get("ar_baseline", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    enc_acc = r.get("encoder_only", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    print(f"  E1 vs AR:           {e1_acc:.4f} vs {ar_acc:.4f} (gap: {abs(e1_acc - ar_acc):.4f})")
    print(f"  E1 vs Encoder-Only: {e1_acc:.4f} vs {enc_acc:.4f} (dynamics value: {e1_acc - enc_acc:+.4f})")


def decision_table(exp_a, exp_b):
    """Apply the decision table from design_revision_r3.md."""
    print_header("DECISION TABLE")

    if not exp_a:
        print("  Cannot evaluate — Experiment A results missing.")
        return

    e1_copy = exp_a.get("track_a_e1", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)

    if e1_copy < 0.90:
        print("  OUTCOME: Exp A FAILS (copy < 90%)")
        print("  CONCLUSION: Dynamics don't converge")
        print("  NEXT STEP: Study Jacobian, try different architecture")
        return

    if not exp_b:
        if e1_copy >= 0.99:
            print(f"  Exp A PASSES (copy acc = {e1_copy:.4f})")
            print("  Exp B pending — run exp_b_reversal.py next")
        else:
            print(f"  Exp A INVESTIGATE (copy acc = {e1_copy:.4f})")
        return

    e1_rev = exp_b.get("track_a_e1", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)

    if e1_rev < 0.70:
        print("  OUTCOME: Exp B FAILS (reversal < 70%)")
        print("  CONCLUSION: Dynamics converge but can't transform")
        print("  NEXT STEP: Study representation capacity")
        return

    # Find best E5 by (accuracy, converged_frac, -rho) — not just WA rate
    best_wa = 1.0
    best_lam_info = None
    for lam in [0.0, 0.1, 1.0, 10.0]:
        key = f"track_b_e5_lam{lam}"
        if key in exp_b:
            ev = exp_b[key]["eval"]
            wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 1.0)
            conv = ev.get("wrong_attractor", {}).get("converged_frac", 0.0)
            rho = ev.get("spectral_radius", {}).get("mean_rho", 1.0)
            if wa < best_wa or (wa == best_wa and conv > (best_lam_info or {}).get("conv", -1)):
                best_wa = wa
                best_lam_info = {"lam": lam, "wa": wa, "conv": conv, "rho": rho}

    if best_lam_info and best_lam_info["conv"] == 0.0:
        print(f"  NOTE: Best WA rate is vacuous (converged_frac=0.0 for lambda={best_lam_info['lam']})")
        print(f"  Using next-best lambda with actual convergence for viability check.")
        for lam in [0.1, 1.0, 10.0, 0.0]:
            key = f"track_b_e5_lam{lam}"
            if key in exp_b:
                ev = exp_b[key]["eval"]
                conv = ev.get("wrong_attractor", {}).get("converged_frac", 0.0)
                if conv > 0.5:
                    wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 1.0)
                    best_wa = wa
                    best_lam_info = {"lam": lam, "wa": wa, "conv": conv}
                    print(f"  Selected lambda={lam} (converged_frac={conv:.4f}, WA={wa:.4f})")
                    break

    if best_wa > 0.20:
        print("  OUTCOME: Exp B passes, E5 wrong-attractor > 20%")
        print("  CONCLUSION: E5 is dead, E1 is the path")
        print("  NEXT STEP: Drop E5, investigate learned energy (Track C)")
    elif best_wa < 0.05:
        print("  OUTCOME: Exp B passes, E5 wrong-attractor < 5%")
        print("  CONCLUSION: E5 viable as proposed")
        print("  NEXT STEP: Proceed to harder tasks, scaling")
    else:
        print(f"  OUTCOME: E5 wrong-attractor rate = {best_wa:.4f} (between 5% and 20%)")
        print("  CONCLUSION: E5 uncertain — needs further investigation")

    enc_acc = exp_b.get("encoder_only", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    if enc_acc > 0.80:
        print(f"\n  WARNING: Encoder-only ablation at {enc_acc:.4f} — dynamics may not be needed")

    ar_acc = exp_b.get("ar_baseline", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    if abs(e1_rev - ar_acc) < 0.05:
        print(f"\n  COMPETITIVE: UESD within 5% of AR ({e1_rev:.4f} vs {ar_acc:.4f})")
    elif e1_rev > ar_acc:
        print(f"\n  UNEXPECTED WIN: UESD > AR ({e1_rev:.4f} vs {ar_acc:.4f}) — validate carefully")

    # Exp C dynamics necessity
    exp_c = load_results("exp_c_sort")
    if exp_c:
        enc_sort = exp_c.get("encoder_only", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
        e1_sort = exp_c.get("track_a_e1", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
        print(f"\n--- Exp C: Dynamics Necessity (Sort) ---")
        if enc_sort < 0.80 and e1_sort >= 0.80:
            print(f"  DYNAMICS NECESSARY: encoder-only={enc_sort:.4f} < 0.80, UESD={e1_sort:.4f}")
            print(f"  The encoder-only confound from Exp A/B is resolved.")
        elif enc_sort >= 0.80:
            print(f"  DYNAMICS NOT YET PROVEN NECESSARY: encoder-only={enc_sort:.4f} on sort")
            print(f"  Need harder tasks to separate dynamics contribution.")
        else:
            print(f"  BOTH STRUGGLE: encoder-only={enc_sort:.4f}, UESD={e1_sort:.4f}")
            print(f"  Sort may be too hard at current scale.")

    # Exp D dynamics necessity (compositional)
    exp_d = load_results("exp_d_compositional")
    if exp_d:
        print(f"\n--- Exp D: Dynamics Necessity (Compositional) ---")
        necessity_confirmed = False
        for task in ["addition", "dedup"]:
            enc_key = f"{task}_enc"
            e1_key = f"{task}_e1"
            if enc_key not in exp_d or e1_key not in exp_d:
                continue
            enc_tok = exp_d[enc_key]["eval"]["token_accuracy"]["token_acc"]
            enc_seq = exp_d[enc_key]["eval"]["token_accuracy"]["seq_acc"]
            best_uesd_tok = exp_d[e1_key]["eval"]["token_accuracy"]["token_acc"]
            best_uesd_seq = exp_d[e1_key]["eval"]["token_accuracy"]["seq_acc"]
            best_uesd_label = "E1"
            for lam in [0.1, 1.0]:
                e5_key = f"{task}_e5_lam{lam}"
                if e5_key in exp_d:
                    e5_tok = exp_d[e5_key]["eval"]["token_accuracy"]["token_acc"]
                    e5_seq = exp_d[e5_key]["eval"]["token_accuracy"]["seq_acc"]
                    if e5_tok > best_uesd_tok or (e5_tok == best_uesd_tok and e5_seq > best_uesd_seq):
                        best_uesd_tok = e5_tok
                        best_uesd_seq = e5_seq
                        best_uesd_label = f"E5(lam={lam})"
            if enc_tok < 0.80 and best_uesd_tok >= 0.80:
                print(f"  {task.upper()}: DYNAMICS NECESSARY (enc={enc_tok:.4f}, best_UESD={best_uesd_tok:.4f} [{best_uesd_label}])")
                necessity_confirmed = True
            elif enc_seq < 0.50 and best_uesd_seq >= 0.50:
                print(f"  {task.upper()}: DYNAMICS NECESSARY by seq_acc (enc_seq={enc_seq:.4f}, best_UESD_seq={best_uesd_seq:.4f} [{best_uesd_label}])")
                necessity_confirmed = True
            elif enc_tok >= 0.80:
                print(f"  {task.upper()}: CONFOUND PERSISTS (enc={enc_tok:.4f}, seq={enc_seq:.4f})")
            else:
                print(f"  {task.upper()}: BOTH STRUGGLE (enc={enc_tok:.4f}, best_UESD={best_uesd_tok:.4f} [{best_uesd_label}])")
        if necessity_confirmed:
            print(f"\n  CONCLUSION: Dynamics necessity CONFIRMED on compositional tasks.")
            print(f"  The encoder-only confound from Exp A/B/C is resolved.")
        else:
            print(f"\n  CONCLUSION: Dynamics necessity still not confirmed.")


def summarize_exp_c():
    r = load_results("exp_c_sort")
    if not r:
        print("\nExperiment C results not found.")
        return

    print_header("EXPERIMENT C: SORT — DYNAMICS NECESSITY TEST")

    print(f"\nDevice: {r.get('device', 'unknown')}")

    print("\n--- Accuracy Comparison ---")
    print(f"  {'Model':25s} {'Token Acc':>10s} {'Seq Acc':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    for key, label in [("track_a_e1", "E1 (embed reg)")]:
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    for lam in [0.1, 1.0]:
        key = f"track_b_e5_lam{lam}"
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {'E5 (lam=' + str(lam) + ')':25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    for key, label in [("ar_baseline", "AR baseline"), ("encoder_only", "Encoder-only")]:
        if key in r:
            ta = r[key]["eval"]["token_accuracy"]
            print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

    # E5 diagnostics
    print("\n--- E5 Lambda Sweep Diagnostics ---")
    print(f"  {'Lambda':>6s} {'Acc':>7s} {'WA Rate':>8s} {'Conv%':>7s} {'Margin':>8s} {'Rho':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    for lam in [0.1, 1.0]:
        key = f"track_b_e5_lam{lam}"
        if key in r:
            ev = r[key]["eval"]
            acc = ev["token_accuracy"]["token_acc"]
            wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 0)
            conv = ev.get("wrong_attractor", {}).get("converged_frac", 0)
            margin = ev.get("decoder_margin", {}).get("mean_margin", 0)
            rho = ev.get("spectral_radius", {}).get("mean_rho", 0)
            print(f"  {lam:>6.1f} {acc:>7.4f} {wa:>8.4f} {conv:>7.4f} {margin:>8.4f} {rho:>7.4f}")

    # Update trajectory (per-step residual norms)
    for key, label in [("track_a_e1", "E1"), ("track_b_e5_lam0.1", "E5 lam=0.1"),
                       ("track_b_e5_lam1.0", "E5 lam=1.0")]:
        if key in r:
            traj = r[key].get("eval", {}).get("update_trajectory", {})
            if traj:
                vals = [traj.get(f"step_{i+1}", 0) for i in range(10)]
                print(f"\n  {label} convergence trajectory:")
                print(f"    " + " -> ".join(f"{v:.4f}" for v in vals))

    if "gates" in r:
        print("\n--- Gates ---")
        for k, v in r["gates"].items():
            print(f"  {k}: {v}")

    # Dynamics necessity verdict
    enc_acc = r.get("encoder_only", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    e1_acc = r.get("track_a_e1", {}).get("eval", {}).get("token_accuracy", {}).get("token_acc", 0)
    print("\n--- DYNAMICS NECESSITY ---")
    if enc_acc < 0.80 and e1_acc >= 0.80:
        print(f"  CONFIRMED: Encoder-only={enc_acc:.4f}, UESD={e1_acc:.4f}")
        print(f"  Dynamics provide {e1_acc - enc_acc:+.4f} improvement over encoder-only")
    elif enc_acc >= 0.80:
        print(f"  NOT CONFIRMED: Encoder-only={enc_acc:.4f} still solves the task")
        print(f"  Need harder tasks (longer sequences, compositional tasks)")
    else:
        print(f"  INCONCLUSIVE: Both struggle (Enc={enc_acc:.4f}, UESD={e1_acc:.4f})")


def summarize_exp_d():
    r = load_results("exp_d_compositional")
    if not r:
        print("\nExperiment D results not found.")
        return

    print_header("EXPERIMENT D: COMPOSITIONAL — DYNAMICS NECESSITY (HARD)")

    print(f"\nDevice: {r.get('device', 'unknown')}")

    for task in ["addition", "dedup"]:
        e1_key = f"{task}_e1"
        if e1_key not in r:
            continue

        print(f"\n--- {task.upper()} ---")
        print(f"  {'Model':25s} {'Token Acc':>10s} {'Seq Acc':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")

        for key, label in [(f"{task}_e1", "E1 (embed reg)")]:
            if key in r:
                ta = r[key]["eval"]["token_accuracy"]
                print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

        for lam in [0.1, 1.0]:
            key = f"{task}_e5_lam{lam}"
            if key in r:
                ta = r[key]["eval"]["token_accuracy"]
                print(f"  {'E5 (lam=' + str(lam) + ')':25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

        for key, label in [(f"{task}_ar", "AR baseline"), (f"{task}_enc", "Encoder-only")]:
            if key in r:
                ta = r[key]["eval"]["token_accuracy"]
                print(f"  {label:25s} {ta['token_acc']:>10.4f} {ta['seq_acc']:>10.4f}")

        # E5 diagnostics
        print(f"\n  E5 Diagnostics:")
        print(f"  {'Lambda':>6s} {'Acc':>7s} {'WA':>7s} {'Conv%':>7s} {'Margin':>8s} {'Rho':>7s}")
        print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
        for lam in [0.1, 1.0]:
            key = f"{task}_e5_lam{lam}"
            if key in r:
                ev = r[key]["eval"]
                acc = ev["token_accuracy"]["token_acc"]
                wa = ev.get("wrong_attractor", {}).get("wrong_attractor_rate", 0)
                conv = ev.get("wrong_attractor", {}).get("converged_frac", 0)
                margin = ev.get("decoder_margin", {}).get("mean_margin", 0)
                rho = ev.get("spectral_radius", {}).get("mean_rho", 0)
                print(f"  {lam:>6.1f} {acc:>7.4f} {wa:>7.4f} {conv:>7.4f} {margin:>8.4f} {rho:>7.4f}")

        # Gates
        gates_key = f"{task}_gates"
        if gates_key in r:
            print(f"\n  Gates:")
            for k, v in r[gates_key].items():
                print(f"    {k}: {v}")

    # Dynamics necessity summary
    print("\n--- DYNAMICS NECESSITY SUMMARY ---")
    for task in ["addition", "dedup"]:
        gates = r.get(f"{task}_gates", {})
        if not gates:
            continue
        enc_tok = gates.get("encoder_only_token_acc", "?")
        enc_seq = gates.get("encoder_only_seq_acc", "?")
        e1_eval = r.get(f"{task}_e1", {}).get("eval", {})
        e1_tok = e1_eval.get("token_accuracy", {}).get("token_acc", 0)
        verdict = gates.get("dynamics_necessity", gates.get("encoder_confound", "?"))
        print(f"  {task.upper()}: UESD={e1_tok:.4f}, Enc-only={enc_tok} | {verdict}")


if __name__ == "__main__":
    exp_a = load_results("exp_a_copy")
    exp_b = load_results("exp_b_reversal")
    exp_c = load_results("exp_c_sort")
    exp_d = load_results("exp_d_compositional")

    if exp_a:
        summarize_exp_a()
    else:
        print("Experiment A results not available yet.")

    if exp_b:
        summarize_exp_b()

    if exp_c:
        summarize_exp_c()

    if exp_d:
        summarize_exp_d()

    decision_table(exp_a, exp_b)
