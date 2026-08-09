"""Audit the UESD status claims against checked-in result JSONs.

Reads JSON artifacts only; no training, no GPU. This is the canonical
validation command for the 06 line (see STATUS.md).

Scope constraint: no check in this file interprets D22 k-suppression as
evidence of correct fixed-point convergence. The two are deliberately
audited as separate claims — D22 is a compute-window robustness result,
and D40 is the controlling (negative) fixed-point result.
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main():
    d40 = json.loads(
        (RESULTS / "exp_d40_extended_convergence.json").read_text(encoding="utf-8")
    )
    d22 = json.loads(
        (RESULTS / "exp_d22_robust_dynamics.json").read_text(encoding="utf-8")
    )

    runs = d40["runs"]
    grid = {(s, l) for s in (42, 137, 256, 512) for l in (0.0, 0.5, 1.0, 3.0)}
    done = {(r["seed"], r["lambda_sc"]) for r in runs}

    # 1. D40 completed 15 of the intended 16-run grid (seed 512 / lambda 3.0 crashed).
    check(
        "d40_15_of_16_grid",
        len(runs) == 15 and done <= grid and grid - done == {(512, 3.0)},
        f"completed={len(runs)}, missing={sorted(grid - done)}",
    )

    multi = {
        (r["seed"], r["lambda_sc"]): r["phases"]["C"]["eval"]["multi_T"] for r in runs
    }

    # 2. Every completed run passes the in-window accuracy gate at T=10.
    worst_t10 = min(m["10"]["h_seq_acc"] for m in multi.values())
    check("d40_in_window_accuracy_T10", worst_t10 >= 0.999, f"min T=10 acc={worst_t10}")

    # 3. Extended-T gates fail: no run converges >=95% and every lambda>=0.5 run
    #    collapses to <=1% sequence accuracy by T=200.
    max_conv = max(m["200"]["converged_frac"] for m in multi.values())
    sc_t200 = max(
        m["200"]["h_seq_acc"] for (s, l), m in multi.items() if l >= 0.5
    )
    check("d40_extended_T_convergence_fails", max_conv < 0.95, f"max conv@200={max_conv}")
    check("d40_extended_T_accuracy_collapses", sc_t200 <= 0.01, f"max sc acc@200={sc_t200}")

    # 4. The only substantially converged case decodes to wrong attractors.
    nv = multi[(42, 3.0)]["200"]
    check(
        "d40_nonvacuous_wrong_attractors",
        nv["converged_frac"] >= 0.90
        and nv["wrong_attractor_rate"] >= 0.99
        and nv["h_seq_acc"] <= 0.001,
        f"conv={nv['converged_frac']}, WA={nv['wrong_attractor_rate']}, acc={nv['h_seq_acc']}",
    )

    # 5. D22 variable-T claim stays bounded to compute-window robustness:
    #    VT holds the window at T=32 where fixed-T baseline degrades.
    vt32 = d22["variable_t"]["summary"]["mean_T32_seq"]
    base32 = d22["baseline"]["summary"]["mean_T32_seq"]
    check(
        "d22_vt_compute_window_robustness",
        vt32 >= 0.99 and base32 <= 0.90 and vt32 > base32,
        f"VT T32={vt32}, baseline T32={base32}",
    )

    # 6. D22 denoising arm is negative (both denoising and combined collapse).
    den10 = d22["denoising"]["summary"]["mean_T10_seq"]
    comb10 = d22["combined"]["summary"]["mean_T10_seq"]
    check("d22_denoising_negative", den10 <= 0.01 and comb10 <= 0.01,
          f"denoising T10={den10}, combined T10={comb10}")

    failed = [c for c in checks if not c["ok"]]
    print(json.dumps({
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
    }, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
