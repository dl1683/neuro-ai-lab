"""Audit the UESD status claims against checked-in evidence JSON.

Reads result artifacts and the JSONL ledger only; no training, no GPU. This is the canonical
validation command for the 06 line (see STATUS.md).

Scope constraint: no check in this file interprets D22 k-suppression as
evidence of correct fixed-point convergence. The two are deliberately
audited as separate claims — D22 is a compute-window robustness result,
and D40 is the controlling (negative) fixed-point result.
"""

import hashlib
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
LEDGER = Path(__file__).resolve().parents[1] / "ledger.jsonl"
SVAMP_TERMINAL_VOID_REASON = (
    "NO_RECOVERABLE_NUMERIC_CONTENT_FOR_PERMITTED_PARSER_REPAIR"
)

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def canonical_lf_sha256(path: Path) -> str:
    """Hash Git's canonical LF representation, independent of checkout EOLs."""
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()


def main():
    d40 = json.loads(
        (RESULTS / "exp_d40_extended_convergence.json").read_text(encoding="utf-8")
    )
    d22 = json.loads(
        (RESULTS / "exp_d22_robust_dynamics.json").read_text(encoding="utf-8")
    )
    e1 = json.loads(
        (RESULTS / "exp_e1_task_band.json").read_text(encoding="utf-8")
    )
    e1_svamp_initial = json.loads(
        (
            RESULTS / "exp_e1_task_band_svamp_initial_parser_miss.json"
        ).read_text(encoding="utf-8")
    )
    ledger_entries = [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    runs = d40["runs"]
    grid = {
        (seed, lambda_sc)
        for seed in (42, 137, 256, 512)
        for lambda_sc in (0.0, 0.5, 1.0, 3.0)
    }
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
        metrics["200"]["h_seq_acc"]
        for (_seed, lambda_sc), metrics in multi.items()
        if lambda_sc >= 0.5
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

    # 7. E1 is a complete canonical 256-example gate, not a smoke result.
    check(
        "e1_complete_canonical_cohort",
        e1["status"] == "COMPLETE"
        and e1["mode"] == "canonical"
        and e1["sample_count"] == 256
        and len(e1["per_example"]) == 256,
        (
            f"status={e1['status']}, mode={e1['mode']}, "
            f"sample_count={e1['sample_count']}, records={len(e1['per_example'])}"
        ),
    )

    # 8. Every example is accounted for with explicit denominators.
    extraction_failures = e1["extraction_failures"]
    accounted = (
        e1["correct_count"]
        + e1["valid_extracted_incorrect_count"]
        + extraction_failures["numerator"]
    )
    check(
        "e1_outcome_accounting",
        e1["correct_count"] == 18
        and e1["valid_extracted_incorrect_count"] == 238
        and extraction_failures == {"numerator": 0, "denominator": 256, "rate": 0.0}
        and accounted == 256
        and e1["exact_answer_accuracy"]
        == {"numerator": 18, "denominator": 256, "rate": 0.0703125},
        (
            f"correct={e1['correct_count']}, "
            f"valid_incorrect={e1['valid_extracted_incorrect_count']}, "
            f"extraction_failures={extraction_failures}, accounted={accounted}"
        ),
    )

    # 9. The initial parser and protocol integrity controls survived.
    answer_extraction = e1["protocol"]["answer_extraction"]
    leakage = e1["protocol"]["leakage_preflight"]
    batch_equivalence = e1["protocol"]["decoding"][
        "batch_vs_unbatched_equivalence"
    ]
    check(
        "e1_protocol_integrity",
        answer_extraction["parser_attempt"] == "initial"
        and leakage["status"] == "PASS"
        and leakage["overlap_count"] == 0
        and batch_equivalence["status"] == "PASS"
        and extraction_failures["rate"] <= 0.05
        and e1["cap_reached"]["denominator"] == 256,
        (
            f"parser={answer_extraction['parser_attempt']}, "
            f"leakage={leakage['status']}, batch={batch_equivalence['status']}, "
            f"extraction_rate={extraction_failures['rate']}"
        ),
    )

    # 9b. The artifact's checkout-stable canonical-LF bytes and documented
    #     protocol fields are bound (dataset revision-pinned, greedy, five
    #     demonstrations, cap count).
    artifact_sha = canonical_lf_sha256(RESULTS / "exp_e1_task_band.json")
    proto = e1["protocol"]
    demo_indices = proto["five_shot_demonstrations"]["indices"]
    if isinstance(demo_indices, str):
        demo_indices = json.loads(demo_indices)
    check(
        "e1_artifact_hash_and_protocol_binding",
        artifact_sha == "d5bd9d3b5bbd0a902625df6341973b0c4c6996d6ebe94f965ef6497cd15a0c62"
        and e1["model"] == "base-A"
        and proto["dataset"] == "GSM8K"
        and proto["dataset_revision"] == "740312add88f781978c0658806c59bc2815b9866"
        and demo_indices == [0, 1, 2, 3, 4]
        and proto["decoding"]["strategy"] == "greedy"
        and proto["answer_extraction"]["parser_attempt"] == "initial"
        and e1["cap_reached"]["numerator"] == 15,
        f"sha={artifact_sha[:12]}..., model={e1.get('model')}, dataset={proto['dataset']}",
    )

    # 10. The frozen gate maps this valid below-band result to ABORT-AND-SWAP.
    verdict = e1["verdict"]
    check(
        "e1_below_band_abort_and_swap",
        e1["correct_count"] < 26
        and e1["correct_count"] < 40
        and verdict["token"] == "ABORT-AND-SWAP"
        and verdict["reason"] == "below_band_or_fewer_than_40_correct"
        and not verdict["terminal"],
        f"correct={e1['correct_count']}, verdict={verdict}",
    )

    # 11. The one allowed SVAMP fallback's initial parser attempt is a full
    #     canonical-mode cohort, but it is not a completed gate result.
    check(
        "e1_svamp_initial_complete_cohort",
        e1_svamp_initial["status"] == "PARSER_REPAIR_REQUIRED"
        and e1_svamp_initial["mode"] == "canonical"
        and e1_svamp_initial["sample_count"] == 256
        and len(e1_svamp_initial["per_example"]) == 256
        and not (RESULTS / "exp_e1_task_band_svamp.json").exists(),
        (
            f"status={e1_svamp_initial['status']}, "
            f"mode={e1_svamp_initial['mode']}, "
            f"sample_count={e1_svamp_initial['sample_count']}, "
            f"records={len(e1_svamp_initial['per_example'])}"
        ),
    )

    # 12. All SVAMP outcomes are accounted for with explicit denominators.
    svamp_extraction_failures = e1_svamp_initial["extraction_failures"]
    svamp_accounted = (
        e1_svamp_initial["correct_count"]
        + e1_svamp_initial["valid_extracted_incorrect_count"]
        + svamp_extraction_failures["numerator"]
    )
    check(
        "e1_svamp_initial_outcome_accounting",
        e1_svamp_initial["correct_count"] == 66
        and e1_svamp_initial["valid_extracted_incorrect_count"] == 177
        and svamp_extraction_failures
        == {"numerator": 13, "denominator": 256, "rate": 0.05078125}
        and svamp_accounted == 256
        and e1_svamp_initial["exact_answer_accuracy"]
        == {"numerator": 66, "denominator": 256, "rate": 0.2578125},
        (
            f"correct={e1_svamp_initial['correct_count']}, "
            "valid_incorrect="
            f"{e1_svamp_initial['valid_extracted_incorrect_count']}, "
            f"extraction_failures={svamp_extraction_failures}, "
            f"accounted={svamp_accounted}"
        ),
    )

    # 13. Bind the immutable initial-attempt bytes and the frozen SVAMP protocol.
    svamp_initial_path = RESULTS / "exp_e1_task_band_svamp_initial_parser_miss.json"
    svamp_initial_sha = canonical_lf_sha256(svamp_initial_path)
    svamp_proto = e1_svamp_initial["protocol"]
    svamp_demo_indices = svamp_proto["five_shot_demonstrations"]["indices"]
    if isinstance(svamp_demo_indices, str):
        svamp_demo_indices = json.loads(svamp_demo_indices)
    check(
        "e1_svamp_initial_artifact_hash_and_protocol_binding",
        svamp_initial_sha
        == "d092eb628a5279085e6c2de2974463937efe95f4c1c7d769ff7a83e21d0b242d"
        and e1_svamp_initial["model"] == "base-A"
        and svamp_proto["dataset"] == "SVAMP"
        and svamp_proto["dataset_revision"]
        == "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
        and svamp_demo_indices == [0, 1, 2, 3, 4]
        and svamp_proto["decoding"]["strategy"] == "greedy"
        and svamp_proto["decoding"]["batch_size"] == 1
        and svamp_proto["decoding"]["repeat_determinism"]["status"] == "PASS"
        and svamp_proto["leakage_preflight"]["status"] == "PASS"
        and svamp_proto["answer_extraction"]["parser_attempt"] == "initial"
        and e1_svamp_initial["cap_reached"]
        == {"numerator": 0, "denominator": 256, "rate": 0.0},
        (
            f"sha={svamp_initial_sha[:12]}..., "
            f"model={e1_svamp_initial.get('model')}, "
            f"dataset={svamp_proto['dataset']}"
        ),
    )

    # 14. Bind the immutable artifact's point-in-time initial-attempt verdict.
    #     The later terminal adjudication is separately bound to the ledger.
    svamp_verdict = e1_svamp_initial["verdict"]
    check(
        "e1_svamp_initial_verdict_mapping",
        26 <= e1_svamp_initial["correct_count"] <= 217
        and e1_svamp_initial["correct_count"] >= 40
        and e1_svamp_initial["valid_extracted_incorrect_count"] >= 40
        and svamp_extraction_failures["rate"] > 0.05
        and svamp_verdict["token"] == "PARSER-REPAIR-REQUIRED"
        and svamp_verdict["reason"]
        == "initial_extraction_failure_rate_above_5_percent"
        and not svamp_verdict["terminal"],
        (
            f"correct={e1_svamp_initial['correct_count']}, "
            f"valid_incorrect={e1_svamp_initial['valid_extracted_incorrect_count']}, "
            f"extraction_rate={svamp_extraction_failures['rate']}, "
            f"verdict={svamp_verdict}"
        ),
    )

    # 15. The extraction failures contain no numeric output for a parser to
    #     recover; all are the literal empty answer prefix emitted by the model.
    svamp_failed_records = [
        record
        for record in e1_svamp_initial["per_example"]
        if record["extraction_failed"]
    ]
    check(
        "e1_svamp_initial_failure_characterization",
        len(svamp_failed_records) == 13
        and all(record["response"] == "Answer:" for record in svamp_failed_records)
        and all(record["generated_tokens"] == 3 for record in svamp_failed_records),
        (
            f"failed_records={len(svamp_failed_records)}, "
            "responses="
            f"{sorted({record['response'] for record in svamp_failed_records})}"
        ),
    )

    # 16. The terminal taxonomy is derived without mutating the immutable
    #     initial-miss artifact: all unextractable records are model-empty, so
    #     there are no parser-recognition failures and all categories exhaust N.
    svamp_correct_count = sum(
        bool(record["correct"]) for record in e1_svamp_initial["per_example"]
    )
    svamp_valid_incorrect_count = sum(
        bool(record["valid_extracted_incorrect"])
        for record in e1_svamp_initial["per_example"]
    )
    svamp_model_empty_count = sum(
        record["extracted_answer"] is None and record["response"] == "Answer:"
        for record in e1_svamp_initial["per_example"]
    )
    svamp_parser_recognition_failure_count = (
        svamp_extraction_failures["numerator"] - svamp_model_empty_count
    )
    check(
        "e1_svamp_terminal_category_accounting",
        svamp_correct_count == 66
        and svamp_valid_incorrect_count == 177
        and svamp_model_empty_count == 13
        and svamp_parser_recognition_failure_count == 0
        and e1_svamp_initial["exact_answer_failure_count"] == 190
        and (
            svamp_correct_count
            + svamp_valid_incorrect_count
            + svamp_model_empty_count
            + svamp_parser_recognition_failure_count
        )
        == 256,
        (
            f"correct={svamp_correct_count}, "
            f"valid_incorrect={svamp_valid_incorrect_count}, "
            f"model_empty={svamp_model_empty_count}, "
            f"parser_recognition_failure={svamp_parser_recognition_failure_count}, "
            f"exact_answer_failures={e1_svamp_initial['exact_answer_failure_count']}"
        ),
    )

    # 17. Bind the terminal adjudication to the immutable artifact hash, exact
    #     category counts, denominators, ceiling, and frozen reason code.
    terminal_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "uesd-e1-task-band-base-a-svamp-terminal-void"
    ]
    terminal_entry = terminal_entries[0] if len(terminal_entries) == 1 else {}
    terminal_metrics = terminal_entry.get("metrics", {})
    check(
        "e1_svamp_terminal_void_binding",
        len(terminal_entries) == 1
        and terminal_entry.get("status") == "void"
        and terminal_metrics.get("artifact_sha256") == svamp_initial_sha
        and terminal_metrics.get("correct")
        == {"numerator": 66, "denominator": 256, "rate": 0.2578125}
        and terminal_metrics.get("valid_extracted_incorrect")
        == {"numerator": 177, "denominator": 256, "rate": 0.69140625}
        and terminal_metrics.get("model_empty_non_answers")
        == {"numerator": 13, "denominator": 256, "rate": 0.05078125}
        and terminal_metrics.get("extraction_failures")
        == {"numerator": 13, "denominator": 256, "rate": 0.05078125}
        and terminal_metrics.get("exact_answer_failures")
        == {"numerator": 190, "denominator": 256, "rate": 0.7421875}
        and terminal_metrics.get("maximum_extraction_failure_count") == 12
        and terminal_metrics.get("verdict") == "VOID"
        and terminal_metrics.get("verdict_reason") == SVAMP_TERMINAL_VOID_REASON,
        (
            f"entry_count={len(terminal_entries)}, "
            f"artifact_sha={terminal_metrics.get('artifact_sha256')}, "
            f"verdict={terminal_metrics.get('verdict')}, "
            f"reason={terminal_metrics.get('verdict_reason')}"
        ),
    )

    failed = [c for c in checks if not c["ok"]]
    print(json.dumps({
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
    }, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
