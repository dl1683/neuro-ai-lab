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
    e1_base_b = json.loads(
        (RESULTS / "exp_e1_task_band_base_b.json").read_text(encoding="utf-8")
    )
    e2 = json.loads(
        (RESULTS / "exp_e2_latch_mechanics.json").read_text(encoding="utf-8")
    )
    e2_diag = json.loads(
        (RESULTS / "exp_e2_diag.json").read_text(encoding="utf-8")
    )
    e2_diag_repair = json.loads(
        (RESULTS / "exp_e2_diag_stage0_instrumented.json").read_text(
            encoding="utf-8"
        )
    )
    e3_preflight = json.loads(
        (RESULTS / "exp_e3_preflight.json").read_text(encoding="utf-8")
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

    # 18. The successor artifact is the single complete canonical base-B run
    #     and is bound to the preregistered disjoint cohort and frozen protocol.
    base_b_path = RESULTS / "exp_e1_task_band_base_b.json"
    base_b_sha = canonical_lf_sha256(base_b_path)
    base_b_proto = e1_base_b["protocol"]
    base_b_selection = base_b_proto["selection"]
    check(
        "e1_base_b_artifact_and_protocol_binding",
        base_b_sha
        == "9c57a10f3aa64c43fa34819255f4cf4e004cc5ab74afaa838c02c4a06a271de2"
        and e1_base_b["status"] == "COMPLETE"
        and e1_base_b["mode"] == "canonical"
        and e1_base_b["model"] == "base-B"
        and e1_base_b["sample_count"] == 256
        and len(e1_base_b["per_example"]) == 256
        and base_b_proto["dataset"] == "GSM8K"
        and base_b_proto["dataset_revision"]
        == "740312add88f781978c0658806c59bc2815b9866"
        and base_b_proto["decoding"]["strategy"] == "greedy"
        and base_b_proto["decoding"]["batch_size"] == 1
        and base_b_proto["decoding"]["repeat_determinism"]["status"] == "PASS"
        and base_b_proto["leakage_preflight"]["status"] == "PASS"
        and base_b_proto["leakage_preflight"]["overlap_count"] == 0
        and base_b_proto["answer_extraction"]["parser_attempt"] == "initial"
        and base_b_selection["canonical_indices_sha256"]
        == "670705ea2936f75f0e90a4048d3f5b5ec3a63b42577c0d7a9df87253b77444ff"
        and base_b_selection["canonical_unique_count"] == 256
        and base_b_selection["overlap_with_base_a_count"] == 0
        and base_b_selection["remaining_unconsumed_official_test_count"] == 807,
        (
            f"sha={base_b_sha[:12]}..., status={e1_base_b['status']}, "
            f"records={len(e1_base_b['per_example'])}, "
            f"cohort_sha={base_b_selection['canonical_indices_sha256'][:12]}..."
        ),
    )

    # 19. Re-derive all four mutually exclusive successor categories from the
    #     per-example records and bind every numerator and denominator.
    base_b_category_names = (
        "correct_numeric",
        "valid_extracted_incorrect",
        "model_empty_non_answer",
        "parser_recognition_failure",
    )
    base_b_derived_counts = {
        name: sum(bool(record[name]) for record in e1_base_b["per_example"])
        for name in base_b_category_names
    }
    base_b_expected_counts = {
        "correct_numeric": 143,
        "valid_extracted_incorrect": 104,
        "model_empty_non_answer": 0,
        "parser_recognition_failure": 9,
    }
    base_b_record_one_hot = all(
        sum(bool(record[name]) for name in base_b_category_names) == 1
        for record in e1_base_b["per_example"]
    )
    check(
        "e1_base_b_four_category_accounting",
        base_b_derived_counts == base_b_expected_counts
        and base_b_record_one_hot
        and sum(base_b_derived_counts.values()) == 256
        and all(
            e1_base_b["outcome_categories"][name]["numerator"] == count
            and e1_base_b["outcome_categories"][name]["denominator"] == 256
            for name, count in base_b_expected_counts.items()
        )
        and e1_base_b["usable_incorrect"]
        == {"numerator": 104, "denominator": 256, "rate": 0.40625},
        f"derived={base_b_derived_counts}, one_hot={base_b_record_one_hot}",
    )

    # 20. Bind the preregistered PASS mapping, including both population floors
    #     and the independent model-empty/parser-recognition ceilings.
    base_b_verdict = e1_base_b["verdict"]
    check(
        "e1_base_b_pass_mapping",
        26 <= e1_base_b["correct_numeric_count"] <= 217
        and e1_base_b["correct_numeric_count"] >= 40
        and e1_base_b["usable_incorrect_count"] >= 40
        and e1_base_b["model_empty_non_answer_count"] <= 12
        and e1_base_b["parser_recognition_failure_count"] <= 12
        and base_b_verdict["token"] == "PASS"
        and base_b_verdict["reason"]
        == "all_successor_task_band_thresholds_satisfied"
        and not base_b_verdict["terminal"],
        (
            f"correct={e1_base_b['correct_numeric_count']}, "
            f"usable_incorrect={e1_base_b['usable_incorrect_count']}, "
            f"model_empty={e1_base_b['model_empty_non_answer_count']}, "
            "parser_recognition_failure="
            f"{e1_base_b['parser_recognition_failure_count']}, "
            f"verdict={base_b_verdict['token']}"
        ),
    )

    # 21. Bind the landed ledger entry to the immutable artifact, exact category
    #     denominators, measured compute, and narrow band-placement verdict.
    base_b_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "uesd-e1-task-band-base-b-gsm8k"
    ]
    base_b_entry = base_b_entries[0] if len(base_b_entries) == 1 else {}
    base_b_metrics = base_b_entry.get("metrics", {})
    check(
        "e1_base_b_ledger_binding",
        len(base_b_entries) == 1
        and base_b_entry.get("status") == "complete"
        and base_b_metrics.get("artifact_sha256") == base_b_sha
        and base_b_metrics.get("correct_numeric")
        == {"numerator": 143, "denominator": 256, "rate": 0.55859375}
        and base_b_metrics.get("valid_extracted_incorrect")
        == {"numerator": 104, "denominator": 256, "rate": 0.40625}
        and base_b_metrics.get("model_empty_non_answers")
        == {"numerator": 0, "denominator": 256, "rate": 0.0}
        and base_b_metrics.get("parser_recognition_failures")
        == {"numerator": 9, "denominator": 256, "rate": 0.03515625}
        and base_b_metrics.get("wall_time_seconds") == 21412.4622991
        and base_b_metrics.get("throughput_generated_tokens_per_second")
        == 1.8718598219711238
        and base_b_metrics.get("verdict") == "PASS",
        (
            f"entry_count={len(base_b_entries)}, "
            f"artifact_sha={base_b_metrics.get('artifact_sha256')}, "
            f"verdict={base_b_metrics.get('verdict')}"
        ),
    )

    # 22. Bind the immutable E2 artifact and the runner-emitted pretest decision.
    #     The controlling post-evidence adjudication is checked separately below.
    e2_path = RESULTS / "exp_e2_latch_mechanics.json"
    e2_sha = canonical_lf_sha256(e2_path)
    e2_decision = e2["decision"]
    check(
        "e2_immutable_artifact_raw_decision_binding",
        e2_sha
        == "7842ca6f69ba3885fe7b03142b694e9c95950f195d31acd73f601d1e3f5a4075"
        and e2["schema_version"] == "1.1.0"
        and e2["experiment_id"] == "exp_e2_latch_mechanics"
        and e2["review_attestation"] == "INDEPENDENT_PRETRAINING_REVIEW_CLEAN"
        and e2["final_token"] == "FAIL"
        and e2_decision["final_token"] == "FAIL"
        and e2_decision["reason"] == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        and e2_decision["failed_seed_numerators"] == [42, 31415]
        and e2_decision["invalid_selector_fit_seed_numerators"] == []
        and e2_decision["model_seed_denominator"] == 2
        and not e2_decision["official_test_inspected"],
        (
            f"sha={e2_sha[:12]}..., token={e2['final_token']}, "
            f"decision={e2_decision}"
        ),
    )

    # 23. Both frozen training/selector populations and integrity boundaries
    #     were present before the scientific provenance miss.
    e2_validity = e2["validity_gates"]
    e2_seed_setup_ok = True
    for seed in (42, 31415):
        seed_key = str(seed)
        training = e2["training"][seed_key]
        e2_seed_setup_ok &= (
            training["common_training"]["processed_nonpadding_tokens"] == 500000
            and training["common_training"]["logical_arm_exposure"] == 500000
            and training["encoder_training"]["processed_nonpadding_tokens"] == 500000
            and training["selector_training_corpus"]["states"] == 491520
            and training["selector_calibration_corpus"]["states"] == 16384
            and training["selector_calibration_corpus"]["correct_states"] == 4096
            and training["selector_calibration_corpus"]["incorrect_states"] == 12288
            and training["selector_calibration_corpus"]["unique_problems"] == 1024
            and training["latent_critic_fit"]["fit_performed"]
            and e2_validity[f"seed_{seed}_selector_fit_performed"]["pass"]
            and not e2_validity[f"seed_{seed}_selector_provenance"]["pass"]
            and e2_validity[f"seed_{seed}_confidence_split_contract"]["pass"]
        )
    check(
        "e2_pretest_integrity_and_population_floors",
        e2_seed_setup_ok
        and e2["generator_audit"]["skeleton_hash_disjoint"]
        and e2["generator_audit"]["name_combination_disjoint"]
        and e2["feature_boundary_audit"]["pass"]
        and e2_validity["generator_and_split_integrity"]["pass"]
        and e2_validity["feature_boundary"]["pass"]
        and e2_validity["both_model_seeds_prepared"]
        == {"observed_numerator": 2, "required_denominator": 2, "pass": True},
        f"seed_setup_ok={e2_seed_setup_ok}, validity={e2_validity}",
    )

    # 24. Rebind every seedwise selector-provenance numerator, denominator,
    #     point estimate, and failed threshold. Undefined matched concordance
    #     has denominator zero and must not be represented as an observed zero.
    expected_e2_provenance = {
        "42": {
            "critic_concordant": 25183159.0,
            "confidence_concordant": 25163785.0,
            "critic_auroc": 0.5003444155057272,
            "confidence_auroc": 0.49995948870976764,
            "auroc_delta": 0.00038492679595952817,
        },
        "31415": {
            "critic_concordant": 25275202.5,
            "confidence_concordant": 25150211.0,
            "critic_auroc": 0.5021731555461884,
            "confidence_auroc": 0.49968979756037396,
            "auroc_delta": 0.002483357985814394,
        },
    }
    e2_provenance_ok = True
    for seed_key, expected in expected_e2_provenance.items():
        provenance = e2["training"][seed_key]["selector_provenance"]
        gates = provenance["gates"]
        e2_provenance_ok &= (
            provenance["critic_auroc"]
            == {
                "concordant_pair_units": expected["critic_concordant"],
                "positive_negative_pairs": 50331648,
                "positive_states": 4096,
                "negative_states": 12288,
                "value": expected["critic_auroc"],
            }
            and provenance["confidence_auroc"]
            == {
                "concordant_pair_units": expected["confidence_concordant"],
                "positive_negative_pairs": 50331648,
                "positive_states": 4096,
                "negative_states": 12288,
                "value": expected["confidence_auroc"],
            }
            and provenance["critic_minus_confidence_auroc"]
            == expected["auroc_delta"]
            and provenance["critic_selection_accuracy"]
            == {"correct": 256, "total": 1024, "value": 0.25}
            and provenance["confidence_selection_accuracy"]
            == {"correct": 256, "total": 1024, "value": 0.25}
            and provenance["critic_minus_confidence_selection_accuracy"] == 0.0
            and provenance["confidence_matched_concordance"]
            == {"concordant_pair_units": 0.0, "qualifying_pairs": 0, "value": None}
            and gates["overall_critic_auroc"]
            == {"threshold": 0.75, "observed": expected["critic_auroc"], "pass": False}
            and gates["confidence_matched_concordance"]["denominator"] == 0
            and gates["confidence_matched_concordance"]["observed"] is None
            and not gates["confidence_matched_concordance"]["pass"]
            and gates["critic_auroc_advantage_over_confidence"]["observed"]
            == expected["auroc_delta"]
            and not gates["critic_auroc_advantage_over_confidence"]["pass"]
            and gates["critic_selection_accuracy_advantage"]["critic_numerator"]
            == 256
            and gates["critic_selection_accuracy_advantage"]["confidence_numerator"]
            == 256
            and gates["critic_selection_accuracy_advantage"]["denominator"] == 1024
            and not gates["critic_selection_accuracy_advantage"]["pass"]
            and gates["on_policy_critic_auroc"]["observed"]
            == expected["critic_auroc"]
            and not gates["on_policy_critic_auroc"]["pass"]
            and not provenance["all_gates_pass"]
        )
    check(
        "e2_seedwise_selector_provenance_failure",
        e2_provenance_ok,
        f"expected={expected_e2_provenance}",
    )

    # 25. Calibration froze at chance with zero gain, arm 4 had no feasible
    #     delta, and every endpoint/test field remained explicitly uninspected.
    e2_t_star = e2["calibration_frozen_t_star"]
    e2_calibration_ok = (
        set(e2_t_star["candidate_metrics"]) == {str(h) for h in range(1, 16)}
        and all(
            metrics["per_seed"]["42"]
            == {"correct": 256, "total": 1024, "value": 0.25}
            and metrics["per_seed"]["31415"]
            == {"correct": 256, "total": 1024, "value": 0.25}
            and metrics["mean_seed_accuracy"] == 0.25
            for metrics in e2_t_star["candidate_metrics"].values()
        )
        and e2_t_star["selected_horizon"] == 1
        and e2_t_star["selected_mean_seed_accuracy"] == 0.25
        and e2_t_star["tie_break"] == "smaller_horizon"
        and not e2_t_star["test_inspected_before_freeze"]
    )
    e2_hysteresis_ok = True
    for seed_key in ("42", "31415"):
        hysteresis = e2["calibration_frozen_hysteresis_by_seed"][seed_key]
        e2_hysteresis_ok &= (
            hysteresis["selected_delta"] == 0.0
            and hysteresis["calibration_constraint_miss"]
            and not hysteresis["test_grid_evaluated"]
            and hysteresis["informational_only"]
            and all(
                candidate["b16_accuracy"]
                == {"numerator": 256, "denominator": 1024, "value": 0.25}
                and candidate["calibration_gain_retention"]
                == {
                    "arm4_minus_t1_correct_numerator": 0,
                    "t_star_minus_t1_correct_denominator": 0,
                    "accuracy_denominator": 1024,
                    "value": None,
                }
                and not candidate["feasible"]
                for candidate in hysteresis["candidate_calibration_metrics"].values()
            )
        )
    e2_evaluation = e2["evaluation"]
    e2_not_applicable_sections = (
        "encoder_control",
        "compute_by_seed",
        "accuracy_grid",
        "stratified_accuracy_grid",
        "trajectory_diagnostics",
        "trajectory_accounting_assertions",
        "endpoint_metrics",
        "paired_counterfactual_group_bootstrap",
    )
    check(
        "e2_calibration_and_uninspected_endpoint_binding",
        e2_calibration_ok
        and e2_hysteresis_ok
        and e2_evaluation["status"] == "not_applicable"
        and e2_evaluation["not_applicable_reason"]
        == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        and not e2_evaluation["official_test_inspected"]
        and all(
            e2_evaluation[section]["status"] == "not_applicable"
            and e2_evaluation[section]["not_applicable_reason"]
            == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
            for section in e2_not_applicable_sections
        )
        and e2["per_example_records"]["records_by_seed"] == {},
        (
            f"calibration_ok={e2_calibration_ok}, "
            f"hysteresis_ok={e2_hysteresis_ok}, "
            f"evaluation_status={e2_evaluation['status']}"
        ),
    )

    # 26. Bind the raw-run chronology to the immutable artifact, live runner and
    #     config hashes, complete seedwise provenance, and N/A endpoint boundary.
    e2_entries = [
        entry for entry in ledger_entries if entry.get("id") == "uesd-e2-latch-mechanics"
    ]
    e2_entry = e2_entries[0] if len(e2_entries) == 1 else {}
    e2_metrics = e2_entry.get("metrics", {})
    e2_ledger_provenance_ok = True
    for seed_key, expected in expected_e2_provenance.items():
        ledger_provenance = e2_metrics.get("selector_provenance", {}).get(seed_key, {})
        artifact_provenance = e2["training"][seed_key]["selector_provenance"]
        e2_ledger_provenance_ok &= (
            ledger_provenance.get("critic_auroc")
            == {
                "concordant_pair_units": expected["critic_concordant"],
                "positive_negative_pairs": 50331648,
                "positive_states": 4096,
                "negative_states": 12288,
                "value": expected["critic_auroc"],
            }
            and ledger_provenance.get("confidence_auroc")
            == artifact_provenance["confidence_auroc"]["value"]
            and ledger_provenance.get("on_policy_critic_auroc")
            == artifact_provenance["gates"]["on_policy_critic_auroc"]["observed"]
            and ledger_provenance.get("critic_minus_confidence_auroc")
            == artifact_provenance["critic_minus_confidence_auroc"]
            and ledger_provenance.get("critic_selection_accuracy")
            == {"numerator": 256, "denominator": 1024, "rate": 0.25}
            and ledger_provenance.get("confidence_selection_accuracy")
            == {"numerator": 256, "denominator": 1024, "rate": 0.25}
            and ledger_provenance.get("selection_accuracy_delta") == 0.0
            and ledger_provenance.get("confidence_matched_concordance")
            == {"numerator": 0.0, "denominator": 0, "rate": None}
        )
    e2_runner_path = Path(__file__).resolve().parent / "exp_e2_latch_mechanics.py"
    e2_config_path = Path(__file__).resolve().parent / "exp_e2_latch_mechanics_config.json"
    live_code_sha = hashlib.sha256(e2_runner_path.read_bytes()).hexdigest()
    live_config_sha = hashlib.sha256(e2_config_path.read_bytes()).hexdigest()
    check(
        "e2_raw_run_ledger_and_live_hash_binding",
        len(e2_entries) == 1
        and e2_entry.get("status") == "fail"
        and e2_entry.get("artifacts")
        == ["experiments/06_uesd/results/exp_e2_latch_mechanics.json"]
        and e2_metrics.get("artifact_sha256") == e2_sha
        and e2_metrics.get("final_token") == "FAIL"
        and e2_metrics.get("verdict_reason")
        == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        and not e2_metrics.get("official_test_inspected")
        and e2_metrics.get("failed_model_seeds")
        == {"numerator": 2, "denominator": 2, "seeds": [42, 31415]}
        and e2_metrics.get("invalid_selector_fits")
        == {"numerator": 0, "denominator": 2}
        and e2_metrics.get("calibration_accuracy", {}).get("pooled")
        == {"numerator": 512, "denominator": 2048, "rate": 0.25}
        and e2_metrics.get("endpoint_metrics", {}).get("status")
        == "not_applicable"
        and e2_metrics.get("endpoint_metrics", {}).get("reason")
        == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        and e2_ledger_provenance_ok
        and e2_metrics.get("code_sha256") == e2["hashes"]["code_sha256"]
        and e2_metrics.get("config_sha256") == e2["hashes"]["config_sha256"]
        and live_code_sha == e2["hashes"]["code_sha256"]
        and live_config_sha == e2["hashes"]["config_sha256"]
        and e2_metrics.get("wall_time_seconds") == 1181.121862999993
        and e2_metrics.get("peak_vram_allocated_bytes") == 1076839936
        and e2_metrics.get("peak_vram_reserved_bytes") == 1256194048,
        (
            f"entry_count={len(e2_entries)}, "
            f"artifact_sha={e2_metrics.get('artifact_sha256')}, "
            f"token={e2_metrics.get('final_token')}, "
            f"provenance_ok={e2_ledger_provenance_ok}, "
            f"live_hashes={(live_code_sha, live_config_sha)}"
        ),
    )

    # 27. The preregistered 200-pair floor precedes FAIL. Both seeds supplied
    #     zero qualifying pairs, so the append-only post-evidence VOID controls
    #     while the raw immutable artifact remains unchanged.
    e2_adjudication_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "uesd-e2-latch-mechanics-post-evidence-adjudication"
    ]
    e2_adjudication = (
        e2_adjudication_entries[0] if len(e2_adjudication_entries) == 1 else {}
    )
    adjudication_metrics = e2_adjudication.get("metrics", {})
    adjudication_provenance = adjudication_metrics.get(
        "selector_provenance_denominator_complete", {}
    )
    adjudication_provenance_ok = True
    for seed_key, expected in expected_e2_provenance.items():
        corrected = adjudication_provenance.get(seed_key, {})
        adjudication_provenance_ok &= (
            corrected.get("critic_auroc")
            == {
                "concordant_pair_units": expected["critic_concordant"],
                "positive_negative_pair_denominator": 50331648,
                "positive_states": 4096,
                "negative_states": 12288,
                "value": expected["critic_auroc"],
            }
            and corrected.get("confidence_auroc")
            == {
                "concordant_pair_units": expected["confidence_concordant"],
                "positive_negative_pair_denominator": 50331648,
                "positive_states": 4096,
                "negative_states": 12288,
                "value": expected["confidence_auroc"],
            }
            and corrected.get("on_policy_critic_auroc")
            == corrected.get("critic_auroc")
            and corrected.get("critic_minus_confidence_auroc")
            == expected["auroc_delta"]
            and corrected.get("critic_selection_accuracy")
            == {"numerator": 256, "denominator": 1024, "rate": 0.25}
            and corrected.get("confidence_selection_accuracy")
            == {"numerator": 256, "denominator": 1024, "rate": 0.25}
            and corrected.get("confidence_matched_concordance")
            == {
                "concordant_pair_units": 0.0,
                "qualifying_pair_denominator": 0,
                "rate": None,
            }
        )
    expected_floor = {
        "42": {
            "observed_numerator": 0,
            "required_denominator": 200,
            "qualifying_pair_denominator": 0,
            "pass": False,
        },
        "31415": {
            "observed_numerator": 0,
            "required_denominator": 200,
            "qualifying_pair_denominator": 0,
            "pass": False,
        },
        "failed_seed_count": {"numerator": 2, "denominator": 2},
    }
    check(
        "e2_controlling_post_evidence_void_binding",
        len(e2_adjudication_entries) == 1
        and e2_adjudication.get("status") == "void"
        and e2_adjudication.get("artifacts")
        == ["experiments/06_uesd/results/exp_e2_latch_mechanics.json"]
        and adjudication_metrics.get("controlling_final_token") == "VOID"
        and adjudication_metrics.get("controlling_reason")
        == "INSUFFICIENT_CONFIDENCE_MATCHED_PAIRS_PER_SEED"
        and adjudication_metrics.get("artifact_emitted_token") == "FAIL"
        and adjudication_metrics.get("artifact_emitted_reason")
        == "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        and not adjudication_metrics.get("artifact_emitted_token_admissible")
        and adjudication_metrics.get("decision_precedence")
        == ["VOID", "FAIL", "PROCEED"]
        and adjudication_metrics.get("matched_pair_floor") == expected_floor
        and not adjudication_metrics.get("official_test_inspected")
        and adjudication_metrics.get("endpoint_metrics")
        == {
            "status": "not_applicable",
            "reason": "PRETEST_TERMINATION_BEFORE_OFFICIAL_TEST",
        }
        and adjudication_metrics.get("artifact_path")
        == "experiments/06_uesd/results/exp_e2_latch_mechanics.json"
        and adjudication_metrics.get("artifact_sha256") == e2_sha
        and adjudication_metrics.get("runner_path")
        == "experiments/06_uesd/exp_e2_latch_mechanics.py"
        and adjudication_metrics.get("code_sha256") == live_code_sha
        and adjudication_metrics.get("config_sha256") == live_config_sha
        and adjudication_provenance_ok,
        (
            f"entry_count={len(e2_adjudication_entries)}, "
            f"token={adjudication_metrics.get('controlling_final_token')}, "
            f"floor={adjudication_metrics.get('matched_pair_floor')}, "
            f"provenance_ok={adjudication_provenance_ok}"
        ),
    )

    # 28. Bind the immutable E2-DIAG artifact to its terminal operational VOID.
    #     This is explicitly not an adjudicated Stage-0 STOP or scientific miss.
    e2_diag_path = RESULTS / "exp_e2_diag.json"
    e2_diag_sha = canonical_lf_sha256(e2_diag_path)
    e2_diag_stage0 = e2_diag["stages"]["stage_0"]
    e2_diag_access = e2_diag["access_confirmation"]
    check(
        "e2_diag_terminal_operational_void_binding",
        e2_diag_sha
        == "41c035dd79fe42e9edb891dcc02664077c7d491cd7a2857fe7cc9ae16dc5ab37"
        and e2_diag["schema_version"] == "1.0.0"
        and e2_diag["experiment_id"] == "exp_e2_diag"
        and e2_diag["diagnostic_only"]
        and not e2_diag["adjudicates_semantic_ratchet"]
        and e2_diag["suite_status"] == "TERMINAL_OPERATIONAL_VOID"
        and e2_diag["final_route_token"] == "VOID_NO_ROUTE"
        and e2_diag_stage0["status"] == "VOID"
        and e2_diag_stage0["reason"]
        == "POST_ENDPOINT_RESULT_SERIALIZATION_FAILURE"
        and e2_diag["stages"]["stage_1"]["status"] == "NOT_RUN"
        and e2_diag["stages"]["stage_2"]["status"] == "NOT_RUN"
        and e2_diag["stages"]["stage_3"]["status"] == "NOT_RUN"
        and not any(e2_diag_access.values()),
        (
            f"sha={e2_diag_sha}, status={e2_diag_stage0['status']}, "
            f"route={e2_diag['final_route_token']}, access={e2_diag_access}"
        ),
    )

    # 29. Preserve the observed endpoint as non-adjudicating execution evidence,
    #     with every unavailable post-crash quantity represented as unavailable.
    e2_diag_training = e2_diag_stage0["training"]
    e2_diag_endpoint = e2_diag_stage0["registered_endpoint_observation"]
    e2_diag_curve = e2_diag_stage0["training_curve"]["observations"]
    check(
        "e2_diag_stage0_observation_and_unavailable_metrics",
        e2_diag_endpoint["update"] == 3000
        and e2_diag_endpoint["training_accuracy"]
        == {"numerator": 32, "denominator": 128, "rate": 0.25}
        and e2_diag_endpoint["cross_entropy_mean_stdout_rounded"] == 1.388611
        and not e2_diag_endpoint["scientific_gate_adjudicated"]
        and e2_diag_endpoint["reason"] == "OPERATIONAL_VOID_PRECEDENCE"
        and len(e2_diag_curve) == 30
        and [row["update"] for row in e2_diag_curve] == list(range(100, 3001, 100))
        and all(
            row["correct_numerator"] == 32
            and row["training_example_denominator"] == 128
            for row in e2_diag_curve
        )
        and min(row["cross_entropy_mean_stdout_rounded"] for row in e2_diag_curve)
        == 1.384766
        and max(row["cross_entropy_mean_stdout_rounded"] for row in e2_diag_curve)
        == 1.52124
        and e2_diag_training["completed_updates"] == 3000
        and e2_diag_training["processed_tokens"] == 2786051
        and e2_diag_training["processed_examples"] == 11998
        and e2_diag_training["pre_clip_gradient_norm"]["status"] == "unavailable"
        and e2_diag_training["clipped_updates"]
        == {
            "numerator": None,
            "denominator": 3000,
            "rate": None,
            "status": "unavailable",
        }
        and e2_diag_stage0["compute"]["status"] == "unavailable"
        and e2_diag["hashes"]["checkpoints"]["status"] == "unavailable"
        and e2_diag["protocol_deviations"][0]["scientific_miss_counted"] is False
        and e2_diag["protocol_deviations"][0]["repeat_permitted"] is False,
        (
            f"endpoint={e2_diag_endpoint}, updates={e2_diag_training['completed_updates']}, "
            f"tokens={e2_diag_training['processed_tokens']}"
        ),
    )

    # 30. Bind the reconstruction hashes and model/data integrity. Both code
    #     hashes are historical now that the separately authorized repair mode
    #     has landed; the original artifact remains immutable.
    e2_diag_runner_path = Path(__file__).resolve().parent / "exp_e2_diag.py"
    e2_diag_live_code_sha = hashlib.sha256(e2_diag_runner_path.read_bytes()).hexdigest()
    e2_diag_config_path = Path(__file__).resolve().parent / (
        "exp_e2_latch_mechanics_config.json"
    )
    e2_diag_live_config_sha = hashlib.sha256(
        e2_diag_config_path.read_bytes()
    ).hexdigest()
    e2_diag_hashes = e2_diag["hashes"]
    check(
        "e2_diag_hash_and_integrity_binding",
        e2_diag_hashes["launch_code_sha256"]
        == "d1648e7a6a1563e89b8cc3d0119902eefcef49ea588923f2656a1835751eaca2"
        and e2_diag_hashes["operational_landing_code_sha256"]
        == "6782e8c976697d2c4ea1c10aa30848a8c4c6fa0e99dfa11f2d09641cef7952c7"
        and e2_diag_live_config_sha == e2_diag_hashes["e2_config_sha256"]
        and e2_diag_hashes["stage_0_ordered_set_sha256"]
        == "7d28ffe443047ce49a2fbacf4d7b78f93000eb30c0f74db28e89232877e39e7b"
        and e2_diag_hashes["stdout_recovery_sha256"]
        == "5d64c739c0c147116b75b9a090421e74c5f33723202958677ea725626ecdca14"
        and e2_diag["model"]["seed"] == 42
        and e2_diag["model"]["trainable_parameter_count"] == 29509636
        and e2_diag["integrity"]["controller_train_examples"] == 8192
        and e2_diag["integrity"]["controller_train_label_counts"]
        == {"0": 2048, "1": 2048, "2": 2048, "3": 2048}
        and e2_diag["integrity"]["memorization_examples"] == 128
        and e2_diag["integrity"]["memorization_label_counts"]
        == {"0": 32, "1": 32, "2": 32, "3": 32}
        and not e2_diag["integrity"]["pass"]
        and e2_diag["integrity"]["post_endpoint_result_publication"]
        == {"pass": False, "reason": "GRADIENT_QUANTILE_Q_DTYPE_MISMATCH"},
        (
            f"live_code={e2_diag_live_code_sha}, "
            f"landing_code={e2_diag_hashes['operational_landing_code_sha256']}, "
            f"live_config={e2_diag_live_config_sha}"
        ),
    )

    # 31. Bind the append-only chronology to the immutable artifact and its
    #     non-adjudication/no-route language.
    e2_diag_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "uesd-e2-diag-stage0-operational-void"
    ]
    e2_diag_entry = e2_diag_entries[0] if len(e2_diag_entries) == 1 else {}
    e2_diag_metrics = e2_diag_entry.get("metrics", {})
    check(
        "e2_diag_ledger_binding",
        len(e2_diag_entries) == 1
        and e2_diag_entry.get("status") == "void"
        and e2_diag_entry.get("artifacts")
        == ["experiments/06_uesd/results/exp_e2_diag.json"]
        and e2_diag_metrics.get("artifact_sha256") == e2_diag_sha
        and e2_diag_metrics.get("suite_status") == "TERMINAL_OPERATIONAL_VOID"
        and e2_diag_metrics.get("final_route_token") == "VOID_NO_ROUTE"
        and e2_diag_metrics.get("reason")
        == "POST_ENDPOINT_RESULT_SERIALIZATION_FAILURE"
        and e2_diag_metrics.get("registered_endpoint_observation", {}).get(
            "training_accuracy"
        )
        == {"numerator": 32, "denominator": 128, "rate": 0.25}
        and not e2_diag_metrics.get("registered_endpoint_observation", {}).get(
            "scientific_gate_adjudicated"
        )
        and e2_diag_metrics.get("gpu_wall_time_seconds") is None
        and not e2_diag_metrics.get("official_test_inspected")
        and not e2_diag_metrics.get("selector_path_accessed")
        and e2_diag_metrics.get("launch_code_sha256")
        == e2_diag_hashes["launch_code_sha256"]
        and e2_diag_metrics.get("operational_landing_code_sha256")
        == e2_diag_hashes["operational_landing_code_sha256"],
        (
            f"entry_count={len(e2_diag_entries)}, "
            f"route={e2_diag_metrics.get('final_route_token')}, "
            f"artifact_sha={e2_diag_metrics.get('artifact_sha256')}"
        ),
    )

    # 32. Bind the one owner-authorized operational repair to the registered
    #     Stage-0 STOP and its immutable artifact.
    e2_diag_repair_path = RESULTS / "exp_e2_diag_stage0_instrumented.json"
    e2_diag_repair_sha = canonical_lf_sha256(e2_diag_repair_path)
    repair_stage0 = e2_diag_repair["stages"]["stage_0"]
    repair_curve = repair_stage0["training_curve"]
    repair_access = e2_diag_repair["access_confirmation"]
    check(
        "e2_diag_operational_repair_stage0_stop_binding",
        e2_diag_repair_sha
        == "0ad4fc5fafe343b37944b462972d32ff201749280a7bdbb506fa3b62536620d7"
        and e2_diag_repair["schema_version"] == "1.0.0"
        and e2_diag_repair["experiment_id"] == "exp_e2_diag"
        and e2_diag_repair["operational_repair_of"]
        == "experiments/06_uesd/results/exp_e2_diag.json"
        and e2_diag_repair["diagnostic_only"]
        and not e2_diag_repair["adjudicates_semantic_ratchet"]
        and e2_diag_repair["suite_status"] == "TERMINAL_STAGE_0_STOP"
        and e2_diag_repair["final_route_token"] == "KILL_FROM_SCRATCH_LINE"
        and repair_stage0["status"] == "STOP"
        and repair_stage0["reason"]
        == "FEWER_THAN_122_OF_128_AFTER_3000_UPDATES"
        and len(repair_curve) == 31
        and [row["update"] for row in repair_curve] == list(range(0, 3001, 100))
        and all(
            row["training_accuracy"]
            == {"numerator": 32, "denominator": 128, "rate": 0.25}
            for row in repair_curve
        )
        and repair_stage0["final_evaluation"]["cross_entropy"]["mean"]
        == 1.388671875
        and not any(repair_access.values()),
        (
            f"sha={e2_diag_repair_sha}, status={repair_stage0['status']}, "
            f"route={e2_diag_repair['final_route_token']}, access={repair_access}"
        ),
    )

    # 33. Bind the diagnostic bug test: every tensor in every requested group
    #     had a nonzero gradient at all 30 samples, while all updates clipped.
    repair_instrumentation = repair_stage0["instrumentation"]
    gradient_rows = repair_instrumentation["gradient_flow_every_100_updates"]
    gradient_groups = (
        "encoder",
        "controller",
        "plan_slots",
        "prefix_projector",
        "answer_decoder",
        "readout_head",
    )
    expected_flips = [
        122, 128, 0, 128, 0, 0, 0, 128, 128, 128,
        0, 128, 128, 128, 128, 0, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 0, 0, 0, 128,
    ]
    check(
        "e2_diag_gradient_flow_and_prediction_flip_binding",
        len(gradient_rows) == 30
        and [row["update"] for row in gradient_rows] == list(range(100, 3001, 100))
        and all(
            sample["pre_clip_gradient_norms"][group]["l2_norm"] > 0.0
            and sample["pre_clip_gradient_norms"][group]["tensors_with_gradient"]
            == sample["pre_clip_gradient_norms"][group]["parameter_tensor_count"]
            and sample["pre_clip_gradient_norms"][group]["nonzero_gradient_tensors"]
            == sample["pre_clip_gradient_norms"][group]["parameter_tensor_count"]
            for sample in gradient_rows
            for group in gradient_groups
        )
        and [
            row["prediction_flip_count_from_previous_checkpoint"]
            for row in repair_curve[1:]
        ]
        == expected_flips
        and sum(expected_flips) == 2682
        and all(
            list(row["prediction_counts"].values()) == [128]
            for row in repair_curve[1:]
        )
        and repair_stage0["final_evaluation"]
        ["readout_head_weight_delta_l2_from_initial"]
        == 1.9251918633863532
        and repair_stage0["training"]["clipped_updates"]
        == {"numerator": 3000, "denominator": 3000, "rate": 1.0},
        (
            f"gradient_samples={len(gradient_rows)}, "
            f"flip_total={sum(expected_flips)}, "
            f"clipped={repair_stage0['training']['clipped_updates']}"
        ),
    )

    # 34. Bind the repaired serialization, frozen identities, and exact loss
    #     movement. The minimum remains above ln(4), so this is chance-floor
    #     relaxation rather than learned tiny-set discrimination.
    repair_hashes = e2_diag_repair["hashes"]
    loss_decrease = repair_instrumentation["loss_decrease"]
    check(
        "e2_diag_repair_integrity_hash_and_loss_binding",
        e2_diag_live_code_sha == repair_hashes["code_sha256"]
        and e2_diag_live_config_sha == repair_hashes["e2_config_sha256"]
        and repair_hashes["stage_0_ordered_set_sha256"]
        == e2_diag_hashes["stage_0_ordered_set_sha256"]
        and repair_hashes["model_initialization_sha256"]
        == e2_diag_hashes["reconstructed_model_initialization_sha256"]
        and e2_diag_repair["model"]["seed"] == 42
        and e2_diag_repair["model"]["trainable_parameter_count"] == 29509636
        and e2_diag_repair["integrity"]["pass"]
        and loss_decrease
        == {
            "initial_cross_entropy_mean": 1.522216796875,
            "minimum_cross_entropy_mean": 1.38671875,
            "minimum_at_update": 2700,
            "absolute_decrease_from_initial": 0.135498046875,
            "any_decrease_from_initial": True,
            "final_cross_entropy_mean": 1.388671875,
        }
        and loss_decrease["minimum_cross_entropy_mean"] > 1.3862943611198906,
        (
            f"live_code={e2_diag_live_code_sha}, "
            f"repair_code={repair_hashes['code_sha256']}, loss={loss_decrease}"
        ),
    )

    # 35. Bind the append-only chronology to the repaired artifact and the
    #     registered line-kill route without extending it into a model fix.
    repair_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "uesd-e2-diag-stage0-instrumented-repair"
    ]
    repair_entry = repair_entries[0] if len(repair_entries) == 1 else {}
    repair_metrics = repair_entry.get("metrics", {})
    check(
        "e2_diag_repair_ledger_binding",
        len(repair_entries) == 1
        and repair_entry.get("status") == "stopped"
        and repair_entry.get("artifacts")
        == [
            "experiments/06_uesd/results/exp_e2_diag_stage0_instrumented.json"
        ]
        and repair_metrics.get("artifact_sha256") == e2_diag_repair_sha
        and repair_metrics.get("stage_status") == "STOP"
        and repair_metrics.get("final_route_token") == "KILL_FROM_SCRATCH_LINE"
        and repair_metrics.get("training_accuracy")
        == {"numerator": 32, "denominator": 128, "rate": 0.25}
        and repair_metrics.get("prediction_flip_total") == 2682
        and repair_metrics.get("clipped_updates")
        == {"numerator": 3000, "denominator": 3000, "rate": 1.0}
        and not repair_metrics.get("wiring_bug_found")
        and not repair_metrics.get("official_test_inspected"),
        (
            f"entry_count={len(repair_entries)}, "
            f"route={repair_metrics.get('final_route_token')}, "
            f"artifact_sha={repair_metrics.get('artifact_sha256')}"
        ),
    )

    # 36. Bind the immutable E3 preflight to the registered probe mapping and
    #     the single controlling competence-smoke denominator.
    e3_path = RESULTS / "exp_e3_preflight.json"
    e3_sha = canonical_lf_sha256(e3_path)
    probe = e3_preflight["instrument_probe"]
    norms = probe["raw_global_preclip_gradient_norms"]
    sorted_norms = sorted(norms)
    probe_median = (sorted_norms[24] + sorted_norms[25]) / 2
    mapped_clip = min(16, max(2, int(probe_median + 0.5)))
    smoke = e3_preflight["competence_smoke"]
    check(
        "e3_preflight_probe_and_smoke_terminal_binding",
        e3_sha
        == "51b7e565fe13967948017c88b7df2ea4738cb35fdc2dba92a402673698130c9a"
        and e3_preflight["final_token"] == "PREFLIGHT_STOP"
        and e3_preflight["decision"]["reason"]
        == "PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS"
        and e3_preflight["decision"]["route"]
        == "DRAFT_REGISTER_INTERFACE_SUPERVISION_DIAGNOSTIC_ONLY"
        and not e3_preflight["decision"]["canonical_launch_authorized_now"]
        and probe["outcome"] == "PASS"
        and probe["completed_updates"] == 50
        and len(norms) == 50
        and all(value > 0 for value in norms)
        and probe["ordinary_median"] == probe_median
        and probe["quantiles"]["median"] == probe_median
        and mapped_clip == 14
        and probe["derived_controller_gradient_clip_norm"] == mapped_clip
        and not probe["forbidden_metrics_computed"]
        and not probe["accuracy_computed"]
        and smoke["outcome"] == "PREFLIGHT_STOP"
        and smoke["completed_optimizer_updates"] == 500
        and smoke["validation_correct"] == 128
        and smoke["validation_denominator"] == 512
        and smoke["validation_percentage"] == 25.0
        and smoke["pass_correct_minimum"] == 205
        and smoke["intermediate_validation_inspections"] == 0
        and len(smoke["training_curve"]) == 10
        and smoke["training_curve"][-1]["completed_update"] == 500
        and e3_preflight["compute"]["total_completed_optimizer_updates"] == 550
        and e3_preflight["compute"]["canonical_retained_updates"] == 0
        and e3_preflight["state_isolation"]["all_throwaway_state_discarded"]
        and e3_preflight["cohort_isolation"][
            "all_forbidden_access_counts_zero"
        ],
        (
            f"artifact_sha={e3_sha}, token={e3_preflight['final_token']}, "
            f"probe_n={len(norms)}, median={probe_median}, clip={mapped_clip}, "
            f"smoke={smoke['validation_correct']}/{smoke['validation_denominator']}"
        ),
    )

    # 37. Bind the reviewed runner/config bytes and append-only chronology to
    #     the immutable E3 stop without promoting it to mechanics evidence.
    e3_runner = Path(__file__).resolve().parent / "exp_e3_pretrained_latch_mechanics.py"
    e3_config = (
        Path(__file__).resolve().parent
        / "exp_e3_pretrained_latch_mechanics_config.json"
    )
    e3_entries = [
        entry for entry in ledger_entries if entry.get("id") == "uesd-e3-preflight"
    ]
    e3_entry = e3_entries[0] if len(e3_entries) == 1 else {}
    e3_metrics = e3_entry.get("metrics", {})
    check(
        "e3_preflight_live_hash_and_ledger_binding",
        canonical_lf_sha256(e3_runner) == e3_preflight["hashes"]["runner_sha256"]
        and canonical_lf_sha256(e3_config)
        == e3_preflight["hashes"]["config_sha256"]
        and len(e3_entries) == 1
        and e3_entry.get("status") == "stopped"
        and e3_entry.get("artifacts")
        == ["experiments/06_uesd/results/exp_e3_preflight.json"]
        and e3_metrics.get("final_token") == "PREFLIGHT_STOP"
        and e3_metrics.get("reason")
        == "PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS"
        and e3_metrics.get("hashes", {}).get("artifact_sha256") == e3_sha
        and e3_metrics.get("instrument_probe", {}).get("derived_clip_norm") == 14
        and e3_metrics.get("competence_smoke", {}).get("validation_correct")
        == {"numerator": 128, "denominator": 512, "rate": 0.25}
        and e3_metrics.get("official_test_rows_inspected") == 0
        and e3_metrics.get("line_07_rows_accessed") == 0
        and not e3_metrics.get("canonical_launch_authorized"),
        (
            f"entry_count={len(e3_entries)}, status={e3_entry.get('status')}, "
            f"runner={canonical_lf_sha256(e3_runner)}, "
            f"config={canonical_lf_sha256(e3_config)}"
        ),
    )

    # 38. Bind the fresh diagnostic as preregistered, hash-bound, not run, and
    #     still launch-blocked at the separate full-pipeline review boundary.
    diagnostic_registration = (
        Path(__file__).resolve().parent
        / "E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTRATION.md"
    )
    diagnostic_config_path = (
        Path(__file__).resolve().parent
        / "exp_e3_interface_supervision_diagnostic_config.json"
    )
    diagnostic_result_path = RESULTS / "exp_e3_interface_supervision_diagnostic.json"
    diagnostic_config = json.loads(
        diagnostic_config_path.read_text(encoding="utf-8")
    )
    diagnostic_registration_text = diagnostic_registration.read_text(
        encoding="utf-8"
    )
    diagnostic_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id") == "E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTERED"
    ]
    diagnostic_entry = (
        diagnostic_entries[0] if len(diagnostic_entries) == 1 else {}
    )
    check(
        "e3_interface_supervision_diagnostic_preregistered_not_run",
        len(diagnostic_entries) == 1
        and diagnostic_entry.get("status") == "designed"
        and diagnostic_entry.get("metrics", {}).get("diagnostic_cells_executed")
        == 0
        and not diagnostic_entry.get("metrics", {}).get("launch_authorized")
        and diagnostic_config.get("experiment_id")
        == "exp_e3_interface_supervision_diagnostic"
        and diagnostic_config.get("status")
        == (
            "PREREGISTERED_NOT_RUN_IMPLEMENTED_HASH_BOUND_"
            "LAUNCH_BLOCKED_ON_INDEPENDENT_REVIEW"
        )
        and diagnostic_config["compute"]["cell_order"]
        == [
            "I0_ONE_HOP_LINEAR_FLOOR",
            "I1_HARD_ANSWER_LINEAR",
            "I2_FACT_TRACE_LINEAR",
            "S0_ONE_HOP_CONTROLLER_FLOOR",
            "S1_HARD_ANSWER_ONLY",
            "S2_HARD_PROCESS_DENSE",
        ]
        and diagnostic_config["compute"]["per_cell_wall_cap_seconds"] == 900
        and diagnostic_config["compute"]["suite_wall_cap_seconds"] == 5400
        and "PENDING" not in json.dumps(diagnostic_config["bindings"])
        and "LAUNCH BLOCKED ON SEPARATE FULL-PIPELINE REVIEW"
        in diagnostic_registration_text
        and not diagnostic_result_path.exists(),
        (
            f"entry_count={len(diagnostic_entries)}, "
            f"result_exists={diagnostic_result_path.exists()}, "
            f"status={diagnostic_config.get('status')}"
        ),
    )

    diagnostic_runner_path = (
        Path(__file__).resolve().parent
        / "exp_e3_interface_supervision_diagnostic.py"
    )
    implementation_entries = [
        entry
        for entry in ledger_entries
        if entry.get("id")
        == "E3_INTERFACE_SUPERVISION_DIAGNOSTIC_IMPLEMENTED_HASH_BOUND"
    ]
    implementation_entry = (
        implementation_entries[0] if len(implementation_entries) == 1 else {}
    )
    implementation_metrics = implementation_entry.get("metrics", {})
    check(
        "e3_interface_supervision_implementation_hash_binding",
        len(implementation_entries) == 1
        and implementation_entry.get("status") == "designed"
        and implementation_metrics.get("runner_sha256")
        == canonical_lf_sha256(diagnostic_runner_path)
        and implementation_metrics.get("config_sha256")
        == canonical_lf_sha256(diagnostic_config_path)
        and implementation_metrics.get("registration_protocol_sha256")
        == diagnostic_config["bindings"]["registration_protocol_sha256"]
        and implementation_metrics.get("split_sha256")
        == diagnostic_config["bindings"]["split_sha256"]
        and implementation_metrics.get("fast_self_test_pass")
        and implementation_metrics.get("full_binding_replay_pass")
        and implementation_metrics.get("diagnostic_cells_executed") == 0
        and implementation_metrics.get("scientific_metrics_computed") == 0
        and not implementation_metrics.get("result_artifact_exists")
        and implementation_metrics.get("line_07_examples_accessed") == 0
        and implementation_metrics.get("official_test_examples_accessed") == 0
        and not implementation_metrics.get("launch_authorized"),
        (
            f"entry_count={len(implementation_entries)}, "
            f"runner={canonical_lf_sha256(diagnostic_runner_path)}, "
            f"config={canonical_lf_sha256(diagnostic_config_path)}"
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
