"""Read-only audit for immutable Line-07 Option-A engineering stops."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "experiments"
    / "07_safe_selection"
    / "results"
    / "exp_bon_safe_selection_cap_preflight.json"
)
PERFORMANCE_FORK_ARTIFACT = (
    ROOT
    / "experiments"
    / "07_safe_selection"
    / "results"
    / "exp_bon_safe_selection_performance_fork.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "8f7d3be695f5503bd5a0a87a09651f52ab38a0adb4a04483d9a4be842d8da0c2"
)
EXPECTED_CAP_RUNNER_SHA256 = (
    "49af8237e6de5373a109eb47e71ff0b39efe805750e3fa2bd0c8f15da51b3b26"
)
EXPECTED_REVIEW_SHA256 = (
    "f2a099ea60d44bb9809fa1e2cfa614c4066c242ba7580c55bdad176a6f59bffc"
)
EXPECTED_PERFORMANCE_FORK_ARTIFACT_SHA256 = (
    "e88194f92a07e85c53cc775f3aba37a108beb7f00dd231f780f0fdb7f352ba8d"
)
EXPECTED_PERFORMANCE_FORK_RUNNER_SHA256 = (
    "07d23d8cc15de79e6e33b9780d2a1e3ee9bc0c0d8a88db86c84211dad9667a98"
)
EXPECTED_PERFORMANCE_FORK_REVIEW_SHA256 = (
    "6e23e774524684908a6a57053cf87204b691761a9c81824b721848727eab1bc5"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    performance_fork = json.loads(
        PERFORMANCE_FORK_ARTIFACT.read_text(encoding="utf-8")
    )
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))

    check(
        "artifact_sha256",
        sha256_canonical_lf(ARTIFACT) == EXPECTED_ARTIFACT_SHA256,
    )
    check("stop_token", result["stop_token"] == "PREFLIGHT_STOP_COMPUTE_CAP")
    check("stop_stage", result["stop_stage"] == "probe")
    check("no_scientific_outcome", result["scientific_outcome"] is False)
    check("scientific_verdict_na", result["scientific_verdict"] == "N/A")
    check("claim_not_authorized", result["claim_authorized"] is False)
    check(
        "runner_binding",
        result["attestation"]["runner_source_sha256"]
        == EXPECTED_CAP_RUNNER_SHA256,
    )
    check(
        "review_binding",
        result["attestation"]["delta_review_sha256"] == EXPECTED_REVIEW_SHA256,
    )
    runner = ROOT / "experiments" / "07_safe_selection" / "exp_f1_bon_safe_selection.py"
    check(
        "live_runner_sha256",
        sha256_file(runner) == EXPECTED_PERFORMANCE_FORK_RUNNER_SHA256,
    )

    check(
        "performance_fork_artifact_sha256",
        sha256_canonical_lf(PERFORMANCE_FORK_ARTIFACT)
        == EXPECTED_PERFORMANCE_FORK_ARTIFACT_SHA256,
    )
    check(
        "performance_fork_interrupted_close",
        performance_fork["status"] == "CLOSED_INTERRUPTED_BRANCH_1_GOVERNS",
    )
    check(
        "performance_fork_reason",
        performance_fork["reason"] == "INTERRUPTED_AFTER_DURABLE_START",
    )
    check(
        "performance_fork_runner_binding",
        performance_fork["started"]["runner_source_sha256"]
        == EXPECTED_PERFORMANCE_FORK_RUNNER_SHA256,
    )
    check(
        "performance_fork_review_binding",
        performance_fork["started"]["prelaunch_review_sha256"]
        == EXPECTED_PERFORMANCE_FORK_REVIEW_SHA256,
    )
    check(
        "performance_fork_no_local_banks",
        performance_fork["local_bank_artifacts_present"]
        == {"batch_8": False, "batch_16": False},
    )
    check(
        "performance_fork_no_retained_data",
        performance_fork["retained_responses_or_scores"] == 0
        and performance_fork["calibration_or_test_outcomes_accessed"] is False,
    )
    check(
        "performance_fork_no_successor_authorization",
        performance_fork["authorization"]
        == {
            "fresh_cap_compliant_successor_registration_authorized": False,
            "owner_exception_necessary": True,
        },
    )

    protocol = result["protocol"]
    check("forward_sequence_length", len(protocol["forward_only_sequence"]) == 6)
    check("last_stage_probe", protocol["last_reached_stage"] == "probe")
    check("stage_ledger_absent", protocol["stage_ledger_created"] is False)
    check("cap_resolution_absent", protocol["cap_resolution_created"] is False)
    check("strict_cap", protocol["authorizing_ceiling_seconds"] == 8100.0)
    check("minimum_problem_count", protocol["minimum_canonical_problem_count"] == 512)
    check("no_selected_batch", protocol["selected_batch_size"] == "N/A")
    check("no_resolved_calibration", protocol["resolved_calibration_count"] == 0)
    check("no_resolved_test", protocol["resolved_test_count"] == 0)
    check(
        "retained_gpu_ledger_absent",
        protocol["retained_gpu_time_ledger_created"] is False,
    )

    for label, expected_hash, expected_maximum in (
        (
            "batch_8",
            "0f184c0c9876938dedc50a146d36804afb9c71e9f88274b641e21879702ca66c",
            486,
        ),
        (
            "batch_16",
            "840fe2dbc6bec1c6793eeda93b73dcfa28f01cdbd693fa259927257132fa78c0",
            171,
        ),
    ):
        probe = result["probe_results"][label]
        check(f"{label}_source_hash", probe["source_artifact_sha256"] == expected_hash)
        check(f"{label}_status_fail", probe["status"] == "FAIL")
        check(f"{label}_technical_pass", probe["technical_eligibility"] is True)
        check(f"{label}_cap_ineligible", probe["cap_eligible"] is False)
        check(f"{label}_vram_pass", probe["vram_pass"] is True)
        check(f"{label}_stopping_pass", probe["per_row_stopping_pass"] is True)
        check(f"{label}_no_cuda_error", probe["nan_or_cuda_error"] is False)
        check(f"{label}_no_process_leak", probe["process_leak"] is False)
        check(
            f"{label}_no_checkpoint_inconsistency",
            probe["checkpoint_inconsistency"] is False,
        )
        check(
            f"{label}_exact_duplicate_match",
            probe["exact_match_candidates"] == {"numerator": 32, "denominator": 32},
        )
        check(f"{label}_two_executions", len(probe["executions"]) == 2)
        projection = probe["projection"]
        check(
            f"{label}_over_cap",
            projection["full_bank_projection_seconds"]
            > protocol["authorizing_ceiling_seconds"],
        )
        check(
            f"{label}_maximum_bound",
            projection["maximum_problem_count_under_ceiling"] == expected_maximum,
        )
        check(
            f"{label}_below_minimum",
            projection["maximum_problem_count_under_ceiling"]
            < protocol["minimum_canonical_problem_count"],
        )
        check(
            f"{label}_launch_not_authorized",
            projection["canonical_launch_size_authorized"] is False,
        )

    access = result["access_and_denominators"]
    for field in (
        "retained_responses_generated",
        "successor_candidates_scored",
        "calibration_gold_answers_used_by_probe",
        "calibration_correctness_or_outcome_fields_computed",
        "test_content_exposed_to_probe",
        "test_gold_answers_used_by_probe",
        "test_correctness_or_outcome_fields_computed",
        "calibration_viability_metrics_computed",
        "outcome_blind_rescore_records",
        "policy_evaluations",
        "bootstrap_replicates",
        "permutations",
    ):
        check(f"zero_{field}", access[field] == 0)
    check(
        "throwaway_denominator",
        access["throwaway_generated_candidate_executions"]
        == {"numerator": 128, "denominator": 128},
    )
    check(
        "no_throwaway_retention",
        access["diagnostic_duplicates_entered_retained_bank"]
        == {"numerator": 0, "denominator": 128},
    )
    check(
        "all_unreached_outputs_na",
        all(value == "N/A" for value in result["unreached_registered_outputs"].values()),
    )
    check("no_deviations", result["deviations"] == [])

    expected_texts = {
        ROOT / "STATUS.md": (
            "PREFLIGHT_STOP_COMPUTE_CAP",
            "CLOSED_INTERRUPTED_BRANCH_1_GOVERNS",
        ),
        ROOT / "experiments" / "EXPERIMENTS.md": (
            "PREFLIGHT_STOP_COMPUTE_CAP",
            "CLOSED_INTERRUPTED_BRANCH_1_GOVERNS",
        ),
        ROOT / "experiments" / "07_safe_selection" / "PREREGISTRATION.md": (
            "exp_bon_safe_selection_cap_preflight.json",
            "exp_bon_safe_selection_performance_fork.json",
        ),
    }
    for path, needles in expected_texts.items():
        text = path.read_text(encoding="utf-8")
        for ordinal, needle in enumerate(needles, start=1):
            check(f"document_binding_{path.name}_{ordinal}", needle in text)

    ledger_entries = []
    for line in (ROOT / "experiments" / "ledger.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        entry = json.loads(line)
        if entry.get("id") == "safe-selection-option-a-cap-preflight-stop":
            ledger_entries.append(entry)
    check("one_ledger_entry", len(ledger_entries) == 1)
    if ledger_entries:
        metrics = ledger_entries[0]["metrics"]
        check("ledger_stop_token", metrics["stop_token"] == result["stop_token"])
        check(
            "ledger_artifact_hash",
            metrics["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256,
        )

    performance_ledger_entries = []
    for line in (ROOT / "experiments" / "ledger.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        entry = json.loads(line)
        if entry.get("id") == "safe-selection-stopping-performance-fork-interrupted":
            performance_ledger_entries.append(entry)
    check("one_performance_fork_ledger_entry", len(performance_ledger_entries) == 1)
    if performance_ledger_entries:
        metrics = performance_ledger_entries[0]["metrics"]
        check(
            "performance_ledger_status",
            metrics["status"] == performance_fork["status"],
        )
        check(
            "performance_ledger_artifact_hash",
            metrics["artifact_sha256"]
            == EXPECTED_PERFORMANCE_FORK_ARTIFACT_SHA256,
        )

    failed = [name for name, passed in checks if not passed]
    payload = {"passed": len(checks) - len(failed), "total": len(checks), "failed": failed}
    print(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
