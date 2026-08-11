"""Read-only audit for the immutable Line-07 Option-A compute-cap stop."""

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
EXPECTED_ARTIFACT_SHA256 = (
    "8f7d3be695f5503bd5a0a87a09651f52ab38a0adb4a04483d9a4be842d8da0c2"
)
EXPECTED_RUNNER_SHA256 = (
    "49af8237e6de5373a109eb47e71ff0b39efe805750e3fa2bd0c8f15da51b3b26"
)
EXPECTED_REVIEW_SHA256 = (
    "f2a099ea60d44bb9809fa1e2cfa614c4066c242ba7580c55bdad176a6f59bffc"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
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
        result["attestation"]["runner_source_sha256"] == EXPECTED_RUNNER_SHA256,
    )
    check(
        "review_binding",
        result["attestation"]["delta_review_sha256"] == EXPECTED_REVIEW_SHA256,
    )
    runner = ROOT / "experiments" / "07_safe_selection" / "exp_f1_bon_safe_selection.py"
    check("live_runner_sha256", sha256_file(runner) == EXPECTED_RUNNER_SHA256)

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
        ROOT / "STATUS.md": "PREFLIGHT_STOP_COMPUTE_CAP",
        ROOT / "experiments" / "EXPERIMENTS.md": "PREFLIGHT_STOP_COMPUTE_CAP",
        ROOT / "experiments" / "07_safe_selection" / "PREREGISTRATION.md": (
            "exp_bon_safe_selection_cap_preflight.json"
        ),
    }
    for path, needle in expected_texts.items():
        check(f"document_binding_{path.name}", needle in path.read_text(encoding="utf-8"))

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

    failed = [name for name, passed in checks if not passed]
    payload = {"passed": len(checks) - len(failed), "total": len(checks), "failed": failed}
    print(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
