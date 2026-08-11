"""Fresh Option-A line-07 Best-of-N safe-selection successor runner.

This is a provenance runner, not a landing command.  The fast paths are:

* ``--cohort-only``: construct the entirely new successor cohorts and bind
  every dataset/cohort/provenance schedule without loading either model;
* ``--self-test-fast``: exercise schedules, all six policies, calibration,
  statistics, schema checks, resume logic, and immutable writing on synthetic
  data;
* ``--scorer-determinism``: run the frozen operational fixture across three
  clean model loads spanning two fresh Python processes;
* ``--batch-preflight`` / ``--resolve-cap``: measure throwaway diagnostic
  duplicates, then immutably bind a cap-safe cohort prefix before retained data;

Every canonical mode is hash-attested and advances one persisted, forward-only
stage ledger.  Test work additionally requires a complete frozen calibration
bank, the preserved headroom gate, and an exact one-shot outcome-blind rescore.
Completed gate and result evidence is never overwritten.

The qualified E1 numeric parser and four-category taxonomy are imported
directly from ``exp_e1_task_band.py``.  The immutable JSON writer and stale-temp
scavenger faithfully port the hardened E2 same-directory fsync/no-clobber
pattern.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

# The workspace cache contract must be established before hub clients import.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HF_HOME = ROOT / ".hf_cache"
PREREGISTRATION_PATH = HERE / "PREREGISTRATION.md"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from datasets import Dataset, disable_progress_bar, load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import TokenClassifierOutput  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402


SCHEMA_VERSION = "3.0.0"
EXPERIMENT_ID = "exp_f1_bon_safe_selection"
PUBLIC_GENERATOR = "base-C"
PUBLIC_VERIFIER = "verifier-V"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_SPLIT = "train"
EXPECTED_SPLIT_SIZE = 7_473
EXPECTED_SPLIT_FINGERPRINT = "c6f812ae33c9159d"
DEMONSTRATION_INDICES = (0, 1, 2, 3, 4)
CALIBRATION_COUNT = 256
TEST_COUNT = 512
CANDIDATE_COUNT = 16
HISTORICAL_SMOKE_PROBLEM_COUNT = 2
HISTORICAL_SMOKE_CANDIDATE_COUNT = 32
# Legacy helpers remain provenance-only and are unreachable pending review.
RETAINED_SMOKE_PROBLEM_COUNT = HISTORICAL_SMOKE_PROBLEM_COUNT
RETAINED_SMOKE_CANDIDATE_COUNT = HISTORICAL_SMOKE_CANDIDATE_COUNT
ELIGIBLE_BATCH_SIZES = (8, 16)
RESOLVED_BATCH_SIZE = 8
CAP_AUTHORIZING_SECONDS = 8_100.0
MAX_PREFLIGHT_RESERVED_BYTES = 20 * 1024**3
HISTORICAL_SMOKE_TIMING_ARTIFACT_SHA256 = (
    "db64bbf7f44d4135fa7960ce37aa6cfb79abe177ad5aa16ed92327490e327112"
)
HISTORICAL_SMOKE_GENERATION_LOAD_SECONDS = 3.162245300001814
HISTORICAL_SMOKE_GENERATION_WALL_SECONDS = 110.15521369999624
HISTORICAL_SMOKE_SCORING_LOAD_SECONDS = 8.827072100000805
HISTORICAL_SMOKE_SCORING_WALL_SECONDS = 2.666824199986877
SCORER_CONTRACT = "verifier-V / maintained-eager-BF16-v1"
SCORER_INTERNAL_REFERENCE = (1.0, 0.1923828125, 0.98046875, 1.0)
SCORER_INTERNAL_REFERENCE_BF16_BITS = (16256, 15941, 16251, 16256)
SCORER_FIXTURE_INPUT_SHA256 = (
    "e14ec22c375a3dbc31596964e53fe1a59b4f7264b4c73f5dcd80a1bbb3f52741"
)
SCORER_DETERMINISM_STOP = "PREFLIGHT_STOP_SCORER_NONDETERMINISM"
SCORER_RESCORE_VOID = "VOID_SCORER_RESCORE_MISMATCH"
PREFIXES = (1, 2, 4, 8, 16)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 0
REPETITION_PENALTY = 1.0
ORIGINAL_CALIBRATION_SELECTION_STRING = (
    "bon-safe-selection-base-c-gsm8k-train-calibration-v1-2026-08-10"
)
ORIGINAL_TEST_SELECTION_STRING = (
    "bon-safe-selection-base-c-gsm8k-train-test-v1-2026-08-10"
)
CALIBRATION_SELECTION_STRING = (
    "bon-safe-selection-successor-a-base-c-gsm8k-train-calibration-v1-2026-08-11"
)
TEST_SELECTION_STRING = (
    "bon-safe-selection-successor-a-base-c-gsm8k-train-test-v1-2026-08-11"
)
GENERATION_SEED_STRING = (
    "bon-safe-selection-successor-a-base-c-generation-seeds-v1-2026-08-11"
)
BATCH_SEED_STRING = "bon-safe-selection-successor-a-batched-generation-v1-2026-08-11"
BOOTSTRAP_STRING = "bon-safe-selection-successor-a-paired-bootstrap-v1-2026-08-11"
PERMUTATION_STRING = "bon-safe-selection-successor-a-order-permutations-v1-2026-08-11"
RESCORE_SELECTION_STRING = (
    "bon-safe-selection-successor-a-outcome-blind-rescore-v1-2026-08-11"
)
RESCORE_RECORD_COUNT = 64
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 1_000
DELTA_GRID = (0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
LAMBDA_GRID = (0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100)
BUDGET_GRID = (1, 2, 4, 8, 16)
FINAL_TOKENS = {"CONFIRM", "KILL", "VOID"}
NOT_APPLICABLE = "N/A"
CLAIM_CONTRACT_SECTION = "Claim-language contract"
STAGE_ORDER = ("probe", "cap_resolution", "calibration", "viability", "rescore", "test")
STAGE_STATES = {
    "probe": ("PASS",),
    "cap_resolution": ("PASS",),
    "calibration": ("STARTED", "PASS"),
    "viability": ("PASS", "STOP"),
    "rescore": ("STARTED", "PASS", "VOID"),
    "test": ("STARTED", "SCORED", "PASS"),
}


class StageTransitionError(RuntimeError):
    """A persisted stage was skipped, repeated, or advanced out of order."""


class VoidIntegrityError(RuntimeError):
    """A post-retention registered integrity failure that requires VOID landing."""

    def __init__(self, reason: str, stage: str):
        super().__init__(reason)
        self.reason = reason
        self.stage = stage

EXPECTED_HASHES = {
    "demonstration_indices": "6484c68c0c85987f9beb3db42175c46955a8abe05170239580fcd1ff8b514452",
    "demonstrations_content": "b310a622f3e0ef1ed3694cf37cc8da38362e48f194d68adc6df34155e702025e",
    "base_a_sorted_indices": "0cc16e5a27c42ed8cab155006c25883a2695cac4131150b7e1d61334e912192b",
    "base_a_content": "1034771b7a0d0b547819e185ebddd7a981d6c000a81aefa1c956341df7c98674",
    "base_b_ordered_indices": "670705ea2936f75f0e90a4048d3f5b5ec3a63b42577c0d7a9df87253b77444ff",
    "base_b_content": "37febcb5eab403fa8f479a3adc76124ddeb78183e5305f944810eb64c25cea86",
    "svamp_demo_content": "f7fe5da1bf49cbb4125949999f89d1ca83773e3c7a3cbb8cc203c0ccc8c686dd",
    "svamp_base_a_ordered_indices": "69af24379818a75c67c3444639acd0e2d7d17091175a6720bb111b73357dd684",
    "svamp_base_a_content": "f674ce1024e890731192f5be4c55655d4c73abc83b395bfd0f3c216b25bbfba0",
    "prior_questions": "c4a3a04eeb926bd89474c1938ea816fa681e66f8cf03a5bdc3e1411b9e77bc62",
    "calibration_pool": "6f3b8138db046b3f0c78dd13896864bea624c063ad48706491adebdd0f8b51c4",
    "calibration_ordered_indices": "9236163527dac17431ac43c9d094874ce644a211219a77de6783917ba15705c7",
    "calibration_sorted_indices": "c424773f8b1246e1030dac9b3b92fd40f6f5f27e49186e03a54e0e58cc356046",
    "calibration_content": "3919f0dc142cc9c61e70da902fb1233c49c07a07ce542e3e3f62e46263ec0376",
    "test_pool": "37b837cd3ddb055ebf59ac83fc0c024a21f9aaae1ebce409545e01268707fc91",
    "test_ordered_indices": "f151f1c46c2421ed78aef43f1cd6e7bd99fc7c8ab5bb945e36adff346bad9c48",
    "test_sorted_indices": "8aa38ad17399eb30526c05174408322b53a05a874ef74805d5b73d18a787b6bf",
    "test_content": "0efd8a2b9c8f7341f3e2677f8da89dd76ad65ae29cab05ae33751b1810f1afac",
    "calibration_then_test": "727debb0c35837935e4686ce6ba30051f5826bca8eef2963b7acbae47953fa2f",
    "unallocated_indices": "aa28e47af683f7fe65dd2cc78eaf29030c55d7de9127fe4995af844f1d31b44c",
    "unallocated_content": "0c78ada0a2b578c25e4361fc592bcf4a956a401fcb0921d5769b02e578f42e98",
    "generation_schedule": "8d3d567ba90da7ec7cdc0eb9c5764b545834975c0d477f0fc030888c9f2f03fd",
    "permutation_schedule": "354b34427aa1ff857b6c281f57cd07a7b24a0ebfd4771f2645f1b3dfa1dcca14",
}

LOCAL_MANIFEST = HERE / "_local_manifest.md"
E1_RUNNER = HERE.parent / "06_uesd" / "exp_e1_task_band.py"
RESULT_PATH = HERE / "results" / "exp_bon_safe_selection.json"
WORK_ROOT = HF_HOME / "line07_safe_selection_successor_a"
CALIBRATION_BANK_PATH = WORK_ROOT / "calibration_bank.json"
TEST_BANK_PATH = WORK_ROOT / "test_bank.json"
CALIBRATION_FREEZE_PATH = WORK_ROOT / "calibration_freeze.json"
BATCH_PREFLIGHT_PATH = WORK_ROOT / "batch_preflight.json"
CAP_RESOLUTION_PATH = WORK_ROOT / "cap_resolution.json"
SCORER_DETERMINISM_PATH = WORK_ROOT / "scorer_determinism.json"
OUTCOME_BLIND_RESCORE_PATH = WORK_ROOT / "outcome_blind_rescore.json"
RESCORE_TERMINAL_FAILURE_PATH = WORK_ROOT / "rescore_terminal_failure.json"
REVIEW_BINDING_PATH = WORK_ROOT / "independent_review_binding.json"
STAGE_LEDGER_DIR = WORK_ROOT / "stage_ledger"
GPU_TIME_LEDGER_PATH = WORK_ROOT / "retained_gpu_time_ledger.jsonl"
IMMUTABLE_TEMP_PREFIX = ".f1-immutable-result-tmp-"
IMMUTABLE_TEMP_SUFFIX = ".tmp"
IMMUTABLE_TEMP_GLOB = f"{IMMUTABLE_TEMP_PREFIX}*{IMMUTABLE_TEMP_SUFFIX}"


def _load_qualified_e1_module():
    module_name = "_line07_qualified_e1_parser"
    spec = importlib.util.spec_from_file_location(module_name, E1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the qualified E1 parser module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


E1 = _load_qualified_e1_module()
PROMPT_PREAMBLE = E1.PROMPT_PREAMBLE


def canonical_json_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comma_hash(indices: Iterable[int]) -> str:
    return sha256_text(",".join(str(index) for index in indices))


def assert_hash(name: str, actual: str) -> str:
    expected = EXPECTED_HASHES[name]
    if actual != expected:
        raise RuntimeError(f"{name} hash changed: expected {expected}, got {actual}")
    return actual


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_runtime() -> None:
    disable_progress_bar()
    transformers_logging.set_verbosity_error()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def normalized_question(question: str) -> str:
    return " ".join(question.casefold().split())


def gsm_row_payload(split: Dataset, indices: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "answer": str(split[index]["answer"]),
            "index": int(index),
            "question": str(split[index]["question"]),
        }
        for index in indices
    ]


def gsm_rows_hash(split: Dataset, indices: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(gsm_row_payload(split, indices)))


def svamp_row_payload(split: Dataset, indices: Sequence[int]) -> list[dict[str, Any]]:
    fields = ("ID", "Body", "Question", "Equation", "Answer", "Type", "question_concat")
    return [
        {"index": int(index), **{field: split[index][field] for field in fields}}
        for index in indices
    ]


def svamp_rows_hash(split: Dataset, indices: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(svamp_row_payload(split, indices)))


def ranked_indices(pool: Sequence[int], selection_string: str, count: int) -> list[int]:
    def key(index: int) -> tuple[bytes, int]:
        payload = f"{DATASET_REVISION}\n{selection_string}\n{index}".encode("ascii")
        return hashlib.sha256(payload).digest(), index

    return sorted(pool, key=key)[:count]


def load_artifact_indices(filename: str, *, ordered: bool = True) -> list[int]:
    path = HERE.parent / "06_uesd" / "results" / filename
    artifact = json.loads(path.read_text(encoding="utf-8"))
    indices = [int(row["dataset_index"]) for row in artifact["per_example"]]
    if not ordered:
        indices.sort()
    if len(indices) != 256 or len(set(indices)) != 256:
        raise RuntimeError(f"immutable prior artifact is malformed: {filename}")
    return indices


def construct_cohorts(
    *,
    expose_pilot_test_content: bool = True,
    resolved_calibration_count: int = CALIBRATION_COUNT,
    resolved_test_count: int = TEST_COUNT,
) -> tuple[Dataset, dict[str, list[int]], dict[str, Any]]:
    if not 0 < resolved_calibration_count <= CALIBRATION_COUNT:
        raise RuntimeError("resolved calibration count is outside the frozen prefix")
    if not 0 < resolved_test_count <= TEST_COUNT:
        raise RuntimeError("resolved test count is outside the frozen prefix")
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        cache_dir=str(HF_HOME / "datasets"),
    )
    train = dataset[DATASET_SPLIT]
    if len(train) != EXPECTED_SPLIT_SIZE:
        raise RuntimeError("revision-pinned GSM8K train split size changed")
    if train._fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            f"revision-pinned GSM8K train fingerprint changed: {train._fingerprint}"
        )

    demonstration_hash = assert_hash(
        "demonstrations_content", gsm_rows_hash(train, DEMONSTRATION_INDICES)
    )
    assert_hash("demonstration_indices", comma_hash(DEMONSTRATION_INDICES))
    original_pool = list(range(5, EXPECTED_SPLIT_SIZE))
    assert len(original_pool) == 7_468
    assert_hash("calibration_pool", comma_hash(original_pool))
    original_calibration = ranked_indices(
        original_pool,
        ORIGINAL_CALIBRATION_SELECTION_STRING,
        CALIBRATION_COUNT,
    )
    assert_hash("calibration_ordered_indices", comma_hash(original_calibration))
    assert_hash("calibration_sorted_indices", comma_hash(sorted(original_calibration)))
    assert_hash("calibration_content", gsm_rows_hash(train, original_calibration))
    original_calibration_set = set(original_calibration)
    original_test_pool = [
        index for index in original_pool if index not in original_calibration_set
    ]
    assert len(original_test_pool) == 7_212
    assert_hash("test_pool", comma_hash(sorted(original_test_pool)))
    original_test = ranked_indices(
        original_test_pool, ORIGINAL_TEST_SELECTION_STRING, TEST_COUNT
    )
    assert_hash("test_ordered_indices", comma_hash(original_test))
    assert_hash("test_sorted_indices", comma_hash(sorted(original_test)))
    assert_hash("test_content", gsm_rows_hash(train, original_test))
    assert_hash(
        "calibration_then_test",
        comma_hash([*original_calibration, *original_test]),
    )
    original_selected = original_calibration_set | set(original_test)
    successor_source_pool = [
        index for index in original_pool if index not in original_selected
    ]
    assert len(successor_source_pool) == 6_700
    assert_hash("unallocated_indices", comma_hash(successor_source_pool))
    assert_hash("unallocated_content", gsm_rows_hash(train, successor_source_pool))

    full_calibration = ranked_indices(
        successor_source_pool, CALIBRATION_SELECTION_STRING, CALIBRATION_COUNT
    )
    full_calibration_set = set(full_calibration)
    test_pool = [
        index for index in successor_source_pool if index not in full_calibration_set
    ]
    if len(test_pool) != 6_444:
        raise RuntimeError("successor test pool has the wrong size")
    full_test = ranked_indices(test_pool, TEST_SELECTION_STRING, TEST_COUNT)
    calibration = full_calibration[:resolved_calibration_count]
    test = full_test[:resolved_test_count]
    calibration_set = set(calibration)
    selected_set = calibration_set | set(test)
    unallocated = [
        index for index in successor_source_pool if index not in selected_set
    ]
    expected_unallocated = len(successor_source_pool) - len(calibration) - len(test)
    if len(unallocated) != expected_unallocated:
        raise RuntimeError("successor unallocated pool has the wrong size")

    calibration_hash = gsm_rows_hash(train, calibration)
    test_hash = (
        gsm_rows_hash(train, test)
        if expose_pilot_test_content
        else registered_slot_value("successor_test_row_content_sha256")
    )
    if test_hash is None:
        raise RuntimeError("successor test content slot is not registered")
    unallocated_hash = gsm_rows_hash(train, unallocated)

    if len(set(calibration)) != resolved_calibration_count:
        raise RuntimeError("calibration cohort is not unique")
    if len(set(test)) != resolved_test_count:
        raise RuntimeError("test cohort is not unique")
    if calibration_set & set(test):
        raise RuntimeError("calibration/test cohort overlap")
    if selected_set & set(DEMONSTRATION_INDICES):
        raise RuntimeError("selected cohort overlaps demonstrations")
    if selected_set & original_selected:
        raise RuntimeError("successor cohort overlaps the original 768 rows")

    registry = validate_prior_consumption_registry(dataset)
    prior_questions = registry.pop("normalized_questions")
    original_questions = {
        normalized_question(str(train[i]["question"])) for i in original_selected
    }
    calibration_questions = [
        normalized_question(train[i]["question"]) for i in calibration
    ]
    if len(set(calibration_questions)) != resolved_calibration_count:
        raise RuntimeError("calibration normalized questions are not unique")
    if set(calibration_questions) & prior_questions:
        raise RuntimeError("calibration questions overlap prior consumption")
    if set(calibration_questions) & original_questions:
        raise RuntimeError("calibration questions overlap original line-07 rows")
    if expose_pilot_test_content:
        test_questions = [normalized_question(train[i]["question"]) for i in test]
        if len(set(test_questions)) != resolved_test_count:
            raise RuntimeError("test normalized questions are not unique")
        if set(calibration_questions) & set(test_questions):
            raise RuntimeError("calibration/test normalized-question overlap")
        if set(test_questions) & prior_questions:
            raise RuntimeError("test questions overlap prior consumption")
        if set(test_questions) & original_questions:
            raise RuntimeError("test questions overlap original line-07 rows")

    cohorts = {"calibration": calibration, "test": test, "unallocated": unallocated}
    evidence = {
        "status": "PASS",
        "dataset": {
            "public_id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "size": len(train),
            "fingerprint": train._fingerprint,
        },
        "demonstrations_content_sha256": demonstration_hash,
        "calibration_selected_row_content_sha256": calibration_hash,
        "test_selected_row_content_sha256": test_hash,
        "remaining_unallocated_row_content_sha256": unallocated_hash,
        "prior_consumption": registry,
        "counts": {
            "calibration": len(calibration),
            "test": len(test),
            "source_pool": len(successor_source_pool),
            "original_rows_excluded": len(original_selected),
            "unallocated": len(unallocated),
            "full_frozen_calibration": len(full_calibration),
            "full_frozen_test": len(full_test),
            "cohorts_are_frozen_ordered_prefixes": True,
            "calibration_test_overlap": 0,
            "selected_demonstration_overlap": 0,
            "calibration_prior_question_overlap": 0,
            "test_prior_question_overlap": (
                0 if expose_pilot_test_content else "registered_not_reaccessed"
            ),
        },
        "resolved_ordered_indices": {
            "calibration": [int(index) for index in calibration],
            "test": [int(index) for index in test],
        },
        "pilot_test_content_revalidated_in_this_process": expose_pilot_test_content,
        "hashes": {
            "successor_source_pool_sorted_indices_sha256": comma_hash(
                successor_source_pool
            ),
            "successor_original_excluded_ordered_indices_sha256": comma_hash(
                [*original_calibration, *original_test]
            ),
            "successor_calibration_ordered_indices_sha256": comma_hash(calibration),
            "successor_calibration_sorted_indices_sha256": comma_hash(
                sorted(calibration)
            ),
            "successor_calibration_row_content_sha256": calibration_hash,
            "successor_test_pool_sorted_indices_sha256": comma_hash(sorted(test_pool)),
            "successor_test_ordered_indices_sha256": comma_hash(test),
            "successor_test_sorted_indices_sha256": comma_hash(sorted(test)),
            "successor_test_row_content_sha256": test_hash,
            "successor_calibration_then_test_ordered_indices_sha256": comma_hash(
                [*calibration, *test]
            ),
            "successor_remaining_unallocated_sorted_indices_sha256": comma_hash(
                sorted(unallocated)
            ),
            "successor_remaining_unallocated_row_content_sha256": (unallocated_hash),
        },
    }
    return train, cohorts, evidence


def validate_prior_consumption_registry(
    gsm_dataset: Mapping[str, Dataset],
) -> dict[str, Any]:
    gsm_train = gsm_dataset["train"]
    gsm_test = gsm_dataset["test"]
    base_a_ordered = load_artifact_indices("exp_e1_task_band.json", ordered=True)
    base_a = sorted(base_a_ordered)
    base_b = load_artifact_indices("exp_e1_task_band_base_b.json", ordered=True)
    assert_hash("base_a_sorted_indices", comma_hash(base_a))
    assert_hash("base_b_ordered_indices", comma_hash(base_b))
    assert_hash("base_a_content", gsm_rows_hash(gsm_test, base_a_ordered))
    assert_hash("base_b_content", gsm_rows_hash(gsm_test, base_b))

    svamp_revision = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
    svamp = load_dataset(
        "ChilleD/SVAMP",
        "default",
        revision=svamp_revision,
        cache_dir=str(HF_HOME / "datasets"),
    )
    svamp_indices = load_artifact_indices(
        "exp_e1_task_band_svamp_initial_parser_miss.json", ordered=True
    )
    assert_hash("svamp_base_a_ordered_indices", comma_hash(svamp_indices))
    assert_hash(
        "svamp_demo_content", svamp_rows_hash(svamp["train"], DEMONSTRATION_INDICES)
    )
    assert_hash("svamp_base_a_content", svamp_rows_hash(svamp["test"], svamp_indices))

    questions = {
        *(
            normalized_question(gsm_train[index]["question"])
            for index in DEMONSTRATION_INDICES
        ),
        *(normalized_question(gsm_test[index]["question"]) for index in base_a),
        *(normalized_question(gsm_test[index]["question"]) for index in base_b),
        *(
            normalized_question(
                " ".join(
                    part
                    for part in (
                        str(svamp["train"][index]["Body"]).strip(),
                        str(svamp["train"][index]["Question"]).strip(),
                    )
                    if part
                )
            )
            for index in DEMONSTRATION_INDICES
        ),
        *(
            normalized_question(
                " ".join(
                    part
                    for part in (
                        str(svamp["test"][index]["Body"]).strip(),
                        str(svamp["test"][index]["Question"]).strip(),
                    )
                    if part
                )
            )
            for index in svamp_indices
        ),
    }
    if len(questions) != 778:
        raise RuntimeError(f"prior normalized-question count changed: {len(questions)}")
    prior_hash = sha256_text("\n".join(sorted(questions)))
    assert_hash("prior_questions", prior_hash)
    return {
        "status": "PASS",
        "unique_normalized_question_count": len(questions),
        "normalized_question_sha256": prior_hash,
        "normalized_questions": questions,
    }


def generation_seed(dataset_index: int, candidate_ordinal: int) -> int:
    payload = (
        f"{DATASET_REVISION}\n{GENERATION_SEED_STRING}\n"
        f"{dataset_index}\n{candidate_ordinal}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def batch_payload(
    partition: str,
    dataset_index: int,
    first_candidate_ordinal: int,
    batch_size: int,
) -> str:
    if partition not in {"calibration", "test"}:
        raise ValueError(f"invalid partition for batch seed: {partition}")
    if batch_size not in ELIGIBLE_BATCH_SIZES:
        raise ValueError(f"invalid amended batch size: {batch_size}")
    if first_candidate_ordinal not in range(1, CANDIDATE_COUNT + 1, batch_size):
        raise ValueError("batch does not start on a frozen candidate boundary")
    ordinals = range(first_candidate_ordinal, first_candidate_ordinal + batch_size)
    if max(ordinals) > CANDIDATE_COUNT:
        raise ValueError("batch crosses a problem boundary")
    identity_seeds = [generation_seed(dataset_index, ordinal) for ordinal in ordinals]
    return "\n".join(
        (
            BATCH_SEED_STRING,
            partition,
            str(dataset_index),
            str(first_candidate_ordinal),
            str(batch_size),
            ",".join(str(seed) for seed in identity_seeds),
        )
    )


def batch_seed(
    partition: str,
    dataset_index: int,
    first_candidate_ordinal: int,
    batch_size: int,
) -> tuple[int, str]:
    payload = batch_payload(
        partition, dataset_index, first_candidate_ordinal, batch_size
    )
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1), digest.hex()


def generation_schedule(cohorts: Mapping[str, Sequence[int]]) -> list[int]:
    return [
        generation_seed(index, ordinal)
        for partition in ("calibration", "test")
        for index in cohorts[partition]
        for ordinal in range(1, CANDIDATE_COUNT + 1)
    ]


def mixed_generation_schedule(
    cohorts: Mapping[str, Sequence[int]], batch_size: int = RESOLVED_BATCH_SIZE
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for partition in ("calibration", "test"):
        for dataset_index in cohorts[partition]:
            for first_ordinal in range(1, CANDIDATE_COUNT + 1, batch_size):
                seed, payload_sha256 = batch_seed(
                    partition, dataset_index, first_ordinal, batch_size
                )
                schedule.append(
                    {
                        "partition": partition,
                        "dataset_index": dataset_index,
                        "first_candidate_ordinal": first_ordinal,
                        "batch_size": batch_size,
                        "candidate_identity_seeds": [
                            generation_seed(dataset_index, ordinal)
                            for ordinal in range(
                                first_ordinal, first_ordinal + batch_size
                            )
                        ],
                        "batch_seed": seed,
                        "batch_payload_sha256": payload_sha256,
                    }
                )
    return schedule


def outcome_blind_rescore_identities(
    calibration_indices: Sequence[int], count: int = RESCORE_RECORD_COUNT
) -> list[tuple[int, int]]:
    identities = [
        (dataset_index, ordinal)
        for dataset_index in calibration_indices
        for ordinal in range(1, CANDIDATE_COUNT + 1)
    ]

    def key(identity: tuple[int, int]) -> tuple[bytes, int, int]:
        dataset_index, ordinal = identity
        payload = (
            f"{DATASET_REVISION}\n{RESCORE_SELECTION_STRING}\n"
            f"{dataset_index}\n{ordinal}"
        ).encode("ascii")
        return hashlib.sha256(payload).digest(), dataset_index, ordinal

    if count > len(identities):
        raise ValueError("outcome-blind rescore count exceeds calibration bank")
    return sorted(identities, key=key)[:count]


def permutation_schedule(count: int = PERMUTATION_REPLICATES) -> list[list[int]]:
    result = []
    for permutation_index in range(count):

        def key(ordinal: int) -> tuple[bytes, int]:
            payload = f"{PERMUTATION_STRING}\n{permutation_index}\n{ordinal}".encode(
                "ascii"
            )
            return hashlib.sha256(payload).digest(), ordinal

        result.append(sorted(range(1, CANDIDATE_COUNT + 1), key=key))
    return result


def validate_schedules(
    cohorts: Mapping[str, Sequence[int]],
    *,
    batch_size: int = RESOLVED_BATCH_SIZE,
    enforce_registered_full_bank: bool = True,
) -> dict[str, Any]:
    seeds = generation_schedule(cohorts)
    seed_hash = comma_hash(seeds)
    expected_seed_count = (
        len(cohorts["calibration"]) + len(cohorts["test"])
    ) * CANDIDATE_COUNT
    if len(seeds) != expected_seed_count:
        raise RuntimeError("generation schedule denominator drift")
    mixed = mixed_generation_schedule(cohorts, batch_size=batch_size)
    mixed_hash = sha256_bytes(canonical_json_bytes(mixed))
    batch_seeds = [int(row["batch_seed"]) for row in mixed if row["batch_seed"]]
    batch_seed_hash = comma_hash(batch_seeds)
    permutations = permutation_schedule()
    permutation_hash = sha256_text(
        "\n".join(",".join(str(value) for value in row) for row in permutations)
    )
    bootstrap_hash = sha256_text(
        "\n".join(
            ",".join(
                str(value)
                for value in bootstrap_positions(replicate, len(cohorts["test"]))
            )
            for replicate in range(BOOTSTRAP_REPLICATES)
        )
    )
    rescore_identities = outcome_blind_rescore_identities(cohorts["calibration"])
    rescore_hash = sha256_text(
        "\n".join(f"{index},{ordinal}" for index, ordinal in rescore_identities)
    )
    bindings = {
        "successor_generation_seed_schedule_sha256": seed_hash,
        "successor_batch_seed_schedule_sha256": batch_seed_hash,
        "successor_bootstrap_schedule_sha256": bootstrap_hash,
        "successor_permutation_schedule_sha256": permutation_hash,
        "successor_outcome_blind_rescore_schedule_sha256": rescore_hash,
    }
    if enforce_registered_full_bank:
        if len(cohorts["calibration"]) != CALIBRATION_COUNT or len(
            cohorts["test"]
        ) != TEST_COUNT:
            raise RuntimeError("only the full bank may use pre-cap registered schedules")
        for name, actual in bindings.items():
            registered = registered_slot_value(name)
            if registered is not None and registered != actual:
                raise RuntimeError(
                    f"registered successor schedule drift for {name}: "
                    f"expected {registered}, got {actual}"
                )
    return {
        "generation_seed_count": len(seeds),
        "generation_schedule_sha256": seed_hash,
        "mixed_generation_call_count": len(mixed),
        "mixed_generation_schedule_sha256": mixed_hash,
        "batch_seed_count": len(batch_seeds),
        "batch_seed_schedule_sha256": batch_seed_hash,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_schedule_sha256": bootstrap_hash,
        "resolved_calibration_count": len(cohorts["calibration"]),
        "resolved_test_count": len(cohorts["test"]),
        "resolved_batch_size": batch_size,
        "permutation_count": len(permutations),
        "permutation_schedule_sha256": permutation_hash,
        "outcome_blind_rescore_count": len(rescore_identities),
        "outcome_blind_rescore_schedule_sha256": rescore_hash,
        "bindings": bindings,
    }


def build_five_shot_messages(train: Dataset, question: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": PROMPT_PREAMBLE}]
    for index in DEMONSTRATION_INDICES:
        row = train[index]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Question: {str(row['question']).strip()}",
                },
                {
                    "role": "assistant",
                    "content": f"Answer:\n{str(row['answer']).strip()}",
                },
            ]
        )
    messages.append(
        {"role": "user", "content": f"Question: {question.strip()}\nAnswer:"}
    )
    return messages


def parse_manifest() -> dict[str, str]:
    if not LOCAL_MANIFEST.is_file():
        raise RuntimeError("the gitignored line-07 local manifest is missing")
    pattern = re.compile(r"^- `([^`]+)`: `([^`]*)`$")
    entries: dict[str, str] = {}
    for line in LOCAL_MANIFEST.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            if match.group(1) in entries:
                raise RuntimeError(f"duplicate manifest key: {match.group(1)}")
            entries[match.group(1)] = match.group(2)
    required = {
        f"{public}-{suffix}"
        for public in (PUBLIC_GENERATOR, PUBLIC_VERIFIER)
        for suffix in (
            "repo-id",
            "revision",
            "tokenizer-revision",
            "resolved-path",
            "license",
            "documentation-url",
            "dtype",
            "selection-provenance",
            "local-content-sha256",
            "files-map-sha256",
        )
    }
    required.update(
        {
            "python-version",
            "torch-version",
            "transformers-version",
            "datasets-version",
            "tokenizers-version",
            "huggingface-hub-version",
            "safetensors-version",
            "cuda-version",
            "cuda-driver-version",
            "gpu-identity",
        }
    )
    missing = sorted(required - set(entries))
    if missing:
        raise RuntimeError(f"local manifest is missing keys: {missing}")
    return entries


def snapshot_content_digest(
    snapshot_path: Path,
) -> tuple[str, dict[str, dict[str, Any]]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot_path).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not files:
        raise RuntimeError("model snapshot is empty")
    digest = hashlib.sha256()
    for relative, evidence in files.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(evidence["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(evidence["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def manifest_identity(entries: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = dict(sorted(entries.items()))
    for public in (PUBLIC_GENERATOR, PUBLIC_VERIFIER):
        path = Path(entries[f"{public}-resolved-path"])
        if not path.is_dir():
            raise RuntimeError(f"{public} resolved snapshot is absent")
        digest, files = snapshot_content_digest(path)
        if digest != entries[f"{public}-local-content-sha256"]:
            raise RuntimeError(f"{public} local snapshot digest changed")
        files_map_hash = sha256_bytes(canonical_json_bytes(files))
        if files_map_hash != entries[f"{public}-files-map-sha256"]:
            raise RuntimeError(f"{public} file-map digest changed")
    return sha256_bytes(canonical_json_bytes(payload)), payload


def prompt_serialization_hash(
    train: Dataset,
    cohorts: Mapping[str, Sequence[int]],
    manifest: Mapping[str, str],
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        manifest[f"{PUBLIC_GENERATOR}-repo-id"],
        revision=manifest[f"{PUBLIC_GENERATOR}-tokenizer-revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError("base-C tokenizer has no native chat template")
    serialized = []
    for partition in ("calibration", "test"):
        for position, index in enumerate(cohorts[partition]):
            prompt = tokenizer.apply_chat_template(
                build_five_shot_messages(train, str(train[index]["question"])),
                tokenize=False,
                add_generation_prompt=True,
            )
            serialized.append(
                {
                    "partition": partition,
                    "cohort_position": position,
                    "dataset_index": index,
                    "prompt": prompt,
                }
            )
    return sha256_bytes(canonical_json_bytes(serialized))


def registered_slot_value(name: str) -> str | None:
    """Return the last pre-data amendment value for a 64-hex slot."""
    pattern = re.compile(rf"^- `{re.escape(name)}`: `([0-9a-f]{{64}})`$", re.MULTILINE)
    values = pattern.findall(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if not values:
        return None
    return values[-1]


def provenance_slots(
    train: Dataset,
    cohorts: Mapping[str, Sequence[int]],
    *,
    require_manifest: bool,
    allow_pilot_test_access: bool,
    validate_bindings: bool = True,
) -> dict[str, str | None]:
    slots: dict[str, str | None] = {
        "dataset_train_split_fingerprint": train._fingerprint,
        "normalized_prior_consumed_questions_sha256": EXPECTED_HASHES[
            "prior_questions"
        ],
        "demonstrations_content_sha256": EXPECTED_HASHES["demonstrations_content"],
        "successor_source_pool_sorted_indices_sha256": EXPECTED_HASHES[
            "unallocated_indices"
        ],
        "successor_original_excluded_ordered_indices_sha256": EXPECTED_HASHES[
            "calibration_then_test"
        ],
        "successor_calibration_ordered_indices_sha256": comma_hash(
            cohorts["calibration"]
        ),
        "successor_calibration_sorted_indices_sha256": comma_hash(
            sorted(cohorts["calibration"])
        ),
        "successor_calibration_row_content_sha256": gsm_rows_hash(
            train, cohorts["calibration"]
        ),
        "successor_test_pool_sorted_indices_sha256": registered_slot_value(
            "successor_test_pool_sorted_indices_sha256"
        ),
        "successor_test_ordered_indices_sha256": comma_hash(cohorts["test"]),
        "successor_test_sorted_indices_sha256": comma_hash(sorted(cohorts["test"])),
        "successor_test_row_content_sha256": None,
        "successor_calibration_then_test_ordered_indices_sha256": comma_hash(
            [*cohorts["calibration"], *cohorts["test"]]
        ),
        "successor_remaining_unallocated_sorted_indices_sha256": comma_hash(
            sorted(cohorts["unallocated"])
        ),
        "successor_remaining_unallocated_row_content_sha256": gsm_rows_hash(
            train, cohorts["unallocated"]
        ),
        "successor_prompt_serialization_sha256": None,
        "parser_source_sha256": sha256_text(E1.parser_source_text()),
        "successor_runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "successor_manifest_identity_digest": None,
        "successor_verifier_identity_bundle_sha256": None,
        "successor_tokenizer_identity_bundle_sha256": None,
        "successor_maintained_scorer_source_sha256": (
            maintained_scorer_source_sha256()
        ),
        "successor_checkpoint_tensor_identity_sha256": None,
    }
    if LOCAL_MANIFEST.is_file():
        entries = parse_manifest()
        identity_digest, _ = manifest_identity(entries)
        slots["successor_manifest_identity_digest"] = identity_digest
        slots.update(scorer_identity_bundles(entries))
        if allow_pilot_test_access:
            slots["successor_test_row_content_sha256"] = gsm_rows_hash(
                train, cohorts["test"]
            )
            slots["successor_prompt_serialization_sha256"] = prompt_serialization_hash(
                train, cohorts, entries
            )
        else:
            cap_slots = None
            if CAP_RESOLUTION_PATH.is_file():
                cap_report = json.loads(CAP_RESOLUTION_PATH.read_text(encoding="utf-8"))
                cap_slots = cap_report.get("resolved_provenance_slots")
            for name in (
                "successor_test_row_content_sha256",
                "successor_prompt_serialization_sha256",
            ):
                slots[name] = (
                    cap_slots.get(name)
                    if isinstance(cap_slots, Mapping)
                    else registered_slot_value(name)
                )
        if SCORER_DETERMINISM_PATH.is_file():
            determinism = json.loads(
                SCORER_DETERMINISM_PATH.read_text(encoding="utf-8")
            )
            if determinism.get("status") != "PASS":
                raise RuntimeError("scorer determinism artifact is not PASS")
            slots["successor_checkpoint_tensor_identity_sha256"] = determinism["loads"][
                0
            ]["loading_report"]["successor_checkpoint_tensor_identity_sha256"]
        else:
            slots["successor_checkpoint_tensor_identity_sha256"] = (
                registered_slot_value("successor_checkpoint_tensor_identity_sha256")
            )
    elif require_manifest:
        raise RuntimeError("manifest-bound provenance slots cannot be filled")

    if not validate_bindings:
        return slots
    full_bank = (
        len(cohorts["calibration"]) == CALIBRATION_COUNT
        and len(cohorts["test"]) == TEST_COUNT
    )
    resolved_binding = None
    if not full_bank:
        resolved_binding = json.loads(CAP_RESOLUTION_PATH.read_text(encoding="utf-8"))
        if resolved_binding.get("status") != "PASS":
            raise RuntimeError("resized provenance requires a passing cap resolution")
    cohort_dependent = {
        "successor_calibration_ordered_indices_sha256",
        "successor_calibration_sorted_indices_sha256",
        "successor_calibration_row_content_sha256",
        "successor_test_ordered_indices_sha256",
        "successor_test_sorted_indices_sha256",
        "successor_test_row_content_sha256",
        "successor_calibration_then_test_ordered_indices_sha256",
        "successor_remaining_unallocated_sorted_indices_sha256",
        "successor_remaining_unallocated_row_content_sha256",
        "successor_prompt_serialization_sha256",
    }
    for name, actual in slots.items():
        if not name.startswith("successor_"):
            continue
        if not full_bank and name in cohort_dependent:
            expected = resolved_binding["resolved_provenance_slots"].get(name)
            if expected != actual:
                raise RuntimeError(
                    f"cap-resolved provenance drift for {name}: expected {expected}, got {actual}"
                )
            continue
        registered = registered_slot_value(name)
        if registered is not None and actual != registered:
            raise RuntimeError(
                f"registered provenance drift for {name}: "
                f"expected {registered}, got {actual}"
            )
    return slots


def split_reasoning_steps(response: str) -> list[str]:
    normalized = response.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [
        step.strip() for step in re.split(r"\n[ \t]*\n+", normalized) if step.strip()
    ]


def verifier_messages(
    question: str, response: str
) -> tuple[list[dict[str, str]], list[str]]:
    steps = split_reasoning_steps(response)
    assistant = "<extra_0>".join(steps) + ("<extra_0>" if steps else "")
    messages = [
        {"role": "system", "content": PROMPT_PREAMBLE},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant},
    ]
    return messages, steps


def verifier_serialization(
    question: str, response: str, tokenizer
) -> tuple[str, list[str]]:
    messages, steps = verifier_messages(question, response)
    return (
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ),
        steps,
    )


@dataclass
class PowerSampler:
    samples_watts: list[float]
    _stop: threading.Event
    _thread: threading.Thread | None = None

    @classmethod
    def create(cls) -> "PowerSampler":
        return cls([], threading.Event())

    def __enter__(self) -> "PowerSampler":
        def sample() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                while not self._stop.wait(0.25):
                    self.samples_watts.append(
                        float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    )
                pynvml.nvmlShutdown()
            except Exception:
                return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self, active_seconds: float) -> dict[str, Any]:
        average = statistics.fmean(self.samples_watts) if self.samples_watts else None
        return {
            "sample_count": len(self.samples_watts),
            "average_active_watts": average,
            "peak_watts": max(self.samples_watts) if self.samples_watts else None,
            "watt_hours": average * active_seconds / 3600
            if average is not None
            else None,
        }


def cuda_telemetry() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def load_generator(manifest: Mapping[str, str], device: torch.device):
    identifier = manifest[f"{PUBLIC_GENERATOR}-repo-id"]
    revision = manifest[f"{PUBLIC_GENERATOR}-revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        revision=manifest[f"{PUBLIC_GENERATOR}-tokenizer-revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError("base-C tokenizer has no native chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        identifier,
        revision=revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    return tokenizer, model


class MaintainedEagerProcessRewardModel(PreTrainedModel):
    """Maintained pseudonymous backbone plus the checkpoint reward head."""

    config_class = PretrainedConfig
    base_model_prefix = "model"

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        config.use_cache = False
        self.model = AutoModel.from_config(
            config,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2),
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
        **kwargs: Any,
    ) -> TokenClassifierOutput | tuple[Any, ...]:
        if use_cache not in (None, False):
            raise RuntimeError("maintained-eager scorer forbids use_cache=True")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        logits = self.score(outputs.last_hidden_state)
        if return_dict is False:
            return (logits, outputs.hidden_states, outputs.attentions)
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def scorer_identity_bundles(manifest: Mapping[str, str]) -> dict[str, str]:
    snapshot = Path(manifest[f"{PUBLIC_VERIFIER}-resolved-path"])
    verifier_payload = {
        "revision": manifest[f"{PUBLIC_VERIFIER}-revision"],
        "local_content_sha256": manifest[f"{PUBLIC_VERIFIER}-local-content-sha256"],
        "files_map_sha256": manifest[f"{PUBLIC_VERIFIER}-files-map-sha256"],
    }
    tokenizer_payload = {
        "revision": manifest[f"{PUBLIC_VERIFIER}-tokenizer-revision"],
        "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(snapshot / "tokenizer_config.json"),
    }
    return {
        "successor_verifier_identity_bundle_sha256": sha256_bytes(
            canonical_json_bytes(verifier_payload)
        ),
        "successor_tokenizer_identity_bundle_sha256": sha256_bytes(
            canonical_json_bytes(tokenizer_payload)
        ),
    }


def maintained_scorer_source_sha256() -> str:
    source_text = "\n\n".join(
        inspect.getsource(item)
        for item in (
            MaintainedEagerProcessRewardModel,
            scorer_identity_bundles,
            load_verifier,
            operational_scorer_output,
        )
    )
    return sha256_text(source_text)


def load_verifier(manifest: Mapping[str, str], device: torch.device):
    snapshot_path = manifest[f"{PUBLIC_VERIFIER}-resolved-path"]
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    config.use_cache = False
    model, loading_info = MaintainedEagerProcessRewardModel.from_pretrained(
        snapshot_path,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    unexpected_keys = list(loading_info.get("unexpected_keys", []))
    omitted_checkpoint_keys = ["lm_head.weight"]
    blocking_loading_fields = {
        "missing_keys": list(loading_info.get("missing_keys", [])),
        "unexpected_keys": sorted(set(unexpected_keys) - set(omitted_checkpoint_keys)),
        "mismatched_keys": list(loading_info.get("mismatched_keys", [])),
        "error_msgs": list(loading_info.get("error_msgs", [])),
    }
    if sorted(unexpected_keys) != omitted_checkpoint_keys:
        blocking_loading_fields["unexpected_keys"] = unexpected_keys
    if any(blocking_loading_fields.values()):
        raise RuntimeError(
            f"maintained scorer checkpoint load mismatch: {blocking_loading_fields}"
        )
    model = model.to(device)
    model.eval()
    state = model.state_dict()
    floating_dtypes = sorted(
        {str(tensor.dtype) for tensor in state.values() if tensor.is_floating_point()}
    )
    if floating_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"maintained scorer dtype drift: {floating_dtypes}")
    state_key_hash = sha256_text("\n".join(sorted(state)))
    tensor_identity_payload = {
        "checkpoint_files_map_sha256": manifest[f"{PUBLIC_VERIFIER}-files-map-sha256"],
        "state_tensor_count": len(state),
        "state_key_sha256": state_key_hash,
        "floating_dtypes": floating_dtypes,
        "missing_keys": [],
        "unexpected_keys": unexpected_keys,
        "omitted_checkpoint_keys": omitted_checkpoint_keys,
        "mismatched_keys": [],
        "operational_language_model_head_present": any(
            key.startswith("lm_head.") for key in state
        ),
    }
    model._line07_loading_report = {  # type: ignore[attr-defined]
        **tensor_identity_payload,
        "successor_checkpoint_tensor_identity_sha256": sha256_bytes(
            canonical_json_bytes(tensor_identity_payload)
        ),
    }
    return tokenizer, model


def _stop_metadata(
    generated_ids: torch.Tensor, criteria: Any, tokenizer, row_index: int = 0
) -> tuple[int, int, str]:
    token_ids = [int(token_id) for token_id in generated_ids.tolist()]
    eos = tokenizer.eos_token_id
    eos_ids = (
        set(int(value) for value in eos)
        if isinstance(eos, (list, tuple))
        else {int(eos)}
    )
    boundary_count = criteria.boundary_token_counts[row_index]
    eos_position = next(
        (p for p, token in enumerate(token_ids) if token in eos_ids), None
    )
    eos_count = eos_position + 1 if eos_position is not None else None
    if boundary_count is not None and (
        eos_count is None or boundary_count <= eos_count
    ):
        return boundary_count, boundary_count, "new_question_boundary"
    if eos_count is not None:
        return eos_count, eos_position, "end_of_message"
    count = len(token_ids)
    reason = "max_new_tokens" if count >= MAX_NEW_TOKENS else "generation_stopped_other"
    return count, count, reason


class PerRowNewQuestionBoundaryCriteria(StoppingCriteria):
    """Sticky, row-local boundary stops with auditable mixed-state history."""

    def __init__(self, tokenizer, generation_start: int, batch_size: int) -> None:
        self.tokenizer = tokenizer
        self.generation_start = generation_start
        self.boundary_token_counts: list[int | None] = [None] * batch_size
        self.sticky_stop_checks = [0] * batch_size
        self.mixed_state_call_count = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        stopped = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for row_index in range(input_ids.shape[0]):
            if self.boundary_token_counts[row_index] is not None:
                stopped[row_index] = True
                self.sticky_stop_checks[row_index] += 1
                continue
            generated_ids = input_ids[row_index, self.generation_start :]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            if E1.NEW_QUESTION_RE.search(generated_text):
                self.boundary_token_counts[row_index] = int(generated_ids.numel())
                stopped[row_index] = True
        if bool(stopped.any().item()) and not bool(stopped.all().item()):
            self.mixed_state_call_count += 1
        return stopped


def _per_row_stop_audit(
    generated_rows: torch.Tensor,
    criteria: PerRowNewQuestionBoundaryCriteria,
    tokenizer,
) -> dict[str, Any]:
    allowed_padding = {int(tokenizer.pad_token_id)}
    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        allowed_padding.update(int(value) for value in eos)
    elif eos is not None:
        allowed_padding.add(int(eos))
    completion_counts = []
    suffix_checks = []
    for row_index, generated in enumerate(generated_rows):
        generated_count, _, _ = _stop_metadata(
            generated, criteria, tokenizer, row_index
        )
        completion_counts.append(generated_count)
        suffix = [int(token) for token in generated[generated_count:].tolist()]
        suffix_checks.append(all(token in allowed_padding for token in suffix))
    batch_end = max(completion_counts)
    completed_before_end = sum(count < batch_end for count in completion_counts)
    return {
        "completion_token_counts": completion_counts,
        "rows_completed_before_batch_end": completed_before_end,
        "post_completion_suffix_padding_only": all(suffix_checks),
        "boundary_rows_with_sticky_rechecks": sum(
            count > 0 for count in criteria.sticky_stop_checks
        ),
        "sticky_stop_recheck_count": sum(criteria.sticky_stop_checks),
        "mixed_row_state_call_count": criteria.mixed_state_call_count,
        "pass": bool(all(suffix_checks)),
    }


def generate_batch(
    train: Dataset,
    partition: str,
    dataset_index: int,
    first_ordinal: int,
    batch_size: int,
    tokenizer,
    model,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer.padding_side != "left":
        raise RuntimeError("amended batched generation requires left padding")
    messages = build_five_shot_messages(train, str(train[dataset_index]["question"]))
    serialized_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [serialized_prompt] * batch_size
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.shape[0] != batch_size or not torch.all(
        attention_mask == attention_mask[0]
    ):
        raise RuntimeError("within-problem prompt batch serialization drift")
    generation_start = int(input_ids.shape[1])
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and generation_start + MAX_NEW_TOKENS > context_limit:
        raise RuntimeError("frozen prompt plus response cap exceeds base-C context")
    criteria = PerRowNewQuestionBoundaryCriteria(
        tokenizer=tokenizer,
        generation_start=generation_start,
        batch_size=batch_size,
    )
    seed, payload_sha256 = batch_seed(
        partition, dataset_index, first_ordinal, batch_size
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([criteria]),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if outputs.shape[0] != batch_size:
        raise RuntimeError("batched generation returned the wrong row count")
    generated_rows = outputs[:, generation_start:]
    stop_audit = _per_row_stop_audit(generated_rows, criteria, tokenizer)
    gold = E1.extract_gold_answer(str(train[dataset_index]["answer"]))
    records = []
    for batch_row, generated in enumerate(generated_rows):
        ordinal = first_ordinal + batch_row
        generated_count, content_count, stop_reason = _stop_metadata(
            generated, criteria, tokenizer, batch_row
        )
        response = tokenizer.decode(
            generated[:content_count], skip_special_tokens=True
        ).strip()
        predicted, extraction_source, scored_segment, segment_stop_reason = (
            E1.extract_predicted_answer(
                response=response, generation_stop_reason=stop_reason
            )
        )
        taxonomy = E1.successor_outcome_taxonomy(predicted, gold, scored_segment)
        records.append(
            {
                "dataset_index": dataset_index,
                "candidate_ordinal": ordinal,
                "seed": generation_seed(dataset_index, ordinal),
                "batch_seed": seed,
                "batch_size": batch_size,
                "batch_row": batch_row,
                "batch_payload_sha256": payload_sha256,
                "question": str(train[dataset_index]["question"]),
                "gold_answer": gold,
                "response": response,
                "scored_response_segment": scored_segment,
                "extracted_answer": predicted,
                "extraction_source": extraction_source,
                "extraction_segment_stop_reason": segment_stop_reason,
                "stop_reason": stop_reason,
                "generated_tokens": generated_count,
                "prompt_tokens": int(attention_mask[batch_row].sum().item()),
                "generation_seconds": seconds / batch_size,
                **taxonomy,
                "verifier_score": None,
                "verifier_step_scores": None,
                "verifier_step_score_bf16_bits": None,
                "verifier_score_bf16_bits": None,
                "verifier_scored_tokens": None,
                "verifier_seconds": None,
                "verifier_input_sha256": None,
                "verifier_token_ids_sha256": None,
                "verifier_marker_positions": None,
                "verifier_output_digest": None,
            }
        )
    return records, {"wall_seconds": seconds, "per_row_stopping": stop_audit}


def generate_one(
    train: Dataset,
    dataset_index: int,
    ordinal: int,
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, Any]:
    seed = generation_seed(dataset_index, ordinal)
    messages = build_five_shot_messages(train, str(train[dataset_index]["question"]))
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generation_start = int(input_ids.shape[1])
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and generation_start + MAX_NEW_TOKENS > context_limit:
        raise RuntimeError("frozen prompt plus response cap exceeds base-C context")
    criteria = E1.NewQuestionBoundaryCriteria(
        tokenizer=tokenizer, generation_start=generation_start, batch_size=1
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([criteria]),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    generated = outputs[0, generation_start:]
    generated_count, content_count, stop_reason = _stop_metadata(
        generated, criteria, tokenizer
    )
    response = tokenizer.decode(
        generated[:content_count], skip_special_tokens=True
    ).strip()
    predicted, extraction_source, scored_segment, segment_stop_reason = (
        E1.extract_predicted_answer(
            response=response, generation_stop_reason=stop_reason
        )
    )
    gold = E1.extract_gold_answer(str(train[dataset_index]["answer"]))
    taxonomy = E1.successor_outcome_taxonomy(predicted, gold, scored_segment)
    return {
        "dataset_index": dataset_index,
        "candidate_ordinal": ordinal,
        "seed": seed,
        "question": str(train[dataset_index]["question"]),
        "gold_answer": gold,
        "response": response,
        "scored_response_segment": scored_segment,
        "extracted_answer": predicted,
        "extraction_source": extraction_source,
        "extraction_segment_stop_reason": segment_stop_reason,
        "stop_reason": stop_reason,
        "generated_tokens": generated_count,
        "prompt_tokens": generation_start,
        "generation_seconds": seconds,
        **taxonomy,
        "verifier_score": None,
        "verifier_step_scores": None,
        "verifier_step_score_bf16_bits": None,
        "verifier_score_bf16_bits": None,
        "verifier_scored_tokens": None,
        "verifier_seconds": None,
        "verifier_input_sha256": None,
        "verifier_token_ids_sha256": None,
        "verifier_marker_positions": None,
        "verifier_output_digest": None,
    }


def score_one(
    record: Mapping[str, Any], tokenizer, model, device: torch.device
) -> dict[str, Any]:
    updated = dict(record)
    response = str(record["scored_response_segment"])
    if not response.strip():
        empty_core = {
            "serialized_utf8_sha256": sha256_text(""),
            "token_ids": [],
            "token_ids_sha256": sha256_text(""),
            "marker_positions": [],
            "step_scores": [],
            "step_score_bf16_bits": [],
            "minimum_score": 0.0,
            "minimum_score_bf16_bits": 0,
            "scored_tokens": 0,
        }
        updated.update(
            verifier_score=0.0,
            verifier_step_scores=[],
            verifier_step_score_bf16_bits=[],
            verifier_score_bf16_bits=0,
            verifier_scored_tokens=0,
            verifier_seconds=0.0,
            verifier_input_sha256=sha256_text(""),
            verifier_token_ids_sha256=sha256_text(""),
            verifier_marker_positions=[],
            verifier_output_digest=sha256_bytes(canonical_json_bytes(empty_core)),
        )
        return updated
    serialized, steps = verifier_serialization(
        str(record["question"]), response, tokenizer
    )

    scorer_output = operational_scorer_output(
        serialized, len(steps), tokenizer, model, device
    )
    step_scores = scorer_output["step_scores"]
    updated.update(
        verifier_score=float(min(step_scores)),
        verifier_step_scores=[float(value) for value in step_scores],
        verifier_step_score_bf16_bits=scorer_output["step_score_bf16_bits"],
        verifier_score_bf16_bits=scorer_output["minimum_score_bf16_bits"],
        verifier_scored_tokens=scorer_output["scored_tokens"],
        verifier_seconds=scorer_output["score_wall_seconds"],
        verifier_input_sha256=sha256_text(serialized),
        verifier_token_ids_sha256=scorer_output["token_ids_sha256"],
        verifier_marker_positions=scorer_output["marker_positions"],
        verifier_output_digest=scorer_output["complete_output_digest"],
    )
    return updated


def operational_scorer_output(
    serialized: str,
    expected_step_count: int,
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = tokenizer.encode(serialized, return_tensors="pt").to(device)
    step_sep_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(step_sep_ids) != 1:
        raise RuntimeError("verifier step marker is not one token")
    mask = input_ids == step_sep_ids[0]
    if int(mask.sum().item()) != expected_step_count:
        raise RuntimeError(
            "verifier step-marker count differs from reasoning-step count"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        if logits.dtype != torch.bfloat16:
            raise RuntimeError("maintained scorer logits are not BF16")
        probabilities = torch.softmax(logits, dim=-1)
        if probabilities.dtype != torch.bfloat16:
            raise RuntimeError("maintained scorer softmax is not native BF16")
        selected = probabilities[0][mask[0]]
        if selected.shape[-1] != 2:
            raise RuntimeError("verifier output does not expose two-class step logits")
        positive = selected[:, 1].contiguous()
        step_scores = positive.detach().cpu().tolist()
        step_bits = positive.view(torch.uint16).detach().cpu().tolist()
        minimum = torch.min(positive).contiguous()
        minimum_bits = int(minimum.view(torch.uint16).item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if len(step_scores) != expected_step_count:
        raise RuntimeError("verifier produced the wrong number of step scores")
    if not step_scores or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in step_scores
    ):
        raise RuntimeError(
            "verifier returned missing, nonfinite, or out-of-range scores"
        )
    token_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    marker_positions = [
        position
        for position, token_id in enumerate(token_ids)
        if token_id == step_sep_ids[0]
    ]
    core = {
        "serialized_utf8_sha256": sha256_text(serialized),
        "token_ids": token_ids,
        "token_ids_sha256": comma_hash(token_ids),
        "marker_positions": marker_positions,
        "step_scores": [float(value) for value in step_scores],
        "step_score_bf16_bits": [int(value) for value in step_bits],
        "minimum_score": float(minimum.item()),
        "minimum_score_bf16_bits": minimum_bits,
        "scored_tokens": int(input_ids.numel()),
    }
    return {
        **core,
        "complete_output_digest": sha256_bytes(canonical_json_bytes(core)),
        "score_wall_seconds": seconds,
    }


def verifier_scores_from_serialized(
    serialized: str,
    expected_step_count: int,
    tokenizer,
    model,
    device: torch.device,
) -> tuple[list[float], int, float]:
    output = operational_scorer_output(
        serialized, expected_step_count, tokenizer, model, device
    )
    return output["step_scores"], output["scored_tokens"], output["score_wall_seconds"]


def successor_scorer_fixture_load(
    manifest: Mapping[str, str],
    *,
    load_ordinal: int,
    process_ordinal: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("successor scorer determinism gate requires CUDA")
    tokenizer, model = load_verifier(manifest, device)
    response_steps = [
        (
            "To find out how many more pink plastic flamingos were out than white "
            "plastic flamingos at noon on Sunday, we can break down the problem "
            "into steps. First, on Friday, the neighbors start with 18 pink "
            "plastic flamingos."
        ),
        (
            "On Saturday, they take back one third of the flamingos. Since there "
            "were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, "
            "they have (18 - 6 = 12) flamingos left in their possession. Then, "
            "they paint these 6 flamingos white and put them back out on Sue's "
            "front yard. Now, Sue has the original 12 pink flamingos plus the 6 "
            "new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) "
            "pink flamingos and 6 white flamingos."
        ),
        (
            "On Sunday, the neighbors add another 18 pink plastic flamingos to "
            "Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = "
            "36) pink flamingos and still 6 white flamingos."
        ),
        (
            "To find the difference, subtract the number of white flamingos from "
            "the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on "
            "Sunday, there were 30 more pink plastic flamingos out than white "
            "plastic flamingos. The answer is (\\boxed{30})."
        ),
    ]
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": (
                "Sue lives in a fun neighborhood.  One weekend, the neighbors "
                "decided to play a prank on Sue.  On Friday morning, the neighbors "
                "placed 18 pink plastic flamingos out on Sue's front yard.  On "
                "Saturday morning, the neighbors took back one third of the "
                "flamingos, painted them white, and put these newly painted white "
                "flamingos back out on Sue's front yard.  Then, on Sunday morning, "
                "they added another 18 pink plastic flamingos to the collection. At "
                "noon on Sunday, how many more pink plastic flamingos were out than "
                "white plastic flamingos?"
            ),
        },
        {
            "role": "assistant",
            "content": "<extra_0>".join(response_steps) + "<extra_0>",
        },
    ]
    serialized = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    output = operational_scorer_output(
        serialized, len(response_steps), tokenizer, model, device
    )
    loading_report = dict(model._line07_loading_report)  # type: ignore[attr-defined]
    source_sha256 = maintained_scorer_source_sha256()
    identity_bundles = scorer_identity_bundles(manifest)
    reference_match = (
        tuple(output["step_score_bf16_bits"]) == SCORER_INTERNAL_REFERENCE_BF16_BITS
        and output["serialized_utf8_sha256"] == SCORER_FIXTURE_INPUT_SHA256
        and output["scored_tokens"] == 454
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if reference_match else SCORER_DETERMINISM_STOP,
        "scorer_contract": SCORER_CONTRACT,
        "process_ordinal": process_ordinal,
        "load_ordinal": load_ordinal,
        "internal_reference": list(SCORER_INTERNAL_REFERENCE),
        "internal_reference_bf16_bits": list(SCORER_INTERNAL_REFERENCE_BF16_BITS),
        "output": output,
        "loading_report": loading_report,
        "successor_maintained_scorer_source_sha256": source_sha256,
        **identity_bundles,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if not reference_match:
        raise RuntimeError(SCORER_DETERMINISM_STOP)
    return report


def _determinism_comparable(report: Mapping[str, Any]) -> dict[str, Any]:
    output = report["output"]
    return {
        "serialized_utf8_sha256": output["serialized_utf8_sha256"],
        "token_ids": output["token_ids"],
        "marker_positions": output["marker_positions"],
        "step_score_bf16_bits": output["step_score_bf16_bits"],
        "minimum_score_bf16_bits": output["minimum_score_bf16_bits"],
        "complete_output_digest": output["complete_output_digest"],
        "checkpoint_tensor_identity_sha256": report["loading_report"][
            "successor_checkpoint_tensor_identity_sha256"
        ],
        "maintained_scorer_source_sha256": report[
            "successor_maintained_scorer_source_sha256"
        ],
        "verifier_identity_bundle_sha256": report[
            "successor_verifier_identity_bundle_sha256"
        ],
        "tokenizer_identity_bundle_sha256": report[
            "successor_tokenizer_identity_bundle_sha256"
        ],
    }


def run_scorer_determinism_gate() -> dict[str, Any]:
    if SCORER_DETERMINISM_PATH.exists():
        raise RuntimeError("scorer determinism gate is one-shot and already resolved")
    manifest = parse_manifest()
    loads: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        loads = [
            successor_scorer_fixture_load(
                manifest, load_ordinal=load_ordinal, process_ordinal=1
            )
            for load_ordinal in (1, 2)
        ]
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--scorer-determinism-worker"],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        child_lines = [line for line in child.stdout.splitlines() if line.strip()]
        if not child_lines:
            raise RuntimeError("scorer determinism worker returned no report")
        loads.append(json.loads(child_lines[-1]))
    except Exception as error:  # terminal by preregistration; no retry path
        failure = f"{type(error).__name__}: {error}"
    comparables = [_determinism_comparable(load) for load in loads]
    bit_exact = (
        failure is None
        and len(comparables) == 3
        and all(item == comparables[0] for item in comparables[1:])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "PASS" if bit_exact else SCORER_DETERMINISM_STOP,
        "terminal": not bit_exact,
        "failure": failure,
        "scorer_contract": SCORER_CONTRACT,
        "clean_loads": len(loads),
        "fresh_processes": 2 if len(loads) == 3 else None,
        "bit_exact_across_loads": bit_exact,
        "loads": loads,
        "comparison_digest": (
            sha256_bytes(canonical_json_bytes(comparables[0])) if comparables else None
        ),
        "retained_generation_performed": False,
        "calibration_rows_accessed": 0,
        "test_rows_accessed": 0,
    }
    write_one_shot_gate(SCORER_DETERMINISM_PATH, report)
    if not bit_exact:
        raise RuntimeError(SCORER_DETERMINISM_STOP)
    return report


def bank_template(
    partition: str, indices: Sequence[int], slots: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "partition": partition,
        "canonical_indices_sha256": comma_hash(indices),
        "expected_problem_count": len(indices),
        "expected_candidate_count": len(indices) * CANDIDATE_COUNT,
        "protocol_slots": dict(slots),
        "records": [],
        "telemetry": {},
        "complete_generation": False,
        "complete_scoring": False,
    }


def bank_path(partition: str) -> Path:
    return CALIBRATION_BANK_PATH if partition == "calibration" else TEST_BANK_PATH


def bank_database_path(partition: str) -> Path:
    return WORK_ROOT / f"{partition}_candidates.sqlite3"


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _bank_metadata(bank: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bank.items() if key != "records"}


def initialize_bank_database(partition: str) -> None:
    path = bank_database_path(partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    position INTEGER PRIMARY KEY,
                    dataset_index INTEGER NOT NULL,
                    candidate_ordinal INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(dataset_index, candidate_ordinal)
                )
                """
            )


def load_bank_database_records(partition: str) -> list[dict[str, Any]]:
    path = bank_database_path(partition)
    if not path.is_file():
        return []
    with contextlib.closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT position, record_json FROM candidates ORDER BY position"
        ).fetchall()
    if [int(row[0]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("resumable candidate database has a position gap")
    return [json.loads(str(row[1])) for row in rows]


def persist_candidate_record(
    partition: str,
    position: int,
    record: Mapping[str, Any],
    *,
    generated: bool,
) -> None:
    path = bank_database_path(partition)
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with contextlib.closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("PRAGMA synchronous=FULL")
            if generated:
                connection.execute(
                    """
                    INSERT INTO candidates(
                        position, dataset_index, candidate_ordinal, seed, record_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        position,
                        int(record["dataset_index"]),
                        int(record["candidate_ordinal"]),
                        int(record["seed"]),
                        serialized,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE candidates SET record_json = ?
                    WHERE position = ? AND dataset_index = ?
                      AND candidate_ordinal = ? AND seed = ?
                    """,
                    (
                        serialized,
                        position,
                        int(record["dataset_index"]),
                        int(record["candidate_ordinal"]),
                        int(record["seed"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "scoring update did not match one generated candidate"
                    )


def persist_candidate_batch(
    partition: str,
    start_position: int,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Commit a generated batch atomically so resume never sees a partial call."""
    path = bank_database_path(partition)
    with contextlib.closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("PRAGMA synchronous=FULL")
            for batch_row, record in enumerate(records):
                connection.execute(
                    """
                    INSERT INTO candidates(
                        position, dataset_index, candidate_ordinal, seed, record_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        start_position + batch_row,
                        int(record["dataset_index"]),
                        int(record["candidate_ordinal"]),
                        int(record["seed"]),
                        json.dumps(record, ensure_ascii=False, allow_nan=False),
                    ),
                )


def persist_scored_batch(
    partition: str,
    start_position: int,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically replace one fully rescored problem after time is durably charged."""
    path = bank_database_path(partition)
    with contextlib.closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("PRAGMA synchronous=FULL")
            for offset, record in enumerate(records):
                cursor = connection.execute(
                    """
                    UPDATE candidates SET record_json = ?
                    WHERE position = ? AND dataset_index = ?
                      AND candidate_ordinal = ? AND seed = ?
                    """,
                    (
                        json.dumps(record, ensure_ascii=False, allow_nan=False),
                        start_position + offset,
                        int(record["dataset_index"]),
                        int(record["candidate_ordinal"]),
                        int(record["seed"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("scored-batch update lost its candidate binding")


def bank_content_digest(bank: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(bank))


def load_or_create_bank(
    partition: str, indices: Sequence[int], slots: Mapping[str, Any]
) -> dict[str, Any]:
    path = bank_path(partition)
    if path.is_file():
        if not bank_database_path(partition).is_file():
            raise RuntimeError(
                "protocol-bound bank metadata exists without its candidate database"
            )
        bank = json.loads(path.read_text(encoding="utf-8"))
        bank["records"] = load_bank_database_records(partition)
        expected = bank_template(partition, indices, slots)
        legacy_smoke = (
            partition == "calibration"
            and bank.get("schema_version") == "1.0.0"
            and len(bank["records"]) == RETAINED_SMOKE_CANDIDATE_COUNT
        )
        for field in (
            "experiment_id",
            "partition",
            "canonical_indices_sha256",
            "expected_problem_count",
            "expected_candidate_count",
        ):
            if bank.get(field) != expected[field]:
                raise RuntimeError(f"resumable bank binding changed at {field}")
        if legacy_smoke:
            previous_slots = dict(bank["protocol_slots"])
            unchanged_slots = {
                key: value
                for key, value in slots.items()
                if key != "runner_source_sha256"
            }
            if {
                key: value
                for key, value in previous_slots.items()
                if key != "runner_source_sha256"
            } != unchanged_slots:
                raise RuntimeError(
                    "retained smoke provenance changed outside runner amendment"
                )
            bank["schema_version"] = SCHEMA_VERSION
            bank["legacy_smoke_protocol_slots"] = previous_slots
            bank["protocol_slots"] = dict(slots)
            bank["generation_schedule"] = "mixed_batch1_prefix_then_amended_batch"
        else:
            for field in ("schema_version", "protocol_slots"):
                if bank.get(field) != expected[field]:
                    raise RuntimeError(f"resumable bank binding changed at {field}")
        return bank
    bank = bank_template(partition, indices, slots)
    if bank_database_path(partition).exists():
        raise RuntimeError(
            "candidate database exists without its protocol-bound metadata"
        )
    initialize_bank_database(partition)
    atomic_replace_json(path, _bank_metadata(bank))
    return bank


def validate_bank_records(bank: Mapping[str, Any], indices: Sequence[int]) -> None:
    expected_pairs = [
        (index, ordinal)
        for index in indices
        for ordinal in range(1, CANDIDATE_COUNT + 1)
    ]
    actual_pairs = [
        (int(record["dataset_index"]), int(record["candidate_ordinal"]))
        for record in bank["records"]
    ]
    if actual_pairs != expected_pairs[: len(actual_pairs)]:
        raise RuntimeError("resumable bank order/identity drift")
    partition = str(bank["partition"])
    for position, record in enumerate(bank["records"]):
        if int(record["seed"]) != generation_seed(
            int(record["dataset_index"]), int(record["candidate_ordinal"])
        ):
            raise RuntimeError("resumable bank seed drift")
        legacy_smoke = (
            partition == "calibration"
            and "legacy_smoke_protocol_slots" in bank
            and position < RETAINED_SMOKE_CANDIDATE_COUNT
        )
        if legacy_smoke:
            if any(
                name in record
                for name in (
                    "batch_seed",
                    "batch_size",
                    "batch_row",
                    "batch_payload_sha256",
                )
            ):
                raise RuntimeError("retained batch-size-1 smoke record was rewritten")
            continue
        ordinal = int(record["candidate_ordinal"])
        batch_size = int(bank.get("resolved_batch_size", RESOLVED_BATCH_SIZE))
        first_ordinal = ((ordinal - 1) // batch_size) * batch_size + 1
        expected_batch_seed, expected_payload_hash = batch_seed(
            partition,
            int(record["dataset_index"]),
            first_ordinal,
            batch_size,
        )
        expected_batch_fields = {
            "batch_seed": expected_batch_seed,
            "batch_size": batch_size,
            "batch_row": ordinal - first_ordinal,
            "batch_payload_sha256": expected_payload_hash,
        }
        if any(
            record.get(key) != value for key, value in expected_batch_fields.items()
        ):
            raise RuntimeError("resumable bank amended batch binding drift")


def generate_bank_prefix(
    train: Dataset,
    partition: str,
    indices: Sequence[int],
    slots: Mapping[str, Any],
    manifest: Mapping[str, str],
    *,
    problem_limit: int | None = None,
    batch_size: int = RESOLVED_BATCH_SIZE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_indices = list(indices[:problem_limit] if problem_limit else indices)
    bank = load_or_create_bank(partition, indices, slots)
    if "resolved_batch_size" not in bank:
        bank["resolved_batch_size"] = batch_size
    elif int(bank["resolved_batch_size"]) != batch_size:
        raise RuntimeError("resumable bank batch-size binding changed")
    validate_bank_records(bank, indices)
    target_count = len(selected_indices) * CANDIDATE_COUNT
    if len(bank["records"]) >= target_count:
        telemetry_rows = bank.get("telemetry", {}).get("generation_invocations", [])
        telemetry = telemetry_rows[-1] if telemetry_rows else None
        if not telemetry:
            raise RuntimeError("generated prefix has no bound telemetry")
        return bank, telemetry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("generation preflight requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer, model = load_generator(manifest, device)
    load_seconds = time.perf_counter() - load_started
    if problem_limit is None:
        record_retained_gpu_time(partition, "generation_model_load", load_seconds)
    generation_started = time.perf_counter()
    accounting_checkpoint = generation_started
    generated_this_run: list[dict[str, Any]] = []
    batch_stop_audits: list[dict[str, Any]] = []
    with PowerSampler.create() as power:
        if problem_limit is not None:
            for offset in range(len(bank["records"]), target_count):
                problem_position, ordinal_offset = divmod(offset, CANDIDATE_COUNT)
                record = generate_one(
                    train,
                    selected_indices[problem_position],
                    ordinal_offset + 1,
                    tokenizer,
                    model,
                    device,
                )
                bank["records"].append(record)
                generated_this_run.append(record)
                persist_candidate_record(partition, offset, record, generated=True)
        else:
            offset = len(bank["records"])
            if offset % batch_size:
                raise RuntimeError("resume position is inside an amended atomic batch")
            while offset < target_count:
                problem_position, ordinal_offset = divmod(offset, CANDIDATE_COUNT)
                if ordinal_offset % batch_size:
                    raise RuntimeError(
                        "amended batch crosses a frozen ordinal boundary"
                    )
                try:
                    records, stop_audit = generate_batch(
                        train,
                        partition,
                        selected_indices[problem_position],
                        ordinal_offset + 1,
                        batch_size,
                        tokenizer,
                        model,
                        device,
                    )
                except Exception:
                    record_retained_gpu_time(
                        partition,
                        "generation_failed_batch",
                        time.perf_counter() - accounting_checkpoint,
                    )
                    raise
                if not stop_audit["per_row_stopping"]["pass"]:
                    record_retained_gpu_time(
                        partition,
                        "generation_failed_stop_audit",
                        time.perf_counter() - accounting_checkpoint,
                    )
                    raise VoidIntegrityError(
                        "VOID_PER_ROW_STOPPING_AUDIT_FAILED", f"{partition}_generation"
                    )
                record_retained_gpu_time(
                    partition,
                    "generation_batch_before_bank_commit",
                    time.perf_counter() - accounting_checkpoint,
                )
                accounting_checkpoint = time.perf_counter()
                try:
                    persist_candidate_batch(partition, offset, records)
                except Exception as error:
                    raise VoidIntegrityError(
                        "VOID_GENERATED_BANK_PERSISTENCE_FAILURE",
                        f"{partition}_generation",
                    ) from error
                bank["records"].extend(records)
                generated_this_run.extend(records)
                batch_stop_audits.append(
                    {
                        "dataset_index": selected_indices[problem_position],
                        "first_candidate_ordinal": ordinal_offset + 1,
                        **stop_audit["per_row_stopping"],
                    }
                )
                offset += len(records)
    if problem_limit is None:
        record_retained_gpu_time(
            partition,
            "generation_finalize_before_metadata_commit",
            time.perf_counter() - accounting_checkpoint,
        )
    generation_seconds = time.perf_counter() - generation_started
    total_generated_tokens = sum(
        int(row["generated_tokens"]) for row in generated_this_run
    )
    active_seconds = sum(float(row["generation_seconds"]) for row in generated_this_run)
    prior_invocations = bank.get("telemetry", {}).get("generation_invocations", [])
    prior_gpu_seconds = sum(
        float(row["gpu_wall_seconds_delta"]) for row in prior_invocations
    )
    gpu_delta = load_seconds + generation_seconds
    telemetry = {
        "model_load_seconds": load_seconds,
        "wall_seconds": generation_seconds,
        "gpu_wall_seconds_delta": gpu_delta,
        "cumulative_partition_gpu_wall_seconds": prior_gpu_seconds + gpu_delta,
        "new_response_count": len(generated_this_run),
        "active_generation_seconds": active_seconds,
        "generated_tokens": total_generated_tokens,
        "tokens_per_second": total_generated_tokens / active_seconds
        if active_seconds
        else None,
        "responses_per_second": len(generated_this_run) / active_seconds
        if active_seconds
        else None,
        "cuda": cuda_telemetry(),
        "power": power.summary(active_seconds),
        "per_row_stopping_batch_count": len(batch_stop_audits),
        "per_row_stopping_pass": all(audit["pass"] for audit in batch_stop_audits),
    }
    bank["telemetry"].setdefault("generation_invocations", []).append(telemetry)
    if (
        problem_limit is None
        and len(bank["records"]) == bank["expected_candidate_count"]
    ):
        bank["complete_generation"] = True
    atomic_replace_json(bank_path(partition), _bank_metadata(bank))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return bank, telemetry


def score_bank_prefix(
    partition: str,
    indices: Sequence[int],
    slots: Mapping[str, Any],
    manifest: Mapping[str, str],
    *,
    problem_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bank = load_or_create_bank(partition, indices, slots)
    validate_bank_records(bank, indices)
    target_count = (problem_limit or len(indices)) * CANDIDATE_COUNT
    if len(bank["records"]) < target_count:
        raise RuntimeError("requested scoring prefix has not been generated")
    if all(
        bank["records"][offset].get("verifier_score") is not None
        for offset in range(target_count)
    ):
        telemetry_rows = bank.get("telemetry", {}).get("scoring_invocations", [])
        telemetry = telemetry_rows[-1] if telemetry_rows else None
        if not telemetry:
            raise RuntimeError("scored smoke prefix has no bound telemetry")
        return bank, telemetry
    scored_flags = [
        bank["records"][offset].get("verifier_score") is not None
        for offset in range(target_count)
    ]
    first_unscored = next(
        (offset for offset, scored in enumerate(scored_flags) if not scored), target_count
    )
    if first_unscored % CANDIDATE_COUNT or any(scored_flags[first_unscored:]):
        raise VoidIntegrityError(
            "VOID_SCORING_RESUME_NOT_PROBLEM_ATOMIC", f"{partition}_scoring"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("verifier scoring requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer, model = load_verifier(manifest, device)
    load_seconds = time.perf_counter() - load_started
    if problem_limit is None:
        record_retained_gpu_time(partition, "scoring_model_load", load_seconds)
    score_started = time.perf_counter()
    accounting_checkpoint = score_started
    scored_this_run: list[dict[str, Any]] = []
    with PowerSampler.create() as power:
        for problem_start in range(first_unscored, target_count, CANDIDATE_COUNT):
            problem_records: list[dict[str, Any]] = []
            try:
                for offset in range(problem_start, problem_start + CANDIDATE_COUNT):
                    problem_records.append(
                        score_one(bank["records"][offset], tokenizer, model, device)
                    )
            except Exception as error:
                if problem_limit is None:
                    record_retained_gpu_time(
                        partition,
                        "scoring_failed_problem",
                        time.perf_counter() - accounting_checkpoint,
                    )
                raise VoidIntegrityError(
                    f"VOID_SCORER_INTEGRITY_FAILURE:{type(error).__name__}",
                    f"{partition}_scoring",
                ) from error
            if problem_limit is None:
                record_retained_gpu_time(
                    partition,
                    "scoring_problem_before_bank_commit",
                    time.perf_counter() - accounting_checkpoint,
                )
                accounting_checkpoint = time.perf_counter()
            try:
                persist_scored_batch(partition, problem_start, problem_records)
            except Exception as error:
                raise VoidIntegrityError(
                    "VOID_SCORED_BANK_PERSISTENCE_FAILURE", f"{partition}_scoring"
                ) from error
            bank["records"][
                problem_start : problem_start + CANDIDATE_COUNT
            ] = problem_records
            scored_this_run.extend(problem_records)
    if problem_limit is None:
        record_retained_gpu_time(
            partition,
            "scoring_finalize_before_metadata_commit",
            time.perf_counter() - accounting_checkpoint,
        )
    score_seconds = time.perf_counter() - score_started
    active_seconds = sum(float(row["verifier_seconds"]) for row in scored_this_run)
    scored_tokens = sum(int(row["verifier_scored_tokens"]) for row in scored_this_run)
    prior_invocations = bank.get("telemetry", {}).get("scoring_invocations", [])
    prior_gpu_seconds = sum(
        float(row["gpu_wall_seconds_delta"]) for row in prior_invocations
    )
    gpu_delta = load_seconds + score_seconds
    telemetry = {
        "model_load_seconds": load_seconds,
        "wall_seconds": score_seconds,
        "gpu_wall_seconds_delta": gpu_delta,
        "cumulative_partition_gpu_wall_seconds": prior_gpu_seconds + gpu_delta,
        "new_score_count": len(scored_this_run),
        "active_scoring_seconds": active_seconds,
        "verifier_scored_tokens": scored_tokens,
        "tokens_per_second": scored_tokens / active_seconds if active_seconds else None,
        "cuda": cuda_telemetry(),
        "power": power.summary(active_seconds),
    }
    bank["telemetry"].setdefault("scoring_invocations", []).append(telemetry)
    if problem_limit is None and target_count == bank["expected_candidate_count"]:
        bank["complete_scoring"] = True
    atomic_replace_json(bank_path(partition), _bank_metadata(bank))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return bank, telemetry


def group_records(records: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []
    for start in range(0, len(records), CANDIDATE_COUNT):
        group = [dict(row) for row in records[start : start + CANDIDATE_COUNT]]
        if len(group) != CANDIDATE_COUNT:
            raise RuntimeError("candidate bank has an incomplete problem")
        if [row["candidate_ordinal"] for row in group] != list(range(1, 17)):
            raise RuntimeError("candidate bank ordinal order changed")
        grouped.append(group)
    return grouped


def _answer_group_choice(
    candidates: Sequence[Mapping[str, Any]],
    seen_positions: Sequence[int],
    previous_selected: int | None,
    *,
    weighted: bool,
) -> tuple[int, bool, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for position in seen_positions:
        answer = candidates[position]["extracted_answer"]
        if answer is not None:
            groups[str(answer)].append(position)
    if not groups:
        if weighted:
            selected = max(
                seen_positions,
                key=lambda p: (float(candidates[p]["verifier_score"]), -p),
            )
        else:
            selected = seen_positions[0]
        return selected, False, {"no_valid_answer": True}

    values = {
        answer: (
            sum(float(candidates[p]["verifier_score"]) for p in positions)
            if weighted
            else float(len(positions))
        )
        for answer, positions in groups.items()
    }
    maximum = max(values.values())
    tied = [answer for answer, value in values.items() if value == maximum]
    previous_answer = (
        candidates[previous_selected]["extracted_answer"]
        if previous_selected is not None
        else None
    )
    if previous_answer is not None and str(previous_answer) in tied:
        winner = str(previous_answer)
    else:
        winner = min(tied, key=lambda answer: min(groups[answer]))
    if weighted:
        selected = max(
            groups[winner],
            key=lambda p: (float(candidates[p]["verifier_score"]), -p),
        )
    else:
        selected = min(groups[winner])
    correct_wrong_counts = Counter(
        str(candidates[p]["extracted_answer"])
        for p in seen_positions
        if candidates[p]["extracted_answer"] is not None
        and not candidates[p]["correct"]
    )
    vote_count = len(groups[winner])
    valid_count = sum(len(positions) for positions in groups.values())
    return (
        selected,
        len(tied) > 1,
        {
            "no_valid_answer": False,
            "distinct_valid_answers": len(groups),
            "winning_answer_vote_count": vote_count,
            "winning_answer_vote_share": vote_count / valid_count,
            "largest_wrong_answer_vote_count": max(
                correct_wrong_counts.values(), default=0
            ),
            "largest_wrong_answer_vote_share": max(
                correct_wrong_counts.values(), default=0
            )
            / valid_count,
            "no_repeated_valid_answer": all(
                len(positions) == 1 for positions in groups.values()
            ),
            "tie_rule_used": len(tied) > 1,
        },
    )


def policy_trace(
    candidates: Sequence[Mapping[str, Any]],
    policy: int,
    parameters: Mapping[str, Any] | None = None,
    order: Sequence[int] | None = None,
) -> dict[str, Any]:
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError("every policy requires exactly 16 candidates")
    parameters = parameters or {}
    stream = [ordinal - 1 for ordinal in (order or range(1, 17))]
    selected = stream[0]
    selected_stream: list[int] = [selected]
    events: list[dict[str, Any]] = []
    if policy in (4, 5):
        selected, _, initial_extra = _answer_group_choice(
            candidates,
            stream[:1],
            None,
            weighted=policy == 5,
        )
        selected_stream[0] = selected
        extras: list[dict[str, Any]] = [initial_extra]
    else:
        extras = [{}]
    for stream_position in range(2, CANDIDATE_COUNT + 1):
        challenger = stream[stream_position - 1]
        previous = selected
        accepted = False
        tie_used = False
        extra: dict[str, Any] = {}
        if policy == 0:
            accepted = float(candidates[challenger]["verifier_score"]) > float(
                candidates[previous]["verifier_score"]
            )
            if accepted:
                selected = challenger
        elif policy == 1:
            delta = float(parameters["delta"])
            accepted = (
                float(candidates[challenger]["verifier_score"])
                - float(candidates[previous]["verifier_score"])
                > delta
            )
            if accepted:
                selected = challenger
        elif policy == 2:
            margin = float(parameters["delta_0"]) + float(
                parameters["lambda"]
            ) * math.sqrt(2 * math.log(stream_position))
            accepted = (
                float(candidates[challenger]["verifier_score"])
                - float(candidates[previous]["verifier_score"])
                > margin
            )
            extra["margin"] = margin
            if accepted:
                selected = challenger
        elif policy == 3:
            budget = int(parameters["budget"])
            if stream_position <= budget:
                accepted = float(candidates[challenger]["verifier_score"]) > float(
                    candidates[previous]["verifier_score"]
                )
                if accepted:
                    selected = challenger
        elif policy in (4, 5):
            selected, tie_used, extra = _answer_group_choice(
                candidates,
                stream[:stream_position],
                previous,
                weighted=policy == 5,
            )
            previous_answer = candidates[previous]["extracted_answer"]
            selected_answer = candidates[selected]["extracted_answer"]
            accepted = (
                selected != previous
                if extra.get("no_valid_answer")
                else selected_answer != previous_answer
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        events.append(
            {
                "stream_position": stream_position,
                "challenger_index": challenger,
                "prior_selected_index": previous,
                "selected_index": selected,
                "accepted_replacement": accepted,
                "harmful_challenge": bool(
                    candidates[previous]["correct"]
                    and not candidates[challenger]["correct"]
                ),
                "beneficial_challenge": bool(
                    not candidates[previous]["correct"]
                    and candidates[challenger]["correct"]
                ),
                "tie_rule_used": tie_used,
                **extra,
            }
        )
        selected_stream.append(selected)
        extras.append(extra)
    return {"selected_stream": selected_stream, "events": events, "extras": extras}


def _rate(numerator: int | float, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def evaluate_policy(
    problems: Sequence[Sequence[Mapping[str, Any]]],
    policy: int,
    parameters: Mapping[str, Any] | None = None,
    order: Sequence[int] | None = None,
    *,
    include_ledgers: bool = True,
) -> dict[str, Any]:
    traces = [policy_trace(problem, policy, parameters, order) for problem in problems]
    result: dict[str, Any] = {
        "policy": policy,
        "parameters": dict(parameters or {}),
        "prefixes": {},
    }
    if include_ledgers:
        result["switch_ledgers"] = [
            {
                "dataset_index": int(problem[0]["dataset_index"]),
                "events": trace["events"],
            }
            for problem, trace in zip(problems, traces, strict=True)
        ]
    stream_indices = [ordinal - 1 for ordinal in (order or range(1, 17))]
    first_position = stream_indices[0]
    for prefix in PREFIXES:
        selected_indices = [trace["selected_stream"][prefix - 1] for trace in traces]
        selected_records = [
            problem[index]
            for problem, index in zip(problems, selected_indices, strict=True)
        ]
        events_by_problem = [trace["events"][: max(0, prefix - 1)] for trace in traces]
        offered_harmful = sum(
            event["harmful_challenge"]
            for events in events_by_problem
            for event in events
        )
        accepted_harmful = sum(
            event["harmful_challenge"] and event["accepted_replacement"]
            for events in events_by_problem
            for event in events
        )
        offered_beneficial = sum(
            event["beneficial_challenge"]
            for events in events_by_problem
            for event in events
        )
        accepted_beneficial = sum(
            event["beneficial_challenge"] and event["accepted_replacement"]
            for events in events_by_problem
            for event in events
        )
        replacements = sum(
            event["accepted_replacement"]
            for events in events_by_problem
            for event in events
        )
        harmful_examples = sum(
            any(
                event["harmful_challenge"] and event["accepted_replacement"]
                for event in events
            )
            for events in events_by_problem
        )
        selected_correct = sum(bool(record["correct"]) for record in selected_records)
        oracle_correct = sum(
            any(
                bool(problem[position]["correct"])
                for position in stream_indices[:prefix]
            )
            for problem in problems
        )
        first_correct_denominator = sum(
            bool(problem[first_position]["correct"]) for problem in problems
        )
        first_incorrect_denominator = len(problems) - first_correct_denominator
        regression = sum(
            bool(problem[first_position]["correct"]) and not bool(record["correct"])
            for problem, record in zip(problems, selected_records, strict=True)
        )
        acquisition = sum(
            not bool(problem[first_position]["correct"]) and bool(record["correct"])
            for problem, record in zip(problems, selected_records, strict=True)
        )
        candidate_prefix = [
            problem[position]
            for problem in problems
            for position in stream_indices[:prefix]
        ]
        actual_per_problem = (
            min(prefix, int(parameters["budget"])) if policy == 3 else prefix
        )
        consumed = [
            problem[position]
            for problem in problems
            for position in stream_indices[:actual_per_problem]
        ]
        correct_incumbent_challenges = sum(
            bool(problems[problem_index][event["prior_selected_index"]]["correct"])
            for problem_index, events in enumerate(events_by_problem)
            for event in events
        )
        correct_incumbent_survived = sum(
            bool(problems[problem_index][event["prior_selected_index"]]["correct"])
            and bool(problems[problem_index][event["selected_index"]]["correct"])
            for problem_index, events in enumerate(events_by_problem)
            for event in events
        )
        prefix_result: dict[str, Any] = {
            "denominator": len(problems),
            "selected_accuracy": _rate(selected_correct, len(problems)),
            "oracle_coverage": _rate(oracle_correct, len(problems)),
            "oracle_to_selected_gap": (oracle_correct - selected_correct)
            / len(problems),
            "selected_index_distribution": dict(
                sorted(Counter(index + 1 for index in selected_indices).items())
            ),
            "selected_state_verifier_optimism": statistics.fmean(
                float(record["verifier_score"]) - int(bool(record["correct"]))
                for record in selected_records
            ),
            "harmful_switches": _rate(accepted_harmful, offered_harmful),
            "harmful_switch_examples": _rate(harmful_examples, len(problems)),
            "beneficial_switches": _rate(accepted_beneficial, offered_beneficial),
            "rejected_beneficial_challenges": _rate(
                offered_beneficial - accepted_beneficial, offered_beneficial
            ),
            "correct_incumbent_survival": _rate(
                correct_incumbent_survived, correct_incumbent_challenges
            ),
            "first_correct_to_selected_incorrect": _rate(
                regression, first_correct_denominator
            ),
            "first_incorrect_to_selected_correct": _rate(
                acquisition, first_incorrect_denominator
            ),
            "total_replacements": _rate(
                replacements, max(0, (prefix - 1) * len(problems))
            ),
            "candidate_model_empty_count": sum(
                bool(row["model_empty_non_answer"]) for row in candidate_prefix
            ),
            "candidate_parser_recognition_failure_count": sum(
                bool(row["parser_recognition_failure"]) for row in candidate_prefix
            ),
            "selected_model_empty_count": sum(
                bool(row["model_empty_non_answer"]) for row in selected_records
            ),
            "selected_parser_recognition_failure_count": sum(
                bool(row["parser_recognition_failure"]) for row in selected_records
            ),
            "correct_to_empty_switches": sum(
                event["accepted_replacement"]
                and bool(problems[i][event["prior_selected_index"]]["correct"])
                and bool(problems[i][event["selected_index"]]["model_empty_non_answer"])
                for i, events in enumerate(events_by_problem)
                for event in events
            ),
            "empty_to_correct_switches": sum(
                event["accepted_replacement"]
                and bool(
                    problems[i][event["prior_selected_index"]]["model_empty_non_answer"]
                )
                and bool(problems[i][event["selected_index"]]["correct"])
                for i, events in enumerate(events_by_problem)
                for event in events
            ),
            "actual_samples_consumed": {
                "per_problem": actual_per_problem,
                "total": actual_per_problem * len(problems),
            },
            "generated_tokens_consumed": sum(
                int(row["generated_tokens"]) for row in consumed
            ),
            "verifier_scored_tokens_consumed": sum(
                int(row["verifier_scored_tokens"] or 0) for row in consumed
            ),
            "compute": {
                "generation_active_seconds_consumed": sum(
                    float(row["generation_seconds"]) for row in consumed
                ),
                "verifier_active_seconds_consumed": sum(
                    float(row["verifier_seconds"] or 0.0) for row in consumed
                ),
                "gpu_power_and_peak_vram": "see artifact.compute stage telemetry",
            },
            "per_problem": [
                {
                    "dataset_index": int(problem[0]["dataset_index"]),
                    "selected_index": selected_index + 1,
                    "selected_correct": bool(selected["correct"]),
                    "selected_score": float(selected["verifier_score"]),
                    "optimism": float(selected["verifier_score"])
                    - int(bool(selected["correct"])),
                    "harmful_switch_indicator": any(
                        event["harmful_challenge"] and event["accepted_replacement"]
                        for event in events
                    ),
                }
                for problem, selected_index, selected, events in zip(
                    problems,
                    selected_indices,
                    selected_records,
                    events_by_problem,
                    strict=True,
                )
            ],
        }
        if policy in (4, 5):
            group_extras = [trace["extras"][prefix - 1] for trace in traces]
            valid_extras = [
                extra for extra in group_extras if not extra.get("no_valid_answer")
            ]
            answer_transitions_cw = 0
            answer_transitions_wc = 0
            for problem, trace in zip(problems, traces, strict=True):
                for event in trace["events"][: max(0, prefix - 1)]:
                    if not event["accepted_replacement"]:
                        continue
                    prior = problem[event["prior_selected_index"]]
                    after = problem[event["selected_index"]]
                    answer_transitions_cw += bool(
                        prior["correct"] and not after["correct"]
                    )
                    answer_transitions_wc += bool(
                        not prior["correct"] and after["correct"]
                    )
            prefix_result["answer_aggregation"] = {
                "distinct_valid_normalized_answers_mean": statistics.fmean(
                    extra.get("distinct_valid_answers", 0) for extra in group_extras
                ),
                "winning_answer_vote_count_mean": statistics.fmean(
                    extra.get("winning_answer_vote_count", 0) for extra in group_extras
                ),
                "winning_answer_vote_share_mean": statistics.fmean(
                    extra.get("winning_answer_vote_share", 0.0)
                    for extra in group_extras
                ),
                "largest_wrong_answer_vote_count_mean": statistics.fmean(
                    extra.get("largest_wrong_answer_vote_count", 0)
                    for extra in group_extras
                ),
                "largest_wrong_answer_vote_share_mean": statistics.fmean(
                    extra.get("largest_wrong_answer_vote_share", 0.0)
                    for extra in group_extras
                ),
                "no_repeated_valid_answer_fraction": sum(
                    bool(extra.get("no_repeated_valid_answer"))
                    for extra in group_extras
                )
                / len(problems),
                "tie_rule_fraction": sum(
                    bool(extra.get("tie_rule_used")) for extra in group_extras
                )
                / len(problems),
                "no_valid_answer_count": len(group_extras) - len(valid_extras),
                "selected_answer_correct_to_wrong_transitions": answer_transitions_cw,
                "selected_answer_wrong_to_correct_transitions": answer_transitions_wc,
            }
        result["prefixes"][str(prefix)] = prefix_result
    return result


def calibration_row(policy_result: Mapping[str, Any]) -> dict[str, Any]:
    final = policy_result["prefixes"]["16"]
    return {
        "parameters": policy_result["parameters"],
        "beneficial_acquisitions": final["first_incorrect_to_selected_correct"][
            "numerator"
        ],
        "accuracy": final["selected_accuracy"]["rate"],
        "harmful_switch_example_rate": final["harmful_switch_examples"]["rate"],
        "accepted_harmful_challenges": final["harmful_switches"],
        "rejected_beneficial_challenges": final["rejected_beneficial_challenges"],
        "total_replacements": final["total_replacements"],
        "selected_state_verifier_optimism": final["selected_state_verifier_optimism"],
    }


def select_calibration_parameters(
    problems: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    argmax = evaluate_policy(problems, 0, include_ledgers=False)
    reference_ba = argmax["prefixes"]["16"]["first_incorrect_to_selected_correct"][
        "numerator"
    ]
    tables: dict[str, list[dict[str, Any]]] = {
        "policy_1": [],
        "policy_2": [],
        "policy_3": [],
    }
    if reference_ba == 0:
        return {
            "status": "KILL",
            "reason": "NO_BENEFICIAL_ACQUISITION_HEADROOM",
            "reference_beneficial_acquisitions": 0,
            "tables": tables,
            "selected": {},
        }
    for delta in DELTA_GRID:
        row = calibration_row(
            evaluate_policy(problems, 1, {"delta": delta}, include_ledgers=False)
        )
        row["retention"] = row["beneficial_acquisitions"] / reference_ba
        row["feasible"] = row["retention"] >= 0.90
        tables["policy_1"].append(row)
    for delta0 in DELTA_GRID:
        for lambda_ in LAMBDA_GRID:
            row = calibration_row(
                evaluate_policy(
                    problems,
                    2,
                    {"delta_0": delta0, "lambda": lambda_},
                    include_ledgers=False,
                )
            )
            row["terminal_margin"] = delta0 + lambda_ * math.sqrt(2 * math.log(16))
            row["retention"] = row["beneficial_acquisitions"] / reference_ba
            row["feasible"] = row["retention"] >= 0.90
            tables["policy_2"].append(row)
    for budget in BUDGET_GRID:
        row = calibration_row(
            evaluate_policy(problems, 3, {"budget": budget}, include_ledgers=False)
        )
        row["retention"] = row["beneficial_acquisitions"] / reference_ba
        row["feasible"] = row["retention"] >= 0.90
        tables["policy_3"].append(row)

    feasible = {
        key: [row for row in rows if row["feasible"]] for key, rows in tables.items()
    }
    if any(not rows for rows in feasible.values()):
        return {
            "status": "KILL",
            "reason": "NO_FEASIBLE_CALIBRATION_PARAMETER",
            "reference_beneficial_acquisitions": reference_ba,
            "tables": tables,
            "selected": {},
        }
    p1 = min(
        feasible["policy_1"],
        key=lambda row: (
            row["harmful_switch_example_rate"],
            -row["accuracy"],
            -row["parameters"]["delta"],
        ),
    )
    p2 = min(
        feasible["policy_2"],
        key=lambda row: (
            row["harmful_switch_example_rate"],
            -row["accuracy"],
            -row["terminal_margin"],
            -row["parameters"]["lambda"],
            -row["parameters"]["delta_0"],
        ),
    )
    p3 = min(
        feasible["policy_3"],
        key=lambda row: (
            row["harmful_switch_example_rate"],
            -row["accuracy"],
            row["parameters"]["budget"],
        ),
    )
    return {
        "status": "PASS",
        "reason": "CALIBRATION_PARAMETERS_FROZEN",
        "reference_beneficial_acquisitions": reference_ba,
        "tables": tables,
        "selected": {
            "policy_1": p1["parameters"],
            "policy_2": p2["parameters"],
            "policy_3": p3["parameters"],
        },
        "multiplicity_term_collapsed": p2["parameters"]["lambda"] == 0,
    }


def all_policy_results(
    problems: Sequence[Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Mapping[str, Any]],
    order: Sequence[int] | None = None,
    *,
    include_ledgers: bool = True,
) -> dict[str, Any]:
    return {
        "policy_0": evaluate_policy(
            problems, 0, order=order, include_ledgers=include_ledgers
        ),
        "policy_1": evaluate_policy(
            problems,
            1,
            selected["policy_1"],
            order,
            include_ledgers=include_ledgers,
        ),
        "policy_2": evaluate_policy(
            problems,
            2,
            selected["policy_2"],
            order,
            include_ledgers=include_ledgers,
        ),
        "policy_3": evaluate_policy(
            problems,
            3,
            selected["policy_3"],
            order,
            include_ledgers=include_ledgers,
        ),
        "policy_4": evaluate_policy(
            problems, 4, order=order, include_ledgers=include_ledgers
        ),
        "policy_5": evaluate_policy(
            problems, 5, order=order, include_ledgers=include_ledgers
        ),
    }


def bootstrap_positions(replicate: int, count: int) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                f"{BOOTSTRAP_STRING}\n{replicate}\n{k}".encode("ascii")
            ).digest()[:8],
            "big",
        )
        % count
        for k in range(count)
    ]


def percentile_interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "lower_2_5": float(np.percentile(array, 2.5, method="linear")),
        "median": float(np.percentile(array, 50, method="linear")),
        "upper_97_5": float(np.percentile(array, 97.5, method="linear")),
    }


def paired_bootstrap(policy_results: Mapping[str, Any]) -> dict[str, Any]:
    final = {
        key: value["prefixes"]["16"]["per_problem"]
        for key, value in policy_results.items()
    }
    count = len(final["policy_2"])
    series: dict[str, list[float]] = defaultdict(list)
    for replicate in range(BOOTSTRAP_REPLICATES):
        positions = bootstrap_positions(replicate, count)
        for baseline in ("policy_0", "policy_4", "policy_5"):
            series[f"accuracy_minus_{baseline}"].append(
                statistics.fmean(
                    int(final["policy_2"][p]["selected_correct"])
                    - int(final[baseline][p]["selected_correct"])
                    for p in positions
                )
            )
        series["policy_0_minus_policy_2_optimism"].append(
            statistics.fmean(
                final["policy_0"][p]["optimism"] - final["policy_2"][p]["optimism"]
                for p in positions
            )
        )
        series["policy_0_minus_policy_2_harmful_indicator"].append(
            statistics.fmean(
                int(final["policy_0"][p]["harmful_switch_indicator"])
                - int(final["policy_2"][p]["harmful_switch_indicator"])
                for p in positions
            )
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed_string": BOOTSTRAP_STRING,
        "percentile_method": "numpy_linear",
        "intervals": {
            key: percentile_interval(values) for key, values in series.items()
        },
        "replicate_statistics": dict(series),
    }


def order_permutation_analysis(
    problems: Sequence[Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for permutation_index, order in enumerate(permutation_schedule()):
        results = all_policy_results(problems, selected, order, include_ledgers=False)
        final = {key: value["prefixes"]["16"] for key, value in results.items()}
        reference_ba = final["policy_0"]["first_incorrect_to_selected_correct"][
            "numerator"
        ]
        policy_ba = final["policy_2"]["first_incorrect_to_selected_correct"][
            "numerator"
        ]
        retention = policy_ba / reference_ba if reference_ba else None
        rows.append(
            {
                "permutation_index": permutation_index,
                "accuracy_advantages": {
                    baseline: final["policy_2"]["selected_accuracy"]["rate"]
                    - final[baseline]["selected_accuracy"]["rate"]
                    for baseline in ("policy_0", "policy_4", "policy_5")
                },
                "beneficial_acquisition_retention": retention,
                "lower_optimism": final["policy_2"]["selected_state_verifier_optimism"]
                < final["policy_0"]["selected_state_verifier_optimism"],
                "fewer_harmful_switch_examples": final["policy_2"][
                    "harmful_switch_examples"
                ]["numerator"]
                < final["policy_0"]["harmful_switch_examples"]["numerator"],
                "replacement_count": final["policy_2"]["total_replacements"][
                    "numerator"
                ],
            }
        )
    baseline_summary = {}
    for baseline in ("policy_0", "policy_4", "policy_5"):
        values = [row["accuracy_advantages"][baseline] for row in rows]
        baseline_summary[baseline] = {
            "median": float(np.percentile(values, 50, method="linear")),
            "fifth_percentile": float(np.percentile(values, 5, method="linear")),
            "passes": float(np.percentile(values, 50, method="linear")) >= 0.03
            and float(np.percentile(values, 5, method="linear")) > 0,
        }
    conditions = {
        "baseline_survival": all(
            value["passes"] for value in baseline_summary.values()
        ),
        "retention_at_least_90_count": sum(
            row["beneficial_acquisition_retention"] is not None
            and row["beneficial_acquisition_retention"] >= 0.90
            for row in rows
        ),
        "lower_optimism_count": sum(row["lower_optimism"] for row in rows),
        "fewer_harmful_switch_examples_count": sum(
            row["fewer_harmful_switch_examples"] for row in rows
        ),
        "replacement_every_permutation": all(
            row["replacement_count"] > 0 for row in rows
        ),
    }
    conditions["passes"] = (
        conditions["baseline_survival"]
        and conditions["retention_at_least_90_count"] >= 950
        and conditions["lower_optimism_count"] >= 950
        and conditions["fewer_harmful_switch_examples_count"] >= 950
        and conditions["replacement_every_permutation"]
    )
    return {
        "replicates": PERMUTATION_REPLICATES,
        "seed_string": PERMUTATION_STRING,
        "schedule_sha256": sha256_text(
            "\n".join(
                ",".join(str(value) for value in row) for row in permutation_schedule()
            )
        ),
        "baseline_summary": baseline_summary,
        "conditions": conditions,
        "rows": rows,
    }


def adjudicate(
    policy_results: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutations: Mapping[str, Any],
) -> dict[str, Any]:
    final = {key: value["prefixes"]["16"] for key, value in policy_results.items()}
    p2 = final["policy_2"]
    p0 = final["policy_0"]
    reference_ba = p0["first_incorrect_to_selected_correct"]["numerator"]
    retention = (
        p2["first_incorrect_to_selected_correct"]["numerator"] / reference_ba
        if reference_ba
        else None
    )
    intervals = bootstrap["intervals"]
    accuracy_conditions = {
        baseline: p2["selected_accuracy"]["rate"]
        - final[baseline]["selected_accuracy"]["rate"]
        >= 0.03
        and intervals[f"accuracy_minus_{baseline}"]["lower_2_5"] > 0
        for baseline in ("policy_0", "policy_4", "policy_5")
    }
    optimism_difference = (
        p0["selected_state_verifier_optimism"] - p2["selected_state_verifier_optimism"]
    )
    p0_harm = p0["harmful_switch_examples"]["rate"]
    p2_harm = p2["harmful_switch_examples"]["rate"]
    relative_harm_reduction = (p0_harm - p2_harm) / p0_harm if p0_harm else None
    conditions = {
        "accuracy_and_bootstrap": all(accuracy_conditions.values()),
        "beneficial_acquisition_retention": retention is not None and retention >= 0.90,
        "actual_replacement": p2["total_replacements"]["numerator"] > 0,
        "beneficial_acquisition": p2["first_incorrect_to_selected_correct"]["numerator"]
        > 0,
        "optimism_reduction": optimism_difference >= 0.02
        and intervals["policy_0_minus_policy_2_optimism"]["lower_2_5"] > 0,
        "harmful_switch_reduction": relative_harm_reduction is not None
        and relative_harm_reduction >= 0.20
        and intervals["policy_0_minus_policy_2_harmful_indicator"]["lower_2_5"] > 0,
        "order_permutation_survival": permutations["conditions"]["passes"],
    }
    token = "CONFIRM" if all(conditions.values()) else "KILL"
    return {
        "final_token": token,
        "conditions": conditions,
        "accuracy_conditions": accuracy_conditions,
        "beneficial_acquisition_retention": retention,
        "optimism_difference": optimism_difference,
        "relative_harmful_switch_reduction": relative_harm_reduction,
    }


def validate_result_schema(result: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "created_at",
        "protocol",
        "calibration",
        "candidate_records",
        "policy_results",
        "bootstrap",
        "permutations",
        "compute",
        "adjudication",
        "claim_language_contract",
        "final_token",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"result schema missing fields: {missing}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version changed")
    if result["final_token"] not in FINAL_TOKENS:
        raise ValueError("result final token is invalid")
    if result["claim_language_contract"] != _claim_language_contract():
        raise ValueError("result is not bound to the canonical claim-language contract")
    scientific_fields = (
        "calibration",
        "candidate_records",
        "policy_results",
        "bootstrap",
        "permutations",
    )
    if result["final_token"] == "VOID":
        if any(result[name] != NOT_APPLICABLE for name in scientific_fields):
            raise ValueError("VOID scientific fields must be N/A")
        adjudication = result["adjudication"]
        if (
            adjudication.get("final_token") != "VOID"
            or adjudication.get("scientific_adjudication") != NOT_APPLICABLE
            or not adjudication.get("reason")
        ):
            raise ValueError("VOID adjudication lacks reason/N-A discipline")
        return
    if any(result[name] == NOT_APPLICABLE for name in scientific_fields):
        raise ValueError("CONFIRM/KILL scientific fields may not be N/A")
    cap_resolution = result["protocol"].get("cap_resolution", {})
    counts = cap_resolution.get("resolved_counts", {})
    calibration_count = int(counts.get("calibration", -1))
    test_count = int(counts.get("test", -1))
    if not 0 < calibration_count <= CALIBRATION_COUNT:
        raise ValueError("result calibration denominator is not cap-resolved")
    if not 0 < test_count <= TEST_COUNT:
        raise ValueError("result test denominator is not cap-resolved")
    records = result["candidate_records"]
    expected_records = (calibration_count + test_count) * CANDIDATE_COUNT
    if len(records) != expected_records:
        raise ValueError(f"result must bind all {expected_records} resolved candidates")
    required_record = {
        "dataset_index",
        "candidate_ordinal",
        "seed",
        "response",
        "extracted_answer",
        "outcome_category",
        "correct",
        "verifier_score",
        "verifier_step_scores",
        "verifier_step_score_bf16_bits",
        "verifier_score_bf16_bits",
        "verifier_input_sha256",
        "verifier_token_ids_sha256",
        "verifier_marker_positions",
        "verifier_output_digest",
        "stop_reason",
        "generated_tokens",
    }
    identities = []
    for position, record in enumerate(records):
        if missing_record := sorted(required_record - set(record)):
            raise ValueError(f"candidate {position} missing fields: {missing_record}")
        score = record["verifier_score"]
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(f"candidate {position} has invalid score")
        ordinal = int(record["candidate_ordinal"])
        if ordinal != position % CANDIDATE_COUNT + 1:
            raise ValueError(f"candidate {position} violates frozen ordinal order")
        identity = (int(record["dataset_index"]), ordinal)
        identities.append(identity)
        if int(record["seed"]) != generation_seed(*identity):
            raise ValueError(f"candidate {position} violates frozen seed schedule")
    if len(set(identities)) != len(identities):
        raise ValueError("result contains duplicate candidate identities")
    for policy in range(6):
        policy_key = f"policy_{policy}"
        if policy_key not in result["policy_results"]:
            raise ValueError(f"result is missing {policy_key}")
        if set(result["policy_results"][policy_key].get("prefixes", {})) != {
            str(prefix) for prefix in PREFIXES
        }:
            raise ValueError(f"{policy_key} does not report every frozen prefix")
        ledgers = result["policy_results"][policy_key].get("switch_ledgers")
        if not isinstance(ledgers, list) or len(ledgers) != test_count:
            raise ValueError(f"{policy_key} lacks {test_count} switch ledgers")
        if any(len(ledger.get("events", [])) != 15 for ledger in ledgers):
            raise ValueError(f"{policy_key} has an incomplete switch ledger")
    if result["adjudication"]["final_token"] != result["final_token"]:
        raise ValueError("adjudication token mismatch")

    def check_denominators(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if {"numerator", "denominator", "rate"} <= set(value):
                numerator = value["numerator"]
                denominator = value["denominator"]
                rate = value["rate"]
                if not isinstance(denominator, int) or denominator < 0:
                    raise ValueError(f"{path} has an invalid denominator")
                if denominator == 0:
                    if rate is not None:
                        raise ValueError(f"{path} reports a vacuous rate")
                elif not isinstance(rate, (int, float)) or not math.isclose(
                    float(rate), float(numerator) / denominator, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"{path} rate does not match numerator/denominator")
            for key, child in value.items():
                check_denominators(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_denominators(child, f"{path}[{index}]")

    for name in ("calibration", "policy_results", "bootstrap", "permutations"):
        check_denominators(result[name], name)


def _open_exclusive_private_temp(directory: Path) -> tuple[Path, Any]:
    for _ in range(128):
        path = (
            directory
            / f"{IMMUTABLE_TEMP_PREFIX}{uuid.uuid4().hex}{IMMUTABLE_TEMP_SUFFIX}"
        )
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        return path, os.fdopen(descriptor, "wb")
    raise FileExistsError("could not allocate immutable result temp")


def _assert_temp_namespace_disjoint(final_path: Path) -> None:
    if IMMUTABLE_TEMP_SUFFIX == ".json" or final_path.suffix != ".json":
        raise AssertionError("immutable temp namespace overlaps JSON finals")
    if final_path.match(IMMUTABLE_TEMP_GLOB):
        raise AssertionError("final path matches immutable temp namespace")


def _fsync_directory(directory: Path) -> bool:
    if os.name == "nt":
        return False
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def scavenge_stale_immutable_temps(directory: Path, final_path: Path) -> list[Path]:
    _assert_temp_namespace_disjoint(final_path)
    if not directory.exists():
        return []
    removed = []
    for candidate in directory.glob(IMMUTABLE_TEMP_GLOB):
        candidate.unlink()
        removed.append(candidate)
    if removed:
        _fsync_directory(directory)
    return removed


def _atomic_no_clobber_bytes(payload: bytes, path: Path) -> None:
    _assert_temp_namespace_disjoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        temporary, handle = _open_exclusive_private_temp(path.parent)
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"immutable result already exists: {path}") from error
        temporary.unlink()
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def write_one_shot_gate(path: Path, report: Mapping[str, Any]) -> None:
    """Write a controlling gate exactly once; PASS and terminal failure are final."""
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        _atomic_no_clobber_bytes(payload, path)
    except RuntimeError as error:
        raise RuntimeError(f"one-shot gate already resolved: {path}") from error


def _claim_language_contract() -> dict[str, Any]:
    text = PREREGISTRATION_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    marker = f"## {CLAIM_CONTRACT_SECTION}\n"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("claim-language contract section is missing")
    end = text.find("\n## ", start + len(marker))
    if end < 0:
        raise RuntimeError("claim-language contract section has no closing heading")
    section = text[start:end].rstrip("\n")
    return {
        "source": "experiments/07_safe_selection/PREREGISTRATION.md",
        "section": CLAIM_CONTRACT_SECTION,
        "section_sha256": sha256_text(section),
        "required_scope_phrase": (
            "on this frozen generator-verifier pair and frozen GSM8K train-split cohort"
        ),
        "ratio_language_requires_named_rates": True,
        "zero_numerator_requires_denominator_and_one_sided_bound": True,
        "forbidden_claims": [
            "never gets worse",
            "cannot abandon a correct answer",
            "eliminates harmful replacement",
            "virtually never",
            "monotonic",
            "anytime-safe",
        ],
    }


def _ledger_events(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.json"))
    if [path.name for path in paths] != [
        f"{index:08d}.json" for index in range(1, len(paths) + 1)
    ]:
        raise StageTransitionError(f"ledger sequence is not contiguous: {directory}")
    events: list[dict[str, Any]] = []
    previous_digest: str | None = None
    for index, path in enumerate(paths, start=1):
        event = json.loads(path.read_text(encoding="utf-8"))
        digest = event.pop("event_sha256", None)
        if event.get("event_index") != index:
            raise StageTransitionError(f"ledger event index drift: {path}")
        if event.get("previous_event_sha256") != previous_digest:
            raise StageTransitionError(f"ledger hash chain drift: {path}")
        actual = sha256_bytes(canonical_json_bytes(event))
        if digest != actual:
            raise StageTransitionError(f"ledger event digest drift: {path}")
        event["event_sha256"] = digest
        events.append(event)
        previous_digest = digest
    return events


def _append_ledger_event(directory: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    events = _ledger_events(directory)
    event = {
        "event_index": len(events) + 1,
        "previous_event_sha256": events[-1]["event_sha256"] if events else None,
        "created_at": utc_now(),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        **dict(payload),
    }
    event["event_sha256"] = sha256_bytes(canonical_json_bytes(event))
    path = directory / f"{event['event_index']:08d}.json"
    _atomic_no_clobber_bytes(
        (json.dumps(event, indent=2, allow_nan=False) + "\n").encode("utf-8"), path
    )
    return event


def _stage_snapshot(directory: Path = STAGE_LEDGER_DIR) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    last_index = -1
    terminal = False
    for event in _ledger_events(directory):
        stage = str(event.get("stage"))
        state = str(event.get("state"))
        if stage not in STAGE_ORDER or state not in STAGE_STATES[stage]:
            raise StageTransitionError("stage ledger contains an unknown transition")
        index = STAGE_ORDER.index(stage)
        if terminal or index < last_index or index > last_index + 1:
            raise StageTransitionError("stage ledger is not forward-only")
        previous = snapshot.get(stage)
        allowed = {
            "calibration": {None: "STARTED", "STARTED": "PASS"},
            "test": {None: "STARTED", "STARTED": "SCORED", "SCORED": "PASS"},
        }.get(stage, {None: state})
        rescore_terminal_reconcile = (
            stage == "rescore"
            and state == "VOID"
            and previous in {None, "STARTED", "PASS"}
            and isinstance(event.get("terminal_failure_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", event["terminal_failure_sha256"])
        )
        rescore_progress = stage == "rescore" and (
            (previous is None and state == "STARTED")
            or (previous == "STARTED" and state == "PASS")
        )
        if stage == "rescore" and state == "VOID" and not rescore_terminal_reconcile:
            raise StageTransitionError("rescore VOID lacks terminal failure evidence")
        if stage == "rescore" and not (
            rescore_progress or rescore_terminal_reconcile
        ):
            raise StageTransitionError(f"invalid repeated transition for {stage}: {state}")
        if (
            allowed.get(previous) != state
            and not rescore_progress
            and not rescore_terminal_reconcile
        ):
            raise StageTransitionError(f"invalid repeated transition for {stage}: {state}")
        if index == last_index + 1:
            if index and snapshot.get(STAGE_ORDER[index - 1]) != "PASS":
                raise StageTransitionError(f"stage {stage} skipped its predecessor")
            last_index = index
        snapshot[stage] = state
        terminal = state in {"STOP", "VOID"}
    return snapshot


def advance_stage(
    stage: str,
    state: str,
    *,
    directory: Path = STAGE_LEDGER_DIR,
    gpu_time_path: Path = GPU_TIME_LEDGER_PATH,
    terminal_failure_sha256: str | None = None,
) -> dict[str, Any]:
    before = _stage_snapshot(directory)
    if stage not in STAGE_ORDER or state not in STAGE_STATES[stage]:
        raise StageTransitionError("unknown stage transition")
    if before.get(stage) == state:
        raise StageTransitionError(f"stage transition is one-shot: {stage} {state}")
    index = STAGE_ORDER.index(stage)
    previous = before.get(stage)
    expected = {
        "calibration": {None: "STARTED", "STARTED": "PASS"},
        "test": {None: "STARTED", "STARTED": "SCORED", "SCORED": "PASS"},
    }.get(stage, {None: state}).get(previous)
    rescore_terminal_reconcile = (
        stage == "rescore"
        and state == "VOID"
        and previous in {None, "STARTED", "PASS"}
        and isinstance(terminal_failure_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", terminal_failure_sha256)
    )
    rescore_progress = stage == "rescore" and (
        (previous is None and state == "STARTED")
        or (previous == "STARTED" and state == "PASS")
    )
    if stage == "rescore" and state == "VOID" and not rescore_terminal_reconcile:
        raise StageTransitionError("rescore VOID lacks terminal failure evidence")
    if stage == "rescore" and not (rescore_progress or rescore_terminal_reconcile):
        raise StageTransitionError(f"invalid transition for {stage}: {previous} -> {state}")
    if expected != state and not rescore_progress and not rescore_terminal_reconcile:
        raise StageTransitionError(f"invalid transition for {stage}: {previous} -> {state}")
    if previous is None and index and before.get(STAGE_ORDER[index - 1]) != "PASS":
        raise StageTransitionError(f"stage {stage} skipped its predecessor")
    if any(STAGE_ORDER.index(name) > index for name in before):
        raise StageTransitionError(f"stage {stage} cannot move backward")
    retained_gpu_at_boundary: float | str
    retained_gpu_ledger_error: str | None = None
    try:
        retained_gpu_at_boundary = retained_gpu_seconds(gpu_time_path)
    except Exception as error:
        if not rescore_terminal_reconcile:
            raise
        retained_gpu_at_boundary = NOT_APPLICABLE
        retained_gpu_ledger_error = f"{type(error).__name__}: {error}"
    event_payload = {
        "stage": stage,
        "state": state,
        "retained_gpu_seconds_at_boundary": retained_gpu_at_boundary,
    }
    if rescore_terminal_reconcile:
        event_payload["terminal_failure_sha256"] = terminal_failure_sha256
        event_payload["retained_gpu_ledger_error"] = retained_gpu_ledger_error
    event = _append_ledger_event(
        directory,
        event_payload,
    )
    _stage_snapshot(directory)
    return event


def require_stage(
    stage: str,
    state: str = "PASS",
    *,
    directory: Path = STAGE_LEDGER_DIR,
) -> None:
    actual = _stage_snapshot(directory).get(stage)
    if actual != state:
        raise StageTransitionError(
            f"stage boundary requires {stage}={state}; observed {actual or 'UNRUN'}"
        )
    assert_retained_gpu_cap()


def ensure_stage_started(stage: str) -> None:
    snapshot = _stage_snapshot()
    if stage not in snapshot:
        advance_stage(stage, "STARTED")
    elif snapshot[stage] != "STARTED":
        raise StageTransitionError(
            f"stage {stage} cannot start from persisted state {snapshot[stage]}"
        )
    assert_retained_gpu_cap()


def _gpu_time_events(path: Path = GPU_TIME_LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous_digest: str | None = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise StageTransitionError("retained GPU ledger has a partial event") from error
        digest = event.pop("event_sha256", None)
        if (
            event.get("event_index") != index
            or event.get("previous_event_sha256") != previous_digest
            or digest != sha256_bytes(canonical_json_bytes(event))
        ):
            raise StageTransitionError("retained GPU ledger hash chain drift")
        event["event_sha256"] = digest
        events.append(event)
        previous_digest = digest
    return events


def retained_gpu_seconds(path: Path = GPU_TIME_LEDGER_PATH) -> float:
    events = _gpu_time_events(path)
    cumulative = 0.0
    for event in events:
        delta = float(event.get("delta_gpu_seconds", -1.0))
        if delta < 0:
            raise StageTransitionError("retained GPU ledger contains a negative delta")
        cumulative += delta
        if not math.isclose(
            float(event.get("cumulative_gpu_seconds", -1.0)),
            cumulative,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise StageTransitionError("retained GPU ledger cumulative total drifted")
    return cumulative


def record_retained_gpu_time(
    stage: str,
    operation: str,
    delta_seconds: float,
    *,
    path: Path = GPU_TIME_LEDGER_PATH,
) -> dict[str, Any]:
    if not math.isfinite(delta_seconds) or delta_seconds < 0:
        raise ValueError("retained GPU time delta must be finite and nonnegative")
    events = _gpu_time_events(path)
    cumulative = retained_gpu_seconds(path) + delta_seconds
    event = {
        "event_index": len(events) + 1,
        "previous_event_sha256": events[-1]["event_sha256"] if events else None,
        "created_at": utc_now(),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "stage": stage,
        "operation": operation,
        "delta_gpu_seconds": delta_seconds,
        "cumulative_gpu_seconds": cumulative,
        "authorizing_ceiling_seconds": CAP_AUTHORIZING_SECONDS,
    }
    event["event_sha256"] = sha256_bytes(canonical_json_bytes(event))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_bytes(event) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    if cumulative > CAP_AUTHORIZING_SECONDS:
        raise VoidIntegrityError("VOID_RETAINED_GPU_TIME_CAP_EXCEEDED", stage)
    return event


def assert_retained_gpu_cap(path: Path = GPU_TIME_LEDGER_PATH) -> float:
    cumulative = retained_gpu_seconds(path)
    if cumulative > CAP_AUTHORIZING_SECONDS:
        raise VoidIntegrityError("VOID_RETAINED_GPU_TIME_CAP_EXCEEDED", "stage_boundary")
    return cumulative


def write_immutable_result(result: Mapping[str, Any], path: Path) -> None:
    validate_result_schema(result)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_no_clobber_bytes(payload, path)


def land_void_result(
    reason: str,
    stage: str,
    *,
    protocol: Mapping[str, Any] | None = None,
    compute: Mapping[str, Any] | None = None,
    result_path: Path = RESULT_PATH,
    stage_directory: Path = STAGE_LEDGER_DIR,
    gpu_time_path: Path = GPU_TIME_LEDGER_PATH,
) -> dict[str, Any]:
    try:
        retained_seconds: float | str = retained_gpu_seconds(gpu_time_path)
        retained_ledger_error: str | None = None
    except Exception as error:
        retained_seconds = NOT_APPLICABLE
        retained_ledger_error = f"{type(error).__name__}: {error}"
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "protocol": {
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "failed_stage": stage,
            "stage_ledger": _stage_snapshot(stage_directory),
            **dict(protocol or {}),
        },
        "calibration": NOT_APPLICABLE,
        "candidate_records": NOT_APPLICABLE,
        "policy_results": NOT_APPLICABLE,
        "bootstrap": NOT_APPLICABLE,
        "permutations": NOT_APPLICABLE,
        "compute": {
            "retained_gpu_seconds": retained_seconds,
            "retained_gpu_ledger_error": retained_ledger_error,
            **dict(compute or {}),
        },
        "adjudication": {
            "final_token": "VOID",
            "reason": reason,
            "scientific_adjudication": NOT_APPLICABLE,
            "claim_authorized": False,
        },
        "claim_language_contract": _claim_language_contract(),
        "final_token": "VOID",
    }
    write_immutable_result(result, result_path)
    return result


def _rescore_terminal_failure_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("artifact_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _rescore_pass_artifact_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("artifact_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def load_rescore_pass_artifact(
    path: Path = OUTCOME_BLIND_RESCORE_PATH,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("terminal") is not False
        or report.get("runner_source_sha256")
        != sha256_file(Path(__file__).resolve())
        or report.get("artifact_sha256") != _rescore_pass_artifact_digest(report)
    ):
        raise VoidIntegrityError("VOID_SCORER_RESCORE_PASS_ARTIFACT_INVALID", "rescore")
    return report


def load_rescore_terminal_failure(
    path: Path = RESCORE_TERMINAL_FAILURE_PATH,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "VOID"
        or report.get("terminal") is not True
        or report.get("failed_stage") != "rescore"
        or not report.get("reason")
        or report.get("runner_source_sha256")
        != sha256_file(Path(__file__).resolve())
        or report.get("artifact_sha256") != _rescore_terminal_failure_digest(report)
    ):
        raise StageTransitionError("rescore terminal failure artifact is invalid")
    return report


def persist_rescore_terminal_failure(
    reason: str,
    failure: str | None,
    *,
    details: Mapping[str, Any] | None = None,
    path: Path = RESCORE_TERMINAL_FAILURE_PATH,
) -> dict[str, Any]:
    """Persist the first terminal rescore failure; it controls every retry."""
    if path.is_file():
        return load_rescore_terminal_failure(path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "VOID",
        "terminal": True,
        "failed_stage": "rescore",
        "reason": reason,
        "failure": failure,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "details": dict(details or {}),
    }
    report["artifact_sha256"] = _rescore_terminal_failure_digest(report)
    write_one_shot_gate(path, report)
    return report


def reconcile_rescore_terminal(
    *,
    failure_path: Path = RESCORE_TERMINAL_FAILURE_PATH,
    stage_directory: Path = STAGE_LEDGER_DIR,
    result_path: Path = RESULT_PATH,
    gpu_time_path: Path = GPU_TIME_LEDGER_PATH,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Finish an interrupted rescore VOID landing without rerunning GPU work."""
    if not failure_path.is_file():
        return None
    failure = load_rescore_terminal_failure(failure_path)
    failure_digest = str(failure["artifact_sha256"])
    snapshot = _stage_snapshot(stage_directory)
    if snapshot.get("rescore") != "VOID":
        if "test" in snapshot:
            raise StageTransitionError("rescore terminal failure appeared after test access")
        advance_stage(
            "rescore",
            "VOID",
            directory=stage_directory,
            gpu_time_path=gpu_time_path,
            terminal_failure_sha256=failure_digest,
        )
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result_schema(result)
        if (
            result.get("final_token") != "VOID"
            or result.get("adjudication", {}).get("reason") != failure["reason"]
            or result.get("protocol", {}).get("rescore_terminal_failure_sha256")
            != failure_digest
        ):
            raise StageTransitionError(
                "immutable result conflicts with rescore terminal failure"
            )
        return result
    return land_void_result(
        str(failure["reason"]),
        "rescore",
        protocol={
            **dict(protocol or {}),
            "rescore_terminal_failure_sha256": failure_digest,
            "rescore_terminal_failure": failure,
        },
        result_path=result_path,
        stage_directory=stage_directory,
        gpu_time_path=gpu_time_path,
    )


def terminalize_rescore(
    reason: str,
    failure: str | None,
    *,
    details: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    failure_path: Path = RESCORE_TERMINAL_FAILURE_PATH,
    stage_directory: Path = STAGE_LEDGER_DIR,
    result_path: Path = RESULT_PATH,
    gpu_time_path: Path = GPU_TIME_LEDGER_PATH,
) -> NoReturn:
    controlling = persist_rescore_terminal_failure(
        reason, failure, details=details, path=failure_path
    )
    reconcile_rescore_terminal(
        failure_path=failure_path,
        stage_directory=stage_directory,
        result_path=result_path,
        gpu_time_path=gpu_time_path,
        protocol=protocol,
    )
    raise VoidIntegrityError(str(controlling["reason"]), "rescore")


def cap_projection_from_batch_preflight(
    batch_execution_seconds: Sequence[float],
    generation_load_seconds: float,
    original_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if len(batch_execution_seconds) != 2:
        raise RuntimeError("cap projection requires exactly two diagnostic executions")
    retained_generation = float(original_preflight["generation"]["wall_seconds"])
    scoring_wall = float(original_preflight["scoring"]["wall_seconds"])
    scoring_load = float(original_preflight["scoring"]["model_load_seconds"])
    g_b = max(batch_execution_seconds) / RETAINED_SMOKE_CANDIDATE_COUNT
    s = scoring_wall / RETAINED_SMOKE_CANDIDATE_COUNT
    model_load_overhead = generation_load_seconds + scoring_load

    def seconds_for(problem_count: int) -> float:
        return (
            retained_generation
            + g_b * (CANDIDATE_COUNT * problem_count - RETAINED_SMOKE_CANDIDATE_COUNT)
            + s * CANDIDATE_COUNT * problem_count
            + model_load_overhead
        )

    full_seconds = seconds_for(CALIBRATION_COUNT + TEST_COUNT)
    fitting_counts = [
        problem_count
        for problem_count in range(CALIBRATION_COUNT + TEST_COUNT + 1)
        if seconds_for(problem_count) <= CAP_AUTHORIZING_SECONDS
    ]
    maximum_problem_count = max(fitting_counts, default=0)
    if maximum_problem_count >= TEST_COUNT + 128:
        resolved_test = TEST_COUNT
        resolved_calibration = maximum_problem_count - resolved_test
    elif maximum_problem_count >= TEST_COUNT:
        resolved_calibration = 128
        resolved_test = maximum_problem_count - resolved_calibration
    else:
        resolved_calibration = 0
        resolved_test = 0
    if full_seconds <= CAP_AUTHORIZING_SECONDS:
        maximum_problem_count = CALIBRATION_COUNT + TEST_COUNT
        resolved_calibration = CALIBRATION_COUNT
        resolved_test = TEST_COUNT
    resolved_seconds = (
        seconds_for(resolved_calibration + resolved_test)
        if resolved_calibration and resolved_test
        else None
    )
    launch_authorized = (
        maximum_problem_count >= TEST_COUNT
        and resolved_seconds is not None
        and resolved_seconds <= CAP_AUTHORIZING_SECONDS
    )
    return {
        "formula": "H_b(M)=smoke_generation+g_b*(16M-32)+s*16M+model_load_overhead",
        "retained_smoke_generation_wall_seconds": retained_generation,
        "g_b_seconds_per_response": g_b,
        "retained_smoke_scoring_seconds_per_response": s,
        "generation_model_load_seconds": generation_load_seconds,
        "scoring_model_load_seconds": scoring_load,
        "combined_model_load_overhead_seconds": model_load_overhead,
        "authorizing_ceiling_seconds": CAP_AUTHORIZING_SECONDS,
        "operational_headroom_seconds": 900.0,
        "full_bank_problem_count": CALIBRATION_COUNT + TEST_COUNT,
        "full_bank_response_count": (CALIBRATION_COUNT + TEST_COUNT) * CANDIDATE_COUNT,
        "full_bank_projection_seconds": full_seconds,
        "full_bank_projection_hours": full_seconds / 3600,
        "full_bank_fits": full_seconds <= CAP_AUTHORIZING_SECONDS,
        "maximum_problem_count_under_ceiling": maximum_problem_count,
        "resolved_calibration_count": resolved_calibration,
        "resolved_test_count": resolved_test,
        "resolved_bank_projection_seconds": resolved_seconds,
        "resolved_bank_projection_under_authorizing_ceiling": launch_authorized,
        "canonical_launch_size_authorized": launch_authorized,
    }


def diagnostic_determinism_payload(
    records: Sequence[Mapping[str, Any]], verifier_tokenizer
) -> list[dict[str, Any]]:
    payload = []
    for record in records:
        serialized, _ = verifier_serialization(
            str(record["question"]),
            str(record["scored_response_segment"]),
            verifier_tokenizer,
        )
        payload.append(
            {
                "dataset_index": int(record["dataset_index"]),
                "candidate_ordinal": int(record["candidate_ordinal"]),
                "response": str(record["response"]),
                "stop_reason": str(record["stop_reason"]),
                "generated_tokens": int(record["generated_tokens"]),
                "extracted_answer": record["extracted_answer"],
                "verifier_input_serialization": serialized,
            }
        )
    return payload


def batch_generation_preflight(
    train: Dataset,
    calibration_indices: Sequence[int],
    manifest: Mapping[str, str],
    requested_batch_size: int,
) -> dict[str, Any]:
    if _stage_snapshot().get("probe") == "PASS":
        raise StageTransitionError("probe stage already passed; no further probe is allowed")
    if requested_batch_size not in ELIGIBLE_BATCH_SIZES:
        raise RuntimeError("preflight batch size must be 8 or 16")
    batch8_path = WORK_ROOT / "batch_preflight_8.json"
    if requested_batch_size == 16:
        if not batch8_path.is_file():
            raise StageTransitionError("batch 16 probe requires the one-shot batch 8 probe")
        batch8 = json.loads(batch8_path.read_text(encoding="utf-8"))
        if batch8.get("eligible") and batch8.get("projection", {}).get(
            "full_bank_fits"
        ):
            raise StageTransitionError(
                "batch 8 already fits the full bank; batch 16 is not registered"
            )
    gate_path = WORK_ROOT / f"batch_preflight_{requested_batch_size}.json"
    if gate_path.exists():
        raise RuntimeError(f"batch-{requested_batch_size} probe is one-shot")
    golden = json.loads(SCORER_DETERMINISM_PATH.read_text(encoding="utf-8"))
    if golden.get("status") != "PASS":
        raise RuntimeError("batch preflight is blocked on the verifier golden replay")
    original_preflight = {
        "generation": {
            "model_load_seconds": HISTORICAL_SMOKE_GENERATION_LOAD_SECONDS,
            "wall_seconds": HISTORICAL_SMOKE_GENERATION_WALL_SECONDS,
        },
        "scoring": {
            "model_load_seconds": HISTORICAL_SMOKE_SCORING_LOAD_SECONDS,
            "wall_seconds": HISTORICAL_SMOKE_SCORING_WALL_SECONDS,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("batched generation preflight requires CUDA")
    child_pids_before = sorted(child.pid for child in multiprocessing.active_children())
    torch.cuda.empty_cache()
    load_started = time.perf_counter()
    tokenizer, model = load_generator(manifest, device)
    generation_load_seconds = time.perf_counter() - load_started
    verifier_tokenizer = AutoTokenizer.from_pretrained(
        manifest[f"{PUBLIC_VERIFIER}-repo-id"],
        revision=manifest[f"{PUBLIC_VERIFIER}-tokenizer-revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    executions: list[dict[str, Any]] = []
    execution_records: list[list[dict[str, Any]]] = []
    for execution_index in range(2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        records: list[dict[str, Any]] = []
        stop_audits: list[dict[str, Any]] = []
        with PowerSampler.create() as power:
            for dataset_index in calibration_indices[:RETAINED_SMOKE_PROBLEM_COUNT]:
                for first_ordinal in range(
                    1, CANDIDATE_COUNT + 1, requested_batch_size
                ):
                    batch_records, batch_telemetry = generate_batch(
                        train,
                        "calibration",
                        int(dataset_index),
                        first_ordinal,
                        requested_batch_size,
                        tokenizer,
                        model,
                        device,
                    )
                    records.extend(batch_records)
                    stop_audits.append(batch_telemetry["per_row_stopping"])
        wall_seconds = time.perf_counter() - started
        payload = diagnostic_determinism_payload(records, verifier_tokenizer)
        executions.append(
            {
                "execution_index": execution_index + 1,
                "wall_seconds": wall_seconds,
                "determinism_sha256": sha256_bytes(canonical_json_bytes(payload)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "rows_completed_before_batch_end": sum(
                    int(audit["rows_completed_before_batch_end"])
                    for audit in stop_audits
                ),
                "mixed_row_state_call_count": sum(
                    int(audit["mixed_row_state_call_count"]) for audit in stop_audits
                ),
                "post_completion_suffix_padding_only": all(
                    bool(audit["post_completion_suffix_padding_only"])
                    for audit in stop_audits
                ),
                "power": power.summary(wall_seconds),
            }
        )
        execution_records.append(records)
    first_payload = diagnostic_determinism_payload(
        execution_records[0], verifier_tokenizer
    )
    second_payload = diagnostic_determinism_payload(
        execution_records[1], verifier_tokenizer
    )
    exact_match = first_payload == second_payload
    field_mismatches = []
    if not exact_match:
        for position, (first, second) in enumerate(
            zip(first_payload, second_payload, strict=True)
        ):
            changed = sorted(key for key in first if first[key] != second[key])
            if changed:
                field_mismatches.append(
                    {"candidate_position": position, "fields": changed}
                )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    child_pids_after = sorted(child.pid for child in multiprocessing.active_children())
    process_leak = child_pids_after != child_pids_before
    stopping_pass = all(
        execution["rows_completed_before_batch_end"] > 0
        and execution["post_completion_suffix_padding_only"]
        for execution in executions
    )
    vram_pass = all(
        execution["peak_reserved_bytes"] <= MAX_PREFLIGHT_RESERVED_BYTES
        for execution in executions
    )
    projection = cap_projection_from_batch_preflight(
        [float(execution["wall_seconds"]) for execution in executions],
        generation_load_seconds,
        original_preflight,
    )
    technical_eligibility = exact_match and vram_pass and stopping_pass and not process_leak
    eligible = technical_eligibility and bool(
        projection["resolved_bank_projection_under_authorizing_ceiling"]
    )
    probe_sequence_complete = bool(
        requested_batch_size == 16 or projection["full_bank_fits"]
    )
    prior_eligible = bool(
        requested_batch_size == 16
        and batch8.get("eligible")
        and batch8.get("under_8100_second_projection")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "mode": "diagnostic_duplicate_batch_preflight",
        "batch_size": requested_batch_size,
        "diagnostic_duplicates_enter_bank_or_metrics": False,
        "diagnostic_successor_rows": [int(v) for v in calibration_indices[:2]],
        "historical_timing_only_artifact_sha256": (
            HISTORICAL_SMOKE_TIMING_ARTIFACT_SHA256
        ),
        "historical_responses_or_scores_read": False,
        "successor_scorer_determinism": golden,
        "generation_model_load_seconds": generation_load_seconds,
        "executions": executions,
        "exact_match_all_32_candidates": exact_match,
        "field_mismatches": field_mismatches,
        "vram_limit_bytes": MAX_PREFLIGHT_RESERVED_BYTES,
        "vram_pass": vram_pass,
        "per_row_stopping_pass": stopping_pass,
        "nan_or_cuda_error": False,
        "process_leak": process_leak,
        "checkpoint_inconsistency": False,
        "technical_eligibility": technical_eligibility,
        "under_8100_second_projection": bool(
            projection["resolved_bank_projection_under_authorizing_ceiling"]
        ),
        "eligible": eligible,
        "probe_sequence_complete": probe_sequence_complete,
        "probe_stage_can_pass": probe_sequence_complete
        and (eligible or prior_eligible),
        "projection": projection,
        "status": "PASS" if eligible else "FAIL",
    }
    write_one_shot_gate(gate_path, report)
    if report["probe_stage_can_pass"]:
        advance_stage("probe", "PASS")
    if not eligible and not prior_eligible:
        raise RuntimeError("requested batch size is ineligible")
    return report


def load_cap_resolution() -> dict[str, Any]:
    require_stage("cap_resolution")
    report = json.loads(CAP_RESOLUTION_PATH.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or not report.get("projection", {}).get(
            "resolved_bank_projection_under_authorizing_ceiling"
        )
        or float(report["projection"]["resolved_bank_projection_seconds"])
        > CAP_AUTHORIZING_SECONDS
    ):
        raise RuntimeError("cap resolution is not launch-authorizing")
    if report.get("runner_source_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("cap resolution is not bound to the current runner")
    return report


def select_cap_probe(
    eligible_reports: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str]:
    if not eligible_reports:
        raise RuntimeError("cap resolution requires an eligible under-cap probe")
    full_bank_reports = [
        report for report in eligible_reports if report["projection"]["full_bank_fits"]
    ]
    if full_bank_reports:
        return (
            min(full_bank_reports, key=lambda item: int(item["batch_size"])),
            "smallest_eligible_batch_fitting_full_bank",
        )
    return (
        min(
            eligible_reports,
            key=lambda item: (
                float(item["projection"]["g_b_seconds_per_response"]),
                int(item["batch_size"]),
            ),
        ),
        "smallest_g_b_among_eligible_resized_batches",
    )


def resolve_cap(
    manifest: Mapping[str, str],
) -> dict[str, Any]:
    require_stage("probe")
    if CAP_RESOLUTION_PATH.exists():
        raise RuntimeError("cap resolution is one-shot and already resolved")
    eligible_reports = []
    for batch_size in ELIGIBLE_BATCH_SIZES:
        path = WORK_ROOT / f"batch_preflight_{batch_size}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") == "PASS" and report.get("eligible"):
            eligible_reports.append(report)
    probe, selection_rule = select_cap_probe(eligible_reports)
    projection = probe["projection"]
    calibration_count = int(projection["resolved_calibration_count"])
    test_count = int(projection["resolved_test_count"])
    if calibration_count + test_count < TEST_COUNT:
        raise RuntimeError("cap resolution does not authorize a canonical launch")
    train, cohorts, cohort_evidence = construct_cohorts(
        expose_pilot_test_content=True,
        resolved_calibration_count=calibration_count,
        resolved_test_count=test_count,
    )
    schedule_evidence = validate_schedules(
        cohorts,
        batch_size=int(probe["batch_size"]),
        enforce_registered_full_bank=False,
    )
    slots = provenance_slots(
        train,
        cohorts,
        require_manifest=True,
        allow_pilot_test_access=True,
        validate_bindings=False,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "PASS",
        "stage": "cap_resolution",
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "selected_batch_size": int(probe["batch_size"]),
        "batch_selection_rule": selection_rule,
        "probe_artifact_sha256": sha256_bytes(canonical_json_bytes(probe)),
        "projection": projection,
        "resolved_counts": {
            "calibration": calibration_count,
            "test": test_count,
            "candidate_denominator": (calibration_count + test_count)
            * CANDIDATE_COUNT,
            "test_problem_denominator": test_count,
        },
        "prefix_derivation": {
            "calibration": "first resolved_calibration_count rows of frozen 256-row order",
            "test": "first resolved_test_count rows of frozen 512-row order",
            "candidate_count_per_problem": CANDIDATE_COUNT,
        },
        "cohort": cohort_evidence,
        "resolved_schedules": schedule_evidence,
        "resolved_provenance_slots": slots,
        "retained_gpu_seconds_before_calibration": assert_retained_gpu_cap(),
        "manifest_identity_digest": manifest_identity(manifest)[0],
    }
    write_one_shot_gate(CAP_RESOLUTION_PATH, report)
    write_one_shot_gate(BATCH_PREFLIGHT_PATH, probe)
    advance_stage("cap_resolution", "PASS")
    return report


def freeze_calibration(bank: Mapping[str, Any]) -> dict[str, Any]:
    require_stage("calibration")
    if CALIBRATION_FREEZE_PATH.exists():
        raise RuntimeError("calibration viability gate is one-shot and already resolved")
    if not bank.get("complete_scoring"):
        raise RuntimeError("calibration bank is not completely scored")
    problems = group_records(bank["records"])
    if len(problems) != int(bank["expected_problem_count"]):
        raise RuntimeError(
            "calibration bank does not contain the resolved problem count"
        )
    correct_candidates = sum(bool(record["correct"]) for record in bank["records"])
    incorrect_candidates = len(bank["records"]) - correct_candidates
    argmax = evaluate_policy(problems, 0, include_ledgers=False)["prefixes"]["16"]
    beneficial = int(argmax["first_incorrect_to_selected_correct"]["numerator"])
    harmful = int(argmax["harmful_switch_examples"]["numerator"])
    distinct_scores = len(
        {float(record["verifier_score"]) for record in bank["records"]}
    )
    finite_scores = all(
        math.isfinite(float(record["verifier_score"])) for record in bank["records"]
    )
    viability = {
        "correct_candidates": correct_candidates,
        "incorrect_candidates": incorrect_candidates,
        "calibration_candidate_denominator": len(bank["records"]),
        "verifier_argmax_beneficial_acquisition_events": {
            "numerator": beneficial,
            "eligible_denominator": int(
                argmax["first_incorrect_to_selected_correct"]["denominator"]
            ),
        },
        "verifier_argmax_harmful_switch_examples": {
            "numerator": harmful,
            "resolved_problem_denominator": int(
                argmax["harmful_switch_examples"]["denominator"]
            ),
        },
        "distinct_finite_candidate_scores": distinct_scores if finite_scores else 0,
        "interpretation": "non-degeneracy/headroom only; not verifier accuracy",
        "floors": {
            "correct_candidates_at_least_40": correct_candidates >= 40,
            "incorrect_candidates_at_least_40": incorrect_candidates >= 40,
            "beneficial_acquisitions_at_least_20": beneficial >= 20,
            "harmful_switch_examples_at_least_20": harmful >= 20,
            "at_least_two_distinct_finite_scores": finite_scores
            and distinct_scores >= 2,
        },
    }
    viability["status"] = (
        "PASS"
        if all(viability["floors"].values())
        else "PREFLIGHT_STOP_INSUFFICIENT_CALIBRATION_HEADROOM"
    )
    selected = (
        select_calibration_parameters(problems)
        if viability["status"] == "PASS"
        else {
            "status": "PREFLIGHT_STOP_INSUFFICIENT_CALIBRATION_HEADROOM",
            "reason": "CALIBRATION_VERIFIER_VIABILITY_FLOOR_MISSED",
            "selected": {},
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "calibration_bank_content_sha256": bank_content_digest(bank),
        "verifier_viability": viability,
        "selection": selected,
    }
    payload["freeze_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_one_shot_gate(CALIBRATION_FREEZE_PATH, payload)
    advance_stage(
        "viability", "PASS" if viability["status"] == "PASS" else "STOP"
    )
    return payload


def load_calibration_freeze() -> dict[str, Any]:
    freeze = json.loads(CALIBRATION_FREEZE_PATH.read_text(encoding="utf-8"))
    digest = freeze.pop("freeze_sha256")
    if sha256_bytes(canonical_json_bytes(freeze)) != digest:
        raise RuntimeError("calibration freeze digest changed")
    freeze["freeze_sha256"] = digest
    current_bank = json.loads(CALIBRATION_BANK_PATH.read_text(encoding="utf-8"))
    current_bank["records"] = load_bank_database_records("calibration")
    if freeze["calibration_bank_content_sha256"] != bank_content_digest(current_bank):
        raise RuntimeError("calibration bank changed after threshold freeze")
    return freeze


def outcome_blind_rescore_calibration(
    calibration_indices: Sequence[int], manifest: Mapping[str, str]
) -> dict[str, Any]:
    if RESCORE_TERMINAL_FAILURE_PATH.is_file():
        controlling = load_rescore_terminal_failure()
        reconcile_rescore_terminal()
        raise VoidIntegrityError(str(controlling["reason"]), "rescore")
    require_stage("viability")
    rescore_state = _stage_snapshot().get("rescore")
    if rescore_state == "STARTED":
        reconciled = reconcile_rescore_started(calibration_indices)
        if reconciled is None:
            raise AssertionError("rescore STARTED reconciliation returned no verdict")
        return reconciled
    if rescore_state is not None:
        raise StageTransitionError(f"rescore is already terminal at {rescore_state}")
    if OUTCOME_BLIND_RESCORE_PATH.exists():
        raise RuntimeError("outcome-blind rescore gate is one-shot and already resolved")
    advance_stage("rescore", "STARTED")
    identities = outcome_blind_rescore_identities(calibration_indices)
    comparison_fields = (
        "verifier_input_sha256",
        "verifier_token_ids_sha256",
        "verifier_marker_positions",
        "verifier_step_score_bf16_bits",
        "verifier_score_bf16_bits",
        "verifier_output_digest",
    )
    bank: dict[str, Any] | None = None
    model = None
    tokenizer = None
    mismatches: list[dict[str, Any]] = []
    failure: str | None = None
    terminal_reason: str | None = None
    load_seconds = 0.0
    rescore_seconds = 0.0
    started = time.perf_counter()
    try:
        bank = json.loads(CALIBRATION_BANK_PATH.read_text(encoding="utf-8"))
        bank["records"] = load_bank_database_records("calibration")
        if not bank.get("complete_scoring"):
            raise RuntimeError("outcome-blind rescore requires a complete scored bank")
        by_identity = {
            (int(record["dataset_index"]), int(record["candidate_ordinal"])): record
            for record in bank["records"]
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("outcome-blind scorer rescore requires CUDA")
        load_started = time.perf_counter()
        tokenizer, model = load_verifier(manifest, device)
        load_seconds = time.perf_counter() - load_started
        rescore_started = time.perf_counter()
        for identity in identities:
            original = by_identity.get(identity)
            if original is None:
                raise RuntimeError(f"outcome-blind rescore identity absent: {identity}")
            blind_record = {
                "question": original["question"],
                "scored_response_segment": original["scored_response_segment"],
            }
            rescored = score_one(blind_record, tokenizer, model, device)
            matches = {
                field: rescored.get(field) == original.get(field)
                for field in comparison_fields
            }
            if not all(matches.values()):
                mismatches.append(
                    {
                        "dataset_index": identity[0],
                        "candidate_ordinal": identity[1],
                        "matches": matches,
                    }
                )
        rescore_seconds = time.perf_counter() - rescore_started
        if mismatches:
            terminal_reason = SCORER_RESCORE_VOID
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        terminal_reason = f"VOID_SCORER_RESCORE_INTEGRITY_FAILURE:{type(error).__name__}"
        rescore_seconds = max(0.0, time.perf_counter() - started - load_seconds)
    finally:
        try:
            if model is not None or tokenizer is not None:
                del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as error:
            terminal_reason = (
                f"VOID_SCORER_RESCORE_CLEANUP_FAILURE:{type(error).__name__}"
            )
            failure = f"{type(error).__name__}: {error}"
    try:
        record_retained_gpu_time(
            "rescore", "outcome_blind_integrity_rescore", load_seconds + rescore_seconds
        )
    except Exception as error:
        terminal_reason = (
            error.reason
            if isinstance(error, VoidIntegrityError)
            else f"VOID_SCORER_RESCORE_GPU_LEDGER_FAILURE:{type(error).__name__}"
        )
        failure = f"{type(error).__name__}: {error}"
    passed = terminal_reason is None
    try:
        calibration_digest = bank_content_digest(bank) if bank is not None else None
    except Exception as error:
        passed = False
        terminal_reason = "VOID_SCORER_RESCORE_BANK_DIGEST_FAILURE"
        failure = f"{type(error).__name__}: {error}"
    try:
        cumulative_retained_gpu_seconds: float | str = retained_gpu_seconds()
    except Exception as error:
        passed = False
        terminal_reason = (
            f"VOID_SCORER_RESCORE_GPU_LEDGER_VALIDATION_FAILURE:{type(error).__name__}"
        )
        failure = f"{type(error).__name__}: {error}"
        cumulative_retained_gpu_seconds = NOT_APPLICABLE
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "PASS" if passed else terminal_reason,
        "terminal": not passed,
        "failure": failure,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "calibration_bank_content_sha256": calibration_digest,
        "outcome_blind": True,
        "records_rescored": len(identities),
        "rescore_schedule_sha256": sha256_text(
            "\n".join(f"{index},{ordinal}" for index, ordinal in identities)
        ),
        "comparison_fields": list(comparison_fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "gpu_time": {
            "model_load_seconds": load_seconds,
            "rescore_wall_seconds": rescore_seconds,
            "delta_gpu_seconds": load_seconds + rescore_seconds,
            "cumulative_retained_gpu_seconds": cumulative_retained_gpu_seconds,
        },
        "gold_answers_read": False,
        "correctness_fields_read": False,
        "policy_outputs_read": False,
    }
    if passed:
        report["artifact_sha256"] = _rescore_pass_artifact_digest(report)
    if not passed:
        controlling = persist_rescore_terminal_failure(
            str(terminal_reason),
            failure,
            details={
                "rescore_report": report,
                "outcome_blind_rescore_artifact_written": False,
            },
        )
        try:
            write_one_shot_gate(OUTCOME_BLIND_RESCORE_PATH, report)
        except Exception:
            pass
        reconcile_rescore_terminal()
        raise VoidIntegrityError(str(controlling["reason"]), "rescore")
    try:
        write_one_shot_gate(OUTCOME_BLIND_RESCORE_PATH, report)
        advance_stage("rescore", "PASS")
    except Exception as error:
        terminalize_rescore(
            f"VOID_SCORER_RESCORE_PASS_COMMIT_FAILURE:{type(error).__name__}",
            f"{type(error).__name__}: {error}",
            details={"rescore_report": report},
        )
    return report


def load_outcome_blind_rescore_pass(
    calibration_indices: Sequence[int],
) -> dict[str, Any]:
    if RESCORE_TERMINAL_FAILURE_PATH.is_file():
        controlling = load_rescore_terminal_failure()
        reconcile_rescore_terminal()
        raise VoidIntegrityError(str(controlling["reason"]), "rescore")
    if _stage_snapshot().get("rescore") == "STARTED":
        reconciled = reconcile_rescore_started(calibration_indices)
        if reconciled is None:
            raise AssertionError("rescore STARTED reconciliation returned no verdict")
        return reconciled
    require_stage("rescore")
    try:
        report = validate_rescore_pass_bindings(calibration_indices)
    except Exception as error:
        terminalize_rescore(
            f"VOID_SCORER_RESCORE_BINDING_VALIDATION_FAILURE:{type(error).__name__}",
            f"{type(error).__name__}: {error}",
        )
    return report


def validate_rescore_pass_bindings(
    calibration_indices: Sequence[int],
    *,
    pass_path: Path = OUTCOME_BLIND_RESCORE_PATH,
) -> dict[str, Any]:
    report = load_rescore_pass_artifact(pass_path)
    expected_schedule = sha256_text(
        "\n".join(
            f"{index},{ordinal}"
            for index, ordinal in outcome_blind_rescore_identities(calibration_indices)
        )
    )
    current_bank = json.loads(CALIBRATION_BANK_PATH.read_text(encoding="utf-8"))
    current_bank["records"] = load_bank_database_records("calibration")
    if (
        report.get("rescore_schedule_sha256") != expected_schedule
        or report.get("calibration_bank_content_sha256")
        != bank_content_digest(current_bank)
    ):
        raise VoidIntegrityError("VOID_SCORER_RESCORE_BINDING_MISMATCH", "rescore")
    return report


def reconcile_rescore_started(
    calibration_indices: Sequence[int],
    *,
    failure_path: Path = RESCORE_TERMINAL_FAILURE_PATH,
    pass_path: Path = OUTCOME_BLIND_RESCORE_PATH,
    stage_directory: Path = STAGE_LEDGER_DIR,
    result_path: Path = RESULT_PATH,
    gpu_time_path: Path = GPU_TIME_LEDGER_PATH,
    validate_bindings: bool = True,
) -> dict[str, Any] | None:
    """Resolve a persisted STARTED state without ever rerunning rescore work."""
    if _stage_snapshot(stage_directory).get("rescore") != "STARTED":
        return None
    if failure_path.is_file():
        controlling = load_rescore_terminal_failure(failure_path)
        reconcile_rescore_terminal(
            failure_path=failure_path,
            stage_directory=stage_directory,
            result_path=result_path,
            gpu_time_path=gpu_time_path,
        )
        raise VoidIntegrityError(str(controlling["reason"]), "rescore")
    if pass_path.is_file():
        try:
            report = (
                validate_rescore_pass_bindings(calibration_indices, pass_path=pass_path)
                if validate_bindings
                else load_rescore_pass_artifact(pass_path)
            )
        except Exception as error:
            terminalize_rescore(
                f"VOID_SCORER_RESCORE_BINDING_VALIDATION_FAILURE:{type(error).__name__}",
                f"{type(error).__name__}: {error}",
                failure_path=failure_path,
                stage_directory=stage_directory,
                result_path=result_path,
                gpu_time_path=gpu_time_path,
            )
        advance_stage(
            "rescore",
            "PASS",
            directory=stage_directory,
            gpu_time_path=gpu_time_path,
        )
        return report
    terminalize_rescore(
        "VOID_SCORER_RESCORE_INTERRUPTED",
        "rescore STARTED persisted without a PASS artifact or terminal failure record",
        failure_path=failure_path,
        stage_directory=stage_directory,
        result_path=result_path,
        gpu_time_path=gpu_time_path,
    )


def canonical_evaluate(
    slots: Mapping[str, Any],
    cohort_evidence: Mapping[str, Any],
    schedule_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    require_stage("test", "SCORED")
    if RESULT_PATH.exists():
        raise RuntimeError("canonical immutable result already exists")
    review_binding = json.loads(REVIEW_BINDING_PATH.read_text(encoding="utf-8"))
    if (
        review_binding.get("runner_source_sha256")
        != slots["successor_runner_source_sha256"]
    ):
        raise RuntimeError("independent review is not bound to the current runner")
    cap_resolution = load_cap_resolution()
    repeat_report = json.loads(BATCH_PREFLIGHT_PATH.read_text(encoding="utf-8"))
    calibration_bank = json.loads(CALIBRATION_BANK_PATH.read_text(encoding="utf-8"))
    calibration_bank["records"] = load_bank_database_records("calibration")
    calibration_indices = [
        int(calibration_bank["records"][offset]["dataset_index"])
        for offset in range(0, len(calibration_bank["records"]), CANDIDATE_COUNT)
    ]
    load_outcome_blind_rescore_pass(calibration_indices)
    test_bank = json.loads(TEST_BANK_PATH.read_text(encoding="utf-8"))
    test_bank["records"] = load_bank_database_records("test")
    if not calibration_bank.get("complete_scoring") or not test_bank.get(
        "complete_scoring"
    ):
        raise RuntimeError("both canonical banks must be complete and scored")
    freeze = load_calibration_freeze()
    if freeze["selection"]["status"] != "PASS":
        raise RuntimeError("calibration terminated before test evaluation")
    calibration_problems = group_records(calibration_bank["records"])
    test_problems = group_records(test_bank["records"])
    selected = freeze["selection"]["selected"]
    calibration_results = all_policy_results(calibration_problems, selected)
    test_results = all_policy_results(test_problems, selected)
    bootstrap = paired_bootstrap(test_results)
    permutations = order_permutation_analysis(test_problems, selected)
    decision = adjudicate(test_results, bootstrap, permutations)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "protocol": {
            "public_models": [PUBLIC_GENERATOR, PUBLIC_VERIFIER],
            "slots": dict(slots),
            "cohort": dict(cohort_evidence),
            "schedules": dict(schedule_evidence),
            "cap_resolution": cap_resolution,
            "decoding": {
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "repetition_penalty": REPETITION_PENALTY,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batch_size": int(cap_resolution["selected_batch_size"]),
                "historical_smoke_excluded": True,
                "dtype": "bfloat16",
            },
            "parser": "qualified E1 exact-rational numeric parser imported directly",
            "verifier_aggregation": "minimum_positive_class_probability_over_steps",
            "repeat_determinism": repeat_report,
            "independent_review": review_binding,
            "outcome_blind_rescore": load_outcome_blind_rescore_pass(
                calibration_indices
            ),
            "stage_ledger_before_landing": _stage_snapshot(),
        },
        "calibration": {"freeze": freeze, "policy_results": calibration_results},
        "candidate_records": [*calibration_bank["records"], *test_bank["records"]],
        "policy_results": test_results,
        "bootstrap": bootstrap,
        "permutations": permutations,
        "compute": {
            "calibration": calibration_bank["telemetry"],
            "test": test_bank["telemetry"],
            "cumulative_retained_gpu_seconds": retained_gpu_seconds(),
            "authorizing_ceiling_seconds": CAP_AUTHORIZING_SECONDS,
        },
        "adjudication": decision,
        "claim_language_contract": _claim_language_contract(),
        "final_token": decision["final_token"],
    }
    validate_result_schema(result)
    write_immutable_result(result, RESULT_PATH)
    return result


def synthetic_candidate(
    ordinal: int,
    score: float,
    correct: bool,
    answer: str | None,
) -> dict[str, Any]:
    empty = answer is None and ordinal % 2 == 0
    return {
        "dataset_index": 10_000,
        "candidate_ordinal": ordinal,
        "seed": ordinal,
        "question": "synthetic",
        "gold_answer": "1",
        "response": "" if empty else "synthetic",
        "scored_response_segment": "" if empty else "synthetic",
        "extracted_answer": answer,
        "extraction_source": None if answer is None else "self_test",
        "extraction_segment_stop_reason": "self_test",
        "stop_reason": "end_of_message",
        "generated_tokens": ordinal,
        "prompt_tokens": 1,
        "generation_seconds": 0.001,
        "outcome_category": (
            "correct_numeric"
            if correct
            else "model_empty_non_answer"
            if empty
            else "parser_recognition_failure"
            if answer is None
            else "valid_extracted_incorrect"
        ),
        "correct_numeric": correct,
        "valid_extracted_incorrect": answer is not None and not correct,
        "model_empty_non_answer": empty,
        "parser_recognition_failure": answer is None and not empty,
        "correct": correct,
        "extraction_failed": answer is None,
        "exact_answer_failure": not correct,
        "verifier_score": score,
        "verifier_step_scores": [score],
        "verifier_scored_tokens": ordinal + 10,
        "verifier_seconds": 0.001,
    }


def policy_rule_self_test() -> dict[str, Any]:
    def make(
        scores: Sequence[float], answers: Sequence[str | None]
    ) -> list[dict[str, Any]]:
        padded_scores = [*scores, *([0.0] * (16 - len(scores)))]
        padded_answers = [*answers, *(["z"] * (16 - len(answers)))]
        return [
            synthetic_candidate(
                ordinal,
                score=padded_scores[ordinal - 1],
                correct=padded_answers[ordinal - 1] == "1",
                answer=padded_answers[ordinal - 1],
            )
            for ordinal in range(1, 17)
        ]

    argmax = policy_trace(make([0.5, 0.5, 0.6], ["a", "b", "c"]), 0)
    if argmax["selected_stream"][:3] != [0, 0, 2]:
        raise AssertionError("Policy 0 tie/argmax semantics changed")

    fixed = policy_trace(
        make([0.25, 0.5, 0.76], ["a", "b", "c"]),
        1,
        {"delta": 0.25},
    )
    if fixed["selected_stream"][:3] != [0, 0, 2]:
        raise AssertionError("Policy 1 strict-margin semantics changed")

    multiplicity_candidates = make([0.0, 0.01, 0.13], ["a", "b", "c"])
    multiplicity = policy_trace(
        multiplicity_candidates,
        2,
        {"delta_0": 0.0, "lambda": 0.1},
        order=[1, 3, 2, *range(4, 17)],
    )
    if multiplicity["selected_stream"][1] != 2:
        raise AssertionError("Policy 2 did not use permuted stream position")

    stopped = policy_trace(
        make([0.1, 0.2, 0.9], ["a", "b", "c"]),
        3,
        {"budget": 2},
    )
    if stopped["selected_stream"][:3] != [0, 1, 1]:
        raise AssertionError("Policy 3 global-budget lock semantics changed")

    plurality = policy_trace(
        make([0.1, 0.2, 0.3], ["a", "b", "b"]),
        4,
    )
    if plurality["selected_stream"][:3] != [0, 0, 1]:
        raise AssertionError("Policy 4 plurality/tie/representative semantics changed")

    weighted = policy_trace(
        make([0.6, 0.4, 0.3], ["a", "b", "b"]),
        5,
    )
    if weighted["selected_stream"][:3] != [0, 0, 1]:
        raise AssertionError("Policy 5 weighted-vote semantics changed")
    fallback = policy_trace(make([0.1, 0.2], [None, None]), 5)
    if (
        fallback["selected_stream"][:2] != [0, 1]
        or not fallback["events"][0]["accepted_replacement"]
    ):
        raise AssertionError("Policy 5 invalid-output argmax fallback changed")
    return {
        "policy_0": "PASS",
        "policy_1": "PASS",
        "policy_2": "PASS",
        "policy_3": "PASS",
        "policy_4": "PASS",
        "policy_5": "PASS",
    }


def self_test_fast() -> dict[str, Any]:
    started = time.perf_counter()
    E1.validate_numeric_extraction()
    replay = E1.replay_gsm8k_parser_guard()
    policy_rules = policy_rule_self_test()
    original_pool = list(range(5, EXPECTED_SPLIT_SIZE))
    original_calibration = ranked_indices(
        original_pool, ORIGINAL_CALIBRATION_SELECTION_STRING, CALIBRATION_COUNT
    )
    original_calibration_set = set(original_calibration)
    original_test = ranked_indices(
        [index for index in original_pool if index not in original_calibration_set],
        ORIGINAL_TEST_SELECTION_STRING,
        TEST_COUNT,
    )
    original_selected = original_calibration_set | set(original_test)
    successor_source_pool = [
        index for index in original_pool if index not in original_selected
    ]
    cohorts = {
        "calibration": ranked_indices(
            successor_source_pool, CALIBRATION_SELECTION_STRING, CALIBRATION_COUNT
        ),
    }
    calibration_set = set(cohorts["calibration"])
    test_pool = [
        index for index in successor_source_pool if index not in calibration_set
    ]
    cohorts["test"] = ranked_indices(test_pool, TEST_SELECTION_STRING, TEST_COUNT)
    if (set(cohorts["calibration"]) | set(cohorts["test"])) & original_selected:
        raise AssertionError("successor cohort overlaps original allocation")
    schedules = validate_schedules(cohorts)
    historical_timing = {
        "generation": {"wall_seconds": HISTORICAL_SMOKE_GENERATION_WALL_SECONDS},
        "scoring": {
            "wall_seconds": HISTORICAL_SMOKE_SCORING_WALL_SECONDS,
            "model_load_seconds": HISTORICAL_SMOKE_SCORING_LOAD_SECONDS,
        },
    }
    resized_projection = cap_projection_from_batch_preflight(
        [20.0, 20.0], HISTORICAL_SMOKE_GENERATION_LOAD_SECONDS, historical_timing
    )
    if (
        resized_projection["full_bank_fits"]
        or not resized_projection["canonical_launch_size_authorized"]
        or resized_projection["resolved_test_count"] != TEST_COUNT
        or not 128 <= resized_projection["resolved_calibration_count"] < CALIBRATION_COUNT
        or resized_projection["resolved_bank_projection_seconds"]
        > CAP_AUTHORIZING_SECONDS
    ):
        raise AssertionError("cap-resize formula did not bind an under-ceiling prefix")
    stopped_projection = cap_projection_from_batch_preflight(
        [400.0, 400.0],
        HISTORICAL_SMOKE_GENERATION_LOAD_SECONDS,
        historical_timing,
    )
    if stopped_projection["canonical_launch_size_authorized"]:
        raise AssertionError("over-cap projection incorrectly authorized a launch")
    selected_resize, resize_rule = select_cap_probe(
        [
            {
                "batch_size": 8,
                "projection": {
                    "full_bank_fits": False,
                    "g_b_seconds_per_response": 0.7,
                },
            },
            {
                "batch_size": 16,
                "projection": {
                    "full_bank_fits": False,
                    "g_b_seconds_per_response": 0.5,
                },
            },
        ]
    )
    selected_full, full_rule = select_cap_probe(
        [
            {
                "batch_size": 8,
                "projection": {
                    "full_bank_fits": True,
                    "g_b_seconds_per_response": 0.7,
                },
            },
            {
                "batch_size": 16,
                "projection": {
                    "full_bank_fits": True,
                    "g_b_seconds_per_response": 0.5,
                },
            },
        ]
    )
    if (
        selected_resize["batch_size"] != 16
        or resize_rule != "smallest_g_b_among_eligible_resized_batches"
        or selected_full["batch_size"] != 8
        or full_rule != "smallest_eligible_batch_fitting_full_bank"
    ):
        raise AssertionError("cap probe selection rule drifted")
    problem = [
        synthetic_candidate(
            ordinal,
            score=min(0.99, 0.04 * ordinal + (0.2 if ordinal in (3, 7) else 0)),
            correct=ordinal in (1, 4, 7, 12),
            answer=(
                "1"
                if ordinal in (1, 4, 7, 12)
                else None
                if ordinal in (5, 10)
                else str(ordinal % 3 + 2)
            ),
        )
        for ordinal in range(1, 17)
    ]
    problems = []
    for problem_index in range(24):
        cloned = [dict(row, dataset_index=10_000 + problem_index) for row in problem]
        if problem_index % 3:
            cloned[0]["correct"] = False
            cloned[0]["correct_numeric"] = False
            cloned[0]["valid_extracted_incorrect"] = True
            cloned[0]["extracted_answer"] = "2"
        problems.append(cloned)
    parameters = {
        "policy_1": {"delta": 0.05},
        "policy_2": {"delta_0": 0.02, "lambda": 0.01},
        "policy_3": {"budget": 8},
    }
    first = all_policy_results(problems, parameters)
    second = all_policy_results(problems, parameters)
    first_hash = sha256_bytes(canonical_json_bytes(first))
    if first_hash != sha256_bytes(canonical_json_bytes(second)):
        raise AssertionError("policy code path is nondeterministic")
    for result in first.values():
        for prefix in PREFIXES:
            metric = result["prefixes"][str(prefix)]
            if (
                metric["selected_accuracy"]["numerator"]
                > metric["oracle_coverage"]["numerator"]
            ):
                raise AssertionError("selected accuracy exceeds oracle coverage")
    calibration_problems = []
    for problem_index in range(24):
        fixture = [
            synthetic_candidate(
                ordinal,
                score=(0.10 if ordinal == 1 else 0.90 if ordinal == 2 else 0.20),
                correct=ordinal == 2,
                answer="1" if ordinal == 2 else str(ordinal + 10),
            )
            for ordinal in range(1, 17)
        ]
        calibration_problems.append(
            [dict(row, dataset_index=20_000 + problem_index) for row in fixture]
        )
    calibration = select_calibration_parameters(calibration_problems)
    if calibration["status"] != "PASS" or set(calibration["selected"]) != {
        "policy_1",
        "policy_2",
        "policy_3",
    }:
        raise AssertionError("calibration selection did not exercise the passing path")
    permutations = permutation_schedule()
    if len({tuple(row) for row in permutations}) != PERMUTATION_REPLICATES:
        raise AssertionError("permutation schedule contains duplicates")
    small_positions_a = [bootstrap_positions(i, len(problems)) for i in range(10)]
    small_positions_b = [bootstrap_positions(i, len(problems)) for i in range(10)]
    if small_positions_a != small_positions_b:
        raise AssertionError("bootstrap schedule is nondeterministic")
    directory_path = HERE / f".f1-self-test-{uuid.uuid4().hex}"
    directory_path.mkdir()
    try:
        final = directory_path / "result.json"
        orphan = (
            directory_path / f"{IMMUTABLE_TEMP_PREFIX}orphan{IMMUTABLE_TEMP_SUFFIX}"
        )
        orphan.write_text("orphan", encoding="utf-8")
        removed = scavenge_stale_immutable_temps(directory_path, final)
        if removed != [orphan] or orphan.exists():
            raise AssertionError("immutable temp scavenger failed")
        _assert_temp_namespace_disjoint(final)
        immutable_payload = b'{"complete":true}\n'
        _atomic_no_clobber_bytes(immutable_payload, final)
        try:
            _atomic_no_clobber_bytes(b'{"changed":true}\n', final)
        except RuntimeError:
            pass
        else:
            raise AssertionError("immutable writer allowed an overwrite")
        if final.read_bytes() != immutable_payload:
            raise AssertionError("immutable writer changed the completed artifact")
        gate = directory_path / "one_shot_gate.json"
        gate_payload = {"status": "PASS", "terminal": False}
        write_one_shot_gate(gate, gate_payload)
        try:
            write_one_shot_gate(gate, {"status": "FAIL", "terminal": True})
        except RuntimeError:
            pass
        else:
            raise AssertionError("one-shot integrity gate allowed a rerun")
        if json.loads(gate.read_text(encoding="utf-8")) != gate_payload:
            raise AssertionError("one-shot integrity gate was clobbered")
        skipped_ledger = directory_path / "skipped_stage_ledger"
        try:
            advance_stage("viability", "PASS", directory=skipped_ledger)
        except StageTransitionError:
            pass
        else:
            raise AssertionError("stage ledger allowed a predecessor skip")
        if skipped_ledger.exists() and list(skipped_ledger.iterdir()):
            raise AssertionError("rejected stage skip mutated the ledger")
        valid_ledger = directory_path / "valid_stage_ledger"
        for stage, state in (
            ("probe", "PASS"),
            ("cap_resolution", "PASS"),
            ("calibration", "STARTED"),
            ("calibration", "PASS"),
            ("viability", "PASS"),
            ("rescore", "STARTED"),
            ("rescore", "PASS"),
            ("test", "STARTED"),
            ("test", "SCORED"),
            ("test", "PASS"),
        ):
            advance_stage(stage, state, directory=valid_ledger)
        if _stage_snapshot(valid_ledger) != {
            "probe": "PASS",
            "cap_resolution": "PASS",
            "calibration": "PASS",
            "viability": "PASS",
            "rescore": "PASS",
            "test": "PASS",
        }:
            raise AssertionError("valid stage ledger did not replay exactly")
        void_fixture = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "created_at": utc_now(),
            "protocol": {"failed_stage": "rescore"},
            "calibration": NOT_APPLICABLE,
            "candidate_records": NOT_APPLICABLE,
            "policy_results": NOT_APPLICABLE,
            "bootstrap": NOT_APPLICABLE,
            "permutations": NOT_APPLICABLE,
            "compute": {"retained_gpu_seconds": 1.0},
            "adjudication": {
                "final_token": "VOID",
                "reason": SCORER_RESCORE_VOID,
                "scientific_adjudication": NOT_APPLICABLE,
            },
            "claim_language_contract": _claim_language_contract(),
            "final_token": "VOID",
        }
        validate_result_schema(void_fixture)
        invalid_void = dict(void_fixture, policy_results={})
        try:
            validate_result_schema(invalid_void)
        except ValueError:
            pass
        else:
            raise AssertionError("VOID schema allowed scientific fields without N/A")
        reconciled_stage_states = []
        for prior_rescore_state in (None, "PASS"):
            suffix = "absent" if prior_rescore_state is None else "pass"
            reconcile_ledger = directory_path / f"reconcile_ledger_{suffix}"
            reconcile_gpu = directory_path / f"reconcile_gpu_{suffix}.jsonl"
            reconcile_failure = directory_path / f"reconcile_failure_{suffix}.json"
            reconcile_result = directory_path / f"reconcile_result_{suffix}.json"
            for stage, state in (
                ("probe", "PASS"),
                ("cap_resolution", "PASS"),
                ("calibration", "STARTED"),
                ("calibration", "PASS"),
                ("viability", "PASS"),
            ):
                advance_stage(
                    stage,
                    state,
                    directory=reconcile_ledger,
                    gpu_time_path=reconcile_gpu,
                )
            if prior_rescore_state is not None:
                advance_stage(
                    "rescore",
                    "STARTED",
                    directory=reconcile_ledger,
                    gpu_time_path=reconcile_gpu,
                )
                advance_stage(
                    "rescore",
                    prior_rescore_state,
                    directory=reconcile_ledger,
                    gpu_time_path=reconcile_gpu,
                )
            reconcile_gpu.write_text("{partial-ledger\n", encoding="utf-8")
            failure_record = persist_rescore_terminal_failure(
                "VOID_SELF_TEST_RESCORE_FAILURE",
                "synthetic crash-window fixture",
                path=reconcile_failure,
            )
            first_void = reconcile_rescore_terminal(
                failure_path=reconcile_failure,
                stage_directory=reconcile_ledger,
                result_path=reconcile_result,
                gpu_time_path=reconcile_gpu,
            )
            first_bytes = reconcile_result.read_bytes()
            second_void = reconcile_rescore_terminal(
                failure_path=reconcile_failure,
                stage_directory=reconcile_ledger,
                result_path=reconcile_result,
                gpu_time_path=reconcile_gpu,
            )
            if (
                first_void != second_void
                or reconcile_result.read_bytes() != first_bytes
                or _stage_snapshot(reconcile_ledger).get("rescore") != "VOID"
                or first_void["final_token"] != "VOID"
                or first_void["compute"]["retained_gpu_seconds"] != NOT_APPLICABLE
                or first_void["protocol"]["rescore_terminal_failure_sha256"]
                != failure_record["artifact_sha256"]
            ):
                raise AssertionError("rescore terminal reconciliation is not idempotent")
            reconciled_stage_states.append(suffix)
        interrupted_ledger = directory_path / "interrupted_rescore_ledger"
        interrupted_gpu = directory_path / "interrupted_rescore_gpu.jsonl"
        interrupted_failure = directory_path / "interrupted_rescore_failure.json"
        interrupted_result = directory_path / "interrupted_rescore_result.json"
        for stage, state in (
            ("probe", "PASS"),
            ("cap_resolution", "PASS"),
            ("calibration", "STARTED"),
            ("calibration", "PASS"),
            ("viability", "PASS"),
            ("rescore", "STARTED"),
        ):
            advance_stage(
                stage,
                state,
                directory=interrupted_ledger,
                gpu_time_path=interrupted_gpu,
            )
        try:
            reconcile_rescore_started(
                [],
                failure_path=interrupted_failure,
                pass_path=directory_path / "missing_rescore_pass.json",
                stage_directory=interrupted_ledger,
                result_path=interrupted_result,
                gpu_time_path=interrupted_gpu,
                validate_bindings=False,
            )
        except VoidIntegrityError as error:
            if error.reason != "VOID_SCORER_RESCORE_INTERRUPTED":
                raise AssertionError("mid-rescore interruption landed the wrong VOID")
        else:
            raise AssertionError("mid-rescore interruption was allowed to rerun")
        if _stage_snapshot(interrupted_ledger).get("rescore") != "VOID":
            raise AssertionError("mid-rescore interruption did not persist VOID")
        orphan_pass_ledger = directory_path / "orphan_pass_ledger"
        orphan_pass_gpu = directory_path / "orphan_pass_gpu.jsonl"
        orphan_pass_failure = directory_path / "orphan_pass_failure.json"
        orphan_pass_result = directory_path / "orphan_pass_result.json"
        orphan_pass_path = directory_path / "orphan_pass.json"
        for stage, state in (
            ("probe", "PASS"),
            ("cap_resolution", "PASS"),
            ("calibration", "STARTED"),
            ("calibration", "PASS"),
            ("viability", "PASS"),
            ("rescore", "STARTED"),
        ):
            advance_stage(
                stage,
                state,
                directory=orphan_pass_ledger,
                gpu_time_path=orphan_pass_gpu,
            )
        orphan_pass = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "terminal": False,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        orphan_pass["artifact_sha256"] = _rescore_pass_artifact_digest(orphan_pass)
        write_one_shot_gate(orphan_pass_path, orphan_pass)
        reconciled_pass = reconcile_rescore_started(
            [],
            failure_path=orphan_pass_failure,
            pass_path=orphan_pass_path,
            stage_directory=orphan_pass_ledger,
            result_path=orphan_pass_result,
            gpu_time_path=orphan_pass_gpu,
            validate_bindings=False,
        )
        if (
            reconciled_pass != orphan_pass
            or _stage_snapshot(orphan_pass_ledger).get("rescore") != "PASS"
        ):
            raise AssertionError("orphan PASS artifact did not reconcile without CUDA")
        mutable = directory_path / "bank.json"
        atomic_replace_json(mutable, {"records": [1]})
        atomic_replace_json(mutable, {"records": [1, 2]})
        if json.loads(mutable.read_text(encoding="utf-8"))["records"] != [1, 2]:
            raise AssertionError("resumable atomic replacement failed")
    finally:
        shutil.rmtree(directory_path)
    database_partition = f"self-test-{uuid.uuid4().hex}"
    database_path = bank_database_path(database_partition)
    try:
        initialize_bank_database(database_partition)
        first_record = synthetic_candidate(1, 0.2, False, "2")
        second_record = synthetic_candidate(2, 0.8, True, "1")
        persist_candidate_record(database_partition, 0, first_record, generated=True)
        persist_candidate_record(database_partition, 1, second_record, generated=True)
        first_record["verifier_score"] = 0.3
        first_record["verifier_step_scores"] = [0.3]
        persist_candidate_record(database_partition, 0, first_record, generated=False)
        restored = load_bank_database_records(database_partition)
        if len(restored) != 2 or restored[0]["verifier_score"] != 0.3:
            raise AssertionError("transactional candidate resume failed")
    finally:
        with contextlib.suppress(FileNotFoundError):
            database_path.unlink()
    return {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "qualified_parser_replay": replay,
        "frozen_policy_rule_checks": policy_rules,
        "schedules": schedules,
        "cap_resize_prefix_fixture": resized_projection,
        "over_cap_stop_fixture": stopped_projection,
        "cap_probe_choice_rules": "PASS",
        "six_policy_determinism_sha256": first_hash,
        "calibration_code_path_status": calibration["status"],
        "bootstrap_schedule_repeat_match": True,
        "permutation_unique_count": len({tuple(row) for row in permutations}),
        "atomic_resume_and_scavenger": "PASS",
        "one_shot_gate_rerun_rejected": True,
        "stage_skip_rejected_without_mutation": True,
        "forward_only_stage_ledger": "PASS",
        "rescore_terminal_reconcile_absent_or_pass": reconciled_stage_states,
        "rescore_terminal_reconcile_corrupt_gpu_ledger": "PASS",
        "rescore_started_without_artifact_lands_void": "PASS",
        "rescore_orphan_pass_reconciles_without_cuda": "PASS",
        "void_na_claim_contract_schema": "PASS",
        "transactional_candidate_resume": "PASS",
    }


def require_review_attestation(args: argparse.Namespace) -> dict[str, str]:
    pattern = re.compile(
        r"FINAL_POST_FIX_RUNNER_SHA256=(?P<runner>[0-9a-f]{64});"
        r"DELTA_REVIEW_SHA256=(?P<review>[0-9a-f]{64})"
    )
    match = pattern.fullmatch(args.review_attestation or "")
    if match is None:
        raise RuntimeError(
            "canonical stages require a FINAL_POST_FIX_RUNNER_SHA256/"
            "DELTA_REVIEW_SHA256 attestation token"
        )
    values = match.groupdict()
    current = sha256_file(Path(__file__).resolve())
    if values["runner"] != current:
        raise RuntimeError(
            f"review token names runner {values['runner']}, current runner is {current}"
        )
    registered = registered_slot_value("successor_runner_source_sha256")
    if registered != current:
        raise RuntimeError("current runner hash is not bound in the canonical registration")
    return values


def bind_review_attestation(
    attestation: Mapping[str, str],
    slots: Mapping[str, Any],
) -> dict[str, Any]:
    binding = {
        "schema_version": SCHEMA_VERSION,
        "delta_review_sha256": attestation["review"],
        "runner_source_sha256": slots["successor_runner_source_sha256"],
        "manifest_identity_digest": slots["successor_manifest_identity_digest"],
        "attestation_format": (
            "FINAL_POST_FIX_RUNNER_SHA256=<runner>;DELTA_REVIEW_SHA256=<review>"
        ),
    }
    if REVIEW_BINDING_PATH.is_file():
        existing = json.loads(REVIEW_BINDING_PATH.read_text(encoding="utf-8"))
        if existing != binding:
            raise RuntimeError("independent review binding changed between stages")
    else:
        write_one_shot_gate(REVIEW_BINDING_PATH, binding)
    return binding


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cohort-only", action="store_true")
    modes.add_argument("--self-test-fast", action="store_true")
    modes.add_argument("--scorer-determinism", action="store_true")
    modes.add_argument(
        "--scorer-determinism-worker", action="store_true", help=argparse.SUPPRESS
    )
    modes.add_argument("--batch-preflight", type=int, choices=ELIGIBLE_BATCH_SIZES)
    modes.add_argument("--resolve-cap", action="store_true")
    modes.add_argument("--rescore", action="store_true")
    modes.add_argument("--generate", choices=("calibration", "test"))
    modes.add_argument("--score", choices=("calibration", "test"))
    modes.add_argument("--evaluate", choices=("calibration", "full"))
    parser.add_argument("--review-attestation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runtime()
    scavenge_stale_immutable_temps(RESULT_PATH.parent, RESULT_PATH)
    if args.self_test_fast:
        print(json.dumps(self_test_fast(), indent=2, allow_nan=False))
        return 0
    if args.scorer_determinism_worker:
        report = successor_scorer_fixture_load(
            parse_manifest(), load_ordinal=3, process_ordinal=2
        )
        print(json.dumps(report, separators=(",", ":"), allow_nan=False))
        return 0
    if args.scorer_determinism:
        print(json.dumps(run_scorer_determinism_gate(), indent=2, allow_nan=False))
        return 0
    if args.cohort_only:
        train, cohorts, cohort_evidence = construct_cohorts(
            expose_pilot_test_content=True
        )
        schedule_evidence = validate_schedules(cohorts)
        slots = provenance_slots(
            train,
            cohorts,
            require_manifest=False,
            allow_pilot_test_access=True,
        )
        print(
            json.dumps(
                {
                    "cohort": cohort_evidence,
                    "schedules": schedule_evidence,
                    "provenance_slots": slots,
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    attestation = require_review_attestation(args)
    if RESCORE_TERMINAL_FAILURE_PATH.is_file():
        controlling = load_rescore_terminal_failure()
        reconcile_rescore_terminal()
        raise VoidIntegrityError(str(controlling["reason"]), "rescore")
    if args.batch_preflight or args.resolve_cap:
        train, cohorts, cohort_evidence = construct_cohorts(
            expose_pilot_test_content=False
        )
        schedule_evidence = validate_schedules(cohorts)
        slots = provenance_slots(
            train,
            cohorts,
            require_manifest=True,
            allow_pilot_test_access=False,
        )
        manifest = parse_manifest()
        bind_review_attestation(attestation, slots)
        if args.batch_preflight:
            print(
                json.dumps(
                    batch_generation_preflight(
                        train,
                        cohorts["calibration"],
                        manifest,
                        args.batch_preflight,
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        print(json.dumps(resolve_cap(manifest), indent=2, allow_nan=False))
        return 0

    cap_resolution = load_cap_resolution()
    calibration_count = int(cap_resolution["resolved_counts"]["calibration"])
    test_count = int(cap_resolution["resolved_counts"]["test"])
    batch_size = int(cap_resolution["selected_batch_size"])
    if _stage_snapshot().get("rescore") == "STARTED":
        reconciled_rescore = reconcile_rescore_started(
            cap_resolution["cohort"]["resolved_ordered_indices"]["calibration"]
        )
        if args.rescore and reconciled_rescore is not None:
            print(json.dumps(reconciled_rescore, indent=2, allow_nan=False))
            return 0
    pilot_test_mode = (
        args.generate == "test" or args.score == "test" or args.evaluate == "full"
    )
    if pilot_test_mode:
        load_calibration_freeze()
        try:
            load_outcome_blind_rescore_pass(
                cap_resolution["cohort"]["resolved_ordered_indices"]["calibration"]
            )
        except VoidIntegrityError as error:
            if not RESULT_PATH.exists():
                land_void_result(error.reason, error.stage)
            raise
    train, cohorts, cohort_evidence = construct_cohorts(
        expose_pilot_test_content=pilot_test_mode,
        resolved_calibration_count=calibration_count,
        resolved_test_count=test_count,
    )
    schedule_evidence = validate_schedules(
        cohorts,
        batch_size=batch_size,
        enforce_registered_full_bank=False,
    )
    if schedule_evidence != cap_resolution["resolved_schedules"]:
        raise RuntimeError("cap-resolved schedule binding changed")
    slots = provenance_slots(
        train,
        cohorts,
        require_manifest=True,
        allow_pilot_test_access=pilot_test_mode,
    )
    manifest = parse_manifest()
    bind_review_attestation(attestation, slots)
    protocol_for_void = {
        "slots": dict(slots),
        "cap_resolution": cap_resolution,
        "cohort": dict(cohort_evidence),
        "schedules": dict(schedule_evidence),
    }
    try:
        if args.rescore:
            print(
                json.dumps(
                    outcome_blind_rescore_calibration(
                        cohorts["calibration"], manifest
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if args.generate:
            if args.generate == "calibration":
                require_stage("cap_resolution")
                ensure_stage_started("calibration")
            else:
                require_stage("rescore")
                load_outcome_blind_rescore_pass(cohorts["calibration"])
                ensure_stage_started("test")
            bank, telemetry = generate_bank_prefix(
                train,
                args.generate,
                cohorts[args.generate],
                slots,
                manifest,
                batch_size=batch_size,
            )
            print(
                json.dumps(
                    {"record_count": len(bank["records"]), "telemetry": telemetry},
                    indent=2,
                )
            )
            return 0
        if args.score:
            if args.score == "calibration":
                require_stage("calibration", "STARTED")
            else:
                require_stage("test", "STARTED")
                load_outcome_blind_rescore_pass(cohorts["calibration"])
            bank, telemetry = score_bank_prefix(
                args.score, cohorts[args.score], slots, manifest
            )
            if args.score == "calibration" and bank.get("complete_scoring"):
                advance_stage("calibration", "PASS")
            elif args.score == "test" and bank.get("complete_scoring"):
                advance_stage("test", "SCORED")
            print(
                json.dumps(
                    {"record_count": len(bank["records"]), "telemetry": telemetry},
                    indent=2,
                )
            )
            return 0
        if args.evaluate == "calibration":
            require_stage("calibration")
            bank = load_or_create_bank("calibration", cohorts["calibration"], slots)
            print(json.dumps(freeze_calibration(bank), indent=2, allow_nan=False))
            return 0
        if args.evaluate == "full":
            require_stage("test", "SCORED")
            load_outcome_blind_rescore_pass(cohorts["calibration"])
            result = canonical_evaluate(slots, cohort_evidence, schedule_evidence)
            advance_stage("test", "PASS")
            print(
                json.dumps(
                    {"final_token": result["final_token"], "result": str(RESULT_PATH)},
                    indent=2,
                )
            )
            return 0
    except VoidIntegrityError as error:
        if not RESULT_PATH.exists():
            land_void_result(
                error.reason,
                error.stage,
                protocol=protocol_for_void,
            )
        raise
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
