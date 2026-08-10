"""Frozen line-07 Best-of-N safe-selection pilot runner.

This is a provenance runner, not a landing command.  The fast paths are:

* ``--cohort-only``: validate every dataset/cohort/provenance binding without
  loading either model;
* ``--self-test-fast``: exercise schedules, all six policies, calibration,
  statistics, schema checks, resume logic, and immutable writing on synthetic
  data;
* ``--smoke``: retain the first two canonical calibration problems, all 16
  registered candidates, and their verifier scores in the ignored resumable
  bank while measuring throughput and memory.

Canonical generation/scoring/evaluation stages require an independent-review
attestation.  Test generation additionally requires a complete, frozen
calibration bank and selected-parameter digest.  Completed immutable evidence
is never overwritten.

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
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from datasets import Dataset, disable_progress_bar, load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)
from transformers.utils import logging as transformers_logging  # noqa: E402


SCHEMA_VERSION = "1.0.0"
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
PREFIXES = (1, 2, 4, 8, 16)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 0
REPETITION_PENALTY = 1.0
CALIBRATION_SELECTION_STRING = (
    "bon-safe-selection-base-c-gsm8k-train-calibration-v1-2026-08-10"
)
TEST_SELECTION_STRING = "bon-safe-selection-base-c-gsm8k-train-test-v1-2026-08-10"
GENERATION_SEED_STRING = "bon-safe-selection-base-c-generation-seeds-v1-2026-08-10"
BOOTSTRAP_STRING = "bon-safe-selection-paired-bootstrap-v1-2026-08-10"
PERMUTATION_STRING = "bon-safe-selection-order-permutations-v1-2026-08-10"
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 1_000
DELTA_GRID = (0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
LAMBDA_GRID = (0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100)
BUDGET_GRID = (1, 2, 4, 8, 16)
FINAL_TOKENS = {"CONFIRM", "KILL", "VOID"}

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
WORK_ROOT = HF_HOME / "line07_safe_selection"
CALIBRATION_BANK_PATH = WORK_ROOT / "calibration_bank.json"
TEST_BANK_PATH = WORK_ROOT / "test_bank.json"
CALIBRATION_FREEZE_PATH = WORK_ROOT / "calibration_freeze.json"
PREFLIGHT_PATH = WORK_ROOT / "preflight.json"
REPEAT_DETERMINISM_PATH = WORK_ROOT / "repeat_determinism.json"
REVIEW_BINDING_PATH = WORK_ROOT / "independent_review_binding.json"
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
    *, expose_pilot_test_content: bool = True
) -> tuple[Dataset, dict[str, list[int]], dict[str, Any]]:
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
    calibration_pool = list(range(5, EXPECTED_SPLIT_SIZE))
    assert len(calibration_pool) == 7_468
    assert_hash("calibration_pool", comma_hash(calibration_pool))
    calibration = ranked_indices(
        calibration_pool, CALIBRATION_SELECTION_STRING, CALIBRATION_COUNT
    )
    assert_hash("calibration_ordered_indices", comma_hash(calibration))
    assert_hash("calibration_sorted_indices", comma_hash(sorted(calibration)))
    calibration_hash = assert_hash(
        "calibration_content", gsm_rows_hash(train, calibration)
    )

    calibration_set = set(calibration)
    test_pool = [index for index in calibration_pool if index not in calibration_set]
    assert len(test_pool) == 7_212
    assert_hash("test_pool", comma_hash(sorted(test_pool)))
    test = ranked_indices(test_pool, TEST_SELECTION_STRING, TEST_COUNT)
    assert_hash("test_ordered_indices", comma_hash(test))
    assert_hash("test_sorted_indices", comma_hash(sorted(test)))
    test_hash = (
        assert_hash("test_content", gsm_rows_hash(train, test))
        if expose_pilot_test_content
        else EXPECTED_HASHES["test_content"]
    )
    assert_hash("calibration_then_test", comma_hash([*calibration, *test]))

    selected_set = calibration_set | set(test)
    unallocated = [index for index in calibration_pool if index not in selected_set]
    assert len(unallocated) == 6_700
    assert_hash("unallocated_indices", comma_hash(unallocated))
    unallocated_hash = assert_hash(
        "unallocated_content", gsm_rows_hash(train, unallocated)
    )

    if len(set(calibration)) != CALIBRATION_COUNT:
        raise RuntimeError("calibration cohort is not unique")
    if len(set(test)) != TEST_COUNT:
        raise RuntimeError("test cohort is not unique")
    if calibration_set & set(test):
        raise RuntimeError("calibration/test cohort overlap")
    if selected_set & set(DEMONSTRATION_INDICES):
        raise RuntimeError("selected cohort overlaps demonstrations")

    registry = validate_prior_consumption_registry(dataset)
    prior_questions = registry.pop("normalized_questions")
    calibration_questions = [
        normalized_question(train[i]["question"]) for i in calibration
    ]
    if len(set(calibration_questions)) != CALIBRATION_COUNT:
        raise RuntimeError("calibration normalized questions are not unique")
    if set(calibration_questions) & prior_questions:
        raise RuntimeError("calibration questions overlap prior consumption")
    if expose_pilot_test_content:
        test_questions = [normalized_question(train[i]["question"]) for i in test]
        if len(set(test_questions)) != TEST_COUNT:
            raise RuntimeError("test normalized questions are not unique")
        if set(calibration_questions) & set(test_questions):
            raise RuntimeError("calibration/test normalized-question overlap")
        if set(test_questions) & prior_questions:
            raise RuntimeError("test questions overlap prior consumption")

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
            "unallocated": len(unallocated),
            "calibration_test_overlap": 0,
            "selected_demonstration_overlap": 0,
            "calibration_prior_question_overlap": 0,
            "test_prior_question_overlap": (
                0 if expose_pilot_test_content else "registered_not_reaccessed"
            ),
        },
        "pilot_test_content_revalidated_in_this_process": expose_pilot_test_content,
        "hashes": {
            name: EXPECTED_HASHES[name]
            for name in (
                "calibration_pool",
                "calibration_ordered_indices",
                "calibration_sorted_indices",
                "test_pool",
                "test_ordered_indices",
                "test_sorted_indices",
                "calibration_then_test",
                "unallocated_indices",
            )
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


def generation_schedule(cohorts: Mapping[str, Sequence[int]]) -> list[int]:
    return [
        generation_seed(index, ordinal)
        for partition in ("calibration", "test")
        for index in cohorts[partition]
        for ordinal in range(1, CANDIDATE_COUNT + 1)
    ]


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


def validate_schedules(cohorts: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    seeds = generation_schedule(cohorts)
    seed_hash = comma_hash(seeds)
    assert len(seeds) == 12_288
    assert_hash("generation_schedule", seed_hash)
    permutations = permutation_schedule()
    permutation_hash = sha256_text(
        "\n".join(",".join(str(value) for value in row) for row in permutations)
    )
    assert_hash("permutation_schedule", permutation_hash)
    return {
        "generation_seed_count": len(seeds),
        "generation_schedule_sha256": seed_hash,
        "permutation_count": len(permutations),
        "permutation_schedule_sha256": permutation_hash,
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
            "huggingface-hub-version",
            "safetensors-version",
            "cuda-version",
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
    pattern = re.compile(
        rf"^- `{re.escape(name)}`: `([0-9a-f]{{64}})`$", re.MULTILINE
    )
    values = pattern.findall(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if not values:
        return None
    if len(set(values)) != 1:
        raise RuntimeError(f"registration contains conflicting values for {name}")
    return values[-1]


def provenance_slots(
    train: Dataset,
    cohorts: Mapping[str, Sequence[int]],
    *,
    require_manifest: bool,
    allow_pilot_test_access: bool,
) -> dict[str, str | None]:
    slots: dict[str, str | None] = {
        "dataset_train_split_fingerprint": train._fingerprint,
        "normalized_prior_consumed_questions_sha256": EXPECTED_HASHES[
            "prior_questions"
        ],
        "demonstrations_content_sha256": EXPECTED_HASHES["demonstrations_content"],
        "calibration_selected_row_content_sha256": EXPECTED_HASHES[
            "calibration_content"
        ],
        "test_selected_row_content_sha256": EXPECTED_HASHES["test_content"],
        "remaining_unallocated_row_content_sha256": EXPECTED_HASHES[
            "unallocated_content"
        ],
        "prompt_serialization_sha256": None,
        "parser_source_sha256": sha256_text(E1.parser_source_text()),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "local_manifest_identity_digest": None,
    }
    if LOCAL_MANIFEST.is_file():
        entries = parse_manifest()
        identity_digest, _ = manifest_identity(entries)
        slots["local_manifest_identity_digest"] = identity_digest
        if allow_pilot_test_access:
            slots["prompt_serialization_sha256"] = prompt_serialization_hash(
                train, cohorts, entries
            )
        else:
            slots["prompt_serialization_sha256"] = registered_slot_value(
                "prompt_serialization_sha256"
            )
    elif require_manifest:
        raise RuntimeError("manifest-bound provenance slots cannot be filled")

    registered_slots = {
        name: registered_slot_value(name)
        for name in (
            "prompt_serialization_sha256",
            "parser_source_sha256",
            "runner_source_sha256",
            "local_manifest_identity_digest",
        )
    }
    if require_manifest or any(registered_slots.values()):
        for name, registered in registered_slots.items():
            if registered is None:
                raise RuntimeError(f"registration slot remains unfilled: {name}")
            if slots[name] != registered:
                raise RuntimeError(
                    f"registered provenance drift for {name}: "
                    f"expected {registered}, got {slots[name]}"
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
    model = AutoModelForCausalLM.from_pretrained(
        identifier,
        revision=revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    return tokenizer, model


def load_verifier(manifest: Mapping[str, str], device: torch.device):
    identifier = manifest[f"{PUBLIC_VERIFIER}-repo-id"]
    revision = manifest[f"{PUBLIC_VERIFIER}-revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        revision=manifest[f"{PUBLIC_VERIFIER}-tokenizer-revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        identifier,
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModel.from_pretrained(
        identifier,
        revision=revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        config=config,
    ).to(device)
    model.eval()
    return tokenizer, model


def _stop_metadata(
    generated_ids: torch.Tensor, criteria: Any, tokenizer
) -> tuple[int, int, str]:
    token_ids = [int(token_id) for token_id in generated_ids.tolist()]
    eos = tokenizer.eos_token_id
    eos_ids = (
        set(int(value) for value in eos)
        if isinstance(eos, (list, tuple))
        else {int(eos)}
    )
    boundary_count = criteria.boundary_token_counts[0]
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
        "verifier_scored_tokens": None,
        "verifier_seconds": None,
    }


def score_one(
    record: Mapping[str, Any], tokenizer, model, device: torch.device
) -> dict[str, Any]:
    updated = dict(record)
    response = str(record["scored_response_segment"])
    if not response.strip():
        updated.update(
            verifier_score=0.0,
            verifier_step_scores=[],
            verifier_scored_tokens=0,
            verifier_seconds=0.0,
            verifier_input_sha256=sha256_text(""),
        )
        return updated
    serialized, steps = verifier_serialization(
        str(record["question"]), response, tokenizer
    )
    input_ids = tokenizer.encode(serialized, return_tensors="pt").to(device)
    step_sep_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(step_sep_ids) != 1:
        raise RuntimeError("verifier step marker is not one token")
    mask = input_ids == step_sep_ids[0]
    if int(mask.sum().item()) != len(steps):
        raise RuntimeError(
            "verifier step-marker count differs from reasoning-step count"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = probabilities[0][mask[0]]
        if selected.shape[-1] != 2:
            raise RuntimeError("verifier output does not expose two-class step logits")
        step_scores = selected[:, 1].detach().cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if len(step_scores) != len(steps):
        raise RuntimeError("verifier produced the wrong number of step scores")
    if not step_scores or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in step_scores
    ):
        raise RuntimeError(
            "verifier returned missing, nonfinite, or out-of-range scores"
        )
    updated.update(
        verifier_score=float(min(step_scores)),
        verifier_step_scores=[float(value) for value in step_scores],
        verifier_scored_tokens=int(input_ids.numel()),
        verifier_seconds=seconds,
        verifier_input_sha256=sha256_text(serialized),
    )
    return updated


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
        expected = bank_template(partition, indices, slots)
        for field in (
            "schema_version",
            "experiment_id",
            "partition",
            "canonical_indices_sha256",
            "expected_problem_count",
            "expected_candidate_count",
            "protocol_slots",
        ):
            if bank.get(field) != expected[field]:
                raise RuntimeError(f"resumable bank binding changed at {field}")
        bank["records"] = load_bank_database_records(partition)
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
    for record in bank["records"]:
        if int(record["seed"]) != generation_seed(
            int(record["dataset_index"]), int(record["candidate_ordinal"])
        ):
            raise RuntimeError("resumable bank seed drift")


def generate_bank_prefix(
    train: Dataset,
    partition: str,
    indices: Sequence[int],
    slots: Mapping[str, Any],
    manifest: Mapping[str, str],
    *,
    problem_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_indices = list(indices[:problem_limit] if problem_limit else indices)
    bank = load_or_create_bank(partition, indices, slots)
    validate_bank_records(bank, indices)
    target_count = len(selected_indices) * CANDIDATE_COUNT
    if len(bank["records"]) >= target_count and problem_limit is not None:
        telemetry = bank.get("telemetry", {}).get("generation")
        if not telemetry:
            raise RuntimeError("generated smoke prefix has no bound telemetry")
        return bank, telemetry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("generation preflight requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer, model = load_generator(manifest, device)
    load_seconds = time.perf_counter() - load_started
    generation_started = time.perf_counter()
    with PowerSampler.create() as power:
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
            persist_candidate_record(partition, offset, record, generated=True)
    generation_seconds = time.perf_counter() - generation_started
    total_generated_tokens = sum(
        int(row["generated_tokens"]) for row in bank["records"][:target_count]
    )
    active_seconds = sum(
        float(row["generation_seconds"]) for row in bank["records"][:target_count]
    )
    telemetry = {
        "model_load_seconds": load_seconds,
        "wall_seconds": generation_seconds,
        "active_generation_seconds": active_seconds,
        "generated_tokens": total_generated_tokens,
        "tokens_per_second": total_generated_tokens / active_seconds
        if active_seconds
        else None,
        "responses_per_second": target_count / active_seconds
        if active_seconds
        else None,
        "cuda": cuda_telemetry(),
        "power": power.summary(active_seconds),
    }
    bank["telemetry"]["generation"] = telemetry
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
    if problem_limit is not None and all(
        bank["records"][offset].get("verifier_score") is not None
        for offset in range(target_count)
    ):
        telemetry = bank.get("telemetry", {}).get("scoring")
        if not telemetry:
            raise RuntimeError("scored smoke prefix has no bound telemetry")
        return bank, telemetry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("verifier scoring requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer, model = load_verifier(manifest, device)
    load_seconds = time.perf_counter() - load_started
    score_started = time.perf_counter()
    with PowerSampler.create() as power:
        for offset in range(target_count):
            if bank["records"][offset].get("verifier_score") is not None:
                continue
            bank["records"][offset] = score_one(
                bank["records"][offset], tokenizer, model, device
            )
            persist_candidate_record(
                partition, offset, bank["records"][offset], generated=False
            )
    score_seconds = time.perf_counter() - score_started
    selected = bank["records"][:target_count]
    active_seconds = sum(float(row["verifier_seconds"]) for row in selected)
    scored_tokens = sum(int(row["verifier_scored_tokens"]) for row in selected)
    telemetry = {
        "model_load_seconds": load_seconds,
        "wall_seconds": score_seconds,
        "active_scoring_seconds": active_seconds,
        "verifier_scored_tokens": scored_tokens,
        "tokens_per_second": scored_tokens / active_seconds if active_seconds else None,
        "cuda": cuda_telemetry(),
        "power": power.summary(active_seconds),
    }
    bank["telemetry"]["scoring"] = telemetry
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
        "schedule_sha256": EXPECTED_HASHES["permutation_schedule"],
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
        "final_token",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"result schema missing fields: {missing}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version changed")
    if result["final_token"] not in FINAL_TOKENS:
        raise ValueError("result final token is invalid")
    records = result["candidate_records"]
    if len(records) != 12_288:
        raise ValueError("result must bind all 12,288 candidate records")
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
        if not isinstance(ledgers, list) or len(ledgers) != TEST_COUNT:
            raise ValueError(f"{policy_key} lacks 512 switch ledgers")
        if any(len(ledger.get("events", [])) != 15 for ledger in ledgers):
            raise ValueError(f"{policy_key} has an incomplete switch ledger")
    if result["adjudication"]["final_token"] != result["final_token"]:
        raise ValueError("adjudication token mismatch")


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


def write_immutable_result(result: Mapping[str, Any], path: Path) -> None:
    validate_result_schema(result)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_no_clobber_bytes(payload, path)


def preflight_projection(
    generation: Mapping[str, Any], response_count: int
) -> dict[str, Any]:
    observed_wall = float(generation["wall_seconds"])
    if response_count <= 0 or observed_wall <= 0:
        raise RuntimeError("preflight has no generation throughput")
    projected_stage = observed_wall / response_count * 12_288
    projected_with_load = projected_stage + float(generation["model_load_seconds"])
    max_token_projection = 12_288 * MAX_NEW_TOKENS / float(
        generation["tokens_per_second"]
    ) + float(generation["model_load_seconds"])
    return {
        "basis": "measured_seconds_per_response_on_2x16_retained_calibration_smoke",
        "observed_responses": response_count,
        "projected_full_bank_seconds": projected_with_load,
        "projected_full_bank_hours": projected_with_load / 3600,
        "max_token_conservative_projection_hours": max_token_projection / 3600,
        "cap_hours": 2.5,
        "within_cap": projected_with_load <= 2.5 * 3600,
        "required_action": (
            "proceed_to_independent_review_before_canonical_run"
            if projected_with_load <= 2.5 * 3600
            else "STOP_AND_APPEND_CONDITIONAL_RESIZE_AMENDMENT"
        ),
    }


def smoke_run(
    train: Dataset,
    cohorts: Mapping[str, list[int]],
    cohort_evidence: Mapping[str, Any],
    schedule_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    slots = provenance_slots(
        train,
        cohorts,
        require_manifest=True,
        allow_pilot_test_access=False,
    )
    manifest = parse_manifest()
    identity_digest, _ = manifest_identity(manifest)
    if identity_digest != slots["local_manifest_identity_digest"]:
        raise RuntimeError("manifest identity digest mismatch")
    bank, generation = generate_bank_prefix(
        train, "calibration", cohorts["calibration"], slots, manifest, problem_limit=2
    )
    bank, scoring = score_bank_prefix(
        "calibration", cohorts["calibration"], slots, manifest, problem_limit=2
    )
    if generation["power"]["sample_count"] == 0:
        raise RuntimeError("generation power telemetry is missing")
    if scoring["power"]["sample_count"] == 0:
        raise RuntimeError("verifier power telemetry is missing")
    records = bank["records"][:32]
    projection = preflight_projection(generation, len(records))
    smoke = {
        "schema_version": SCHEMA_VERSION,
        "mode": "retained_calibration_smoke",
        "status": "PASS" if projection["within_cap"] else "STOP_OVER_CAP",
        "created_at": utc_now(),
        "public_models": [PUBLIC_GENERATOR, PUBLIC_VERIFIER],
        "cohort": cohort_evidence,
        "schedules": schedule_evidence,
        "protocol_slots": slots,
        "problem_count": 2,
        "candidate_count": len(records),
        "generation": generation,
        "scoring": scoring,
        "projection": projection,
        "outcome_taxonomy": dict(Counter(row["outcome_category"] for row in records)),
        "score_range": [
            min(row["verifier_score"] for row in records),
            max(row["verifier_score"] for row in records),
        ],
        "retained_bank_path": str(CALIBRATION_BANK_PATH.relative_to(ROOT)),
        "retained_bank_content_sha256": bank_content_digest(bank),
    }
    atomic_replace_json(PREFLIGHT_PATH, smoke)
    return smoke


def repeat_determinism_check(
    train: Dataset,
    calibration_indices: Sequence[int],
    slots: Mapping[str, Any],
    manifest: Mapping[str, str],
) -> dict[str, Any]:
    bank = load_or_create_bank("calibration", calibration_indices, slots)
    if len(bank["records"]) < 2 * CANDIDATE_COUNT:
        raise RuntimeError(
            "repeat check requires the retained two-problem calibration smoke"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("repeat determinism requires CUDA")
    tokenizer, model = load_generator(manifest, device)
    repeated = []
    for position in range(2 * CANDIDATE_COUNT):
        expected = bank["records"][position]
        actual = generate_one(
            train,
            int(expected["dataset_index"]),
            int(expected["candidate_ordinal"]),
            tokenizer,
            model,
            device,
        )
        repeated.append(actual)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    verifier_tokenizer = AutoTokenizer.from_pretrained(
        manifest[f"{PUBLIC_VERIFIER}-repo-id"],
        revision=manifest[f"{PUBLIC_VERIFIER}-tokenizer-revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    mismatches = []
    for position, (expected, actual) in enumerate(
        zip(bank["records"][:32], repeated, strict=True)
    ):
        expected_serialized, _ = verifier_serialization(
            str(expected["question"]),
            str(expected["scored_response_segment"]),
            verifier_tokenizer,
        )
        actual_serialized, _ = verifier_serialization(
            str(actual["question"]),
            str(actual["scored_response_segment"]),
            verifier_tokenizer,
        )
        fields = {
            "response": expected["response"] == actual["response"],
            "stop_reason": expected["stop_reason"] == actual["stop_reason"],
            "extracted_answer": expected["extracted_answer"]
            == actual["extracted_answer"],
            "verifier_input_serialization": expected_serialized == actual_serialized,
        }
        if not all(fields.values()):
            mismatches.append({"candidate_position": position, "matches": fields})
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "PASS" if not mismatches else "FAIL",
        "diagnostic_duplicate_count": len(repeated),
        "duplicates_enter_metrics": False,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "calibration_bank_content_sha256": bank_content_digest(bank),
    }
    atomic_replace_json(REPEAT_DETERMINISM_PATH, report)
    if mismatches:
        raise RuntimeError("registered repeat-determinism check failed")
    return report


def freeze_calibration(bank: Mapping[str, Any]) -> dict[str, Any]:
    if not bank.get("complete_scoring"):
        raise RuntimeError("calibration bank is not completely scored")
    problems = group_records(bank["records"])
    if len(problems) != CALIBRATION_COUNT:
        raise RuntimeError("calibration bank does not contain 256 problems")
    selected = select_calibration_parameters(problems)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "calibration_bank_content_sha256": bank_content_digest(bank),
        "selection": selected,
    }
    payload["freeze_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    if CALIBRATION_FREEZE_PATH.exists():
        existing = json.loads(CALIBRATION_FREEZE_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "calibration freeze already exists with different content"
            )
    else:
        atomic_replace_json(CALIBRATION_FREEZE_PATH, payload)
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


def canonical_evaluate(
    slots: Mapping[str, Any],
    cohort_evidence: Mapping[str, Any],
    schedule_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if RESULT_PATH.exists():
        raise RuntimeError("canonical immutable result already exists")
    review_binding = json.loads(REVIEW_BINDING_PATH.read_text(encoding="utf-8"))
    if review_binding.get("runner_source_sha256") != slots["runner_source_sha256"]:
        raise RuntimeError("independent review is not bound to the current runner")
    repeat_report = json.loads(REPEAT_DETERMINISM_PATH.read_text(encoding="utf-8"))
    if repeat_report.get("status") != "PASS":
        raise RuntimeError("canonical evaluation requires passing repeat determinism")
    calibration_bank = json.loads(CALIBRATION_BANK_PATH.read_text(encoding="utf-8"))
    calibration_bank["records"] = load_bank_database_records("calibration")
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
            "decoding": {
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "repetition_penalty": REPETITION_PENALTY,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batch_size": 1,
                "dtype": "bfloat16",
            },
            "parser": "qualified E1 exact-rational numeric parser imported directly",
            "verifier_aggregation": "minimum_positive_class_probability_over_steps",
            "repeat_determinism": repeat_report,
            "independent_review": review_binding,
        },
        "calibration": {"freeze": freeze, "policy_results": calibration_results},
        "candidate_records": [*calibration_bank["records"], *test_bank["records"]],
        "policy_results": test_results,
        "bootstrap": bootstrap,
        "permutations": permutations,
        "compute": {
            "calibration": calibration_bank["telemetry"],
            "test": test_bank["telemetry"],
        },
        "adjudication": decision,
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
    cohorts = {
        "calibration": ranked_indices(
            list(range(5, EXPECTED_SPLIT_SIZE)), CALIBRATION_SELECTION_STRING, 256
        ),
    }
    calibration_set = set(cohorts["calibration"])
    test_pool = [
        index for index in range(5, EXPECTED_SPLIT_SIZE) if index not in calibration_set
    ]
    cohorts["test"] = ranked_indices(test_pool, TEST_SELECTION_STRING, 512)
    schedules = validate_schedules(cohorts)
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
        "six_policy_determinism_sha256": first_hash,
        "calibration_code_path_status": calibration["status"],
        "bootstrap_schedule_repeat_match": True,
        "permutation_unique_count": len({tuple(row) for row in permutations}),
        "atomic_resume_and_scavenger": "PASS",
        "transactional_candidate_resume": "PASS",
    }


def require_review_attestation(args: argparse.Namespace) -> None:
    if not args.review_attestation or not re.fullmatch(
        r"[0-9a-f]{64}", args.review_attestation
    ):
        raise RuntimeError(
            "canonical stages require --review-attestation with the independent review SHA-256"
        )


def bind_review_attestation(
    attestation_sha256: str, slots: Mapping[str, Any]
) -> dict[str, Any]:
    binding = {
        "schema_version": SCHEMA_VERSION,
        "independent_review_attestation_sha256": attestation_sha256,
        "runner_source_sha256": slots["runner_source_sha256"],
        "manifest_identity_digest": slots["local_manifest_identity_digest"],
    }
    if REVIEW_BINDING_PATH.is_file():
        existing = json.loads(REVIEW_BINDING_PATH.read_text(encoding="utf-8"))
        if existing != binding:
            raise RuntimeError("independent review binding changed between stages")
    else:
        atomic_replace_json(REVIEW_BINDING_PATH, binding)
    return binding


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cohort-only", action="store_true")
    modes.add_argument("--self-test-fast", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--repeat-determinism", action="store_true")
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
    pilot_test_mode = (
        args.generate == "test"
        or args.score == "test"
        or args.evaluate == "full"
    )
    if pilot_test_mode:
        require_review_attestation(args)
        pre_access_freeze = load_calibration_freeze()
        if pre_access_freeze["selection"]["status"] != "PASS":
            raise RuntimeError(
                "pilot-test content access blocked by calibration outcome"
            )
    expose_pilot_test_content = bool(args.cohort_only or pilot_test_mode)
    train, cohorts, cohort_evidence = construct_cohorts(
        expose_pilot_test_content=expose_pilot_test_content
    )
    schedule_evidence = validate_schedules(cohorts)
    if args.cohort_only:
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
    if args.smoke:
        print(
            json.dumps(
                smoke_run(train, cohorts, cohort_evidence, schedule_evidence),
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    require_review_attestation(args)
    slots = provenance_slots(
        train,
        cohorts,
        require_manifest=True,
        allow_pilot_test_access=expose_pilot_test_content,
    )
    manifest = parse_manifest()
    bind_review_attestation(args.review_attestation, slots)
    if args.repeat_determinism:
        print(
            json.dumps(
                repeat_determinism_check(
                    train, cohorts["calibration"], slots, manifest
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if args.generate:
        if args.generate == "test":
            freeze = load_calibration_freeze()
            if freeze["selection"]["status"] != "PASS":
                raise RuntimeError("test generation blocked by calibration outcome")
            repeat_report = json.loads(
                REPEAT_DETERMINISM_PATH.read_text(encoding="utf-8")
            )
            if repeat_report.get("status") != "PASS":
                raise RuntimeError("test generation blocked by repeat determinism")
            preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
            if preflight.get("protocol_slots") != slots:
                raise RuntimeError("test generation blocked by preflight binding drift")
            if not preflight["projection"]["within_cap"]:
                raise RuntimeError("test generation blocked by over-cap preflight")
        bank, telemetry = generate_bank_prefix(
            train, args.generate, cohorts[args.generate], slots, manifest
        )
        print(
            json.dumps(
                {"record_count": len(bank["records"]), "telemetry": telemetry}, indent=2
            )
        )
        return 0
    if args.score:
        bank, telemetry = score_bank_prefix(
            args.score, cohorts[args.score], slots, manifest
        )
        print(
            json.dumps(
                {"record_count": len(bank["records"]), "telemetry": telemetry}, indent=2
            )
        )
        return 0
    if args.evaluate == "calibration":
        bank = load_or_create_bank("calibration", cohorts["calibration"], slots)
        print(json.dumps(freeze_calibration(bank), indent=2, allow_nan=False))
        return 0
    if args.evaluate == "full":
        result = canonical_evaluate(slots, cohort_evidence, schedule_evidence)
        print(
            json.dumps(
                {"final_token": result["final_token"], "result": str(RESULT_PATH)},
                indent=2,
            )
        )
        return 0
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
