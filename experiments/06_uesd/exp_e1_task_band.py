"""E1 task-band evaluator for the preregistered semantic-ratchet direction.

Base-A protocol (``--dataset gsm8k`` by default):
  - deterministically select 256 examples without replacement from the
    selected dataset's test split using seed 20260809;
  - use the first five training examples from that dataset as a fixed
    five-shot prompt;
  - decode greedily from frozen base-A with at most 256 new tokens;
  - apply a fixed exact-numeric extractor and compute the preregistered gate;
  - create the dataset-specific result artifact, refusing to overwrite
    evidence.

Smoke protocol (``--smoke N``):
  - evaluate the first N examples of the canonical selected cohort through the
    identical data, prompt, generation, and extraction path;
  - report diagnostic metrics to stdout with verdict SMOKE_ONLY;
  - never write the canonical result artifact unless the canonical-cohort
    leakage preflight detects the preregistered terminal VOID condition.

Successor protocol (``--successor-base-b``):
  - use the preregistered revision-pinned GSM8K task only;
  - construct the precommitted 256-example cohort disjoint from base-A;
  - decode from frozen base-B under the same five-shot/parser contract;
  - apply the four-category outcome taxonomy and successor PASS/VOID rules;
  - write only ``exp_e1_task_band_base_b.json`` (or the qualified initial
    parser-miss sidecar), refusing to overwrite evidence.

Repaired parser protocol (canonical ``--parser-attempt repaired`` only):
  - load the exact immutable responses from the initial parser-miss artifact;
  - perform extraction and outcome classification only, with no model load or
    generation;
  - require byte-identical response-set, cohort, prompt, decoding, and stopping
    evidence before a repaired verdict can be written.

The ``--cohort-only`` successor path loads the cached dataset, runs every
cohort/provenance assertion, prints the evidence, and exits before any model
coordinates are read or any generation code is reached.

The exact base-A and base-B identifiers are private repository-local
configuration. This script reads them from the gitignored
``_local_manifest.md`` and never logs or serializes them.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

# Keep dependency output free of private checkpoint identifiers and make smoke
# output machine-readable. These must be set before importing hub clients.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from datasets import Dataset, disable_progress_bar, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from transformers.utils import logging as transformers_logging


EXPERIMENT = "E1 task-band gate"
BASE_A_PUBLIC_NAME = "base-A"
BASE_B_PUBLIC_NAME = "base-B"
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    dataset_id: str
    revision: str
    config: str
    result_filename: str
    initial_parser_miss_filename: str
    fallback: bool
    batch_size: int


DATASET_SPECS = {
    "gsm8k": DatasetSpec(
        key="gsm8k",
        name="GSM8K",
        dataset_id="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        config="main",
        result_filename="exp_e1_task_band.json",
        initial_parser_miss_filename="exp_e1_task_band_initial_parser_miss.json",
        fallback=False,
        batch_size=8,
    ),
    "svamp": DatasetSpec(
        key="svamp",
        name="SVAMP",
        dataset_id="ChilleD/SVAMP",
        revision="5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
        config="default",
        result_filename="exp_e1_task_band_svamp.json",
        initial_parser_miss_filename=(
            "exp_e1_task_band_svamp_initial_parser_miss.json"
        ),
        fallback=True,
        # Do not let padding shape or co-batched prompts change greedy answers.
        batch_size=1,
    ),
}

SEED = 20260809
CANONICAL_SAMPLE_COUNT = 256
DEMONSTRATION_INDICES = (0, 1, 2, 3, 4)
MAX_NEW_TOKENS = 256
EQUIVALENCE_CHECK_COUNT = 2
MAX_SMOKE_RESPONSE_CHARS = 1_200
MAX_VRAM_FRACTION = 0.80

PASS_MIN_CORRECT = 26
PASS_MAX_CORRECT = 217
MIN_CORRECT_POPULATION = 40
MIN_INCORRECT_POPULATION = 40
MAX_EXTRACTION_FAILURE_RATE = 0.05
MAX_FAILURE_COUNT = 12

HERE = Path(__file__).resolve().parent
LOCAL_MANIFEST_PATH = HERE / "_local_manifest.md"

BASE_A_GSM8K_ARTIFACT = HERE / "results" / "exp_e1_task_band.json"
BASE_A_SORTED_INDICES_SHA256 = (
    "0cc16e5a27c42ed8cab155006c25883a2695cac4131150b7e1d61334e912192b"
)
BASE_B_SELECTION_STRING = "semantic-ratchet-base-b-gsm8k-v1-2026-08-09"
BASE_B_ORDERED_INDICES_SHA256 = (
    "670705ea2936f75f0e90a4048d3f5b5ec3a63b42577c0d7a9df87253b77444ff"
)
GSM8K_TEST_SPLIT_SIZE = 1319
BASE_B_RESULT_FILENAME = "exp_e1_task_band_base_b.json"
BASE_B_INITIAL_PARSER_MISS_FILENAME = (
    "exp_e1_task_band_base_b_initial_parser_miss.json"
)

TERMINAL_ASSERTION_REASONS = {
    "cohort_assertion_failure",
    "provenance_assertion_failure",
    "accounting_assertion_failure",
    "determinism_assertion_failure",
}


class TerminalAssertionError(RuntimeError):
    """An amendment-defined canonical failure that must land as terminal VOID."""

    def __init__(
        self,
        reason: str,
        stage: str,
        message: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if reason not in TERMINAL_ASSERTION_REASONS:
            raise ValueError(f"unsupported terminal assertion reason: {reason}")
        super().__init__(message)
        self.reason = reason
        self.stage = stage
        self.public_message = message
        self.evidence = evidence or {}

UNSIGNED_DECIMAL_PATTERN = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
SIGNED_DECIMAL_PATTERN = rf"[-+]?{UNSIGNED_DECIMAL_PATTERN}"
# A single ASCII slash may have horizontal whitespace on either side. Unicode
# fraction/division slashes and line-spanning fractions are deliberately invalid.
NUMBER_PATTERN = (
    rf"{SIGNED_DECIMAL_PATTERN}"
    rf"(?:[ \t]*/[ \t]*{SIGNED_DECIMAL_PATTERN})?"
    r"%?"
)
STRICT_NUMBER_RE = re.compile(rf"\A{NUMBER_PATTERN}\Z")
ANSWER_BOUNDARY_LINE_RE = re.compile(r"(?m)^[ \t]*####[^\r\n]*(?:\r?$)")
ANY_NUMBER_RE = re.compile(NUMBER_PATTERN)
NEW_QUESTION_RE = re.compile(r"(?m)^\s*Question\s*:")
PROMPT_PREAMBLE = (
    "Solve each grade-school math problem step by step. End every answer with "
    "a separate line in exactly this form:\n#### <numeric answer>"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        default="gsm8k",
        help="task-band dataset; defaults to the immutable GSM8K primary path",
    )
    parser.add_argument(
        "--successor-base-b",
        action="store_true",
        help="use the preregistered disjoint GSM8K successor gate for base-B",
    )
    parser.add_argument(
        "--cohort-only",
        action="store_true",
        help=(
            "assert and report the successor cohort without loading a model or "
            "generating responses; requires --successor-base-b"
        ),
    )
    parser.add_argument(
        "--smoke",
        type=int,
        metavar="N",
        help=(
            "run the first N examples of the canonical cohort without writing "
            "the canonical result artifact unless leakage is detected"
        ),
    )
    parser.add_argument(
        "--parser-attempt",
        choices=("initial", "repaired"),
        default="initial",
        help=(
            "mark the canonical parser attempt; 'repaired' re-adjudicates only "
            "the immutable initial-attempt responses and never regenerates"
        ),
    )
    args = parser.parse_args(argv)
    if args.smoke is not None and not 1 <= args.smoke <= CANONICAL_SAMPLE_COUNT:
        parser.error(f"--smoke N must satisfy 1 <= N <= {CANONICAL_SAMPLE_COUNT}")
    if args.smoke is not None and args.parser_attempt != "initial":
        parser.error("--parser-attempt repaired is only valid for the canonical run")
    if args.successor_base_b and args.dataset != "gsm8k":
        parser.error("--successor-base-b is restricted to the frozen GSM8K task")
    if args.cohort_only and not args.successor_base_b:
        parser.error("--cohort-only requires --successor-base-b")
    if args.cohort_only and args.smoke is not None:
        parser.error("--cohort-only and --smoke are mutually exclusive")
    if args.cohort_only and args.parser_attempt != "initial":
        parser.error("--cohort-only does not perform a parser attempt")
    return args


def configure_runtime() -> None:
    disable_progress_bar()
    transformers_logging.set_verbosity_error()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def load_private_model_coordinates(
    public_model_name: str,
) -> tuple[str, str, str | None]:
    if not LOCAL_MANIFEST_PATH.is_file():
        raise RuntimeError("the gitignored local model manifest is missing")

    model_entry_re = re.compile(
        rf"^-\s*`{re.escape(public_model_name)}`:\s*`([^`]+)`\s*$"
    )
    revision_entry_re = re.compile(
        rf"^-\s*`{re.escape(public_model_name)}-revision`:\s*`([0-9a-f]{{40}})`\s*$"
    )
    digest_entry_re = re.compile(
        rf"^-\s*`{re.escape(public_model_name)}-local-content-sha256`:\s*"
        r"`([0-9a-f]{64})`\s*$"
    )

    model_matches = []
    revision_matches = []
    digest_matches = []
    for line in LOCAL_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        model_match = model_entry_re.fullmatch(stripped)
        if model_match:
            model_matches.append(model_match.group(1).strip())
        revision_match = revision_entry_re.fullmatch(stripped)
        if revision_match:
            revision_matches.append(revision_match.group(1))
        digest_match = digest_entry_re.fullmatch(stripped)
        if digest_match:
            digest_matches.append(digest_match.group(1))

    if len(model_matches) != 1 or not model_matches[0]:
        raise RuntimeError(
            f"the local manifest must contain exactly one {public_model_name} entry"
        )
    if len(revision_matches) != 1:
        raise RuntimeError(
            "the local manifest must contain exactly one "
            f"{public_model_name}-revision entry"
        )
    if public_model_name == BASE_B_PUBLIC_NAME and len(digest_matches) != 1:
        raise RuntimeError(
            "the local manifest must contain exactly one base-B content digest"
        )
    return (
        model_matches[0],
        revision_matches[0],
        digest_matches[0] if digest_matches else None,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_content_digest(snapshot_path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in snapshot_path.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("the private checkpoint snapshot is empty")
    for path in files:
        relative_path = path.relative_to(snapshot_path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_local_checkpoint_snapshot(
    model_identifier: str,
    model_revision: str,
    recorded_content_digest: str | None,
    public_model_name: str,
) -> None:
    if public_model_name != BASE_B_PUBLIC_NAME:
        return
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("base-B execution requires HF_HUB_OFFLINE=1")

    configured_home = os.environ.get("HF_HOME")
    if not configured_home:
        raise RuntimeError("base-B execution requires the workspace-local HF_HOME")
    expected_home = (HERE.parent.parent / ".hf_cache").resolve()
    if Path(configured_home).resolve() != expected_home:
        raise RuntimeError("base-B execution requires HF_HOME=<repo>/.hf_cache")

    cache_key = f"models--{model_identifier.replace('/', '--')}"
    snapshot_path = expected_home / "hub" / cache_key / "snapshots" / model_revision
    if not snapshot_path.is_dir():
        raise RuntimeError("the revision-pinned base-B snapshot is absent offline")
    if not recorded_content_digest:
        raise RuntimeError("the base-B local content digest is absent")
    if snapshot_content_digest(snapshot_path) != recorded_content_digest:
        raise RuntimeError("the base-B local snapshot content digest changed")


def result_path(dataset_spec: DatasetSpec, successor_base_b: bool = False) -> Path:
    if successor_base_b:
        return HERE / "results" / BASE_B_RESULT_FILENAME
    return HERE / "results" / dataset_spec.result_filename


def initial_parser_miss_path(
    dataset_spec: DatasetSpec,
    successor_base_b: bool = False,
) -> Path:
    if successor_base_b:
        return HERE / "results" / BASE_B_INITIAL_PARSER_MISS_FILENAME
    return HERE / "results" / dataset_spec.initial_parser_miss_filename


def evaluation_batch_size(
    dataset_spec: DatasetSpec,
    successor_base_b: bool,
) -> int:
    # Base-B is padding-sensitive on GSM8K. Keep the frozen base-A GSM8K
    # batch-8 path intact while forcing the successor through batch-1.
    return 1 if successor_base_b else dataset_spec.batch_size


def uses_repeat_determinism(
    dataset_spec: DatasetSpec,
    successor_base_b: bool,
) -> bool:
    return dataset_spec.key == "svamp" or successor_base_b


def generation_verification_field(
    dataset_spec: DatasetSpec,
    successor_base_b: bool,
) -> str:
    return (
        "repeat_determinism"
        if uses_repeat_determinism(dataset_spec, successor_base_b)
        else "batch_vs_unbatched_equivalence"
    )


def row_question(row: dict[str, Any], dataset_spec: DatasetSpec) -> str:
    if dataset_spec.key == "gsm8k":
        return str(row["question"]).strip()
    if dataset_spec.key == "svamp":
        return " ".join(
            part for part in (str(row["Body"]).strip(), str(row["Question"]).strip())
            if part
        )
    raise RuntimeError("unsupported task-band dataset")


def row_gold_answer(row: dict[str, Any], dataset_spec: DatasetSpec) -> str:
    if dataset_spec.key == "gsm8k":
        return extract_gold_answer(str(row["answer"]))
    if dataset_spec.key == "svamp":
        normalized = normalize_number(str(row["Answer"]))
        if normalized is None:
            raise RuntimeError("an official SVAMP numeric answer is invalid")
        return normalized
    raise RuntimeError("unsupported task-band dataset")


def row_demonstration_answer(
    row: dict[str, Any],
    dataset_spec: DatasetSpec,
) -> str:
    if dataset_spec.key == "gsm8k":
        return str(row["answer"]).strip()
    if dataset_spec.key == "svamp":
        gold = row_gold_answer(row, dataset_spec)
        equation = str(row["Equation"]).strip()
        return f"{equation} = {gold}\n#### {gold}"
    raise RuntimeError("unsupported task-band dataset")


def canonical_test_indices(split_size: int) -> list[int]:
    if split_size < CANONICAL_SAMPLE_COUNT:
        raise RuntimeError("the official test split is smaller than the canonical cohort")
    rng = random.Random(SEED)
    return rng.sample(range(split_size), CANONICAL_SAMPLE_COUNT)


def selection_digest(indices: Sequence[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_base_a_consumed_gsm8k_indices() -> list[int]:
    try:
        artifact = json.loads(BASE_A_GSM8K_ARTIFACT.read_text(encoding="utf-8"))
        indices = sorted(
            int(record["dataset_index"]) for record in artifact["per_example"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("the immutable base-A GSM8K cohort is invalid") from error

    if len(indices) != CANONICAL_SAMPLE_COUNT or len(set(indices)) != len(indices):
        raise RuntimeError("the immutable base-A GSM8K cohort is not 256 unique rows")
    if selection_digest(indices) != BASE_A_SORTED_INDICES_SHA256:
        raise RuntimeError("the immutable base-A GSM8K cohort hash changed")
    return indices


def successor_base_b_test_indices(split_size: int) -> tuple[list[int], dict[str, Any]]:
    if split_size != GSM8K_TEST_SPLIT_SIZE:
        raise RuntimeError("the revision-pinned GSM8K test split size changed")

    consumed = load_base_a_consumed_gsm8k_indices()
    consumed_set = set(consumed)
    pool = [index for index in range(split_size) if index not in consumed_set]
    if len(pool) != 1063:
        raise RuntimeError("the disjoint base-B candidate pool must contain 1063 rows")

    def ranking_key(index: int) -> tuple[bytes, int]:
        payload = (
            f"{DATASET_SPECS['gsm8k'].revision}\n"
            f"{BASE_B_SELECTION_STRING}\n{index}"
        ).encode("ascii")
        return hashlib.sha256(payload).digest(), index

    selected = sorted(pool, key=ranking_key)[:CANONICAL_SAMPLE_COUNT]
    selected_hash = selection_digest(selected)
    overlap_count = len(set(selected) & consumed_set)
    remaining_count = split_size - len(consumed_set | set(selected))
    if len(selected) != CANONICAL_SAMPLE_COUNT:
        raise RuntimeError("the base-B cohort does not contain 256 rows")
    if len(set(selected)) != CANONICAL_SAMPLE_COUNT:
        raise RuntimeError("the base-B cohort contains duplicate rows")
    if overlap_count != 0:
        raise RuntimeError("the base-B cohort overlaps the base-A cohort")
    if remaining_count != 807:
        raise RuntimeError("the reserved final-evaluation pool must contain 807 rows")
    if selected_hash != BASE_B_ORDERED_INDICES_SHA256:
        raise RuntimeError("the base-B ordered cohort hash changed")

    return selected, {
        "method": "sha256_ranked_disjoint_official_test_pool",
        "selection_string": BASE_B_SELECTION_STRING,
        "base_a_consumed_count": len(consumed),
        "base_a_sorted_indices_sha256": selection_digest(consumed),
        "candidate_pool_count": len(pool),
        "canonical_sample_count": len(selected),
        "canonical_unique_count": len(set(selected)),
        "canonical_indices_sha256": selected_hash,
        "overlap_with_base_a_count": overlap_count,
        "remaining_unconsumed_official_test_count": remaining_count,
        "smoke_uses_canonical_prefix": True,
    }


def build_five_shot_messages(
    train_split: Dataset,
    question: str,
    dataset_spec: DatasetSpec,
) -> list[dict[str, str]]:
    if len(train_split) <= max(DEMONSTRATION_INDICES):
        raise RuntimeError("the official training split lacks fixed demonstrations")

    messages = [{"role": "system", "content": PROMPT_PREAMBLE}]
    for index in DEMONSTRATION_INDICES:
        row = train_split[index]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Question: {row_question(row, dataset_spec)}",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Answer:\n{row_demonstration_answer(row, dataset_spec)}"
                    ),
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question.strip()}\nAnswer:",
        }
    )
    return messages


def normalized_question(question: str) -> str:
    return " ".join(question.casefold().split())


def row_provenance(row: dict[str, Any], dataset_spec: DatasetSpec) -> dict[str, Any]:
    if dataset_spec.key == "gsm8k":
        return {
            "question": row["question"],
            "answer": row["answer"],
        }
    if dataset_spec.key == "svamp":
        return {
            field: row[field]
            for field in (
                "ID",
                "Body",
                "Question",
                "Equation",
                "Answer",
                "Type",
                "question_concat",
            )
        }
    raise RuntimeError("unsupported task-band dataset")


def rows_content_digest(
    split: Dataset,
    indices: Sequence[int],
    dataset_spec: DatasetSpec,
) -> str:
    rows = [
        {
            "index": int(index),
            **row_provenance(split[index], dataset_spec),
        }
        for index in indices
    ]
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def leakage_preflight(
    train_split: Dataset,
    test_split: Dataset,
    selected_indices: Sequence[int],
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    demonstration_questions: dict[str, list[int]] = {}
    for index in DEMONSTRATION_INDICES:
        normalized = normalized_question(row_question(train_split[index], dataset_spec))
        demonstration_questions.setdefault(normalized, []).append(index)

    cohort_questions: dict[str, list[dict[str, int]]] = {}
    for cohort_position, dataset_index in enumerate(selected_indices):
        normalized = normalized_question(
            row_question(test_split[dataset_index], dataset_spec)
        )
        cohort_questions.setdefault(normalized, []).append(
            {
                "cohort_position": cohort_position,
                "dataset_index": dataset_index,
            }
        )

    overlapping_questions = sorted(demonstration_questions.keys() & cohort_questions.keys())
    overlap_evidence = [
        {
            "normalized_question": question,
            "demonstration_indices": demonstration_questions[question],
            "cohort_examples": cohort_questions[question],
        }
        for question in overlapping_questions
    ]
    return {
        "status": "FAIL" if overlap_evidence else "PASS",
        "comparison": "normalized_exact_question_text",
        "demonstration_count": len(demonstration_questions),
        "cohort_count": len(cohort_questions),
        "overlap_count": len(overlap_evidence),
        "overlaps": overlap_evidence,
    }


def normalize_number(raw: str) -> str | None:
    """Parse one complete token under the anchored numeric grammar.

    The accepted grammar is a signed decimal, optionally a ratio of two signed
    decimals separated by one ASCII slash with horizontal whitespace, followed
    by at most one percent sign. The percent sign is a presentation suffix and
    is stripped rather than scaled, preserving the recorded ``#### 25%`` GSM8K
    behavior. Fractions are evaluated exactly with :class:`fractions.Fraction`.
    """
    if STRICT_NUMBER_RE.fullmatch(raw) is None:
        return None

    candidate = raw[:-1] if raw.endswith("%") else raw
    parts = re.split(r"[ \t]*/[ \t]*", candidate)
    try:
        exact_parts = [Fraction(part.replace(",", "")) for part in parts]
    except (ValueError, ZeroDivisionError):
        return None

    value = exact_parts[0]
    if len(exact_parts) == 2:
        if exact_parts[1] == 0:
            return None
        value /= exact_parts[1]
    if value == 0:
        return "0"
    if value.denominator == 1:
        return str(value.numerator)

    remaining_denominator = value.denominator
    power_of_two = 0
    power_of_five = 0
    while remaining_denominator % 2 == 0:
        remaining_denominator //= 2
        power_of_two += 1
    while remaining_denominator % 5 == 0:
        remaining_denominator //= 5
        power_of_five += 1
    if remaining_denominator != 1:
        return f"{value.numerator}/{value.denominator}"

    decimal_places = max(power_of_two, power_of_five)
    scaled_numerator = value.numerator * (10**decimal_places // value.denominator)
    sign = "-" if scaled_numerator < 0 else ""
    digits = str(abs(scaled_numerator)).zfill(decimal_places + 1)
    integer_part = digits[:-decimal_places]
    fractional_part = digits[-decimal_places:].rstrip("0")
    return f"{sign}{integer_part}.{fractional_part}"


def numeric_candidate_is_isolated(text: str, start: int, end: int) -> bool:
    """Reject fragments embedded in malformed numeric-looking syntax."""
    slash_characters = "/\u2044\u2215"

    left = start - 1
    while left >= 0 and text[left].isspace():
        left -= 1
    if left >= 0 and text[left] in slash_characters:
        return False

    right = end
    while right < len(text) and text[right].isspace():
        right += 1
    if right < len(text) and text[right] in slash_characters:
        return False

    if start:
        previous = text[start - 1]
        if previous.isalnum() or previous in f"_%{slash_characters}":
            return False
        if previous == ".":
            return False
        if previous == "," and start >= 2 and text[start - 2].isdigit():
            return False

    if end < len(text):
        following = text[end]
        if following.isalnum() or following in f"_%{slash_characters}":
            return False
        if following == "." and end + 1 < len(text) and text[end + 1].isdigit():
            return False
        if following == "," and end + 1 < len(text) and text[end + 1].isdigit():
            return False

    return True


def numeric_candidates(text: str) -> list[str]:
    """Return only complete, isolated candidates; never rescan rejected fragments."""
    return [
        match.group(0)
        for match in ANY_NUMBER_RE.finditer(text)
        if numeric_candidate_is_isolated(text, match.start(), match.end())
    ]


def extract_gold_answer(answer: str) -> str:
    matches = list(ANSWER_BOUNDARY_LINE_RE.finditer(answer))
    if not matches:
        raise RuntimeError("an official GSM8K answer lacks its numeric marker")
    raw_answer = matches[-1].group(0).split("####", maxsplit=1)[1].strip()
    normalized = normalize_number(raw_answer)
    if normalized is None:
        raise RuntimeError("an official GSM8K numeric answer is invalid")
    return normalized


def first_answer_segment(
    response: str,
    generation_stop_reason: str,
) -> tuple[str, str]:
    normalized_response = response.replace("\r\n", "\n")
    question_match = NEW_QUESTION_RE.search(normalized_response)
    question_boundary = question_match.start() if question_match else len(normalized_response)
    before_question = normalized_response[:question_boundary]

    answer_match = ANSWER_BOUNDARY_LINE_RE.search(before_question)
    if answer_match:
        return before_question[: answer_match.end()], "first_final_answer_line"
    if question_match:
        return before_question, "new_question_boundary"
    return before_question, generation_stop_reason


def extract_predicted_answer(
    response: str,
    generation_stop_reason: str,
) -> tuple[str | None, str | None, str, str]:
    segment, segment_stop_reason = first_answer_segment(
        response=response,
        generation_stop_reason=generation_stop_reason,
    )
    answer_match = ANSWER_BOUNDARY_LINE_RE.search(segment)
    if answer_match:
        raw_answer = answer_match.group(0).split("####", maxsplit=1)[1].strip()
        normalized = normalize_number(raw_answer)
        if normalized is None:
            return None, None, segment, segment_stop_reason
        return normalized, "first_final_hash_answer", segment, segment_stop_reason

    for raw in reversed(numeric_candidates(segment)):
        normalized = normalize_number(raw)
        if normalized is not None:
            return normalized, "last_numeric_token_in_first_segment", segment, segment_stop_reason
    return None, None, segment, segment_stop_reason


def successor_outcome_taxonomy(
    predicted: str | None,
    gold: str,
    scored_response_segment: str,
) -> dict[str, bool | str]:
    correct = predicted == gold
    valid_incorrect = predicted is not None and not correct
    if predicted is None:
        model_empty = not bool(numeric_candidates(scored_response_segment))
        parser_failure = not model_empty
    else:
        model_empty = False
        parser_failure = False

    categories = {
        "correct_numeric": correct,
        "valid_extracted_incorrect": valid_incorrect,
        "model_empty_non_answer": model_empty,
        "parser_recognition_failure": parser_failure,
    }
    if sum(categories.values()) != 1:
        raise TerminalAssertionError(
            "accounting_assertion_failure",
            "successor_outcome_taxonomy",
            "successor outcome taxonomy is not mutually exclusive",
            {
                "predicted_answer_present": predicted is not None,
                "gold_answer_present": bool(gold),
                "category_flags": categories,
            },
        )
    category = next(name for name, active in categories.items() if active)
    return {
        "outcome_category": category,
        **categories,
        "correct": correct,
        "extraction_failed": predicted is None,
        "exact_answer_failure": not correct,
    }


def validate_numeric_extraction() -> None:
    exact_cases = (
        ("1/2", "0.5"),
        ("-1/2", "-0.5"),
        ("392/196", "2"),
        ("1.5/0.5", "3"),
        ("32.0", "32"),
        ("0.5", "0.5"),
        ("25%", "25"),
        ("1", "1"),
        ("1,234", "1234"),
        ("3 / 2", "1.5"),
        ("+3/-2", "-1.5"),
    )
    for raw, expected in exact_cases:
        if normalize_number(raw) != expected:
            raise RuntimeError(f"numeric normalizer self-check failed for {raw!r}")
        if extract_gold_answer(f"#### {raw}") != expected:
            raise RuntimeError(f"gold-answer parser self-check failed for {raw!r}")
        primary, _, _, _ = extract_predicted_answer(
            f"#### {raw}",
            "end_of_message",
        )
        fallback, _, _, _ = extract_predicted_answer(
            f"Answer: {raw}",
            "end_of_message",
        )
        if primary != expected or fallback != expected:
            raise RuntimeError(f"prediction parser self-check failed for {raw!r}")

    negative_cases = (
        "3/",
        "/2",
        "3//2",
        "3/x",
        "12/3x",
        "1/2/3",
        "1\u20442",
        "1/2e3",
        "1/2_3",
        "1_000",
        "3 / / 2",
        "3 /",
        "/ 2",
        "3 / x",
        "3/0",
        "8 /\n 2",
        "8/\n2",
        "\n/2",
        "3 /\r\n x",
    )
    for raw in negative_cases:
        if normalize_number(raw) is not None:
            raise RuntimeError(
                f"numeric normalizer negative self-check failed for {raw!r}"
            )
        try:
            extract_gold_answer(f"#### {raw}")
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f"gold-answer parser negative self-check failed for {raw!r}"
            )
        primary, _, _, _ = extract_predicted_answer(
            f"#### {raw}",
            "end_of_message",
        )
        fallback, _, _, _ = extract_predicted_answer(
            f"Answer: {raw}",
            "end_of_message",
        )
        if primary is not None or fallback is not None:
            raise RuntimeError(
                f"prediction parser negative self-check failed for {raw!r}"
            )


def replay_gsm8k_parser_guard() -> dict[str, Any]:
    """Preserve every recorded GSM8K extraction outcome and correctness label."""
    artifact_path = result_path(DATASET_SPECS["gsm8k"])
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        records = artifact["per_example"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the immutable GSM8K replay artifact is invalid") from error

    if (
        artifact.get("mode") != "canonical"
        or artifact.get("sample_count") != CANONICAL_SAMPLE_COUNT
        or not isinstance(records, list)
        or len(records) != CANONICAL_SAMPLE_COUNT
    ):
        raise RuntimeError("the immutable GSM8K replay artifact is incomplete")

    extraction_outcome_mismatches = []
    correctness_mismatches = []
    for position, record in enumerate(records):
        try:
            gold_answer = record["gold_answer"]
            expected_extraction_failed = record["extraction_failed"]
            expected_correct = record["correct"]
            actual_extraction, _, _, _ = extract_predicted_answer(
                response=record["response"],
                generation_stop_reason=record["stop_reason"],
            )
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"GSM8K replay record {position} is invalid"
            ) from error

        actual_extraction_failed = actual_extraction is None
        if actual_extraction_failed != expected_extraction_failed:
            extraction_outcome_mismatches.append(
                {
                    "cohort_position": position,
                    "expected_extraction_failed": expected_extraction_failed,
                    "actual_extraction_failed": actual_extraction_failed,
                }
            )
        actual_correct = actual_extraction == gold_answer
        if actual_correct != expected_correct:
            correctness_mismatches.append(
                {
                    "cohort_position": position,
                    "expected": expected_correct,
                    "actual": actual_correct,
                }
            )

    if extraction_outcome_mismatches or correctness_mismatches:
        mismatch_summary = {
            "extraction_outcome_mismatch_count": len(
                extraction_outcome_mismatches
            ),
            "correctness_mismatch_count": len(correctness_mismatches),
            "first_extraction_outcome_mismatches": (
                extraction_outcome_mismatches[:5]
            ),
            "first_correctness_mismatches": correctness_mismatches[:5],
        }
        raise RuntimeError(
            "GSM8K parser replay guard failed: "
            f"{json.dumps(mismatch_summary, sort_keys=True)}"
        )

    relative_path = artifact_path.relative_to(HERE.parent.parent)
    return {
        "status": "PASS",
        "artifact": str(relative_path),
        "extraction_outcome_matches": CANONICAL_SAMPLE_COUNT,
        "correctness_label_matches": CANONICAL_SAMPLE_COUNT,
        "total": CANONICAL_SAMPLE_COUNT,
    }


def parser_source_text() -> str:
    components = (
        normalize_number,
        numeric_candidate_is_isolated,
        numeric_candidates,
        first_answer_segment,
        extract_predicted_answer,
    )
    source_text = "\n\n".join(
        inspect.getsource(component) for component in components
    )
    return f"NUMBER_PATTERN = {NUMBER_PATTERN!r}\n\n{source_text}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def repair_identity_payloads(artifact: dict[str, Any]) -> dict[str, Any]:
    protocol = artifact["protocol"]
    records = artifact["per_example"]
    return {
        "response_set": [
            {
                "cohort_position": record["cohort_position"],
                "dataset_index": record["dataset_index"],
                "response": record["response"],
            }
            for record in records
        ],
        "cohort": {
            "sample_count": artifact["sample_count"],
            "dataset_provenance": protocol["dataset_provenance"],
            "selection": protocol["selection"],
            "per_example": [
                {
                    "cohort_position": record["cohort_position"],
                    "dataset_index": record["dataset_index"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                }
                for record in records
            ],
        },
        "prompt": {
            "five_shot_demonstrations": protocol["five_shot_demonstrations"],
            "prompt_serialization": protocol["prompt_serialization"],
        },
        "decoding": protocol["decoding"],
        "stopping": {
            "per_sequence_stops": protocol["decoding"]["per_sequence_stops"],
            "per_example": [
                {
                    "cohort_position": record["cohort_position"],
                    "stop_reason": record["stop_reason"],
                    "cap_reached": record["cap_reached"],
                    "generated_tokens": record["generated_tokens"],
                }
                for record in records
            ],
        },
    }


INITIAL_RECORD_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "cohort_position": (int,),
    "dataset_index": (int,),
    "question": (str,),
    "gold_answer": (str,),
    "response": (str,),
    "scored_response_segment": (str,),
    "extracted_answer": (str, type(None)),
    "extraction_source": (str, type(None)),
    "extraction_segment_stop_reason": (str,),
    "stop_reason": (str,),
    "cap_reached": (bool,),
    "generated_tokens": (int,),
    "extraction_failed": (bool,),
    "valid_extracted_incorrect": (bool,),
    "correct": (bool,),
    "prompt_tokens": (int,),
    "generation_seconds_share": (int, float),
}
SUCCESSOR_INITIAL_RECORD_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "outcome_category": (str,),
    "correct_numeric": (bool,),
    "model_empty_non_answer": (bool,),
    "parser_recognition_failure": (bool,),
    "exact_answer_failure": (bool,),
}


def expected_type_label(expected_types: tuple[type, ...]) -> str:
    return " | ".join(expected_type.__name__ for expected_type in expected_types)


def validate_initial_artifact_records(artifact: dict[str, Any]) -> None:
    required_types = dict(INITIAL_RECORD_FIELD_TYPES)
    if artifact.get("model") == BASE_B_PUBLIC_NAME:
        required_types.update(SUCCESSOR_INITIAL_RECORD_FIELD_TYPES)

    record_errors: list[dict[str, Any]] = []
    for record_index, record in enumerate(artifact["per_example"]):
        if not isinstance(record, dict):
            record_errors.append(
                {
                    "record_index": record_index,
                    "missing_fields": sorted(required_types),
                    "invalid_field_types": {
                        "__record__": {
                            "expected": "dict",
                            "actual": type(record).__name__,
                        }
                    },
                }
            )
            continue

        missing_fields = sorted(set(required_types) - set(record))
        invalid_field_types = {}
        for field, expected_types in required_types.items():
            if field not in record:
                continue
            value = record[field]
            valid_type = type(value) in expected_types
            if not valid_type:
                invalid_field_types[field] = {
                    "expected": expected_type_label(expected_types),
                    "actual": type(value).__name__,
                }
        if missing_fields or invalid_field_types:
            record_errors.append(
                {
                    "record_index": record_index,
                    "missing_fields": missing_fields,
                    "invalid_field_types": invalid_field_types,
                }
            )

    if record_errors:
        raise TerminalAssertionError(
            "provenance_assertion_failure",
            "initial_parser_miss_record_structure",
            "initial parser-miss records failed required-field and type validation",
            {
                "record_count": len(artifact["per_example"]),
                "malformed_record_count": len(record_errors),
                "malformed_record_indices": [
                    record_error["record_index"] for record_error in record_errors
                ],
                "record_errors": record_errors,
            },
        )


def immutable_generation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in (
            "cohort_position",
            "dataset_index",
            "question",
            "gold_answer",
            "response",
            "stop_reason",
            "cap_reached",
            "generated_tokens",
            "prompt_tokens",
            "generation_seconds_share",
        )
    }


def load_initial_parser_miss(
    parser_miss_path: Path,
) -> tuple[dict[str, Any], str, str]:
    try:
        artifact = json.loads(parser_miss_path.read_text(encoding="utf-8"))
        parser_provenance = artifact["protocol"]["answer_extraction"][
            "parser_provenance"
        ]
        initial_source = parser_provenance["source_text"]
        initial_fingerprint = parser_provenance["source_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the initial parser-miss artifact is invalid") from error

    valid_artifact = (
        artifact.get("experiment") == EXPERIMENT
        and artifact.get("mode") == "canonical"
        and artifact.get("sample_count") == CANONICAL_SAMPLE_COUNT
        and artifact.get("status") == "PARSER_REPAIR_REQUIRED"
        and artifact.get("verdict", {}).get("token") == "PARSER-REPAIR-REQUIRED"
        and artifact.get("extraction_failures", {}).get("rate", 0)
        > MAX_EXTRACTION_FAILURE_RATE
        and isinstance(artifact.get("per_example"), list)
        and len(artifact["per_example"]) == CANONICAL_SAMPLE_COUNT
        and parser_provenance.get("attempt") == "initial"
        and isinstance(initial_source, str)
        and sha256_text(initial_source) == initial_fingerprint
    )
    if not valid_artifact:
        raise RuntimeError("the initial parser-miss artifact failed provenance checks")
    validate_initial_artifact_records(artifact)
    return artifact, initial_source, initial_fingerprint


def parser_provenance(
    mode: str,
    attempt: str,
    parser_miss_path: Path,
) -> dict[str, Any]:
    current_source = parser_source_text()
    current_fingerprint = sha256_text(current_source)
    provenance: dict[str, Any] = {
        "attempt": attempt if mode == "canonical" else "smoke",
        "source_components": [
            "NUMBER_PATTERN",
            "normalize_number",
            "numeric_candidate_is_isolated",
            "numeric_candidates",
            "first_answer_segment",
            "extract_predicted_answer",
        ],
        "source_sha256": current_fingerprint,
        "source_text": current_source,
    }
    if mode != "canonical":
        return provenance

    relative_attempt_path = str(parser_miss_path.relative_to(HERE.parent.parent))
    if attempt == "initial":
        if parser_miss_path.exists():
            raise RuntimeError(
                "the initial parser attempt is already recorded; use the repaired attempt"
            )
        provenance["initial_miss_artifact_path"] = relative_attempt_path
        provenance["change_from_initial"] = None
        return provenance

    if not parser_miss_path.is_file():
        raise RuntimeError("a repaired parser attempt requires the initial-miss artifact")
    _, initial_source, initial_fingerprint = load_initial_parser_miss(parser_miss_path)
    if current_fingerprint == initial_fingerprint:
        raise RuntimeError("the repaired parser source is identical to the initial parser")

    source_diff = "".join(
        difflib.unified_diff(
            initial_source.splitlines(keepends=True),
            current_source.splitlines(keepends=True),
            fromfile=f"initial/{initial_fingerprint}",
            tofile=f"repaired/{current_fingerprint}",
        )
    )
    if not source_diff:
        raise RuntimeError("the repaired parser has no mechanically recorded source diff")
    provenance.update(
        {
            "initial_miss_artifact_path": relative_attempt_path,
            "initial_miss_artifact_sha256": sha256_file(parser_miss_path),
            "initial_source_sha256": initial_fingerprint,
            "source_changed_from_initial": True,
            "change_from_initial": source_diff,
        }
    )
    return provenance


def re_adjudicate_initial_responses(
    initial_artifact: dict[str, Any],
    parser_miss_path: Path,
    parser_evidence: dict[str, Any],
    dataset_spec: DatasetSpec,
    public_model_name: str,
    successor_base_b: bool,
) -> dict[str, Any]:
    re_adjudication_started = time.perf_counter()
    initial_protocol = initial_artifact["protocol"]
    if (
        initial_artifact.get("model") != public_model_name
        or initial_protocol.get("dataset_key") != dataset_spec.key
        or initial_protocol.get("dataset_revision") != dataset_spec.revision
    ):
        raise TerminalAssertionError(
            "provenance_assertion_failure",
            "repaired_parser_initial_artifact_binding",
            "the initial parser-miss artifact is not bound to this model and dataset",
            {
                "expected_model": public_model_name,
                "recorded_model": initial_artifact.get("model"),
                "expected_dataset_key": dataset_spec.key,
                "recorded_dataset_key": initial_protocol.get("dataset_key"),
                "expected_dataset_revision": dataset_spec.revision,
                "recorded_dataset_revision": initial_protocol.get(
                    "dataset_revision"
                ),
            },
        )

    repaired_records: list[dict[str, Any]] = []
    per_example_comparisons: list[dict[str, Any]] = []
    for initial_record in initial_artifact["per_example"]:
        repaired_record = copy.deepcopy(initial_record)
        predicted, extraction_source, scored_segment, segment_stop_reason = (
            extract_predicted_answer(
                response=initial_record["response"],
                generation_stop_reason=initial_record["stop_reason"],
            )
        )
        repaired_record.update(
            {
                "scored_response_segment": scored_segment,
                "extracted_answer": predicted,
                "extraction_source": extraction_source,
                "extraction_segment_stop_reason": segment_stop_reason,
            }
        )
        if successor_base_b:
            repaired_record.update(
                successor_outcome_taxonomy(
                    predicted=predicted,
                    gold=initial_record["gold_answer"],
                    scored_response_segment=scored_segment,
                )
            )
        else:
            repaired_record.update(
                {
                    "extraction_failed": predicted is None,
                    "valid_extracted_incorrect": (
                        predicted is not None
                        and predicted != initial_record["gold_answer"]
                    ),
                    "correct": predicted == initial_record["gold_answer"],
                }
            )

        immutable_fields_match = json_value_bytes(
            immutable_generation_record(initial_record)
        ) == json_value_bytes(immutable_generation_record(repaired_record))
        comparison = {
            "cohort_position": initial_record["cohort_position"],
            "dataset_index": initial_record["dataset_index"],
            "response_sha256": sha256_text(initial_record["response"]),
            "immutable_generation_fields_byte_identical": immutable_fields_match,
            "initial_extracted_answer": initial_record["extracted_answer"],
            "repaired_extracted_answer": repaired_record["extracted_answer"],
            "initial_extraction_failed": initial_record["extraction_failed"],
            "repaired_extraction_failed": repaired_record["extraction_failed"],
            "initial_correct": initial_record["correct"],
            "repaired_correct": repaired_record["correct"],
        }
        if successor_base_b:
            comparison.update(
                {
                    "initial_outcome_category": initial_record["outcome_category"],
                    "repaired_outcome_category": repaired_record[
                        "outcome_category"
                    ],
                }
            )
        per_example_comparisons.append(comparison)
        repaired_records.append(repaired_record)

    repaired_identity_artifact = {
        "sample_count": initial_artifact["sample_count"],
        "protocol": copy.deepcopy(initial_protocol),
        "per_example": repaired_records,
    }
    initial_payloads = repair_identity_payloads(initial_artifact)
    repaired_payloads = repair_identity_payloads(repaired_identity_artifact)
    identity_checks = {}
    for field in ("response_set", "cohort", "prompt", "decoding", "stopping"):
        initial_bytes = json_value_bytes(initial_payloads[field])
        repaired_bytes = json_value_bytes(repaired_payloads[field])
        identity_checks[field] = {
            "status": "PASS" if initial_bytes == repaired_bytes else "FAIL",
            "byte_identical": initial_bytes == repaired_bytes,
            "initial_sha256": hashlib.sha256(initial_bytes).hexdigest(),
            "repaired_sha256": hashlib.sha256(repaired_bytes).hexdigest(),
        }

    failed_identity_checks = [
        field
        for field, check in identity_checks.items()
        if not check["byte_identical"]
    ]
    failed_record_checks = [
        comparison["cohort_position"]
        for comparison in per_example_comparisons
        if not comparison["immutable_generation_fields_byte_identical"]
    ]
    if failed_identity_checks or failed_record_checks:
        raise TerminalAssertionError(
            "provenance_assertion_failure",
            "repaired_parser_immutable_response_reuse",
            "the repaired parser attempt changed immutable generation evidence",
            {
                "failed_identity_checks": failed_identity_checks,
                "failed_record_positions": failed_record_checks[:20],
                "identity_checks": identity_checks,
            },
        )

    response_reuse = {
        "status": "PASS",
        "generation_performed": False,
        "initial_miss_artifact_path": str(
            parser_miss_path.relative_to(HERE.parent.parent)
        ),
        "initial_miss_artifact_sha256": sha256_file(parser_miss_path),
        "checked_examples": len(per_example_comparisons),
        "all_per_example_immutable_generation_fields_byte_identical": True,
        "identity_checks": identity_checks,
        "per_example_comparisons": per_example_comparisons,
    }
    parser_evidence["immutable_response_reuse"] = response_reuse

    verification_field = generation_verification_field(
        dataset_spec,
        successor_base_b,
    )
    result = summarize(
        records=repaired_records,
        wall_time_seconds=initial_artifact["wall_time_seconds"],
        evaluation_wall_time_seconds=initial_artifact[
            "evaluation_wall_time_seconds"
        ],
        generation_wall_time_seconds=initial_artifact[
            "generation_wall_time_seconds"
        ],
        verification_wall_time_seconds=initial_artifact[
            "verification_wall_time_seconds"
        ],
        canonical_indices=[
            int(record["dataset_index"])
            for record in initial_artifact["per_example"]
        ],
        mode="canonical",
        parser_evidence=parser_evidence,
        dataset_provenance=copy.deepcopy(initial_protocol["dataset_provenance"]),
        leakage_check=copy.deepcopy(initial_protocol["leakage_preflight"]),
        batch_equivalence=copy.deepcopy(
            initial_protocol["decoding"][verification_field]
        ),
        vram=copy.deepcopy(initial_artifact["peak_vram"]),
        dataset_spec=dataset_spec,
        parser_miss_path=parser_miss_path,
        public_model_name=public_model_name,
        successor_base_b=successor_base_b,
        selection_evidence=copy.deepcopy(initial_protocol["selection"]),
    )
    for field in (
        "dataset",
        "dataset_key",
        "dataset_id",
        "dataset_revision",
        "dataset_config",
        "evaluation_split",
        "dataset_provenance",
        "leakage_preflight",
        "selection",
        "five_shot_demonstrations",
        "prompt_serialization",
        "decoding",
        "thresholds",
    ):
        result["protocol"][field] = copy.deepcopy(initial_protocol[field])
    result["protocol"]["answer_extraction"]["parser_attempt"] = "repaired"
    result["protocol"]["answer_extraction"]["parser_source_sha256"] = (
        parser_evidence["source_sha256"]
    )
    result["protocol"]["answer_extraction"]["parser_provenance"] = parser_evidence
    result["generation_performed"] = False
    result["generation_metrics_reused_from_initial_attempt"] = True
    result["re_adjudication_wall_time_seconds"] = (
        time.perf_counter() - re_adjudication_started
    )
    result["immutable_response_reuse"] = response_reuse
    return result


class NewQuestionBoundaryCriteria(StoppingCriteria):
    """Stop each sequence after it starts a generated Question block."""

    def __init__(self, tokenizer, generation_start: int, batch_size: int) -> None:
        self.tokenizer = tokenizer
        self.generation_start = generation_start
        self.boundary_token_counts: list[int | None] = [None] * batch_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for row_index in range(input_ids.shape[0]):
            if self.boundary_token_counts[row_index] is not None:
                stopped[row_index] = True
                continue
            generated_ids = input_ids[row_index, self.generation_start :]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            if NEW_QUESTION_RE.search(generated_text):
                self.boundary_token_counts[row_index] = int(generated_ids.numel())
                stopped[row_index] = True
        return stopped


def percentile_nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def response_length_stats(lengths: Sequence[int]) -> dict[str, int | float]:
    if not lengths:
        return {
            "count": 0,
            "total": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95_nearest_rank": 0,
        }
    return {
        "count": len(lengths),
        "total": sum(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "p95_nearest_rank": percentile_nearest_rank(lengths, 0.95),
    }


REPAIR_IDENTITY_FIELDS = (
    "response_set",
    "cohort",
    "prompt",
    "decoding",
    "stopping",
)


def repaired_attempt_is_qualified(
    parser_evidence: dict[str, Any],
    parser_miss_path: Path,
) -> bool:
    reuse = parser_evidence.get("immutable_response_reuse", {})
    identity_checks = reuse.get("identity_checks", {})
    required_identity_fields = set(REPAIR_IDENTITY_FIELDS)
    return bool(
        parser_evidence.get("attempt") == "repaired"
        and parser_evidence.get("source_changed_from_initial") is True
        and parser_miss_path.is_file()
        and parser_evidence.get("initial_miss_artifact_sha256")
        == sha256_file(parser_miss_path)
        and reuse.get("status") == "PASS"
        and reuse.get("generation_performed") is False
        and reuse.get("initial_miss_artifact_sha256")
        == sha256_file(parser_miss_path)
        and reuse.get(
            "all_per_example_immutable_generation_fields_byte_identical"
        )
        is True
        and set(identity_checks) == required_identity_fields
        and all(
            identity_checks[field].get("status") == "PASS"
            and identity_checks[field].get("byte_identical") is True
            and identity_checks[field].get("initial_sha256")
            == identity_checks[field].get("repaired_sha256")
            for field in required_identity_fields
        )
    )


def repaired_attempt_qualification_evidence(
    parser_evidence: dict[str, Any],
    parser_miss_path: Path,
) -> dict[str, Any]:
    reuse = parser_evidence.get("immutable_response_reuse", {})
    identity_checks = reuse.get("identity_checks", {})
    artifact_exists = parser_miss_path.is_file()
    live_artifact_sha256 = sha256_file(parser_miss_path) if artifact_exists else None
    identity_hash_comparisons = {
        field: {
            "status": identity_checks.get(field, {}).get("status"),
            "byte_identical": identity_checks.get(field, {}).get(
                "byte_identical"
            ),
            "initial_sha256": identity_checks.get(field, {}).get(
                "initial_sha256"
            ),
            "repaired_sha256": identity_checks.get(field, {}).get(
                "repaired_sha256"
            ),
            "hashes_match": (
                identity_checks.get(field, {}).get("initial_sha256")
                == identity_checks.get(field, {}).get("repaired_sha256")
                and identity_checks.get(field, {}).get("initial_sha256")
                is not None
            ),
        }
        for field in REPAIR_IDENTITY_FIELDS
    }
    qualified = repaired_attempt_is_qualified(parser_evidence, parser_miss_path)
    return {
        "status": "PASS" if qualified else "FAIL",
        "checked_immediately_before_canonical_write": True,
        "initial_miss_artifact_exists": artifact_exists,
        "recorded_initial_miss_artifact_sha256": parser_evidence.get(
            "initial_miss_artifact_sha256"
        ),
        "live_initial_miss_artifact_sha256": live_artifact_sha256,
        "initial_miss_artifact_hash_matches_live": (
            artifact_exists
            and parser_evidence.get("initial_miss_artifact_sha256")
            == live_artifact_sha256
        ),
        "required_identity_fields": list(REPAIR_IDENTITY_FIELDS),
        "identity_hash_comparisons": identity_hash_comparisons,
        "all_five_identity_hash_comparisons_pass": all(
            comparison["status"] == "PASS"
            and comparison["byte_identical"] is True
            and comparison["hashes_match"] is True
            for comparison in identity_hash_comparisons.values()
        ),
    }


def canonical_verdict(
    correct_count: int,
    valid_extracted_incorrect_count: int,
    extraction_failure_rate: float,
    parser_evidence: dict[str, Any],
    dataset_spec: DatasetSpec,
    parser_miss_path: Path,
) -> dict[str, Any]:
    if extraction_failure_rate > MAX_EXTRACTION_FAILURE_RATE:
        if parser_evidence["attempt"] == "initial":
            return {
                "token": "PARSER-REPAIR-REQUIRED",
                "reason": "initial_extraction_failure_rate_above_5_percent",
                "next_action": (
                    "repair and qualify the parser once, then rerun with "
                    "--parser-attempt repaired; the initial-miss artifact was "
                    "written but no canonical result was written"
                ),
                "terminal": False,
            }
        if not repaired_attempt_is_qualified(parser_evidence, parser_miss_path):
            raise RuntimeError(
                "terminal extraction-failure VOID requires a qualified repaired attempt"
            )
        return {
            "token": "VOID",
            "reason": "repaired_extraction_failure_rate_above_5_percent",
            "next_action": "stop the task-band program after the repeated parser miss",
            "terminal": True,
        }

    low_band = correct_count < PASS_MIN_CORRECT or correct_count < MIN_CORRECT_POPULATION
    high_band = (
        correct_count > PASS_MAX_CORRECT
        or valid_extracted_incorrect_count < MIN_INCORRECT_POPULATION
    )
    if low_band:
        if dataset_spec.fallback:
            return {
                "token": "VOID",
                "reason": "fallback_below_band_or_fewer_than_40_correct",
                "next_action": "stop the task-band program after the fallback miss",
                "terminal": True,
            }
        return {
            "token": "ABORT-AND-SWAP",
            "reason": "below_band_or_fewer_than_40_correct",
            "next_action": "swap to SVAMP and rerun the same 256-example gate once",
            "terminal": False,
        }
    if high_band:
        if dataset_spec.fallback:
            return {
                "token": "VOID",
                "reason": (
                    "fallback_above_band_or_fewer_than_40_valid_extracted_incorrect"
                ),
                "next_action": "stop the task-band program after the fallback miss",
                "terminal": True,
            }
        return {
            "token": "ABORT-AND-SWAP",
            "reason": "above_band_or_fewer_than_40_valid_extracted_incorrect",
            "next_action": "swap to GSM-Hard and rerun the same 256-example gate once",
            "terminal": False,
        }
    return {
        "token": "PASS",
        "reason": "all_preregistered_task_band_thresholds_satisfied",
        "next_action": "task-band gate permits the separately reviewed mechanics pilot",
        "terminal": False,
    }


def successor_canonical_verdict(
    correct_count: int,
    usable_incorrect_count: int,
    model_empty_non_answer_count: int,
    parser_recognition_failure_count: int,
    parser_evidence: dict[str, Any],
    parser_miss_path: Path,
) -> dict[str, Any]:
    if model_empty_non_answer_count > MAX_FAILURE_COUNT:
        return {
            "token": "VOID",
            "reason": "model_empty_non_answer_count_above_12_of_256",
            "next_action": "return to steering; no successor fallback is permitted",
            "terminal": True,
        }

    if parser_recognition_failure_count > MAX_FAILURE_COUNT:
        if parser_evidence["attempt"] == "initial":
            return {
                "token": "PARSER-REPAIR-REQUIRED",
                "reason": "initial_parser_recognition_failure_count_above_12_of_256",
                "next_action": (
                    "independently review the failed scored segments and perform "
                    "at most one eligible parser repair under the frozen scope"
                ),
                "terminal": False,
            }
        if not repaired_attempt_is_qualified(parser_evidence, parser_miss_path):
            raise RuntimeError(
                "terminal parser-recognition VOID requires a qualified repair"
            )
        return {
            "token": "VOID",
            "reason": "repaired_parser_recognition_failure_count_above_12_of_256",
            "next_action": "return to steering; no successor fallback is permitted",
            "terminal": True,
        }

    if correct_count < PASS_MIN_CORRECT or correct_count < MIN_CORRECT_POPULATION:
        return {
            "token": "VOID",
            "reason": "below_band_or_fewer_than_40_correct",
            "next_action": "return to steering; no successor fallback is permitted",
            "terminal": True,
        }
    if correct_count > PASS_MAX_CORRECT or usable_incorrect_count < 40:
        return {
            "token": "VOID",
            "reason": "above_band_or_fewer_than_40_usable_incorrect",
            "next_action": "return to steering; no successor fallback is permitted",
            "terminal": True,
        }
    return {
        "token": "PASS",
        "reason": "all_successor_task_band_thresholds_satisfied",
        "next_action": (
            "the full program still requires an admissible mechanics outcome, "
            "replacement transfer-set preregistration, and launch review"
        ),
        "terminal": False,
    }


def load_frozen_base(
    model_identifier: str,
    model_revision: str,
    device: torch.device,
    public_model_name: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        revision=model_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{public_model_name} tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                f"{public_model_name} tokenizer has neither a pad nor EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        revision=model_revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def validate_context_window(
    model,
    prompt_tokens: int,
    public_model_name: str,
) -> None:
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and prompt_tokens + MAX_NEW_TOKENS > context_limit:
        raise RuntimeError(
            f"the fixed five-shot prompt exceeds {public_model_name}'s context window"
        )


def evaluate_examples(
    train_split: Dataset,
    test_split: Dataset,
    selected_indices: Sequence[int],
    dataset_spec: DatasetSpec,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    public_model_name: str,
    successor_base_b: bool,
) -> tuple[list[dict[str, Any]], float, float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    generation_wall_time_seconds = 0.0
    eos_token_id = tokenizer.eos_token_id
    eos_token_ids = (
        {int(token_id) for token_id in eos_token_id}
        if isinstance(eos_token_id, (list, tuple))
        else {int(eos_token_id)}
    )

    for batch_start in range(0, len(selected_indices), batch_size):
        batch_indices = selected_indices[batch_start : batch_start + batch_size]
        rows = [test_split[index] for index in batch_indices]
        conversations = [
            build_five_shot_messages(
                train_split,
                row_question(row, dataset_spec),
                dataset_spec,
            )
            for row in rows
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        generation_start = int(input_ids.shape[1])
        validate_context_window(model, generation_start, public_model_name)
        boundary_criteria = NewQuestionBoundaryCriteria(
            tokenizer=tokenizer,
            generation_start=generation_start,
            batch_size=len(batch_indices),
        )

        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[boundary_criteria],
            )
        torch.cuda.synchronize(device)
        batch_generation_seconds = time.perf_counter() - generation_started
        generation_wall_time_seconds += batch_generation_seconds

        generated_batch = output_ids[:, generation_start:]
        batch_metadata: list[dict[str, Any]] = []
        for row_index, generated_ids in enumerate(generated_batch):
            token_ids = [int(token_id) for token_id in generated_ids.tolist()]
            boundary_count = boundary_criteria.boundary_token_counts[row_index]
            eos_position = next(
                (
                    position
                    for position, token_id in enumerate(token_ids)
                    if token_id in eos_token_ids
                ),
                None,
            )
            eos_count = eos_position + 1 if eos_position is not None else None

            if boundary_count is not None and (
                eos_count is None or boundary_count <= eos_count
            ):
                generated_token_count = boundary_count
                content_token_count = boundary_count
                stop_reason = "new_question_boundary"
            elif eos_count is not None:
                generated_token_count = eos_count
                content_token_count = eos_position
                stop_reason = "end_of_message"
            else:
                generated_token_count = len(token_ids)
                content_token_count = generated_token_count
                stop_reason = (
                    "max_new_tokens"
                    if generated_token_count >= MAX_NEW_TOKENS
                    else "generation_stopped_other"
                )

            response = tokenizer.decode(
                generated_ids[:content_token_count],
                skip_special_tokens=True,
            ).strip()
            predicted, extraction_source, scored_segment, segment_stop_reason = (
                extract_predicted_answer(
                    response=response,
                    generation_stop_reason=stop_reason,
                )
            )
            batch_metadata.append(
                {
                    "response": response,
                    "scored_response_segment": scored_segment,
                    "extracted_answer": predicted,
                    "extraction_source": extraction_source,
                    "extraction_segment_stop_reason": segment_stop_reason,
                    "stop_reason": stop_reason,
                    "cap_reached": stop_reason == "max_new_tokens",
                    "generated_tokens": generated_token_count,
                }
            )

        batch_token_total = sum(
            int(metadata["generated_tokens"]) for metadata in batch_metadata
        )
        for row_index, (dataset_index, row, metadata) in enumerate(
            zip(batch_indices, rows, batch_metadata, strict=True)
        ):
            gold = row_gold_answer(row, dataset_spec)
            predicted = metadata["extracted_answer"]
            if successor_base_b:
                outcome_fields = successor_outcome_taxonomy(
                    predicted=predicted,
                    gold=gold,
                    scored_response_segment=metadata["scored_response_segment"],
                )
            else:
                outcome_fields = {
                    "extraction_failed": predicted is None,
                    "valid_extracted_incorrect": (
                        predicted is not None and predicted != gold
                    ),
                    "correct": predicted == gold,
                }
            token_share = (
                int(metadata["generated_tokens"]) / batch_token_total
                if batch_token_total
                else 1.0 / len(batch_indices)
            )
            records.append(
                {
                    "cohort_position": batch_start + row_index,
                    "dataset_index": int(dataset_index),
                    "question": row_question(row, dataset_spec),
                    "gold_answer": gold,
                    **metadata,
                    **outcome_fields,
                    "prompt_tokens": int(attention_mask[row_index].sum().item()),
                    "generation_seconds_share": batch_generation_seconds * token_share,
                }
            )

    return (
        records,
        time.perf_counter() - started,
        generation_wall_time_seconds,
    )


def summarize(
    records: Sequence[dict[str, Any]],
    wall_time_seconds: float,
    evaluation_wall_time_seconds: float,
    generation_wall_time_seconds: float,
    verification_wall_time_seconds: float,
    canonical_indices: Sequence[int],
    mode: str,
    parser_evidence: dict[str, Any],
    dataset_provenance: dict[str, Any],
    leakage_check: dict[str, Any],
    batch_equivalence: dict[str, Any],
    vram: dict[str, Any],
    dataset_spec: DatasetSpec,
    parser_miss_path: Path,
    public_model_name: str,
    successor_base_b: bool,
    selection_evidence: dict[str, Any],
) -> dict[str, Any]:
    verification_field = generation_verification_field(
        dataset_spec,
        successor_base_b,
    )
    denominator = len(records)
    correct_count = sum(bool(record["correct"]) for record in records)
    exact_answer_failure_count = denominator - correct_count
    failure_count = sum(bool(record["extraction_failed"]) for record in records)
    valid_extracted_incorrect_count = sum(
        bool(record["valid_extracted_incorrect"]) for record in records
    )
    generated_lengths = [int(record["generated_tokens"]) for record in records]
    total_generated_tokens = sum(generated_lengths)
    cap_reached_count = sum(bool(record["cap_reached"]) for record in records)
    stop_reason_distribution = dict(
        sorted(Counter(record["stop_reason"] for record in records).items())
    )
    setup_wall_time_seconds = max(
        0.0,
        wall_time_seconds
        - evaluation_wall_time_seconds
        - verification_wall_time_seconds,
    )

    frequencies = Counter(
        record["extracted_answer"]
        if record["extracted_answer"] is not None
        else "__EXTRACTION_FAILURE__"
        for record in records
    )
    frequency_distribution = dict(
        sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    )

    extraction_failure_rate = failure_count / denominator
    successor_counts: dict[str, int] = {}
    if successor_base_b:
        successor_counts = {
            category: sum(record[category] for record in records)
            for category in (
                "correct_numeric",
                "valid_extracted_incorrect",
                "model_empty_non_answer",
                "parser_recognition_failure",
            )
        }
        if sum(successor_counts.values()) != denominator:
            raise TerminalAssertionError(
                "accounting_assertion_failure",
                "successor_aggregate_accounting",
                "successor outcome categories do not exhaust the cohort",
                {
                    "sample_count": denominator,
                    "category_counts": successor_counts,
                    "category_sum": sum(successor_counts.values()),
                },
            )
        if successor_counts["correct_numeric"] != correct_count:
            raise TerminalAssertionError(
                "accounting_assertion_failure",
                "successor_aggregate_accounting",
                "successor correct-category accounting diverged",
                {
                    "correct_count": correct_count,
                    "correct_numeric_count": successor_counts["correct_numeric"],
                },
            )
        if (
            successor_counts["model_empty_non_answer"]
            + successor_counts["parser_recognition_failure"]
            != failure_count
        ):
            raise TerminalAssertionError(
                "accounting_assertion_failure",
                "successor_aggregate_accounting",
                "successor extraction-failure accounting diverged",
                {
                    "extraction_failure_count": failure_count,
                    "model_empty_non_answer_count": successor_counts[
                        "model_empty_non_answer"
                    ],
                    "parser_recognition_failure_count": successor_counts[
                        "parser_recognition_failure"
                    ],
                },
            )
        usable_incorrect_count = (
            successor_counts["valid_extracted_incorrect"]
            + successor_counts["model_empty_non_answer"]
        )
    else:
        usable_incorrect_count = valid_extracted_incorrect_count
    if mode == "canonical":
        if successor_base_b:
            verdict = successor_canonical_verdict(
                correct_count=correct_count,
                usable_incorrect_count=usable_incorrect_count,
                model_empty_non_answer_count=successor_counts[
                    "model_empty_non_answer"
                ],
                parser_recognition_failure_count=successor_counts[
                    "parser_recognition_failure"
                ],
                parser_evidence=parser_evidence,
                parser_miss_path=parser_miss_path,
            )
        else:
            verdict = canonical_verdict(
                correct_count=correct_count,
                valid_extracted_incorrect_count=valid_extracted_incorrect_count,
                extraction_failure_rate=extraction_failure_rate,
                parser_evidence=parser_evidence,
                dataset_spec=dataset_spec,
                parser_miss_path=parser_miss_path,
            )
    else:
        verdict = {
            "token": "SMOKE_ONLY",
            "reason": "canonical thresholds require exactly 256 examples",
            "next_action": "obtain independent pipeline review before the full gate",
            "terminal": False,
        }

    return {
        "experiment": EXPERIMENT,
        "status": (
            "PARSER_REPAIR_REQUIRED"
            if verdict["token"] == "PARSER-REPAIR-REQUIRED"
            else ("COMPLETE" if mode == "canonical" else "SMOKE_ONLY")
        ),
        "mode": mode,
        "model": public_model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": dataset_spec.name,
            "dataset_key": dataset_spec.key,
            "dataset_id": dataset_spec.dataset_id,
            "dataset_revision": dataset_spec.revision,
            "dataset_config": dataset_spec.config,
            "evaluation_split": TEST_SPLIT,
            "dataset_provenance": dataset_provenance,
            "leakage_preflight": leakage_check,
            "selection": selection_evidence,
            "five_shot_demonstrations": {
                "split": TRAIN_SPLIT,
                "indices": list(DEMONSTRATION_INDICES),
                "prompt_preamble": PROMPT_PREAMBLE,
            },
            "prompt_serialization": {
                "method": "tokenizer.apply_chat_template",
                "system_preamble": True,
                "demonstrations": "user_assistant_pairs",
                "target": "final_user_message",
                "add_generation_prompt": True,
            },
            "decoding": {
                "strategy": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": "bfloat16",
                "device": "cuda",
                "batch_size": evaluation_batch_size(
                    dataset_spec,
                    successor_base_b,
                ),
                "padding_side": "left",
                "per_sequence_stops": [
                    "end_of_message",
                    "new_question_boundary",
                    "max_new_tokens",
                ],
                verification_field: batch_equivalence,
            },
            "answer_extraction": {
                "segmentation": (
                    "truncate at first final-answer line, end-of-message, or "
                    "generated new-question delimiter"
                ),
                "primary": "first valid final-line #### answer in the first segment",
                "fallback": "last numeric token inside that same first segment",
                "normalization": (
                    "remove thousands separators and compare finite decimals "
                    "and fractions as exact rational values"
                ),
                "parser_attempt": parser_evidence["attempt"],
                "parser_source_sha256": parser_evidence["source_sha256"],
                "parser_provenance": parser_evidence,
            },
            "thresholds": (
                {
                    "correct_count_inclusive": [PASS_MIN_CORRECT, PASS_MAX_CORRECT],
                    "minimum_correct_population": MIN_CORRECT_POPULATION,
                    "minimum_usable_incorrect_population": MIN_INCORRECT_POPULATION,
                    "maximum_model_empty_non_answer_count": MAX_FAILURE_COUNT,
                    "maximum_parser_recognition_failure_count": MAX_FAILURE_COUNT,
                    "denominator": CANONICAL_SAMPLE_COUNT,
                }
                if successor_base_b
                else {
                    "correct_count_inclusive": [PASS_MIN_CORRECT, PASS_MAX_CORRECT],
                    "minimum_correct_population": MIN_CORRECT_POPULATION,
                    "minimum_valid_extracted_incorrect_population": (
                        MIN_INCORRECT_POPULATION
                    ),
                    "maximum_extraction_failure_rate": MAX_EXTRACTION_FAILURE_RATE,
                }
            ),
        },
        "sample_count": denominator,
        "correct_count": correct_count,
        "exact_answer_failure_count": exact_answer_failure_count,
        "valid_extracted_incorrect_count": valid_extracted_incorrect_count,
        **(
            {
                "correct_numeric_count": successor_counts["correct_numeric"],
                "model_empty_non_answer_count": successor_counts[
                    "model_empty_non_answer"
                ],
                "parser_recognition_failure_count": successor_counts[
                    "parser_recognition_failure"
                ],
                "usable_incorrect_count": usable_incorrect_count,
                "outcome_categories": {
                    category: {
                        "numerator": count,
                        "denominator": denominator,
                        "rate": count / denominator,
                    }
                    for category, count in successor_counts.items()
                },
                "model_empty_non_answers": {
                    "numerator": successor_counts["model_empty_non_answer"],
                    "denominator": denominator,
                    "rate": successor_counts["model_empty_non_answer"] / denominator,
                },
                "parser_recognition_failures": {
                    "numerator": successor_counts["parser_recognition_failure"],
                    "denominator": denominator,
                    "rate": (
                        successor_counts["parser_recognition_failure"] / denominator
                    ),
                },
                "usable_incorrect": {
                    "numerator": usable_incorrect_count,
                    "denominator": denominator,
                    "rate": usable_incorrect_count / denominator,
                },
            }
            if successor_base_b
            else {}
        ),
        "exact_answer_accuracy": {
            "numerator": correct_count,
            "denominator": denominator,
            "rate": correct_count / denominator,
        },
        "extraction_failures": {
            "numerator": failure_count,
            "denominator": denominator,
            "rate": extraction_failure_rate,
        },
        "response_length_tokens": response_length_stats(generated_lengths),
        "stop_reasons": stop_reason_distribution,
        "cap_reached": {
            "numerator": cap_reached_count,
            "denominator": denominator,
            "rate": cap_reached_count / denominator,
        },
        "answer_frequency_distribution": frequency_distribution,
        "wall_time_seconds": wall_time_seconds,
        "setup_wall_time_seconds": setup_wall_time_seconds,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "verification_wall_time_seconds": verification_wall_time_seconds,
        "generation_wall_time_seconds": generation_wall_time_seconds,
        "throughput_generated_tokens_per_second": (
            total_generated_tokens / generation_wall_time_seconds
            if generation_wall_time_seconds > 0
            else 0.0
        ),
        "peak_vram": vram,
        "verdict": verdict,
        "per_example": list(records),
    }


def leakage_void_result(
    trigger_mode: str,
    canonical_indices: Sequence[int],
    dataset_provenance: dict[str, Any],
    leakage_check: dict[str, Any],
    parser_evidence: dict[str, Any],
    dataset_spec: DatasetSpec,
    public_model_name: str,
    selection_evidence: dict[str, Any],
) -> dict[str, Any]:
    if leakage_check["status"] != "FAIL" or leakage_check["overlap_count"] < 1:
        raise RuntimeError("leakage VOID requires positive leakage evidence")
    return {
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "mode": "canonical",
        "trigger_mode": trigger_mode,
        "model": public_model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": dataset_spec.name,
            "dataset_key": dataset_spec.key,
            "dataset_id": dataset_spec.dataset_id,
            "dataset_revision": dataset_spec.revision,
            "dataset_config": dataset_spec.config,
            "evaluation_split": TEST_SPLIT,
            "dataset_provenance": dataset_provenance,
            "leakage_preflight": leakage_check,
            "selection": selection_evidence,
            "five_shot_demonstrations": {
                "split": TRAIN_SPLIT,
                "indices": list(DEMONSTRATION_INDICES),
                "prompt_preamble": PROMPT_PREAMBLE,
            },
            "answer_extraction": {
                "parser_attempt": parser_evidence["attempt"],
                "parser_source_sha256": parser_evidence["source_sha256"],
                "parser_provenance": parser_evidence,
            },
        },
        "sample_count": 0,
        "verdict": {
            "token": "VOID",
            "reason": "demonstration_or_cohort_leakage_detected",
            "next_action": "stop the task-band program",
            "terminal": True,
        },
        "per_example": [],
    }


def terminal_assertion_void_result(
    error: TerminalAssertionError,
    dataset_spec: DatasetSpec,
    public_model_name: str,
    successor_base_b: bool,
    selection_evidence: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    parser_evidence: dict[str, Any] | None = None,
    records: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    evidence = {
        "assertion_kind": error.reason,
        "stage": error.stage,
        "message": error.public_message,
        "details": error.evidence,
    }
    return {
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "mode": "canonical",
        "model": public_model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": dataset_spec.name,
            "dataset_key": dataset_spec.key,
            "dataset_id": dataset_spec.dataset_id,
            "dataset_revision": dataset_spec.revision,
            "dataset_config": dataset_spec.config,
            "evaluation_split": TEST_SPLIT,
            "dataset_provenance": dataset_provenance,
            "selection": selection_evidence,
            "decoding": {
                "strategy": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batch_size": evaluation_batch_size(
                    dataset_spec,
                    successor_base_b,
                ),
                "verification_kind": generation_verification_field(
                    dataset_spec,
                    successor_base_b,
                ),
            },
            "answer_extraction": {
                "parser_provenance": parser_evidence,
            },
        },
        "sample_count": len(records),
        "terminal_failure_evidence": evidence,
        "verdict": {
            "token": "VOID",
            "reason": error.reason,
            "next_action": "return to steering; no successor fallback is permitted",
            "terminal": True,
        },
        "per_example": list(records),
    }


def write_terminal_assertion_void(
    error: TerminalAssertionError,
    canonical_result_path: Path,
    parser_miss_path: Path,
    dataset_spec: DatasetSpec,
    public_model_name: str,
    successor_base_b: bool,
    selection_evidence: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    parser_evidence: dict[str, Any] | None = None,
    records: Sequence[dict[str, Any]] = (),
) -> int:
    result = terminal_assertion_void_result(
        error=error,
        dataset_spec=dataset_spec,
        public_model_name=public_model_name,
        successor_base_b=successor_base_b,
        selection_evidence=selection_evidence,
        dataset_provenance=dataset_provenance,
        parser_evidence=parser_evidence,
        records=records,
    )
    write_canonical_result(result, canonical_result_path, parser_miss_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": public_model_name,
                "sample_count": result["sample_count"],
                "verdict": result["verdict"],
                "failure_evidence": result["terminal_failure_evidence"],
                "result_path": str(
                    canonical_result_path.relative_to(HERE.parent.parent)
                ),
            },
            indent=2,
        )
    )
    print(
        f"ERROR: canonical task-band terminal VOID ({error.reason}).",
        file=sys.stderr,
    )
    return 2


def write_evidence_artifact(
    path: Path,
    result: dict[str, Any],
    evidence_label: str,
) -> None:
    if path.exists():
        raise RuntimeError(f"{evidence_label} already exists and will not be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(result, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        if path.exists():
            raise RuntimeError(f"{evidence_label} appeared during the run")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_canonical_result(
    result: dict[str, Any],
    canonical_result_path: Path,
    parser_miss_path: Path,
) -> dict[str, Any]:
    parser_evidence = (
        result.get("protocol", {})
        .get("answer_extraction", {})
        .get("parser_provenance", {})
    )
    if isinstance(parser_evidence, dict) and parser_evidence.get("attempt") == "repaired":
        qualification = repaired_attempt_qualification_evidence(
            parser_evidence,
            parser_miss_path,
        )
        result["repaired_attempt_qualification"] = qualification
        if qualification["status"] != "PASS":
            attempted_verdict = copy.deepcopy(result.get("verdict", {}))
            protocol = result.get("protocol", {})
            dataset_key = protocol.get("dataset_key")
            dataset_spec = DATASET_SPECS.get(dataset_key)
            if dataset_spec is None:
                raise RuntimeError(
                    "a repaired result lacks a recognized dataset binding"
                )
            error = TerminalAssertionError(
                "provenance_assertion_failure",
                "repaired_parser_write_qualification",
                "repaired outcome failed qualification immediately before canonical write",
                {
                    "attempted_verdict": attempted_verdict,
                    "qualification": qualification,
                },
            )
            result = terminal_assertion_void_result(
                error=error,
                dataset_spec=dataset_spec,
                public_model_name=str(result.get("model")),
                successor_base_b=result.get("model") == BASE_B_PUBLIC_NAME,
                selection_evidence=copy.deepcopy(protocol.get("selection")),
                dataset_provenance=copy.deepcopy(
                    protocol.get("dataset_provenance")
                ),
                parser_evidence=copy.deepcopy(parser_evidence),
            )
            result["repaired_attempt_qualification"] = qualification
            result["attempted_repaired_outcome"] = {
                "status": "REJECTED",
                "verdict": attempted_verdict,
            }

    verdict = result.get("verdict", {})
    if verdict.get("token") == "VOID":
        reason = verdict.get("reason")
        if reason in {
            "repaired_extraction_failure_rate_above_5_percent",
            "repaired_parser_recognition_failure_count_above_12_of_256",
        }:
            parser_evidence = result.get("protocol", {}).get(
                "answer_extraction", {}
            ).get("parser_provenance", {})
            recorded_artifact_hash = parser_evidence.get(
                "initial_miss_artifact_sha256"
            )
            _, _, initial_source_fingerprint = load_initial_parser_miss(
                parser_miss_path
            )
            current_source_fingerprint = sha256_text(parser_source_text())
            qualified_repair = (
                repaired_attempt_is_qualified(parser_evidence, parser_miss_path)
                and recorded_artifact_hash == sha256_file(parser_miss_path)
                and parser_evidence.get("initial_source_sha256")
                == initial_source_fingerprint
                and parser_evidence.get("source_sha256")
                == current_source_fingerprint
                and initial_source_fingerprint != current_source_fingerprint
            )
            if not qualified_repair:
                raise RuntimeError(
                    "terminal extraction-failure VOID lacks qualified repair evidence"
                )
            if (
                reason
                == "repaired_parser_recognition_failure_count_above_12_of_256"
                and result.get("parser_recognition_failure_count", 0)
                <= MAX_FAILURE_COUNT
            ):
                raise RuntimeError(
                    "terminal parser-recognition VOID lacks a threshold miss"
                )
        elif reason == "demonstration_or_cohort_leakage_detected":
            leakage_check = result.get("protocol", {}).get("leakage_preflight", {})
            if (
                leakage_check.get("status") != "FAIL"
                or leakage_check.get("overlap_count", 0) < 1
                or not leakage_check.get("overlaps")
            ):
                raise RuntimeError("terminal leakage VOID lacks leakage evidence")
        elif reason in TERMINAL_ASSERTION_REASONS:
            failure_evidence = result.get("terminal_failure_evidence", {})
            if (
                result.get("mode") != "canonical"
                or result.get("status") != "COMPLETE"
                or failure_evidence.get("assertion_kind") != reason
                or not failure_evidence.get("stage")
                or not failure_evidence.get("message")
                or not isinstance(failure_evidence.get("details"), dict)
            ):
                raise RuntimeError(
                    "terminal assertion VOID lacks embedded failure evidence"
                )
        elif result.get("model") == BASE_B_PUBLIC_NAME and reason in {
            "model_empty_non_answer_count_above_12_of_256",
            "below_band_or_fewer_than_40_correct",
            "above_band_or_fewer_than_40_usable_incorrect",
        }:
            correct_count = result.get("correct_count")
            usable_incorrect_count = result.get("usable_incorrect_count")
            model_empty_count = result.get("model_empty_non_answer_count")
            parser_failure_count = result.get("parser_recognition_failure_count")
            successor_counts = (
                result.get("correct_numeric_count", 0)
                + result.get("valid_extracted_incorrect_count", 0)
                + (model_empty_count or 0)
                + (parser_failure_count or 0)
            )
            reason_is_supported = (
                (
                    reason == "model_empty_non_answer_count_above_12_of_256"
                    and isinstance(model_empty_count, int)
                    and model_empty_count > MAX_FAILURE_COUNT
                )
                or (
                    reason == "below_band_or_fewer_than_40_correct"
                    and isinstance(correct_count, int)
                    and (
                        correct_count < PASS_MIN_CORRECT
                        or correct_count < MIN_CORRECT_POPULATION
                    )
                )
                or (
                    reason == "above_band_or_fewer_than_40_usable_incorrect"
                    and isinstance(correct_count, int)
                    and isinstance(usable_incorrect_count, int)
                    and (
                        correct_count > PASS_MAX_CORRECT
                        or usable_incorrect_count < MIN_INCORRECT_POPULATION
                    )
                )
            )
            if (
                result.get("sample_count") != CANONICAL_SAMPLE_COUNT
                or successor_counts != CANONICAL_SAMPLE_COUNT
                or not reason_is_supported
            ):
                raise RuntimeError("terminal successor VOID lacks qualifying evidence")
        elif reason in {
            "fallback_below_band_or_fewer_than_40_correct",
            "fallback_above_band_or_fewer_than_40_valid_extracted_incorrect",
        }:
            protocol = result.get("protocol", {})
            correct_count = result.get("correct_count")
            incorrect_count = result.get("valid_extracted_incorrect_count")
            extraction_failure_rate = result.get("extraction_failures", {}).get(
                "rate"
            )
            below_band = (
                isinstance(correct_count, int)
                and (
                    correct_count < PASS_MIN_CORRECT
                    or correct_count < MIN_CORRECT_POPULATION
                )
            )
            above_band = (
                isinstance(correct_count, int)
                and isinstance(incorrect_count, int)
                and (
                    correct_count > PASS_MAX_CORRECT
                    or incorrect_count < MIN_INCORRECT_POPULATION
                )
            )
            valid_fallback_void = (
                protocol.get("dataset_key") == "svamp"
                and result.get("sample_count") == CANONICAL_SAMPLE_COUNT
                and isinstance(extraction_failure_rate, (int, float))
                and extraction_failure_rate <= MAX_EXTRACTION_FAILURE_RATE
                and (
                    (reason.startswith("fallback_below") and below_band)
                    or (reason.startswith("fallback_above") and above_band)
                )
            )
            if not valid_fallback_void:
                raise RuntimeError("terminal fallback VOID lacks qualifying evidence")
        else:
            raise RuntimeError("unrecognized terminal VOID cannot be written")
    write_evidence_artifact(canonical_result_path, result, "canonical E1 evidence")
    return result


def write_initial_parser_miss(
    result: dict[str, Any],
    parser_miss_path: Path,
) -> None:
    parser_evidence = result.get("protocol", {}).get("answer_extraction", {}).get(
        "parser_provenance", {}
    )
    if (
        result.get("verdict", {}).get("token") != "PARSER-REPAIR-REQUIRED"
        or result.get("sample_count") != CANONICAL_SAMPLE_COUNT
        or len(result.get("per_example", [])) != CANONICAL_SAMPLE_COUNT
        or result.get("extraction_failures", {}).get("rate", 0)
        <= MAX_EXTRACTION_FAILURE_RATE
        or parser_evidence.get("attempt") != "initial"
        or sha256_text(parser_evidence.get("source_text", ""))
        != parser_evidence.get("source_sha256")
        or parser_evidence.get("source_sha256") != sha256_text(parser_source_text())
    ):
        raise RuntimeError("initial parser-miss evidence is incomplete")
    write_evidence_artifact(
        parser_miss_path,
        result,
        "initial parser-miss evidence",
    )


def smoke_console_summary(
    result: dict[str, Any],
    replay_guard: dict[str, Any],
) -> dict[str, Any]:
    verification_field = (
        "repeat_determinism"
        if (
            result["protocol"]["dataset_key"] == "svamp"
            or result["model"] == BASE_B_PUBLIC_NAME
        )
        else "batch_vs_unbatched_equivalence"
    )
    diagnostics = []
    for record in result["per_example"]:
        response = str(record["response"])
        diagnostics.append(
            {
                "cohort_position": record["cohort_position"],
                "dataset_index": record["dataset_index"],
                "response": response[:MAX_SMOKE_RESPONSE_CHARS],
                "response_truncated_for_console": (
                    len(response) > MAX_SMOKE_RESPONSE_CHARS
                ),
                "gold_answer": record["gold_answer"],
                "extracted_answer": record["extracted_answer"],
                "extraction_source": record["extraction_source"],
                "extraction_segment_stop_reason": record[
                    "extraction_segment_stop_reason"
                ],
                "correct": record["correct"],
                "generated_tokens": record["generated_tokens"],
                "stop_reason": record["stop_reason"],
                "cap_reached": record["cap_reached"],
            }
        )
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "model": result["model"],
        "dataset": result["protocol"]["dataset"],
        "dataset_revision": result["protocol"]["dataset_revision"],
        "sample_count": result["sample_count"],
        "correct_count": result["correct_count"],
        "exact_answer_failure_count": result["exact_answer_failure_count"],
        "valid_extracted_incorrect_count": result[
            "valid_extracted_incorrect_count"
        ],
        "exact_answer_accuracy": result["exact_answer_accuracy"],
        "extraction_failures": result["extraction_failures"],
        "response_length_tokens": result["response_length_tokens"],
        "stop_reasons": result["stop_reasons"],
        "cap_reached": result["cap_reached"],
        "wall_time_seconds": result["wall_time_seconds"],
        "setup_wall_time_seconds": result["setup_wall_time_seconds"],
        "evaluation_wall_time_seconds": result["evaluation_wall_time_seconds"],
        "generation_wall_time_seconds": result["generation_wall_time_seconds"],
        "throughput_generated_tokens_per_second": result[
            "throughput_generated_tokens_per_second"
        ],
        "projected_256_wall_time_seconds": (
            result["setup_wall_time_seconds"]
            + result["verification_wall_time_seconds"]
            + result["evaluation_wall_time_seconds"]
            * CANONICAL_SAMPLE_COUNT
            / result["sample_count"]
        ),
        "peak_vram": result["peak_vram"],
        verification_field: result["protocol"]["decoding"][
            verification_field
        ],
        "gsm8k_replay_guard": replay_guard,
        "verdict": result["verdict"],
        "per_example_diagnostics": diagnostics,
    }


def verify_batch_equivalence(
    batched_records: Sequence[dict[str, Any]],
    train_split: Dataset,
    test_split: Dataset,
    selected_indices: Sequence[int],
    dataset_spec: DatasetSpec,
    evaluation_batch_size: int,
    tokenizer,
    model,
    device: torch.device,
    public_model_name: str,
    successor_base_b: bool,
) -> tuple[dict[str, Any], float]:
    repeat_determinism = uses_repeat_determinism(
        dataset_spec,
        successor_base_b,
    )
    check_count = min(EQUIVALENCE_CHECK_COUNT, len(selected_indices))
    if check_count == 0:
        raise RuntimeError("batch-equivalence check received no examples")
    check_indices = selected_indices[:check_count]
    unbatched_records, verification_seconds, _ = evaluate_examples(
        train_split=train_split,
        test_split=test_split,
        selected_indices=check_indices,
        dataset_spec=dataset_spec,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
        public_model_name=public_model_name,
        successor_base_b=successor_base_b,
    )
    comparisons = []
    for batched, unbatched in zip(
        batched_records[:check_count],
        unbatched_records,
        strict=True,
    ):
        answers_match = (
            batched["extracted_answer"] == unbatched["extracted_answer"]
        )
        responses_match = batched["response"] == unbatched["response"]
        comparison = {
            "cohort_position": batched["cohort_position"],
            "answers_match": answers_match,
            "responses_match": responses_match,
        }
        if repeat_determinism:
            comparison.update(
                {
                    "first_stop_reason": batched["stop_reason"],
                    "repeat_stop_reason": unbatched["stop_reason"],
                }
            )
        else:
            comparison.update(
                {
                    "batched_stop_reason": batched["stop_reason"],
                    "unbatched_stop_reason": unbatched["stop_reason"],
                }
            )
        comparisons.append(comparison)
    passed = all(
        comparison["answers_match"] and comparison["responses_match"]
        for comparison in comparisons
    )
    if not passed:
        reason = (
            "repeated batch-1 greedy generation diverged"
            if repeat_determinism
            else "batched and unbatched greedy generation diverged"
        )
        raise TerminalAssertionError(
            "determinism_assertion_failure",
            "generation_determinism",
            reason,
            {
                "verification_kind": (
                    "repeat_determinism"
                    if repeat_determinism
                    else "batch_vs_unbatched_equivalence"
                ),
                "checked_examples": check_count,
                "evaluation_batch_size": evaluation_batch_size,
                "comparisons": comparisons,
            },
        )
    if repeat_determinism:
        verification = {
            "status": "PASS",
            "checked_examples": check_count,
            "repeat_batch_size": 1,
            "requires_identical_response_and_extracted_answer": True,
            "comparisons": comparisons,
        }
    else:
        verification = {
            "status": "PASS",
            "checked_examples": check_count,
            "batched_size": evaluation_batch_size,
            "unbatched_size": 1,
            "requires_identical_response_and_extracted_answer": True,
            "comparisons": comparisons,
        }
    return (
        verification,
        verification_seconds,
    )


def run(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    configure_runtime()
    validate_numeric_extraction()
    replay_guard = replay_gsm8k_parser_guard()
    print(
        "GSM8K parser replay guard: "
        f"{replay_guard['status']} "
        f"({replay_guard['extraction_outcome_matches']}/{replay_guard['total']} "
        "extraction outcomes, "
        f"{replay_guard['correctness_label_matches']}/{replay_guard['total']} labels).",
        flush=True,
    )

    dataset_spec = DATASET_SPECS[args.dataset]
    public_model_name = (
        BASE_B_PUBLIC_NAME if args.successor_base_b else BASE_A_PUBLIC_NAME
    )
    canonical_result_path = result_path(dataset_spec, args.successor_base_b)
    parser_miss_path = initial_parser_miss_path(
        dataset_spec,
        args.successor_base_b,
    )
    mode = (
        "cohort_only"
        if args.cohort_only
        else ("smoke" if args.smoke is not None else "canonical")
    )
    if mode == "canonical" and canonical_result_path.exists():
        raise RuntimeError(
            f"canonical {dataset_spec.name} E1 evidence already exists and will not "
            "be overwritten"
        )

    if mode == "canonical" and args.parser_attempt == "repaired":
        try:
            parser_evidence = parser_provenance(
                mode,
                args.parser_attempt,
                parser_miss_path,
            )
            initial_artifact, _, _ = load_initial_parser_miss(parser_miss_path)
            result = re_adjudicate_initial_responses(
                initial_artifact=initial_artifact,
                parser_miss_path=parser_miss_path,
                parser_evidence=parser_evidence,
                dataset_spec=dataset_spec,
                public_model_name=public_model_name,
                successor_base_b=args.successor_base_b,
            )
        except TerminalAssertionError as error:
            return write_terminal_assertion_void(
                error=error,
                canonical_result_path=canonical_result_path,
                parser_miss_path=parser_miss_path,
                dataset_spec=dataset_spec,
                public_model_name=public_model_name,
                successor_base_b=args.successor_base_b,
            )
        except RuntimeError as error:
            return write_terminal_assertion_void(
                error=TerminalAssertionError(
                    "provenance_assertion_failure",
                    "repaired_parser_qualification",
                    str(error),
                    {"exception_type": type(error).__name__},
                ),
                canonical_result_path=canonical_result_path,
                parser_miss_path=parser_miss_path,
                dataset_spec=dataset_spec,
                public_model_name=public_model_name,
                successor_base_b=args.successor_base_b,
            )
        written_result = write_canonical_result(
            result,
            canonical_result_path,
            parser_miss_path,
        )
        if (
            written_result.get("terminal_failure_evidence", {}).get("stage")
            == "repaired_parser_write_qualification"
        ):
            print(
                json.dumps(
                    {
                        "status": written_result["status"],
                        "model": public_model_name,
                        "sample_count": written_result["sample_count"],
                        "verdict": written_result["verdict"],
                        "failure_evidence": written_result[
                            "terminal_failure_evidence"
                        ],
                        "result_path": str(
                            canonical_result_path.relative_to(HERE.parent.parent)
                        ),
                    },
                    indent=2,
                )
            )
            print(
                "ERROR: repaired outcome rejected by final qualification.",
                file=sys.stderr,
            )
            return 2
        result = written_result
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model": public_model_name,
                    "sample_count": result["sample_count"],
                    "verdict": result["verdict"],
                    "generation_performed": result["generation_performed"],
                    "immutable_response_reuse": {
                        "status": result["immutable_response_reuse"]["status"],
                        "checked_examples": result["immutable_response_reuse"][
                            "checked_examples"
                        ],
                        "identity_checks": result["immutable_response_reuse"][
                            "identity_checks"
                        ],
                    },
                    "result_path": str(
                        canonical_result_path.relative_to(HERE.parent.parent)
                    ),
                },
                indent=2,
            )
        )
        return 0

    print(
        f"Loading revision-pinned {dataset_spec.name} train/test splits.",
        flush=True,
    )
    dataset = load_dataset(
        dataset_spec.dataset_id,
        dataset_spec.config,
        revision=dataset_spec.revision,
    )
    train_split = dataset[TRAIN_SPLIT]
    test_split = dataset[TEST_SPLIT]
    try:
        if args.successor_base_b:
            canonical_indices, selection_evidence = successor_base_b_test_indices(
                len(test_split)
            )
        else:
            canonical_indices = canonical_test_indices(len(test_split))
            selection_evidence = {
                "method": "python_random_sample_without_replacement",
                "seed": SEED,
                "canonical_sample_count": CANONICAL_SAMPLE_COUNT,
                "canonical_indices_sha256": selection_digest(canonical_indices),
                "smoke_uses_canonical_prefix": True,
            }
    except RuntimeError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "cohort_assertion_failure",
                "canonical_cohort_construction",
                str(error),
                {
                    "test_split_size": len(test_split),
                    "successor_base_b": args.successor_base_b,
                },
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
        )
    selected_indices = (
        canonical_indices[: args.smoke]
        if args.smoke is not None
        else canonical_indices
    )
    try:
        leakage_check = leakage_preflight(
            train_split=train_split,
            test_split=test_split,
            selected_indices=canonical_indices,
            dataset_spec=dataset_spec,
        )
        dataset_provenance = {
            "train_split_fingerprint": train_split._fingerprint,
            "test_split_fingerprint": test_split._fingerprint,
            "demonstrations_content_sha256": rows_content_digest(
                train_split,
                DEMONSTRATION_INDICES,
                dataset_spec,
            ),
            "canonical_cohort_content_sha256": rows_content_digest(
                test_split,
                canonical_indices,
                dataset_spec,
            ),
            "evaluated_cohort_content_sha256": rows_content_digest(
                test_split,
                selected_indices,
                dataset_spec,
            ),
        }
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError) as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "provenance_assertion_failure",
                "dataset_and_prompt_provenance",
                str(error),
                {
                    "canonical_index_count": len(canonical_indices),
                    "evaluated_index_count": len(selected_indices),
                },
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
        )
    selection_evidence.update(
        {
            "selected_row_content_sha256": dataset_provenance[
                "canonical_cohort_content_sha256"
            ],
            "dataset_split_fingerprint": dataset_provenance[
                "test_split_fingerprint"
            ],
        }
    )
    if args.cohort_only:
        print(
            json.dumps(
                {
                    "status": "COHORT_ASSERTIONS_PASS",
                    "model": public_model_name,
                    "dataset": dataset_spec.name,
                    "dataset_revision": dataset_spec.revision,
                    "selection": selection_evidence,
                    "dataset_provenance": dataset_provenance,
                    "generation_performed": False,
                    "model_coordinates_read": False,
                },
                indent=2,
            )
        )
        return 0

    try:
        parser_evidence = parser_provenance(
            mode,
            args.parser_attempt,
            parser_miss_path,
        )
    except RuntimeError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "provenance_assertion_failure",
                "parser_provenance",
                str(error),
                {"parser_attempt": args.parser_attempt},
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
        )
    if leakage_check["status"] == "FAIL":
        result = leakage_void_result(
            trigger_mode=mode,
            canonical_indices=canonical_indices,
            dataset_provenance=dataset_provenance,
            leakage_check=leakage_check,
            parser_evidence=parser_evidence,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            selection_evidence=selection_evidence,
        )
        write_canonical_result(
            result,
            canonical_result_path,
            parser_miss_path,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model": public_model_name,
                    "sample_count": result["sample_count"],
                    "verdict": result["verdict"],
                    "leakage_evidence": leakage_check,
                    "result_path": str(
                        canonical_result_path.relative_to(HERE.parent.parent)
                    ),
                },
                indent=2,
            )
        )
        return 2

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for the {public_model_name} task-band evaluator"
        )
    try:
        model_identifier, model_revision, recorded_content_digest = (
            load_private_model_coordinates(public_model_name)
        )
        validate_local_checkpoint_snapshot(
            model_identifier=model_identifier,
            model_revision=model_revision,
            recorded_content_digest=recorded_content_digest,
            public_model_name=public_model_name,
        )
    except RuntimeError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "provenance_assertion_failure",
                "frozen_checkpoint_provenance",
                str(error),
                {
                    "public_model_name": public_model_name,
                    "private_coordinates_serialized": False,
                },
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
        )
    print(f"Loading {public_model_name} on CUDA.", flush=True)
    device = torch.device("cuda:0")
    tokenizer, model = load_frozen_base(
        model_identifier,
        model_revision,
        device,
        public_model_name,
    )

    batch_size = evaluation_batch_size(dataset_spec, args.successor_base_b)
    print(
        (
            f"Evaluating {len(selected_indices)} example(s) in {mode} mode "
            f"with batch size {batch_size}."
        ),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    try:
        records, evaluation_wall_time_seconds, generation_wall_time_seconds = (
            evaluate_examples(
                train_split=train_split,
                test_split=test_split,
                selected_indices=selected_indices,
                dataset_spec=dataset_spec,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=batch_size,
                public_model_name=public_model_name,
                successor_base_b=args.successor_base_b,
            )
        )
    except TerminalAssertionError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=error,
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
        )
    except (KeyError, TypeError, ValueError) as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "accounting_assertion_failure",
                "per_example_record_construction",
                "per-example generation records failed accounting construction",
                {"exception_type": type(error).__name__},
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
        )
    if uses_repeat_determinism(dataset_spec, args.successor_base_b):
        print("Checking two greedy answers for repeat determinism.", flush=True)
    else:
        print("Checking two greedy answers batched versus unbatched.", flush=True)
    try:
        batch_equivalence, verification_wall_time_seconds = verify_batch_equivalence(
            batched_records=records,
            train_split=train_split,
            test_split=test_split,
            selected_indices=selected_indices,
            dataset_spec=dataset_spec,
            evaluation_batch_size=batch_size,
            tokenizer=tokenizer,
            model=model,
            device=device,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
        )
    except TerminalAssertionError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=error,
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
            records=records,
        )
    except (KeyError, TypeError, ValueError) as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "determinism_assertion_failure",
                "generation_determinism",
                "determinism comparison evidence was structurally invalid",
                {"exception_type": type(error).__name__},
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
            records=records,
        )
    device_total_bytes = torch.cuda.get_device_properties(device).total_memory
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    vram = {
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "device_total_bytes": device_total_bytes,
        "peak_allocated_fraction": peak_allocated_bytes / device_total_bytes,
        "peak_reserved_fraction": peak_reserved_bytes / device_total_bytes,
        "maximum_allowed_fraction": MAX_VRAM_FRACTION,
    }
    if vram["peak_reserved_fraction"] >= MAX_VRAM_FRACTION:
        raise RuntimeError("peak reserved VRAM reached the repository safety limit")

    wall_time_seconds = time.perf_counter() - run_started
    try:
        result = summarize(
            records=records,
            wall_time_seconds=wall_time_seconds,
            evaluation_wall_time_seconds=evaluation_wall_time_seconds,
            generation_wall_time_seconds=generation_wall_time_seconds,
            verification_wall_time_seconds=verification_wall_time_seconds,
            canonical_indices=canonical_indices,
            mode=mode,
            parser_evidence=parser_evidence,
            dataset_provenance=dataset_provenance,
            leakage_check=leakage_check,
            batch_equivalence=batch_equivalence,
            vram=vram,
            dataset_spec=dataset_spec,
            parser_miss_path=parser_miss_path,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
        )
    except TerminalAssertionError as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=error,
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
            records=records,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        if mode != "canonical":
            raise
        return write_terminal_assertion_void(
            error=TerminalAssertionError(
                "accounting_assertion_failure",
                "canonical_summary_accounting",
                "canonical summary accounting was structurally invalid",
                {"exception_type": type(error).__name__},
            ),
            canonical_result_path=canonical_result_path,
            parser_miss_path=parser_miss_path,
            dataset_spec=dataset_spec,
            public_model_name=public_model_name,
            successor_base_b=args.successor_base_b,
            selection_evidence=selection_evidence,
            dataset_provenance=dataset_provenance,
            parser_evidence=parser_evidence,
            records=records,
        )

    if mode == "canonical":
        wrote_result = result["verdict"]["token"] != "PARSER-REPAIR-REQUIRED"
        if wrote_result:
            write_canonical_result(
                result,
                canonical_result_path,
                parser_miss_path,
            )
            written_result_path = canonical_result_path
        else:
            write_initial_parser_miss(result, parser_miss_path)
            written_result_path = parser_miss_path
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model": public_model_name,
                    "sample_count": result["sample_count"],
                    "verdict": result["verdict"],
                    "result_path": str(
                        written_result_path.relative_to(HERE.parent.parent)
                    ),
                    "canonical_result_written": wrote_result,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(smoke_console_summary(result, replay_guard), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("ERROR: task-band evaluation interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        # Exception messages from dependency clients can contain the private
        # identifier. Report only the public alias and exception class.
        print(
            f"ERROR: task-band evaluation failed ({type(error).__name__}).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
