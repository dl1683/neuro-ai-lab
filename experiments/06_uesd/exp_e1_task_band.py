"""E1 task-band evaluator for the preregistered semantic-ratchet direction.

Canonical protocol (``--dataset gsm8k`` by default):
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

The exact base-A identifier is private repository-local configuration. This
script reads it from the gitignored ``_local_manifest.md`` and never logs or
serializes it.
"""

from __future__ import annotations

import argparse
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
from decimal import Decimal, InvalidOperation
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
PUBLIC_MODEL_NAME = "base-A"
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

HERE = Path(__file__).resolve().parent
LOCAL_MANIFEST_PATH = HERE / "_local_manifest.md"

UNSIGNED_DECIMAL_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUMBER_PATTERN = (
    rf"[-+]?{UNSIGNED_DECIMAL_PATTERN}"
    rf"(?:\s*/\s*[-+]?{UNSIGNED_DECIMAL_PATTERN})?"
)
ANSWER_BOUNDARY_LINE_RE = re.compile(r"(?m)^[ \t]*####[^\r\n]*(?:\r?$)")
ANY_NUMBER_RE = re.compile(rf"(?<!/)({NUMBER_PATTERN})(?!/)")
NEW_QUESTION_RE = re.compile(r"(?m)^\s*Question\s*:")
MANIFEST_MODEL_ENTRY_RE = re.compile(r"^-\s*`base-A`:\s*`([^`]+)`\s*$")
MANIFEST_REVISION_ENTRY_RE = re.compile(
    r"^-\s*`base-A-revision`:\s*`([0-9a-f]{40})`\s*$"
)

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
            "mark the canonical parser attempt; only a repeated extraction "
            "miss on 'repaired' may emit terminal VOID"
        ),
    )
    args = parser.parse_args(argv)
    if args.smoke is not None and not 1 <= args.smoke <= CANONICAL_SAMPLE_COUNT:
        parser.error(f"--smoke N must satisfy 1 <= N <= {CANONICAL_SAMPLE_COUNT}")
    if args.smoke is not None and args.parser_attempt != "initial":
        parser.error("--parser-attempt repaired is only valid for the canonical run")
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


def load_private_model_coordinates() -> tuple[str, str]:
    if not LOCAL_MANIFEST_PATH.is_file():
        raise RuntimeError("the gitignored local manifest for base-A is missing")

    model_matches = []
    revision_matches = []
    for line in LOCAL_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        model_match = MANIFEST_MODEL_ENTRY_RE.fullmatch(stripped)
        if model_match:
            model_matches.append(model_match.group(1).strip())
        revision_match = MANIFEST_REVISION_ENTRY_RE.fullmatch(stripped)
        if revision_match:
            revision_matches.append(revision_match.group(1))

    if len(model_matches) != 1 or not model_matches[0]:
        raise RuntimeError("the local manifest must contain exactly one base-A entry")
    if len(revision_matches) != 1:
        raise RuntimeError(
            "the local manifest must contain exactly one base-A-revision entry"
        )
    return model_matches[0], revision_matches[0]


def result_path(dataset_spec: DatasetSpec) -> Path:
    return HERE / "results" / dataset_spec.result_filename


def initial_parser_miss_path(dataset_spec: DatasetSpec) -> Path:
    return HERE / "results" / dataset_spec.initial_parser_miss_filename


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
    candidate = re.sub(r"\s+", "", raw.replace(",", ""))
    parts = candidate.split("/")
    if len(parts) not in (1, 2) or any(not part for part in parts):
        return None
    try:
        decimal_parts = [Decimal(part) for part in parts]
    except InvalidOperation:
        return None
    if any(not part.is_finite() for part in decimal_parts):
        return None
    value = Fraction(decimal_parts[0])
    if len(decimal_parts) == 2:
        denominator_value = Fraction(decimal_parts[1])
        if denominator_value == 0:
            return None
        value /= denominator_value
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

    numbers = ANY_NUMBER_RE.findall(segment)
    for raw in reversed(numbers):
        normalized = normalize_number(raw)
        if normalized is not None:
            return normalized, "last_numeric_token_in_first_segment", segment, segment_stop_reason
    return None, None, segment, segment_stop_reason


def validate_numeric_extraction() -> None:
    exact_cases = (
        ("32.0", "32"),
        ("0.5", "0.5"),
        ("1/2", "0.5"),
        ("392/196", "2"),
    )
    for raw, expected in exact_cases:
        if normalize_number(raw) != expected:
            raise RuntimeError("numeric normalizer self-check failed")
        if extract_gold_answer(f"#### {raw}") != expected:
            raise RuntimeError("gold-answer parser self-check failed")
        primary, _, _, _ = extract_predicted_answer(
            f"#### {raw}",
            "end_of_message",
        )
        fallback, _, _, _ = extract_predicted_answer(
            f"Answer: {raw}",
            "end_of_message",
        )
        if primary != expected or fallback != expected:
            raise RuntimeError("prediction parser self-check failed")

    negative_cases = ("3/", "3//2", "3/x", "/2", "3/0")
    for raw in negative_cases:
        if normalize_number(raw) is not None:
            raise RuntimeError("numeric normalizer negative self-check failed")
        try:
            extract_gold_answer(f"#### {raw}")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("gold-answer parser negative self-check failed")
        primary, _, _, _ = extract_predicted_answer(
            f"#### {raw}",
            "end_of_message",
        )
        fallback, _, _, _ = extract_predicted_answer(
            f"Answer: {raw}",
            "end_of_message",
        )
        if primary is not None or fallback is not None:
            raise RuntimeError("prediction parser negative self-check failed")


def parser_source_text() -> str:
    components = (
        normalize_number,
        first_answer_segment,
        extract_predicted_answer,
    )
    return "\n\n".join(inspect.getsource(component) for component in components)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "normalize_number",
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
        repaired_attempt_is_qualified = (
            parser_evidence["attempt"] == "repaired"
            and parser_evidence.get("source_changed_from_initial") is True
            and bool(parser_evidence.get("initial_miss_artifact_sha256"))
            and parser_miss_path.is_file()
        )
        if not repaired_attempt_is_qualified:
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


def load_base_a(
    model_identifier: str,
    model_revision: str,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        revision=model_revision,
        trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError("base-A tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("base-A tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def validate_context_window(model, prompt_tokens: int) -> None:
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and prompt_tokens + MAX_NEW_TOKENS > context_limit:
        raise RuntimeError("the fixed five-shot prompt exceeds base-A's context window")


def evaluate_examples(
    train_split: Dataset,
    test_split: Dataset,
    selected_indices: Sequence[int],
    dataset_spec: DatasetSpec,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
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
        validate_context_window(model, generation_start)
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
                    "extraction_failed": predicted is None,
                    "valid_extracted_incorrect": (
                        predicted is not None and predicted != gold
                    ),
                    "correct": predicted == gold,
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
) -> dict[str, Any]:
    generation_verification_field = (
        "repeat_determinism"
        if dataset_spec.key == "svamp"
        else "batch_vs_unbatched_equivalence"
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
    if mode == "canonical":
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
        "model": PUBLIC_MODEL_NAME,
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
            "selection": {
                "method": "python_random_sample_without_replacement",
                "seed": SEED,
                "canonical_sample_count": CANONICAL_SAMPLE_COUNT,
                "canonical_indices_sha256": selection_digest(canonical_indices),
                "smoke_uses_canonical_prefix": True,
            },
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
                "batch_size": dataset_spec.batch_size,
                "padding_side": "left",
                "per_sequence_stops": [
                    "end_of_message",
                    "new_question_boundary",
                    "max_new_tokens",
                ],
                generation_verification_field: batch_equivalence,
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
            "thresholds": {
                "correct_count_inclusive": [PASS_MIN_CORRECT, PASS_MAX_CORRECT],
                "minimum_correct_population": MIN_CORRECT_POPULATION,
                "minimum_valid_extracted_incorrect_population": (
                    MIN_INCORRECT_POPULATION
                ),
                "maximum_extraction_failure_rate": MAX_EXTRACTION_FAILURE_RATE,
            },
        },
        "sample_count": denominator,
        "correct_count": correct_count,
        "exact_answer_failure_count": exact_answer_failure_count,
        "valid_extracted_incorrect_count": valid_extracted_incorrect_count,
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
) -> dict[str, Any]:
    if leakage_check["status"] != "FAIL" or leakage_check["overlap_count"] < 1:
        raise RuntimeError("leakage VOID requires positive leakage evidence")
    return {
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "mode": "canonical",
        "trigger_mode": trigger_mode,
        "model": PUBLIC_MODEL_NAME,
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
            "selection": {
                "method": "python_random_sample_without_replacement",
                "seed": SEED,
                "canonical_sample_count": CANONICAL_SAMPLE_COUNT,
                "canonical_indices_sha256": selection_digest(canonical_indices),
            },
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
) -> None:
    verdict = result.get("verdict", {})
    if verdict.get("token") == "VOID":
        reason = verdict.get("reason")
        if reason == "repaired_extraction_failure_rate_above_5_percent":
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
                parser_evidence.get("attempt") == "repaired"
                and parser_evidence.get("source_changed_from_initial") is True
                and parser_miss_path.is_file()
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
        elif reason == "demonstration_or_cohort_leakage_detected":
            leakage_check = result.get("protocol", {}).get("leakage_preflight", {})
            if (
                leakage_check.get("status") != "FAIL"
                or leakage_check.get("overlap_count", 0) < 1
                or not leakage_check.get("overlaps")
            ):
                raise RuntimeError("terminal leakage VOID lacks leakage evidence")
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


def smoke_console_summary(result: dict[str, Any]) -> dict[str, Any]:
    generation_verification_field = (
        "repeat_determinism"
        if result["protocol"]["dataset_key"] == "svamp"
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
        generation_verification_field: result["protocol"]["decoding"][
            generation_verification_field
        ],
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
) -> tuple[dict[str, Any], float]:
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
        if dataset_spec.key == "svamp":
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
        if dataset_spec.key == "svamp":
            raise RuntimeError("repeated batch-1 greedy generation diverged")
        raise RuntimeError("batched and unbatched greedy generation diverged")
    if dataset_spec.key == "svamp":
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

    dataset_spec = DATASET_SPECS[args.dataset]
    canonical_result_path = result_path(dataset_spec)
    parser_miss_path = initial_parser_miss_path(dataset_spec)
    mode = "smoke" if args.smoke is not None else "canonical"
    if mode == "canonical" and canonical_result_path.exists():
        raise RuntimeError(
            f"canonical {dataset_spec.name} E1 evidence already exists and will not "
            "be overwritten"
        )
    parser_evidence = parser_provenance(
        mode,
        args.parser_attempt,
        parser_miss_path,
    )

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
    canonical_indices = canonical_test_indices(len(test_split))
    selected_indices = (
        canonical_indices[: args.smoke]
        if args.smoke is not None
        else canonical_indices
    )
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
    if leakage_check["status"] == "FAIL":
        result = leakage_void_result(
            trigger_mode=mode,
            canonical_indices=canonical_indices,
            dataset_provenance=dataset_provenance,
            leakage_check=leakage_check,
            parser_evidence=parser_evidence,
            dataset_spec=dataset_spec,
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
                    "model": PUBLIC_MODEL_NAME,
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
        raise RuntimeError("CUDA is required for the base-A task-band evaluator")
    model_identifier, model_revision = load_private_model_coordinates()
    print(f"Loading {PUBLIC_MODEL_NAME} on CUDA.", flush=True)
    device = torch.device("cuda:0")
    tokenizer, model = load_base_a(model_identifier, model_revision, device)

    print(
        (
            f"Evaluating {len(selected_indices)} example(s) in {mode} mode "
            f"with batch size {dataset_spec.batch_size}."
        ),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    records, evaluation_wall_time_seconds, generation_wall_time_seconds = (
        evaluate_examples(
            train_split=train_split,
            test_split=test_split,
            selected_indices=selected_indices,
            dataset_spec=dataset_spec,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=dataset_spec.batch_size,
        )
    )
    if dataset_spec.key == "svamp":
        print("Checking two greedy answers for repeat determinism.", flush=True)
    else:
        print("Checking two greedy answers batched versus unbatched.", flush=True)
    batch_equivalence, verification_wall_time_seconds = verify_batch_equivalence(
        batched_records=records,
        train_split=train_split,
        test_split=test_split,
        selected_indices=selected_indices,
        dataset_spec=dataset_spec,
        evaluation_batch_size=dataset_spec.batch_size,
        tokenizer=tokenizer,
        model=model,
        device=device,
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
                    "model": PUBLIC_MODEL_NAME,
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
        print(json.dumps(smoke_console_summary(result), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("ERROR: base-A task-band evaluation interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        # Exception messages from dependency clients can contain the private
        # identifier. Report only the public alias and exception class.
        print(
            f"ERROR: base-A task-band evaluation failed ({type(error).__name__}).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
