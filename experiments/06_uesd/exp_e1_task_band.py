"""E1 task-band evaluator for the preregistered semantic-ratchet direction.

Canonical protocol (no arguments):
  - deterministically select 256 examples without replacement from the
    official GSM8K test split using seed 20260809;
  - use the first five official training examples as a fixed five-shot prompt;
  - decode greedily from frozen base-A with at most 256 new tokens;
  - apply a fixed exact-numeric extractor and compute the preregistered gate;
  - create results/exp_e1_task_band.json, refusing to overwrite evidence.

Smoke protocol (``--smoke N``):
  - evaluate the first N examples of the canonical selected cohort through the
    identical data, prompt, generation, and extraction path;
  - report diagnostic metrics to stdout with verdict SMOKE_ONLY;
  - never write the canonical result artifact.

The exact base-A identifier is private repository-local configuration. This
script reads it from the gitignored ``_local_manifest.md`` and never logs or
serializes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

# Keep dependency output free of private checkpoint identifiers and make smoke
# output machine-readable. These must be set before importing hub clients.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from datasets import Dataset, disable_progress_bar, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


EXPERIMENT = "E1 task-band gate"
PUBLIC_MODEL_NAME = "base-A"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"

SEED = 20260809
CANONICAL_SAMPLE_COUNT = 256
DEMONSTRATION_INDICES = (0, 1, 2, 3, 4)
MAX_NEW_TOKENS = 256

PASS_MIN_CORRECT = 26
PASS_MAX_CORRECT = 217
MIN_CORRECT_POPULATION = 40
MIN_INCORRECT_POPULATION = 40
MAX_EXTRACTION_FAILURE_RATE = 0.05

HERE = Path(__file__).resolve().parent
LOCAL_MANIFEST_PATH = HERE / "_local_manifest.md"
RESULT_PATH = HERE / "results" / "exp_e1_task_band.json"

NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
HASH_ANSWER_RE = re.compile(rf"####\s*({NUMBER_PATTERN})")
ANY_NUMBER_RE = re.compile(NUMBER_PATTERN)
MANIFEST_ENTRY_RE = re.compile(r"^-\s*`base-A`:\s*`([^`]+)`\s*$")

PROMPT_PREAMBLE = (
    "Solve each grade-school math problem step by step. End every answer with "
    "a separate line in exactly this form:\n#### <numeric answer>"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        type=int,
        metavar="N",
        help=(
            "run the first N examples of the canonical cohort without writing "
            "the canonical result artifact"
        ),
    )
    args = parser.parse_args(argv)
    if args.smoke is not None and not 1 <= args.smoke <= CANONICAL_SAMPLE_COUNT:
        parser.error(f"--smoke N must satisfy 1 <= N <= {CANONICAL_SAMPLE_COUNT}")
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


def load_private_model_identifier() -> str:
    if not LOCAL_MANIFEST_PATH.is_file():
        raise RuntimeError("the gitignored local manifest for base-A is missing")

    matches = []
    for line in LOCAL_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_ENTRY_RE.fullmatch(line.strip())
        if match:
            matches.append(match.group(1).strip())

    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("the local manifest must contain exactly one base-A entry")
    return matches[0]


def canonical_test_indices(split_size: int) -> list[int]:
    if split_size < CANONICAL_SAMPLE_COUNT:
        raise RuntimeError("the official test split is smaller than the canonical cohort")
    rng = random.Random(SEED)
    return rng.sample(range(split_size), CANONICAL_SAMPLE_COUNT)


def selection_digest(indices: Sequence[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_five_shot_prefix(train_split: Dataset) -> str:
    if len(train_split) <= max(DEMONSTRATION_INDICES):
        raise RuntimeError("the official training split lacks fixed demonstrations")

    blocks = [PROMPT_PREAMBLE]
    for index in DEMONSTRATION_INDICES:
        row = train_split[index]
        blocks.append(f"Question: {row['question'].strip()}\nAnswer:\n{row['answer'].strip()}")
    return "\n\n".join(blocks)


def build_prompt(five_shot_prefix: str, question: str) -> str:
    return f"{five_shot_prefix}\n\nQuestion: {question.strip()}\nAnswer:\n"


def normalize_number(raw: str) -> str | None:
    candidate = raw.replace(",", "").strip()
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def extract_gold_answer(answer: str) -> str:
    matches = HASH_ANSWER_RE.findall(answer)
    if not matches:
        raise RuntimeError("an official GSM8K answer lacks its numeric marker")
    normalized = normalize_number(matches[-1])
    if normalized is None:
        raise RuntimeError("an official GSM8K numeric answer is invalid")
    return normalized


def extract_predicted_answer(response: str) -> tuple[str | None, str | None]:
    marked = HASH_ANSWER_RE.findall(response)
    if marked:
        normalized = normalize_number(marked[-1])
        if normalized is not None:
            return normalized, "hash_answer"

    numbers = ANY_NUMBER_RE.findall(response)
    for raw in reversed(numbers):
        normalized = normalize_number(raw)
        if normalized is not None:
            return normalized, "last_numeric_token"
    return None, None


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
    incorrect_count: int,
    extraction_failure_rate: float,
) -> dict[str, Any]:
    if extraction_failure_rate > MAX_EXTRACTION_FAILURE_RATE:
        return {
            "token": "VOID",
            "reason": "extraction_failure_rate_above_5_percent",
            "next_action": "repair the parser once; a repeated miss remains VOID",
        }

    low_band = correct_count < PASS_MIN_CORRECT or correct_count < MIN_CORRECT_POPULATION
    high_band = (
        correct_count > PASS_MAX_CORRECT
        or incorrect_count < MIN_INCORRECT_POPULATION
    )
    if low_band:
        return {
            "token": "ABORT-AND-SWAP",
            "reason": "below_band_or_fewer_than_40_correct",
            "next_action": "swap to SVAMP and rerun the same 256-example gate once",
        }
    if high_band:
        return {
            "token": "ABORT-AND-SWAP",
            "reason": "above_band_or_fewer_than_40_incorrect",
            "next_action": "swap to GSM-Hard and rerun the same 256-example gate once",
        }
    return {
        "token": "PASS",
        "reason": "all_preregistered_task_band_thresholds_satisfied",
        "next_action": "task-band gate permits the separately reviewed mechanics pilot",
    }


def load_base_a(model_identifier: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("base-A tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
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
    test_split: Dataset,
    selected_indices: Sequence[int],
    five_shot_prefix: str,
    tokenizer,
    model,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for cohort_position, dataset_index in enumerate(selected_indices):
        row = test_split[dataset_index]
        prompt = build_prompt(five_shot_prefix, row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        validate_context_window(model, input_ids.shape[1])

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
            )
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started

        generated_ids = output_ids[0, input_ids.shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted, extraction_source = extract_predicted_answer(response)
        gold = extract_gold_answer(row["answer"])

        records.append(
            {
                "cohort_position": cohort_position,
                "dataset_index": dataset_index,
                "question": row["question"],
                "gold_answer": gold,
                "response": response,
                "extracted_answer": predicted,
                "extraction_source": extraction_source,
                "extraction_failed": predicted is None,
                "correct": predicted == gold,
                "generated_tokens": int(generated_ids.numel()),
                "generation_seconds": generation_seconds,
            }
        )

    return records, time.perf_counter() - started


def summarize(
    records: Sequence[dict[str, Any]],
    wall_time_seconds: float,
    evaluation_wall_time_seconds: float,
    canonical_indices: Sequence[int],
    mode: str,
) -> dict[str, Any]:
    denominator = len(records)
    correct_count = sum(bool(record["correct"]) for record in records)
    incorrect_count = denominator - correct_count
    failure_count = sum(bool(record["extraction_failed"]) for record in records)
    generated_lengths = [int(record["generated_tokens"]) for record in records]
    total_generated_tokens = sum(generated_lengths)
    generation_seconds = sum(
        float(record["generation_seconds"]) for record in records
    )
    setup_wall_time_seconds = max(0.0, wall_time_seconds - evaluation_wall_time_seconds)

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
            incorrect_count=incorrect_count,
            extraction_failure_rate=extraction_failure_rate,
        )
    else:
        verdict = {
            "token": "SMOKE_ONLY",
            "reason": "canonical thresholds require exactly 256 examples",
            "next_action": "obtain independent pipeline review before the full gate",
        }

    return {
        "experiment": EXPERIMENT,
        "status": "COMPLETE" if mode == "canonical" else "SMOKE_ONLY",
        "mode": mode,
        "model": PUBLIC_MODEL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "GSM8K",
            "dataset_config": DATASET_CONFIG,
            "evaluation_split": TEST_SPLIT,
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
            "decoding": {
                "strategy": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": "bfloat16",
                "device": "cuda",
            },
            "answer_extraction": {
                "primary": "last numeric value after the final #### marker",
                "fallback": "last numeric token in the response",
                "normalization": "remove thousands separators and compare finite decimals exactly",
            },
            "thresholds": {
                "correct_count_inclusive": [PASS_MIN_CORRECT, PASS_MAX_CORRECT],
                "minimum_correct_population": MIN_CORRECT_POPULATION,
                "minimum_incorrect_population": MIN_INCORRECT_POPULATION,
                "maximum_extraction_failure_rate": MAX_EXTRACTION_FAILURE_RATE,
            },
        },
        "sample_count": denominator,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
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
        "answer_frequency_distribution": frequency_distribution,
        "wall_time_seconds": wall_time_seconds,
        "setup_wall_time_seconds": setup_wall_time_seconds,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "generation_wall_time_seconds": generation_seconds,
        "throughput_generated_tokens_per_second": (
            total_generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        ),
        "verdict": verdict,
        "per_example": list(records),
    }


def write_canonical_result(result: dict[str, Any]) -> None:
    if RESULT_PATH.exists():
        raise RuntimeError("canonical E1 evidence already exists and will not be overwritten")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(result, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        if RESULT_PATH.exists():
            raise RuntimeError("canonical E1 evidence appeared during the run")
        temporary_path.replace(RESULT_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def smoke_console_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "model": result["model"],
        "sample_count": result["sample_count"],
        "correct_count": result["correct_count"],
        "exact_answer_accuracy": result["exact_answer_accuracy"],
        "extraction_failures": result["extraction_failures"],
        "response_length_tokens": result["response_length_tokens"],
        "wall_time_seconds": result["wall_time_seconds"],
        "setup_wall_time_seconds": result["setup_wall_time_seconds"],
        "evaluation_wall_time_seconds": result["evaluation_wall_time_seconds"],
        "generation_wall_time_seconds": result["generation_wall_time_seconds"],
        "throughput_generated_tokens_per_second": result[
            "throughput_generated_tokens_per_second"
        ],
        "projected_256_wall_time_seconds": (
            result["setup_wall_time_seconds"]
            + result["evaluation_wall_time_seconds"]
            * CANONICAL_SAMPLE_COUNT
            / result["sample_count"]
        ),
        "verdict": result["verdict"],
    }


def run(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the base-A task-band evaluator")

    mode = "smoke" if args.smoke is not None else "canonical"
    if mode == "canonical" and RESULT_PATH.exists():
        raise RuntimeError("canonical E1 evidence already exists and will not be overwritten")

    model_identifier = load_private_model_identifier()
    print(f"Loading {PUBLIC_MODEL_NAME} on CUDA.", flush=True)
    device = torch.device("cuda:0")
    tokenizer, model = load_base_a(model_identifier, device)

    print("Loading official GSM8K splits.", flush=True)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    train_split = dataset[TRAIN_SPLIT]
    test_split = dataset[TEST_SPLIT]
    canonical_indices = canonical_test_indices(len(test_split))
    selected_indices = (
        canonical_indices[: args.smoke]
        if args.smoke is not None
        else canonical_indices
    )
    five_shot_prefix = build_five_shot_prefix(train_split)

    print(
        f"Evaluating {len(selected_indices)} example(s) in {mode} mode.",
        flush=True,
    )
    records, evaluation_wall_time_seconds = evaluate_examples(
        test_split=test_split,
        selected_indices=selected_indices,
        five_shot_prefix=five_shot_prefix,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    wall_time_seconds = time.perf_counter() - run_started
    result = summarize(
        records=records,
        wall_time_seconds=wall_time_seconds,
        evaluation_wall_time_seconds=evaluation_wall_time_seconds,
        canonical_indices=canonical_indices,
        mode=mode,
    )

    if mode == "canonical":
        write_canonical_result(result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model": PUBLIC_MODEL_NAME,
                    "sample_count": result["sample_count"],
                    "verdict": result["verdict"],
                    "result_path": str(RESULT_PATH.relative_to(HERE.parent.parent)),
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
