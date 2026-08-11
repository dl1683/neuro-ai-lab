"""Execute the preregistered E2-DIAG competence diagnostics.

Only Stage 0 is implemented here.  The diagnostic reuses the frozen E2
generator, tokenizer, common recurrent model, optimizer, batching, and
autocast machinery without changing the E2 mechanics runner.

Stage-0 PASS is not a completed E2-DIAG suite.  In that case this runner
stores untracked recovery provenance under ``checkpoints/exp_e2_diag`` and
does not publish the suite's sole immutable result artifact.  Stage-0 STOP or
an operational/integrity VOID is terminal and publishes ``exp_e2_diag.json``
with atomic no-clobber semantics.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import exp_e2_latch_mechanics as e2


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PREREGISTRATION = HERE / "E2_DIAG_PREREGISTRATION.md"
E2_CONFIG = HERE / "exp_e2_latch_mechanics_config.json"
RESULT_PATH = HERE / "results" / "exp_e2_diag.json"
REPAIR_RESULT_PATH = HERE / "results" / "exp_e2_diag_stage0_instrumented.json"
RECOVERY_DIR = HERE / "checkpoints" / "exp_e2_diag"
LAUNCH_MARKER = RECOVERY_DIR / "stage0_launch.json"
RECOVERY_RECORD = RECOVERY_DIR / "stage0_provenance.json"
RECOVERY_CHECKPOINT = RECOVERY_DIR / "stage0_final.pt"
STDOUT_RECOVERY = RECOVERY_DIR / "stage0_stdout_recovery.json"
REPAIR_RECOVERY_DIR = HERE / "checkpoints" / "exp_e2_diag_stage0_instrumented"
REPAIR_LAUNCH_MARKER = REPAIR_RECOVERY_DIR / "stage0_launch.json"
REPAIR_RECOVERY_RECORD = REPAIR_RECOVERY_DIR / "stage0_provenance.json"
REPAIR_RECOVERY_CHECKPOINT = REPAIR_RECOVERY_DIR / "stage0_final.pt"

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "exp_e2_diag"
MODEL_SEED = 42
CONTROLLER_TRAIN_EXAMPLES = 8192
MEMORIZATION_EXAMPLES = 128
EXAMPLES_PER_LABEL = 32
MAX_UPDATES = 3000
EVAL_INTERVAL = 100
GATE_CORRECT = 122
TARGET_TOKENS_PER_UPDATE = 1024
WALL_TIME_CAP_SECONDS = 29 * 60
REVIEW_ATTESTATION = "E2_DIAG_STAGE0_PRELAUNCH_REVIEW_CLEAN"
REPAIR_REVIEW_ATTESTATION = "E2_DIAG_STAGE0_OPERATIONAL_REPAIR_REVIEW_CLEAN"

GRADIENT_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "encoder": (
        "embedding",
        "position",
        "encoder",
        "encoder_norm",
        "prompt_to_plan",
    ),
    "controller": ("controller_layers", "controller_norm"),
    "plan_slots": ("plan_slots",),
    "prefix_projector": ("prefix_projector",),
    "answer_decoder": ("answer_decoder", "answer_query"),
    "readout_head": ("answer_norm", "choice_head"),
}


class IntegrityFailure(RuntimeError):
    """Raised when a frozen Stage-0 integrity condition is false."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("ascii"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json_no_clobber(payload: Mapping[str, Any], path: Path) -> None:
    """Publish JSON through same-directory fsync and a no-clobber hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"immutable file already exists: {path}") from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
    }


def _not_run(reason: str) -> dict[str, str]:
    return {"status": "NOT_RUN", "reason": reason}


def _example_identity(example: e2.DeductionExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "skeleton_hash": example.skeleton_hash,
        "counterfactual_member": example.counterfactual_member,
        "answer_position": example.answer_position,
        "rendered_text_sha256": _sha256_bytes(example.rendered_text.encode("utf-8")),
    }


def _generate_controller_train(
    config: Mapping[str, Any],
) -> tuple[list[e2.DeductionExample], dict[str, Any]]:
    """Generate only the public controller-training split; never touch test."""

    generator = e2.DeductionGenerator(config)
    examples = generator._generate_split(  # noqa: SLF001 - frozen generator path
        "controller_train",
        CONTROLLER_TRAIN_EXAMPLES,
        set(),
        seed_offset=0,
    )
    audit = generator.audit_dataset({"controller_train": examples})
    return examples, audit


def _select_memorization_set(
    examples: Sequence[e2.DeductionExample],
) -> list[e2.DeductionExample]:
    grouped: dict[int, list[e2.DeductionExample]] = {
        answer_position: [] for answer_position in range(4)
    }
    for example in examples:
        grouped[example.answer_position].append(example)
    selected: list[e2.DeductionExample] = []
    for answer_position in range(4):
        ordered = sorted(
            grouped[answer_position],
            key=lambda example: (example.skeleton_hash, example.example_id),
        )
        if len(ordered) < EXAMPLES_PER_LABEL:
            raise IntegrityFailure(
                f"answer position {answer_position} has only {len(ordered)} examples"
            )
        selected.extend(ordered[:EXAMPLES_PER_LABEL])
    if len(selected) != MEMORIZATION_EXAMPLES:
        raise IntegrityFailure("memorization set does not contain exactly 128 examples")
    return selected


def _tokenizer_integrity(
    examples: Sequence[e2.DeductionExample], tokenizer: e2.LocalTokenizer
) -> dict[str, Any]:
    round_trip_passes = 0
    encoded_without_unknown = 0
    for example in examples:
        if tokenizer.round_trip_tokens(example.rendered_text):
            round_trip_passes += 1
        encoded = tokenizer.encode(example.rendered_text)
        if tokenizer.unk_id not in encoded:
            encoded_without_unknown += 1
    return {
        "round_trip": _rate(round_trip_passes, len(examples)),
        "unknown_token_free": _rate(encoded_without_unknown, len(examples)),
        "pass": round_trip_passes == len(examples)
        and encoded_without_unknown == len(examples),
    }


def _build_training_batches(
    examples: Sequence[e2.DeductionExample], tokenizer: e2.LocalTokenizer
) -> tuple[list[list[int]], list[int], int]:
    nominal_budget = MAX_UPDATES * TARGET_TOKENS_PER_UPDATE
    schedule, lengths = e2.exact_token_schedule(
        examples,
        tokenizer,
        nominal_budget,
        MODEL_SEED,
    )
    packed = e2.pack_dynamic_batches(schedule, lengths, TARGET_TOKENS_PER_UPDATE)
    if len(packed) < MAX_UPDATES:
        raise IntegrityFailure(
            f"fixed schedule produced {len(packed)} batches, need {MAX_UPDATES}"
        )
    batches = packed[:MAX_UPDATES]
    total_tokens = sum(lengths[index] for batch in batches for index in batch)
    if any(
        sum(lengths[index] for index in batch) > TARGET_TOKENS_PER_UPDATE
        for batch in batches
    ):
        raise IntegrityFailure("a training batch exceeds the 1,024-token target")
    return batches, lengths, total_tokens


@torch.no_grad()
def _evaluate(
    model: e2.CommonRecurrentModel,
    examples: Sequence[e2.DeductionExample],
    tokenizer: e2.LocalTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    ce_sum = 0.0
    predictions: list[int] = []
    for batch_indices in e2.iter_index_batches(list(range(len(examples))), 16):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = e2.collate_examples(batch, tokenizer, device)
        with e2.training_autocast(device):
            logits, _, _ = model(tokens, mask, horizon=4)
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
        batch_predictions = logits.argmax(dim=-1)
        correct += int((batch_predictions == labels).sum().item())
        ce_sum += float(loss_sum.float().item())
        predictions.extend(int(value) for value in batch_predictions.cpu().tolist())
    return {
        "horizon": 4,
        "training_accuracy": _rate(correct, len(examples)),
        "cross_entropy": {
            "sum": ce_sum,
            "mean": ce_sum / len(examples),
            "example_denominator": len(examples),
        },
        "prediction_counts": {
            str(key): value for key, value in sorted(Counter(predictions).items())
        },
        "predictions_sha256": _sha256_json(predictions),
        "predictions": predictions,
    }


def _gradient_summary(values: Sequence[float]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=tensor.dtype
        ),
    )
    return {
        "update_denominator": len(values),
        "mean": float(tensor.mean().item()),
        "standard_deviation_population": float(tensor.std(unbiased=False).item()),
        "quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "p25", "median", "p75", "p95", "p99", "maximum"),
                quantiles.tolist(),
                strict=True,
            )
        },
    }


def _parameter_groups(
    model: e2.CommonRecurrentModel,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups = {name: [] for name in GRADIENT_GROUP_PREFIXES}
    assigned: set[str] = set()
    for parameter_name, parameter in model.named_parameters():
        matches = [
            group_name
            for group_name, prefixes in GRADIENT_GROUP_PREFIXES.items()
            if any(
                parameter_name == prefix or parameter_name.startswith(f"{prefix}.")
                for prefix in prefixes
            )
        ]
        if len(matches) != 1:
            raise IntegrityFailure(
                f"parameter {parameter_name!r} matched gradient groups {matches}"
            )
        groups[matches[0]].append((parameter_name, parameter))
        assigned.add(parameter_name)
    if assigned != {name for name, _ in model.named_parameters()}:
        raise IntegrityFailure("gradient instrumentation did not cover every parameter")
    if any(not parameters for parameters in groups.values()):
        raise IntegrityFailure("gradient instrumentation contains an empty group")
    return groups


def _group_gradient_norms(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for group_name, parameters in groups.items():
        squared_norm = 0.0
        tensors_with_gradient = 0
        nonzero_gradient_tensors = 0
        for _, parameter in parameters:
            if parameter.grad is None:
                continue
            tensors_with_gradient += 1
            gradient = parameter.grad.detach().float()
            tensor_squared_norm = float(torch.sum(gradient * gradient).item())
            squared_norm += tensor_squared_norm
            if tensor_squared_norm > 0.0:
                nonzero_gradient_tensors += 1
        summaries[group_name] = {
            "l2_norm": math.sqrt(squared_norm),
            "parameter_tensor_count": len(parameters),
            "tensors_with_gradient": tensors_with_gradient,
            "nonzero_gradient_tensors": nonzero_gradient_tensors,
        }
    return summaries


def _snapshot_parameters(
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in parameters
    }


def _parameter_delta_norm(
    parameters: Sequence[tuple[str, nn.Parameter]],
    initial: Mapping[str, torch.Tensor],
) -> float:
    squared_norm = 0.0
    for name, parameter in parameters:
        delta = parameter.detach().cpu().float() - initial[name]
        squared_norm += float(torch.sum(delta * delta).item())
    return math.sqrt(squared_norm)


def _prediction_flips(current: Sequence[int], reference: Sequence[int]) -> int:
    if len(current) != len(reference):
        raise IntegrityFailure("prediction vectors have unequal lengths")
    return sum(left != right for left, right in zip(current, reference, strict=True))


def _base_result(
    *,
    started_utc: str,
    hashes: Mapping[str, Any],
    parameter_count: int,
    model_initialization_hash: str,
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "adjudicates_semantic_ratchet": False,
        "started_utc": started_utc,
        "completed_utc": None,
        "preregistration": "experiments/06_uesd/E2_DIAG_PREREGISTRATION.md",
        "review_attestation": REVIEW_ATTESTATION,
        "scope": "STAGE_0_ONLY",
        "hashes": {
            **dict(hashes),
            "model_initialization_sha256": model_initialization_hash,
            "checkpoints": {},
        },
        "model": {
            "seed": MODEL_SEED,
            "trainable_parameter_count": parameter_count,
            "registered_parameter_range": [28_000_000, 30_000_000],
        },
        "integrity": dict(integrity),
        "stages": {
            "stage_0": None,
            "stage_1": _not_run("STAGE_0_ONLY_EXECUTION_SCOPE"),
            "stage_2": {
                **_not_run("STAGE_0_ONLY_EXECUTION_SCOPE"),
                "optimizer_matrix": {
                    "A": {"gradient_clip_norm": 1.0, "schedule": "cosine"},
                    "B": {"gradient_clip_norm": 10.0, "schedule": "cosine"},
                    "C": {"gradient_clip_norm": 1.0, "schedule": "plateau"},
                    "D": {"gradient_clip_norm": 10.0, "schedule": "plateau"},
                },
            },
            "stage_3": _not_run("STAGE_0_ONLY_EXECUTION_SCOPE"),
        },
        "final_route_token": None,
        "suite_status": None,
        "compute": None,
        "protocol_deviations": [],
        "operational_stops": [],
        "access_confirmation": {
            "original_e2_official_test_accessed": False,
            "gsm8k_official_test_accessed": False,
            "svamp_official_test_accessed": False,
            "selector_path_accessed": False,
            "critic_trained": False,
            "latch_trained_or_evaluated": False,
        },
    }


def _validate_prelaunch(
    config: Mapping[str, Any],
    controller_train: Sequence[e2.DeductionExample],
    selected: Sequence[e2.DeductionExample],
    generator_audit: Mapping[str, Any],
    tokenizer_audit: Mapping[str, Any],
    parameter_count: int,
    repeated_eval_equal: bool,
) -> dict[str, Any]:
    controller_labels = Counter(example.answer_position for example in controller_train)
    selected_labels = Counter(example.answer_position for example in selected)
    checks = {
        "controller_train_examples_exact": len(controller_train)
        == CONTROLLER_TRAIN_EXAMPLES,
        "controller_train_label_counts_exact": controller_labels
        == Counter({0: 2048, 1: 2048, 2: 2048, 3: 2048}),
        "memorization_examples_exact": len(selected) == MEMORIZATION_EXAMPLES,
        "memorization_label_counts_exact": selected_labels
        == Counter({0: 32, 1: 32, 2: 32, 3: 32}),
        "memorization_example_ids_unique": len(
            {example.example_id for example in selected}
        )
        == MEMORIZATION_EXAMPLES,
        "parameter_count_in_range": 28_000_000 <= parameter_count <= 30_000_000,
        "tokenizer_round_trip_and_unknown_checks": bool(tokenizer_audit["pass"]),
        "generator_symbolic_verification": generator_audit[
            "symbolically_verified_examples"
        ]
        == CONTROLLER_TRAIN_EXAMPLES,
        "generator_label_randomization": generator_audit[
            "answer_position_balanced_within_strata"
        ],
        "deterministic_repeated_initial_evaluation": repeated_eval_equal,
        "training_horizons_exact": tuple(config["training"]["train_horizons"])
        == (1, 2, 4),
        "base_optimizer_exact": config["training"]["optimizer"] == "AdamW"
        and config["training"]["betas"] == [0.9, 0.95]
        and math.isclose(config["training"]["weight_decay"], 0.1)
        and math.isclose(config["training"]["epsilon"], 1e-8)
        and math.isclose(config["training"]["gradient_clip_norm"], 1.0),
        "official_test_loader_absent": True,
        "selector_path_absent": True,
        "wall_time_stop_internal": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise IntegrityFailure(f"prelaunch integrity checks failed: {failed}")
    return {
        "checks": checks,
        "controller_train_label_counts": {
            str(key): value for key, value in sorted(controller_labels.items())
        },
        "memorization_label_counts": {
            str(key): value for key, value in sorted(selected_labels.items())
        },
        "generator_audit": dict(generator_audit),
        "tokenizer_audit": dict(tokenizer_audit),
        "pass": True,
    }


def _run_stage_0(
    review_attestation: str, *, operational_repair: bool = False
) -> tuple[dict[str, Any], bool, Path]:
    expected_attestation = (
        REPAIR_REVIEW_ATTESTATION if operational_repair else REVIEW_ATTESTATION
    )
    result_path = REPAIR_RESULT_PATH if operational_repair else RESULT_PATH
    launch_marker = REPAIR_LAUNCH_MARKER if operational_repair else LAUNCH_MARKER
    recovery_record = (
        REPAIR_RECOVERY_RECORD if operational_repair else RECOVERY_RECORD
    )
    recovery_checkpoint = (
        REPAIR_RECOVERY_CHECKPOINT if operational_repair else RECOVERY_CHECKPOINT
    )
    if review_attestation != expected_attestation:
        raise RuntimeError(
            "launch blocked: exact Stage-0 prelaunch review attestation required"
        )
    if result_path.exists():
        raise RuntimeError(f"immutable result already exists: {result_path}")
    if launch_marker.exists() or recovery_record.exists():
        raise RuntimeError(
            "Stage 0 has prior launch provenance; repeating or resuming is forbidden"
        )
    if operational_repair and not RESULT_PATH.exists():
        raise RuntimeError("operational repair requires the immutable original VOID")

    os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
    config = e2.load_config(E2_CONFIG)
    controller_train, generator_audit = _generate_controller_train(config)
    selected = _select_memorization_set(controller_train)
    tokenizer = e2.LocalTokenizer(config)
    tokenizer_audit = _tokenizer_integrity(controller_train, tokenizer)
    batches, lengths, scheduled_tokens = _build_training_batches(selected, tokenizer)

    code_hash = _sha256_file(Path(__file__).resolve())
    config_hash = _sha256_file(E2_CONFIG)
    full_split_hash = e2.dataset_content_hash({"controller_train": controller_train})
    selected_identities = [_example_identity(example) for example in selected]
    selected_hash = _sha256_json(selected_identities)
    selected_ids_hash = _sha256_json(
        [identity["example_id"] for identity in selected_identities]
    )
    hashes = {
        "code_sha256": code_hash,
        "e2_config_sha256": config_hash,
        "registration_sha256": _sha256_file(PREREGISTRATION),
        "generator_version_sha256": _sha256_json(config["generator"]),
        "controller_train_split_sha256": full_split_hash,
        "stage_0_ordered_set_sha256": selected_hash,
        "stage_0_ordered_example_ids_sha256": selected_ids_hash,
        "tokenizer_vocabulary_sha256": tokenizer.vocabulary_hash(),
    }

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json_no_clobber(
        {
            "experiment_id": EXPERIMENT_ID,
            "stage": 0,
            "started_utc": started_utc,
            "code_sha256": code_hash,
            "stage_0_ordered_set_sha256": selected_hash,
            "review_attestation": review_attestation,
        },
        launch_marker,
    )

    cuda_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0 requires CUDA; CPU execution is forbidden")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["training"]["tf32"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(MODEL_SEED)
    random.seed(MODEL_SEED)
    torch.cuda.reset_peak_memory_stats(device)

    model = e2.CommonRecurrentModel(config, len(tokenizer.id_to_token))
    parameter_count = e2.trainable_parameter_count(model)
    initialization_hash = _model_state_hash(model)
    model.to(device)
    optimizer = e2.make_optimizer(model, config)
    parameter_groups = _parameter_groups(model)
    readout_initial = _snapshot_parameters(parameter_groups["readout_head"])

    initial_eval_a = _evaluate(model, selected[:16], tokenizer, device)
    initial_eval_b = _evaluate(model, selected[:16], tokenizer, device)
    repeated_eval_equal = initial_eval_a == initial_eval_b
    integrity = _validate_prelaunch(
        config,
        controller_train,
        selected,
        generator_audit,
        tokenizer_audit,
        parameter_count,
        repeated_eval_equal,
    )
    result = _base_result(
        started_utc=started_utc,
        hashes=hashes,
        parameter_count=parameter_count,
        model_initialization_hash=initialization_hash,
        integrity=integrity,
    )

    curve: list[dict[str, Any]] = []
    gradient_norms: list[float] = []
    clipped_updates = 0
    processed_tokens = 0
    processed_examples = 0
    first_threshold_update: int | None = None
    stop_reason: str | None = None
    initial_full_eval = _evaluate(model, selected, tokenizer, device)
    initial_predictions = initial_full_eval["predictions"]
    previous_predictions = initial_predictions
    initial_full_eval["prediction_flip_count_from_initial"] = 0
    initial_full_eval["prediction_flip_count_from_previous_checkpoint"] = 0
    initial_full_eval["readout_head_weight_delta_l2_from_initial"] = 0.0
    curve.append({"update": 0, "processed_tokens": 0, **initial_full_eval})
    gradient_flow: list[dict[str, Any]] = []

    horizon_schedule = (1, 2, 4)
    completed_updates = 0
    for update_index, batch_indices in enumerate(batches, start=1):
        if time.perf_counter() - cuda_started >= WALL_TIME_CAP_SECONDS:
            stop_reason = "STAGE_0_WALL_TIME_CAP"
            break
        model.train()
        batch = [selected[index] for index in batch_indices]
        tokens, mask, labels, batch_tokens = e2.collate_examples(
            batch, tokenizer, device
        )
        optimizer.zero_grad(set_to_none=True)
        horizon = horizon_schedule[(update_index - 1) % len(horizon_schedule)]
        with e2.training_autocast(device):
            logits, _, _ = model(tokens, mask, horizon)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        checkpoint_gradient_norms = None
        if update_index % EVAL_INTERVAL == 0 or update_index == MAX_UPDATES:
            checkpoint_gradient_norms = _group_gradient_norms(parameter_groups)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        gradient_norm_value = float(torch.as_tensor(gradient_norm).float().item())
        gradient_norms.append(gradient_norm_value)
        if gradient_norm_value > float(config["training"]["gradient_clip_norm"]):
            clipped_updates += 1
        processed_tokens += batch_tokens
        processed_examples += len(batch)
        learning_rate = e2.learning_rate_at_tokens(
            processed_tokens, scheduled_tokens, config
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        completed_updates = update_index

        if update_index % EVAL_INTERVAL == 0 or update_index == MAX_UPDATES:
            evaluation = _evaluate(model, selected, tokenizer, device)
            current_predictions = evaluation["predictions"]
            flip_from_initial = _prediction_flips(
                current_predictions, initial_predictions
            )
            flip_from_previous = _prediction_flips(
                current_predictions, previous_predictions
            )
            readout_delta = _parameter_delta_norm(
                parameter_groups["readout_head"], readout_initial
            )
            evaluation["prediction_flip_count_from_initial"] = flip_from_initial
            evaluation["prediction_flip_count_from_previous_checkpoint"] = (
                flip_from_previous
            )
            evaluation["readout_head_weight_delta_l2_from_initial"] = readout_delta
            gradient_flow.append(
                {
                    "update": update_index,
                    "training_horizon": horizon,
                    "pre_clip_gradient_norms": checkpoint_gradient_norms,
                    "readout_head_weight_delta_l2_from_initial": readout_delta,
                }
            )
            curve.append(
                {
                    "update": update_index,
                    "processed_tokens": processed_tokens,
                    "processed_examples": processed_examples,
                    "learning_rate": learning_rate,
                    "last_training_loss_mean": float(loss.detach().float().item()),
                    **evaluation,
                }
            )
            previous_predictions = current_predictions
            correct = evaluation["training_accuracy"]["numerator"]
            print(
                f"stage0 update={update_index} correct={correct}/128 "
                f"ce={evaluation['cross_entropy']['mean']:.6f}",
                flush=True,
            )
            if correct >= GATE_CORRECT:
                first_threshold_update = update_index
                break
            if time.perf_counter() - cuda_started >= WALL_TIME_CAP_SECONDS:
                stop_reason = "STAGE_0_WALL_TIME_CAP"
                break

    torch.cuda.synchronize(device)
    final_eval = curve[-1]
    if stop_reason is not None:
        stage_status = "VOID"
        stage_reason = stop_reason
        final_route_token = "VOID_NO_ROUTE"
        suite_status = "TERMINAL_VOID"
    elif first_threshold_update is not None:
        stage_status = "PASS"
        stage_reason = "AT_LEAST_122_OF_128_BY_UPDATE_3000"
        final_route_token = None
        suite_status = "INCOMPLETE_STAGE_1_NEXT"
    elif completed_updates == MAX_UPDATES:
        stage_status = "STOP"
        stage_reason = "FEWER_THAN_122_OF_128_AFTER_3000_UPDATES"
        final_route_token = "KILL_FROM_SCRATCH_LINE"
        suite_status = "TERMINAL_STAGE_0_STOP"
    else:
        raise AssertionError("Stage 0 ended without a registered state")

    cuda_wall_time = time.perf_counter() - cuda_started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    initial_ce = float(curve[0]["cross_entropy"]["mean"])
    minimum_ce_row = min(curve, key=lambda row: row["cross_entropy"]["mean"])
    loss_decrease = {
        "initial_cross_entropy_mean": initial_ce,
        "minimum_cross_entropy_mean": float(
            minimum_ce_row["cross_entropy"]["mean"]
        ),
        "minimum_at_update": int(minimum_ce_row["update"]),
        "absolute_decrease_from_initial": initial_ce
        - float(minimum_ce_row["cross_entropy"]["mean"]),
        "any_decrease_from_initial": float(
            minimum_ce_row["cross_entropy"]["mean"]
        )
        < initial_ce,
        "final_cross_entropy_mean": float(final_eval["cross_entropy"]["mean"]),
    }
    stage_record = {
        "status": stage_status,
        "reason": stage_reason,
        "gate": {
            "required_correct_numerator": GATE_CORRECT,
            "training_example_denominator": MEMORIZATION_EXAMPLES,
            "maximum_update_denominator": MAX_UPDATES,
        },
        "dataset": {
            "source_split": "controller_train",
            "source_examples": CONTROLLER_TRAIN_EXAMPLES,
            "selection_rule": (
                "group_by_answer_position_then_sort_by_skeleton_hash_"
                "with_example_id_tie_break_then_first_32"
            ),
            "ordered_examples": selected_identities,
            "answer_position_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(example.answer_position for example in selected).items()
                )
            },
        },
        "training": {
            "seed": MODEL_SEED,
            "precision": "bfloat16",
            "tf32": True,
            "horizon_schedule": [1, 2, 4],
            "optimizer": "AdamW",
            "gradient_clip_norm": 1.0,
            "learning_rate_schedule": "linear_warmup_cosine_by_processed_tokens",
            "scheduled_token_budget_for_3000_updates": scheduled_tokens,
            "completed_updates": completed_updates,
            "processed_tokens": processed_tokens,
            "processed_examples": processed_examples,
            "first_evaluated_threshold_update": first_threshold_update,
            "pre_clip_gradient_norm": _gradient_summary(gradient_norms),
            "clipped_updates": _rate(clipped_updates, completed_updates),
        },
        "instrumentation": {
            "informational_only": True,
            "gradient_group_definitions": {
                group_name: list(prefixes)
                for group_name, prefixes in GRADIENT_GROUP_PREFIXES.items()
            },
            "gradient_flow_every_100_updates": gradient_flow,
            "loss_decrease": loss_decrease,
        },
        "training_curve": curve,
        "final_evaluation": {
            key: final_eval[key]
            for key in (
                "update",
                "processed_tokens",
                "training_accuracy",
                "cross_entropy",
                "prediction_counts",
                "predictions_sha256",
                "predictions",
                "prediction_flip_count_from_initial",
                "prediction_flip_count_from_previous_checkpoint",
                "readout_head_weight_delta_l2_from_initial",
            )
        },
        "compute": {
            "gpu_wall_time_seconds": cuda_wall_time,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
        },
    }
    result["stages"]["stage_0"] = stage_record
    result["final_route_token"] = final_route_token
    result["suite_status"] = suite_status
    result["compute"] = {
        "runs": [
            {
                "stage": 0,
                "gpu_wall_time_seconds": cuda_wall_time,
                "peak_vram_allocated_bytes": peak_allocated,
                "peak_vram_reserved_bytes": peak_reserved,
            }
        ],
        "cumulative_gpu_wall_time_seconds": cuda_wall_time,
        "cumulative_gpu_wall_time_cap_seconds": 5400,
    }
    if stop_reason is not None:
        result["operational_stops"].append(stop_reason)
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    checkpoint_payload = e2.checkpoint_payload(
        model,
        optimizer,
        processed_tokens=processed_tokens,
        config_hash=config_hash,
        code_hash=code_hash,
        seed=MODEL_SEED,
        history=curve,
    )
    if recovery_checkpoint.exists():
        raise RuntimeError(f"recovery checkpoint already exists: {recovery_checkpoint}")
    e2.atomic_torch_save(checkpoint_payload, recovery_checkpoint)
    checkpoint_hash = _sha256_file(recovery_checkpoint)
    result["hashes"]["checkpoints"] = {
        "stage_0_final_sha256": checkpoint_hash,
        "stage_0_final_processed_tokens": processed_tokens,
        "stage_0_final_completed_updates": completed_updates,
    }
    if operational_repair:
        result["operational_repair_of"] = (
            "experiments/06_uesd/results/exp_e2_diag.json"
        )
        result["protocol_deviations"].extend(
            [
                {
                    "stage": 0,
                    "type": "OWNER_AUTHORIZED_OPERATIONAL_REPAIR_RERUN",
                    "scientific_configuration_changed": False,
                },
                {
                    "stage": 0,
                    "type": "INFORMATIONAL_GRADIENT_FLOW_INSTRUMENTATION",
                    "scientific_branching_metric": False,
                },
            ]
        )
    _atomic_json_no_clobber(result, recovery_record)

    del model, optimizer
    torch.cuda.empty_cache()
    terminal = stage_status in {"STOP", "VOID"}
    return result, terminal, result_path


def _land_operational_void() -> dict[str, Any]:
    """Land the terminal VOID after the post-endpoint serialization crash."""

    if RESULT_PATH.exists():
        raise RuntimeError(f"immutable result already exists: {RESULT_PATH}")
    if not LAUNCH_MARKER.exists() or not STDOUT_RECOVERY.exists():
        raise RuntimeError("operational VOID landing requires launch and stdout records")
    if RECOVERY_RECORD.exists() or RECOVERY_CHECKPOINT.exists():
        raise RuntimeError("unexpected completed recovery provenance exists")

    launch = json.loads(LAUNCH_MARKER.read_text(encoding="utf-8"))
    observations = json.loads(STDOUT_RECOVERY.read_text(encoding="utf-8"))
    curve = observations["training_curve_stdout_observations"]
    if [row["update"] for row in curve] != list(range(100, 3001, 100)):
        raise IntegrityFailure("stdout recovery does not cover every 100-update check")
    if any(row["correct_numerator"] != 32 for row in curve):
        raise IntegrityFailure("stdout recovery has an unexpected correct count")
    if any(row["training_example_denominator"] != 128 for row in curve):
        raise IntegrityFailure("stdout recovery has an unexpected denominator")
    if observations.get("process_exit_code") != 1:
        raise IntegrityFailure("stdout recovery must bind the observed nonzero exit")

    config = e2.load_config(E2_CONFIG)
    controller_train, generator_audit = _generate_controller_train(config)
    selected = _select_memorization_set(controller_train)
    tokenizer = e2.LocalTokenizer(config)
    tokenizer_audit = _tokenizer_integrity(controller_train, tokenizer)
    batches, lengths, scheduled_tokens = _build_training_batches(selected, tokenizer)
    selected_identities = [_example_identity(example) for example in selected]
    selected_hash = _sha256_json(selected_identities)
    if selected_hash != launch["stage_0_ordered_set_sha256"]:
        raise IntegrityFailure("reconstructed Stage-0 set does not match launch marker")

    torch.manual_seed(MODEL_SEED)
    random.seed(MODEL_SEED)
    model = e2.CommonRecurrentModel(config, len(tokenizer.id_to_token))
    parameter_count = e2.trainable_parameter_count(model)
    initialization_hash = _model_state_hash(model)
    processed_examples = sum(len(batch) for batch in batches)
    full_split_hash = e2.dataset_content_hash({"controller_train": controller_train})
    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "adjudicates_semantic_ratchet": False,
        "started_utc": launch["started_utc"],
        "completed_utc": completed_utc,
        "completed_utc_semantics": (
            "operational_void_landing_time; process exit timestamp unavailable"
        ),
        "preregistration": "experiments/06_uesd/E2_DIAG_PREREGISTRATION.md",
        "review_attestation": launch["review_attestation"],
        "scope": "STAGE_0_ONLY",
        "hashes": {
            "launch_code_sha256": launch["code_sha256"],
            "operational_landing_code_sha256": _sha256_file(Path(__file__).resolve()),
            "e2_config_sha256": _sha256_file(E2_CONFIG),
            "registration_sha256": _sha256_file(PREREGISTRATION),
            "generator_version_sha256": _sha256_json(config["generator"]),
            "controller_train_split_sha256": full_split_hash,
            "stage_0_ordered_set_sha256": selected_hash,
            "stage_0_ordered_example_ids_sha256": _sha256_json(
                [identity["example_id"] for identity in selected_identities]
            ),
            "tokenizer_vocabulary_sha256": tokenizer.vocabulary_hash(),
            "reconstructed_model_initialization_sha256": initialization_hash,
            "stdout_recovery_sha256": _sha256_file(STDOUT_RECOVERY),
            "checkpoints": {
                "status": "unavailable",
                "reason": "PROCESS_EXITED_BEFORE_RECOVERY_CHECKPOINT_PUBLICATION",
            },
        },
        "model": {
            "seed": MODEL_SEED,
            "trainable_parameter_count": parameter_count,
            "registered_parameter_range": [28_000_000, 30_000_000],
        },
        "integrity": {
            "prelaunch_review_passed": True,
            "controller_train_examples": len(controller_train),
            "controller_train_label_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        example.answer_position for example in controller_train
                    ).items()
                )
            },
            "memorization_examples": len(selected),
            "memorization_label_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(example.answer_position for example in selected).items()
                )
            },
            "generator_audit": generator_audit,
            "tokenizer_audit": tokenizer_audit,
            "post_endpoint_result_publication": {
                "pass": False,
                "reason": "GRADIENT_QUANTILE_Q_DTYPE_MISMATCH",
            },
            "pass": False,
        },
        "stages": {
            "stage_0": {
                "status": "VOID",
                "reason": "POST_ENDPOINT_RESULT_SERIALIZATION_FAILURE",
                "gate": {
                    "required_correct_numerator": GATE_CORRECT,
                    "training_example_denominator": MEMORIZATION_EXAMPLES,
                    "maximum_update_denominator": MAX_UPDATES,
                },
                "dataset": {
                    "source_split": "controller_train",
                    "source_examples": CONTROLLER_TRAIN_EXAMPLES,
                    "selection_rule": (
                        "group_by_answer_position_then_sort_by_skeleton_hash_"
                        "with_example_id_tie_break_then_first_32"
                    ),
                    "ordered_examples": selected_identities,
                },
                "training": {
                    "seed": MODEL_SEED,
                    "precision": "bfloat16",
                    "tf32": True,
                    "horizon_schedule": [1, 2, 4],
                    "optimizer": "AdamW",
                    "gradient_clip_norm": 1.0,
                    "learning_rate_schedule": (
                        "linear_warmup_cosine_by_processed_tokens"
                    ),
                    "scheduled_token_budget_for_3000_updates": scheduled_tokens,
                    "completed_updates": MAX_UPDATES,
                    "processed_tokens": scheduled_tokens,
                    "processed_examples": processed_examples,
                    "pre_clip_gradient_norm": {
                        "status": "unavailable",
                        "reason": "PROCESS_EXITED_BEFORE_RESULT_PUBLICATION",
                    },
                    "clipped_updates": {
                        "numerator": None,
                        "denominator": MAX_UPDATES,
                        "rate": None,
                        "status": "unavailable",
                    },
                },
                "training_curve": {
                    "source": observations["source"],
                    "cross_entropy_precision": observations[
                        "cross_entropy_precision"
                    ],
                    "observations": curve,
                },
                "registered_endpoint_observation": {
                    "update": 3000,
                    "training_accuracy": _rate(32, 128),
                    "cross_entropy_mean_stdout_rounded": curve[-1][
                        "cross_entropy_mean_stdout_rounded"
                    ],
                    "scientific_gate_adjudicated": False,
                    "reason": "OPERATIONAL_VOID_PRECEDENCE",
                },
                "compute": {
                    "gpu_wall_time_seconds": None,
                    "peak_vram_allocated_bytes": None,
                    "peak_vram_reserved_bytes": None,
                    "status": "unavailable",
                    "reason": "PROCESS_EXITED_BEFORE_RESULT_PUBLICATION",
                },
            },
            "stage_1": _not_run("STAGE_0_OPERATIONAL_VOID"),
            "stage_2": {
                **_not_run("STAGE_0_OPERATIONAL_VOID"),
                "optimizer_matrix": {
                    "A": {"gradient_clip_norm": 1.0, "schedule": "cosine"},
                    "B": {"gradient_clip_norm": 10.0, "schedule": "cosine"},
                    "C": {"gradient_clip_norm": 1.0, "schedule": "plateau"},
                    "D": {"gradient_clip_norm": 10.0, "schedule": "plateau"},
                },
            },
            "stage_3": _not_run("STAGE_0_OPERATIONAL_VOID"),
        },
        "suite_status": "TERMINAL_OPERATIONAL_VOID",
        "final_route_token": "VOID_NO_ROUTE",
        "compute": {
            "runs": [
                {
                    "stage": 0,
                    "gpu_wall_time_seconds": None,
                    "peak_vram_allocated_bytes": None,
                    "peak_vram_reserved_bytes": None,
                    "status": "unavailable_after_process_exit",
                }
            ],
            "cumulative_gpu_wall_time_seconds": None,
            "cumulative_gpu_wall_time_cap_seconds": 5400,
        },
        "protocol_deviations": [
            {
                "stage": 0,
                "type": "POST_ENDPOINT_RESULT_SERIALIZATION_FAILURE",
                "detail": (
                    "torch.quantile rejected a float32 q tensor for float64 "
                    "gradient-norm input after update 3000"
                ),
                "scientific_miss_counted": False,
                "repeat_permitted": False,
            }
        ],
        "operational_stops": ["POST_ENDPOINT_RESULT_SERIALIZATION_FAILURE"],
        "access_confirmation": {
            "original_e2_official_test_accessed": False,
            "gsm8k_official_test_accessed": False,
            "svamp_official_test_accessed": False,
            "selector_path_accessed": False,
            "critic_trained": False,
            "latch_trained_or_evaluated": False,
        },
    }
    _atomic_json_no_clobber(result, RESULT_PATH)
    return result


def _self_test() -> int:
    synthetic = [
        type(
            "SyntheticExample",
            (),
            {
                "example_id": f"example-{position}-{index}",
                "skeleton_hash": f"{index:064x}",
                "answer_position": position,
            },
        )()
        for position in range(4)
        for index in range(40)
    ]
    selected = _select_memorization_set(synthetic)
    assert len(selected) == 128
    assert Counter(item.answer_position for item in selected) == Counter(
        {0: 32, 1: 32, 2: 32, 3: 32}
    )
    assert _rate(122, 128)["rate"] == 0.953125
    assert _not_run("reason") == {"status": "NOT_RUN", "reason": "reason"}
    summary = _gradient_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["quantiles"]["median"] == 2.5
    assert _prediction_flips([0, 1, 2], [0, 2, 2]) == 1
    print(json.dumps({"passed": 6, "total": 6, "failed": []}, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-0", action="store_true")
    parser.add_argument("--stage-0-operational-repair", action="store_true")
    parser.add_argument("--land-operational-void", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--review-attestation", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        if (
            args.stage_0
            or args.stage_0_operational_repair
            or args.land_operational_void
            or args.review_attestation
        ):
            raise ValueError("self-test accepts no launch options")
        return _self_test()
    if args.land_operational_void:
        if args.stage_0 or args.stage_0_operational_repair or args.review_attestation:
            raise ValueError("operational VOID landing accepts no launch options")
        result = _land_operational_void()
        print(f"published_operational_void={RESULT_PATH}")
        print(f"final_route_token={result['final_route_token']}")
        return 0
    if args.stage_0 and args.stage_0_operational_repair:
        raise ValueError("select only one Stage-0 launch mode")
    if not args.stage_0 and not args.stage_0_operational_repair:
        raise RuntimeError("only the explicit --stage-0 diagnostic is implemented")
    result, terminal, result_path = _run_stage_0(
        args.review_attestation,
        operational_repair=args.stage_0_operational_repair,
    )
    if terminal or args.stage_0_operational_repair:
        _atomic_json_no_clobber(result, result_path)
        print(f"published_stage0_result={result_path}")
    else:
        print("stage0_passed=true; immutable_suite_result_published=false")
        print(f"untracked_recovery_record={RECOVERY_RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
