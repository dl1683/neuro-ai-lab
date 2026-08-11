"""E3 interface-supervision diagnostic provenance runner.

The six scientific cells form one ordered suite. The run path remains blocked
until this implementation is hash-bound, separately reviewed, and projected
inside the registered cap. Fast self-tests and binding preparation are CPU-only
and compute no scientific outcome. No line-07 file or cohort is opened here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import exp_e2_latch_mechanics as e2
import exp_e3_pretrained_latch_mechanics as e3


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "exp_e3_interface_supervision_diagnostic_config.json"
REGISTRATION_PATH = HERE / "E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTRATION.md"
E2_RUNNER_PATH = HERE / "exp_e2_latch_mechanics.py"
E3_RUNNER_PATH = HERE / "exp_e3_pretrained_latch_mechanics.py"
E3_PREFLIGHT_RESULT = HERE / "results" / "exp_e3_preflight.json"
RESULT_PATH = HERE / "results" / "exp_e3_interface_supervision_diagnostic.json"
PROTOCOL_HASH_MARKER = "\n## Implementation binding attestation"
PROOF_TRACE_CONTRACT = (
    "unique-shortest-proof-replay-v1:premises-available-before-use;"
    "typed-rule-application;conclusion-follows;terminal-entails-gold;"
    "trace-length-equals-minimum-cost;shortest-proof-count-equals-one;"
    "four-type-and-arity-matched-candidates-exactly-one-replay-valid"
)
CELL_NAMES = (
    "I0_ONE_HOP_LINEAR_FLOOR",
    "I1_HARD_ANSWER_LINEAR",
    "I2_FACT_TRACE_LINEAR",
    "S0_ONE_HOP_CONTROLLER_FLOOR",
    "S1_HARD_ANSWER_ONLY",
    "S2_HARD_PROCESS_DENSE",
)
SPLIT_NAMES = (
    "easy_train",
    "easy_validation",
    "hard_train",
    "hard_validation",
    "fact_probe_train",
    "fact_probe_validation",
)
FORBIDDEN_ACCESS_COUNTS = {
    "line_07_calibration": 0,
    "line_07_test": 0,
    "gsm8k_official_test": 0,
    "svamp_official_test": 0,
    "e2_test": 0,
    "e3_smoke": 0,
}


class IntegrityFailure(RuntimeError):
    """A frozen integrity condition failed."""


@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    prompt: str
    label: int
    span_text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CachedRepresentation:
    record_id: str
    token_states: Tensor
    prompt_mean: Tensor
    final_token: Tensor
    span_mean: Tensor
    label: int
    base_tokens: int

    @property
    def cache_bytes(self) -> int:
        tensors = (
            self.token_states,
            self.prompt_mean,
            self.final_token,
            self.span_mean,
        )
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True)
class GroundedApplication:
    rule_index: int
    rule: e2.Rule
    premises: tuple[e2.UnaryAtom, ...]
    conclusion: e2.UnaryAtom


@dataclass(frozen=True)
class ProcessRecord:
    example: e2.DeductionExample
    trace: tuple[GroundedApplication, ...]
    candidates: tuple[tuple[GroundedApplication, ...], ...]
    candidate_labels: tuple[int, ...]


@dataclass(frozen=True)
class DiagnosticData:
    easy_train: tuple[PromptRecord, ...]
    easy_validation: tuple[PromptRecord, ...]
    hard_train: tuple[e2.DeductionExample, ...]
    hard_validation: tuple[e2.DeductionExample, ...]
    fact_train: tuple[PromptRecord, ...]
    fact_validation: tuple[PromptRecord, ...]
    process_train: tuple[ProcessRecord, ...]
    process_validation: tuple[ProcessRecord, ...]
    generator_audit: Mapping[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def protocol_sha256() -> str:
    text = REGISTRATION_PATH.read_text(encoding="utf-8")
    body = text.split(PROTOCOL_HASH_MARKER, maxsplit=1)[0]
    return sha256_bytes(body.encode("utf-8"))


def _contains_pending(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_pending(item) for item in value)
    return value == "PENDING"


def load_config(*, allow_pending: bool = False) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config, allow_pending=allow_pending)
    return config


def validate_config(config: Mapping[str, Any], *, allow_pending: bool) -> None:
    if config.get("schema_version") != "1.0.0":
        raise IntegrityFailure("diagnostic config schema changed")
    if config.get("experiment_id") != "exp_e3_interface_supervision_diagnostic":
        raise IntegrityFailure("wrong diagnostic experiment id")
    if (
        config["substrate"]["alias"] != "base-B"
        or int(config["substrate"]["expected_hidden_width"]) != 1536
    ):
        raise IntegrityFailure("frozen substrate binding changed")
    expected_counts = {
        "easy_train": 2048,
        "easy_validation": 1024,
        "hard_train": 2048,
        "hard_validation": 512,
        "fact_probe_train_per_distance": 2048,
        "fact_probe_validation_per_distance": 512,
    }
    if config["generator"]["split_counts"] != expected_counts:
        raise IntegrityFailure("frozen split grid changed")
    probes = config["probes"]
    if (
        tuple(probes["views"])
        != (
            "final_nonpadding_token",
            "masked_prompt_mean",
            "query_choices_span_mean",
        )
        or probes["fact_view"] != "atomic_query_span_mean"
        or int(probes["updates"]) != 1000
        or int(probes["batch_size"]) != 64
        or float(probes["learning_rate"]) != 1e-3
        or float(probes["weight_decay"]) != 0.0
        or float(probes["gradient_clip_norm"]) != 1.0
    ):
        raise IntegrityFailure("affine-probe protocol changed")
    training = config["controller_training"]
    if (
        int(training["updates"]) != 500
        or int(training["batch_size"]) != 8
        or float(training["peak_learning_rate"]) != 3e-4
        or int(training["warmup_updates"]) != 50
        or tuple(training["betas"]) != (0.9, 0.95)
        or float(training["epsilon"]) != 1e-8
        or float(training["weight_decay"]) != 0.01
        or float(training["gradient_clip_norm"]) != 14.0
        or int(training["s0_horizon"]) != 1
        or int(training["hard_horizon"]) != 5
    ):
        raise IntegrityFailure("controller protocol changed")
    compute = config["compute"]
    if (
        tuple(compute["cell_order"]) != CELL_NAMES
        or int(compute["per_cell_wall_cap_seconds"]) != 900
        or int(compute["suite_wall_cap_seconds"]) != 5400
    ):
        raise IntegrityFailure("cell order or compute caps changed")
    if config["result_artifact"]["write_mode"] != (
        "same_directory_fsync_atomic_no_clobber"
    ):
        raise IntegrityFailure("immutable publication contract changed")
    if not allow_pending and _contains_pending(config["bindings"]):
        raise IntegrityFailure("implementation/data bindings remain pending")


def _diagnostic_namespace_owner(value: str) -> str:
    bucket = int(sha256_bytes(f"e3-interface-owner-v1:{value}".encode())[:16], 16)
    return SPLIT_NAMES[bucket % len(SPLIT_NAMES)]


class DiagnosticHardGenerator(e3.E3DeductionGenerator):
    """E2 hard construction under six diagnostic-only name owners."""

    def _build_name_pools(self) -> dict[str, dict[str, list[str]]]:
        pools: dict[str, dict[str, list[str]]] = {
            split: defaultdict(list) for split in SPLIT_NAMES
        }
        split_index = {name: index for index, name in enumerate(SPLIT_NAMES)}
        for first in self.syllables:
            for second in self.syllables:
                if first == second:
                    continue
                owner = _diagnostic_namespace_owner(f"name:{first}-{second}")
                pools[owner][first].append(f"isd{split_index[owner]}-{first}-{second}")
        for split, grouped in pools.items():
            if not any(len(names) >= 2 for names in grouped.values()):
                raise IntegrityFailure(f"no same-prefix name pair for {split}")
        return pools

    def generate_split(
        self,
        split: str,
        count: int,
        used_source_skeletons: set[str],
        *,
        seed_offset: int,
    ) -> list[e2.DeductionExample]:
        pair_specs = self._allocate_pair_specs(count)
        examples: list[e2.DeductionExample] = []
        local_sources: set[str] = set()
        nonce = seed_offset * 10_000_000
        for pair_index, spec in enumerate(pair_specs):
            for _ in range(int(self.gcfg["max_generation_attempts_per_pair"])):
                rng_seed = (
                    int(self.gcfg["seed"])
                    + seed_offset * 1_000_003
                    + nonce * 97
                    + pair_index
                )
                pair = self._make_pair(split, spec, nonce, random.Random(rng_seed))
                nonce += 1
                source = pair[0].skeleton_hash
                if source in local_sources or source in used_source_skeletons:
                    continue
                self._verify_pair(pair)
                examples.extend(_namespace_example(row, split, source) for row in pair)
                local_sources.add(source)
                break
            else:
                raise IntegrityFailure(f"could not construct diagnostic pair: {split}")
        used_source_skeletons.update(local_sources)
        examples.sort(key=lambda row: row.example_id)
        if len(examples) != count:
            raise IntegrityFailure(f"{split} constructed {len(examples)} != {count}")
        return examples


def _namespace_example(
    row: e2.DeductionExample, split: str, source_skeleton: str
) -> e2.DeductionExample:
    skeleton = sha256_json(
        {
            "namespace": "e3-interface-supervision-diagnostic-v1",
            "split": split,
            "source_skeleton": source_skeleton,
        }
    )
    group = f"{split}-{skeleton[:16]}"
    return dataclasses.replace(
        row,
        example_id=f"{group}-{row.counterfactual_member}",
        counterfactual_group=group,
        split=split,
        skeleton_hash=skeleton,
    )


def _hard_generator_config(
    config: Mapping[str, Any], *, proof_depths: Sequence[int] | None = None
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["generator"]["types"] = {
        "entity": "person",
        "unary_predicate": "person_to_boolean_property",
        "binary_relation": "person_x_person_to_boolean",
    }
    if proof_depths is not None:
        result["generator"]["proof_depths"] = list(proof_depths)
    return result


def build_easy_records(
    split: str, count: int, config: Mapping[str, Any]
) -> tuple[PromptRecord, ...]:
    properties = tuple(config["generator"]["property_words"])
    records: list[PromptRecord] = []
    for index in range(count):
        label = index % 4
        choices = list(properties[:4])
        rotation = (index // 4) % 4
        choices = choices[rotation:] + choices[:rotation]
        answer_property = choices[label]
        source_property = properties[4 + index % (len(properties) - 4)]
        entity = f"easy{0 if split == 'easy_train' else 1}-{index:05d}"
        fact = e2.UNARY_FACT_TEMPLATES[index % 6].format(
            entity=entity, property=source_property
        )
        rule = e2.UNARY_RULE_TEMPLATES[(index + 1) % 6].format(
            body=source_property, head=answer_property
        )
        query_choices = (
            f"Query: Which property must hold for {entity}?\n"
            + "Choices: "
            + " / ".join(choices)
        )
        deduction = "\n".join(("Facts:", fact, "Rules:", rule, query_choices))
        prompt = str(config["generator"]["hard_prompt_template"]).format(
            deduction=deduction
        )
        records.append(
            PromptRecord(
                f"{split}-{index:05d}",
                prompt,
                label,
                query_choices,
                {
                    "split": split,
                    "answer_position": label,
                    "proof_depth": 1,
                    "one_supporting_fact": True,
                    "one_unary_implication": True,
                },
            )
        )
    expected = Counter({label: count // 4 for label in range(4)})
    if Counter(row.label for row in records) != expected:
        raise IntegrityFailure(f"{split} is not exactly label balanced")
    if len({row.prompt for row in records}) != count:
        raise IntegrityFailure(f"{split} prompts are not unique")
    return tuple(records)


def hard_prompt(example: e2.DeductionExample, config: Mapping[str, Any]) -> str:
    return str(config["generator"]["hard_prompt_template"]).format(
        deduction=example.rendered_text
    )


def query_choices_span(example: e2.DeductionExample) -> str:
    return example.rendered_text[example.rendered_text.index("Query:") :]


def _candidate_applications_for_atom(
    example: e2.DeductionExample,
    atom: e2.UnaryAtom,
    closure: Mapping[e2.UnaryAtom, e2.ProofRecord],
) -> list[GroundedApplication]:
    target_record = closure[atom]
    candidates: list[GroundedApplication] = []
    for rule_index, rule in enumerate(example.rules):
        if rule.head != atom.property:
            continue
        premise_sets: list[tuple[e2.UnaryAtom, ...]] = []
        if rule.kind == "unary":
            premise_sets.append((e2.UnaryAtom(atom.entity, rule.body),))
        elif rule.kind == "conjunction":
            if rule.body2 is None:
                raise IntegrityFailure("conjunction lacks second premise")
            premise_sets.append(
                (
                    e2.UnaryAtom(atom.entity, rule.body),
                    e2.UnaryAtom(atom.entity, rule.body2),
                )
            )
        else:
            if rule.relation is None:
                raise IntegrityFailure("relational rule lacks relation")
            for relation in example.relation_facts:
                if relation.relation != rule.relation:
                    continue
                if rule.kind == "rel_out_self" and relation.source == atom.entity:
                    premise_sets.append((e2.UnaryAtom(relation.source, rule.body),))
                elif rule.kind == "rel_in_self" and relation.target == atom.entity:
                    premise_sets.append((e2.UnaryAtom(relation.target, rule.body),))
                elif rule.kind == "rel_out_other" and relation.target == atom.entity:
                    premise_sets.append((e2.UnaryAtom(relation.source, rule.body),))
        for premises in premise_sets:
            records = [closure.get(premise) for premise in premises]
            if any(record is None for record in records):
                continue
            cost = sum(record.cost for record in records if record is not None) + 1
            proof_count = math.prod(
                record.shortest_proof_count for record in records if record is not None
            )
            if cost == target_record.cost and proof_count > 0:
                candidates.append(GroundedApplication(rule_index, rule, premises, atom))
    return candidates


def reconstruct_unique_trace(
    example: e2.DeductionExample,
) -> tuple[GroundedApplication, ...]:
    closure = example.verifier().closure()
    target = e2.UnaryAtom(example.target_entity, example.answer_property)
    target_record = closure.get(target)
    if (
        target_record is None
        or target_record.cost != example.proof_depth
        or target_record.shortest_proof_count != 1
    ):
        raise IntegrityFailure("target does not have the registered unique proof")
    emitted: set[e2.UnaryAtom] = set()

    def visit(atom: e2.UnaryAtom) -> list[GroundedApplication]:
        record = closure[atom]
        if record.cost == 0 or atom in emitted:
            return []
        candidates = _candidate_applications_for_atom(example, atom, closure)
        if len(candidates) != 1:
            raise IntegrityFailure(
                f"proof reconstruction found {len(candidates)} applications"
            )
        application = candidates[0]
        trace: list[GroundedApplication] = []
        for premise in application.premises:
            trace.extend(visit(premise))
        trace.append(application)
        emitted.add(atom)
        return trace

    trace = tuple(visit(target))
    if len(trace) != example.proof_depth:
        raise IntegrityFailure("trace length differs from minimum proof cost")
    replay_trace(example, trace)
    return trace


def _application_is_valid(
    example: e2.DeductionExample,
    application: GroundedApplication,
    available: set[e2.UnaryAtom],
) -> bool:
    if application.rule_index < 0 or application.rule_index >= len(example.rules):
        return False
    if example.rules[application.rule_index] != application.rule:
        return False
    if not all(premise in available for premise in application.premises):
        return False
    verifier = e2.SymbolicVerifier(
        example.entities,
        tuple(available),
        example.relation_facts,
        (application.rule,),
    )
    return application.conclusion in verifier.closure()


def replay_trace(
    example: e2.DeductionExample, trace: Sequence[GroundedApplication]
) -> None:
    available = set(example.unary_facts)
    for application in trace:
        if not _application_is_valid(example, application, available):
            raise IntegrityFailure("proof step failed independent symbolic replay")
        available.add(application.conclusion)
    target = e2.UnaryAtom(example.target_entity, example.answer_property)
    closure = example.verifier().closure()
    if target not in available:
        raise IntegrityFailure("replayed trace does not entail the gold answer")
    record = closure[target]
    if record.cost != len(trace) or record.shortest_proof_count != 1:
        raise IntegrityFailure("replay disagrees with shortest-proof certificate")


def application_text(
    application: GroundedApplication, config: Mapping[str, Any]
) -> str:
    premises = " and ".join(
        f"{atom.entity} is {atom.property}" for atom in application.premises
    )
    conclusion = f"{application.conclusion.entity} is {application.conclusion.property}"
    return str(config["generator"]["candidate_application_template"]).format(
        premises=premises, conclusion=conclusion
    )


def build_process_records(
    examples: Sequence[e2.DeductionExample],
    config: Mapping[str, Any],
) -> tuple[ProcessRecord, ...]:
    counters = [0] * 5
    output: list[ProcessRecord] = []
    for example in examples:
        trace = reconstruct_unique_trace(example)
        closure = example.verifier().closure()
        available = set(example.unary_facts)
        step_candidates: list[tuple[GroundedApplication, ...]] = []
        labels: list[int] = []
        prompt_properties = sorted(
            {
                *(fact.property for fact in example.unary_facts),
                *(rule.body for rule in example.rules),
                *(rule.head for rule in example.rules),
                *(rule.body2 for rule in example.rules if rule.body2 is not None),
            }
        )
        for step_index, gold in enumerate(trace):
            gold_position = counters[step_index] % 4
            counters[step_index] += 1
            unavailable = [
                e2.UnaryAtom(entity, property_name)
                for entity in example.entities
                for property_name in prompt_properties
                if e2.UnaryAtom(entity, property_name) not in available
            ]
            distractors: list[GroundedApplication] = []
            for atom in unavailable:
                candidate = GroundedApplication(
                    gold.rule_index,
                    gold.rule,
                    (atom, *gold.premises[1:]),
                    gold.conclusion,
                )
                if not _application_is_valid(example, candidate, available):
                    distractors.append(candidate)
                if len(distractors) == 3:
                    break
            if len(distractors) != 3:
                raise IntegrityFailure("could not construct three invalid candidates")
            row = list(distractors)
            row.insert(gold_position, gold)
            valid = [
                index
                for index, candidate in enumerate(row)
                if _application_is_valid(example, candidate, available)
            ]
            if valid != [gold_position]:
                raise IntegrityFailure("candidate set lacks exactly one valid item")
            step_candidates.append(tuple(row))
            labels.append(gold_position)
            available.add(gold.conclusion)
        if any(atom not in closure for atom in available):
            raise IntegrityFailure("process replay introduced a non-derivable atom")
        output.append(
            ProcessRecord(example, trace, tuple(step_candidates), tuple(labels))
        )
    for step_index, counter in enumerate(counters):
        labels = [
            row.candidate_labels[step_index]
            for row in output
            if len(row.trace) > step_index
        ]
        if counter and len(set(Counter(labels).values())) != 1:
            raise IntegrityFailure(
                f"candidate positions unbalanced at step {step_index + 1}"
            )
    return tuple(output)


def build_fact_records(
    examples: Sequence[e2.DeductionExample],
    split: str,
    config: Mapping[str, Any],
    *,
    per_distance: int | None = None,
) -> tuple[PromptRecord, ...]:
    records: list[PromptRecord] = []
    if per_distance is None:
        per_distance = int(
            config["generator"]["split_counts"][
                "fact_probe_train_per_distance"
                if split == "fact_probe_train"
                else "fact_probe_validation_per_distance"
            ]
        )
    if len(examples) * 2 != per_distance:
        raise IntegrityFailure("fact base count changed")
    for example_index, example in enumerate(examples):
        trace = reconstruct_unique_trace(example)
        if len(trace) != 5:
            raise IntegrityFailure("fact base example is not depth five")
        positives = (trace[0].premises[0],) + tuple(
            application.conclusion for application in trace
        )
        closure = example.verifier().closure()
        properties = sorted(
            {
                *(fact.property for fact in example.unary_facts),
                *(rule.body for rule in example.rules),
                *(rule.head for rule in example.rules),
            }
        )
        for distance, positive in enumerate(positives):
            negative = next(
                (
                    e2.UnaryAtom(positive.entity, property_name)
                    for property_name in properties
                    if e2.UnaryAtom(positive.entity, property_name) not in closure
                ),
                None,
            )
            if negative is None:
                raise IntegrityFailure("fact query lacks a lexical matched negative")
            pair = ((positive, 1), (negative, 0))
            if example_index % 2:
                pair = tuple(reversed(pair))
            for pair_index, (atom, label) in enumerate(pair):
                suffix = str(config["generator"]["fact_query_suffix_template"]).format(
                    entity=atom.entity, property=atom.property
                )
                records.append(
                    PromptRecord(
                        f"{split}-d{distance}-{example.example_id}-{pair_index}",
                        hard_prompt(example, config) + suffix,
                        label,
                        suffix.strip(),
                        {
                            "split": split,
                            "distance": distance,
                            "entailed": bool(label),
                            "base_example_id": example.example_id,
                        },
                    )
                )
    distance_counts = Counter(int(row.metadata["distance"]) for row in records)
    if distance_counts != Counter({distance: per_distance for distance in range(6)}):
        raise IntegrityFailure("fact distance denominators changed")
    for distance in range(6):
        labels = Counter(
            row.label for row in records if int(row.metadata["distance"]) == distance
        )
        if labels != Counter({0: per_distance // 2, 1: per_distance // 2}):
            raise IntegrityFailure(f"fact distance {distance} is not balanced")
    return tuple(records)


def build_diagnostic_data(
    config: Mapping[str, Any], *, mini: bool = False
) -> DiagnosticData:
    counts = dict(config["generator"]["split_counts"])
    if mini:
        size = int(config["self_test"]["examples_per_split"])
        counts = {
            "easy_train": size,
            "easy_validation": size,
            "hard_train": size,
            "hard_validation": size,
            "fact_probe_train_per_distance": size,
            "fact_probe_validation_per_distance": size,
        }
    easy_train = build_easy_records("easy_train", counts["easy_train"], config)
    easy_validation = build_easy_records(
        "easy_validation", counts["easy_validation"], config
    )
    used_sources: set[str] = set()
    hard_generator = DiagnosticHardGenerator(_hard_generator_config(config))
    hard_train = tuple(
        hard_generator.generate_split(
            "hard_train", counts["hard_train"], used_sources, seed_offset=101
        )
    )
    hard_validation = tuple(
        hard_generator.generate_split(
            "hard_validation",
            counts["hard_validation"],
            used_sources,
            seed_offset=202,
        )
    )
    fact_generator = DiagnosticHardGenerator(
        _hard_generator_config(config, proof_depths=(5,))
    )
    fact_train_base = fact_generator.generate_split(
        "fact_probe_train",
        counts["fact_probe_train_per_distance"] // 2,
        used_sources,
        seed_offset=303,
    )
    fact_validation_base = fact_generator.generate_split(
        "fact_probe_validation",
        counts["fact_probe_validation_per_distance"] // 2,
        used_sources,
        seed_offset=404,
    )
    process_train = build_process_records(hard_train, config)
    process_validation = build_process_records(hard_validation, config)
    fact_train = build_fact_records(
        fact_train_base,
        "fact_probe_train",
        config,
        per_distance=counts["fact_probe_train_per_distance"],
    )
    fact_validation = build_fact_records(
        fact_validation_base,
        "fact_probe_validation",
        config,
        per_distance=counts["fact_probe_validation_per_distance"],
    )
    all_hard = {
        "hard_train": hard_train,
        "hard_validation": hard_validation,
        "fact_probe_train": fact_train_base,
        "fact_probe_validation": fact_validation_base,
    }
    audit = hard_generator.audit_dataset(all_hard)
    validation_depths = Counter(row.proof_depth for row in hard_validation)
    expected_per_depth = len(hard_validation) // 4
    if not mini and validation_depths != Counter(
        {depth: expected_per_depth for depth in (2, 3, 4, 5)}
    ):
        raise IntegrityFailure("hard validation is not exactly depth balanced")
    return DiagnosticData(
        easy_train,
        easy_validation,
        hard_train,
        hard_validation,
        fact_train,
        fact_validation,
        process_train,
        process_validation,
        {
            **audit,
            "hard_validation_depth_counts": dict(sorted(validation_depths.items())),
            "proof_trace_contract_sha256": sha256_bytes(
                PROOF_TRACE_CONTRACT.encode("ascii")
            ),
            "forbidden_access_counts": dict(FORBIDDEN_ACCESS_COUNTS),
            "all_forbidden_access_counts_zero": True,
        },
    )


def _prompt_payload(records: Sequence[PromptRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.record_id,
            "prompt": row.prompt,
            "label": row.label,
            "span": row.span_text,
            "metadata": dict(row.metadata),
        }
        for row in records
    ]


def _hard_payload(
    examples: Sequence[e2.DeductionExample],
    process: Sequence[ProcessRecord],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    process_by_id = {row.example.example_id: row for row in process}
    return [
        {
            "id": row.example_id,
            "counterfactual_group": row.counterfactual_group,
            "skeleton": row.skeleton_hash,
            "prompt": hard_prompt(row, config),
            "gold": row.answer_position,
            "proof_depth": row.proof_depth,
            "trace": [
                {
                    "rule_index": app.rule_index,
                    "rule": dataclasses.asdict(app.rule),
                    "premises": [dataclasses.asdict(atom) for atom in app.premises],
                    "conclusion": dataclasses.asdict(app.conclusion),
                }
                for app in process_by_id[row.example_id].trace
            ],
            "candidate_labels": list(process_by_id[row.example_id].candidate_labels),
            "candidates": [
                [application_text(candidate, config) for candidate in step]
                for step in process_by_id[row.example_id].candidates
            ],
        }
        for row in examples
    ]


def split_hashes(data: DiagnosticData, config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "easy_train": sha256_json(_prompt_payload(data.easy_train)),
        "easy_validation": sha256_json(_prompt_payload(data.easy_validation)),
        "hard_train": sha256_json(
            _hard_payload(data.hard_train, data.process_train, config)
        ),
        "hard_validation": sha256_json(
            _hard_payload(data.hard_validation, data.process_validation, config)
        ),
        "fact_probe_train": sha256_json(_prompt_payload(data.fact_train)),
        "fact_probe_validation": sha256_json(_prompt_payload(data.fact_validation)),
    }


def expected_bindings(
    config: Mapping[str, Any], data: DiagnosticData
) -> dict[str, Any]:
    preflight = json.loads(E3_PREFLIGHT_RESULT.read_text(encoding="utf-8"))
    return {
        "registration_protocol_sha256": protocol_sha256(),
        "e2_generator_source_sha256": sha256_file(E2_RUNNER_PATH),
        "e3_runner_source_sha256": sha256_file(E3_RUNNER_PATH),
        "template_inventory_sha256": e2.template_inventory_hash(),
        "hard_prompt_sha256": sha256_bytes(
            str(config["generator"]["hard_prompt_template"]).encode("utf-8")
        ),
        "fact_query_suffix_sha256": sha256_bytes(
            str(config["generator"]["fact_query_suffix_template"]).encode("utf-8")
        ),
        "candidate_application_template_sha256": sha256_bytes(
            str(config["generator"]["candidate_application_template"]).encode("utf-8")
        ),
        "proof_trace_contract_sha256": sha256_bytes(
            PROOF_TRACE_CONTRACT.encode("ascii")
        ),
        "substrate_local_content_sha256": preflight["hashes"][
            "substrate_local_content_sha256"
        ],
        "split_sha256": split_hashes(data, config),
    }


def validate_bindings(config: Mapping[str, Any], data: DiagnosticData) -> None:
    observed = config["bindings"]
    expected = expected_bindings(config, data)
    if observed != expected:
        differences = {
            key: {"observed": observed.get(key), "expected": expected.get(key)}
            for key in expected
            if observed.get(key) != expected.get(key)
        }
        raise IntegrityFailure(f"diagnostic bindings mismatch: {differences}")
    text = REGISTRATION_PATH.read_text(encoding="utf-8")
    tick = chr(96)
    expected_slots = {
        "E3-interface-diagnostic-runner-sha256": sha256_file(Path(__file__)),
        "E3-interface-diagnostic-config-sha256": sha256_file(CONFIG_PATH),
    }
    for key, value in expected_slots.items():
        prefix = f"- {tick}{key}{tick}: {tick}"
        matches = [
            line[len(prefix) : -1]
            for line in text.splitlines()
            if line.startswith(prefix) and line.endswith(tick)
        ]
        if matches != [value]:
            raise IntegrityFailure(f"registration binding slot mismatch: {key}")


@torch.inference_mode()
def build_representations(
    records: Sequence[PromptRecord],
    config: Mapping[str, Any],
    tokenizer: Any,
    substrate: nn.Module,
    device: torch.device,
    *,
    batch_size: int = 8,
    cap_check: Callable[[str], None] | None = None,
) -> tuple[list[CachedRepresentation], dict[str, int]]:
    cache: list[CachedRepresentation] = []
    token_count = 0
    for start in range(0, len(records), batch_size):
        if cap_check is not None:
            cap_check("representation construction")
        batch = records[start : start + batch_size]
        encoded = tokenizer(
            [row.prompt for row in batch],
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        outputs = substrate.model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if hidden.shape[-1] != int(config["substrate"]["expected_hidden_width"]):
            raise IntegrityFailure("substrate representation width changed")
        for row_index, row in enumerate(batch):
            length = int(attention[row_index].sum().item())
            states = hidden[row_index, :length].detach().cpu().contiguous()
            span_start = row.prompt.rfind(row.span_text)
            if span_start < 0:
                raise IntegrityFailure("registered prompt span text is absent")
            span_end = span_start + len(row.span_text)
            span_indices = [
                index
                for index, (token_start, token_end) in enumerate(
                    offsets[row_index, :length].tolist()
                )
                if token_end > span_start and token_start < span_end
            ]
            if not span_indices:
                raise IntegrityFailure("registered prompt span could not be localized")
            cache.append(
                CachedRepresentation(
                    row.record_id,
                    states,
                    states.float().mean(0).to(torch.bfloat16),
                    states[-1].contiguous(),
                    states[span_indices].float().mean(0).to(torch.bfloat16),
                    row.label,
                    length,
                )
            )
            token_count += length
    cache_bytes = sum(row.cache_bytes for row in cache)
    if cache_bytes > int(config["substrate"]["maximum_cache_bytes"]):
        raise IntegrityFailure("representation cache exceeds registered bound")
    return cache, {
        "examples": len(cache),
        "generated_base_tokens": token_count,
        "cache_bytes": cache_bytes,
    }


def hard_prompt_records(
    examples: Sequence[e2.DeductionExample],
    config: Mapping[str, Any],
) -> tuple[PromptRecord, ...]:
    return tuple(
        PromptRecord(
            row.example_id,
            hard_prompt(row, config),
            row.answer_position,
            query_choices_span(row),
            {
                "split": row.split,
                "proof_depth": row.proof_depth,
                "answer_position": row.answer_position,
            },
        )
        for row in examples
    )


def cache_view(
    cache: Sequence[CachedRepresentation], view: str
) -> tuple[Tensor, Tensor]:
    attribute = {
        "final_nonpadding_token": "final_token",
        "masked_prompt_mean": "prompt_mean",
        "query_choices_span_mean": "span_mean",
        "atomic_query_span_mean": "span_mean",
    }[view]
    features = torch.stack([getattr(row, attribute).float() for row in cache], dim=0)
    labels = torch.tensor([row.label for row in cache], dtype=torch.long)
    return features, labels


def fixed_batches(
    length: int, batch_size: int, updates: int, seed: int
) -> list[list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule: list[int] = []
    while len(schedule) < updates * batch_size:
        schedule.extend(torch.randperm(length, generator=generator).tolist())
    return [
        schedule[start : start + batch_size]
        for start in range(0, updates * batch_size, batch_size)
    ]


def wilson_interval(numerator: int, denominator: int) -> tuple[float, float]:
    return e3.wilson_interval(numerator, denominator)


def confusion_matrix(
    predictions: Tensor, labels: Tensor, classes: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for gold, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[int(gold)][int(predicted)] += 1
    return matrix


def metric_record(correct: int, denominator: int) -> dict[str, Any]:
    lower, upper = wilson_interval(correct, denominator)
    return {
        "correct": correct,
        "denominator": denominator,
        "percentage": 100.0 * correct / denominator,
        "wilson_95_interval": {"lower": lower, "upper": upper},
    }


def train_affine_probe(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    classes: int,
    seed: int,
    config: Mapping[str, Any],
    device: torch.device,
    cap_check: Callable[[str], None],
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], Tensor | None]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    probe = nn.Linear(train_features.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config["probes"]["learning_rate"]),
        weight_decay=0.0,
    )
    batches = fixed_batches(
        len(train_features),
        int(config["probes"]["batch_size"]),
        int(config["probes"]["updates"]),
        seed + 1,
    )
    raw_norms: list[float] = []
    clipped = 0
    probe.train()
    for indices in batches:
        cap_check("affine probe update")
        x = train_features[indices].to(device)
        y = train_labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        with e3.training_autocast(device):
            loss = F.cross_entropy(probe(x), y)
        if not bool(torch.isfinite(loss)):
            raise IntegrityFailure("nonfinite affine-probe loss")
        loss.backward()
        raw = torch.nn.utils.clip_grad_norm_(
            probe.parameters(), float(config["probes"]["gradient_clip_norm"])
        )
        raw_value = float(raw.detach().cpu())
        if not math.isfinite(raw_value):
            raise IntegrityFailure("nonfinite affine-probe gradient")
        raw_norms.append(raw_value)
        clipped += int(raw_value > float(config["probes"]["gradient_clip_norm"]))
        optimizer.step()
    probe.eval()
    predictions: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(validation_features), 256):
            cap_check("affine probe final validation")
            logits = probe(validation_features[start : start + 256].to(device))
            predictions.append(logits.argmax(-1).cpu())
    predicted = torch.cat(predictions)
    correct = int((predicted == validation_labels).sum().item())
    result = {
        **metric_record(correct, len(validation_labels)),
        "confusion_matrix_gold_rows_predicted_columns": confusion_matrix(
            predicted, validation_labels, classes
        ),
        "updates": len(batches),
        "batch_size": int(config["probes"]["batch_size"]),
        "optimizer": "AdamW",
        "learning_rate": float(config["probes"]["learning_rate"]),
        "weight_decay": 0.0,
        "gradient_norm": {
            "minimum": min(raw_norms),
            "median": statistics.median(raw_norms),
            "maximum": max(raw_norms),
        },
        "clipped_updates": {"numerator": clipped, "denominator": len(batches)},
        "final_checkpoint_only": True,
        "intermediate_validation_inspections": 0,
    }
    del probe, optimizer
    e3.release_cuda()
    return result, predicted if return_predictions else None


def controller_records(
    cache: Sequence[CachedRepresentation],
) -> list[e3.FrozenExampleRepresentation]:
    return [
        e3.FrozenExampleRepresentation(
            row.record_id,
            row.token_states,
            row.prompt_mean,
            row.label,
            row.base_tokens,
        )
        for row in cache
    ]


def model_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(canonical_json(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ProcessHead(nn.Module):
    def __init__(self, substrate_width: int, width: int) -> None:
        super().__init__()
        self.candidate_adapter = nn.Linear(substrate_width, width)
        self.state_adapter = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.scale = math.sqrt(width)

    def forward(self, state: Tensor, candidates: Tensor) -> Tensor:
        pooled = state.mean(dim=1)
        query = self.norm(self.state_adapter(pooled))
        keys = self.norm(self.candidate_adapter(candidates))
        return torch.einsum("bd,bcd->bc", query, keys) / self.scale


class PairedController(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.core = e3.E3RecurrentController(config)
        self.process_head = ProcessHead(
            int(config["common_model"]["substrate_width"]),
            int(config["common_model"]["width"]),
        )


def controller_optimizer(
    model: nn.Module, config: Mapping[str, Any]
) -> torch.optim.AdamW:
    cfg = config["controller_training"]
    return torch.optim.AdamW(
        e3.optimizer_groups(model, float(cfg["weight_decay"])),
        lr=float(cfg["peak_learning_rate"]),
        betas=tuple(float(value) for value in cfg["betas"]),
        eps=float(cfg["epsilon"]),
    )


def _set_warmup_lr(
    optimizer: torch.optim.Optimizer, update: int, config: Mapping[str, Any]
) -> float:
    cfg = config["controller_training"]
    lr = float(cfg["peak_learning_rate"]) * min(
        1.0, update / int(cfg["warmup_updates"])
    )
    e3.set_learning_rate(optimizer, lr)
    return lr


@torch.inference_mode()
def build_candidate_representations(
    process_records: Sequence[ProcessRecord],
    config: Mapping[str, Any],
    tokenizer: Any,
    substrate: nn.Module,
    device: torch.device,
    cap_check: Callable[[str], None],
) -> tuple[dict[str, Tensor], dict[str, int]]:
    prompts: list[PromptRecord] = []
    identities: list[tuple[str, int, int]] = []
    for process in process_records:
        for step_index, candidates in enumerate(process.candidates):
            for candidate_index, application in enumerate(candidates):
                text = application_text(application, config)
                record_id = (
                    f"{process.example.example_id}-s{step_index}-c{candidate_index}"
                )
                prompts.append(PromptRecord(record_id, text, 0, text, {}))
                identities.append(
                    (process.example.example_id, step_index, candidate_index)
                )
    cache, compute = build_representations(
        prompts,
        config,
        tokenizer,
        substrate,
        device,
        batch_size=16,
        cap_check=cap_check,
    )
    by_key = {
        f"{example_id}:{step}:{candidate}": row.prompt_mean
        for (example_id, step, candidate), row in zip(identities, cache, strict=True)
    }
    grouped: dict[str, Tensor] = {}
    for process in process_records:
        tensor = torch.zeros(
            5,
            4,
            int(config["substrate"]["expected_hidden_width"]),
            dtype=torch.bfloat16,
        )
        for step_index, candidates in enumerate(process.candidates):
            for candidate_index, _ in enumerate(candidates):
                tensor[step_index, candidate_index] = by_key[
                    f"{process.example.example_id}:{step_index}:{candidate_index}"
                ]
        grouped[process.example.example_id] = tensor
    return grouped, compute


def train_controller_cell(
    train_cache: Sequence[CachedRepresentation],
    validation_cache: Sequence[CachedRepresentation],
    config: Mapping[str, Any],
    device: torch.device,
    cap_check: Callable[[str], None],
    *,
    horizon: int,
    dense: bool,
    process_train: Mapping[str, ProcessRecord] | None = None,
    process_validation: Mapping[str, ProcessRecord] | None = None,
    candidate_train: Mapping[str, Tensor] | None = None,
    candidate_validation: Mapping[str, Tensor] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    cfg = config["controller_training"]
    seed = int(cfg["seed"])
    e3.configure_runtime(seed)
    model = PairedController(config).to(device)
    initial_core_hash = model_state_sha256(model.core)
    initial_process_hash = model_state_sha256(model.process_head)
    optimizer = controller_optimizer(model, config)
    train_rows = controller_records(train_cache)
    validation_rows = controller_records(validation_cache)
    batches = fixed_batches(
        len(train_rows), int(cfg["batch_size"]), int(cfg["updates"]), seed + 1
    )
    raw_norms: list[float] = []
    clipped = 0
    model.train()
    for update, indices in enumerate(batches, start=1):
        cap_check(f"controller update {update}")
        rows = [train_rows[index] for index in indices]
        states, padding_mask, _frozen_prompt, labels = e3.collate_representations(
            rows, device
        )
        optimizer.zero_grad(set_to_none=True)
        _set_warmup_lr(optimizer, update, config)
        with e3.training_autocast(device):
            logits, trajectory, _prompt = model.core(states, padding_mask, horizon)
            answer_loss = F.cross_entropy(logits, labels)
            loss = answer_loss
            if dense:
                if (
                    process_train is None
                    or candidate_train is None
                    or len(trajectory) != 5
                ):
                    raise IntegrityFailure("dense process inputs are incomplete")
                per_example_losses: list[Tensor] = []
                for batch_index, row in enumerate(rows):
                    process = process_train[row.example_id]
                    candidates = candidate_train[row.example_id].to(device)
                    step_losses = [
                        F.cross_entropy(
                            model.process_head(
                                trajectory[step][batch_index : batch_index + 1],
                                candidates[step].unsqueeze(0),
                            ),
                            torch.tensor(
                                [process.candidate_labels[step]],
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                        for step in range(len(process.trace))
                    ]
                    per_example_losses.append(torch.stack(step_losses).mean())
                loss = answer_loss + torch.stack(per_example_losses).mean()
        if not bool(torch.isfinite(loss)):
            raise IntegrityFailure("nonfinite controller loss")
        loss.backward()
        raw = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["gradient_clip_norm"])
        )
        raw_value = float(raw.detach().cpu())
        if not math.isfinite(raw_value):
            raise IntegrityFailure("nonfinite controller gradient")
        raw_norms.append(raw_value)
        clipped += int(raw_value > float(cfg["gradient_clip_norm"]))
        optimizer.step()
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []
    process_step_correct = Counter()
    process_step_denominator = Counter()
    exact_trace_correct = 0
    with torch.inference_mode():
        for start in range(0, len(validation_rows), 16):
            cap_check("controller final validation")
            rows = validation_rows[start : start + 16]
            states, padding_mask, _frozen_prompt, labels = e3.collate_representations(
                rows, device
            )
            with e3.training_autocast(device):
                logits, trajectory, _prompt = model.core(states, padding_mask, horizon)
            predictions.extend(int(value) for value in logits.argmax(-1).cpu().tolist())
            gold.extend(int(value) for value in labels.cpu().tolist())
            if dense:
                if process_validation is None or candidate_validation is None:
                    raise IntegrityFailure("dense validation inputs are incomplete")
                for batch_index, row in enumerate(rows):
                    process = process_validation[row.example_id]
                    candidates = candidate_validation[row.example_id].to(device)
                    all_correct = True
                    for step in range(len(process.trace)):
                        scores = model.process_head(
                            trajectory[step][batch_index : batch_index + 1],
                            candidates[step].unsqueeze(0),
                        )
                        predicted = int(scores.argmax(-1).item())
                        is_correct = predicted == process.candidate_labels[step]
                        process_step_correct[step + 1] += int(is_correct)
                        process_step_denominator[step + 1] += 1
                        all_correct = all_correct and is_correct
                    exact_trace_correct += int(all_correct)
    predictions_tensor = torch.tensor(predictions)
    gold_tensor = torch.tensor(gold)
    correct = int((predictions_tensor == gold_tensor).sum().item())
    result: dict[str, Any] = {
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "answer": {
            **metric_record(correct, len(gold)),
            "confusion_matrix_gold_rows_predicted_columns": confusion_matrix(
                predictions_tensor, gold_tensor, 4
            ),
        },
        "completed_optimizer_updates": len(batches),
        "example_exposures": len(batches) * int(cfg["batch_size"]),
        "batch_size": int(cfg["batch_size"]),
        "horizon": horizon,
        "optimizer": "AdamW",
        "peak_learning_rate": float(cfg["peak_learning_rate"]),
        "warmup_updates": int(cfg["warmup_updates"]),
        "gradient_clip_norm": float(cfg["gradient_clip_norm"]),
        "gradient_norm": {
            "minimum": min(raw_norms),
            "median": statistics.median(raw_norms),
            "maximum": max(raw_norms),
        },
        "clipped_updates": {"numerator": clipped, "denominator": len(batches)},
        "initial_core_sha256": initial_core_hash,
        "initial_answer_path_sha256": initial_core_hash,
        "initial_process_head_sha256": initial_process_hash,
        "process_head_optimizer_updates": len(batches) if dense else 0,
        "final_checkpoint_only": True,
        "intermediate_validation_inspections": 0,
    }
    if dense:
        result["process"] = {
            "steps": {
                str(step): metric_record(
                    process_step_correct[step], process_step_denominator[step]
                )
                for step in range(1, 6)
            },
            "exact_trace": metric_record(exact_trace_correct, len(gold)),
        }
    del model, optimizer
    e3.release_cuda()
    return result, predictions


def one_sided_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(n01, discordant + 1)) / (
        2**discordant
    )


def _cell_timer(
    name: str,
    config: Mapping[str, Any],
    device: torch.device,
    suite_started: float,
) -> tuple[
    float,
    Callable[[str], None],
    torch.cuda.Event | None,
    torch.cuda.Event | None,
]:
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
    else:
        cuda_start = cuda_end = None

    def check(stage: str) -> None:
        if time.perf_counter() - started > float(
            config["compute"]["per_cell_wall_cap_seconds"]
        ):
            raise TimeoutError(f"{name} exceeded its cap during {stage}")
        if time.perf_counter() - suite_started > float(
            config["compute"]["suite_wall_cap_seconds"]
        ):
            raise TimeoutError(f"suite exceeded total cap during {name}/{stage}")

    return started, check, cuda_start, cuda_end


def _finish_cell(
    started: float,
    cuda_start: torch.cuda.Event | None,
    cuda_end: torch.cuda.Event | None,
    device: torch.device,
) -> dict[str, Any]:
    if cuda_start is not None and cuda_end is not None:
        cuda_end.record()
        torch.cuda.synchronize(device)
        cuda_seconds = cuda_start.elapsed_time(cuda_end) / 1000.0
    else:
        cuda_seconds = 0.0
    return {
        "wall_time_seconds": time.perf_counter() - started,
        "cuda_time_seconds": cuda_seconds,
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0,
    }


def _atomic_json_no_clobber(payload: Mapping[str, Any], path: Path) -> None:
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
            raise RuntimeError(f"immutable result already exists: {path}") from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def route_result(cells: Mapping[str, Any]) -> dict[str, Any]:
    i0 = bool(cells[CELL_NAMES[0]]["i0_pass"])
    i1 = bool(cells[CELL_NAMES[1]]["i1_pass"])
    fact = str(cells[CELL_NAMES[2]]["fact_class"])
    s0 = bool(cells[CELL_NAMES[3]]["s0_pass"])
    s1 = bool(cells[CELL_NAMES[4]]["s1_competent"])
    s2 = cells[CELL_NAMES[5]]
    s2_competent = bool(s2["s2_competent"])
    process = str(s2["process_class"])
    dense_gain = bool(s2["dense_gain"])
    if i0 and s0 and s2_competent and process == "PROCESS_FULL" and dense_gain:
        token = "REGISTER_DENSE_SUPERVISION_E3B"
    elif (
        (not i0)
        or (i0 and not s0)
        or ((i1 or fact == "FACT_FULL") and not s1 and not s2_competent)
    ):
        token = "REGISTER_INTERFACE_REDESIGN"
    elif s1 and not dense_gain:
        token = "REGISTER_INTERFACE_REDESIGN"
    elif (
        i0
        and s0
        and not i1
        and not s1
        and not s2_competent
        and (fact == "FACT_PARTIAL" or process == "PROCESS_PARTIAL")
    ):
        token = "REGISTER_TASK_FAMILY_CHANGE"
    elif (
        i0
        and s0
        and not i1
        and not s1
        and not s2_competent
        and fact == "FACT_NONE"
        and process == "PROCESS_NONE"
    ):
        token = "KILL_SYNTHETIC_DEDUCTION_FAMILY"
    else:
        token = "VOID_NO_ROUTE / MIXED_DIAGNOSTIC_PATTERN"
    return {
        "token": token,
        "branch_table_evaluated_top_to_bottom": True,
        "successor_execution_authorized": False,
        "canonical_e3_authorized": False,
        "full_0_5b_launch_authorized": False,
    }


def run_probe_views(
    train_cache: Sequence[CachedRepresentation],
    validation_cache: Sequence[CachedRepresentation],
    config: Mapping[str, Any],
    device: torch.device,
    check: Callable[[str], None],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for view in config["probes"]["views"]:
        train_x, train_y = cache_view(train_cache, view)
        validation_x, validation_y = cache_view(validation_cache, view)
        results[view], _ = train_affine_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            4,
            int(config["probes"]["initialization_seeds"][view]),
            config,
            device,
            check,
        )
    return results


def _load_projection(config: Mapping[str, Any]) -> Mapping[str, Any]:
    path = REPO_ROOT / str(config["review"]["projection_path"])
    if not path.exists():
        raise IntegrityFailure("reviewed completion projection is missing")
    projection = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "review_attestation": config["review"]["required_attestation"],
    }
    for key, value in expected.items():
        if projection.get(key) != value:
            raise IntegrityFailure(f"projection binding mismatch: {key}")
    if float(projection["projected_suite_wall_seconds"]) > float(
        config["compute"]["suite_wall_cap_seconds"]
    ):
        raise IntegrityFailure("measured projection exceeds suite cap")
    return projection


def run_suite(
    config: Mapping[str, Any],
    review_attestation: str,
    clean_launch_attestation: str,
) -> int:
    if RESULT_PATH.exists():
        raise RuntimeError(f"immutable diagnostic result exists: {RESULT_PATH}")
    if review_attestation != config["review"]["required_attestation"]:
        raise IntegrityFailure("independent full-pipeline review attestation missing")
    if clean_launch_attestation != config["review"]["clean_launch_attestation"]:
        raise IntegrityFailure("explicit clean-launch attestation missing")
    data = build_diagnostic_data(config)
    validate_bindings(config, data)
    projection = _load_projection(config)
    if not torch.cuda.is_available():
        raise IntegrityFailure("scientific diagnostic requires CUDA")
    device = torch.device("cuda")
    coordinates = e3.resolve_and_validate_substrate(config)
    if (
        coordinates["content_digest"]
        != config["bindings"]["substrate_local_content_sha256"]
    ):
        raise IntegrityFailure("frozen substrate content digest changed")
    suite_started = time.perf_counter()
    started_utc = utc_now()
    cells: dict[str, Any] = {}
    cell_order_completed: list[str] = []
    tokenizer = substrate = None
    easy_train_cache: list[CachedRepresentation] = []
    easy_validation_cache: list[CachedRepresentation] = []
    hard_train_cache: list[CachedRepresentation] = []
    hard_validation_cache: list[CachedRepresentation] = []
    try:
        # I0 charges substrate loading and the shared easy cache.
        name = CELL_NAMES[0]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        tokenizer, substrate = e3.load_frozen_substrate(coordinates, device)
        easy_train_cache, train_compute = build_representations(
            data.easy_train, config, tokenizer, substrate, device, cap_check=check
        )
        easy_validation_cache, validation_compute = build_representations(
            data.easy_validation,
            config,
            tokenizer,
            substrate,
            device,
            cap_check=check,
        )
        views = run_probe_views(
            easy_train_cache, easy_validation_cache, config, device, check
        )
        cells[name] = {
            "diagnostic_only": True,
            "views": views,
            "i0_pass": max(row["correct"] for row in views.values())
            >= int(config["probes"]["i0_correct_floor"]),
            "correct_floor": int(config["probes"]["i0_correct_floor"]),
            "representation_compute": {
                "train": train_compute,
                "validation": validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        cell_order_completed.append(name)

        # I1 charges the shared hard-answer cache.
        name = CELL_NAMES[1]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        hard_train_cache, train_compute = build_representations(
            hard_prompt_records(data.hard_train, config),
            config,
            tokenizer,
            substrate,
            device,
            cap_check=check,
        )
        hard_validation_cache, validation_compute = build_representations(
            hard_prompt_records(data.hard_validation, config),
            config,
            tokenizer,
            substrate,
            device,
            cap_check=check,
        )
        views = run_probe_views(
            hard_train_cache, hard_validation_cache, config, device, check
        )
        cells[name] = {
            "diagnostic_only": True,
            "views": views,
            "i1_pass": max(row["correct"] for row in views.values())
            >= int(config["probes"]["i1_correct_floor"]),
            "correct_floor": int(config["probes"]["i1_correct_floor"]),
            "representation_compute": {
                "train": train_compute,
                "validation": validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        cell_order_completed.append(name)

        # I2 trains exactly one shared binary affine probe.
        name = CELL_NAMES[2]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        fact_train_cache, train_compute = build_representations(
            data.fact_train, config, tokenizer, substrate, device, cap_check=check
        )
        fact_validation_cache, validation_compute = build_representations(
            data.fact_validation,
            config,
            tokenizer,
            substrate,
            device,
            cap_check=check,
        )
        train_x, train_y = cache_view(fact_train_cache, "atomic_query_span_mean")
        validation_x, validation_y = cache_view(
            fact_validation_cache, "atomic_query_span_mean"
        )
        overall, predictions = train_affine_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            2,
            int(config["probes"]["initialization_seeds"]["fact_trace"]),
            config,
            device,
            check,
            return_predictions=True,
        )
        if predictions is None:
            raise IntegrityFailure("fact probe predictions missing")
        by_distance: dict[str, Any] = {}
        distance_pass: dict[str, bool] = {}
        floor = int(config["probes"]["i2_distance_correct_floor"])
        for distance in range(6):
            indices = [
                index
                for index, row in enumerate(data.fact_validation)
                if int(row.metadata["distance"]) == distance
            ]
            distance_predictions = predictions[indices]
            distance_labels = validation_y[indices]
            correct = int((distance_predictions == distance_labels).sum().item())
            by_distance[str(distance)] = {
                **metric_record(correct, len(indices)),
                "confusion_matrix_gold_rows_predicted_columns": confusion_matrix(
                    distance_predictions, distance_labels, 2
                ),
            }
            distance_pass[str(distance)] = correct >= floor
        if all(distance_pass.values()):
            fact_class = "FACT_FULL"
        elif distance_pass["0"] and distance_pass["1"]:
            fact_class = "FACT_PARTIAL"
        else:
            fact_class = "FACT_NONE"
        cells[name] = {
            "diagnostic_only": True,
            "shared_probe_overall": overall,
            "by_distance": by_distance,
            "distance_pass": distance_pass,
            "correct_floor_per_512": floor,
            "fact_class": fact_class,
            "representation_compute": {
                "train": train_compute,
                "validation": validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        del fact_train_cache, fact_validation_cache
        e3.release_cuda()
        cell_order_completed.append(name)

        # S0 reuses the I0 cache.
        name = CELL_NAMES[3]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        s0, _ = train_controller_cell(
            easy_train_cache,
            easy_validation_cache,
            config,
            device,
            check,
            horizon=int(config["controller_training"]["s0_horizon"]),
            dense=False,
        )
        cells[name] = {
            **s0,
            "diagnostic_only": True,
            "s0_pass": s0["answer"]["correct"]
            >= int(config["controller_training"]["s0_correct_floor"]),
            "correct_floor": int(config["controller_training"]["s0_correct_floor"]),
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        del easy_train_cache, easy_validation_cache
        e3.release_cuda()
        cell_order_completed.append(name)

        process_train = {row.example.example_id: row for row in data.process_train}
        process_validation = {
            row.example.example_id: row for row in data.process_validation
        }

        # S1 is the paired answer-only arm.
        name = CELL_NAMES[4]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        s1, s1_predictions = train_controller_cell(
            hard_train_cache,
            hard_validation_cache,
            config,
            device,
            check,
            horizon=int(config["controller_training"]["hard_horizon"]),
            dense=False,
            process_train=process_train,
            process_validation=process_validation,
        )
        cells[name] = {
            **s1,
            "diagnostic_only": True,
            "supervision": "answer_cross_entropy_only",
            "s1_competent": s1["answer"]["correct"]
            >= int(config["controller_training"]["hard_correct_floor"]),
            "correct_floor": int(config["controller_training"]["hard_correct_floor"]),
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        cell_order_completed.append(name)

        # S2 charges the candidate representation cache.
        name = CELL_NAMES[5]
        started, check, cuda_start, cuda_end = _cell_timer(
            name, config, device, suite_started
        )
        candidate_train, candidate_train_compute = build_candidate_representations(
            data.process_train,
            config,
            tokenizer,
            substrate,
            device,
            check,
        )
        candidate_validation, candidate_validation_compute = (
            build_candidate_representations(
                data.process_validation,
                config,
                tokenizer,
                substrate,
                device,
                check,
            )
        )
        s2, s2_predictions = train_controller_cell(
            hard_train_cache,
            hard_validation_cache,
            config,
            device,
            check,
            horizon=int(config["controller_training"]["hard_horizon"]),
            dense=True,
            process_train=process_train,
            process_validation=process_validation,
            candidate_train=candidate_train,
            candidate_validation=candidate_validation,
        )
        if s1["initial_core_sha256"] != s2["initial_core_sha256"]:
            raise IntegrityFailure("S1/S2 core and answer initialization differ")
        gold = [row.label for row in hard_validation_cache]
        n01 = sum(
            left != target and right == target
            for left, right, target in zip(
                s1_predictions, s2_predictions, gold, strict=True
            )
        )
        n10 = sum(
            left == target and right != target
            for left, right, target in zip(
                s1_predictions, s2_predictions, gold, strict=True
            )
        )
        p_value = one_sided_mcnemar_p(n01, n10)
        step_passes = {
            step: s2["process"]["steps"][step]["correct"]
            >= int(config["process"]["per_step_correct_floors"][step])
            for step in ("1", "2", "3", "4", "5")
        }
        exact_pass = s2["process"]["exact_trace"]["correct"] >= int(
            config["process"]["exact_trace_correct_floor"]
        )
        if all(step_passes.values()) and exact_pass:
            process_class = "PROCESS_FULL"
        elif step_passes["1"]:
            process_class = "PROCESS_PARTIAL"
        else:
            process_class = "PROCESS_NONE"
        answer_gain = s2["answer"]["correct"] - s1["answer"]["correct"]
        s2_competent = s2["answer"]["correct"] >= int(
            config["controller_training"]["hard_correct_floor"]
        )
        dense_gain = (
            s2_competent
            and process_class == "PROCESS_FULL"
            and answer_gain
            >= int(config["controller_training"]["dense_gain_correct_floor"])
            and p_value
            <= float(config["controller_training"]["dense_gain_mcnemar_p_maximum"])
        )
        cells[name] = {
            **s2,
            "diagnostic_only": True,
            "supervision": "answer_ce_plus_mean_verified_process_ce",
            "answer_loss_coefficient": 1.0,
            "process_loss_coefficient": 1.0,
            "s2_competent": s2_competent,
            "process_step_pass": step_passes,
            "exact_trace_pass": exact_pass,
            "process_class": process_class,
            "paired_gain": {
                "s2_minus_s1_correct": answer_gain,
                "denominator": 512,
                "percentage_points": 100.0 * answer_gain / 512,
                "n01_s1_wrong_s2_correct": n01,
                "n10_s1_correct_s2_wrong": n10,
                "one_sided_exact_mcnemar_binomial_p": p_value,
            },
            "dense_gain": dense_gain,
            "candidate_representation_compute": {
                "train": candidate_train_compute,
                "validation": candidate_validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device),
        }
        cell_order_completed.append(name)

        suite_wall = time.perf_counter() - suite_started
        if suite_wall > float(config["compute"]["suite_wall_cap_seconds"]):
            raise TimeoutError("suite exceeded total cap before publication")
        decision = route_result(cells)
        result = {
            "schema_version": "1.0.0",
            "experiment_id": config["experiment_id"],
            "diagnostic_only": True,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "preregistration": config["preregistration"],
            "review_attestation": review_attestation,
            "clean_launch_attestation": clean_launch_attestation,
            "hashes": {
                "runner_sha256": sha256_file(Path(__file__)),
                "config_sha256": sha256_file(CONFIG_PATH),
                **dict(config["bindings"]),
            },
            "access_attestation": {
                "counts": dict(FORBIDDEN_ACCESS_COUNTS),
                "all_zero": True,
            },
            "generator_audit": data.generator_audit,
            "parameter_counts": {
                "controller_trainable": e3.parameter_count(
                    e3.E3RecurrentController(config)
                ),
                "frozen_substrate": e3.parameter_count(substrate, trainable_only=False),
                "process_head_trainable": e3.parameter_count(
                    ProcessHead(
                        int(config["common_model"]["substrate_width"]),
                        int(config["common_model"]["width"]),
                    )
                ),
            },
            "cell_order_registered": list(CELL_NAMES),
            "cell_order_completed": cell_order_completed,
            "cells": cells,
            "decision": decision,
            "final_route_token": decision["token"],
            "compute": {
                "suite_wall_time_seconds": suite_wall,
                "suite_wall_cap_seconds": int(
                    config["compute"]["suite_wall_cap_seconds"]
                ),
                "per_cell_wall_cap_seconds": int(
                    config["compute"]["per_cell_wall_cap_seconds"]
                ),
                "projection": projection,
            },
            "state_disposal": {
                "checkpoints_retained": 0,
                "optimizers_retained": 0,
                "probes_retained": 0,
                "process_heads_retained": 0,
                "caches_retained": 0,
                "rng_states_retained": 0,
                "all_discarded": True,
            },
        }
        _atomic_json_no_clobber(result, RESULT_PATH)
        return 0
    except Exception as error:
        suite_wall = time.perf_counter() - suite_started
        void = {
            "schema_version": "1.0.0",
            "experiment_id": config["experiment_id"],
            "diagnostic_only": True,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "final_route_token": "VOID_NO_ROUTE",
            "reason": type(error).__name__,
            "error": str(error),
            "hashes": {
                "runner_sha256": sha256_file(Path(__file__)),
                "config_sha256": sha256_file(CONFIG_PATH),
                **dict(config["bindings"]),
            },
            "cell_order_registered": list(CELL_NAMES),
            "cell_order_completed": cell_order_completed,
            "partial_cells": cells,
            "access_attestation": {
                "counts": dict(FORBIDDEN_ACCESS_COUNTS),
                "all_zero": all(
                    value == 0 for value in FORBIDDEN_ACCESS_COUNTS.values()
                ),
            },
            "compute": {
                "suite_wall_time_seconds": suite_wall,
                "suite_wall_cap_seconds": int(
                    config["compute"]["suite_wall_cap_seconds"]
                ),
                "per_cell_wall_cap_seconds": int(
                    config["compute"]["per_cell_wall_cap_seconds"]
                ),
                "projection": projection,
            },
            "state_disposal": {
                "status": "best_effort_cleanup_scheduled_in_finally",
            },
            "scientific_interpretation_authorized": False,
            "successor_authorized": False,
        }
        if not RESULT_PATH.exists():
            with contextlib.suppress(Exception):
                _atomic_json_no_clobber(void, RESULT_PATH)
        raise
    finally:
        del tokenizer, substrate
        e3.release_cuda()


def _projection_probe_kernel(
    features: Tensor,
    labels: Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    updates: int,
) -> float:
    torch.manual_seed(int(config["self_test"]["seed"]))
    probe = nn.Linear(features.shape[1], 4).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config["probes"]["learning_rate"]),
        weight_decay=0.0,
    )
    batches = fixed_batches(
        len(features),
        int(config["probes"]["batch_size"]),
        updates,
        int(config["self_test"]["seed"]) + 1,
    )
    started = time.perf_counter()
    for indices in batches:
        optimizer.zero_grad(set_to_none=True)
        with e3.training_autocast(device):
            loss = F.cross_entropy(
                probe(features[indices].to(device)),
                labels[indices].to(device),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            probe.parameters(), float(config["probes"]["gradient_clip_norm"])
        )
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del probe, optimizer
    e3.release_cuda()
    return elapsed


def _projection_controller_kernel(
    cache: Sequence[CachedRepresentation],
    config: Mapping[str, Any],
    device: torch.device,
    updates: int,
    *,
    dense: bool,
    process: Mapping[str, ProcessRecord] | None = None,
    candidates: Mapping[str, Tensor] | None = None,
) -> float:
    seed = int(config["self_test"]["seed"]) + int(dense)
    e3.configure_runtime(seed)
    model = PairedController(config).to(device)
    optimizer = controller_optimizer(model, config)
    rows = controller_records(cache)
    batches = fixed_batches(
        len(rows),
        int(config["controller_training"]["batch_size"]),
        updates,
        seed + 1,
    )
    started = time.perf_counter()
    for update, indices in enumerate(batches, start=1):
        batch = [rows[index] for index in indices]
        states, padding_mask, _frozen_prompt, labels = e3.collate_representations(
            batch, device
        )
        optimizer.zero_grad(set_to_none=True)
        _set_warmup_lr(optimizer, update, config)
        with e3.training_autocast(device):
            logits, trajectory, _prompt = model.core(
                states,
                padding_mask,
                int(config["controller_training"]["hard_horizon"]),
            )
            loss = F.cross_entropy(logits, labels)
            if dense:
                if process is None or candidates is None:
                    raise IntegrityFailure("projection dense inputs missing")
                process_losses: list[Tensor] = []
                for batch_index, row in enumerate(batch):
                    proof = process[row.example_id]
                    candidate_rows = candidates[row.example_id].to(device)
                    step_losses = [
                        F.cross_entropy(
                            model.process_head(
                                trajectory[step][batch_index : batch_index + 1],
                                candidate_rows[step].unsqueeze(0),
                            ),
                            torch.tensor(
                                [proof.candidate_labels[step]],
                                device=device,
                            ),
                        )
                        for step in range(len(proof.trace))
                    ]
                    process_losses.append(torch.stack(step_losses).mean())
                loss = loss + torch.stack(process_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["controller_training"]["gradient_clip_norm"]),
        )
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del model, optimizer
    e3.release_cuda()
    return elapsed


def run_projection(config: Mapping[str, Any], review_attestation: str) -> int:
    if review_attestation != config["review"]["required_attestation"]:
        raise IntegrityFailure("projection requires the independent review attestation")
    full_data = build_diagnostic_data(config)
    validate_bindings(config, full_data)
    if not torch.cuda.is_available():
        raise IntegrityFailure("projection preflight requires CUDA")
    data = build_diagnostic_data(config, mini=True)
    device = torch.device("cuda")
    coordinates = e3.resolve_and_validate_substrate(config)
    overall_started = time.perf_counter()
    load_started = time.perf_counter()
    tokenizer, substrate = e3.load_frozen_substrate(coordinates, device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    phase_started = time.perf_counter()
    easy_cache, easy_compute = build_representations(
        (*data.easy_train, *data.easy_validation),
        config,
        tokenizer,
        substrate,
        device,
        batch_size=8,
    )
    torch.cuda.synchronize(device)
    easy_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    hard_cache, hard_compute = build_representations(
        (
            *hard_prompt_records(data.hard_train, config),
            *hard_prompt_records(data.hard_validation, config),
        ),
        config,
        tokenizer,
        substrate,
        device,
        batch_size=8,
    )
    torch.cuda.synchronize(device)
    hard_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    fact_cache, fact_compute = build_representations(
        (*data.fact_train, *data.fact_validation),
        config,
        tokenizer,
        substrate,
        device,
        batch_size=8,
    )
    torch.cuda.synchronize(device)
    fact_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    candidate_cache, candidate_compute = build_candidate_representations(
        (*data.process_train, *data.process_validation),
        config,
        tokenizer,
        substrate,
        device,
        lambda _stage: None,
    )
    torch.cuda.synchronize(device)
    candidate_seconds = time.perf_counter() - phase_started

    probe_features, probe_labels = cache_view(
        easy_cache[: len(data.easy_train)], "masked_prompt_mean"
    )
    probe_updates = 50
    probe_seconds = _projection_probe_kernel(
        probe_features,
        probe_labels,
        config,
        device,
        probe_updates,
    )
    controller_updates = 25
    answer_seconds = _projection_controller_kernel(
        hard_cache[: len(data.hard_train)],
        config,
        device,
        controller_updates,
        dense=False,
    )
    mini_process = {
        row.example.example_id: row
        for row in (*data.process_train, *data.process_validation)
    }
    dense_seconds = _projection_controller_kernel(
        hard_cache[: len(data.hard_train)],
        config,
        device,
        controller_updates,
        dense=True,
        process=mini_process,
        candidates=candidate_cache,
    )
    full_candidate_count = sum(
        len(row.trace) * 4
        for row in (*full_data.process_train, *full_data.process_validation)
    )
    mini_candidate_count = sum(
        len(row.trace) * 4 for row in (*data.process_train, *data.process_validation)
    )
    representation_projection = (
        easy_seconds * (3072 / len(easy_cache))
        + hard_seconds * (2560 / len(hard_cache))
        + fact_seconds * (15360 / len(fact_cache))
        + candidate_seconds * (full_candidate_count / mini_candidate_count)
    )
    training_projection = (
        probe_seconds * (7000 / probe_updates)
        + answer_seconds * (1000 / controller_updates)
        + dense_seconds * (500 / controller_updates)
    )
    multiplier = float(config["compute"]["projection_safety_multiplier"])
    projected = (
        load_seconds + representation_projection + training_projection
    ) * multiplier
    result = {
        "schema_version": "1.0.0",
        "diagnostic_only": True,
        "scientific_metrics_computed": False,
        "review_attestation": review_attestation,
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "measured_sample_wall_seconds": time.perf_counter() - overall_started,
        "measured_phase_seconds": {
            "substrate_load": load_seconds,
            "easy_representations": easy_seconds,
            "hard_representations": hard_seconds,
            "fact_representations": fact_seconds,
            "candidate_representations": candidate_seconds,
            "probe_50_updates": probe_seconds,
            "answer_controller_25_updates": answer_seconds,
            "dense_controller_25_updates": dense_seconds,
        },
        "projection_components_seconds_before_safety_multiplier": {
            "substrate_load": load_seconds,
            "all_representations": representation_projection,
            "all_training_updates": training_projection,
        },
        "safety_multiplier": multiplier,
        "projected_suite_wall_seconds": projected,
        "suite_wall_cap_seconds": int(config["compute"]["suite_wall_cap_seconds"]),
        "representation_compute": {
            "easy": easy_compute,
            "hard": hard_compute,
            "fact": fact_compute,
            "candidates": candidate_compute,
        },
        "scientific_predictions_or_accuracy_computed": False,
        "completed_utc": utc_now(),
    }
    path = REPO_ROOT / str(config["review"]["projection_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del tokenizer, substrate, easy_cache, hard_cache, fact_cache, candidate_cache
    e3.release_cuda()
    print(canonical_json(result))
    return 0


def run_self_test() -> int:
    config = load_config(allow_pending=True)
    torch.set_num_threads(int(config["self_test"]["torch_threads"]))
    data = build_diagnostic_data(config, mini=True)
    hashes = split_hashes(data, config)
    if tuple(hashes) != SPLIT_NAMES:
        raise AssertionError("self-test split hash surface changed")
    model = PairedController(config)
    if e3.parameter_count(model.core) > 30_000_000:
        raise AssertionError("E3 controller parameter cap exceeded")
    sample = data.process_validation[0]
    if len(sample.trace) not in (2, 3, 4, 5):
        raise AssertionError("self-test trace depth changed")
    replay_trace(sample.example, sample.trace)
    available = set(sample.example.unary_facts)
    for step, candidates in enumerate(sample.candidates):
        valid = [
            index
            for index, candidate in enumerate(candidates)
            if _application_is_valid(sample.example, candidate, available)
        ]
        if valid != [sample.candidate_labels[step]]:
            raise AssertionError("self-test candidate replay failed")
        available.add(sample.trace[step].conclusion)
    scratch = REPO_ROOT / ".e3_interface_diag_self_test_tmp"
    scratch.mkdir(exist_ok=False)
    try:
        path = scratch / "immutable.json"
        _atomic_json_no_clobber({"self_test": True}, path)
        try:
            _atomic_json_no_clobber({"self_test": False}, path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("atomic writer permitted overwrite")
    finally:
        for child in scratch.iterdir():
            child.unlink()
        scratch.rmdir()
    print(
        canonical_json(
            {
                "status": "PASS",
                "tier": "fast_cpu",
                "scientific_metrics_computed": False,
                "split_hash_count": len(hashes),
                "trace_replay": True,
                "candidate_exactly_one_valid": True,
                "atomic_no_clobber": True,
            }
        )
    )
    return 0


def prepare_bindings() -> int:
    config = load_config(allow_pending=True)
    data = build_diagnostic_data(config)
    print(json.dumps(expected_bindings(config, data), indent=2, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test-fast", action="store_true")
    actions.add_argument("--prepare-bindings", action="store_true")
    actions.add_argument("--projection-preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--review-attestation")
    parser.add_argument("--clean-launch-attestation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test_fast:
        return run_self_test()
    if args.prepare_bindings:
        return prepare_bindings()
    config = load_config()
    if args.projection_preflight:
        if args.clean_launch_attestation is not None:
            raise IntegrityFailure(
                "projection does not accept clean-launch attestation"
            )
        return run_projection(config, str(args.review_attestation))
    return run_suite(
        config,
        str(args.review_attestation),
        str(args.clean_launch_attestation),
    )


if __name__ == "__main__":
    raise SystemExit(main())
