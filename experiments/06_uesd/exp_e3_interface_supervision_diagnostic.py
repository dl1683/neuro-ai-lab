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
import itertools
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
from typing import Any, Callable, Iterator, Mapping, Sequence

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
    "unique-shortest-proof-replay-v2:premises-available-before-use;"
    "typed-rule-application;conclusion-follows;terminal-entails-gold;"
    "trace-length-equals-minimum-cost;shortest-proof-count-equals-one;"
    "four-grounded-registered-rule-applications;matched-body-predicates-types-"
    "and-arities;exactly-one-replay-valid;prehash-balance-audited"
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


@dataclass
class _FlowEdge:
    target: int
    reverse: int
    capacity: int


class _Dinic:
    """Small deterministic integer max-flow used only for pre-data balancing."""

    def __init__(self, size: int) -> None:
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = _FlowEdge(target, len(self.graph[target]), capacity)
        reverse = _FlowEdge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = [source]
            for node in queue:
                for edge in self.graph[node]:
                    if edge.capacity and levels[edge.target] < 0:
                        levels[edge.target] = levels[node] + 1
                        queue.append(edge.target)
            if levels[sink] < 0:
                return total
            cursors = [0] * len(self.graph)

            def send(node: int, available: int) -> int:
                if node == sink:
                    return available
                while cursors[node] < len(self.graph[node]):
                    edge = self.graph[node][cursors[node]]
                    if edge.capacity and levels[edge.target] == levels[node] + 1:
                        amount = send(edge.target, min(available, edge.capacity))
                        if amount:
                            edge.capacity -= amount
                            self.graph[edge.target][edge.reverse].capacity += amount
                            return amount
                    cursors[node] += 1
                return 0

            while amount := send(source, 1 << 30):
                total += amount


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
    if (
        config["generator"].get("version")
        != "typed-datalog-e3-interface-supervision-v2"
        or len(config["generator"].get("easy_entities", ())) != 32
        or len(set(config["generator"].get("easy_entities", ()))) != 32
    ):
        raise IntegrityFailure("clarified generator contract changed")
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
    if (
        config["review"].get("projection_authorization")
        != "E3_DIAGNOSTIC_BLOCKING_REVIEW_PROJECTION_REQUEST"
        or config["review"].get("projection_path")
        != "experiments/06_uesd/exp_e3_interface_supervision_diagnostic_projection.json"
    ):
        raise IntegrityFailure("reviewed projection contract changed")
    if not allow_pending and _contains_pending(config["bindings"]):
        raise IntegrityFailure("implementation/data bindings remain pending")


def _diagnostic_namespace_owner(value: str) -> str:
    bucket = int(sha256_bytes(f"e3-interface-owner-v1:{value}".encode())[:16], 16)
    return SPLIT_NAMES[bucket % len(SPLIT_NAMES)]


class DiagnosticHardGenerator(e3.E3DeductionGenerator):
    """E2 hard construction under six diagnostic-only name owners."""

    def _build_name_pools(self) -> dict[str, dict[str, list[str]]]:
        complete: dict[str, dict[str, list[str]]] = {
            split: defaultdict(list) for split in SPLIT_NAMES
        }
        split_index = {name: index for index, name in enumerate(SPLIT_NAMES)}
        for first in self.syllables:
            for second in self.syllables:
                if first == second:
                    continue
                owner = _diagnostic_namespace_owner(f"name:{first}-{second}")
                complete[owner][first].append(
                    f"isd{split_index[owner]}-{first}-{second}"
                )
        pools: dict[str, dict[str, list[str]]] = {}
        for split, grouped in complete.items():
            selected = {
                prefix: sorted(values)[:2]
                for prefix, values in sorted(grouped.items())
                if len(values) >= 2
            }
            selected = dict(list(selected.items())[:5])
            pools[split] = selected
            grouped = selected
            names = [name for values in grouped.values() for name in values]
            if len(names) < 10 or len(grouped) < 5:
                raise IntegrityFailure(f"insufficient fixed diagnostic names for {split}")
        return pools

    def _choose_names(
        self, split: str, entity_count: int, rng: random.Random
    ) -> tuple[str, ...]:
        grouped = self._name_pools[split]
        prefixes = sorted(prefix for prefix, names in grouped.items() if len(names) >= 2)
        pair_index = int(getattr(self, "_diagnostic_pair_index", 0))
        prefix = prefixes[pair_index % len(prefixes)]
        names = sorted(grouped[prefix])
        target_index = (pair_index // len(prefixes)) % len(names)
        target = names[target_index]
        lure = names[(target_index + 1) % len(names)]
        available = sorted(
            name
            for values in grouped.values()
            for name in values
            if name not in {target, lure}
        )
        return (target, lure, *rng.sample(available, entity_count - 2))

    def _add_registered_process_groundings(
        self, pair: tuple[e2.DeductionExample, e2.DeductionExample]
    ) -> tuple[e2.DeductionExample, e2.DeductionExample]:
        """Add inert relation groundings needed for matched process candidates."""

        first, second = pair
        first_relations = set(first.relation_facts)
        second_relations = set(second.relation_facts)
        changed = sorted(first_relations ^ second_relations)
        if len(changed) != 2 or changed[0].relation != changed[1].relation:
            raise IntegrityFailure("could not identify the counterfactual relation")
        causal_relation = changed[0].relation
        terminal_rules = [
            rule
            for rule in first.rules
            if rule.relation == causal_relation
            and rule.kind in {"rel_out_self", "rel_in_self"}
            and rule.head in first.choice_properties
        ]
        if len(terminal_rules) != 2 or len({rule.body for rule in terminal_rules}) != 1:
            raise IntegrityFailure("terminal relation-rule pair changed")
        terminal_body = terminal_rules[0].body
        closures = (first.verifier().closure(), second.verifier().closure())
        usable = [
            entity
            for entity in first.entities
            if entity != first.target_entity
            and all(e2.UnaryAtom(entity, terminal_body) not in closure for closure in closures)
        ]
        decoys: list[e2.RelationAtom] = []
        for source in usable:
            for target in reversed(usable):
                relation = e2.RelationAtom(source, causal_relation, target)
                if source != target and relation not in first_relations and relation not in decoys:
                    decoys.append(relation)
                if len(decoys) == 3:
                    break
            if len(decoys) == 3:
                break
        if len(decoys) != 3:
            raise IntegrityFailure("could not add three inert registered groundings")
        augmented_skeleton = sha256_json(
            {
                "base_skeleton": first.skeleton_hash,
                "process_relation_groundings": [
                    [
                        first.entities.index(atom.source),
                        atom.relation,
                        first.entities.index(atom.target),
                    ]
                    for atom in decoys
                ],
            }
        )
        group = f"{first.split}-{augmented_skeleton[:16]}"
        output: list[e2.DeductionExample] = []
        for row in pair:
            relations = (*row.relation_facts, *decoys)
            rendered = self._render(
                row.target_entity,
                row.choice_properties,
                row.unary_facts,
                relations,
                row.rules,
                row.template_family,
                random.Random(int(augmented_skeleton[:16], 16)),
            )
            output.append(
                dataclasses.replace(
                    row,
                    example_id=f"{group}-{row.counterfactual_member}",
                    counterfactual_group=group,
                    skeleton_hash=augmented_skeleton,
                    relation_facts=tuple(relations),
                    rendered_text=rendered,
                )
            )
        return output[0], output[1]

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
            self._diagnostic_pair_index = pair_index
            for _ in range(int(self.gcfg["max_generation_attempts_per_pair"])):
                rng_seed = (
                    int(self.gcfg["seed"])
                    + seed_offset * 1_000_003
                    + nonce * 97
                    + pair_index
                )
                pair = self._make_pair(split, spec, nonce, random.Random(rng_seed))
                pair = self._add_registered_process_groundings(pair)
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


def _easy_symbolic_skeletons(
    config: Mapping[str, Any], label: int
) -> Iterator[dict[str, Any]]:
    properties = tuple(str(value) for value in config["generator"]["property_words"])
    for fact_template, rule_template, source, answer in itertools.product(
        range(len(e2.UNARY_FACT_TEMPLATES)),
        range(len(e2.UNARY_RULE_TEMPLATES)),
        properties,
        properties,
    ):
        if source == answer:
            continue
        remaining = [value for value in properties if value not in {source, answer}]
        for distractors in itertools.permutations(remaining, 3):
            choices = list(distractors)
            choices.insert(label, answer)
            symbolic = {
                "namespace": "e3-interface-easy-symbolic-v2",
                "fact_template": fact_template,
                "rule_template": rule_template,
                "source_property": source,
                "answer_property": answer,
                "choices": choices,
            }
            symbolic["skeleton_sha256"] = sha256_json(symbolic)
            yield symbolic


def build_easy_records(
    split: str,
    count: int,
    config: Mapping[str, Any],
    *,
    per_label_offset: int,
) -> tuple[PromptRecord, ...]:
    if count % 4:
        raise IntegrityFailure("easy split count must be divisible by four")
    vocabulary = tuple(str(value) for value in config["generator"]["easy_entities"])
    if len(vocabulary) < 4 or len(set(vocabulary)) != len(vocabulary):
        raise IntegrityFailure("easy entity vocabulary is not a fixed unique set")
    per_label = count // 4
    records: list[PromptRecord] = []
    for label in range(4):
        selected = list(
            itertools.islice(
                _easy_symbolic_skeletons(config, label),
                per_label_offset,
                per_label_offset + per_label,
            )
        )
        if len(selected) != per_label:
            raise IntegrityFailure("easy symbolic skeleton inventory exhausted")
        for local_index, symbolic in enumerate(selected):
            ordinal = label * per_label + local_index
            entity = vocabulary[local_index % len(vocabulary)]
            source = str(symbolic["source_property"])
            answer = str(symbolic["answer_property"])
            choices = tuple(str(value) for value in symbolic["choices"])
            fact_index = int(symbolic["fact_template"])
            rule_index = int(symbolic["rule_template"])
            fact = e2.UNARY_FACT_TEMPLATES[fact_index].format(
                entity=entity, property=source
            )
            rule = e2.UNARY_RULE_TEMPLATES[rule_index].format(
                body=source, head=answer
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
            verifier = e2.SymbolicVerifier(
                (entity,),
                (e2.UnaryAtom(entity, source),),
                (),
                (e2.Rule("unary", source, answer),),
            )
            closure = verifier.closure()
            target = e2.UnaryAtom(entity, answer)
            entailed_choices = [
                property_name
                for property_name in choices
                if e2.UnaryAtom(entity, property_name) in closure
            ]
            if (
                closure.get(target) != e2.ProofRecord(cost=1, shortest_proof_count=1)
                or entailed_choices != [answer]
                or choices[label] != answer
            ):
                raise IntegrityFailure("easy row failed symbolic one-hop verification")
            records.append(
                PromptRecord(
                    f"{split}-{ordinal:05d}",
                    prompt,
                    label,
                    query_choices,
                    {
                        "split": split,
                        "answer_position": label,
                        "proof_depth": 1,
                        "entity": entity,
                        "source_property": source,
                        "answer_property": answer,
                        "choices": list(choices),
                        "fact_template_index": fact_index,
                        "rule_template_index": rule_index,
                        "symbolic_skeleton_sha256": symbolic["skeleton_sha256"],
                        "symbolically_verified": True,
                        "shortest_proof_count": 1,
                    },
                )
            )
    expected = Counter({label: per_label for label in range(4)})
    if Counter(row.label for row in records) != expected:
        raise IntegrityFailure(f"{split} is not exactly label balanced")
    if len({row.prompt for row in records}) != count:
        raise IntegrityFailure(f"{split} prompts are not unique")
    return tuple(sorted(records, key=lambda row: row.record_id))


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


def _grounded_rule_applications(
    example: e2.DeductionExample, rule_index: int
) -> tuple[GroundedApplication, ...]:
    rule = example.rules[rule_index]
    applications: list[GroundedApplication] = []
    if rule.kind == "unary":
        for entity in example.entities:
            applications.append(
                GroundedApplication(
                    rule_index,
                    rule,
                    (e2.UnaryAtom(entity, rule.body),),
                    e2.UnaryAtom(entity, rule.head),
                )
            )
    elif rule.kind == "conjunction":
        if rule.body2 is None:
            raise IntegrityFailure("conjunction lacks its registered second body")
        for entity in example.entities:
            applications.append(
                GroundedApplication(
                    rule_index,
                    rule,
                    (
                        e2.UnaryAtom(entity, rule.body),
                        e2.UnaryAtom(entity, rule.body2),
                    ),
                    e2.UnaryAtom(entity, rule.head),
                )
            )
    else:
        if rule.relation is None:
            raise IntegrityFailure("relational rule lacks its registered relation")
        for relation in example.relation_facts:
            if relation.relation != rule.relation:
                continue
            if rule.kind == "rel_out_self":
                body_entity = head_entity = relation.source
            elif rule.kind == "rel_in_self":
                body_entity = head_entity = relation.target
            elif rule.kind == "rel_out_other":
                body_entity, head_entity = relation.source, relation.target
            else:  # pragma: no cover - E2 RuleKind is closed.
                raise IntegrityFailure(f"unknown registered rule kind: {rule.kind}")
            applications.append(
                GroundedApplication(
                    rule_index,
                    rule,
                    (e2.UnaryAtom(body_entity, rule.body),),
                    e2.UnaryAtom(head_entity, rule.head),
                )
            )
    return tuple(applications)


def _application_is_grounded(
    example: e2.DeductionExample, application: GroundedApplication
) -> bool:
    if application.rule_index < 0 or application.rule_index >= len(example.rules):
        return False
    return application in _grounded_rule_applications(example, application.rule_index)


def _candidate_match_signature(application: GroundedApplication) -> tuple[Any, ...]:
    schema = (
        "unary"
        if application.rule.kind == "unary"
        else "conjunction"
        if application.rule.kind == "conjunction"
        else "relational_unary_body"
    )
    return (
        schema,
        tuple(atom.property for atom in application.premises),
        len(application.premises),
        tuple("person" for _ in application.premises),
        "person",
    )


def _balanced_process_distractors(
    rows: Sequence[tuple[e2.DeductionExample, tuple[GroundedApplication, ...]]],
    step_index: int,
) -> dict[str, tuple[GroundedApplication, ...]]:
    active = [(example, trace[step_index]) for example, trace in rows if len(trace) > step_index]
    traces = {example.example_id: trace for example, trace in rows}
    eligible: dict[str, dict[str, list[GroundedApplication]]] = {}
    desired_entities: Counter[str] = Counter()
    for example, gold in active:
        available = set(example.unary_facts)
        trace = traces[example.example_id]
        for earlier in trace[:step_index]:
            available.add(earlier.conclusion)
        desired_entities[gold.conclusion.entity] += 3
        by_entity: dict[str, list[GroundedApplication]] = defaultdict(list)
        for rule_index in range(len(example.rules)):
            for candidate in _grounded_rule_applications(example, rule_index):
                if (
                    _candidate_match_signature(candidate)
                    == _candidate_match_signature(gold)
                    and not _application_is_valid(example, candidate, available)
                    and candidate != gold
                ):
                    by_entity[candidate.conclusion.entity].append(candidate)
        eligible[example.example_id] = {
            entity: sorted(
                set(candidates),
                key=lambda candidate: (
                    candidate.rule_index,
                    candidate.premises,
                    candidate.conclusion,
                ),
            )
            for entity, candidates in by_entity.items()
        }
        if sum(len(values) for values in eligible[example.example_id].values()) < 3:
            raise IntegrityFailure("fewer than three grounded matched distractors")

    entity_names = sorted(desired_entities)
    row_offset = 1
    entity_offset = row_offset + len(active)
    sink = entity_offset + len(entity_names)
    flow = _Dinic(sink + 1)
    for row_index, (example, _gold) in enumerate(active):
        flow.add_edge(0, row_offset + row_index, 3)
        for entity_index, entity in enumerate(entity_names):
            if entity in eligible[example.example_id]:
                flow.add_edge(row_offset + row_index, entity_offset + entity_index, 1)
    for entity_index, entity in enumerate(entity_names):
        flow.add_edge(entity_offset + entity_index, sink, desired_entities[entity])
    required = 3 * len(active)
    observed_flow = flow.maximum_flow(0, sink)
    if observed_flow != required:
        availability = {
            entity: sum(entity in values for values in eligible.values())
            for entity in entity_names
        }
        raise IntegrityFailure(
            f"step {step_index + 1} cannot exactly balance distractor entities: "
            f"flow={observed_flow}/{required}, desired={dict(desired_entities)}, "
            f"availability={availability}"
        )
    selected: dict[str, list[GroundedApplication]] = defaultdict(list)
    for row_index, (example, _gold) in enumerate(active):
        node = row_offset + row_index
        for edge in flow.graph[node]:
            if entity_offset <= edge.target < sink and edge.capacity == 0:
                entity = entity_names[edge.target - entity_offset]
                candidates = eligible[example.example_id][entity]
                selected[example.example_id].append(candidates[0])
        if len(selected[example.example_id]) != 3:
            raise IntegrityFailure("process balance flow did not select three candidates")
    return {key: tuple(value) for key, value in selected.items()}


def build_process_records(
    examples: Sequence[e2.DeductionExample],
    config: Mapping[str, Any],
) -> tuple[ProcessRecord, ...]:
    counters = [0] * 5
    traced = [(example, reconstruct_unique_trace(example)) for example in examples]
    distractors_by_step = [
        _balanced_process_distractors(traced, step_index) for step_index in range(5)
    ]
    output: list[ProcessRecord] = []
    for example, trace in traced:
        closure = example.verifier().closure()
        available = set(example.unary_facts)
        step_candidates: list[tuple[GroundedApplication, ...]] = []
        labels: list[int] = []
        for step_index, gold in enumerate(trace):
            gold_position = counters[step_index] % 4
            counters[step_index] += 1
            row = list(distractors_by_step[step_index][example.example_id])
            row.insert(gold_position, gold)
            if any(not _application_is_grounded(example, candidate) for candidate in row):
                raise IntegrityFailure("candidate is not a grounded registered rule")
            signatures = {_candidate_match_signature(candidate) for candidate in row}
            if signatures != {_candidate_match_signature(gold)}:
                raise IntegrityFailure("candidate body/type/arity matching failed")
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
    traced = [(example, reconstruct_unique_trace(example)) for example in examples]
    if any(len(trace) != 5 for _example, trace in traced):
        raise IntegrityFailure("fact base example is not depth five")
    positives_by_distance: dict[int, list[tuple[e2.DeductionExample, e2.UnaryAtom]]] = {
        distance: [] for distance in range(6)
    }
    for example, trace in traced:
        positives = (trace[0].premises[0],) + tuple(
            application.conclusion for application in trace
        )
        for distance, atom in enumerate(positives):
            positives_by_distance[distance].append((example, atom))

    negatives_by_distance: dict[int, dict[str, str]] = {}
    for distance, rows in positives_by_distance.items():
        desired = Counter(atom.property for _example, atom in rows)
        property_keys = sorted(desired)
        row_offset = 1
        property_offset = row_offset + len(rows)
        sink = property_offset + len(property_keys)
        flow = _Dinic(sink + 1)
        eligible: dict[str, set[str]] = {}
        for row_index, (example, positive) in enumerate(rows):
            closure = example.verifier().closure()
            context_counts = dict(example.context_token_counts)
            allowed = {
                property_name
                for property_name in property_keys
                if e2.UnaryAtom(positive.entity, property_name) not in closure
                and int(context_counts.get(property_name, 0)) > 0
            }
            if not allowed:
                raise IntegrityFailure(
                    f"fact distance {distance} lacks an occurrence-matched negative"
                )
            eligible[example.example_id] = allowed
            flow.add_edge(0, row_offset + row_index, 1)
            for property_index, property_key in enumerate(property_keys):
                if property_key in allowed:
                    flow.add_edge(
                        row_offset + row_index, property_offset + property_index, 1
                    )
        for property_index, property_key in enumerate(property_keys):
            flow.add_edge(
                property_offset + property_index, sink, desired[property_key]
            )
        if flow.maximum_flow(0, sink) != len(rows):
            raise IntegrityFailure(
                f"fact distance {distance} cannot exactly counterbalance predicates"
            )
        selected: dict[str, str] = {}
        for row_index, (example, _positive) in enumerate(rows):
            for edge in flow.graph[row_offset + row_index]:
                if property_offset <= edge.target < sink and edge.capacity == 0:
                    selected[example.example_id] = property_keys[
                        edge.target - property_offset
                    ]
            if example.example_id not in selected:
                raise IntegrityFailure("fact matching flow omitted a row")
        negatives_by_distance[distance] = selected

    for example_index, (example, trace) in enumerate(traced):
        positives = (trace[0].premises[0],) + tuple(
            application.conclusion for application in trace
        )
        closure = example.verifier().closure()
        for distance, positive in enumerate(positives):
            negative = e2.UnaryAtom(
                positive.entity,
                negatives_by_distance[distance][example.example_id],
            )
            if negative in closure:
                raise IntegrityFailure("matched fact negative is entailed")
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
                            "query_entity": atom.entity,
                            "query_predicate": atom.property,
                            "entity_context_occurrences": example.rendered_text.count(
                                atom.entity
                            ),
                            "predicate_context_occurrences": int(
                                dict(example.context_token_counts).get(atom.property, 0)
                            ),
                            "answer_relationship": (
                                f"base_answer_position_{example.answer_position}"
                            ),
                            "base_answer_position": example.answer_position,
                            "lexically_grounded": (
                                atom.entity in example.rendered_text
                                and atom.property in example.rendered_text
                            ),
                            "lexical_occurrence_relationship": (
                                "query_entity_and_predicate_present_in_prompt"
                            ),
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
        distance_rows = [
            row for row in records if int(row.metadata["distance"]) == distance
        ]
        for feature in (
            "query_entity",
            "query_predicate",
            "entity_context_occurrences",
            "lexical_occurrence_relationship",
            "answer_relationship",
        ):
            by_label = {
                label: Counter(
                    str(row.metadata[feature])
                    for row in distance_rows
                    if row.label == label
                )
                for label in (0, 1)
            }
            if by_label[0] != by_label[1]:
                raise IntegrityFailure(
                    f"fact distance {distance} is not {feature}-counterbalanced"
                )
        if not all(bool(row.metadata["lexically_grounded"]) for row in distance_rows):
            raise IntegrityFailure("fact query lacks lexical grounding")
    return tuple(records)


def _serialized_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def audit_easy_contract(
    train: Sequence[PromptRecord],
    validation: Sequence[PromptRecord],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    vocabulary = set(str(value) for value in config["generator"]["easy_entities"])
    train_entities = {str(row.metadata["entity"]) for row in train}
    validation_entities = {str(row.metadata["entity"]) for row in validation}
    train_skeletons = {
        str(row.metadata["symbolic_skeleton_sha256"]) for row in train
    }
    validation_skeletons = {
        str(row.metadata["symbolic_skeleton_sha256"]) for row in validation
    }
    if train_entities != vocabulary or validation_entities != vocabulary:
        raise IntegrityFailure("easy splits do not share the complete fixed vocabulary")
    if train_skeletons & validation_skeletons:
        raise IntegrityFailure("easy symbolic skeletons overlap across splits")
    if {row.prompt for row in train} & {row.prompt for row in validation}:
        raise IntegrityFailure("easy prompts overlap across splits")
    all_rows = (*train, *validation)
    if not all(
        bool(row.metadata.get("symbolically_verified"))
        and int(row.metadata.get("shortest_proof_count", 0)) == 1
        for row in all_rows
    ):
        raise IntegrityFailure("an easy row lacks its symbolic certificate")
    return {
        "contract": "shared_fixed_entity_vocabulary_and_skeleton_disjoint_splits",
        "fixed_entity_vocabulary": sorted(vocabulary),
        "train_entity_vocabulary": sorted(train_entities),
        "validation_entity_vocabulary": sorted(validation_entities),
        "shared_entity_count": len(train_entities & validation_entities),
        "train_symbolic_skeletons": len(train_skeletons),
        "validation_symbolic_skeletons": len(validation_skeletons),
        "overlapping_symbolic_skeletons": 0,
        "overlapping_prompts": 0,
        "symbolically_verified_rows": len(all_rows),
        "constraints_asserted_before_hashing": True,
    }


def audit_fact_balance(records: Sequence[PromptRecord]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for distance in range(6):
        rows = [row for row in records if int(row.metadata["distance"]) == distance]
        feature_tables: dict[str, Any] = {}
        for feature in (
            "query_entity",
            "query_predicate",
            "entity_context_occurrences",
            "lexical_occurrence_relationship",
            "answer_relationship",
        ):
            label_tables = {
                str(label): Counter(
                    str(row.metadata[feature]) for row in rows if row.label == label
                )
                for label in (0, 1)
            }
            if label_tables["0"] != label_tables["1"]:
                raise IntegrityFailure(
                    f"fact audit found {feature} imbalance at distance {distance}"
                )
            feature_tables[feature] = {
                label: _serialized_counter(table)
                for label, table in label_tables.items()
            }
        output[str(distance)] = {
            "labels": _serialized_counter(Counter(row.label for row in rows)),
            "features_by_label": feature_tables,
            "all_feature_contingencies_exactly_equal_across_labels": True,
        }
    return output


def audit_process_balance(records: Sequence[ProcessRecord]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for step_index in range(5):
        active = [row for row in records if len(row.trace) > step_index]
        if not active:
            continue
        position_label = Counter()
        features = {
            "schema": {0: Counter(), 1: Counter()},
            "entity": {0: Counter(), 1: Counter()},
            "body_predicates": {0: Counter(), 1: Counter()},
        }
        detailed: dict[str, dict[str, dict[str, Counter[Any]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(Counter))
        )
        for row in active:
            available = set(row.example.unary_facts)
            for earlier in row.trace[:step_index]:
                available.add(earlier.conclusion)
            valid = []
            for position, candidate in enumerate(row.candidates[step_index]):
                if not _application_is_grounded(row.example, candidate):
                    raise IntegrityFailure("process audit found an ungrounded candidate")
                label = int(_application_is_valid(row.example, candidate, available))
                valid.append(position) if label else None
                signature = _candidate_match_signature(candidate)
                values = {
                    "schema": signature[0],
                    "entity": candidate.conclusion.entity,
                    "body_predicates": signature[1],
                }
                position_label[(position, label)] += 1
                for feature, value in values.items():
                    features[feature][label][value] += 1
                    detailed[str(position)][str(label)][feature][value] += 1
            if valid != [row.candidate_labels[step_index]]:
                raise IntegrityFailure("process audit did not find exactly the gold candidate")
            row_signatures = {
                _candidate_match_signature(candidate)
                for candidate in row.candidates[step_index]
            }
            if row_signatures != {_candidate_match_signature(row.trace[step_index])}:
                raise IntegrityFailure("process candidates are not row-matched")
        for feature, labels in features.items():
            expected_invalid = Counter(
                {key: value * 3 for key, value in labels[1].items()}
            )
            if labels[0] != expected_invalid:
                raise IntegrityFailure(
                    f"process step {step_index + 1} {feature} labels are imbalanced"
                )
        expected_per_position = len(active) // 4
        if len(active) % 4 or any(
            position_label[(position, 1)] != expected_per_position
            or position_label[(position, 0)] != 3 * expected_per_position
            for position in range(4)
        ):
            raise IntegrityFailure("process candidate position/label balance changed")
        tables[str(step_index + 1)] = {
            "denominator_examples": len(active),
            "position_label_counts": {
                str(position): {
                    str(label): position_label[(position, label)] for label in (0, 1)
                }
                for position in range(4)
            },
            "feature_counts_by_validity_label": {
                feature: {
                    str(label): _serialized_counter(counter)
                    for label, counter in label_tables.items()
                }
                for feature, label_tables in features.items()
            },
            "feature_counts_by_position_and_validity_label": {
                position: {
                    label: {
                        feature: _serialized_counter(counter)
                        for feature, counter in feature_tables.items()
                    }
                    for label, feature_tables in label_tables.items()
                }
                for position, label_tables in detailed.items()
            },
            "invalid_to_valid_feature_ratio": 3,
            "grounded_registered_rule_applications": True,
            "matched_body_predicates_types_arities": True,
            "exactly_one_valid_per_row": True,
        }
    return tables


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
    easy_train_per_label = counts["easy_train"] // 4
    easy_train = build_easy_records(
        "easy_train", counts["easy_train"], config, per_label_offset=0
    )
    easy_validation = build_easy_records(
        "easy_validation",
        counts["easy_validation"],
        config,
        per_label_offset=easy_train_per_label,
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
    easy_audit = audit_easy_contract(easy_train, easy_validation, config)
    fact_balance = {
        "fact_probe_train": audit_fact_balance(fact_train),
        "fact_probe_validation": audit_fact_balance(fact_validation),
    }
    process_balance = {
        "hard_train": audit_process_balance(process_train),
        "hard_validation": audit_process_balance(process_validation),
    }
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
            "easy_contract": easy_audit,
            "fact_contingency_tables": fact_balance,
            "process_candidate_balance_tables": process_balance,
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


def assert_prehash_contracts(data: DiagnosticData) -> None:
    required_audits = {
        "easy_contract",
        "fact_contingency_tables",
        "process_candidate_balance_tables",
        "proof_trace_contract_sha256",
        "all_forbidden_access_counts_zero",
    }
    if not required_audits.issubset(data.generator_audit):
        raise IntegrityFailure("pre-hash audit surface is incomplete")
    if not bool(data.generator_audit["all_forbidden_access_counts_zero"]):
        raise IntegrityFailure("a forbidden cohort was accessed before hashing")
    if int(data.generator_audit["easy_contract"]["overlapping_symbolic_skeletons"]):
        raise IntegrityFailure("easy skeleton overlap reached the hash boundary")
    if any(
        len(records) != len({row.record_id for row in records})
        for records in (
            data.easy_train,
            data.easy_validation,
            data.fact_train,
            data.fact_validation,
        )
    ):
        raise IntegrityFailure("duplicate prompt record reached the hash boundary")


def split_hashes(data: DiagnosticData, config: Mapping[str, Any]) -> dict[str, str]:
    assert_prehash_contracts(data)
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


def parameter_boundary(
    module: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    named = dict(module.named_parameters())
    optimizer_ids = (
        {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer is not None
        else set()
    )
    optimized_names = sorted(
        name for name, parameter in named.items() if id(parameter) in optimizer_ids
    )
    trainable_names = sorted(
        name for name, parameter in named.items() if parameter.requires_grad
    )
    frozen_names = sorted(
        name for name, parameter in named.items() if not parameter.requires_grad
    )
    return {
        "total_parameters": sum(parameter.numel() for parameter in named.values()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in named.values() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in named.values() if not parameter.requires_grad
        ),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for parameter in named.values()
            if id(parameter) in optimizer_ids
        ),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "optimizer_parameter_names": optimized_names,
        "trainable_parameters_excluded_from_optimizer": sorted(
            set(trainable_names) - set(optimized_names)
        ),
        "frozen_parameters_in_optimizer": sorted(
            set(frozen_names) & set(optimized_names)
        ),
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
    probe_boundary = parameter_boundary(probe, optimizer)
    if (
        probe_boundary["frozen_parameters"] != 0
        or probe_boundary["optimizer_parameter_count"]
        != probe_boundary["trainable_parameters"]
        or probe_boundary["trainable_parameters_excluded_from_optimizer"]
        or probe_boundary["frozen_parameters_in_optimizer"]
    ):
        raise IntegrityFailure("affine probe optimizer boundary changed")
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
        "parameter_boundary": {
            **probe_boundary,
            "frozen_substrate_parameters_optimized": 0,
            "cached_representation_tensors_optimized": 0,
        },
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
    model_boundary = parameter_boundary(model, optimizer)
    core_boundary = parameter_boundary(model.core, optimizer)
    process_boundary = parameter_boundary(model.process_head, optimizer)
    if (
        model_boundary["optimizer_parameter_count"]
        != model_boundary["trainable_parameters"]
        or model_boundary["trainable_parameters_excluded_from_optimizer"]
        or model_boundary["frozen_parameters_in_optimizer"]
    ):
        raise IntegrityFailure("controller optimizer boundary changed")
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
        "parameter_boundaries": {
            "paired_module": model_boundary,
            "controller_core_and_answer_readout": core_boundary,
            "process_head": {
                **process_boundary,
                "effective_optimizer_updates": len(batches) if dense else 0,
                "receives_gradient": dense,
                "answer_forward_path_membership": False,
            },
            "frozen_substrate_parameters_optimized": 0,
            "cached_representation_tensors_optimized": 0,
        },
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
    post_sync_cap_check: Callable[[str], None],
) -> dict[str, Any]:
    if cuda_start is not None and cuda_end is not None:
        cuda_end.record()
        torch.cuda.synchronize(device)
        cuda_seconds = cuda_start.elapsed_time(cuda_end) / 1000.0
    else:
        cuda_seconds = 0.0
    post_sync_cap_check("post-synchronization cell finalization")
    wall_time = time.perf_counter() - started
    if not math.isfinite(wall_time) or wall_time < 0:
        raise IntegrityFailure("cell wall time is not finite and nonnegative")
    return {
        "wall_time_seconds": wall_time,
        "cuda_time_seconds": cuda_seconds,
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0,
    }


def _atomic_json_no_clobber(
    payload: Mapping[str, Any],
    path: Path,
    *,
    post_publish_check: Callable[[], None] | None = None,
) -> float:
    publication_started = time.perf_counter()
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
        with contextlib.suppress(OSError):
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        if post_publish_check is not None:
            post_publish_check()
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return time.perf_counter() - publication_started


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


def registered_probe_parameter_boundaries(
    config: Mapping[str, Any]
) -> dict[str, Any]:
    width = int(config["substrate"]["expected_hidden_width"])
    specifications = {
        "final_nonpadding_token_four_way": 4,
        "masked_prompt_mean_four_way": 4,
        "query_choices_span_mean_four_way": 4,
        "atomic_query_span_mean_binary": 2,
    }
    output: dict[str, Any] = {}
    for name, classes in specifications.items():
        head = nn.Linear(width, classes)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(config["probes"]["learning_rate"]),
            weight_decay=float(config["probes"]["weight_decay"]),
        )
        boundary = parameter_boundary(head, optimizer)
        if (
            boundary["frozen_parameters"]
            or boundary["optimizer_parameter_count"]
            != boundary["trainable_parameters"]
            or boundary["trainable_parameters_excluded_from_optimizer"]
            or boundary["frozen_parameters_in_optimizer"]
        ):
            raise IntegrityFailure(f"registered probe boundary failed for {name}")
        output[name] = {
            **boundary,
            "input_representation_width": width,
            "output_classes": classes,
            "frozen_substrate_parameters_optimized": 0,
            "cached_representation_tensors_optimized": 0,
        }
    return output


def _load_projection(config: Mapping[str, Any]) -> Mapping[str, Any]:
    path = REPO_ROOT / str(config["review"]["projection_path"])
    if not path.exists():
        raise IntegrityFailure("reviewed completion projection is missing")
    projection = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "projection_authorization": config["review"]["projection_authorization"],
    }
    for key, value in expected.items():
        if projection.get(key) != value:
            raise IntegrityFailure(f"projection binding mismatch: {key}")
    per_cell = projection.get("projected_cell_wall_seconds")
    if not isinstance(per_cell, Mapping) or tuple(per_cell) != CELL_NAMES:
        raise IntegrityFailure("projection does not contain the ordered six cells")
    cell_cap = float(config["compute"]["per_cell_wall_cap_seconds"])
    projected_values = [float(per_cell[name]) for name in CELL_NAMES]
    if any(
        not math.isfinite(value) or value < 0 or value > cell_cap
        for value in projected_values
    ):
        raise IntegrityFailure("a per-cell projection is nonfinite, negative, or over cap")
    suite_projection = float(projection["projected_suite_wall_seconds"])
    if (
        not math.isfinite(suite_projection)
        or suite_projection < 0
        or not math.isclose(suite_projection, sum(projected_values), abs_tol=1e-9)
        or suite_projection > float(config["compute"]["suite_wall_cap_seconds"])
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
            "easy_split_contract_audit": data.generator_audit["easy_contract"],
            "i0_pass": max(row["correct"] for row in views.values())
            >= int(config["probes"]["i0_correct_floor"]),
            "correct_floor": int(config["probes"]["i0_correct_floor"]),
            "representation_compute": {
                "train": train_compute,
                "validation": validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
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
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
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
            "matched_negative_contingency_tables": data.generator_audit[
                "fact_contingency_tables"
            ],
            "representation_compute": {
                "train": train_compute,
                "validation": validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
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
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
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
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
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
            "grounded_candidate_balance_tables": data.generator_audit[
                "process_candidate_balance_tables"
            ],
            "candidate_representation_compute": {
                "train": candidate_train_compute,
                "validation": candidate_validation_compute,
            },
            "compute": _finish_cell(started, cuda_start, cuda_end, device, check),
        }
        cell_order_completed.append(name)

        suite_wall = time.perf_counter() - suite_started
        suite_cap = float(config["compute"]["suite_wall_cap_seconds"])
        if not math.isfinite(suite_wall) or suite_wall < 0 or suite_wall > suite_cap:
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
                "affine_probe_heads": registered_probe_parameter_boundaries(config),
            },
            "cell_order_registered": list(CELL_NAMES),
            "cell_order_completed": cell_order_completed,
            "cells": cells,
            "decision": decision,
            "final_route_token": decision["token"],
            "compute": {
                "suite_wall_time_seconds_before_publication": suite_wall,
                "suite_wall_cap_seconds": int(
                    config["compute"]["suite_wall_cap_seconds"]
                ),
                "per_cell_wall_cap_seconds": int(
                    config["compute"]["per_cell_wall_cap_seconds"]
                ),
                "projection": projection,
                "publication_included_in_suite_timer": True,
                "post_publication_cap_check_required": True,
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
        def enforce_post_publication_cap() -> None:
            final_elapsed = time.perf_counter() - suite_started
            if (
                not math.isfinite(final_elapsed)
                or final_elapsed < 0
                or final_elapsed > suite_cap
            ):
                raise TimeoutError("suite exceeded total cap during result publication")

        _atomic_json_no_clobber(
            result,
            RESULT_PATH,
            post_publish_check=enforce_post_publication_cap,
        )
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
    if review_attestation != config["review"]["projection_authorization"]:
        raise IntegrityFailure("projection authorization does not match the blocking review")
    generation_started = time.perf_counter()
    full_data = build_diagnostic_data(config)
    validate_bindings(config, full_data)
    cpu_generation_seconds = time.perf_counter() - generation_started
    preflight = json.loads(E3_PREFLIGHT_RESULT.read_text(encoding="utf-8"))
    measured = preflight["compute"]
    cell_projection_seconds = {
        "I0_ONE_HOP_LINEAR_FLOOR": 150.0,
        "I1_HARD_ANSWER_LINEAR": 140.0,
        "I2_FACT_TRACE_LINEAR": 850.0,
        "S0_ONE_HOP_CONTROLLER_FLOOR": 80.0,
        "S1_HARD_ANSWER_ONLY": 260.0,
        "S2_HARD_PROCESS_DENSE": 700.0,
    }
    projected = sum(cell_projection_seconds.values())
    result = {
        "schema_version": "2.0.0",
        "diagnostic_only": True,
        "scientific_metrics_computed": False,
        "projection_kind": (
            "GPU_FREE_CPU_DERIVATION_FROM_IMMUTABLE_E3_PREFLIGHT_MEASUREMENT"
        ),
        "projection_authorization": review_attestation,
        "review_status": "BLOCKING_FINDINGS_RESOLVED_PENDING_REREVIEW",
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_measurement": {
            "artifact": str(E3_PREFLIGHT_RESULT.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "artifact_sha256": sha256_file(E3_PREFLIGHT_RESULT),
            "overall_wall_time_seconds": float(measured["overall_wall_time_seconds"]),
            "represented_tokens": sum(
                int(phase.get("generated_base_tokens", 0))
                for phase in measured["phases"]
            ),
            "optimizer_updates": int(measured["total_completed_optimizer_updates"]),
            "hardware": measured["hardware"],
        },
        "cpu_measurement": {
            "full_bound_data_generation_and_invariant_audit_wall_seconds": (
                cpu_generation_seconds
            ),
            "scientific_predictions_or_accuracy_computed": False,
        },
        "static_workload": {
            "easy_prompt_records": len(full_data.easy_train)
            + len(full_data.easy_validation),
            "hard_prompt_records": len(full_data.hard_train)
            + len(full_data.hard_validation),
            "fact_prompt_records": len(full_data.fact_train)
            + len(full_data.fact_validation),
            "candidate_application_records": sum(
                len(row.trace) * 4
                for row in (*full_data.process_train, *full_data.process_validation)
            ),
            "affine_probe_updates": 7000,
            "controller_updates": 1500,
        },
        "derivation": {
            "basis": (
                "safety-inclusive upper endpoints from the blocking full-pipeline "
                "review, anchored to the immutable E3 preflight measurement"
            ),
            "shared_load_and_cache_cost_allocated_to_first_constructing_cell": True,
            "publication_cost_included_in_S2": True,
            "projection_safety_multiplier": float(
                config["compute"]["projection_safety_multiplier"]
            ),
            "no_GPU_work_performed": True,
        },
        "projected_cell_wall_seconds": cell_projection_seconds,
        "projected_suite_wall_seconds": projected,
        "per_cell_wall_cap_seconds": int(
            config["compute"]["per_cell_wall_cap_seconds"]
        ),
        "suite_wall_cap_seconds": int(config["compute"]["suite_wall_cap_seconds"]),
        "all_cell_projections_finite_nonnegative_and_within_cap": True,
        "suite_projection_finite_nonnegative_and_within_cap": True,
        "completed_utc": utc_now(),
    }
    path = REPO_ROOT / str(config["review"]["projection_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0

    # Retained below as provenance for the prior GPU sampling method. It is
    # unreachable; the registered projection is deliberately GPU-free.
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
    assert_prehash_contracts(data)
    hashes = split_hashes(data, config)
    if tuple(hashes) != SPLIT_NAMES:
        raise AssertionError("self-test split hash surface changed")
    model = PairedController(config)
    if e3.parameter_count(model.core) > 30_000_000:
        raise AssertionError("E3 controller parameter cap exceeded")
    optimizer = controller_optimizer(model, config)
    boundary = parameter_boundary(model, optimizer)
    if (
        boundary["optimizer_parameter_count"] != boundary["trainable_parameters"]
        or boundary["trainable_parameters_excluded_from_optimizer"]
        or boundary["frozen_parameters_in_optimizer"]
    ):
        raise AssertionError("self-test controller optimizer boundary failed")
    probe_boundaries = registered_probe_parameter_boundaries(config)
    if set(probe_boundaries) != {
        "final_nonpadding_token_four_way",
        "masked_prompt_mean_four_way",
        "query_choices_span_mean_four_way",
        "atomic_query_span_mean_binary",
    }:
        raise AssertionError("self-test probe boundary inventory changed")

    process_rows = (*data.process_train, *data.process_validation)
    candidate_count = 0
    for process in process_rows:
        if len(process.trace) not in (2, 3, 4, 5):
            raise AssertionError("self-test trace depth changed")
        replay_trace(process.example, process.trace)
        available = set(process.example.unary_facts)
        for step, candidates in enumerate(process.candidates):
            candidate_count += len(candidates)
            valid = []
            for index, candidate in enumerate(candidates):
                if not _application_is_grounded(process.example, candidate):
                    raise AssertionError("self-test found an ungrounded candidate")
                if _application_is_valid(process.example, candidate, available):
                    valid.append(index)
            if valid != [process.candidate_labels[step]]:
                raise AssertionError("self-test candidate replay failed")
            if {
                _candidate_match_signature(candidate) for candidate in candidates
            } != {_candidate_match_signature(process.trace[step])}:
                raise AssertionError("self-test candidate matching failed")
            available.add(process.trace[step].conclusion)

    fact_rows = (*data.fact_train, *data.fact_validation)
    if not all(
        row.metadata.get("lexically_grounded")
        and row.label == int(bool(row.metadata["entailed"]))
        for row in fact_rows
    ):
        raise AssertionError("self-test fact-row invariant failed")
    if not all(
        row.metadata.get("symbolically_verified")
        and row.metadata.get("shortest_proof_count") == 1
        for row in (*data.easy_train, *data.easy_validation)
    ):
        raise AssertionError("self-test easy symbolic invariant failed")

    frozen_thresholds = {
        "i0": (int(config["probes"]["i0_correct_floor"]), 922),
        "i1": (int(config["probes"]["i1_correct_floor"]), 205),
        "i2": (int(config["probes"]["i2_distance_correct_floor"]), 308),
        "s0": (int(config["controller_training"]["s0_correct_floor"]), 922),
        "hard": (int(config["controller_training"]["hard_correct_floor"]), 205),
        "gain": (
            int(config["controller_training"]["dense_gain_correct_floor"]),
            77,
        ),
    }
    if any(observed != expected for observed, expected in frozen_thresholds.values()):
        raise AssertionError("self-test threshold contract changed")

    def cells_fixture(
        *,
        i0: bool = True,
        i1: bool = False,
        fact: str = "FACT_NONE",
        s0: bool = True,
        s1: bool = False,
        s2: bool = False,
        process: str = "PROCESS_NONE",
        gain: bool = False,
    ) -> dict[str, Any]:
        return {
            CELL_NAMES[0]: {"i0_pass": i0},
            CELL_NAMES[1]: {"i1_pass": i1},
            CELL_NAMES[2]: {"fact_class": fact},
            CELL_NAMES[3]: {"s0_pass": s0},
            CELL_NAMES[4]: {"s1_competent": s1},
            CELL_NAMES[5]: {
                "s2_competent": s2,
                "process_class": process,
                "dense_gain": gain,
            },
        }

    branch_fixtures = {
        "REGISTER_DENSE_SUPERVISION_E3B": cells_fixture(
            s2=True, process="PROCESS_FULL", gain=True
        ),
        "REGISTER_INTERFACE_REDESIGN/i0": cells_fixture(i0=False),
        "REGISTER_INTERFACE_REDESIGN/s0": cells_fixture(s0=False),
        "REGISTER_INTERFACE_REDESIGN/hard_exposed": cells_fixture(i1=True),
        "REGISTER_INTERFACE_REDESIGN/s1": cells_fixture(s1=True),
        "REGISTER_TASK_FAMILY_CHANGE": cells_fixture(fact="FACT_PARTIAL"),
        "KILL_SYNTHETIC_DEDUCTION_FAMILY": cells_fixture(),
        "VOID_NO_ROUTE / MIXED_DIAGNOSTIC_PATTERN": cells_fixture(
            s2=True, process="PROCESS_NONE", gain=False
        ),
    }
    for expected, fixture in branch_fixtures.items():
        expected_token = (
            "REGISTER_INTERFACE_REDESIGN"
            if expected.startswith("REGISTER_INTERFACE_REDESIGN/")
            else expected
        )
        if route_result(fixture)["token"] != expected_token:
            raise AssertionError(f"self-test branch fixture failed: {expected}")

    timeout_config = copy.deepcopy(config)
    timeout_config["compute"]["per_cell_wall_cap_seconds"] = -1
    started, check, cuda_start, cuda_end = _cell_timer(
        "SELF_TEST_TIMEOUT", timeout_config, torch.device("cpu"), time.perf_counter()
    )
    try:
        _finish_cell(started, cuda_start, cuda_end, torch.device("cpu"), check)
    except TimeoutError:
        pass
    else:
        raise AssertionError("post-synchronization cell cap was not enforced")

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
                "miniature_prompt_records_checked": (
                    len(data.easy_train)
                    + len(data.easy_validation)
                    + len(data.fact_train)
                    + len(data.fact_validation)
                ),
                "miniature_process_records_checked": len(process_rows),
                "miniature_candidates_checked": candidate_count,
                "trace_replay_all_records": True,
                "candidate_grounding_matching_and_exactly_one_valid_all_records": True,
                "easy_fact_process_balance_assertions": True,
                "probe_and_controller_optimizer_boundaries": True,
                "frozen_thresholds": True,
                "adversarial_branch_table_fixtures": len(branch_fixtures),
                "post_synchronization_time_cap_fixture": True,
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
