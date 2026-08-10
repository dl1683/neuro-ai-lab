"""E2 shortcut-resistant latch-mechanics pilot.

This is a provenance runner, not a landing command.  ``--self-test`` is the
only CPU-only verification entry point.  ``--run`` performs the preregistered
two-seed training/evaluation and must not be used until the independent
pre-training review required by AGENTS.md has passed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "exp_e2_latch_mechanics_config.json"
RESULT_PATH = HERE / "results" / "exp_e2_latch_mechanics.json"
FINAL_TOKENS = {"PROCEED", "FAIL", "VOID"}
PUBLIC_SPLITS = ("controller_train", "selector_harvest", "selector_calibration")
IMMUTABLE_RESULT_TEMP_PREFIX = ".e2-immutable-result-tmp-"
IMMUTABLE_RESULT_TEMP_SUFFIX = ".tmp"
IMMUTABLE_RESULT_TEMP_GLOB = (
    f"{IMMUTABLE_RESULT_TEMP_PREFIX}*{IMMUTABLE_RESULT_TEMP_SUFFIX}"
)
ATOMIC_CRASH_WINDOWS = (
    "after_fsync_before_link",
    "after_link_before_unlink",
)


UNARY_FACT_TEMPLATES = (
    "{entity} is {property}.",
    "The person called {entity} is {property}.",
    "It is known that {entity} is {property}.",
    "One established trait of {entity} is {property}.",
    "Records describe {entity} as {property}.",
    "We are told {entity} is {property}.",
)
RELATION_FACT_TEMPLATES = (
    "{source} {relation} {target}.",
    "The person {source} {relation} the person {target}.",
    "It is known that {source} {relation} {target}.",
    "Records say {source} {relation} {target}.",
    "We are told {source} {relation} {target}.",
    "The relation is that {source} {relation} {target}.",
)
UNARY_RULE_TEMPLATES = (
    "Every {body} person is also {head}.",
    "Anyone who is {body} must be {head}.",
    "Being {body} implies being {head}.",
    "All people described as {body} are {head}.",
    "If someone is {body}, then that person is {head}.",
    "A {body} person is necessarily {head}.",
)
CONJUNCTION_RULE_TEMPLATES = (
    "Anyone both {body} and {body2} is {head}.",
    "If someone is {body} and {body2}, that person is {head}.",
    "The traits {body} and {body2} together imply {head}.",
    "Every person who is {body} as well as {body2} is {head}.",
    "A person is {head} whenever they are both {body} and {body2}.",
    "Being simultaneously {body} and {body2} makes a person {head}.",
)
REL_OUT_SELF_RULE_TEMPLATES = (
    "If a {body} person {relation} someone, the first person is {head}.",
    "Every {body} person who {relation} another person is {head}.",
    "A person is {head} if they are {body} and {relation} someone.",
    "When a {body} person {relation} another, that person is {head}.",
    "Being {body} while one {relation} someone implies being {head}.",
    "Whoever is {body} and {relation} somebody must be {head}.",
)
REL_IN_SELF_RULE_TEMPLATES = (
    "If someone {relation} a {body} person, that second person is {head}.",
    "Every {body} person whom another person {relation} is {head}.",
    "A {body} person is {head} when somebody {relation} them.",
    "When another person {relation} someone {body}, the latter is {head}.",
    "Being {body} and being someone another {relation} implies {head}.",
    "Whoever is {body} and is {relation} by somebody must be {head}.",
)
REL_OUT_OTHER_RULE_TEMPLATES = (
    "If a {body} person {relation} someone, the other person is {head}.",
    "Anyone whom a {body} person {relation} is {head}.",
    "A person becomes {head} when a {body} person {relation} them.",
    "When someone {body} {relation} another, the latter is {head}.",
    "Being the person a {body} individual {relation} implies being {head}.",
    "Whoever a {body} person {relation} must be {head}.",
)

RULE_TEMPLATES = {
    "unary": UNARY_RULE_TEMPLATES,
    "conjunction": CONJUNCTION_RULE_TEMPLATES,
    "rel_out_self": REL_OUT_SELF_RULE_TEMPLATES,
    "rel_in_self": REL_IN_SELF_RULE_TEMPLATES,
    "rel_out_other": REL_OUT_OTHER_RULE_TEMPLATES,
}


def template_inventory_hash() -> str:
    return sha256_json(
        {
            "unary_fact": UNARY_FACT_TEMPLATES,
            "relation_fact": RELATION_FACT_TEMPLATES,
            "rules": RULE_TEMPLATES,
        }
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("experiment_id") != "exp_e2_latch_mechanics":
        raise ValueError("wrong mechanics-pilot configuration")
    if config.get("schema_version") != "1.1.0":
        raise ValueError("mechanics-pilot config schema version changed")
    expected_templates = config["generator"]["template_inventory_sha256"]
    if template_inventory_hash() != expected_templates:
        raise ValueError("rendering-template inventory differs from the frozen config")
    if not math.isclose(sum(config["generator"]["lure_mixture"].values()), 1.0):
        raise ValueError("lure mixture does not sum to one")
    if tuple(config["training"]["model_seeds"]) != (42, 31415):
        raise ValueError("the two preregistered model seeds changed")
    if tuple(config["training"]["train_horizons"]) != (1, 2, 4):
        raise ValueError("controller training horizons changed")
    if tuple(config["selectors"]["fit_horizons"]) != tuple(range(1, 17)):
        raise ValueError("selector-fit horizons must be exactly 1..16")
    if tuple(config["evaluation"]["horizons"]) != tuple(range(1, 33)):
        raise ValueError("evaluation horizons must be exactly 1..32")
    if tuple(config["evaluation"]["t_star_candidates"]) != tuple(range(1, 16)):
        raise ValueError("t-star candidates must be exactly 1..15")
    if int(config["training"]["shared_controller_tokens"]) != 500_000:
        raise ValueError("shared controller token budget changed")
    if int(config["training"]["logical_arm_exposure"]) != 500_000:
        raise ValueError("logical arm exposure changed")
    if int(config["generator"]["examples_per_split"]["test"]) != 4096:
        raise ValueError("fixed test size changed")
    if int(config["generator"]["test_counterfactual_pairs"]) * 2 != 4096:
        raise ValueError("test counterfactual-pair count changed")
    selector_mix = float(config["selectors"]["on_policy_fraction"]) + float(
        config["selectors"]["checkpoint_replay_fraction"]
    )
    if not math.isclose(selector_mix, 1.0):
        raise ValueError("on-policy and replay fractions do not sum to one")
    hysteresis = config["selectors"]["hysteretic_latch"]
    if tuple(float(value) for value in hysteresis["delta_grid"]) != (
        0.0,
        0.02,
        0.05,
        0.1,
    ):
        raise ValueError("hysteresis delta grid changed")
    if int(hysteresis["selection_horizon"]) != 16:
        raise ValueError("hysteresis selection horizon must be 16")
    if int(hysteresis["freeze_through_horizon"]) != 32:
        raise ValueError("hysteresis delta must remain frozen through horizon 32")
    if not math.isclose(float(hysteresis["minimum_calibration_gain_retention"]), 0.9):
        raise ValueError("hysteresis calibration gain-retention floor changed")
    if not bool(hysteresis["informational_only"]) or not bool(
        hysteresis["excluded_from_adjudication"]
    ):
        raise ValueError("arm 4 must remain informational and non-adjudicating")
    configured_result = Path(config["result_artifact"]["path"])
    expected_result = Path("experiments/06_uesd/results/exp_e2_latch_mechanics.json")
    if configured_result != expected_result:
        raise ValueError("immutable result path changed")
    if config["result_artifact"].get("schema_version") != "1.1.0":
        raise ValueError("immutable result schema version changed")
    if (
        config["result_artifact"].get("write_mode")
        != "same_directory_fsync_atomic_no_clobber"
    ):
        raise ValueError("immutable result write mode changed")
    return config


@dataclass(frozen=True, order=True)
class UnaryAtom:
    entity: str
    property: str


@dataclass(frozen=True, order=True)
class RelationAtom:
    source: str
    relation: str
    target: str


RuleKind = Literal[
    "unary", "conjunction", "rel_out_self", "rel_in_self", "rel_out_other"
]


@dataclass(frozen=True)
class Rule:
    kind: RuleKind
    body: str
    head: str
    body2: str | None = None
    relation: str | None = None


@dataclass(frozen=True)
class ProofRecord:
    cost: int
    shortest_proof_count: int


class SymbolicVerifier:
    """Finite typed-Datalog closure with minimum rule-application costs."""

    def __init__(
        self,
        entities: Sequence[str],
        unary_facts: Sequence[UnaryAtom],
        relation_facts: Sequence[RelationAtom],
        rules: Sequence[Rule],
    ) -> None:
        self.entities = tuple(entities)
        self.unary_facts = tuple(unary_facts)
        self.relation_facts = tuple(relation_facts)
        self.rules = tuple(rules)
        entity_set = set(self.entities)
        if len(entity_set) != len(self.entities):
            raise ValueError("typed Datalog entity identifiers must be unique")
        if any(fact.entity not in entity_set for fact in self.unary_facts):
            raise TypeError("unary fact has a non-person argument")
        if any(
            fact.source not in entity_set or fact.target not in entity_set
            for fact in self.relation_facts
        ):
            raise TypeError("binary relation has a non-person argument")
        for rule in self.rules:
            if rule.kind.startswith("rel_") and rule.relation is None:
                raise TypeError("binary rule lacks its typed person relation")
            if rule.kind == "conjunction" and rule.body2 is None:
                raise TypeError("conjunction rule lacks its second unary predicate")

    def closure(self) -> dict[UnaryAtom, ProofRecord]:
        records: dict[UnaryAtom, ProofRecord] = {
            fact: ProofRecord(cost=0, shortest_proof_count=1)
            for fact in self.unary_facts
        }
        max_rounds = max(1, len(self.entities) * (len(self.rules) + 1))
        for _ in range(max_rounds):
            changed = False
            additions: list[tuple[UnaryAtom, int, int]] = []
            for rule in self.rules:
                additions.extend(self._applications(rule, records))
            for atom, cost, proof_count in additions:
                previous = records.get(atom)
                if previous is None or cost < previous.cost:
                    records[atom] = ProofRecord(cost, 0)
                    changed = True
            if not changed:
                break
        else:
            raise RuntimeError("symbolic closure did not stabilize")

        counted: dict[UnaryAtom, ProofRecord] = {
            atom: ProofRecord(record.cost, 1 if record.cost == 0 else 0)
            for atom, record in records.items()
        }
        maximum_cost = max((record.cost for record in counted.values()), default=0)
        for cost in range(1, maximum_cost + 1):
            contributions: Counter[UnaryAtom] = Counter()
            for rule in self.rules:
                for atom, candidate_cost, proof_count in self._applications(
                    rule, counted
                ):
                    if (
                        candidate_cost == cost
                        and counted.get(atom, ProofRecord(-1, 0)).cost == cost
                    ):
                        contributions[atom] += proof_count
            for atom, proof_count in contributions.items():
                counted[atom] = ProofRecord(cost, min(2, proof_count))
        return counted

    def _applications(
        self, rule: Rule, records: Mapping[UnaryAtom, ProofRecord]
    ) -> list[tuple[UnaryAtom, int, int]]:
        output: list[tuple[UnaryAtom, int, int]] = []
        if rule.kind == "unary":
            for entity in self.entities:
                premise = records.get(UnaryAtom(entity, rule.body))
                if premise:
                    output.append(
                        (
                            UnaryAtom(entity, rule.head),
                            premise.cost + 1,
                            premise.shortest_proof_count,
                        )
                    )
        elif rule.kind == "conjunction":
            if rule.body2 is None:
                raise ValueError("conjunction rule lacks body2")
            for entity in self.entities:
                first = records.get(UnaryAtom(entity, rule.body))
                second = records.get(UnaryAtom(entity, rule.body2))
                if first and second:
                    output.append(
                        (
                            UnaryAtom(entity, rule.head),
                            first.cost + second.cost + 1,
                            first.shortest_proof_count * second.shortest_proof_count,
                        )
                    )
        else:
            if rule.relation is None:
                raise ValueError("relation rule lacks relation")
            for relation in self.relation_facts:
                if relation.relation != rule.relation:
                    continue
                if rule.kind == "rel_out_self":
                    body_entity, head_entity = relation.source, relation.source
                elif rule.kind == "rel_in_self":
                    body_entity, head_entity = relation.target, relation.target
                elif rule.kind == "rel_out_other":
                    body_entity, head_entity = relation.source, relation.target
                else:  # pragma: no cover - RuleKind makes this unreachable.
                    raise ValueError(f"unknown rule kind: {rule.kind}")
                premise = records.get(UnaryAtom(body_entity, rule.body))
                if premise:
                    output.append(
                        (
                            UnaryAtom(head_entity, rule.head),
                            premise.cost + 1,
                            premise.shortest_proof_count,
                        )
                    )
        return output


@dataclass(frozen=True)
class DeductionExample:
    example_id: str
    counterfactual_group: str
    counterfactual_member: int
    split: str
    skeleton_hash: str
    proof_depth: int
    lure_type: str
    template_family: int
    target_entity: str
    choice_properties: tuple[str, str, str, str]
    answer_position: int
    answer_property: str
    entities: tuple[str, ...]
    unary_facts: tuple[UnaryAtom, ...]
    relation_facts: tuple[RelationAtom, ...]
    rules: tuple[Rule, ...]
    rendered_text: str
    context_token_counts: tuple[tuple[str, int], ...]
    late_lure_proof_depth: int | None

    def verifier(self) -> SymbolicVerifier:
        return SymbolicVerifier(
            self.entities, self.unary_facts, self.relation_facts, self.rules
        )

    def result_metadata(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "counterfactual_group": self.counterfactual_group,
            "counterfactual_member": self.counterfactual_member,
            "split": self.split,
            "skeleton_hash": self.skeleton_hash,
            "proof_depth": self.proof_depth,
            "lure_type": self.lure_type,
            "template_family": self.template_family,
            "answer_position": self.answer_position,
            "gold_label": self.answer_position,
            "answer_property": self.answer_property,
        }


class PairSpec(NamedTuple):
    depth: int
    lure_type: str
    template_family: int
    first_position: int
    second_position: int


def _split_for_skeleton(skeleton_hash: str) -> str:
    percentile = int(skeleton_hash[:8], 16) % 100
    if percentile < 70:
        return "controller_train"
    if percentile < 85:
        return "selector_harvest"
    return "selector_calibration"


def _name_owner(first: str, second: str) -> str:
    value = int(sha256_bytes(f"name-v1:{first}-{second}".encode("ascii"))[:8], 16)
    bucket = value % 100
    if bucket < 45:
        return "controller_train"
    if bucket < 65:
        return "selector_harvest"
    if bucket < 82:
        return "selector_calibration"
    return "test"


class DeductionGenerator:
    """Deterministic paired generator with symbolic shortcut checks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.gcfg = config["generator"]
        self.properties = tuple(self.gcfg["property_words"])
        self.relations = tuple(self.gcfg["relation_verbs"])
        self.syllables = tuple(self.gcfg["name_syllables"])
        self._name_pools = self._build_name_pools()

    def _build_name_pools(self) -> dict[str, dict[str, list[str]]]:
        pools: dict[str, dict[str, list[str]]] = {
            split: defaultdict(list) for split in (*PUBLIC_SPLITS, "test")
        }
        for first in self.syllables:
            for second in self.syllables:
                if first == second:
                    continue
                owner = _name_owner(first, second)
                pools[owner][first].append(f"{first}-{second}")
        for split, grouped in pools.items():
            if not any(len(names) >= 2 for names in grouped.values()):
                raise AssertionError(f"no same-syllable name pair for {split}")
        return pools

    def generate_dataset(
        self,
        examples_per_split: Mapping[str, int] | None = None,
        *,
        seed_offset: int = 0,
    ) -> dict[str, list[DeductionExample]]:
        counts = dict(examples_per_split or self.gcfg["examples_per_split"])
        datasets: dict[str, list[DeductionExample]] = {}
        used_hashes: set[str] = set()
        for split in (*PUBLIC_SPLITS, "test"):
            count = int(counts[split])
            if count % 4:
                raise ValueError("each split size must be divisible by four")
            datasets[split] = self._generate_split(
                split, count, used_hashes, seed_offset=seed_offset
            )
            used_hashes.update(example.skeleton_hash for example in datasets[split])
        self.audit_dataset(datasets)
        return datasets

    def _generate_split(
        self,
        split: str,
        example_count: int,
        globally_used_hashes: set[str],
        *,
        seed_offset: int,
    ) -> list[DeductionExample]:
        pair_specs = self._allocate_pair_specs(example_count)
        examples: list[DeductionExample] = []
        local_hashes: set[str] = set()
        nonce = seed_offset * 10_000_000
        for pair_index, spec in enumerate(pair_specs):
            for _ in range(int(self.gcfg["max_generation_attempts_per_pair"])):
                rng_seed = (
                    int(self.gcfg["seed"])
                    + seed_offset * 1_000_003
                    + nonce * 97
                    + pair_index
                )
                candidate = self._make_pair(split, spec, nonce, random.Random(rng_seed))
                nonce += 1
                skeleton_hash = candidate[0].skeleton_hash
                assigned = (
                    "test" if split == "test" else _split_for_skeleton(skeleton_hash)
                )
                if assigned != split:
                    continue
                if (
                    skeleton_hash in local_hashes
                    or skeleton_hash in globally_used_hashes
                ):
                    continue
                self._verify_pair(candidate)
                examples.extend(candidate)
                local_hashes.add(skeleton_hash)
                break
            else:
                raise RuntimeError(f"could not generate valid pair for {split}: {spec}")
        if len(examples) != example_count:
            raise AssertionError("split construction produced the wrong example count")
        examples.sort(key=lambda example: example.example_id)
        return examples

    def _allocate_pair_specs(self, example_count: int) -> list[PairSpec]:
        pair_count = example_count // 2
        lure_pair_counts = {
            lure: int(round(pair_count * float(weight)))
            for lure, weight in self.gcfg["lure_mixture"].items()
        }
        difference = pair_count - sum(lure_pair_counts.values())
        lure_pair_counts["late"] += difference
        if any(count % 2 for count in lure_pair_counts.values()):
            raise ValueError("lure counts must permit two-pair position blocks")

        specs: list[PairSpec] = []
        depths = tuple(int(value) for value in self.gcfg["proof_depths"])
        templates = tuple(range(int(self.gcfg["template_families"])))
        global_block_index = 0
        for lure_index, lure in enumerate(("late", "direction", "clean")):
            lure_blocks = lure_pair_counts[lure] // 2
            depth_base, depth_remainder = divmod(lure_blocks, len(depths))
            depth_counts = {
                depth: depth_base
                + int((depth_index - lure_index) % len(depths) < depth_remainder)
                for depth_index, depth in enumerate(depths)
            }
            block_index = 0
            for depth_index, depth in enumerate(depths):
                depth_blocks = depth_counts[depth]
                for _ in range(depth_blocks):
                    template = templates[global_block_index % len(templates)]
                    rotation = (block_index + lure_index) % 4
                    positions = tuple((rotation + offset) % 4 for offset in range(4))
                    specs.append(
                        PairSpec(depth, lure, template, positions[0], positions[1])
                    )
                    specs.append(
                        PairSpec(depth, lure, template, positions[2], positions[3])
                    )
                    block_index += 1
                    global_block_index += 1
        return specs

    def _choose_names(
        self, split: str, entity_count: int, rng: random.Random
    ) -> tuple[str, ...]:
        grouped = self._name_pools[split]
        shared_firsts = [first for first, names in grouped.items() if len(names) >= 2]
        shared_first = rng.choice(shared_firsts)
        target, lure = rng.sample(grouped[shared_first], 2)
        available = sorted(
            {
                name
                for names in grouped.values()
                for name in names
                if name not in {target, lure}
            }
        )
        others = rng.sample(available, entity_count - 2)
        return (target, lure, *others)

    def _make_pair(
        self, split: str, spec: PairSpec, nonce: int, rng: random.Random
    ) -> tuple[DeductionExample, DeductionExample]:
        entity_count = rng.randint(
            int(self.gcfg["entities_min"]), int(self.gcfg["entities_max"])
        )
        entity_count = max(entity_count, 8)
        names = self._choose_names(split, entity_count, rng)
        target, lure, switch_aux, lure_aux, direction_aux, *witnesses = names
        if len(witnesses) < 3:
            raise AssertionError("generator needs at least three witness entities")

        shuffled_properties = list(self.properties)
        rng.shuffle(shuffled_properties)
        options = tuple(shuffled_properties[:4])
        non_options = shuffled_properties[4:]
        option_a, option_b, option_c, _option_d = options
        chain = non_options[: spec.depth]
        remaining_non_options = non_options[spec.depth :]
        relation_order = list(self.relations)
        rng.shuffle(relation_order)
        causal_relation = relation_order[0]

        unary_facts = [UnaryAtom(target, chain[0])]
        base_relations = [RelationAtom(target, causal_relation, switch_aux)]
        rules: list[Rule] = [
            Rule("unary", chain[index], chain[index + 1])
            for index in range(len(chain) - 1)
        ]
        rules.extend(
            [
                Rule("rel_out_self", chain[-1], option_a, relation=causal_relation),
                Rule("rel_in_self", chain[-1], option_b, relation=causal_relation),
            ]
        )
        late_lure_depth: int | None = None

        if spec.lure_type == "late":
            extra = rng.choice(
                tuple(int(x) for x in self.gcfg["late_lure_extra_steps"])
            )
            extension_steps = extra + 1
            if len(remaining_non_options) < extension_steps - 1:
                raise AssertionError("not enough predicates for the late lure")
            unary_facts.append(UnaryAtom(lure, chain[0]))
            late_relation = relation_order[1]
            base_relations.append(RelationAtom(lure, late_relation, lure_aux))
            previous = chain[-1]
            for index in range(extension_steps):
                head = (
                    option_c
                    if index == extension_steps - 1
                    else remaining_non_options[index]
                )
                rules.append(
                    Rule("rel_out_self", previous, head, relation=late_relation)
                )
                previous = head
            late_lure_depth = spec.depth + extra
        elif spec.lure_type == "direction":
            if not remaining_non_options:
                raise AssertionError("not enough predicates for direction lure")
            guard = remaining_non_options[0]
            direction_relation = relation_order[1]
            unary_facts.append(UnaryAtom(direction_aux, guard))
            base_relations.append(
                RelationAtom(target, direction_relation, direction_aux)
            )
            rules.append(
                Rule("rel_out_other", guard, option_c, relation=direction_relation)
            )

        option_rule_counts = Counter()
        for rule in rules:
            if rule.head in options:
                option_rule_counts[rule.head] += 1
            if rule.body in options:
                option_rule_counts[rule.body] += 1
            if rule.body2 in options:
                option_rule_counts[rule.body2] += 1
        equalized_count = max(option_rule_counts.values(), default=0) + 1
        witness_cycle = [switch_aux, lure_aux, direction_aux, *witnesses]
        witness_index = 0
        for option in options:
            for _ in range(equalized_count - option_rule_counts[option]):
                entity = witness_cycle[witness_index % len(witness_cycle)]
                if entity == target:
                    raise AssertionError("option witness cannot be the target")
                unary_facts.append(UnaryAtom(entity, option))
                witness_index += 1

        target_rule_count = rng.randint(
            int(self.gcfg["rules_min"]), int(self.gcfg["rules_max"])
        )
        while len(rules) < target_rule_count:
            body, body2 = rng.sample(non_options, 2)
            rules.append(Rule("conjunction", body, chain[0], body2=body2))
        if len(rules) > int(self.gcfg["rules_max"]):
            raise AssertionError("constructed too many rules")

        canonical_skeleton = {
            "namespace": "independent-test-v1" if split == "test" else "public-v1",
            "depth": spec.depth,
            "lure": spec.lure_type,
            "entity_count": entity_count,
            "facts": [
                [
                    names.index(f.entity),
                    non_options.index(f.property)
                    if f.property in non_options
                    else 100 + options.index(f.property),
                ]
                for f in unary_facts
            ],
            "relations": [
                [
                    names.index(f.source),
                    relation_order.index(f.relation),
                    names.index(f.target),
                ]
                for f in base_relations
            ],
            "rules": [
                [
                    rule.kind,
                    non_options.index(rule.body)
                    if rule.body in non_options
                    else 100 + options.index(rule.body),
                    non_options.index(rule.head)
                    if rule.head in non_options
                    else 100 + options.index(rule.head),
                    None if rule.body2 is None else non_options.index(rule.body2),
                    None
                    if rule.relation is None
                    else relation_order.index(rule.relation),
                ]
                for rule in rules
            ],
        }
        for unordered_field in ("facts", "relations", "rules"):
            canonical_skeleton[unordered_field] = sorted(
                canonical_skeleton[unordered_field], key=canonical_json
            )
        skeleton_hash = sha256_json(canonical_skeleton)
        group_id = f"{split}-{skeleton_hash[:16]}"

        remaining_positions = [
            position
            for position in range(4)
            if position not in {spec.first_position, spec.second_position}
        ]
        rng.shuffle(remaining_positions)
        choices: list[str | None] = [None, None, None, None]
        choices[spec.first_position] = option_a
        choices[spec.second_position] = option_b
        choices[remaining_positions[0]] = option_c
        choices[remaining_positions[1]] = options[3]
        fixed_choices = tuple(str(choice) for choice in choices)

        pair: list[DeductionExample] = []
        for member, reverse in enumerate((False, True)):
            relation_facts = list(base_relations)
            causal = relation_facts[0]
            if reverse:
                relation_facts[0] = RelationAtom(
                    causal.target, causal.relation, causal.source
                )
            answer_property = option_b if reverse else option_a
            answer_position = fixed_choices.index(answer_property)
            rendered = self._render(
                target,
                fixed_choices,
                unary_facts,
                relation_facts,
                rules,
                spec.template_family,
                random.Random(int(skeleton_hash[:16], 16)),
            )
            logical_counts = Counter(fact.property for fact in unary_facts)
            for rule in rules:
                logical_counts[rule.body] += 1
                logical_counts[rule.head] += 1
                if rule.body2 is not None:
                    logical_counts[rule.body2] += 1
            pair.append(
                DeductionExample(
                    example_id=f"{group_id}-{member}",
                    counterfactual_group=group_id,
                    counterfactual_member=member,
                    split=split,
                    skeleton_hash=skeleton_hash,
                    proof_depth=spec.depth,
                    lure_type=spec.lure_type,
                    template_family=spec.template_family,
                    target_entity=target,
                    choice_properties=fixed_choices,
                    answer_position=answer_position,
                    answer_property=answer_property,
                    entities=tuple(names),
                    unary_facts=tuple(unary_facts),
                    relation_facts=tuple(relation_facts),
                    rules=tuple(rules),
                    rendered_text=rendered,
                    context_token_counts=tuple(sorted(logical_counts.items())),
                    late_lure_proof_depth=late_lure_depth,
                )
            )
        return pair[0], pair[1]

    def _render(
        self,
        target: str,
        choices: tuple[str, str, str, str],
        unary_facts: Sequence[UnaryAtom],
        relation_facts: Sequence[RelationAtom],
        rules: Sequence[Rule],
        template_family: int,
        rng: random.Random,
    ) -> str:
        facts: list[str] = []
        for index, fact in enumerate(unary_facts):
            template = UNARY_FACT_TEMPLATES[(template_family + index) % 6]
            facts.append(template.format(entity=fact.entity, property=fact.property))
        for index, fact in enumerate(relation_facts):
            template = RELATION_FACT_TEMPLATES[(template_family + index + 2) % 6]
            facts.append(
                template.format(
                    source=fact.source, relation=fact.relation, target=fact.target
                )
            )
        rendered_rules: list[str] = []
        for index, rule in enumerate(rules):
            template = RULE_TEMPLATES[rule.kind][(template_family + index + 1) % 6]
            rendered_rules.append(
                template.format(
                    body=rule.body,
                    body2=rule.body2,
                    head=rule.head,
                    relation=rule.relation,
                )
            )
        rng.shuffle(facts)
        rng.shuffle(rendered_rules)
        return "\n".join(
            [
                "Facts:",
                *facts,
                "Rules:",
                *rendered_rules,
                f"Query: Which property must hold for {target}?",
                "Choices: " + " / ".join(choices),
            ]
        )

    def _verify_pair(self, pair: tuple[DeductionExample, DeductionExample]) -> None:
        first, second = pair
        if first.skeleton_hash != second.skeleton_hash:
            raise AssertionError("counterfactual twins crossed skeleton groups")
        if first.answer_property == second.answer_property:
            raise AssertionError("counterfactual did not change the answer")
        first_relations = list(first.relation_facts)
        second_relations = list(second.relation_facts)
        differing = [
            (left, right)
            for left, right in zip(first_relations, second_relations)
            if left != right
        ]
        if len(differing) != 1:
            raise AssertionError("counterfactual must change one relation only")
        left, right = differing[0]
        if (left.source, left.relation, left.target) != (
            right.target,
            right.relation,
            right.source,
        ):
            raise AssertionError("counterfactual change is not a direction reversal")
        if Counter(_word_tokens(first.rendered_text)) != Counter(
            _word_tokens(second.rendered_text)
        ):
            raise AssertionError("counterfactual word counts changed")
        for example in pair:
            self._verify_example(example)

    def _verify_example(self, example: DeductionExample) -> None:
        closure = example.verifier().closure()
        target_options = {
            option: closure.get(UnaryAtom(example.target_entity, option))
            for option in example.choice_properties
        }
        derivable = {option: proof for option, proof in target_options.items() if proof}
        if set(derivable) != {example.answer_property}:
            raise AssertionError(f"target choices are not unique: {derivable}")
        proof = derivable[example.answer_property]
        if proof.cost != example.proof_depth:
            raise AssertionError(
                f"wrong shortest proof depth: {proof.cost} != {example.proof_depth}"
            )
        if proof.shortest_proof_count != 1:
            raise AssertionError("target proof is not unique at minimum cost")
        context_counts = dict(example.context_token_counts)
        option_counts = [context_counts[option] for option in example.choice_properties]
        if len(set(option_counts)) != 1:
            raise AssertionError(f"option context counts differ: {option_counts}")
        for option in example.choice_properties:
            if not any(
                entity != example.target_entity and UnaryAtom(entity, option) in closure
                for entity in example.entities
            ):
                raise AssertionError(f"option lacks non-target witness: {option}")
        if example.lure_type == "late":
            depths = [
                record.cost
                for atom, record in closure.items()
                if atom.entity != example.target_entity
                and atom.property in example.choice_properties
                and atom.property != example.answer_property
                and record.cost > 0
            ]
            if example.late_lure_proof_depth not in depths:
                raise AssertionError("late lure proof is missing")
            if example.late_lure_proof_depth - example.proof_depth not in (2, 3):
                raise AssertionError("late lure offset is outside the frozen mixture")
        if example.lure_type == "direction":
            direction_lure_found = False
            for relation_index in range(1, len(example.relation_facts)):
                reversed_relations = list(example.relation_facts)
                relation = reversed_relations[relation_index]
                reversed_relations[relation_index] = RelationAtom(
                    relation.target, relation.relation, relation.source
                )
                reversed_closure = SymbolicVerifier(
                    example.entities,
                    example.unary_facts,
                    reversed_relations,
                    example.rules,
                ).closure()
                if any(
                    option != example.answer_property
                    and UnaryAtom(example.target_entity, option) in reversed_closure
                    for option in example.choice_properties
                ):
                    direction_lure_found = True
                    break
            if not direction_lure_found:
                raise AssertionError("direction lure does not activate under reversal")
        if not (
            int(self.gcfg["ground_facts_min"])
            <= len(example.unary_facts) + len(example.relation_facts)
            <= int(self.gcfg["ground_facts_max"])
        ):
            raise AssertionError("ground-fact count outside the frozen range")
        if not (
            int(self.gcfg["rules_min"])
            <= len(example.rules)
            <= int(self.gcfg["rules_max"])
        ):
            raise AssertionError("rule count outside the frozen range")

    def audit_dataset(
        self, datasets: Mapping[str, Sequence[DeductionExample]]
    ) -> dict[str, Any]:
        skeleton_sets = {
            split: {example.skeleton_hash for example in examples}
            for split, examples in datasets.items()
        }
        name_sets = {
            split: {entity for example in examples for entity in example.entities}
            for split, examples in datasets.items()
        }
        for index, left in enumerate(datasets):
            for right in tuple(datasets)[index + 1 :]:
                if skeleton_sets[left] & skeleton_sets[right]:
                    raise AssertionError(f"skeleton leakage: {left} / {right}")
                if name_sets[left] & name_sets[right]:
                    raise AssertionError(f"name-combination leakage: {left} / {right}")
        split_audits: dict[str, Any] = {}
        for split, examples in datasets.items():
            lure_counts = Counter(example.lure_type for example in examples)
            expected = {
                lure: round(len(examples) * float(weight))
                for lure, weight in self.gcfg["lure_mixture"].items()
            }
            if lure_counts != Counter(expected):
                raise AssertionError(
                    f"lure mixture mismatch in {split}: {lure_counts} != {expected}"
                )
            pair_counts = Counter(example.counterfactual_group for example in examples)
            if set(pair_counts.values()) != {2}:
                raise AssertionError(
                    "counterfactual groups must contain exactly two twins"
                )
            position_strata: dict[tuple[int, str, int], Counter[int]] = defaultdict(
                Counter
            )
            for example in examples:
                position_strata[
                    (example.proof_depth, example.lure_type, example.template_family)
                ][example.answer_position] += 1
            for stratum, positions in position_strata.items():
                if set(positions) != {0, 1, 2, 3} or len(set(positions.values())) != 1:
                    raise AssertionError(
                        f"answer positions unbalanced in {split} {stratum}: {positions}"
                    )
            split_audits[split] = {
                "examples": len(examples),
                "counterfactual_groups": len(pair_counts),
                "unique_skeleton_hashes": len(skeleton_sets[split]),
                "unique_name_combinations": len(name_sets[split]),
                "lure_counts": dict(sorted(lure_counts.items())),
                "proof_depth_counts": dict(
                    sorted(Counter(x.proof_depth for x in examples).items())
                ),
                "template_counts": dict(
                    sorted(Counter(x.template_family for x in examples).items())
                ),
                "answer_position_counts": dict(
                    sorted(Counter(x.answer_position for x in examples).items())
                ),
            }
        return {
            "splits": split_audits,
            "symbolically_verified_examples": sum(map(len, datasets.values())),
            "symbolically_verified_counterfactual_pairs": sum(
                len(examples) // 2 for examples in datasets.values()
            ),
            "typed_datalog_signature": self.gcfg["types"],
            "skeleton_hash_disjoint": True,
            "name_combination_disjoint": True,
            "counterfactual_group_atomic": True,
            "counterfactual_single_relation_reversal": True,
            "counterfactual_word_count_delta_max": 0,
            "unique_shortest_proof_and_single_target_choice": True,
            "equal_option_context_occurrences": True,
            "every_option_has_non_target_derivation": True,
            "late_lure_depth_offset_two_or_three": True,
            "direction_lure_reversal_verified": True,
            "label_randomization_construction": {
                "names_sampled_before_answer_assignment": True,
                "choice_positions_assigned_independently_of_property_shuffle": True,
                "premise_order_seeded_only_by_skeleton": True,
                "paraphrase_family_balanced_before_rendering": True,
            },
            "answer_position_balanced_within_strata": True,
            "lure_mixture_exact": True,
        }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[0-9]+", text.lower())


class LocalTokenizer:
    """Fixed regex/subword tokenizer; pseudonames split at the hyphen."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        tcfg = config["tokenizer"]
        self.pattern = re.compile(str(tcfg["pattern"]))
        self.lowercase = bool(tcfg["lowercase"])
        special = list(tcfg["special_tokens"])
        vocabulary = set(special)
        sources: list[str] = []
        for templates in (
            UNARY_FACT_TEMPLATES,
            RELATION_FACT_TEMPLATES,
            UNARY_RULE_TEMPLATES,
            CONJUNCTION_RULE_TEMPLATES,
            REL_OUT_SELF_RULE_TEMPLATES,
            REL_IN_SELF_RULE_TEMPLATES,
            REL_OUT_OTHER_RULE_TEMPLATES,
        ):
            sources.extend(templates)
        sources.extend(
            [
                "Facts Rules Query Which property must hold for Choices",
                ": . ? / - ,",
                *config["generator"]["property_words"],
                *config["generator"]["relation_verbs"],
                *config["generator"]["name_syllables"],
            ]
        )
        for source in sources:
            vocabulary.update(self.tokenize(source))
        ordered = special + sorted(vocabulary - set(special))
        self.id_to_token = tuple(ordered)
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def tokenize(self, text: str) -> list[str]:
        normalized = text.lower() if self.lowercase else text
        return self.pattern.findall(normalized)

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        ids = [
            self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)
        ]
        if self.unk_id in ids:
            unknown = [
                token for token in self.tokenize(text) if token not in self.token_to_id
            ]
            raise ValueError(
                f"fixed tokenizer encountered unknown tokens: {unknown[:8]}"
            )
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        maximum = int(self.config["tokenizer"]["max_sequence_tokens"])
        if len(ids) > maximum:
            raise ValueError(f"example has {len(ids)} tokens, maximum is {maximum}")
        return ids

    def decode_tokens(
        self, ids: Sequence[int], *, skip_special: bool = True
    ) -> list[str]:
        specials = set(self.config["tokenizer"]["special_tokens"])
        tokens = [self.id_to_token[index] for index in ids]
        return [token for token in tokens if not skip_special or token not in specials]

    def round_trip_tokens(self, text: str) -> bool:
        return self.decode_tokens(self.encode(text)) == self.tokenize(text)

    def vocabulary_hash(self) -> str:
        return sha256_json(self.id_to_token)


def collate_examples(
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    encoded = [tokenizer.encode(example.rendered_text) for example in examples]
    max_length = max(map(len, encoded))
    tokens = torch.full(
        (len(examples), max_length), tokenizer.pad_id, dtype=torch.long, device=device
    )
    padding_mask = torch.ones(
        (len(examples), max_length), dtype=torch.bool, device=device
    )
    for row, ids in enumerate(encoded):
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        padding_mask[row, : len(ids)] = False
    labels = torch.tensor(
        [example.answer_position for example in examples],
        dtype=torch.long,
        device=device,
    )
    nonpadding_tokens = sum(len(ids) + 1 for ids in encoded)  # + answer token
    return tokens, padding_mask, labels, nonpadding_tokens


class CommonRecurrentModel(nn.Module):
    def __init__(self, config: Mapping[str, Any], vocabulary_size: int) -> None:
        super().__init__()
        cfg = config["common_model"]
        width = int(cfg["width"])
        heads = int(cfg["attention_heads"])
        ffn = int(cfg["ffn_width"])
        dropout = float(cfg["dropout"])
        self.width = width
        self.plan_slots_count = int(cfg["latent_plan_slots"])
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.position = nn.Parameter(
            torch.zeros(1, int(config["tokenizer"]["max_sequence_tokens"]), width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            int(cfg["input_encoder_layers"]),
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(width)
        self.plan_slots = nn.Parameter(torch.randn(self.plan_slots_count, width) * 0.02)
        self.prompt_to_plan = nn.Linear(width, width)

        def controller_layer() -> nn.TransformerDecoderLayer:
            return nn.TransformerDecoderLayer(
                width,
                heads,
                ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.controller_layers = nn.ModuleList(
            controller_layer() for _ in range(int(cfg["controller_layers"]))
        )
        self.controller_norm = nn.LayerNorm(width)
        hidden = int(cfg["prefix_projector_hidden"])
        self.prefix_projector = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
        )
        answer_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.answer_decoder = answer_layer
        self.answer_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.answer_norm = nn.LayerNorm(width)
        self.choice_head = nn.Linear(width, int(cfg["answer_choices"]))

    def encode_context(
        self, tokens: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        context = self.encoder(embedded, src_key_padding_mask=padding_mask)
        context = self.encoder_norm(context)
        valid = (~padding_mask).unsqueeze(-1).to(context.dtype)
        prompt = (context * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return context, prompt

    def recurrent_states(
        self,
        context: Tensor,
        prompt: Tensor,
        padding_mask: Tensor,
        max_horizon: int,
    ) -> list[Tensor]:
        plan = self.plan_slots.unsqueeze(0).expand(context.shape[0], -1, -1)
        plan = plan + self.prompt_to_plan(prompt).unsqueeze(1)
        states: list[Tensor] = []
        for _ in range(max_horizon):
            for layer in self.controller_layers:
                plan = layer(plan, context, memory_key_padding_mask=padding_mask)
            plan = self.controller_norm(plan)
            states.append(plan)
        return states

    def logits_from_state(self, state: Tensor) -> Tensor:
        prefix = self.prefix_projector(state)
        query = self.answer_query.unsqueeze(0).expand(state.shape[0], -1, -1)
        decoded = self.answer_decoder(query, prefix)
        return self.choice_head(self.answer_norm(decoded[:, 0]))

    def forward(
        self, tokens: Tensor, padding_mask: Tensor, horizon: int
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        context, prompt = self.encode_context(tokens, padding_mask)
        states = self.recurrent_states(context, prompt, padding_mask, horizon)
        return self.logits_from_state(states[-1]), states, prompt


class EncoderOnlyControl(nn.Module):
    def __init__(self, config: Mapping[str, Any], vocabulary_size: int) -> None:
        super().__init__()
        cfg = config["encoder_control"]
        width = int(cfg["width"])
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.position = nn.Parameter(
            torch.zeros(1, int(config["tokenizer"]["max_sequence_tokens"]), width)
        )
        layer = nn.TransformerEncoderLayer(
            width,
            int(cfg["attention_heads"]),
            int(cfg["ffn_width"]),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, int(cfg["layers"]), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, int(config["common_model"]["answer_choices"]))

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        embedded = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        states = self.norm(self.encoder(embedded, src_key_padding_mask=padding_mask))
        valid = (~padding_mask).unsqueeze(-1).to(states.dtype)
        pooled = (states * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.head(pooled)


class ConfidenceCalibrator(nn.Module):
    FEATURE_NAMES = (
        "maximum_answer_probability",
        "top_two_probability_margin",
        "entropy",
        "t_over_16",
    )

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)


class LatentProgressCritic(nn.Module):
    ALLOWED_FEATURE_NAMES = (
        "prompt_conditioned_pooled_latent",
        "prompt_conditioned_pooled_delta",
        "frozen_prompt_representation",
        "update_norm",
        "consecutive_update_cosine",
        "cross_horizon_latent_agreement",
        "t_over_16",
    )
    FORBIDDEN_FEATURE_NAMES = (
        "raw_decoder_logits",
        "answer_probability",
        "top_two_margin",
        "entropy",
        "answer_identity",
        "sampled_answer_frequency",
        "response_log_probability",
    )

    def __init__(self, config: Mapping[str, Any], width: int = 512) -> None:
        super().__init__()
        cfg = config["critic"]
        self.width = width
        input_width = int(cfg["input_width"])
        bottleneck = int(cfg["input_adapter_bottleneck"])
        self.input_norm = nn.LayerNorm(input_width)
        self.input_adapter = nn.Sequential(
            nn.Linear(input_width, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, input_width),
        )
        first, second = (int(value) for value in cfg["mlp_widths"])
        self.mlp = nn.Sequential(
            nn.Linear(int(cfg["input_width"]), first),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(second, 1),
        )

    def _pool(self, state: Tensor, prompt: Tensor) -> Tensor:
        weights = torch.softmax(
            (state * prompt.unsqueeze(1)).sum(-1) / math.sqrt(self.width),
            dim=1,
        )
        return (weights.unsqueeze(-1) * state).sum(1)

    def features_for_trajectory(
        self, states: Sequence[Tensor], prompt: Tensor, step_denominator: float = 16.0
    ) -> Tensor:
        pooled = [self._pool(state, prompt) for state in states]
        rows: list[Tensor] = []
        zeros = torch.zeros_like(pooled[0])
        for index, current in enumerate(pooled):
            previous = pooled[index - 1] if index else zeros
            delta = current - previous
            previous_delta = (
                pooled[index - 1] - pooled[index - 2] if index >= 2 else zeros
            )
            update_norm = delta.norm(dim=-1, keepdim=True)
            update_cosine = F.cosine_similarity(
                delta, previous_delta, dim=-1, eps=1e-8
            ).unsqueeze(-1)
            if index:
                prior_mean = torch.stack(pooled[:index], dim=0).mean(0)
                agreement = F.cosine_similarity(
                    current, prior_mean, dim=-1, eps=1e-8
                ).unsqueeze(-1)
            else:
                agreement = torch.zeros_like(update_norm)
            step = torch.full_like(update_norm, (index + 1) / step_denominator)
            row = torch.cat(
                [
                    current,
                    delta,
                    prompt,
                    update_norm,
                    update_cosine,
                    agreement,
                    step,
                ],
                dim=-1,
            )
            if row.shape[-1] != 1540:
                raise AssertionError(
                    f"critic feature width is {row.shape[-1]}, not 1540"
                )
            rows.append(row)
        return torch.stack(rows, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.input_norm(features)
        adapted = normalized + self.input_adapter(normalized)
        return self.mlp(adapted).squeeze(-1)


def confidence_features(logits: Tensor, horizon: int) -> Tensor:
    probabilities = logits.softmax(-1)
    top_two = probabilities.topk(2, dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    step = torch.full_like(top_two[:, 0], horizon / 16.0)
    return torch.stack(
        [top_two[:, 0], top_two[:, 0] - top_two[:, 1], entropy, step], dim=-1
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    processed_tokens: int,
    config_hash: str,
    code_hash: str,
    seed: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model_uses_cuda = any(parameter.is_cuda for parameter in model.parameters())
    return {
        "schema_version": 1,
        "processed_nonpadding_tokens": processed_tokens,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if model_uses_cuda else [],
        "history": list(history),
    }


def restore_checkpoint_rng_states(
    payload: Mapping[str, Any], *, model_uses_cuda: bool
) -> None:
    try:
        python_rng_state = payload["python_rng_state"]
        torch_rng_state = payload["torch_rng_state"]
        cuda_rng_state_all = payload["cuda_rng_state_all"]
    except KeyError as error:
        raise RuntimeError("checkpoint is missing a required RNG state") from error
    random.setstate(python_rng_state)
    torch.random.set_rng_state(torch_rng_state.cpu())
    if model_uses_cuda:
        if not cuda_rng_state_all:
            raise RuntimeError("CUDA training checkpoint is missing CUDA RNG states")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])


def load_checked_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config_hash: str,
    code_hash: str,
    expected_seed: int,
    expected_processed_tokens: int,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("config_hash") != config_hash
        or payload.get("code_hash") != code_hash
        or payload.get("seed") != expected_seed
        or payload.get("processed_nonpadding_tokens") != expected_processed_tokens
    ):
        raise RuntimeError(f"checkpoint provenance mismatch: {path}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        restore_checkpoint_rng_states(
            payload,
            model_uses_cuda=any(parameter.is_cuda for parameter in model.parameters()),
        )
    return payload


def _exact_fill(
    length_to_indices: Mapping[int, Sequence[int]], target: int
) -> list[int] | None:
    if target == 0:
        return []
    lengths = sorted(length_to_indices)
    previous: list[tuple[int, int] | None] = [None] * (target + 1)
    previous[0] = (0, -1)
    for total in range(target + 1):
        if previous[total] is None:
            continue
        for length in lengths:
            new_total = total + length
            if new_total > target:
                break
            if previous[new_total] is None:
                previous[new_total] = (total, length)
    if previous[target] is None:
        return None
    chosen_lengths: list[int] = []
    cursor = target
    while cursor:
        prior, length = previous[cursor]  # type: ignore[misc]
        chosen_lengths.append(length)
        cursor = prior
    counters: Counter[int] = Counter()
    indices: list[int] = []
    for length in reversed(chosen_lengths):
        choices = length_to_indices[length]
        indices.append(choices[counters[length] % len(choices)])
        counters[length] += 1
    return indices


def exact_token_schedule(
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    token_budget: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Select examples whose non-padding input-plus-answer count is exact."""

    lengths = [len(tokenizer.encode(example.rendered_text)) + 1 for example in examples]
    length_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        length_to_indices[length].append(index)
    rng = random.Random(seed)
    schedule: list[int] = []
    remaining = token_budget
    tail_limit = min(token_budget, max(8192, max(lengths) * 24))
    while remaining > tail_limit:
        eligible = [
            index
            for index, length in enumerate(lengths)
            if length <= remaining - tail_limit
        ]
        if not eligible:
            break
        index = rng.choice(eligible)
        length = lengths[index]
        schedule.append(index)
        remaining -= length
    fill = _exact_fill(length_to_indices, remaining)
    while fill is None and schedule:
        index = schedule.pop()
        remaining += lengths[index]
        if remaining > min(token_budget, 65536):
            break
        fill = _exact_fill(length_to_indices, remaining)
    if fill is None:
        raise RuntimeError(
            f"could not construct exact {token_budget}-token schedule; remainder={remaining}"
        )
    schedule.extend(fill)
    actual = sum(lengths[index] for index in schedule)
    if actual != token_budget:
        raise AssertionError(f"token schedule mismatch: {actual} != {token_budget}")
    return schedule, lengths


def pack_dynamic_batches(
    schedule: Sequence[int], lengths: Sequence[int], target_tokens: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index in schedule:
        length = lengths[index]
        if current and current_tokens + length > target_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += length
    if current:
        batches.append(current)
    return batches


def learning_rate_at_tokens(
    processed: int, total: int, config: Mapping[str, Any]
) -> float:
    cfg = config["training"]
    peak = float(cfg["peak_learning_rate"])
    minimum = float(cfg["minimum_learning_rate"])
    warmup = max(1, round(total * float(cfg["warmup_token_fraction"])))
    if processed <= warmup:
        return peak * processed / warmup
    progress = min(1.0, (processed - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    cfg = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["peak_learning_rate"]),
        betas=tuple(float(value) for value in cfg["betas"]),
        weight_decay=float(cfg["weight_decay"]),
        eps=float(cfg["epsilon"]),
    )


@contextlib.contextmanager
def training_autocast(device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        with contextlib.nullcontext():
            yield


def iter_index_batches(indices: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start : start + batch_size])


@torch.no_grad()
def recurrent_validation_loss(
    model: CommonRecurrentModel,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> dict[str, float]:
    model.eval()
    losses = {1: 0.0, 2: 0.0, 4: 0.0}
    total = 0
    indices = list(range(len(examples)))
    for batch_indices in iter_index_batches(indices, batch_size):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = collate_examples(batch, tokenizer, device)
        context, prompt = model.encode_context(tokens, mask)
        states = model.recurrent_states(context, prompt, mask, 4)
        for horizon in losses:
            logits = model.logits_from_state(states[horizon - 1])
            losses[horizon] += F.cross_entropy(logits, labels, reduction="sum").item()
        total += len(batch)
    return {
        "horizon_1": losses[1] / total,
        "horizon_2": losses[2] / total,
        "horizon_4": losses[4] / total,
        "mean": sum(losses.values()) / (3 * total),
    }


@torch.no_grad()
def encoder_validation_loss(
    model: EncoderOnlyControl,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> float:
    model.eval()
    loss_sum = 0.0
    for batch_indices in iter_index_batches(list(range(len(examples))), batch_size):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = collate_examples(batch, tokenizer, device)
        loss_sum += F.cross_entropy(model(tokens, mask), labels, reduction="sum").item()
    return loss_sum / len(examples)


def train_common_model(
    model: CommonRecurrentModel,
    train_examples: Sequence[DeductionExample],
    calibration_examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    seed: int,
    checkpoint_dir: Path,
    config_hash: str,
    code_hash: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model.to(device)
    optimizer = make_optimizer(model, config)
    total_budget = int(config["training"]["shared_controller_tokens"])
    fractions = [
        float(value) for value in config["training"]["checkpoint_token_fractions"]
    ]
    checkpoint_tokens = [round(total_budget * fraction) for fraction in fractions]
    history: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    processed = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_path = checkpoint_dir / "common_tokens_000000.pt"
    if not initial_path.exists():
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                processed_tokens=0,
                config_hash=config_hash,
                code_hash=code_hash,
                seed=seed,
                history=history,
            ),
            initial_path,
        )
    else:
        load_checked_checkpoint(
            initial_path,
            model,
            optimizer=optimizer,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=0,
            map_location=device,
            restore_rng=True,
        )
    checkpoint_records.append(
        {
            "processed_tokens": 0,
            "path": str(initial_path),
            "sha256": sha256_file(initial_path),
        }
    )
    started = time.perf_counter()
    horizon_schedule = tuple(
        int(value) for value in config["training"]["train_horizons"]
    )
    update_index = 0
    for target_tokens in checkpoint_tokens:
        path = checkpoint_dir / f"common_tokens_{target_tokens:06d}.pt"
        if path.exists():
            payload = load_checked_checkpoint(
                path,
                model,
                optimizer=optimizer,
                config_hash=config_hash,
                code_hash=code_hash,
                expected_seed=seed,
                expected_processed_tokens=target_tokens,
                map_location=device,
                restore_rng=True,
            )
            processed = int(payload["processed_nonpadding_tokens"])
            history = list(payload.get("history", []))
            update_index = len(history)
        else:
            segment_budget = target_tokens - processed
            schedule, lengths = exact_token_schedule(
                train_examples,
                tokenizer,
                segment_budget,
                seed + processed * 17,
            )
            batches = pack_dynamic_batches(
                schedule,
                lengths,
                int(config["training"]["target_nonpadding_tokens_per_update"]),
            )
            model.train()
            for batch_indices in batches:
                batch = [train_examples[index] for index in batch_indices]
                tokens, mask, labels, batch_tokens = collate_examples(
                    batch, tokenizer, device
                )
                horizon = horizon_schedule[update_index % len(horizon_schedule)]
                optimizer.zero_grad(set_to_none=True)
                with training_autocast(device):
                    logits, _, _ = model(tokens, mask, horizon)
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                processed += batch_tokens
                learning_rate = learning_rate_at_tokens(processed, total_budget, config)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                history.append(
                    {
                        "update": update_index,
                        "horizon": horizon,
                        "processed_nonpadding_tokens": processed,
                        "batch_nonpadding_tokens": batch_tokens,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
                        "learning_rate": learning_rate,
                    }
                )
                update_index += 1
            if processed != target_tokens:
                raise AssertionError(
                    f"token accounting failed: {processed} != {target_tokens}"
                )
            validation = recurrent_validation_loss(
                model, calibration_examples, tokenizer, device
            )
            history[-1]["checkpoint_validation"] = validation
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    processed_tokens=processed,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    seed=seed,
                    history=history,
                ),
                path,
            )
        checkpoint_records.append(
            {
                "processed_tokens": target_tokens,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    if processed != total_budget:
        raise AssertionError("shared controller did not reach the frozen token budget")
    return {
        "processed_nonpadding_tokens": processed,
        "logical_arm_exposure": int(config["training"]["logical_arm_exposure"]),
        "wall_time_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoints": checkpoint_records,
        "selected_checkpoint": checkpoint_records[-1],
        "selected_rule": config["training"]["common_checkpoint_selection"],
    }


def train_encoder_control(
    model: EncoderOnlyControl,
    train_examples: Sequence[DeductionExample],
    calibration_examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    seed: int,
    checkpoint_dir: Path,
    config_hash: str,
    code_hash: str,
) -> dict[str, Any]:
    torch.manual_seed(seed + 1_000_000)
    random.seed(seed + 1_000_000)
    model.to(device)
    optimizer = make_optimizer(model, config)
    total_budget = int(config["training"]["encoder_control_tokens"])
    checkpoint_tokens = [
        round(total_budget * float(value))
        for value in config["training"]["checkpoint_token_fractions"]
    ]
    processed = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()
    for target_tokens in checkpoint_tokens:
        path = checkpoint_dir / f"encoder_tokens_{target_tokens:06d}.pt"
        if path.exists():
            payload = load_checked_checkpoint(
                path,
                model,
                optimizer=optimizer,
                config_hash=config_hash,
                code_hash=code_hash,
                expected_seed=seed,
                expected_processed_tokens=target_tokens,
                map_location=device,
                restore_rng=True,
            )
            processed = int(payload["processed_nonpadding_tokens"])
            history = list(payload.get("history", []))
            validation_loss = float(history[-1]["checkpoint_validation_loss"])
            checkpoints.append(
                {
                    "processed_tokens": processed,
                    "validation_loss": validation_loss,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
            continue
        segment_budget = target_tokens - processed
        schedule, lengths = exact_token_schedule(
            train_examples,
            tokenizer,
            segment_budget,
            seed + 1_000_000 + processed * 19,
        )
        batches = pack_dynamic_batches(
            schedule,
            lengths,
            int(config["training"]["target_nonpadding_tokens_per_update"]),
        )
        model.train()
        for batch_indices in batches:
            batch = [train_examples[index] for index in batch_indices]
            tokens, mask, labels, batch_tokens = collate_examples(
                batch, tokenizer, device
            )
            optimizer.zero_grad(set_to_none=True)
            with training_autocast(device):
                loss = F.cross_entropy(model(tokens, mask), labels)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            processed += batch_tokens
            learning_rate = learning_rate_at_tokens(processed, total_budget, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            history.append(
                {
                    "processed_nonpadding_tokens": processed,
                    "batch_nonpadding_tokens": batch_tokens,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
                    "learning_rate": learning_rate,
                }
            )
        if processed != target_tokens:
            raise AssertionError("encoder token accounting failed")
        validation_loss = encoder_validation_loss(
            model, calibration_examples, tokenizer, device
        )
        history[-1]["checkpoint_validation_loss"] = validation_loss
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                processed_tokens=processed,
                config_hash=config_hash,
                code_hash=code_hash,
                seed=seed,
                history=history,
            ),
            path,
        )
        checkpoints.append(
            {
                "processed_tokens": processed,
                "validation_loss": validation_loss,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    selected = min(
        checkpoints,
        key=lambda item: (item["validation_loss"], item["processed_tokens"]),
    )
    load_checked_checkpoint(
        Path(selected["path"]),
        model,
        config_hash=config_hash,
        code_hash=code_hash,
        expected_seed=seed,
        expected_processed_tokens=int(selected["processed_tokens"]),
        map_location=device,
    )
    return {
        "processed_nonpadding_tokens": processed,
        "wall_time_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoints": checkpoints,
        "selected_checkpoint": selected,
        "selected_rule": config["encoder_control"]["checkpoint_selection"],
    }


@dataclass
class SelectorCorpus:
    critic_features: Tensor
    confidence_features: Tensor
    correctness: Tensor
    problem_indices: Tensor
    horizons: Tensor
    on_policy: Tensor
    source_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.correctness.shape[0])

    def audit(self) -> dict[str, Any]:
        return {
            "states": len(self),
            "correct_states": int(self.correctness.sum().item()),
            "incorrect_states": int((1 - self.correctness).sum().item()),
            "on_policy_states": int(self.on_policy.sum().item()),
            "checkpoint_replay_states": int((~self.on_policy.bool()).sum().item()),
            "unique_problems": int(self.problem_indices.unique().numel()),
            "horizon_counts": {
                str(horizon): int((self.horizons == horizon).sum().item())
                for horizon in sorted(self.horizons.unique().tolist())
            },
            "source_counts": dict(sorted(Counter(self.source_names).items())),
        }


def _selector_source_assignments(
    example_count: int,
    horizons: Sequence[int],
    checkpoints: Sequence[Mapping[str, Any]],
    rollouts: Sequence[str],
) -> dict[tuple[int, str], list[tuple[int, int]]]:
    combinations = [
        (example_index, horizon)
        for example_index in range(example_count)
        for horizon in horizons
    ]
    assignments: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for checkpoint in checkpoints:
        for rollout in rollouts:
            assignments[(int(checkpoint["processed_tokens"]), rollout)] = list(
                combinations
            )
    return assignments


@torch.no_grad()
def _harvest_assignment_source(
    model: CommonRecurrentModel,
    critic: LatentProgressCritic,
    examples: Sequence[DeductionExample],
    assignments: Sequence[tuple[int, int]],
    tokenizer: LocalTokenizer,
    device: torch.device,
    *,
    rollout: str,
    rollout_seed: int,
    batch_size: int = 24,
) -> dict[str, Tensor]:
    if rollout == "greedy":
        model.eval()
    else:
        model.train()
        torch.manual_seed(rollout_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(rollout_seed)
    by_problem: dict[int, list[int]] = defaultdict(list)
    for problem_index, horizon in assignments:
        by_problem[problem_index].append(horizon)
    problem_indices = sorted(by_problem)
    critic_blocks: list[Tensor] = []
    confidence_blocks: list[Tensor] = []
    correctness_blocks: list[Tensor] = []
    problem_blocks: list[Tensor] = []
    horizon_blocks: list[Tensor] = []
    for batch_indices in iter_index_batches(problem_indices, batch_size):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = collate_examples(batch, tokenizer, device)
        context, prompt = model.encode_context(tokens, mask)
        max_horizon = max(max(by_problem[index]) for index in batch_indices)
        states = model.recurrent_states(context, prompt, mask, max_horizon)
        critic_feature_grid = critic.features_for_trajectory(states, prompt)
        logits_grid = torch.stack(
            [model.logits_from_state(state) for state in states], dim=1
        )
        selected_rows: list[int] = []
        selected_horizons: list[int] = []
        selected_problems: list[int] = []
        for row_index, problem_index in enumerate(batch_indices):
            for horizon in sorted(by_problem[problem_index]):
                selected_rows.append(row_index)
                selected_horizons.append(horizon - 1)
                selected_problems.append(problem_index)
        row_tensor = torch.tensor(selected_rows, device=device)
        horizon_tensor = torch.tensor(selected_horizons, device=device)
        selected_logits = logits_grid[row_tensor, horizon_tensor]
        selected_labels = labels[row_tensor]
        selected_confidence = torch.stack(
            [
                confidence_features(selected_logits[index : index + 1], horizon + 1)[0]
                for index, horizon in enumerate(selected_horizons)
            ]
        )
        critic_blocks.append(
            critic_feature_grid[row_tensor, horizon_tensor].cpu().to(torch.float16)
        )
        confidence_blocks.append(selected_confidence.cpu())
        correctness_blocks.append(
            (selected_logits.argmax(-1) == selected_labels).cpu().to(torch.float32)
        )
        problem_blocks.append(torch.tensor(selected_problems, dtype=torch.long))
        horizon_blocks.append(
            torch.tensor(
                [horizon + 1 for horizon in selected_horizons], dtype=torch.long
            )
        )
    return {
        "critic_features": torch.cat(critic_blocks),
        "confidence_features": torch.cat(confidence_blocks),
        "correctness": torch.cat(correctness_blocks),
        "problem_indices": torch.cat(problem_blocks),
        "horizons": torch.cat(horizon_blocks),
    }


def harvest_selector_training_corpus(
    model: CommonRecurrentModel,
    critic: LatentProgressCritic,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    config_hash: str,
    code_hash: str,
    seed: int,
) -> SelectorCorpus:
    horizons = tuple(int(value) for value in config["selectors"]["fit_horizons"])
    rollouts = tuple(str(value) for value in config["selectors"]["harvest_rollouts"])
    assignments = _selector_source_assignments(
        len(examples), horizons, checkpoints, rollouts
    )
    checkpoint_by_tokens = {
        int(record["processed_tokens"]): checkpoint_dir / Path(str(record["path"])).name
        for record in checkpoints
    }
    source_blocks: list[dict[str, Tensor]] = []
    source_names: list[str] = []
    on_policy_blocks: list[Tensor] = []
    final_tokens = int(checkpoints[-1]["processed_tokens"])
    for source_index, ((tokens, rollout), source_assignments) in enumerate(
        sorted(assignments.items())
    ):
        load_checked_checkpoint(
            checkpoint_by_tokens[tokens],
            model,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=tokens,
            map_location=device,
        )
        block = _harvest_assignment_source(
            model,
            critic,
            examples,
            source_assignments,
            tokenizer,
            device,
            rollout=rollout,
            rollout_seed=seed
            + source_index * int(config["selectors"]["rollout_seed_stride"]),
        )
        source_name = f"tokens_{tokens}:{rollout}"
        on_policy = tokens == final_tokens
        source_blocks.append(block)
        source_names.extend(source_name for _ in range(len(block["correctness"])))
        on_policy_blocks.append(
            torch.full((len(block["correctness"]),), on_policy, dtype=torch.bool)
        )
    expected_states = len(examples) * len(horizons) * len(checkpoints) * len(rollouts)
    actual_states = sum(len(block["correctness"]) for block in source_blocks)
    if actual_states != expected_states:
        raise AssertionError("selector harvest lost assigned states")
    return SelectorCorpus(
        critic_features=torch.cat(
            [block["critic_features"] for block in source_blocks]
        ),
        confidence_features=torch.cat(
            [block["confidence_features"] for block in source_blocks]
        ),
        correctness=torch.cat([block["correctness"] for block in source_blocks]),
        problem_indices=torch.cat(
            [block["problem_indices"] for block in source_blocks]
        ),
        horizons=torch.cat([block["horizons"] for block in source_blocks]),
        on_policy=torch.cat(on_policy_blocks),
        source_names=tuple(source_names),
    )


def harvest_fixed_selector_corpus(
    model: CommonRecurrentModel,
    critic: LatentProgressCritic,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    config: Mapping[str, Any],
    device: torch.device,
) -> SelectorCorpus:
    horizons = tuple(int(value) for value in config["selectors"]["fit_horizons"])
    assignments = [
        (problem_index, horizon)
        for problem_index in range(len(examples))
        for horizon in horizons
    ]
    block = _harvest_assignment_source(
        model,
        critic,
        examples,
        assignments,
        tokenizer,
        device,
        rollout="greedy",
        rollout_seed=0,
    )
    return SelectorCorpus(
        critic_features=block["critic_features"],
        confidence_features=block["confidence_features"],
        correctness=block["correctness"],
        problem_indices=block["problem_indices"],
        horizons=block["horizons"],
        on_policy=torch.ones(len(block["correctness"]), dtype=torch.bool),
        source_names=tuple("final:greedy" for _ in range(len(block["correctness"]))),
    )


def fit_confidence_calibrator(
    calibrator: ConfidenceCalibrator,
    corpus: SelectorCorpus,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    seed: int,
    corpus_split: str,
) -> dict[str, Any]:
    if corpus_split != "selector_calibration":
        raise ValueError("confidence calibrator must fit on selector_calibration")
    torch.manual_seed(seed + 30_000_000)
    calibrator.to(device)
    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=float(config["selectors"]["confidence_learning_rate"]),
    )
    steps = int(config["selectors"]["confidence_optimizer_steps"])
    batch_size = min(1024, len(corpus))
    generator = torch.Generator().manual_seed(seed + 31_000_000)
    history: list[dict[str, float | int]] = []
    calibrator.train()
    for step in range(steps):
        indices = torch.randint(len(corpus), (batch_size,), generator=generator)
        features = corpus.confidence_features[indices].to(device)
        labels = corpus.correctness[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = calibrator(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    calibrator.eval()
    for parameter in calibrator.parameters():
        parameter.requires_grad_(False)
    return {
        "optimizer_steps": steps,
        "history": history,
        "fit_corpus_split": corpus_split,
        "frozen_before_external_scoring": True,
    }


@torch.no_grad()
def calibrator_scores(
    calibrator: ConfidenceCalibrator,
    corpus: SelectorCorpus,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> Tensor:
    if calibrator.training or any(
        parameter.requires_grad for parameter in calibrator.parameters()
    ):
        raise RuntimeError("confidence calibrator must be frozen before scoring")
    scores: list[Tensor] = []
    for start in range(0, len(corpus), batch_size):
        features = corpus.confidence_features[start : start + batch_size].to(device)
        scores.append(torch.sigmoid(calibrator(features)).cpu())
    return torch.cat(scores)


def binary_auroc(scores: Tensor, labels: Tensor) -> dict[str, float | int | None]:
    scores = scores.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.int64)
    positives = int(labels.sum().item())
    negatives = int((labels == 0).sum().item())
    denominator = positives * negatives
    if denominator == 0:
        return {
            "concordant_pair_units": None,
            "positive_negative_pairs": 0,
            "positive_states": positives,
            "negative_states": negatives,
            "value": None,
        }
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    negative_before = 0
    concordant = 0.0
    index = 0
    while index < len(sorted_scores):
        end = index + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group = sorted_labels[index:end]
        group_positive = int(group.sum().item())
        group_negative = int((group == 0).sum().item())
        concordant += group_positive * negative_before
        concordant += 0.5 * group_positive * group_negative
        negative_before += group_negative
        index = end
    return {
        "concordant_pair_units": concordant,
        "positive_negative_pairs": denominator,
        "positive_states": positives,
        "negative_states": negatives,
        "value": concordant / denominator,
    }


def construct_within_problem_pairs(
    corpus: SelectorCorpus,
    confidence_scores: Tensor,
    config: Mapping[str, Any],
    *,
    matched_only: bool,
    maximum: int | None,
) -> list[tuple[int, int]]:
    by_problem: dict[int, list[int]] = defaultdict(list)
    for index, problem in enumerate(corpus.problem_indices.tolist()):
        by_problem[int(problem)].append(index)
    tolerance = float(config["selectors"]["confidence_match_absolute_difference_max"])
    deciles = int(config["selectors"]["confidence_match_deciles"])
    candidates: list[tuple[int, int, int]] = []
    bounded_heap: list[tuple[int, int, int]] = []
    for problem, indices in by_problem.items():
        positive = [index for index in indices if corpus.correctness[index] == 1]
        negative = [index for index in indices if corpus.correctness[index] == 0]
        for positive_index in positive:
            for negative_index in negative:
                if matched_only:
                    positive_score = float(confidence_scores[positive_index])
                    negative_score = float(confidence_scores[negative_index])
                    positive_decile = min(deciles - 1, int(positive_score * deciles))
                    negative_decile = min(deciles - 1, int(negative_score * deciles))
                    if positive_decile != negative_decile:
                        continue
                    if abs(positive_score - negative_score) > tolerance:
                        continue
                key = int(
                    sha256_bytes(
                        (
                            f"{config['selectors']['matched_pair_subsample_seed']}:"
                            f"{problem}:{int(corpus.horizons[positive_index])}:"
                            f"{int(corpus.horizons[negative_index])}"
                        ).encode("ascii")
                    ),
                    16,
                )
                if maximum is None:
                    candidates.append((key, positive_index, negative_index))
                elif len(bounded_heap) < maximum:
                    heapq.heappush(bounded_heap, (-key, positive_index, negative_index))
                elif key < -bounded_heap[0][0]:
                    heapq.heapreplace(
                        bounded_heap, (-key, positive_index, negative_index)
                    )
    if maximum is not None:
        candidates = [
            (-negative_key, positive, negative)
            for negative_key, positive, negative in bounded_heap
        ]
    candidates.sort()
    return [(positive, negative) for _, positive, negative in candidates]


def fit_latent_critic(
    critic: LatentProgressCritic,
    training_corpus: SelectorCorpus,
    confidence_scores: Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    cfg = config["selectors"]
    torch.manual_seed(seed + 40_000_000)
    critic.to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(cfg["critic_learning_rate"]),
        weight_decay=float(cfg["critic_weight_decay"]),
    )
    matched_pairs = construct_within_problem_pairs(
        training_corpus,
        confidence_scores,
        config,
        matched_only=True,
        maximum=int(cfg["critic_training_pairs_max"]),
    )
    all_pairs = construct_within_problem_pairs(
        training_corpus,
        confidence_scores,
        config,
        matched_only=False,
        maximum=int(cfg["critic_training_pairs_max"]),
    )
    if not matched_pairs or not all_pairs:
        return {
            "fit_performed": False,
            "reason": "INSUFFICIENT_OPPOSITE_CORRECTNESS_TRAINING_PAIRS",
            "epochs": 0,
            "history": [],
            "available_confidence_matched_training_pairs": len(matched_pairs),
            "available_all_opposite_correctness_training_pairs": len(all_pairs),
        }
    hard_pairs = [
        pair
        for pair in all_pairs
        if float(confidence_scores[pair[1]]) >= float(confidence_scores[pair[0]])
    ]
    on_policy_indices = torch.where(training_corpus.on_policy)[0].tolist()
    replay_indices = torch.where(~training_corpus.on_policy)[0].tolist()
    batch_size = int(cfg["critic_batch_size"])
    epochs = int(cfg["critic_epochs"])
    rng = random.Random(seed + 41_000_000)
    history: list[dict[str, Any]] = []
    critic.train()
    for epoch in range(epochs):
        rng.shuffle(on_policy_indices)
        rng.shuffle(replay_indices)
        state_batches = max(1, math.ceil(len(training_corpus) / batch_size))
        epoch_matched_pairs = 0
        epoch_total_pairs = 0
        for batch_index in range(state_batches):
            on_policy_count = round(batch_size * float(cfg["on_policy_fraction"]))
            on_indices = [
                on_policy_indices[
                    (batch_index * on_policy_count + offset) % len(on_policy_indices)
                ]
                for offset in range(on_policy_count)
            ]
            replay = [
                replay_indices[
                    (batch_index * (batch_size - on_policy_count) + offset)
                    % len(replay_indices)
                ]
                for offset in range(batch_size - on_policy_count)
            ]
            state_indices = torch.tensor(on_indices + replay, dtype=torch.long)
            features = training_corpus.critic_features[state_indices].to(
                device=device, dtype=torch.float32
            )
            labels = training_corpus.correctness[state_indices].to(device)
            pair_count = max(2, batch_size // 4)
            matched_count = math.ceil(
                pair_count * float(cfg["confidence_matched_pair_fraction_min"])
            )
            sampled_pairs = [rng.choice(matched_pairs) for _ in range(matched_count)]
            unmatched_pool = hard_pairs or all_pairs
            unmatched_count = pair_count - matched_count
            hard_count = round(
                unmatched_count * float(cfg["hard_contrast_pair_fraction_of_unmatched"])
            )
            sampled_pairs.extend(rng.choice(unmatched_pool) for _ in range(hard_count))
            sampled_pairs.extend(
                rng.choice(all_pairs) for _ in range(unmatched_count - hard_count)
            )
            pair_positive = torch.tensor([pair[0] for pair in sampled_pairs])
            pair_negative = torch.tensor([pair[1] for pair in sampled_pairs])
            positive_features = training_corpus.critic_features[pair_positive].to(
                device=device, dtype=torch.float32
            )
            negative_features = training_corpus.critic_features[pair_negative].to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            state_scores = critic(features)
            bce = F.binary_cross_entropy_with_logits(state_scores, labels)
            positive_scores = critic(positive_features)
            negative_scores = critic(negative_features)
            ranking = F.softplus(-(positive_scores - negative_scores)).mean()
            loss = (
                float(config["critic"]["bce_weight"]) * bce
                + float(config["critic"]["pairwise_ranking_weight"]) * ranking
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                critic.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_matched_pairs += matched_count
            epoch_total_pairs += pair_count
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "bce_loss": float(bce.detach().cpu()),
                "ranking_loss": float(ranking.detach().cpu()),
                "matched_pair_minibatch_numerator": epoch_matched_pairs,
                "ranking_pair_minibatch_denominator": epoch_total_pairs,
                "matched_pair_fraction": epoch_matched_pairs / epoch_total_pairs,
                "on_policy_state_numerator_per_batch": on_policy_count,
                "state_batch_denominator": batch_size,
            }
        )
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return {
        "fit_performed": True,
        "reason": None,
        "epochs": epochs,
        "history": history,
        "available_confidence_matched_training_pairs": len(matched_pairs),
        "available_all_opposite_correctness_training_pairs": len(all_pairs),
        "available_hard_contrast_training_pairs": len(hard_pairs),
    }


@torch.no_grad()
def critic_scores(
    critic: LatentProgressCritic,
    corpus: SelectorCorpus,
    device: torch.device,
    *,
    batch_size: int = 2048,
) -> Tensor:
    scores: list[Tensor] = []
    for start in range(0, len(corpus), batch_size):
        features = corpus.critic_features[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        scores.append(torch.sigmoid(critic(features)).cpu())
    return torch.cat(scores)


def selection_accuracy_from_corpus(
    corpus: SelectorCorpus, scores: Tensor
) -> dict[str, int | float]:
    correct = 0
    problems = sorted(set(int(value) for value in corpus.problem_indices.tolist()))
    for problem in problems:
        indices = torch.where(corpus.problem_indices == problem)[0]
        selected = int(indices[torch.argmax(scores[indices])])
        correct += int(corpus.correctness[selected].item())
    return {
        "correct": correct,
        "total": len(problems),
        "value": correct / len(problems),
    }


def confidence_matched_concordance(
    corpus: SelectorCorpus,
    confidence: Tensor,
    critic: Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = construct_within_problem_pairs(
        corpus,
        confidence,
        config,
        matched_only=True,
        maximum=int(config["selectors"]["matched_pair_max_per_seed"]),
    )
    concordant = 0.0
    for positive, negative in pairs:
        positive_score = float(critic[positive])
        negative_score = float(critic[negative])
        if positive_score > negative_score:
            concordant += 1.0
        elif positive_score == negative_score:
            concordant += 0.5
    return {
        "concordant_pair_units": concordant,
        "qualifying_pairs": len(pairs),
        "value": None if not pairs else concordant / len(pairs),
        "pair_indices": pairs,
    }


def selector_provenance_metrics(
    corpus: SelectorCorpus,
    confidence: Tensor,
    critic: Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    critic_auc = binary_auroc(critic, corpus.correctness)
    confidence_auc = binary_auroc(confidence, corpus.correctness)
    matched = confidence_matched_concordance(corpus, confidence, critic, config)
    critic_selection = selection_accuracy_from_corpus(corpus, critic)
    confidence_selection = selection_accuracy_from_corpus(corpus, confidence)
    critic_auc_value = critic_auc["value"]
    confidence_auc_value = confidence_auc["value"]
    advantage = (
        None
        if critic_auc_value is None or confidence_auc_value is None
        else float(critic_auc_value) - float(confidence_auc_value)
    )
    selection_advantage = float(critic_selection["value"]) - float(
        confidence_selection["value"]
    )
    thresholds = config["selectors"]["provenance_gates"]
    gates = {
        "overall_critic_auroc": {
            "threshold": thresholds["minimum_overall_critic_auroc"],
            "observed": critic_auc_value,
            "pass": critic_auc_value is not None
            and float(critic_auc_value)
            >= float(thresholds["minimum_overall_critic_auroc"]),
        },
        "confidence_matched_concordance": {
            "threshold": thresholds["minimum_confidence_matched_concordance"],
            "observed": matched["value"],
            "numerator": matched["concordant_pair_units"],
            "denominator": matched["qualifying_pairs"],
            "pass": matched["value"] is not None
            and float(matched["value"])
            >= float(thresholds["minimum_confidence_matched_concordance"]),
        },
        "critic_auroc_advantage_over_confidence": {
            "threshold": thresholds["minimum_critic_auroc_advantage_over_confidence"],
            "observed": advantage,
            "pass": advantage is not None
            and advantage
            >= float(thresholds["minimum_critic_auroc_advantage_over_confidence"]),
        },
        "critic_selection_accuracy_advantage": {
            "threshold": thresholds["minimum_critic_selection_accuracy_advantage"],
            "observed": selection_advantage,
            "critic_numerator": critic_selection["correct"],
            "confidence_numerator": confidence_selection["correct"],
            "denominator": critic_selection["total"],
            "pass": selection_advantage
            >= float(thresholds["minimum_critic_selection_accuracy_advantage"]),
        },
        "on_policy_critic_auroc": {
            "threshold": thresholds["minimum_on_policy_critic_auroc"],
            "observed": critic_auc_value,
            "pass": critic_auc_value is not None
            and float(critic_auc_value)
            >= float(thresholds["minimum_on_policy_critic_auroc"]),
        },
    }
    return {
        "critic_auroc": critic_auc,
        "confidence_auroc": confidence_auc,
        "critic_minus_confidence_auroc": advantage,
        "critic_selection_accuracy": critic_selection,
        "confidence_selection_accuracy": confidence_selection,
        "critic_minus_confidence_selection_accuracy": selection_advantage,
        "confidence_matched_concordance": {
            key: value for key, value in matched.items() if key != "pair_indices"
        },
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates.values()),
    }


@dataclass
class TrajectoryEvaluation:
    example_metadata: tuple[dict[str, Any], ...]
    gold: Tensor
    no_latch_predictions: Tensor
    confidence_latch_predictions: Tensor
    critic_latch_predictions: Tensor
    hysteretic_critic_latch_predictions: Tensor
    confidence_selected_horizons: Tensor
    critic_selected_horizons: Tensor
    hysteretic_selected_horizons: Tensor
    maximum_answer_probabilities: Tensor
    confidence_scores: Tensor
    critic_scores: Tensor

    @property
    def example_count(self) -> int:
        return int(self.gold.shape[0])

    @property
    def horizon_count(self) -> int:
        return int(self.no_latch_predictions.shape[1])


def cumulative_first_argmax(scores: Tensor) -> Tensor:
    """Cumulative argmax with ties frozen toward the smaller horizon."""

    best_scores = scores[:, 0]
    best_indices = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    outputs = [best_indices.clone()]
    for index in range(1, scores.shape[1]):
        improves = scores[:, index] > best_scores
        best_scores = torch.where(improves, scores[:, index], best_scores)
        best_indices = torch.where(
            improves, torch.full_like(best_indices, index), best_indices
        )
        outputs.append(best_indices.clone())
    return torch.stack(outputs, dim=1)


def hysteretic_incumbent_indices(scores: Tensor, delta: float) -> Tensor:
    """Sequential incumbent replacement with strict delta-threshold crossing."""

    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("hysteretic scores must have shape [examples, horizons>=1]")
    if delta < 0.0:
        raise ValueError("hysteresis delta must be nonnegative")
    incumbent = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    outputs = [incumbent.clone()]
    for challenger in range(1, scores.shape[1]):
        incumbent_scores = scores.gather(1, incumbent.unsqueeze(1)).squeeze(1)
        replace = scores[:, challenger] - incumbent_scores > delta
        incumbent = torch.where(
            replace, torch.full_like(incumbent, challenger), incumbent
        )
        outputs.append(incumbent.clone())
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def evaluate_trajectories(
    model: CommonRecurrentModel,
    calibrator: ConfidenceCalibrator,
    critic: LatentProgressCritic,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    device: torch.device,
    *,
    max_horizon: int,
    hysteresis_delta: float = 0.0,
    batch_size: int = 16,
) -> TrajectoryEvaluation:
    model.eval()
    calibrator.eval()
    critic.eval()
    gold_rows: list[Tensor] = []
    prediction_rows: list[Tensor] = []
    confidence_prediction_rows: list[Tensor] = []
    critic_prediction_rows: list[Tensor] = []
    hysteretic_prediction_rows: list[Tensor] = []
    confidence_horizon_rows: list[Tensor] = []
    critic_horizon_rows: list[Tensor] = []
    hysteretic_horizon_rows: list[Tensor] = []
    maximum_probability_rows: list[Tensor] = []
    confidence_score_rows: list[Tensor] = []
    critic_score_rows: list[Tensor] = []
    for batch_indices in iter_index_batches(list(range(len(examples))), batch_size):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = collate_examples(batch, tokenizer, device)
        context, prompt = model.encode_context(tokens, mask)
        states = model.recurrent_states(context, prompt, mask, max_horizon)
        logits = torch.stack(
            [model.logits_from_state(state) for state in states], dim=1
        )
        predictions = logits.argmax(-1)
        probability = logits.softmax(-1)
        maximum_probability = probability.max(-1).values
        confidence_feature_grid = torch.stack(
            [
                confidence_features(logits[:, index], index + 1)
                for index in range(max_horizon)
            ],
            dim=1,
        )
        confidence_score_grid = torch.sigmoid(calibrator(confidence_feature_grid))
        critic_feature_grid = critic.features_for_trajectory(states, prompt)
        critic_score_grid = torch.sigmoid(critic(critic_feature_grid))
        confidence_selected = cumulative_first_argmax(confidence_score_grid)
        critic_selected = cumulative_first_argmax(critic_score_grid)
        hysteretic_selected = hysteretic_incumbent_indices(
            critic_score_grid, hysteresis_delta
        )
        confidence_predictions = torch.gather(predictions, 1, confidence_selected)
        critic_predictions = torch.gather(predictions, 1, critic_selected)
        hysteretic_predictions = torch.gather(predictions, 1, hysteretic_selected)
        gold_rows.append(labels.cpu())
        prediction_rows.append(predictions.cpu())
        confidence_prediction_rows.append(confidence_predictions.cpu())
        critic_prediction_rows.append(critic_predictions.cpu())
        hysteretic_prediction_rows.append(hysteretic_predictions.cpu())
        confidence_horizon_rows.append((confidence_selected + 1).cpu())
        critic_horizon_rows.append((critic_selected + 1).cpu())
        hysteretic_horizon_rows.append((hysteretic_selected + 1).cpu())
        maximum_probability_rows.append(maximum_probability.cpu())
        confidence_score_rows.append(confidence_score_grid.cpu())
        critic_score_rows.append(critic_score_grid.cpu())
    return TrajectoryEvaluation(
        example_metadata=tuple(example.result_metadata() for example in examples),
        gold=torch.cat(gold_rows),
        no_latch_predictions=torch.cat(prediction_rows),
        confidence_latch_predictions=torch.cat(confidence_prediction_rows),
        critic_latch_predictions=torch.cat(critic_prediction_rows),
        hysteretic_critic_latch_predictions=torch.cat(hysteretic_prediction_rows),
        confidence_selected_horizons=torch.cat(confidence_horizon_rows),
        critic_selected_horizons=torch.cat(critic_horizon_rows),
        hysteretic_selected_horizons=torch.cat(hysteretic_horizon_rows),
        maximum_answer_probabilities=torch.cat(maximum_probability_rows),
        confidence_scores=torch.cat(confidence_score_rows),
        critic_scores=torch.cat(critic_score_rows),
    )


@torch.no_grad()
def evaluate_encoder_control(
    model: EncoderOnlyControl,
    examples: Sequence[DeductionExample],
    tokenizer: LocalTokenizer,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> dict[str, int | float]:
    model.eval()
    correct = 0
    for batch_indices in iter_index_batches(list(range(len(examples))), batch_size):
        batch = [examples[index] for index in batch_indices]
        tokens, mask, labels, _ = collate_examples(batch, tokenizer, device)
        correct += int((model(tokens, mask).argmax(-1) == labels).sum().item())
    return {
        "correct": correct,
        "total": len(examples),
        "value": correct / len(examples),
    }


def accuracy_count(predictions: Tensor, gold: Tensor) -> dict[str, int | float]:
    correct = int((predictions == gold).sum().item())
    total = int(gold.numel())
    return {"correct": correct, "total": total, "value": correct / total}


def accuracy_grid(evaluation: TrajectoryEvaluation) -> dict[str, Any]:
    output: dict[str, Any] = {}
    arms = {
        "no_latch": evaluation.no_latch_predictions,
        "confidence_plus_schedule_latch": evaluation.confidence_latch_predictions,
        "latent_critic_latch": evaluation.critic_latch_predictions,
        "hysteretic_critic_latch_informational": (
            evaluation.hysteretic_critic_latch_predictions
        ),
    }
    for arm, predictions in arms.items():
        output[arm] = {
            str(horizon + 1): accuracy_count(predictions[:, horizon], evaluation.gold)
            for horizon in range(evaluation.horizon_count)
        }
    return output


def stratified_accuracy_grid(evaluation: TrajectoryEvaluation) -> dict[str, Any]:
    fields = ("proof_depth", "lure_type", "template_family", "answer_position")
    output: dict[str, Any] = {}
    for field in fields:
        values = sorted(
            {metadata[field] for metadata in evaluation.example_metadata}, key=str
        )
        output[field] = {}
        for value in values:
            indices = torch.tensor(
                [
                    index
                    for index, metadata in enumerate(evaluation.example_metadata)
                    if metadata[field] == value
                ],
                dtype=torch.long,
            )
            subset = TrajectoryEvaluation(
                example_metadata=tuple(
                    evaluation.example_metadata[int(index)] for index in indices
                ),
                gold=evaluation.gold[indices],
                no_latch_predictions=evaluation.no_latch_predictions[indices],
                confidence_latch_predictions=evaluation.confidence_latch_predictions[
                    indices
                ],
                critic_latch_predictions=evaluation.critic_latch_predictions[indices],
                hysteretic_critic_latch_predictions=(
                    evaluation.hysteretic_critic_latch_predictions[indices]
                ),
                confidence_selected_horizons=evaluation.confidence_selected_horizons[
                    indices
                ],
                critic_selected_horizons=evaluation.critic_selected_horizons[indices],
                hysteretic_selected_horizons=(
                    evaluation.hysteretic_selected_horizons[indices]
                ),
                maximum_answer_probabilities=evaluation.maximum_answer_probabilities[
                    indices
                ],
                confidence_scores=evaluation.confidence_scores[indices],
                critic_scores=evaluation.critic_scores[indices],
            )
            output[field][str(value)] = accuracy_grid(subset)
    return output


def select_t_star(
    calibration_evaluations: Mapping[int, TrajectoryEvaluation],
    candidates: Sequence[int],
) -> tuple[int, dict[str, Any]]:
    rows: dict[str, Any] = {}
    best_horizon: int | None = None
    best_mean = -1.0
    for horizon in candidates:
        per_seed = {
            str(seed): accuracy_count(
                evaluation.no_latch_predictions[:, horizon - 1], evaluation.gold
            )
            for seed, evaluation in calibration_evaluations.items()
        }
        mean_accuracy = statistics.mean(
            float(value["value"]) for value in per_seed.values()
        )
        rows[str(horizon)] = {"per_seed": per_seed, "mean_seed_accuracy": mean_accuracy}
        if mean_accuracy > best_mean:
            best_horizon = horizon
            best_mean = mean_accuracy
    if best_horizon is None:
        raise AssertionError("t-star candidate set is empty")
    return best_horizon, {
        "candidate_metrics": rows,
        "selected_horizon": best_horizon,
        "selected_mean_seed_accuracy": best_mean,
        "tie_break": "smaller_horizon",
        "test_inspected_before_freeze": False,
    }


def _rate(numerator: int | float, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def selector_switch_hazards(
    state_predictions: Tensor, gold: Tensor, selected_horizons: Tensor
) -> dict[str, Any]:
    """Count offered-challenge outcomes for a sequential state-bank selector."""

    if state_predictions.shape != selected_horizons.shape:
        raise ValueError("predictions and selected horizons must have equal shape")
    examples, horizons = state_predictions.shape
    harmful_examples = torch.zeros(examples, dtype=torch.bool)
    totals = Counter()
    per_transition: dict[str, Any] = {}
    rows = torch.arange(examples)
    for challenger_index in range(1, horizons):
        previous = selected_horizons[:, challenger_index - 1] - 1
        current = selected_horizons[:, challenger_index] - 1
        if bool(((previous < 0) | (previous >= challenger_index)).any()):
            raise AssertionError("selector incumbent horizon is not causal")
        accepted = current == challenger_index
        if not torch.equal(current != previous, accepted):
            raise AssertionError("selector replacement accounting mismatch")
        incumbent_correct = state_predictions[rows, previous] == gold
        challenger_correct = state_predictions[:, challenger_index] == gold
        selected_correct = state_predictions[rows, current] == gold
        harmful_offer = incumbent_correct & ~challenger_correct
        beneficial_offer = ~incumbent_correct & challenger_correct
        accepted_harmful = accepted & harmful_offer
        rejected_beneficial = ~accepted & beneficial_offer
        correct_survival = incumbent_correct & selected_correct
        harmful_examples |= accepted_harmful
        counts = {
            "accepted_harmful": (
                int(accepted_harmful.sum().item()),
                int(harmful_offer.sum().item()),
            ),
            "rejected_beneficial": (
                int(rejected_beneficial.sum().item()),
                int(beneficial_offer.sum().item()),
            ),
            "correct_incumbent_survival": (
                int(correct_survival.sum().item()),
                int(incumbent_correct.sum().item()),
            ),
            "total_replacements": (int(accepted.sum().item()), examples),
        }
        per_transition[f"{challenger_index}_to_{challenger_index + 1}"] = {
            name: _rate(numerator, denominator)
            for name, (numerator, denominator) in counts.items()
        }
        for name, (numerator, denominator) in counts.items():
            totals[f"{name}_numerator"] += numerator
            totals[f"{name}_denominator"] += denominator
    aggregate = {
        name: _rate(totals[f"{name}_numerator"], totals[f"{name}_denominator"])
        for name in (
            "accepted_harmful",
            "rejected_beneficial",
            "correct_incumbent_survival",
            "total_replacements",
        )
    }
    aggregate["harmful_switch_examples"] = _rate(
        int(harmful_examples.sum().item()), examples
    )
    return {
        "challenge_definition": "each state at t>=2 offered against incumbent",
        "per_transition": per_transition,
        "aggregate": aggregate,
    }


def select_hysteresis_delta(
    state_predictions: Tensor,
    gold: Tensor,
    critic_score_grid: Tensor,
    t_star: int,
    config: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Freeze one seed-local delta using calibration trajectories only."""

    hcfg = config["selectors"]["hysteretic_latch"]
    selection_horizon = int(hcfg["selection_horizon"])
    if state_predictions.shape[1] < selection_horizon:
        raise ValueError("calibration trajectory is shorter than horizon 16")
    predictions = state_predictions[:, :selection_horizon]
    scores = critic_score_grid[:, :selection_horizon]
    no_t1_correct = int((predictions[:, 0] == gold).sum().item())
    no_t_star_correct = int((predictions[:, t_star - 1] == gold).sum().item())
    gain_denominator = no_t_star_correct - no_t1_correct
    candidate_metrics: dict[str, Any] = {}
    feasible: list[tuple[float, float, float]] = []
    for candidate in (float(value) for value in hcfg["delta_grid"]):
        selected = hysteretic_incumbent_indices(scores, candidate) + 1
        selected_predictions = torch.gather(predictions, 1, selected - 1)
        final_correct = int(
            (selected_predictions[:, selection_horizon - 1] == gold).sum().item()
        )
        gain_numerator = final_correct - no_t1_correct
        gain_retention = (
            None if gain_denominator == 0 else gain_numerator / gain_denominator
        )
        switch_hazards = selector_switch_hazards(predictions, gold, selected)
        harmful_rate = switch_hazards["aggregate"]["harmful_switch_examples"]["value"]
        accuracy = final_correct / int(gold.numel())
        is_feasible = gain_retention is not None and gain_retention >= float(
            hcfg["minimum_calibration_gain_retention"]
        )
        candidate_metrics[f"{candidate:.2f}"] = {
            "delta": candidate,
            "b16_accuracy": _rate(final_correct, int(gold.numel())),
            "calibration_gain_retention": {
                "arm4_minus_t1_correct_numerator": gain_numerator,
                "t_star_minus_t1_correct_denominator": gain_denominator,
                "accuracy_denominator": int(gold.numel()),
                "value": gain_retention,
            },
            "switch_hazards": switch_hazards,
            "feasible": is_feasible,
        }
        if is_feasible:
            if harmful_rate is None:
                raise AssertionError("harmful-switch-example rate lacks denominator")
            feasible.append((float(harmful_rate), -accuracy, -candidate))
    constraint_miss = not feasible
    selected_delta = (
        float(hcfg["no_feasible_fallback_delta"])
        if constraint_miss
        else -min(feasible)[2]
    )
    return selected_delta, {
        "selected_delta": selected_delta,
        "calibration_constraint_miss": constraint_miss,
        "delta_grid": [float(value) for value in hcfg["delta_grid"]],
        "selection_horizon": selection_horizon,
        "minimum_gain_retention": hcfg["minimum_calibration_gain_retention"],
        "selection_lexicographic": hcfg["selection_lexicographic"],
        "candidate_calibration_metrics": candidate_metrics,
        "test_grid_evaluated": False,
        "frozen_through_horizon": hcfg["freeze_through_horizon"],
        "informational_only": True,
    }


def trajectory_diagnostics(
    evaluations: Sequence[TrajectoryEvaluation],
) -> dict[str, Any]:
    """Compute pooled-or-single-seed headroom and transition diagnostics."""

    if not evaluations:
        raise ValueError("trajectory diagnostics require at least one evaluation")
    horizon_count = evaluations[0].horizon_count
    if any(evaluation.horizon_count != horizon_count for evaluation in evaluations):
        raise ValueError("trajectory horizon grids differ across seeds")
    gold = torch.cat([evaluation.gold for evaluation in evaluations])
    raw = torch.cat(
        [evaluation.no_latch_predictions for evaluation in evaluations], dim=0
    )
    selected_predictions = {
        "no_latch": raw,
        "confidence_plus_schedule_latch": torch.cat(
            [evaluation.confidence_latch_predictions for evaluation in evaluations]
        ),
        "latent_critic_latch": torch.cat(
            [evaluation.critic_latch_predictions for evaluation in evaluations]
        ),
        "hysteretic_critic_latch_informational": torch.cat(
            [
                evaluation.hysteretic_critic_latch_predictions
                for evaluation in evaluations
            ]
        ),
    }
    no_latch_horizons = torch.arange(1, horizon_count + 1).repeat(len(gold), 1)
    selected_horizons = {
        "no_latch": no_latch_horizons,
        "confidence_plus_schedule_latch": torch.cat(
            [evaluation.confidence_selected_horizons for evaluation in evaluations]
        ),
        "latent_critic_latch": torch.cat(
            [evaluation.critic_selected_horizons for evaluation in evaluations]
        ),
        "hysteretic_critic_latch_informational": torch.cat(
            [evaluation.hysteretic_selected_horizons for evaluation in evaluations]
        ),
    }
    ever_correct = (raw == gold.unsqueeze(1)).cummax(dim=1).values
    budgets: dict[str, Any] = {}
    for index in range(horizon_count):
        oracle_numerator = int(ever_correct[:, index].sum().item())
        arms: dict[str, Any] = {}
        for arm, predictions in selected_predictions.items():
            final_numerator = int((predictions[:, index] == gold).sum().item())
            arms[arm] = {
                "selected_correct": _rate(final_numerator, len(gold)),
                "oracle_headroom": _rate(oracle_numerator - final_numerator, len(gold)),
            }
        budgets[str(index + 1)] = {
            "oracle_reachable_correct": _rate(oracle_numerator, len(gold)),
            "arms": arms,
        }
    raw_transitions: dict[str, Any] = {}
    for index in range(horizon_count - 1):
        current_correct = raw[:, index] == gold
        next_correct = raw[:, index + 1] == gold
        raw_transitions[f"{index + 1}_to_{index + 2}"] = {
            "correct_to_wrong": _rate(
                int((current_correct & ~next_correct).sum().item()),
                int(current_correct.sum().item()),
            ),
            "wrong_to_correct": _rate(
                int((~current_correct & next_correct).sum().item()),
                int((~current_correct).sum().item()),
            ),
        }
    return {
        "examples": len(gold),
        "budgets_b1_b32": budgets,
        "raw_transition_hazards": raw_transitions,
        "selector_switch_hazards": {
            arm: selector_switch_hazards(raw, gold, horizons)
            for arm, horizons in selected_horizons.items()
        },
        "model_empty_transition_hazards": {
            "applicable": False,
            "reason": "four_way_typed_choice_interface_has_no_empty_output",
        },
    }


def trajectory_accounting_assertions(
    diagnostics_by_scope: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[int, TrajectoryEvaluation],
    t_star: int,
    endpoints: Sequence[int],
) -> dict[str, Any]:
    violations: list[str] = []
    budget_checks = 0
    for scope, diagnostics in diagnostics_by_scope.items():
        for budget, row in diagnostics["budgets_b1_b32"].items():
            oracle = int(row["oracle_reachable_correct"]["numerator"])
            for arm, arm_row in row["arms"].items():
                selected = int(arm_row["selected_correct"]["numerator"])
                headroom = int(arm_row["oracle_headroom"]["numerator"])
                budget_checks += 1
                if selected > oracle or headroom < 0 or headroom != oracle - selected:
                    violations.append(f"{scope}:B{budget}:{arm}:F/O/H")
    endpoint_checks: dict[str, Any] = {}
    scope_evaluations = {
        **{str(seed): [evaluation] for seed, evaluation in evaluations.items()},
        "pooled": list(evaluations.values()),
    }
    for scope, scope_rows in scope_evaluations.items():
        gold = torch.cat([row.gold for row in scope_rows])
        raw = torch.cat([row.no_latch_predictions for row in scope_rows])
        endpoint_checks[scope] = {}
        for endpoint in endpoints:
            oracle = int(
                ((raw[:, :endpoint] == gold.unsqueeze(1)).any(dim=1)).sum().item()
            )
            final = int((raw[:, endpoint - 1] == gold).sum().item())
            earlier = raw[:, t_star - 1] == gold
            regression = int((earlier & (raw[:, endpoint - 1] != gold)).sum().item())
            headroom = oracle - final
            passed = headroom >= regression
            endpoint_checks[scope][str(endpoint)] = {
                "no_latch_headroom_numerator": headroom,
                "a_t_star_times_regression_numerator": regression,
                "common_accuracy_denominator": len(gold),
                "comparison": ">=",
                "pass": passed,
            }
            if not passed:
                violations.append(f"{scope}:H{endpoint}:headroom_regression_identity")
    return {
        "f_le_oracle_and_nonnegative_headroom_checks": budget_checks,
        "no_latch_headroom_regression_checks": endpoint_checks,
        "violation_count": len(violations),
        "violations": violations,
        "all_pass": not violations,
        "failure_semantics": "evaluator_bug_VOID_integrity_not_scientific_floor",
        "minimum_headroom_validity_floor": None,
    }


def evaluation_matched_concordance(
    evaluation: TrajectoryEvaluation,
    config: Mapping[str, Any],
    endpoint: int,
) -> dict[str, Any]:
    problem_indices = torch.arange(evaluation.example_count).repeat_interleave(endpoint)
    horizons = torch.arange(1, endpoint + 1).repeat(evaluation.example_count)
    gold = evaluation.gold.repeat_interleave(endpoint)
    predictions = evaluation.no_latch_predictions[:, :endpoint].reshape(-1)
    correctness = (predictions == gold).to(torch.float32)
    confidence = evaluation.confidence_scores[:, :endpoint].reshape(-1)
    critic = evaluation.critic_scores[:, :endpoint].reshape(-1)
    dummy = SelectorCorpus(
        critic_features=torch.empty((len(correctness), 0)),
        confidence_features=torch.empty((len(correctness), 0)),
        correctness=correctness,
        problem_indices=problem_indices,
        horizons=horizons,
        on_policy=torch.ones(len(correctness), dtype=torch.bool),
        source_names=tuple("test" for _ in range(len(correctness))),
    )
    matched = confidence_matched_concordance(dummy, confidence, critic, config)
    return {key: value for key, value in matched.items() if key != "pair_indices"}


def endpoint_counts(
    evaluation: TrajectoryEvaluation,
    t_star: int,
    endpoint: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gold = evaluation.gold
    no_t_star = evaluation.no_latch_predictions[:, t_star - 1]
    no_one = evaluation.no_latch_predictions[:, 0]
    no_endpoint = evaluation.no_latch_predictions[:, endpoint - 1]
    confidence_endpoint = evaluation.confidence_latch_predictions[:, endpoint - 1]
    critic_endpoint = evaluation.critic_latch_predictions[:, endpoint - 1]
    hysteretic_endpoint = evaluation.hysteretic_critic_latch_predictions[
        :, endpoint - 1
    ]
    earlier_correct = no_t_star == gold
    denominator = int(earlier_correct.sum().item())
    no_regressions = int((earlier_correct & (no_endpoint != gold)).sum().item())
    critic_regressions = int((earlier_correct & (critic_endpoint != gold)).sum().item())
    hysteretic_regressions = int(
        (earlier_correct & (hysteretic_endpoint != gold)).sum().item()
    )
    no_one_correct = int((no_one == gold).sum().item())
    no_t_star_correct = int((no_t_star == gold).sum().item())
    no_endpoint_correct = int((no_endpoint == gold).sum().item())
    confidence_correct = int((confidence_endpoint == gold).sum().item())
    critic_correct = int((critic_endpoint == gold).sum().item())
    hysteretic_correct = int((hysteretic_endpoint == gold).sum().item())
    total = evaluation.example_count
    gain_denominator = no_t_star_correct - no_one_correct
    gain_numerator = critic_correct - no_one_correct
    matched = evaluation_matched_concordance(evaluation, config, endpoint)
    return {
        "examples": total,
        "earlier_correct": {
            "numerator": denominator,
            "denominator": total,
            "value": denominator / total,
        },
        "no_latch_regression": {
            "numerator": no_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else no_regressions / denominator,
        },
        "critic_latch_regression": {
            "numerator": critic_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else critic_regressions / denominator,
        },
        "hysteretic_critic_latch_regression_informational": {
            "numerator": hysteretic_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else hysteretic_regressions / denominator,
        },
        "regression_reduction": {
            "critic_regression_numerator": critic_regressions,
            "no_latch_regression_numerator": no_regressions,
            "earlier_correct_denominator": denominator,
            "value": None
            if no_regressions == 0
            else 1.0 - critic_regressions / no_regressions,
        },
        "no_latch_t1_accuracy": {
            "numerator": no_one_correct,
            "denominator": total,
            "value": no_one_correct / total,
        },
        "no_latch_t_star_accuracy": {
            "numerator": no_t_star_correct,
            "denominator": total,
            "value": no_t_star_correct / total,
        },
        "no_latch_endpoint_accuracy": {
            "numerator": no_endpoint_correct,
            "denominator": total,
            "value": no_endpoint_correct / total,
        },
        "confidence_latch_endpoint_accuracy": {
            "numerator": confidence_correct,
            "denominator": total,
            "value": confidence_correct / total,
        },
        "critic_latch_endpoint_accuracy": {
            "numerator": critic_correct,
            "denominator": total,
            "value": critic_correct / total,
        },
        "hysteretic_critic_latch_endpoint_accuracy_informational": {
            "numerator": hysteretic_correct,
            "denominator": total,
            "value": hysteretic_correct / total,
        },
        "gain_retention": {
            "critic_minus_t1_correct_numerator": gain_numerator,
            "t_star_minus_t1_correct_denominator": gain_denominator,
            "accuracy_denominator": total,
            "value": None
            if gain_denominator == 0
            else gain_numerator / gain_denominator,
        },
        "hysteretic_gain_retention_informational": {
            "arm4_minus_t1_correct_numerator": hysteretic_correct - no_one_correct,
            "t_star_minus_t1_correct_denominator": gain_denominator,
            "accuracy_denominator": total,
            "value": None
            if gain_denominator == 0
            else (hysteretic_correct - no_one_correct) / gain_denominator,
        },
        "critic_minus_confidence_accuracy": {
            "critic_correct_numerator": critic_correct,
            "confidence_correct_numerator": confidence_correct,
            "denominator": total,
            "value": (critic_correct - confidence_correct) / total,
        },
        "confidence_matched_critic_concordance": matched,
    }


def pooled_endpoint_counts(
    per_seed: Mapping[int, Mapping[str, Any]], endpoint: int
) -> dict[str, Any]:
    total = sum(int(metrics["examples"]) for metrics in per_seed.values())
    denominator = sum(
        int(metrics["earlier_correct"]["numerator"]) for metrics in per_seed.values()
    )
    no_regressions = sum(
        int(metrics["no_latch_regression"]["numerator"])
        for metrics in per_seed.values()
    )
    critic_regressions = sum(
        int(metrics["critic_latch_regression"]["numerator"])
        for metrics in per_seed.values()
    )
    hysteretic_regressions = sum(
        int(metrics["hysteretic_critic_latch_regression_informational"]["numerator"])
        for metrics in per_seed.values()
    )
    no_one = sum(
        int(metrics["no_latch_t1_accuracy"]["numerator"])
        for metrics in per_seed.values()
    )
    no_t_star = sum(
        int(metrics["no_latch_t_star_accuracy"]["numerator"])
        for metrics in per_seed.values()
    )
    no_endpoint = sum(
        int(metrics["no_latch_endpoint_accuracy"]["numerator"])
        for metrics in per_seed.values()
    )
    confidence = sum(
        int(metrics["confidence_latch_endpoint_accuracy"]["numerator"])
        for metrics in per_seed.values()
    )
    critic = sum(
        int(metrics["critic_latch_endpoint_accuracy"]["numerator"])
        for metrics in per_seed.values()
    )
    hysteretic = sum(
        int(
            metrics["hysteretic_critic_latch_endpoint_accuracy_informational"][
                "numerator"
            ]
        )
        for metrics in per_seed.values()
    )
    matched_numerator = sum(
        float(metrics["confidence_matched_critic_concordance"]["concordant_pair_units"])
        for metrics in per_seed.values()
    )
    matched_denominator = sum(
        int(metrics["confidence_matched_critic_concordance"]["qualifying_pairs"])
        for metrics in per_seed.values()
    )
    return {
        "endpoint": endpoint,
        "examples": total,
        "earlier_correct": {
            "numerator": denominator,
            "denominator": total,
            "value": denominator / total,
        },
        "no_latch_regression": {
            "numerator": no_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else no_regressions / denominator,
        },
        "critic_latch_regression": {
            "numerator": critic_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else critic_regressions / denominator,
        },
        "hysteretic_critic_latch_regression_informational": {
            "numerator": hysteretic_regressions,
            "denominator": denominator,
            "value": None if denominator == 0 else hysteretic_regressions / denominator,
        },
        "regression_reduction": {
            "critic_regression_numerator": critic_regressions,
            "no_latch_regression_numerator": no_regressions,
            "earlier_correct_denominator": denominator,
            "value": None
            if no_regressions == 0
            else 1.0 - critic_regressions / no_regressions,
        },
        "no_latch_t1_accuracy": {
            "numerator": no_one,
            "denominator": total,
            "value": no_one / total,
        },
        "no_latch_t_star_accuracy": {
            "numerator": no_t_star,
            "denominator": total,
            "value": no_t_star / total,
        },
        "no_latch_endpoint_accuracy": {
            "numerator": no_endpoint,
            "denominator": total,
            "value": no_endpoint / total,
        },
        "confidence_latch_endpoint_accuracy": {
            "numerator": confidence,
            "denominator": total,
            "value": confidence / total,
        },
        "critic_latch_endpoint_accuracy": {
            "numerator": critic,
            "denominator": total,
            "value": critic / total,
        },
        "hysteretic_critic_latch_endpoint_accuracy_informational": {
            "numerator": hysteretic,
            "denominator": total,
            "value": hysteretic / total,
        },
        "gain_retention": {
            "critic_minus_t1_correct_numerator": critic - no_one,
            "t_star_minus_t1_correct_denominator": no_t_star - no_one,
            "accuracy_denominator": total,
            "value": None
            if no_t_star == no_one
            else (critic - no_one) / (no_t_star - no_one),
        },
        "hysteretic_gain_retention_informational": {
            "arm4_minus_t1_correct_numerator": hysteretic - no_one,
            "t_star_minus_t1_correct_denominator": no_t_star - no_one,
            "accuracy_denominator": total,
            "value": None
            if no_t_star == no_one
            else (hysteretic - no_one) / (no_t_star - no_one),
        },
        "critic_minus_confidence_accuracy": {
            "critic_correct_numerator": critic,
            "confidence_correct_numerator": confidence,
            "denominator": total,
            "value": (critic - confidence) / total,
        },
        "confidence_matched_critic_concordance": {
            "concordant_pair_units": matched_numerator,
            "qualifying_pairs": matched_denominator,
            "value": None
            if matched_denominator == 0
            else matched_numerator / matched_denominator,
        },
    }


def numeric_gate_results(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    definitions = {
        "no_latch_regression_rate": (
            metrics["no_latch_regression"]["value"],
            float(evaluation["minimum_no_latch_regression_rate"]),
        ),
        "regression_reduction": (
            metrics["regression_reduction"]["value"],
            float(evaluation["minimum_regression_reduction"]),
        ),
        "gain_retention": (
            metrics["gain_retention"]["value"],
            float(evaluation["minimum_gain_retention"]),
        ),
        "critic_minus_confidence_accuracy": (
            metrics["critic_minus_confidence_accuracy"]["value"],
            float(evaluation["minimum_critic_accuracy_advantage"]),
        ),
        "confidence_matched_critic_concordance": (
            metrics["confidence_matched_critic_concordance"]["value"],
            float(evaluation["minimum_confidence_matched_concordance"]),
        ),
    }
    return {
        name: {
            "observed": observed,
            "threshold": threshold,
            "comparison": ">=",
            "pass": observed is not None and float(observed) >= threshold,
        }
        for name, (observed, threshold) in definitions.items()
    }


def adjudicate(
    endpoint_metrics: Mapping[int, Mapping[str, Mapping[str, Any]]],
    validity_gates: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    failed_validity = [
        name for name, gate in validity_gates.items() if not gate["pass"]
    ]
    if failed_validity:
        return {
            "final_token": "VOID",
            "reason": "VALIDITY_GATE_FAILURE",
            "failed_validity_gates": failed_validity,
            "selected_endpoint": None,
            "numeric_gates": None,
        }
    threshold = float(config["evaluation"]["minimum_no_latch_regression_rate"])

    def regression_floor_pass(endpoint: int) -> bool:
        scopes = endpoint_metrics[endpoint]
        return all(
            metrics["no_latch_regression"]["value"] is not None
            and float(metrics["no_latch_regression"]["value"]) >= threshold
            for metrics in scopes.values()
        )

    primary = int(config["evaluation"]["primary_endpoint"])
    conditional = int(config["evaluation"]["conditional_endpoint"])
    if regression_floor_pass(primary):
        selected = primary
    elif regression_floor_pass(conditional):
        selected = conditional
    else:
        nonregression_names = {
            "gain_retention",
            "critic_minus_confidence_accuracy",
            "confidence_matched_critic_concordance",
        }
        gates = {
            scope: numeric_gate_results(metrics, config)
            for scope, metrics in endpoint_metrics[conditional].items()
        }
        nonregression_pass = all(
            gate["pass"]
            for scope_gates in gates.values()
            for name, gate in scope_gates.items()
            if name in nonregression_names
        )
        if nonregression_pass:
            return {
                "final_token": "VOID",
                "reason": config["evaluation"]["regression_only_void_reason"],
                "failed_validity_gates": [],
                "selected_endpoint": None,
                "diagnostic_endpoint": conditional,
                "numeric_gates": gates,
            }
        return {
            "final_token": "FAIL",
            "reason": "INDEPENDENTLY_MEASURABLE_SELECTOR_CRITERION_FAILED",
            "failed_validity_gates": [],
            "selected_endpoint": None,
            "diagnostic_endpoint": conditional,
            "numeric_gates": gates,
        }
    gates = {
        scope: numeric_gate_results(metrics, config)
        for scope, metrics in endpoint_metrics[selected].items()
    }
    all_pass = all(gate["pass"] for scope in gates.values() for gate in scope.values())
    return {
        "final_token": "PROCEED" if all_pass else "FAIL",
        "reason": "ALL_PROCEED_GATES_PASS"
        if all_pass
        else "NUMERIC_PROCEED_GATE_MISSED",
        "failed_validity_gates": [],
        "selected_endpoint": selected,
        "numeric_gates": gates,
        "t32_rescue_permitted": selected == conditional,
    }


def _group_contributions(
    evaluation: TrajectoryEvaluation, t_star: int, endpoint: int
) -> tuple[Tensor, tuple[str, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(evaluation.example_metadata):
        grouped[str(metadata["counterfactual_group"])].append(index)
    rows: list[list[float]] = []
    names = (
        "examples",
        "earlier_correct",
        "no_regressions",
        "critic_regressions",
        "no_t1_correct",
        "no_t_star_correct",
        "confidence_endpoint_correct",
        "critic_endpoint_correct",
    )
    for group in sorted(grouped):
        indices = torch.tensor(grouped[group], dtype=torch.long)
        gold = evaluation.gold[indices]
        no_t_star = evaluation.no_latch_predictions[indices, t_star - 1]
        earlier = no_t_star == gold
        rows.append(
            [
                float(len(indices)),
                float(earlier.sum()),
                float(
                    (
                        earlier
                        & (
                            evaluation.no_latch_predictions[indices, endpoint - 1]
                            != gold
                        )
                    ).sum()
                ),
                float(
                    (
                        earlier
                        & (
                            evaluation.critic_latch_predictions[indices, endpoint - 1]
                            != gold
                        )
                    ).sum()
                ),
                float((evaluation.no_latch_predictions[indices, 0] == gold).sum()),
                float((no_t_star == gold).sum()),
                float(
                    (
                        evaluation.confidence_latch_predictions[indices, endpoint - 1]
                        == gold
                    ).sum()
                ),
                float(
                    (
                        evaluation.critic_latch_predictions[indices, endpoint - 1]
                        == gold
                    ).sum()
                ),
            ]
        )
    return torch.tensor(rows, dtype=torch.float64), names


def paired_group_bootstrap(
    evaluations: Sequence[TrajectoryEvaluation],
    t_star: int,
    endpoint: int,
    config: Mapping[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    contributions = torch.cat(
        [
            _group_contributions(evaluation, t_star, endpoint)[0]
            for evaluation in evaluations
        ]
    )
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    generator = torch.Generator().manual_seed(
        int(config["evaluation"]["bootstrap_seed"]) + seed_offset
    )
    values: dict[str, list[Tensor]] = {
        "regression_reduction": [],
        "gain_retention": [],
        "critic_minus_confidence_accuracy": [],
    }
    group_count = contributions.shape[0]
    for start in range(0, replicates, 128):
        chunk = min(128, replicates - start)
        samples = torch.randint(group_count, (chunk, group_count), generator=generator)
        totals = contributions[samples].sum(1)
        no_regressions = totals[:, 2]
        gain_denominator = totals[:, 5] - totals[:, 4]
        total_examples = totals[:, 0]
        reduction = torch.where(
            no_regressions > 0,
            1.0 - totals[:, 3] / no_regressions,
            torch.nan,
        )
        retention = torch.where(
            gain_denominator != 0,
            (totals[:, 7] - totals[:, 4]) / gain_denominator,
            torch.nan,
        )
        advantage = (totals[:, 7] - totals[:, 6]) / total_examples
        values["regression_reduction"].append(reduction)
        values["gain_retention"].append(retention)
        values["critic_minus_confidence_accuracy"].append(advantage)
    output: dict[str, Any] = {
        "replicates_requested": replicates,
        "counterfactual_groups": group_count,
        "sampling_unit": "counterfactual_group_within_seed",
    }
    for name, chunks in values.items():
        sample = torch.cat(chunks)
        valid = sample[~torch.isnan(sample)]
        output[name] = {
            "valid_replicates_numerator": int(valid.numel()),
            "requested_replicates_denominator": replicates,
            "lower_95": None if not len(valid) else float(torch.quantile(valid, 0.025)),
            "median": None if not len(valid) else float(torch.quantile(valid, 0.5)),
            "upper_95": None if not len(valid) else float(torch.quantile(valid, 0.975)),
        }
    return output


def per_example_result_records(
    evaluation: TrajectoryEvaluation, seed: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, metadata in enumerate(evaluation.example_metadata):
        record = dict(metadata)
        record.update(
            {
                "model_seed": seed,
                "no_latch_predictions_t1_t32": evaluation.no_latch_predictions[
                    index
                ].tolist(),
                "confidence_latch_predictions_b1_b32": evaluation.confidence_latch_predictions[
                    index
                ].tolist(),
                "critic_latch_predictions_b1_b32": evaluation.critic_latch_predictions[
                    index
                ].tolist(),
                "hysteretic_critic_latch_predictions_b1_b32": (
                    evaluation.hysteretic_critic_latch_predictions[index].tolist()
                ),
                "confidence_selected_horizons_b1_b32": evaluation.confidence_selected_horizons[
                    index
                ].tolist(),
                "critic_selected_horizons_b1_b32": evaluation.critic_selected_horizons[
                    index
                ].tolist(),
                "hysteretic_selected_horizons_b1_b32": (
                    evaluation.hysteretic_selected_horizons[index].tolist()
                ),
                "maximum_answer_probabilities_t1_t32": [
                    float(value)
                    for value in evaluation.maximum_answer_probabilities[index]
                ],
                "confidence_plus_schedule_scores_t1_t32": [
                    float(value) for value in evaluation.confidence_scores[index]
                ],
                "critic_scores_t1_t32": [
                    float(value) for value in evaluation.critic_scores[index]
                ],
            }
        )
        records.append(record)
    return records


def feature_boundary_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    configured_allowed = tuple(config["critic"]["allowed_features"])
    configured_forbidden = tuple(config["critic"]["forbidden_features"])
    configured_confidence = tuple(config["selectors"]["confidence_features"])
    allowed_match = configured_allowed == LatentProgressCritic.ALLOWED_FEATURE_NAMES
    forbidden_match = (
        configured_forbidden == LatentProgressCritic.FORBIDDEN_FEATURE_NAMES
    )
    confidence_match = configured_confidence == ConfidenceCalibrator.FEATURE_NAMES
    forbidden_overlap = sorted(set(configured_allowed) & set(configured_forbidden))
    critic_confidence_overlap = sorted(
        set(configured_allowed)
        & {
            "raw_decoder_logits",
            "answer_probability",
            "maximum_answer_probability",
            "top_two_margin",
            "top_two_probability_margin",
            "entropy",
            "answer_identity",
        }
    )
    return {
        "configured_allowed_features": configured_allowed,
        "implemented_allowed_features": LatentProgressCritic.ALLOWED_FEATURE_NAMES,
        "configured_forbidden_features": configured_forbidden,
        "implemented_forbidden_features": LatentProgressCritic.FORBIDDEN_FEATURE_NAMES,
        "confidence_features": configured_confidence,
        "equal_step_coordinate": "t_over_16" in configured_allowed
        and "t_over_16" in configured_confidence,
        "forbidden_overlap": forbidden_overlap,
        "critic_confidence_feature_overlap_excluding_schedule": critic_confidence_overlap,
        "pass": allowed_match
        and forbidden_match
        and confidence_match
        and not forbidden_overlap
        and not critic_confidence_overlap,
    }


def dataset_content_hash(datasets: Mapping[str, Sequence[DeductionExample]]) -> str:
    return sha256_json(
        {
            split: [
                {
                    "id": example.example_id,
                    "skeleton": example.skeleton_hash,
                    "text": example.rendered_text,
                    "gold": example.answer_position,
                }
                for example in examples
            ]
            for split, examples in datasets.items()
        }
    )


def dataset_token_audit(
    datasets: Mapping[str, Sequence[DeductionExample]], tokenizer: LocalTokenizer
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, examples in datasets.items():
        lengths = [
            len(tokenizer.encode(example.rendered_text)) + 1 for example in examples
        ]
        output[split] = {
            "examples": len(examples),
            "nonpadding_input_plus_answer_tokens": sum(lengths),
            "minimum_tokens_per_example": min(lengths),
            "maximum_tokens_per_example": max(lengths),
            "mean_tokens_per_example": statistics.mean(lengths),
        }
    return output


def generator_integrity_gate(generator_audit: Mapping[str, Any]) -> dict[str, Any]:
    required_observations = (
        "skeleton_hash_disjoint",
        "name_combination_disjoint",
        "counterfactual_group_atomic",
        "answer_position_balanced_within_strata",
        "lure_mixture_exact",
    )
    observed = all(bool(generator_audit[key]) for key in required_observations)
    return {
        "observed": observed,
        "required": True,
        "required_observations": list(required_observations),
        "pass": observed,
    }


def cuda_peak_record(
    device: torch.device, *, phase: str, seed: int, wall_time_seconds: float
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model_seed": seed,
        "wall_time_seconds": wall_time_seconds,
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def summarize_vram_phases(phases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not phases:
        raise ValueError("run-level VRAM accounting requires at least one phase")
    return {
        "peak_vram_allocated_bytes": max(
            int(phase["peak_vram_allocated_bytes"]) for phase in phases
        ),
        "peak_vram_reserved_bytes": max(
            int(phase["peak_vram_reserved_bytes"]) for phase in phases
        ),
        "vram_phases": [dict(phase) for phase in phases],
    }


def not_applicable_record(reason: str) -> dict[str, str]:
    return {"status": "not_applicable", "not_applicable_reason": reason}


def pretest_terminal_evaluation(
    reason: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    arm4_not_applicable = not_applicable_record(reason)
    return {
        "status": "not_applicable",
        "not_applicable_reason": reason,
        "official_test_inspected": False,
        "horizons": list(config["evaluation"]["horizons"]),
        "encoder_control": not_applicable_record(reason),
        "compute_by_seed": not_applicable_record(reason),
        "accuracy_grid": {
            **not_applicable_record(reason),
            "hysteretic_critic_latch_informational": arm4_not_applicable,
        },
        "stratified_accuracy_grid": not_applicable_record(reason),
        "trajectory_diagnostics": {
            **not_applicable_record(reason),
            "selector_switch_hazards": {
                "hysteretic_critic_latch_informational": not_applicable_record(
                    reason
                )
            },
        },
        "trajectory_accounting_assertions": not_applicable_record(reason),
        "endpoint_metrics": {
            **not_applicable_record(reason),
            "hysteretic_critic_latch_endpoint_accuracy_informational": (
                not_applicable_record(reason)
            ),
        },
        "paired_counterfactual_group_bootstrap": not_applicable_record(reason),
    }


def pretest_terminal_per_example_records(reason: str) -> dict[str, Any]:
    return {
        **not_applicable_record(reason),
        "official_test_inspected": False,
        "records_by_seed": {},
        "hysteretic_selected_horizons_b1_b32": not_applicable_record(reason),
    }


def _require_mapping(
    container: Mapping[str, Any], key: str, *, context: str
) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"schema 1.1.0 requires mapping {context}.{key}")
    return value


def _validate_not_applicable(value: Mapping[str, Any], *, context: str) -> None:
    if value.get("status") != "not_applicable" or not isinstance(
        value.get("not_applicable_reason"), str
    ):
        raise ValueError(f"schema 1.1.0 requires explicit not_applicable at {context}")


def validate_result_schema(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected_schema = config["result_artifact"]["schema_version"]
    if result.get("schema_version") != expected_schema or expected_schema != "1.1.0":
        raise ValueError("result does not satisfy schema 1.1.0")
    required_top_level = (
        "experiment_id",
        "started_utc",
        "completed_utc",
        "preregistration",
        "review_attestation",
        "hashes",
        "parameter_counts",
        "feature_boundary_audit",
        "generator_audit",
        "dataset_token_audit",
        "training",
        "calibration_frozen_t_star",
        "calibration_frozen_hysteresis_by_seed",
        "arm_definitions",
        "evaluation",
        "validity_gates",
        "decision",
        "compute",
        "per_example_records",
        "final_token",
    )
    missing = [key for key in required_top_level if result.get(key) is None]
    if missing:
        raise ValueError(f"schema 1.1.0 missing required sections: {missing}")
    if result.get("experiment_id") != config["experiment_id"]:
        raise ValueError("result experiment id does not match config")
    if not isinstance(result.get("started_utc"), str) or not isinstance(
        result.get("completed_utc"), str
    ):
        raise ValueError("result timestamps must be complete strings")

    final_token = result.get("final_token")
    if final_token not in FINAL_TOKENS:
        raise ValueError(f"invalid final token: {final_token}")
    decision = _require_mapping(result, "decision", context="result")
    if decision.get("final_token") != final_token:
        raise ValueError("decision and result final tokens differ")
    if not isinstance(decision.get("reason"), str):
        raise ValueError("decision requires an explicit reason")
    official_test_inspected = decision.get("official_test_inspected")
    if not isinstance(official_test_inspected, bool):
        raise ValueError("decision must state whether official test was inspected")

    for key in (
        "hashes",
        "parameter_counts",
        "feature_boundary_audit",
        "generator_audit",
        "dataset_token_audit",
        "arm_definitions",
        "validity_gates",
    ):
        _require_mapping(result, key, context="result")
    validity_gates = _require_mapping(result, "validity_gates", context="result")
    if not validity_gates:
        raise ValueError("validity_gates must not be empty")
    for name, gate in validity_gates.items():
        if not isinstance(gate, Mapping) or not isinstance(gate.get("pass"), bool):
            raise ValueError(f"validity gate {name} lacks a boolean pass value")

    training = _require_mapping(result, "training", context="result")
    expected_seed_keys = {str(seed) for seed in config["training"]["model_seeds"]}
    if set(training) != expected_seed_keys:
        raise ValueError("training section does not contain exactly both model seeds")
    for seed_key, record_value in training.items():
        if not isinstance(record_value, Mapping):
            raise ValueError(f"training record for seed {seed_key} is not a mapping")
        for section in (
            "selector_training_corpus",
            "selector_calibration_corpus",
            "confidence_calibrator_fit",
            "confidence_score_provenance",
            "selector_provenance",
        ):
            _require_mapping(record_value, section, context=f"training.{seed_key}")
        confidence_fit = record_value["confidence_calibrator_fit"]
        confidence_provenance = record_value["confidence_score_provenance"]
        if (
            confidence_fit.get("fit_corpus_split") != "selector_calibration"
            or confidence_fit.get("frozen_before_external_scoring") is not True
            or confidence_provenance.get("calibrator_fit_split")
            != "selector_calibration"
            or confidence_provenance.get("calibrator_frozen_before_scoring") is not True
            or confidence_provenance.get("critic_pair_construction_scored_split")
            != "selector_harvest"
        ):
            raise ValueError("confidence-calibrator split provenance is invalid")

    _require_mapping(result, "calibration_frozen_t_star", context="result")
    _require_mapping(
        result, "calibration_frozen_hysteresis_by_seed", context="result"
    )
    evaluation = _require_mapping(result, "evaluation", context="result")
    for section in config["result_artifact"]["required_evaluation_sections"]:
        _require_mapping(evaluation, section, context="evaluation")
    per_example = _require_mapping(result, "per_example_records", context="result")

    compute = _require_mapping(result, "compute", context="result")
    phases = compute.get("vram_phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("compute.vram_phases must record every measured phase")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("each VRAM phase must be a mapping")
        for key in (
            "phase",
            "model_seed",
            "wall_time_seconds",
            "peak_vram_allocated_bytes",
            "peak_vram_reserved_bytes",
        ):
            if key not in phase:
                raise ValueError(f"VRAM phase is missing {key}")
    expected_allocated = max(int(phase["peak_vram_allocated_bytes"]) for phase in phases)
    expected_reserved = max(int(phase["peak_vram_reserved_bytes"]) for phase in phases)
    if (
        int(compute.get("peak_vram_allocated_bytes", -1)) != expected_allocated
        or int(compute.get("peak_vram_reserved_bytes", -1)) != expected_reserved
    ):
        raise ValueError("run-level peak VRAM is not the maximum across phases")

    if not official_test_inspected:
        _validate_not_applicable(evaluation, context="evaluation")
        for section in config["result_artifact"]["required_evaluation_sections"]:
            _validate_not_applicable(
                evaluation[section], context=f"evaluation.{section}"
            )
        _validate_not_applicable(per_example, context="per_example_records")
        _validate_not_applicable(
            evaluation["accuracy_grid"]["hysteretic_critic_latch_informational"],
            context="evaluation.accuracy_grid.arm4",
        )
        _validate_not_applicable(
            evaluation["endpoint_metrics"][
                "hysteretic_critic_latch_endpoint_accuracy_informational"
            ],
            context="evaluation.endpoint_metrics.arm4",
        )
        _validate_not_applicable(
            evaluation["trajectory_diagnostics"]["selector_switch_hazards"][
                "hysteretic_critic_latch_informational"
            ],
            context="evaluation.trajectory_diagnostics.selector_switch_hazards.arm4",
        )
        _validate_not_applicable(
            per_example["hysteretic_selected_horizons_b1_b32"],
            context="per_example_records.arm4",
        )
        return

    if evaluation.get("status") != "complete":
        raise ValueError("completed official-test evaluation requires status=complete")
    accuracy = evaluation["accuracy_grid"]
    endpoint_metrics = evaluation["endpoint_metrics"]
    diagnostics = evaluation["trajectory_diagnostics"]
    for seed_key in expected_seed_keys:
        if "hysteretic_critic_latch_informational" not in accuracy.get(seed_key, {}):
            raise ValueError(f"accuracy grid is missing arm 4 for seed {seed_key}")
        records = per_example.get(seed_key)
        if not isinstance(records, list) or not records:
            raise ValueError(f"per-example records missing for seed {seed_key}")
        if any("hysteretic_selected_horizons_b1_b32" not in row for row in records):
            raise ValueError(f"per-example arm-4 selections missing for seed {seed_key}")
    for endpoint in ("16", "32"):
        for scope in (*sorted(expected_seed_keys), "pooled"):
            metrics = endpoint_metrics.get(endpoint, {}).get(scope, {})
            if "hysteretic_critic_latch_endpoint_accuracy_informational" not in metrics:
                raise ValueError(f"endpoint {endpoint}/{scope} is missing arm 4")
    for scope in (*sorted(expected_seed_keys), "pooled"):
        switches = diagnostics.get(scope, {}).get("selector_switch_hazards", {})
        if "hysteretic_critic_latch_informational" not in switches:
            raise ValueError(f"trajectory diagnostics are missing arm 4 for {scope}")


def _assert_immutable_temp_namespace_is_disjoint(final_path: Path) -> None:
    """Prove that the scavenger's namespace cannot contain a final artifact."""
    assert IMMUTABLE_RESULT_TEMP_SUFFIX != ".json"
    if final_path.suffix != ".json":
        raise ValueError("immutable result artifacts must use the .json suffix")
    if final_path.match(IMMUTABLE_RESULT_TEMP_GLOB):
        raise AssertionError("immutable temp pattern overlaps a final artifact name")


def scavenge_stale_immutable_result_temps(
    directory: Path, *, final_path: Path
) -> list[Path]:
    """Remove only stale files in the writer's disjoint private namespace."""
    _assert_immutable_temp_namespace_is_disjoint(final_path)
    if not directory.exists():
        return []
    removed: list[Path] = []
    for candidate in directory.glob(IMMUTABLE_RESULT_TEMP_GLOB):
        candidate.unlink()
        removed.append(candidate)
    if removed:
        _fsync_directory(directory)
    return removed


def _fsync_directory(directory: Path) -> bool:
    """Durably commit directory entries where Python exposes that operation.

    Windows does not permit opening a directory with ``os.open`` for
    ``os.fsync``. Returning ``False`` records that limitation honestly; it
    does not claim directory-entry durability that the runtime cannot provide.
    """
    if os.name == "nt":
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def write_immutable_result(
    result: Mapping[str, Any],
    path: Path,
    *,
    _crash_probe: str | None = None,
) -> None:
    final_token = result.get("final_token")
    if final_token not in FINAL_TOKENS:
        raise ValueError(f"invalid final token: {final_token}")
    if _crash_probe is not None and _crash_probe not in ATOMIC_CRASH_WINDOWS:
        raise ValueError(f"unknown atomic crash probe: {_crash_probe}")
    _assert_immutable_temp_namespace_is_disjoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=IMMUTABLE_RESULT_TEMP_PREFIX,
            suffix=IMMUTABLE_RESULT_TEMP_SUFFIX,
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if _crash_probe == "after_fsync_before_link":
            os._exit(91)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"immutable result already exists: {path}") from error
        if _crash_probe == "after_link_before_unlink":
            os._exit(92)
        temporary.unlink()
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def _checkpoint_lookup(
    records: Sequence[Mapping[str, Any]], tokens: int
) -> Mapping[str, Any]:
    for record in records:
        if int(record["processed_tokens"]) == tokens:
            return record
    raise KeyError(tokens)


def run_pilot(config_path: Path, review_attestation: str) -> int:
    required_attestation = "INDEPENDENT_PRETRAINING_REVIEW_CLEAN"
    if review_attestation != required_attestation:
        raise RuntimeError(
            "launch blocked: pass the exact independent-review attestation only after review"
        )
    if RESULT_PATH.exists():
        raise RuntimeError(f"immutable result already exists: {RESULT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the preregistered pilot requires CUDA; CPU launch is forbidden"
        )
    os.environ["HF_HOME"] = str(HERE.parent.parent / ".hf_cache")
    config = load_config(config_path)
    code_hash = sha256_file(Path(__file__).resolve())
    config_hash = sha256_file(config_path.resolve())
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["training"]["tf32"])
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    generator = DeductionGenerator(config)
    datasets = generator.generate_dataset()
    generator_audit = generator.audit_dataset(datasets)
    split_hash = dataset_content_hash(datasets)
    tokenizer = LocalTokenizer(config)
    token_audit = dataset_token_audit(datasets, tokenizer)
    feature_audit = feature_boundary_audit(config)

    torch.manual_seed(0)
    common_probe = CommonRecurrentModel(config, len(tokenizer.id_to_token))
    critic_probe = LatentProgressCritic(config)
    encoder_probe = EncoderOnlyControl(config, len(tokenizer.id_to_token))
    parameter_counts = {
        "common_recurrent": trainable_parameter_count(common_probe),
        "latent_critic": trainable_parameter_count(critic_probe),
        "common_plus_critic": trainable_parameter_count(common_probe)
        + trainable_parameter_count(critic_probe),
        "encoder_only_control": trainable_parameter_count(encoder_probe),
    }
    parameter_counts["encoder_relative_difference"] = (
        abs(
            parameter_counts["encoder_only_control"]
            - parameter_counts["common_recurrent"]
        )
        / parameter_counts["common_recurrent"]
    )
    del common_probe, critic_probe, encoder_probe

    seed_preparation: dict[int, dict[str, Any]] = {}
    calibration_evaluations: dict[int, TrajectoryEvaluation] = {}
    vram_phases: list[dict[str, Any]] = []
    model_seeds = tuple(int(seed) for seed in config["training"]["model_seeds"])
    checkpoint_root = HERE / "checkpoints" / "exp_e2_latch_mechanics"
    for seed in model_seeds:
        seed_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        seed_dir = checkpoint_root / f"seed_{seed}"
        torch.manual_seed(seed)
        common = CommonRecurrentModel(config, len(tokenizer.id_to_token))
        common_training = train_common_model(
            common,
            datasets["controller_train"],
            datasets["selector_calibration"],
            tokenizer,
            config,
            device,
            seed=seed,
            checkpoint_dir=seed_dir,
            config_hash=config_hash,
            code_hash=code_hash,
        )
        torch.manual_seed(seed + 1_000_000)
        encoder = EncoderOnlyControl(config, len(tokenizer.id_to_token))
        encoder_training = train_encoder_control(
            encoder,
            datasets["controller_train"],
            datasets["selector_calibration"],
            tokenizer,
            config,
            device,
            seed=seed,
            checkpoint_dir=seed_dir,
            config_hash=config_hash,
            code_hash=code_hash,
        )
        del encoder
        torch.manual_seed(seed + 2_000_000)
        critic = LatentProgressCritic(config).to(device)
        calibrator = ConfidenceCalibrator().to(device)
        training_corpus = harvest_selector_training_corpus(
            common,
            critic,
            datasets["selector_harvest"],
            tokenizer,
            config,
            device,
            checkpoints=common_training["checkpoints"],
            checkpoint_dir=seed_dir,
            config_hash=config_hash,
            code_hash=code_hash,
            seed=seed,
        )
        final_common = _checkpoint_lookup(
            common_training["checkpoints"],
            int(config["training"]["shared_controller_tokens"]),
        )
        load_checked_checkpoint(
            seed_dir / Path(str(final_common["path"])).name,
            common,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=int(final_common["processed_tokens"]),
            map_location=device,
        )
        calibration_corpus = harvest_fixed_selector_corpus(
            common,
            critic,
            datasets["selector_calibration"],
            tokenizer,
            config,
            device,
        )
        confidence_fit = fit_confidence_calibrator(
            calibrator,
            calibration_corpus,
            config,
            device,
            seed=seed,
            corpus_split="selector_calibration",
        )
        training_confidence_scores = calibrator_scores(
            calibrator, training_corpus, device
        )
        critic_fit = fit_latent_critic(
            critic,
            training_corpus,
            training_confidence_scores,
            config,
            device,
            seed=seed,
        )
        calibration_confidence_scores = calibrator_scores(
            calibrator, calibration_corpus, device
        )
        calibration_critic_scores = critic_scores(critic, calibration_corpus, device)
        provenance = selector_provenance_metrics(
            calibration_corpus,
            calibration_confidence_scores,
            calibration_critic_scores,
            config,
        )
        selector_path = seed_dir / "selectors.pt"
        atomic_torch_save(
            {
                "schema_version": 1,
                "config_hash": config_hash,
                "code_hash": code_hash,
                "seed": seed,
                "confidence_calibrator_state": calibrator.state_dict(),
                "latent_critic_state": critic.state_dict(),
                "feature_boundary_audit": feature_audit,
            },
            selector_path,
        )
        preparation_phase = cuda_peak_record(
            device,
            phase="training_and_selector_fitting",
            seed=seed,
            wall_time_seconds=time.perf_counter() - seed_started,
        )
        vram_phases.append(preparation_phase)
        torch.cuda.reset_peak_memory_stats(device)
        calibration_started = time.perf_counter()
        calibration_evaluations[seed] = evaluate_trajectories(
            common,
            calibrator,
            critic,
            datasets["selector_calibration"],
            tokenizer,
            device,
            max_horizon=16,
        )
        calibration_phase = cuda_peak_record(
            device,
            phase="selector_calibration_evaluation",
            seed=seed,
            wall_time_seconds=time.perf_counter() - calibration_started,
        )
        vram_phases.append(calibration_phase)
        seed_preparation[seed] = {
            "common_training": common_training,
            "encoder_training": encoder_training,
            "selector_training_corpus": training_corpus.audit(),
            "selector_calibration_corpus": calibration_corpus.audit(),
            "confidence_calibrator_fit": confidence_fit,
            "confidence_score_provenance": {
                "calibrator_fit_split": "selector_calibration",
                "calibrator_frozen_before_scoring": True,
                "critic_pair_construction_scored_split": "selector_harvest",
            },
            "latent_critic_fit": critic_fit,
            "selector_provenance": provenance,
            "selector_checkpoint": {
                "path": str(selector_path),
                "sha256": sha256_file(selector_path),
            },
            "compute": {
                "wall_time_seconds": time.perf_counter() - seed_started,
                **summarize_vram_phases(
                    [preparation_phase, calibration_phase]
                ),
            },
        }
        del common, critic, calibrator, training_corpus, calibration_corpus
        torch.cuda.empty_cache()

    t_star, t_star_record = select_t_star(
        calibration_evaluations,
        tuple(int(value) for value in config["evaluation"]["t_star_candidates"]),
    )
    frozen_hysteresis: dict[int, dict[str, Any]] = {}
    for seed, evaluation in calibration_evaluations.items():
        selected_delta, record = select_hysteresis_delta(
            evaluation.no_latch_predictions,
            evaluation.gold,
            evaluation.critic_scores,
            t_star,
            config,
        )
        frozen_hysteresis[seed] = record
        if not math.isclose(selected_delta, float(record["selected_delta"])):
            raise AssertionError("frozen hysteresis delta record mismatch")
    failed_provenance_seeds = [
        seed
        for seed, record in seed_preparation.items()
        if not record["selector_provenance"]["all_gates_pass"]
    ]
    invalid_selector_fit_seeds = [
        seed
        for seed, record in seed_preparation.items()
        if not record["latent_critic_fit"]["fit_performed"]
    ]
    base_result: dict[str, Any] = {
        "schema_version": config["result_artifact"]["schema_version"],
        "experiment_id": config["experiment_id"],
        "started_utc": started_utc,
        "completed_utc": None,
        "preregistration": config["preregistration"],
        "review_attestation": required_attestation,
        "hashes": {
            "code_sha256": code_hash,
            "config_sha256": config_hash,
            "generator_version_sha256": sha256_json(config["generator"]),
            "split_content_sha256": split_hash,
            "tokenizer_vocabulary_sha256": tokenizer.vocabulary_hash(),
            "checkpoints_by_seed": {
                str(seed): {
                    "common_selected_sha256": seed_preparation[seed]["common_training"][
                        "selected_checkpoint"
                    ]["sha256"],
                    "encoder_selected_sha256": seed_preparation[seed][
                        "encoder_training"
                    ]["selected_checkpoint"]["sha256"],
                    "selectors_sha256": seed_preparation[seed]["selector_checkpoint"][
                        "sha256"
                    ],
                }
                for seed in model_seeds
            },
        },
        "parameter_counts": parameter_counts,
        "feature_boundary_audit": feature_audit,
        "generator_audit": generator_audit,
        "dataset_token_audit": token_audit,
        "training": {str(seed): record for seed, record in seed_preparation.items()},
        "calibration_frozen_t_star": t_star_record,
        "calibration_frozen_hysteresis_by_seed": {
            str(seed): record for seed, record in frozen_hysteresis.items()
        },
        "arm_definitions": {
            "1": "final_horizon_no_latch",
            "2": "confidence_plus_schedule_latch",
            "3": "latent_critic_latch_adjudicating",
            "4": "hysteretic_critic_latch_informational_only",
        },
        "evaluation": None,
        "validity_gates": None,
        "decision": None,
        "compute": None,
        "per_example_records": None,
        "final_token": None,
    }
    if invalid_selector_fit_seeds or failed_provenance_seeds:
        final_token = "VOID" if invalid_selector_fit_seeds else "FAIL"
        reason = (
            "INSUFFICIENT_OPPOSITE_CORRECTNESS_SELECTOR_TRAINING_PAIRS"
            if invalid_selector_fit_seeds
            else "PRETEST_SELECTOR_PROVENANCE_GATE_MISSED"
        )
        base_result["decision"] = {
            "final_token": final_token,
            "reason": reason,
            "invalid_selector_fit_seed_numerators": invalid_selector_fit_seeds,
            "failed_seed_numerators": failed_provenance_seeds,
            "model_seed_denominator": len(model_seeds),
            "official_test_inspected": False,
        }
        pretest_validity_gates: dict[str, dict[str, Any]] = {
            "generator_and_split_integrity": generator_integrity_gate(
                generator_audit
            ),
            "feature_boundary": {
                "observed": feature_audit["pass"],
                "required": True,
                "pass": bool(feature_audit["pass"]),
            },
            "both_model_seeds_prepared": {
                "observed_numerator": len(seed_preparation),
                "required_denominator": len(model_seeds),
                "pass": set(seed_preparation) == set(model_seeds),
            },
        }
        for seed in model_seeds:
            training_record = seed_preparation[seed]
            pretest_validity_gates.update(
                {
                    f"seed_{seed}_shared_controller_token_accounting": {
                        "observed": training_record["common_training"][
                            "processed_nonpadding_tokens"
                        ],
                        "required": config["training"]["shared_controller_tokens"],
                        "pass": training_record["common_training"][
                            "processed_nonpadding_tokens"
                        ]
                        == int(config["training"]["shared_controller_tokens"]),
                    },
                    f"seed_{seed}_selector_training_states": {
                        "observed": training_record["selector_training_corpus"][
                            "states"
                        ],
                        "minimum": config["selectors"][
                            "minimum_training_states_per_seed"
                        ],
                        "pass": training_record["selector_training_corpus"][
                            "states"
                        ]
                        >= int(
                            config["selectors"]["minimum_training_states_per_seed"]
                        ),
                    },
                    f"seed_{seed}_selector_calibration_states": {
                        "observed": training_record["selector_calibration_corpus"][
                            "states"
                        ],
                        "minimum": config["selectors"][
                            "minimum_calibration_states_per_seed"
                        ],
                        "pass": training_record["selector_calibration_corpus"][
                            "states"
                        ]
                        >= int(
                            config["selectors"][
                                "minimum_calibration_states_per_seed"
                            ]
                        ),
                    },
                    f"seed_{seed}_selector_fit_performed": {
                        "observed": training_record["latent_critic_fit"][
                            "fit_performed"
                        ],
                        "required": True,
                        "pass": bool(
                            training_record["latent_critic_fit"]["fit_performed"]
                        ),
                    },
                    f"seed_{seed}_selector_provenance": {
                        "observed": training_record["selector_provenance"][
                            "all_gates_pass"
                        ],
                        "required": True,
                        "pass": bool(
                            training_record["selector_provenance"]["all_gates_pass"]
                        ),
                    },
                    f"seed_{seed}_confidence_split_contract": {
                        "observed": training_record["confidence_score_provenance"],
                        "required": {
                            "calibrator_fit_split": "selector_calibration",
                            "calibrator_frozen_before_scoring": True,
                            "critic_pair_construction_scored_split": (
                                "selector_harvest"
                            ),
                        },
                        "pass": training_record["confidence_score_provenance"]
                        == {
                            "calibrator_fit_split": "selector_calibration",
                            "calibrator_frozen_before_scoring": True,
                            "critic_pair_construction_scored_split": (
                                "selector_harvest"
                            ),
                        },
                    },
                }
            )
        base_result["evaluation"] = pretest_terminal_evaluation(reason, config)
        base_result["validity_gates"] = pretest_validity_gates
        base_result["per_example_records"] = pretest_terminal_per_example_records(
            reason
        )
        base_result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        base_result["compute"] = {
            "wall_time_seconds": time.perf_counter() - run_started,
            **summarize_vram_phases(vram_phases),
        }
        base_result["final_token"] = final_token
        validate_result_schema(base_result, config)
        write_immutable_result(base_result, RESULT_PATH)
        print(final_token)
        return 0

    test_evaluations: dict[int, TrajectoryEvaluation] = {}
    encoder_results: dict[int, dict[str, int | float]] = {}
    evaluation_compute_by_seed: dict[int, dict[str, Any]] = {}
    for seed in model_seeds:
        torch.cuda.reset_peak_memory_stats(device)
        evaluation_started = time.perf_counter()
        seed_dir = checkpoint_root / f"seed_{seed}"
        torch.manual_seed(seed)
        common = CommonRecurrentModel(config, len(tokenizer.id_to_token)).to(device)
        final_record = _checkpoint_lookup(
            seed_preparation[seed]["common_training"]["checkpoints"],
            int(config["training"]["shared_controller_tokens"]),
        )
        load_checked_checkpoint(
            seed_dir / Path(str(final_record["path"])).name,
            common,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=int(final_record["processed_tokens"]),
            map_location=device,
        )
        critic = LatentProgressCritic(config).to(device)
        calibrator = ConfidenceCalibrator().to(device)
        selector_payload = torch.load(
            seed_dir / "selectors.pt", map_location=device, weights_only=False
        )
        if (
            selector_payload["config_hash"] != config_hash
            or selector_payload["code_hash"] != code_hash
        ):
            raise RuntimeError("selector checkpoint provenance mismatch")
        critic.load_state_dict(selector_payload["latent_critic_state"])
        calibrator.load_state_dict(selector_payload["confidence_calibrator_state"])
        test_evaluations[seed] = evaluate_trajectories(
            common,
            calibrator,
            critic,
            datasets["test"],
            tokenizer,
            device,
            max_horizon=32,
            hysteresis_delta=float(frozen_hysteresis[seed]["selected_delta"]),
        )
        encoder = EncoderOnlyControl(config, len(tokenizer.id_to_token)).to(device)
        selected_encoder = seed_preparation[seed]["encoder_training"][
            "selected_checkpoint"
        ]
        load_checked_checkpoint(
            Path(str(selected_encoder["path"])),
            encoder,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=int(selected_encoder["processed_tokens"]),
            map_location=device,
        )
        encoder_results[seed] = evaluate_encoder_control(
            encoder, datasets["test"], tokenizer, device
        )
        evaluation_phase = cuda_peak_record(
            device,
            phase="official_test_evaluation",
            seed=seed,
            wall_time_seconds=time.perf_counter() - evaluation_started,
        )
        vram_phases.append(evaluation_phase)
        evaluation_compute_by_seed[seed] = evaluation_phase
        del common, critic, calibrator, encoder
        torch.cuda.empty_cache()

    endpoint_metrics: dict[int, dict[str, Mapping[str, Any]]] = {}
    for endpoint in (
        int(config["evaluation"]["primary_endpoint"]),
        int(config["evaluation"]["conditional_endpoint"]),
    ):
        per_seed = {
            seed: endpoint_counts(evaluation, t_star, endpoint, config)
            for seed, evaluation in test_evaluations.items()
        }
        endpoint_metrics[endpoint] = {
            **{str(seed): metrics for seed, metrics in per_seed.items()},
            "pooled": pooled_endpoint_counts(per_seed, endpoint),
        }

    trajectory_diagnostics_by_scope = {
        **{
            str(seed): trajectory_diagnostics([evaluation])
            for seed, evaluation in test_evaluations.items()
        },
        "pooled": trajectory_diagnostics(list(test_evaluations.values())),
    }
    trajectory_assertions = trajectory_accounting_assertions(
        trajectory_diagnostics_by_scope,
        test_evaluations,
        t_star,
        (
            int(config["evaluation"]["primary_endpoint"]),
            int(config["evaluation"]["conditional_endpoint"]),
        ),
    )

    eval_cfg = config["evaluation"]
    common_range = config["common_model"]["target_parameter_range"]
    critic_range = config["critic"]["target_parameter_range"]
    validity_gates: dict[str, dict[str, Any]] = {
        "generator_and_split_integrity": generator_integrity_gate(generator_audit),
        "feature_boundary": {
            "observed": feature_audit["pass"],
            "required": True,
            "pass": bool(feature_audit["pass"]),
        },
        "common_parameter_range": {
            "observed": parameter_counts["common_recurrent"],
            "minimum": common_range[0],
            "maximum": common_range[1],
            "pass": common_range[0]
            <= parameter_counts["common_recurrent"]
            <= common_range[1],
        },
        "critic_parameter_range": {
            "observed": parameter_counts["latent_critic"],
            "minimum": critic_range[0],
            "maximum": critic_range[1],
            "pass": critic_range[0]
            <= parameter_counts["latent_critic"]
            <= critic_range[1],
        },
        "encoder_parameter_match": {
            "observed": parameter_counts["encoder_relative_difference"],
            "maximum": config["encoder_control"]["parameter_match_relative_tolerance"],
            "pass": parameter_counts["encoder_relative_difference"]
            <= float(config["encoder_control"]["parameter_match_relative_tolerance"]),
        },
        "both_model_seeds_present": {
            "observed_numerator": len(test_evaluations),
            "required_denominator": len(model_seeds),
            "pass": set(test_evaluations) == set(model_seeds),
        },
        "trajectory_accounting_identities": {
            "observed_violation_numerator": trajectory_assertions["violation_count"],
            "required_violation_denominator": 0,
            "pass": trajectory_assertions["all_pass"],
            "failure_semantics": "evaluator_bug_VOID_integrity",
        },
    }
    for seed in model_seeds:
        metrics16 = endpoint_metrics[16][str(seed)]
        training_record = seed_preparation[seed]
        validity_gates.update(
            {
                f"seed_{seed}_shared_controller_token_accounting": {
                    "observed": training_record["common_training"][
                        "processed_nonpadding_tokens"
                    ],
                    "required": config["training"]["shared_controller_tokens"],
                    "pass": training_record["common_training"][
                        "processed_nonpadding_tokens"
                    ]
                    == int(config["training"]["shared_controller_tokens"]),
                },
                f"seed_{seed}_logical_arm_exposure_accounting": {
                    "observed": training_record["common_training"][
                        "logical_arm_exposure"
                    ],
                    "required": config["training"]["logical_arm_exposure"],
                    "pass": training_record["common_training"]["logical_arm_exposure"]
                    == int(config["training"]["logical_arm_exposure"]),
                },
                f"seed_{seed}_selector_training_states": {
                    "observed": training_record["selector_training_corpus"]["states"],
                    "minimum": config["selectors"]["minimum_training_states_per_seed"],
                    "pass": training_record["selector_training_corpus"]["states"]
                    >= int(config["selectors"]["minimum_training_states_per_seed"]),
                },
                f"seed_{seed}_selector_calibration_states": {
                    "observed": training_record["selector_calibration_corpus"][
                        "states"
                    ],
                    "minimum": config["selectors"][
                        "minimum_calibration_states_per_seed"
                    ],
                    "pass": training_record["selector_calibration_corpus"]["states"]
                    >= int(config["selectors"]["minimum_calibration_states_per_seed"]),
                },
                f"seed_{seed}_t_star_competence": {
                    "numerator": metrics16["no_latch_t_star_accuracy"]["numerator"],
                    "denominator": metrics16["no_latch_t_star_accuracy"]["denominator"],
                    "observed": metrics16["no_latch_t_star_accuracy"]["value"],
                    "minimum": eval_cfg["minimum_t_star_accuracy_per_seed"],
                    "pass": metrics16["no_latch_t_star_accuracy"]["value"]
                    >= float(eval_cfg["minimum_t_star_accuracy_per_seed"]),
                },
                f"seed_{seed}_t_star_gain": {
                    "t_star_correct_numerator": metrics16["no_latch_t_star_accuracy"][
                        "numerator"
                    ],
                    "t1_correct_numerator": metrics16["no_latch_t1_accuracy"][
                        "numerator"
                    ],
                    "denominator": metrics16["examples"],
                    "observed": metrics16["no_latch_t_star_accuracy"]["value"]
                    - metrics16["no_latch_t1_accuracy"]["value"],
                    "minimum": eval_cfg["minimum_t_star_gain_per_seed"],
                    "pass": metrics16["no_latch_t_star_accuracy"]["value"]
                    - metrics16["no_latch_t1_accuracy"]["value"]
                    >= float(eval_cfg["minimum_t_star_gain_per_seed"]),
                },
                f"seed_{seed}_earlier_correct_denominator": {
                    "observed": metrics16["earlier_correct"]["numerator"],
                    "minimum": eval_cfg["minimum_earlier_correct_denominator_per_seed"],
                    "pass": metrics16["earlier_correct"]["numerator"]
                    >= int(eval_cfg["minimum_earlier_correct_denominator_per_seed"]),
                },
                f"seed_{seed}_confidence_matched_pairs_h16": {
                    "observed": metrics16["confidence_matched_critic_concordance"][
                        "qualifying_pairs"
                    ],
                    "minimum": config["selectors"][
                        "minimum_confidence_matched_pairs_per_seed"
                    ],
                    "pass": metrics16["confidence_matched_critic_concordance"][
                        "qualifying_pairs"
                    ]
                    >= int(
                        config["selectors"]["minimum_confidence_matched_pairs_per_seed"]
                    ),
                },
                f"seed_{seed}_confidence_matched_pairs_h32": {
                    "observed": endpoint_metrics[32][str(seed)][
                        "confidence_matched_critic_concordance"
                    ]["qualifying_pairs"],
                    "minimum": config["selectors"][
                        "minimum_confidence_matched_pairs_per_seed"
                    ],
                    "pass": endpoint_metrics[32][str(seed)][
                        "confidence_matched_critic_concordance"
                    ]["qualifying_pairs"]
                    >= int(
                        config["selectors"]["minimum_confidence_matched_pairs_per_seed"]
                    ),
                },
                f"seed_{seed}_encoder_control_ceiling": {
                    "numerator": encoder_results[seed]["correct"],
                    "denominator": encoder_results[seed]["total"],
                    "observed": encoder_results[seed]["value"],
                    "void_if_strictly_above": config["encoder_control"][
                        "void_accuracy_strictly_above"
                    ],
                    "pass": encoder_results[seed]["value"]
                    <= float(config["encoder_control"]["void_accuracy_strictly_above"]),
                },
            }
        )
    decision = adjudicate(endpoint_metrics, validity_gates, config)
    decision["official_test_inspected"] = True
    bootstraps: dict[str, Any] = {}
    for endpoint in (16, 32):
        bootstraps[str(endpoint)] = {
            str(seed): paired_group_bootstrap(
                [test_evaluations[seed]],
                t_star,
                endpoint,
                config,
                seed_offset=seed + endpoint,
            )
            for seed in model_seeds
        }
        bootstraps[str(endpoint)]["pooled"] = paired_group_bootstrap(
            list(test_evaluations.values()),
            t_star,
            endpoint,
            config,
            seed_offset=endpoint,
        )
    base_result["evaluation"] = {
        "status": "complete",
        "official_test_inspected": True,
        "horizons": list(range(1, 33)),
        "encoder_control": {
            str(seed): result for seed, result in encoder_results.items()
        },
        "compute_by_seed": {
            str(seed): record for seed, record in evaluation_compute_by_seed.items()
        },
        "accuracy_grid": {
            str(seed): accuracy_grid(evaluation)
            for seed, evaluation in test_evaluations.items()
        },
        "stratified_accuracy_grid": {
            str(seed): stratified_accuracy_grid(evaluation)
            for seed, evaluation in test_evaluations.items()
        },
        "trajectory_diagnostics": trajectory_diagnostics_by_scope,
        "trajectory_accounting_assertions": trajectory_assertions,
        "endpoint_metrics": {
            str(endpoint): scopes for endpoint, scopes in endpoint_metrics.items()
        },
        "paired_counterfactual_group_bootstrap": bootstraps,
    }
    base_result["validity_gates"] = validity_gates
    base_result["decision"] = decision
    base_result["per_example_records"] = {
        str(seed): per_example_result_records(evaluation, seed)
        for seed, evaluation in test_evaluations.items()
    }
    base_result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    base_result["compute"] = {
        "wall_time_seconds": time.perf_counter() - run_started,
        **summarize_vram_phases(vram_phases),
    }
    base_result["final_token"] = decision["final_token"]
    validate_result_schema(base_result, config)
    write_immutable_result(base_result, RESULT_PATH)
    print(decision["final_token"])
    return 0


def _hand_checked_symbolic_tests() -> dict[str, Any]:
    chain = SymbolicVerifier(
        entities=("a", "b"),
        unary_facts=(UnaryAtom("a", "red"),),
        relation_facts=(),
        rules=(Rule("unary", "red", "blue"), Rule("unary", "blue", "green")),
    ).closure()
    assert chain[UnaryAtom("a", "green")] == ProofRecord(2, 1)
    duplicate = SymbolicVerifier(
        entities=("a",),
        unary_facts=(UnaryAtom("a", "red"),),
        relation_facts=(),
        rules=(Rule("unary", "red", "green"), Rule("unary", "red", "green")),
    ).closure()
    assert duplicate[UnaryAtom("a", "green")] == ProofRecord(1, 2)
    relational = SymbolicVerifier(
        entities=("a", "b"),
        unary_facts=(UnaryAtom("a", "ready"), UnaryAtom("b", "calm")),
        relation_facts=(RelationAtom("a", "guides", "b"),),
        rules=(
            Rule("rel_out_self", "ready", "careful", relation="guides"),
            Rule("rel_in_self", "calm", "patient", relation="guides"),
            Rule("rel_out_other", "ready", "quiet", relation="guides"),
            Rule("conjunction", "quiet", "wise", body2="calm"),
        ),
    ).closure()
    assert relational[UnaryAtom("a", "careful")].cost == 1
    assert relational[UnaryAtom("b", "patient")].cost == 1
    assert relational[UnaryAtom("b", "quiet")].cost == 1
    assert relational[UnaryAtom("b", "wise")].cost == 2
    return {
        "chain_minimum_cost": 2,
        "chain_shortest_proof_count": 1,
        "duplicate_shortest_proof_count": 2,
        "relation_rule_types_checked": 3,
        "conjunction_cost_checked": 2,
        "pass": True,
    }


def _synthetic_metrics(
    regression: float,
    reduction: float,
    retention: float,
    advantage: float,
    concordance: float,
) -> dict[str, Any]:
    return {
        "no_latch_regression": {"value": regression},
        "regression_reduction": {"value": reduction},
        "gain_retention": {"value": retention},
        "critic_minus_confidence_accuracy": {"value": advantage},
        "confidence_matched_critic_concordance": {"value": concordance},
    }


def _resume_determinism_self_test() -> dict[str, Any]:
    seed = 20260813
    config_hash = "self-test-config"
    code_hash = "self-test-code"

    def initialize() -> tuple[nn.Module, torch.optim.Optimizer]:
        random.seed(seed)
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(8, 2),
        )
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train_updates(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: list[dict[str, Any]],
        start: int,
        stop: int,
    ) -> None:
        model.train()
        for update in range(start, stop):
            python_scale = random.random()
            features = torch.randn(6, 4)
            targets = torch.randn(6, 2)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features) * python_scale
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "update": update,
                    "python_scale": python_scale,
                    "loss": float(loss.detach()),
                }
            )

    uninterrupted, uninterrupted_optimizer = initialize()
    uninterrupted_history: list[dict[str, Any]] = []
    train_updates(
        uninterrupted, uninterrupted_optimizer, uninterrupted_history, 0, 6
    )

    interrupted, interrupted_optimizer = initialize()
    interrupted_history: list[dict[str, Any]] = []
    train_updates(interrupted, interrupted_optimizer, interrupted_history, 0, 3)
    with tempfile.TemporaryDirectory(prefix="e2-resume-self-test-") as directory:
        checkpoint = Path(directory) / "interrupted.pt"
        atomic_torch_save(
            checkpoint_payload(
                interrupted,
                interrupted_optimizer,
                processed_tokens=3,
                config_hash=config_hash,
                code_hash=code_hash,
                seed=seed,
                history=interrupted_history,
            ),
            checkpoint,
        )
        provenance_mismatches_rejected = 0
        for wrong_seed, wrong_tokens in ((seed + 1, 3), (seed, 4)):
            probe, _ = initialize()
            try:
                load_checked_checkpoint(
                    checkpoint,
                    probe,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    expected_seed=wrong_seed,
                    expected_processed_tokens=wrong_tokens,
                )
            except RuntimeError:
                provenance_mismatches_rejected += 1
            else:
                raise AssertionError("checkpoint provenance mismatch was accepted")

        random.seed(seed + 99)
        torch.manual_seed(seed + 99)
        resumed = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(8, 2),
        )
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        payload = load_checked_checkpoint(
            checkpoint,
            resumed,
            optimizer=resumed_optimizer,
            config_hash=config_hash,
            code_hash=code_hash,
            expected_seed=seed,
            expected_processed_tokens=3,
            restore_rng=True,
        )
        resumed_history = list(payload["history"])
        train_updates(resumed, resumed_optimizer, resumed_history, 3, 6)

    histories_identical = resumed_history == uninterrupted_history
    weights_identical = all(
        torch.equal(resumed.state_dict()[name], value)
        for name, value in uninterrupted.state_dict().items()
    )
    assert histories_identical
    assert weights_identical
    assert provenance_mismatches_rejected == 2
    return {
        "updates": 6,
        "interruption_after_update": 3,
        "python_cpu_torch_rng_restored": True,
        "cuda_rng_not_initialized_by_cpu_test": True,
        "provenance_mismatches_rejected": provenance_mismatches_rejected,
        "histories_identical": histories_identical,
        "weights_identical": weights_identical,
        "pass": True,
    }


def _hysteresis_rule_self_test() -> dict[str, Any]:
    tie = hysteretic_incumbent_indices(torch.tensor([[0.5, 0.5]]), 0.0)
    assert tie.tolist() == [[0, 0]]
    threshold = hysteretic_incumbent_indices(torch.tensor([[0.50, 0.53, 0.56]]), 0.02)
    assert threshold.tolist() == [[0, 1, 2]]
    strict_boundary = hysteretic_incumbent_indices(
        torch.tensor([[0.50, 0.75, 0.80]], dtype=torch.float64), 0.25
    )
    assert strict_boundary.tolist() == [[0, 0, 2]]
    persistence = hysteretic_incumbent_indices(
        torch.tensor([[0.60, 0.69, 0.65, 0.80]]), 0.10
    )
    assert persistence.tolist() == [[0, 0, 0, 3]]
    return {
        "tie_retains_incumbent": tie.tolist(),
        "delta_thresholding": threshold.tolist(),
        "strict_boundary_retention": strict_boundary.tolist(),
        "incumbent_persistence": persistence.tolist(),
        "pass": True,
    }


def _delta_selection_self_test(config: Mapping[str, Any]) -> dict[str, Any]:
    examples = 20
    horizons = 16
    gold = torch.ones(examples, dtype=torch.long)
    predictions = torch.ones((examples, horizons), dtype=torch.long)
    predictions[10:, 0] = 0
    predictions[:4, 2:] = 0
    scores = torch.full((examples, horizons), 0.56)
    scores[:, 0] = 0.50
    scores[:4, 2:] = 0.59
    selected, record = select_hysteresis_delta(predictions, gold, scores, 2, config)
    assert math.isclose(selected, 0.05)
    assert record["candidate_calibration_metrics"]["0.05"]["feasible"]
    assert not record["candidate_calibration_metrics"]["0.10"]["feasible"]
    assert not record["calibration_constraint_miss"]
    return {"selected_delta": selected, "selection_record": record, "pass": True}


def _diagnostics_self_test() -> dict[str, Any]:
    gold = torch.ones(4, dtype=torch.long)
    raw = torch.tensor(
        [[1, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]],
        dtype=torch.long,
    )
    scores = torch.tensor(
        [
            [0.6, 0.7, 0.8, 0.8],
            [0.5, 0.7, 0.8, 0.8],
            [0.5, 0.5, 0.7, 0.7],
            [0.7, 0.7, 0.8, 0.8],
        ]
    )
    critic_selected = cumulative_first_argmax(scores) + 1
    hysteretic_selected = hysteretic_incumbent_indices(scores, 0.05) + 1
    critic_predictions = torch.gather(raw, 1, critic_selected - 1)
    hysteretic_predictions = torch.gather(raw, 1, hysteretic_selected - 1)
    evaluation = TrajectoryEvaluation(
        example_metadata=tuple(
            {"counterfactual_group": str(index)} for index in range(len(gold))
        ),
        gold=gold,
        no_latch_predictions=raw,
        confidence_latch_predictions=critic_predictions,
        critic_latch_predictions=critic_predictions,
        hysteretic_critic_latch_predictions=hysteretic_predictions,
        confidence_selected_horizons=critic_selected,
        critic_selected_horizons=critic_selected,
        hysteretic_selected_horizons=hysteretic_selected,
        maximum_answer_probabilities=torch.zeros((4, 4)),
        confidence_scores=scores,
        critic_scores=scores,
    )
    diagnostics = trajectory_diagnostics([evaluation])
    assertions = trajectory_accounting_assertions(
        {"42": diagnostics, "pooled": diagnostics}, {42: evaluation}, 2, (4,)
    )
    assert assertions["all_pass"]
    assert (
        diagnostics["budgets_b1_b32"]["4"]["arms"]["no_latch"]["oracle_headroom"][
            "numerator"
        ]
        == 2
    )
    assert (
        assertions["no_latch_headroom_regression_checks"]["42"]["4"][
            "a_t_star_times_regression_numerator"
        ]
        == 2
    )
    return {
        "f_le_o_h_nonnegative": assertions,
        "synthetic_no_latch_b4_headroom_numerator": 2,
        "pass": True,
    }


def _decision_logic_self_test(config: Mapping[str, Any]) -> dict[str, Any]:
    validity = {"integrity": {"pass": True}}

    def endpoints(r16: float, r32: float, *, advantage: float = 0.04) -> dict[int, Any]:
        return {
            16: {
                scope: _synthetic_metrics(r16, 0.85, 0.95, advantage, 0.75)
                for scope in ("42", "31415", "pooled")
            },
            32: {
                scope: _synthetic_metrics(r32, 0.85, 0.95, advantage, 0.75)
                for scope in ("42", "31415", "pooled")
            },
        }

    at_16 = adjudicate(endpoints(0.10, 0.20), validity, config)
    assert at_16["final_token"] == "PROCEED" and at_16["selected_endpoint"] == 16
    at_32 = adjudicate(endpoints(0.05, 0.10), validity, config)
    assert at_32["final_token"] == "PROCEED" and at_32["selected_endpoint"] == 32
    regression_void = adjudicate(endpoints(0.05, 0.09), validity, config)
    assert regression_void["final_token"] == "VOID"
    assert regression_void["reason"] == "UNMEASURABLE_REGRESSION"
    measurable_failure = adjudicate(
        endpoints(0.05, 0.09, advantage=0.02), validity, config
    )
    assert measurable_failure["final_token"] == "FAIL"
    validity_void = adjudicate(
        endpoints(0.10, 0.20), {"integrity": {"pass": False}}, config
    )
    assert validity_void["final_token"] == "VOID"
    arm4_adversarial = endpoints(0.10, 0.20)
    for scopes in arm4_adversarial.values():
        for metrics in scopes.values():
            metrics["arm4_informational_adversarial_value"] = {
                "accuracy": -999.0,
                "regression": 999.0,
            }
    arm4_ignored = adjudicate(arm4_adversarial, validity, config)
    assert arm4_ignored == at_16
    return {
        "h16_precedence": at_16,
        "conditional_h32": at_32,
        "regression_only_void": regression_void,
        "independent_selector_failure": measurable_failure,
        "validity_precedence": validity_void,
        "arm4_cannot_change_proceed_precedence": arm4_ignored,
        "pass": True,
    }


def _generator_integrity_false_observation_self_test(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    synthetic_audit = {
        "skeleton_hash_disjoint": True,
        "name_combination_disjoint": True,
        "counterfactual_group_atomic": True,
        "answer_position_balanced_within_strata": True,
        "lure_mixture_exact": False,
    }
    gate = generator_integrity_gate(synthetic_audit)
    passing_endpoints = {
        endpoint: {
            scope: _synthetic_metrics(0.10, 0.85, 0.95, 0.04, 0.75)
            for scope in ("42", "31415", "pooled")
        }
        for endpoint in (16, 32)
    }
    decision = adjudicate(
        passing_endpoints,
        {"generator_and_split_integrity": gate},
        config,
    )
    assert gate["observed"] is False
    assert gate["pass"] is False
    assert decision["final_token"] == "VOID"
    assert decision["reason"] == "VALIDITY_GATE_FAILURE"
    return {
        "synthetic_false_observation": "lure_mixture_exact",
        "gate": gate,
        "decision": decision,
        "pass": True,
    }


def _vram_accounting_self_test() -> dict[str, Any]:
    phases = [
        {
            "phase": "training_and_selector_fitting",
            "model_seed": 42,
            "wall_time_seconds": 1.0,
            "peak_vram_allocated_bytes": 300,
            "peak_vram_reserved_bytes": 700,
        },
        {
            "phase": "official_test_evaluation",
            "model_seed": 31415,
            "wall_time_seconds": 1.0,
            "peak_vram_allocated_bytes": 500,
            "peak_vram_reserved_bytes": 600,
        },
    ]
    summary = summarize_vram_phases(phases)
    assert summary["peak_vram_allocated_bytes"] == 500
    assert summary["peak_vram_reserved_bytes"] == 700
    assert summary["vram_phases"] == phases
    return {**summary, "pass": True}


def _atomic_hard_crash_self_test() -> dict[str, Any]:
    probe_payload = {
        "final_token": "VOID",
        "probe": "atomic-hard-crash",
        "complete": True,
    }
    reports: list[dict[str, Any]] = []
    for window, expected_exit, expected_final in (
        ("after_fsync_before_link", 91, False),
        ("after_link_before_unlink", 92, True),
    ):
        with tempfile.TemporaryDirectory(prefix=f"e2-{window}-") as directory:
            probe_directory = Path(directory)
            final_path = probe_directory / f"{window}.json"
            preserved_final = probe_directory / "preexisting-final-artifact.json"
            preserved_payload = b'{"preserved": true}\n'
            preserved_final.write_bytes(preserved_payload)
            _assert_immutable_temp_namespace_is_disjoint(final_path)
            _assert_immutable_temp_namespace_is_disjoint(preserved_final)

            crash = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_atomic-crash-probe",
                    window,
                    "--_atomic-probe-directory",
                    str(probe_directory),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if crash.returncode != expected_exit:
                raise AssertionError(
                    f"{window} probe exited {crash.returncode}, expected {expected_exit}; "
                    f"stdout={crash.stdout!r}; stderr={crash.stderr!r}"
                )
            orphan_temps = list(probe_directory.glob(IMMUTABLE_RESULT_TEMP_GLOB))
            final_exists = final_path.exists()
            final_complete = (
                final_exists
                and json.loads(final_path.read_text(encoding="utf-8")) == probe_payload
            )
            if final_exists != expected_final:
                raise AssertionError(f"unexpected final-artifact state after {window}")
            if final_exists and not final_complete:
                raise AssertionError(f"partial final artifact after {window}")
            if len(orphan_temps) != 1:
                raise AssertionError(f"{window} did not leave exactly one crash temp")
            if preserved_final.read_bytes() != preserved_payload:
                raise AssertionError(
                    "crash probe changed a pre-existing final artifact"
                )

            next_startup = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_atomic-scavenge-probe",
                    "--_atomic-probe-directory",
                    str(probe_directory),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if next_startup.returncode != 0:
                raise AssertionError(
                    f"startup scavenger failed after {window}; "
                    f"stdout={next_startup.stdout!r}; stderr={next_startup.stderr!r}"
                )
            remaining_temps = list(probe_directory.glob(IMMUTABLE_RESULT_TEMP_GLOB))
            final_preserved = final_path.exists() == expected_final
            if final_path.exists():
                final_preserved = final_preserved and (
                    json.loads(final_path.read_text(encoding="utf-8")) == probe_payload
                )
            preexisting_final_preserved = (
                preserved_final.read_bytes() == preserved_payload
            )
            if remaining_temps:
                raise AssertionError(f"startup scavenger missed a temp after {window}")
            if not final_preserved or not preexisting_final_preserved:
                raise AssertionError("startup scavenger touched a final artifact")
            reports.append(
                {
                    "window": window,
                    "subprocess_exit_code": crash.returncode,
                    "post_crash": {
                        "final_exists": final_exists,
                        "final_complete": final_complete,
                        "orphan_temp_count": len(orphan_temps),
                    },
                    "after_next_startup": {
                        "final_exists": final_path.exists(),
                        "final_complete_or_absent": final_preserved,
                        "orphan_temp_count": len(remaining_temps),
                        "preexisting_final_preserved": (preexisting_final_preserved),
                    },
                }
            )
    return {
        "temp_pattern": IMMUTABLE_RESULT_TEMP_GLOB,
        "temp_pattern_disjoint_from_json_finals": True,
        "directory_fsync": {
            "supported": os.name != "nt",
            "platform": os.name,
            "windows_limitation": (
                "Python os.open/os.fsync cannot fsync directory handles on Windows"
                if os.name == "nt"
                else None
            ),
        },
        "crash_windows": reports,
        "pass": True,
    }


def _result_schema_and_atomic_write_self_test(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reason = "SELF_TEST_PRETEST_TERMINAL"
    phases = [
        {
            "phase": "training_and_selector_fitting",
            "model_seed": 42,
            "wall_time_seconds": 1.0,
            "peak_vram_allocated_bytes": 100,
            "peak_vram_reserved_bytes": 200,
        }
    ]
    training = {
        str(seed): {
            "selector_training_corpus": {"states": 20_000},
            "selector_calibration_corpus": {"states": 10_000},
            "confidence_calibrator_fit": {
                "fit_corpus_split": "selector_calibration",
                "frozen_before_external_scoring": True,
            },
            "confidence_score_provenance": {
                "calibrator_fit_split": "selector_calibration",
                "calibrator_frozen_before_scoring": True,
                "critic_pair_construction_scored_split": "selector_harvest",
            },
            "selector_provenance": {"all_gates_pass": False},
        }
        for seed in config["training"]["model_seeds"]
    }
    result = {
        "schema_version": "1.1.0",
        "experiment_id": config["experiment_id"],
        "started_utc": "2026-08-10T00:00:00+00:00",
        "completed_utc": "2026-08-10T00:00:01+00:00",
        "preregistration": config["preregistration"],
        "review_attestation": "INDEPENDENT_PRETRAINING_REVIEW_CLEAN",
        "hashes": {"self_test": True},
        "parameter_counts": {"self_test": True},
        "feature_boundary_audit": {"pass": True},
        "generator_audit": {"pass": True},
        "dataset_token_audit": {"self_test": True},
        "training": training,
        "calibration_frozen_t_star": {"selected_t_star": 1},
        "calibration_frozen_hysteresis_by_seed": {
            str(seed): {"selected_delta": 0.0}
            for seed in config["training"]["model_seeds"]
        },
        "arm_definitions": {"4": "hysteretic_critic_latch_informational_only"},
        "evaluation": pretest_terminal_evaluation(reason, config),
        "validity_gates": {
            "self_test_pretest_terminal": {"observed": True, "pass": True}
        },
        "decision": {
            "final_token": "VOID",
            "reason": reason,
            "official_test_inspected": False,
        },
        "compute": {"wall_time_seconds": 1.0, **summarize_vram_phases(phases)},
        "per_example_records": pretest_terminal_per_example_records(reason),
        "final_token": "VOID",
    }
    validate_result_schema(result, config)
    seed_keys = [str(seed) for seed in config["training"]["model_seeds"]]
    scopes = [*seed_keys, "pooled"]
    complete = json.loads(json.dumps(result))
    complete["decision"] = {
        "final_token": "FAIL",
        "reason": "SELF_TEST_COMPLETE_TERMINAL",
        "official_test_inspected": True,
    }
    complete["final_token"] = "FAIL"
    complete["evaluation"] = {
        "status": "complete",
        "official_test_inspected": True,
        "horizons": list(range(1, 33)),
        "encoder_control": {},
        "compute_by_seed": {},
        "accuracy_grid": {
            seed_key: {"hysteretic_critic_latch_informational": {}}
            for seed_key in seed_keys
        },
        "stratified_accuracy_grid": {},
        "trajectory_diagnostics": {
            scope: {
                "selector_switch_hazards": {
                    "hysteretic_critic_latch_informational": {}
                }
            }
            for scope in scopes
        },
        "trajectory_accounting_assertions": {},
        "endpoint_metrics": {
            endpoint: {
                scope: {
                    "hysteretic_critic_latch_endpoint_accuracy_informational": {}
                }
                for scope in scopes
            }
            for endpoint in ("16", "32")
        },
        "paired_counterfactual_group_bootstrap": {},
    }
    complete["per_example_records"] = {
        seed_key: [{"hysteretic_selected_horizons_b1_b32": [1] * 32}]
        for seed_key in seed_keys
    }
    validate_result_schema(complete, config)
    invalid = json.loads(json.dumps(result))
    invalid["evaluation"]["endpoint_metrics"] = None
    invalid_schema_rejected = False
    try:
        validate_result_schema(invalid, config)
    except ValueError:
        invalid_schema_rejected = True
    if not invalid_schema_rejected:
        raise AssertionError("invalid schema 1.1.0 result was accepted")

    with tempfile.TemporaryDirectory(prefix="e2-result-self-test-") as directory:
        path = Path(directory) / "result.json"
        write_immutable_result(result, path)
        installed = json.loads(path.read_text(encoding="utf-8"))
        no_clobber_rejected = False
        try:
            write_immutable_result(result, path)
        except RuntimeError:
            no_clobber_rejected = True
        if not no_clobber_rejected:
            raise AssertionError("immutable result overwrite was accepted")
        orphan_temps = list(path.parent.glob(IMMUTABLE_RESULT_TEMP_GLOB))
    assert installed == result
    assert not orphan_temps
    return {
        "schema_1_1_0_validated": True,
        "complete_terminal_variant_validated": True,
        "invalid_schema_rejected": invalid_schema_rejected,
        "atomic_no_clobber_rejected_overwrite": no_clobber_rejected,
        "installed_payload_complete": installed == result,
        "orphan_temporary_files": len(orphan_temps),
        "pass": True,
    }


def run_self_tests(config_path: Path) -> int:
    config = load_config(config_path)
    torch.set_num_threads(int(config["self_test"]["torch_threads"]))
    counts = {
        "controller_train": int(config["self_test"]["examples_per_public_split"]),
        "selector_harvest": int(config["self_test"]["examples_per_public_split"]),
        "selector_calibration": int(config["self_test"]["examples_per_public_split"]),
        "test": int(config["self_test"]["test_examples"]),
    }
    generator = DeductionGenerator(config)
    first = generator.generate_dataset(counts, seed_offset=1)
    second = generator.generate_dataset(counts, seed_offset=1)
    first_hash = dataset_content_hash(first)
    second_hash = dataset_content_hash(second)
    if first_hash != second_hash:
        raise AssertionError("generator is not deterministic")
    generator_audit = generator.audit_dataset(first)
    tokenizer = LocalTokenizer(config)
    sample_texts = [example.rendered_text for example in first["test"][:8]]
    tokenizer_round_trip = all(
        tokenizer.round_trip_tokens(text) for text in sample_texts
    )
    if not tokenizer_round_trip:
        raise AssertionError("tokenizer token round-trip failed")
    token_schedule, token_lengths = exact_token_schedule(
        first["controller_train"],
        tokenizer,
        10_000,
        int(config["self_test"]["seed"]),
    )
    exact_scheduled_tokens = sum(token_lengths[index] for index in token_schedule)
    if exact_scheduled_tokens != 10_000:
        raise AssertionError("exact token-budget scheduler failed")

    torch.manual_seed(int(config["self_test"]["seed"]))
    common = CommonRecurrentModel(config, len(tokenizer.id_to_token)).eval()
    critic = LatentProgressCritic(config).eval()
    calibrator = ConfidenceCalibrator().eval()
    encoder = EncoderOnlyControl(config, len(tokenizer.id_to_token)).eval()
    counts_parameters = {
        "common_recurrent": trainable_parameter_count(common),
        "latent_critic": trainable_parameter_count(critic),
        "encoder_only_control": trainable_parameter_count(encoder),
        "confidence_calibrator": trainable_parameter_count(calibrator),
    }
    common_range = config["common_model"]["target_parameter_range"]
    critic_range = config["critic"]["target_parameter_range"]
    assert common_range[0] <= counts_parameters["common_recurrent"] <= common_range[1]
    assert critic_range[0] <= counts_parameters["latent_critic"] <= critic_range[1]
    relative_difference = (
        abs(
            counts_parameters["encoder_only_control"]
            - counts_parameters["common_recurrent"]
        )
        / counts_parameters["common_recurrent"]
    )
    assert relative_difference <= float(
        config["encoder_control"]["parameter_match_relative_tolerance"]
    )

    toy = first["test"][:2]
    tokens, mask, labels, token_count = collate_examples(
        toy, tokenizer, torch.device("cpu")
    )
    with torch.inference_mode():
        logits, states, prompt = common(tokens, mask, horizon=2)
        critic_feature_grid = critic.features_for_trajectory(states, prompt)
        critic_logits = critic(critic_feature_grid)
        confidence_grid = torch.stack(
            [
                confidence_features(common.logits_from_state(state), index + 1)
                for index, state in enumerate(states)
            ],
            dim=1,
        )
        confidence_logits = calibrator(confidence_grid)
        encoder_logits = encoder(tokens, mask)
        t32_states = common.recurrent_states(
            *common.encode_context(tokens, mask), mask, 32
        )
        t32_logits = torch.stack(
            [common.logits_from_state(state) for state in t32_states], dim=1
        )
        t32_confidence_features = confidence_features(t32_logits[:, -1], 32)
        t32_critic_features = critic.features_for_trajectory(t32_states, prompt)
    assert logits.shape == (2, 4)
    assert len(states) == 2 and states[0].shape == (2, 8, 512)
    assert critic_feature_grid.shape == (2, 2, 1540)
    assert critic_logits.shape == (2, 2)
    assert confidence_logits.shape == (2, 2)
    assert encoder_logits.shape == (2, 4)
    assert t32_logits.shape == (2, 32, 4)
    assert torch.all(t32_confidence_features[:, -1] == 2.0)
    assert torch.all(t32_critic_features[:, -1, -1] == 2.0)
    assert labels.shape == (2,)
    feature_audit = feature_boundary_audit(config)
    assert feature_audit["pass"]

    report = {
        "status": "ok",
        "cpu_only": True,
        "generator_determinism": {
            "first_sha256": first_hash,
            "second_sha256": second_hash,
            "pass": first_hash == second_hash,
        },
        "symbolic_verifier": _hand_checked_symbolic_tests(),
        "split_disjointness": generator_audit,
        "tokenizer": {
            "vocabulary_size": len(tokenizer.id_to_token),
            "vocabulary_sha256": tokenizer.vocabulary_hash(),
            "round_trip_examples_numerator": len(sample_texts),
            "round_trip_examples_denominator": len(sample_texts),
            "pass": tokenizer_round_trip,
        },
        "token_accounting": {
            "scheduled_examples": len(token_schedule),
            "observed_nonpadding_input_plus_answer_tokens": exact_scheduled_tokens,
            "required_nonpadding_input_plus_answer_tokens": 10000,
            "pass": exact_scheduled_tokens == 10000,
        },
        "parameter_counts": {
            **counts_parameters,
            "encoder_relative_difference": relative_difference,
            "common_target_range": common_range,
            "critic_target_range": critic_range,
            "encoder_relative_tolerance": config["encoder_control"][
                "parameter_match_relative_tolerance"
            ],
            "pass": True,
        },
        "toy_forward_passes": {
            "batch_examples": len(toy),
            "nonpadding_input_plus_answer_tokens": token_count,
            "common_logits_shape": list(logits.shape),
            "recurrent_state_shapes": [list(state.shape) for state in states],
            "critic_feature_shape": list(critic_feature_grid.shape),
            "critic_score_shape": list(critic_logits.shape),
            "confidence_score_shape": list(confidence_logits.shape),
            "encoder_logits_shape": list(encoder_logits.shape),
            "t1_t32_logits_shape": list(t32_logits.shape),
            "t32_confidence_step_coordinate": float(t32_confidence_features[0, -1]),
            "t32_critic_step_coordinate": float(t32_critic_features[0, -1, -1]),
            "pass": True,
        },
        "feature_boundary": feature_audit,
        "resume_determinism": _resume_determinism_self_test(),
        "hysteresis_rule": _hysteresis_rule_self_test(),
        "hysteresis_delta_selection": _delta_selection_self_test(config),
        "trajectory_diagnostics": _diagnostics_self_test(),
        "decision_logic": _decision_logic_self_test(config),
        "generator_integrity_false_observation": (
            _generator_integrity_false_observation_self_test(config)
        ),
        "run_level_vram_accounting": _vram_accounting_self_test(),
        "atomic_hard_crash": _atomic_hard_crash_self_test(),
        "result_schema_and_atomic_write": (
            _result_schema_and_atomic_write_self_test(config)
        ),
        "training_or_gpu_executed": False,
    }
    print(json.dumps(report, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test", action="store_true", help="run CPU-only structural tests"
    )
    mode.add_argument(
        "--run", action="store_true", help="launch the reviewed CUDA pilot"
    )
    mode.add_argument(
        "--_atomic-crash-probe",
        choices=ATOMIC_CRASH_WINDOWS,
        help=argparse.SUPPRESS,
    )
    mode.add_argument(
        "--_atomic-scavenge-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_atomic-probe-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--review-attestation",
        default="",
        help="exact pre-training review attestation required by --run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args._atomic_crash_probe or args._atomic_scavenge_probe:
        if args._atomic_probe_directory is None:
            raise ValueError("internal atomic probe requires a probe directory")
        probe_directory = args._atomic_probe_directory.resolve()
        probe_name = args._atomic_crash_probe or "startup-scavenger"
        probe_path = probe_directory / f"{probe_name}.json"
        scavenge_stale_immutable_result_temps(probe_directory, final_path=probe_path)
        if args._atomic_scavenge_probe:
            return 0
        probe_payload = {
            "final_token": "VOID",
            "probe": "atomic-hard-crash",
            "complete": True,
        }
        write_immutable_result(
            probe_payload,
            probe_path,
            _crash_probe=args._atomic_crash_probe,
        )
        raise AssertionError("atomic crash probe returned instead of exiting")
    if args._atomic_probe_directory is not None:
        raise ValueError("probe directory is valid only for internal atomic probes")
    scavenge_stale_immutable_result_temps(RESULT_PATH.parent, final_path=RESULT_PATH)
    if args.self_test:
        if args.review_attestation:
            raise ValueError("self-tests do not accept a launch attestation")
        return run_self_tests(args.config.resolve())
    return run_pilot(args.config.resolve(), args.review_attestation)


if __name__ == "__main__":
    raise SystemExit(main())
