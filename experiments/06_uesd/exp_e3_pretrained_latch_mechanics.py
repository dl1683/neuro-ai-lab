"""E3 frozen-pretrained-substrate latch-mechanics runner.

The only launchable path in this build is ``--preflight``: the registered
outcome-blind 50-update gradient probe followed, on PASS only, by the single
500-update competence smoke. Canonical retained training is deliberately
blocked at the independent-review boundary. ``--self-test-fast`` is CPU-only.

Tracked output never contains private model coordinates. They are resolved
only from the ignored line-06 manifest, and model loading is offline from the
workspace cache.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_CONFIG = HERE / "exp_e3_pretrained_latch_mechanics_config.json"
RESULT_PATH = HERE / "results" / "exp_e3_preflight.json"
LOCAL_MANIFEST_PATH = HERE / "_local_manifest.md"
E2_RUNNER_PATH = HERE / "exp_e2_latch_mechanics.py"
E3_REGISTRATION_PATH = HERE / "E3_PRETRAINED_MECHANICS_PREREGISTRATION.md"
WORKSPACE_HF_HOME = REPO_ROOT / ".hf_cache"

# Set the cache boundary before importing Transformers or any E2 module.
os.environ["HF_HOME"] = str(WORKSPACE_HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

from exp_e1_task_band import (  # noqa: E402
    load_private_model_coordinates,
    snapshot_content_digest,
    validate_local_checkpoint_snapshot,
)
from exp_e2_latch_mechanics import (  # noqa: E402
    DeductionExample,
    DeductionGenerator,
    SelectorCorpus,
    binary_auroc,
    confidence_matched_concordance,
    construct_within_problem_pairs,
    cumulative_first_argmax,
    dataset_content_hash,
    hysteretic_incumbent_indices,
    select_hysteresis_delta,
    selector_provenance_metrics,
    selector_switch_hazards,
    sha256_json,
    template_inventory_hash,
    trajectory_accounting_assertions,
    trajectory_diagnostics,
)

PREFLIGHT_FINAL_TOKENS = ("PASS", "PREFLIGHT_STOP", "VOID_NO_ROUTE")
E3_SPLITS = (
    "controller_train",
    "selector_harvest",
    "selector_calibration",
    "test",
    "instrument_probe_train",
    "competence_smoke_train",
    "competence_smoke_validation",
)
FORBIDDEN_COHORTS = (
    "line_07_calibration",
    "line_07_test",
    "gsm8k_official_test",
    "svamp_official_test",
    "e2_test",
)
TEMP_PREFIX = ".exp-e3-preflight-"
TEMP_SUFFIX = ".immutable-result.tmp"
TEMP_GLOB = f"{TEMP_PREFIX}*{TEMP_SUFFIX}"
PROMPT_LABELS = ("1", "2", "3", "4")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "1.0.0":
        raise ValueError("E3 config schema version changed")
    if config.get("experiment_id") != "exp_e3_pretrained_latch_mechanics":
        raise ValueError("wrong E3 experiment id")
    if config.get("preflight_id") != "exp_e3_preflight":
        raise ValueError("wrong E3 preflight id")
    if config["substrate"]["alias"] != "base-B":
        raise ValueError("E3 substrate alias must remain base-B")
    if int(config["substrate"]["expected_hidden_width"]) != 1536:
        raise ValueError("base-B hidden width binding changed")
    counts = config["generator"]["examples_per_split"]
    if tuple(counts) != E3_SPLITS or any(int(counts[name]) % 4 for name in E3_SPLITS):
        raise ValueError("E3 split names/order/count divisibility changed")
    if int(counts["instrument_probe_train"]) < 400:
        raise ValueError("instrument split cannot supply 50 disjoint batches")
    smoke = config["competence_smoke"]
    if (
        int(smoke["train_examples"]) != 2048
        or int(smoke["validation_examples"]) != 512
        or int(smoke["batch_size"]) != 8
        or int(smoke["updates"]) != 500
        or tuple(smoke["horizon_schedule"]) != (1, 2, 4)
        or int(smoke["warmup_updates"]) != 50
        or int(smoke["validation_horizon"]) != 4
        or int(smoke["pass_correct_minimum"]) != 205
        or int(smoke["wall_time_cap_seconds"]) != 900
    ):
        raise ValueError("competence-smoke contract changed")
    probe = config["instrument_probe"]
    if (
        tuple(probe["initialization_seeds"]) != (73001, 73002)
        or int(probe["updates_per_initialization"]) != 25
        or int(probe["total_updates"]) != 50
        or float(probe["provisional_safety_clip"]) != 16.0
    ):
        raise ValueError("instrument-probe contract changed")
    training = config["training"]
    if (
        tuple(training["model_seeds"]) != (42, 31415)
        or int(training["batch_size"]) != 8
        or int(training["controller_updates"]) != 2500
        or int(training["example_exposures"]) != 20000
        or tuple(training["train_horizons"]) != (1, 2, 4)
        or float(training["peak_learning_rate"]) != 3e-4
        or tuple(training["betas"]) != (0.9, 0.95)
        or float(training["epsilon"]) != 1e-8
        or float(training["weight_decay"]) != 0.01
        or int(training["warmup_updates"]) != 500
        or int(training["plateau_end_update"]) != 2000
        or training["controller_gradient_clip_norm"]
        != "INSTRUMENT_DERIVED_FROM_IMMUTABLE_PREFLIGHT"
    ):
        raise ValueError("canonical optimizer/accounting contract changed")
    if tuple(config["evaluation"]["horizons"]) != tuple(range(1, 33)):
        raise ValueError("evaluation horizons must remain 1..32")
    if tuple(config["evaluation"]["t_star_candidates"]) != tuple(range(1, 16)):
        raise ValueError("t-star candidates must remain 1..15")
    if config["result_artifact"]["write_mode"] != (
        "same_directory_fsync_atomic_no_clobber"
    ):
        raise ValueError("immutable publication contract changed")
    if tuple(config["result_artifact"]["preflight_final_tokens"]) != (
        PREFLIGHT_FINAL_TOKENS
    ):
        raise ValueError("preflight final tokens changed")
    if template_inventory_hash() != config["generator"]["template_inventory_sha256"]:
        raise ValueError("E2 template inventory hash changed")


def configure_runtime(seed: int) -> None:
    transformers_logging.set_verbosity_error()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _namespace_owner(value: str) -> str:
    bucket = int(sha256_bytes(f"e3-owner-v1:{value}".encode("utf-8"))[:16], 16)
    return E3_SPLITS[bucket % len(E3_SPLITS)]


class E3DeductionGenerator(DeductionGenerator):
    """E2 verified construction with seven disjoint E3 namespace owners."""

    def _build_name_pools(self) -> dict[str, dict[str, list[str]]]:
        pools: dict[str, dict[str, list[str]]] = {
            split: defaultdict(list) for split in E3_SPLITS
        }
        for first in self.syllables:
            for second in self.syllables:
                if first == second:
                    continue
                owner = _namespace_owner(f"name:{first}-{second}")
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
        if tuple(counts) != E3_SPLITS:
            raise ValueError("all and only the seven frozen E3 namespaces are required")
        datasets: dict[str, list[DeductionExample]] = {}
        used_hashes: set[str] = set()
        for split_index, split in enumerate(E3_SPLITS):
            datasets[split] = self._generate_e3_split(
                split,
                int(counts[split]),
                used_hashes,
                seed_offset=seed_offset + split_index * 100_003,
            )
            used_hashes.update(row.skeleton_hash for row in datasets[split])
        self.audit_dataset(datasets)
        return datasets

    def _generate_e3_split(
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
                skeleton = candidate[0].skeleton_hash
                if skeleton in local_hashes or skeleton in globally_used_hashes:
                    continue
                self._verify_pair(candidate)
                examples.extend(candidate)
                local_hashes.add(skeleton)
                break
            else:
                raise RuntimeError(f"could not generate valid E3 pair for {split}")
        examples.sort(key=lambda row: row.example_id)
        return examples


def rendered_prompt(example: DeductionExample, config: Mapping[str, Any]) -> str:
    return str(config["generator"]["prompt_template"]).format(
        deduction=example.rendered_text
    )


def split_hash(examples: Sequence[DeductionExample], config: Mapping[str, Any]) -> str:
    return sha256_json(
        [
            {
                "id": row.example_id,
                "counterfactual_group": row.counterfactual_group,
                "skeleton": row.skeleton_hash,
                "prompt": rendered_prompt(row, config),
                "gold": row.answer_position,
            }
            for row in examples
        ]
    )


@dataclass(frozen=True)
class FrozenExampleRepresentation:
    example_id: str
    token_states: Tensor
    frozen_prompt: Tensor
    label: int
    base_tokens: int

    @property
    def cache_bytes(self) -> int:
        return (
            self.token_states.numel() * self.token_states.element_size()
            + self.frozen_prompt.numel() * self.frozen_prompt.element_size()
        )


def resolve_and_validate_substrate(config: Mapping[str, Any]) -> dict[str, Any]:
    alias = str(config["substrate"]["alias"])
    identifier, revision, content_digest = load_private_model_coordinates(alias)
    validate_local_checkpoint_snapshot(identifier, revision, content_digest, alias)
    cache_key = f"models--{identifier.replace('/', '--')}"
    snapshot = WORKSPACE_HF_HOME / "hub" / cache_key / "snapshots" / revision
    if snapshot_content_digest(snapshot) != content_digest:
        raise RuntimeError("base-B local content digest changed after validation")
    return {
        "identifier": identifier,
        "revision": revision,
        "content_digest": content_digest,
        "snapshot": snapshot,
    }


def validate_pending_manifest_slot(
    config_path: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    text = LOCAL_MANIFEST_PATH.read_text(encoding="utf-8")

    def value(key: str) -> str:
        matches = re.findall(
            rf"^-\s*`{re.escape(key)}`:\s*`([^`]+)`\s*$",
            text,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise RuntimeError(f"local manifest requires exactly one {key} slot")
        return matches[0]

    observed = {
        "runner_sha256": value("E3-runner-sha256"),
        "config_sha256": value("E3-config-sha256"),
        "controller_gradient_clip_norm": value(
            "E3-controller-gradient-clip-norm"
        ),
        "clip_mapping": value("E3-clip-mapping"),
    }
    expected = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "controller_gradient_clip_norm": "PENDING_INSTRUMENT_PROBE",
        "clip_mapping": str(config["instrument_probe"]["mapping"]),
    }
    if observed != expected:
        raise RuntimeError("E3 pending manifest slot does not match runner/config/mapping")
    return observed


def load_frozen_substrate(coordinates: Mapping[str, Any], device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        coordinates["identifier"],
        revision=coordinates["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("base-B tokenizer lacks both pad and EOS tokens")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        coordinates["identifier"],
        revision=coordinates["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("frozen substrate exposes trainable parameters")
    return tokenizer, model


@torch.inference_mode()
def build_frozen_representations(
    examples: Sequence[DeductionExample],
    config: Mapping[str, Any],
    tokenizer,
    substrate,
    device: torch.device,
    *,
    batch_size: int = 8,
) -> tuple[list[FrozenExampleRepresentation], dict[str, int]]:
    records: list[FrozenExampleRepresentation] = []
    base_token_count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [rendered_prompt(row, config) for row in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        context_limit = int(getattr(substrate.config, "max_position_embeddings", 0))
        if context_limit and input_ids.shape[1] > context_limit:
            raise RuntimeError("E3 prompt exceeds frozen substrate context window")
        outputs = substrate.model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if hidden.shape[-1] != int(config["substrate"]["expected_hidden_width"]):
            raise RuntimeError("frozen representation width mismatch")
        for row_index, example in enumerate(batch):
            length = int(attention[row_index].sum().item())
            states = hidden[row_index, :length].detach().cpu().contiguous()
            prompt = states.float().mean(dim=0).to(torch.bfloat16)
            records.append(
                FrozenExampleRepresentation(
                    example_id=example.example_id,
                    token_states=states,
                    frozen_prompt=prompt,
                    label=example.answer_position,
                    base_tokens=length,
                )
            )
            base_token_count += length
    cache_bytes = sum(row.cache_bytes for row in records)
    if cache_bytes > int(config["substrate"]["maximum_cache_bytes"]):
        raise RuntimeError("frozen representation cache exceeds 32 GiB")
    return records, {
        "examples": len(records),
        "generated_base_tokens": base_token_count,
        "cache_bytes": cache_bytes,
    }


def collate_representations(
    records: Sequence[FrozenExampleRepresentation], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    maximum = max(row.token_states.shape[0] for row in records)
    width = records[0].token_states.shape[1]
    states = torch.zeros(
        len(records), maximum, width, dtype=torch.bfloat16, device=device
    )
    padding_mask = torch.ones(len(records), maximum, dtype=torch.bool, device=device)
    for index, row in enumerate(records):
        length = row.token_states.shape[0]
        states[index, :length].copy_(row.token_states.to(device))
        padding_mask[index, :length] = False
    prompt = torch.stack([row.frozen_prompt for row in records]).to(device)
    labels = torch.tensor([row.label for row in records], dtype=torch.long, device=device)
    return states, padding_mask, prompt, labels


def _decoder_layer(width: int, heads: int, ffn: int, dropout: float):
    return nn.TransformerDecoderLayer(
        width,
        heads,
        ffn,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class E3RecurrentController(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = config["common_model"]
        substrate_width = int(cfg["substrate_width"])
        width = int(cfg["width"])
        heads = int(cfg["attention_heads"])
        ffn = int(cfg["ffn_width"])
        dropout = float(cfg["dropout"])
        self.width = width
        self.projection = nn.Linear(substrate_width, width)
        self.projection_norm = nn.LayerNorm(width)
        self.plan_slots = nn.Parameter(
            torch.randn(int(cfg["latent_plan_slots"]), width) * 0.02
        )
        self.prompt_to_plan = nn.Linear(width, width)
        self.controller_layers = nn.ModuleList(
            _decoder_layer(width, heads, ffn, dropout)
            for _ in range(int(cfg["controller_layers"]))
        )
        self.controller_norm = nn.LayerNorm(width)
        self.answer_decoder = _decoder_layer(width, heads, ffn, dropout)
        self.answer_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.answer_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, int(cfg["answer_choices"]))

    def encode_context(
        self, frozen_states: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        context = self.projection_norm(self.projection(frozen_states))
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
        outputs: list[Tensor] = []
        for _ in range(max_horizon):
            for layer in self.controller_layers:
                plan = layer(plan, context, memory_key_padding_mask=padding_mask)
            plan = self.controller_norm(plan)
            outputs.append(plan)
        return outputs

    def logits_from_state(self, state: Tensor) -> Tensor:
        query = self.answer_query.unsqueeze(0).expand(state.shape[0], -1, -1)
        decoded = self.answer_decoder(query, state)
        return self.readout(self.answer_norm(decoded[:, 0]))

    def forward(
        self, frozen_states: Tensor, padding_mask: Tensor, horizon: int
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        context, prompt = self.encode_context(frozen_states, padding_mask)
        states = self.recurrent_states(context, prompt, padding_mask, horizon)
        return self.logits_from_state(states[-1]), states, prompt


class E3NonRecurrentControl(E3RecurrentController):
    """Parameter-identical fixed-depth control with no recurrent reuse."""

    def recurrent_states(
        self,
        context: Tensor,
        prompt: Tensor,
        padding_mask: Tensor,
        max_horizon: int = 1,
    ) -> list[Tensor]:
        del max_horizon
        plan = self.plan_slots.unsqueeze(0).expand(context.shape[0], -1, -1)
        plan = plan + self.prompt_to_plan(prompt).unsqueeze(1)
        for layer in self.controller_layers:
            plan = layer(plan, context, memory_key_padding_mask=padding_mask)
        return [self.controller_norm(plan)]


class E3LatentProgressCritic(nn.Module):
    ALLOWED_FEATURE_NAMES = (
        "prompt_conditioned_pooled_latent",
        "prompt_conditioned_pooled_update",
        "frozen_prompt_representation",
        "update_norm",
        "consecutive_update_cosine",
        "cross_horizon_latent_agreement",
        "t_over_16",
    )
    FORBIDDEN_FEATURE_NAMES = (
        "raw_answer_logits",
        "answer_probabilities",
        "top_two_margin",
        "entropy",
        "answer_identity",
        "sampled_answer_frequency",
        "response_log_probability",
    )

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        ccfg = config["common_model"]
        cfg = config["critic"]
        width = int(ccfg["width"])
        prompt_width = int(cfg["frozen_prompt_projection_width"])
        self.width = width
        self.frozen_prompt_adapter = nn.Linear(
            int(ccfg["substrate_width"]), prompt_width
        )
        self.input_width = width * 2 + prompt_width + 4
        first, second = (int(value) for value in cfg["mlp_widths"])
        self.input_norm = nn.LayerNorm(self.input_width)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_width, first),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(second, 1),
        )

    def _pool(self, state: Tensor, prompt: Tensor) -> Tensor:
        weights = torch.softmax(
            (state * prompt.unsqueeze(1)).sum(-1) / math.sqrt(self.width), dim=1
        )
        return (weights.unsqueeze(-1) * state).sum(1)

    def features_for_trajectory(
        self,
        states: Sequence[Tensor],
        prompt: Tensor,
        frozen_prompt: Tensor,
    ) -> Tensor:
        pooled = [self._pool(state, prompt) for state in states]
        frozen = self.frozen_prompt_adapter(frozen_prompt)
        zeros = torch.zeros_like(pooled[0])
        rows: list[Tensor] = []
        for index, current in enumerate(pooled):
            previous = pooled[index - 1] if index else zeros
            delta = current - previous
            previous_delta = pooled[index - 1] - pooled[index - 2] if index >= 2 else zeros
            norm = delta.norm(dim=-1, keepdim=True)
            cosine = F.cosine_similarity(delta, previous_delta, dim=-1, eps=1e-8)
            cosine = cosine.unsqueeze(-1)
            if index:
                prior = torch.stack(pooled[:index]).mean(0)
                agreement = F.cosine_similarity(
                    current, prior, dim=-1, eps=1e-8
                ).unsqueeze(-1)
            else:
                agreement = torch.zeros_like(norm)
            step = torch.full_like(norm, (index + 1) / 16.0)
            row = torch.cat(
                [current, delta, frozen, norm, cosine, agreement, step], dim=-1
            )
            if row.shape[-1] != self.input_width:
                raise AssertionError("latent critic feature width mismatch")
            rows.append(row)
        return torch.stack(rows, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(self.input_norm(features)).squeeze(-1)


def parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_boundary(model: nn.Module) -> dict[str, Any]:
    rows = [
        {"name": name, "shape": list(parameter.shape), "count": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_parameters": sum(int(row["count"]) for row in rows),
        "tensors": len(rows),
        "sha256": sha256_json(rows),
        "names_and_shapes": rows,
    }


def validate_parameter_boundaries(
    model: E3RecurrentController,
    critic: E3LatentProgressCritic,
    control: E3NonRecurrentControl,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    common = parameter_count(model)
    critic_count = parameter_count(critic)
    control_count = parameter_count(control)
    if common > int(config["common_model"]["target_parameter_maximum"]):
        raise ValueError("E3 common model exceeds 30M trainable parameters")
    critic_low, critic_high = config["critic"]["target_parameter_range"]
    if not int(critic_low) <= critic_count <= int(critic_high):
        raise ValueError("E3 critic is outside the registered parameter range")
    relative = abs(control_count - common) / common
    if relative > float(
        config["nonrecurrent_control"]["parameter_match_relative_tolerance"]
    ):
        raise ValueError("E3 non-recurrent control is not parameter matched")
    return {
        "common_trainable": common,
        "critic_trainable": critic_count,
        "nonrecurrent_control_trainable": control_count,
        "control_relative_difference": relative,
        "common_maximum": int(config["common_model"]["target_parameter_maximum"]),
        "critic_range": [int(critic_low), int(critic_high)],
        "all_pass": True,
    }


def optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decayed: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if name.endswith("bias") or "norm" in lowered or "plan_slots" in lowered:
            excluded.append(parameter)
        else:
            decayed.append(parameter)
    if not decayed or not excluded:
        raise AssertionError("AdamW exclusion groups are incomplete")
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": excluded, "weight_decay": 0.0},
    ]


def make_optimizer(
    model: nn.Module, config: Mapping[str, Any]
) -> torch.optim.AdamW:
    cfg = config["training"]
    return torch.optim.AdamW(
        optimizer_groups(model, float(cfg["weight_decay"])),
        lr=float(cfg["peak_learning_rate"]),
        betas=tuple(float(value) for value in cfg["betas"]),
        eps=float(cfg["epsilon"]),
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def canonical_learning_rate(update: int, config: Mapping[str, Any]) -> float:
    cfg = config["training"]
    peak = float(cfg["peak_learning_rate"])
    warmup = int(cfg["warmup_updates"])
    plateau = int(cfg["plateau_end_update"])
    total = int(cfg["controller_updates"])
    minimum = float(cfg["minimum_learning_rate"])
    if update <= warmup:
        return peak * update / warmup
    if update <= plateau:
        return peak
    progress = (update - plateau) / (total - plateau)
    return peak + progress * (minimum - peak)


@contextlib.contextmanager
def training_autocast(device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def release_cuda(*objects: Any) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fixed_index_batches(
    size: int, batch_size: int, updates: int, seed: int
) -> Iterator[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    produced = 0
    while produced < updates:
        order = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size, batch_size):
            if produced >= updates:
                return
            batch = order[start : start + batch_size]
            if len(batch) < batch_size:
                continue
            produced += 1
            yield batch


def run_gradient_probe(
    examples: Sequence[DeductionExample],
    config: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    cfg = config["instrument_probe"]
    seeds = tuple(int(seed) for seed in cfg["initialization_seeds"])
    observations: list[float] = []
    phases: list[dict[str, Any]] = []
    boundary_hash: str | None = None
    for seed_index, seed in enumerate(seeds):
        configure_runtime(seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        phase_start = time.perf_counter()
        tokenizer, substrate = load_frozen_substrate(coordinates, device)
        subset_start = seed_index * 200
        subset = examples[subset_start : subset_start + 200]
        representations, representation_compute = build_frozen_representations(
            subset, config, tokenizer, substrate, device
        )
        frozen_parameter_count = parameter_count(substrate, trainable_only=False)
        del tokenizer, substrate
        release_cuda()
        # Bind the throwaway controller initialization directly to its seed,
        # independent of any RNG use inside substrate construction.
        configure_runtime(seed)
        model = E3RecurrentController(config).to(device)
        boundary = parameter_boundary(model)
        if boundary_hash is None:
            boundary_hash = str(boundary["sha256"])
        elif boundary_hash != boundary["sha256"]:
            raise RuntimeError("throwaway trainable parameter boundaries differ")
        optimizer = make_optimizer(model, config)
        model.train()
        for local_update in range(1, int(cfg["updates_per_initialization"]) + 1):
            batch_start = (local_update - 1) * int(config["training"]["batch_size"])
            batch = representations[
                batch_start : batch_start + int(config["training"]["batch_size"])
            ]
            states, mask, _frozen_prompt, labels = collate_representations(batch, device)
            optimizer.zero_grad(set_to_none=True)
            set_learning_rate(optimizer, canonical_learning_rate(local_update, config))
            horizon = int(config["training"]["train_horizons"][(local_update - 1) % 3])
            with training_autocast(device):
                logits, _, _ = model(states, mask, horizon)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["provisional_safety_clip"])
            )
            raw = float(norm.detach().cpu())
            if not math.isfinite(raw) or raw <= 0:
                raise RuntimeError("instrument probe produced a nonfinite/nonpositive norm")
            observations.append(raw)
            optimizer.step()
        phase_wall = time.perf_counter() - phase_start
        phases.append(
            {
                "phase": "instrument_probe_throwaway_initialization",
                "seed": seed,
                "updates": int(cfg["updates_per_initialization"]),
                "wall_time_seconds": phase_wall,
                "peak_vram_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                )
                if device.type == "cuda"
                else 0,
                "peak_vram_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                )
                if device.type == "cuda"
                else 0,
                "frozen_base_parameters": frozen_parameter_count,
                **representation_compute,
            }
        )
        del model, optimizer, representations
        release_cuda()
    if len(observations) != 50:
        raise RuntimeError("instrument probe did not produce exactly 50 norms")
    sorted_values = sorted(observations)
    median = (sorted_values[24] + sorted_values[25]) / 2.0
    clip = min(16, max(2, math.floor(median + 0.5)))
    tensor = torch.tensor(observations, dtype=torch.float64)
    quantiles = {
        name: float(torch.quantile(tensor, q, interpolation="linear").item())
        for name, q in (
            ("minimum", 0.0),
            ("p05", 0.05),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p95", 0.95),
            ("maximum", 1.0),
        )
    }
    if not math.isclose(quantiles["median"], median, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("ordinary median and quantile median differ")
    record = {
        "outcome": "PASS",
        "integrity_token": "E3_INSTRUMENT_PROBE_PASS",
        "initialization_seeds": list(seeds),
        "updates_per_initialization": 25,
        "completed_updates": 50,
        "batch_size": int(config["training"]["batch_size"]),
        "horizon_schedule": list(config["training"]["train_horizons"]),
        "provisional_safety_clip": float(cfg["provisional_safety_clip"]),
        "raw_global_preclip_gradient_norms": observations,
        "quantiles": quantiles,
        "ordinary_median": median,
        "mapping": str(cfg["mapping"]),
        "derived_controller_gradient_clip_norm": clip,
        "trainable_parameter_boundary_sha256": boundary_hash,
        "forbidden_metrics_computed": [],
        "accuracy_computed": False,
        "prediction_inspected": False,
        "validation_or_test_consumed": False,
        "state_crossed_to_smoke": ["scalar_C", "integrity_token", "audit_record"],
        "throwaway_state_discarded": True,
    }
    return record, clip, phases


def wilson_interval(numerator: int, denominator: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    radius = z * math.sqrt(
        p * (1 - p) / denominator + z2 / (4 * denominator * denominator)
    ) / (1 + z2 / denominator)
    return center - radius, center + radius


def run_competence_smoke(
    train_examples: Sequence[DeductionExample],
    validation_examples: Sequence[DeductionExample],
    config: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    device: torch.device,
    clip: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = config["competence_smoke"]
    seed = int(cfg["seed"])
    configure_runtime(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def ensure_within_cap(stage: str) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > float(cfg["wall_time_cap_seconds"]):
            raise TimeoutError(f"competence smoke exceeded cap during {stage}")

    tokenizer, substrate = load_frozen_substrate(coordinates, device)
    train_representations, train_compute = build_frozen_representations(
        train_examples, config, tokenizer, substrate, device
    )
    ensure_within_cap("smoke training representation construction")
    validation_representations, validation_compute = build_frozen_representations(
        validation_examples, config, tokenizer, substrate, device
    )
    ensure_within_cap("smoke validation representation construction")
    frozen_parameter_count = parameter_count(substrate, trainable_only=False)
    del tokenizer, substrate
    release_cuda()

    # The registered smoke seed controls the controller initialization itself.
    configure_runtime(seed)
    model = E3RecurrentController(config).to(device)
    optimizer = make_optimizer(model, config)
    model.train()
    curve: list[dict[str, Any]] = []
    window_losses: list[float] = []
    batches = _fixed_index_batches(
        len(train_representations),
        int(cfg["batch_size"]),
        int(cfg["updates"]),
        seed + 1,
    )
    for update, indices in enumerate(batches, start=1):
        ensure_within_cap(f"smoke update {update}")
        batch = [train_representations[index] for index in indices]
        states, mask, _frozen_prompt, labels = collate_representations(batch, device)
        optimizer.zero_grad(set_to_none=True)
        peak = float(config["training"]["peak_learning_rate"])
        learning_rate = peak * min(1.0, update / int(cfg["warmup_updates"]))
        set_learning_rate(optimizer, learning_rate)
        horizon = int(cfg["horizon_schedule"][(update - 1) % 3])
        with training_autocast(device):
            logits, _, _ = model(states, mask, horizon)
            loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite smoke training loss")
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        if not bool(torch.isfinite(raw_norm)):
            raise FloatingPointError("nonfinite smoke gradient norm")
        optimizer.step()
        window_losses.append(float(loss.detach().cpu()))
        if update % 50 == 0:
            curve.append(
                {
                    "completed_update": update,
                    "mean_training_cross_entropy_last_50_updates": statistics.mean(
                        window_losses
                    ),
                    "learning_rate": learning_rate,
                }
            )
            window_losses.clear()
    if len(curve) != 10 or curve[-1]["completed_update"] != 500:
        raise RuntimeError("smoke did not complete exactly 500 updates")
    ensure_within_cap("pre-validation boundary")
    model.eval()
    correct = 0
    evaluated = 0
    with torch.inference_mode():
        for start in range(0, len(validation_representations), 16):
            ensure_within_cap("single T=4 validation evaluation")
            batch = validation_representations[start : start + 16]
            states, mask, _frozen_prompt, labels = collate_representations(batch, device)
            with training_autocast(device):
                logits, _, _ = model(states, mask, int(cfg["validation_horizon"]))
            correct += int((logits.argmax(-1) == labels).sum().item())
            evaluated += len(batch)
    wall = time.perf_counter() - started
    if wall > float(cfg["wall_time_cap_seconds"]):
        raise TimeoutError("competence smoke exceeded cap at validation completion")
    if evaluated != 512:
        raise RuntimeError("smoke validation denominator is not 512")
    lower, upper = wilson_interval(correct, evaluated)
    outcome = "PASS" if correct >= int(cfg["pass_correct_minimum"]) else "PREFLIGHT_STOP"
    reason = (
        "PRETRAINED_INTERFACE_COMPETENCE_SMOKE_PASS"
        if outcome == "PASS"
        else "PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS"
    )
    record = {
        "outcome": outcome,
        "reason": reason,
        "seed": seed,
        "train_examples": len(train_representations),
        "completed_optimizer_updates": 500,
        "example_exposures": 4000,
        "batch_size": 8,
        "horizon_schedule": [1, 2, 4],
        "loss": "four_way_cross_entropy_only",
        "optimizer": "AdamW",
        "peak_learning_rate": 3e-4,
        "warmup_updates": 50,
        "gradient_clip_norm": clip,
        "training_curve": curve,
        "intermediate_validation_inspections": 0,
        "validation_horizon": 4,
        "validation_correct": correct,
        "validation_denominator": evaluated,
        "validation_percentage": 100.0 * correct / evaluated,
        "wilson_95_interval": {"lower": lower, "upper": upper},
        "pass_correct_minimum": 205,
        "wall_time_seconds": wall,
        "wall_time_cap_seconds": 900,
        "frozen_base_parameters": frozen_parameter_count,
        "throwaway_state_discarded": True,
        "canonical_state_initialized_from_smoke": False,
    }
    compute = {
        "phase": "competence_smoke",
        "wall_time_seconds": wall,
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0,
        "cache_bytes": train_compute["cache_bytes"]
        + validation_compute["cache_bytes"],
        "generated_base_tokens": train_compute["generated_base_tokens"]
        + validation_compute["generated_base_tokens"],
        "processed_examples": train_compute["examples"]
        + validation_compute["examples"],
    }
    del model, optimizer, train_representations, validation_representations
    release_cuda()
    return record, compute


def hash_bindings(
    config_path: Path,
    config: Mapping[str, Any],
    datasets: Mapping[str, Sequence[DeductionExample]],
    coordinates: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_slot = validate_pending_manifest_slot(config_path, config)
    model = E3RecurrentController(config)
    critic = E3LatentProgressCritic(config)
    control = E3NonRecurrentControl(config)
    parameters = validate_parameter_boundaries(model, critic, control, config)
    boundary = parameter_boundary(model)
    split_hashes = {name: split_hash(datasets[name], config) for name in E3_SPLITS}
    hashes = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "registration_sha256_before_slot_amendment": sha256_file(
            E3_REGISTRATION_PATH
        ),
        "e2_generator_source_sha256": sha256_file(E2_RUNNER_PATH),
        "generator_contract_sha256": sha256_json(config["generator"]),
        "rendered_prompt_template_sha256": sha256_bytes(
            str(config["generator"]["prompt_template"]).encode("utf-8")
        ),
        "all_splits_content_sha256": dataset_content_hash(datasets),
        "split_sha256": split_hashes,
        "substrate_alias": config["substrate"]["alias"],
        "substrate_revision_sha256": sha256_bytes(
            str(coordinates["revision"]).encode("ascii")
        ),
        "substrate_local_content_sha256": coordinates["content_digest"],
        "tokenizer_binding": "included_in_substrate_local_content_sha256",
        "common_architecture_sha256": sha256_json(config["common_model"]),
        "critic_architecture_sha256": sha256_json(config["critic"]),
        "trainable_parameter_boundary_sha256": boundary["sha256"],
    }
    del model, critic, control
    return {
        "hashes": hashes,
        "parameter_counts": parameters,
        "pending_manifest_slot": manifest_slot,
    }


def feature_boundary_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = tuple(config["critic"]["allowed_features"])
    forbidden = tuple(config["critic"]["forbidden_features"])
    overlap = sorted(set(allowed) & set(forbidden))
    return {
        "configured_allowed_features": list(allowed),
        "implemented_allowed_features": list(
            E3LatentProgressCritic.ALLOWED_FEATURE_NAMES
        ),
        "configured_forbidden_features": list(forbidden),
        "implemented_forbidden_features": list(
            E3LatentProgressCritic.FORBIDDEN_FEATURE_NAMES
        ),
        "overlap": overlap,
        "pass": allowed == E3LatentProgressCritic.ALLOWED_FEATURE_NAMES
        and forbidden == E3LatentProgressCritic.FORBIDDEN_FEATURE_NAMES
        and not overlap,
    }


def validate_preflight_result(result: Mapping[str, Any]) -> None:
    required = (
        "schema_version",
        "preflight_id",
        "experiment_id",
        "started_utc",
        "completed_utc",
        "preregistration",
        "hashes",
        "parameter_counts",
        "pending_manifest_slot",
        "feature_boundary_audit",
        "generator_audit",
        "cohort_isolation",
        "instrument_probe",
        "competence_smoke",
        "compute",
        "state_isolation",
        "decision",
        "final_token",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"E3 preflight schema missing fields: {missing}")
    token = result["final_token"]
    if token not in PREFLIGHT_FINAL_TOKENS:
        raise ValueError("invalid E3 preflight final token")
    if result["decision"].get("final_token") != token:
        raise ValueError("preflight decision/result token mismatch")
    if result["preflight_id"] != "exp_e3_preflight":
        raise ValueError("wrong preflight id")
    if not result["cohort_isolation"].get("all_forbidden_access_counts_zero"):
        raise ValueError("forbidden cohort access occurred")
    probe = result["instrument_probe"]
    if probe.get("outcome") == "PASS":
        norms = probe.get("raw_global_preclip_gradient_norms")
        if not isinstance(norms, list) or len(norms) != 50:
            raise ValueError("probe PASS requires exactly 50 raw norms")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in norms):
            raise ValueError("probe PASS contains an invalid gradient norm")
        median = (sorted(float(value) for value in norms)[24:26])
        ordinary = sum(median) / 2
        expected_clip = min(16, max(2, math.floor(ordinary + 0.5)))
        if expected_clip != probe.get("derived_controller_gradient_clip_norm"):
            raise ValueError("probe clip mapping was not reproduced exactly")
    smoke = result["competence_smoke"]
    if token == "PASS":
        if probe.get("outcome") != "PASS" or smoke.get("outcome") != "PASS":
            raise ValueError("preflight PASS requires both gates to PASS")
        if int(smoke.get("validation_correct", -1)) < 205:
            raise ValueError("preflight PASS is below the smoke integer threshold")
    if result["state_isolation"].get("all_throwaway_state_discarded") is not True:
        raise ValueError("preflight did not confirm throwaway-state disposal")


def _fsync_directory(directory: Path) -> bool:
    if os.name == "nt":
        return False
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def scavenge_stale_preflight_temps(directory: Path) -> list[Path]:
    removed: list[Path] = []
    if not directory.exists():
        return removed
    for candidate in directory.glob(TEMP_GLOB):
        candidate.unlink()
        removed.append(candidate)
    if removed:
        _fsync_directory(directory)
    return removed


def write_immutable_preflight(result: Mapping[str, Any], path: Path) -> None:
    validate_preflight_result(result)
    if path.suffix != ".json" or path.match(TEMP_GLOB):
        raise ValueError("immutable preflight path/temp namespace overlap")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.parent / f"{TEMP_PREFIX}{uuid.uuid4().hex}{TEMP_SUFFIX}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"immutable result already exists: {path}") from error
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def generator_audit_with_isolation(
    generator: E3DeductionGenerator,
    datasets: Mapping[str, Sequence[DeductionExample]],
) -> dict[str, Any]:
    audit = generator.audit_dataset(datasets)
    ids = {name: {row.example_id for row in rows} for name, rows in datasets.items()}
    groups = {
        name: {row.counterfactual_group for row in rows}
        for name, rows in datasets.items()
    }
    texts = {name: {row.rendered_text for row in rows} for name, rows in datasets.items()}
    for index, left in enumerate(E3_SPLITS):
        for right in E3_SPLITS[index + 1 :]:
            if ids[left] & ids[right]:
                raise AssertionError(f"identifier overlap: {left}/{right}")
            if groups[left] & groups[right]:
                raise AssertionError(f"counterfactual-group overlap: {left}/{right}")
            if texts[left] & texts[right]:
                raise AssertionError(f"rendered-question overlap: {left}/{right}")
    smoke_labels = Counter(
        row.answer_position for row in datasets["competence_smoke_validation"]
    )
    smoke_count = len(datasets["competence_smoke_validation"])
    expected_per_class = smoke_count // 4
    if smoke_labels != Counter({index: expected_per_class for index in range(4)}):
        raise AssertionError("smoke validation is not exactly class balanced")
    audit.update(
        {
            "all_seven_namespaces_pairwise_disjoint": True,
            "identifier_disjoint": True,
            "rendered_question_disjoint": True,
            "smoke_validation_answer_class_counts": dict(smoke_labels),
            "smoke_validation_exactly_128_per_class": smoke_count == 512
            and expected_per_class == 128,
        }
    )
    return audit


def run_preflight(config_path: Path) -> int:
    if RESULT_PATH.exists():
        raise RuntimeError(f"immutable E3 preflight already exists: {RESULT_PATH}")
    started_utc = utc_now()
    overall_started = time.perf_counter()
    config = load_config(config_path)
    configure_runtime(int(config["generator"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("E3 preflight requires CUDA")
    device = torch.device("cuda")
    scavenge_stale_preflight_temps(RESULT_PATH.parent)
    coordinates = resolve_and_validate_substrate(config)
    generator = E3DeductionGenerator(config)
    datasets = generator.generate_dataset()
    generator_audit = generator_audit_with_isolation(generator, datasets)
    bindings = hash_bindings(config_path, config, datasets, coordinates)
    feature_audit = feature_boundary_audit(config)
    if not feature_audit["pass"]:
        raise RuntimeError("latent critic feature boundary audit failed")

    phases: list[dict[str, Any]] = []
    clip: int | None = None
    try:
        probe, clip, phases = run_gradient_probe(
            datasets["instrument_probe_train"], config, coordinates, device
        )
    except Exception as error:
        # The registration makes any incomplete/invalid instrument observation
        # terminal. Do not retry, select an alternative clip, or enter smoke.
        failure_type = type(error).__name__
        del error
        release_cuda()
        probe = {
            "outcome": "PREFLIGHT_STOP",
            "reason": "INSTRUMENT_CALIBRATION_INVALID",
            "completed_updates": None,
            "raw_global_preclip_gradient_norms": [],
            "derived_controller_gradient_clip_norm": None,
            "failure_type": failure_type,
            "forbidden_metrics_computed": [],
            "accuracy_computed": False,
            "prediction_inspected": False,
            "validation_or_test_consumed": False,
            "throwaway_state_discarded": True,
        }
        smoke = {
            "outcome": "NOT_RUN",
            "reason": "BLOCKED_BY_INSTRUMENT_CALIBRATION_INVALID",
            "completed_optimizer_updates": 0,
            "validation_inspected": False,
            "throwaway_state_discarded": True,
        }
        final_token = "PREFLIGHT_STOP"
        reason = "INSTRUMENT_CALIBRATION_INVALID"
        route = "FRESH_PRE_DATA_AMENDMENT_AND_REVIEW_REQUIRED"
    else:
        smoke_started = time.perf_counter()
        try:
            smoke, smoke_compute = run_competence_smoke(
                datasets["competence_smoke_train"],
                datasets["competence_smoke_validation"],
                config,
                coordinates,
                device,
                clip,
            )
        except Exception as error:
            # Hash/accounting/hardware/nonfinite/cache/cap failures have the
            # frozen operational VOID route and no scientific interpretation.
            failure_type = type(error).__name__
            del error
            release_cuda()
            smoke = {
                "outcome": "VOID_NO_ROUTE",
                "reason": "COMPETENCE_SMOKE_OPERATIONAL_FAILURE",
                "failure_type": failure_type,
                "completed_optimizer_updates": None,
                "validation_inspected": False,
                "wall_time_seconds": time.perf_counter() - smoke_started,
                "wall_time_cap_seconds": 900,
                "throwaway_state_discarded": True,
            }
            final_token = "VOID_NO_ROUTE"
            reason = "COMPETENCE_SMOKE_OPERATIONAL_FAILURE"
            route = "NO_SCIENTIFIC_INTERPRETATION_NO_SUCCESSOR_ROUTE"
        else:
            phases.append(smoke_compute)
            final_token = str(smoke["outcome"])
            if final_token == "PASS":
                reason = "INSTRUMENT_AND_COMPETENCE_SMOKE_PASS"
                route = "STOP_AT_INDEPENDENT_REVIEW_BOUNDARY"
            else:
                reason = str(smoke["reason"])
                route = "DRAFT_REGISTER_INTERFACE_SUPERVISION_DIAGNOSTIC_ONLY"
    canonical_authorized = False
    result = {
        "schema_version": "1.0.0",
        "preflight_id": config["preflight_id"],
        "experiment_id": config["experiment_id"],
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "preregistration": config["preregistration"],
        **bindings,
        "feature_boundary_audit": feature_audit,
        "generator_audit": generator_audit,
        "cohort_isolation": {
            "fresh_synthetic_namespaces_only": list(E3_SPLITS),
            "forbidden_access_counts": {name: 0 for name in FORBIDDEN_COHORTS},
            "all_forbidden_access_counts_zero": True,
            "official_test_rows_inspected": 0,
            "preflight_runner_line_07_cohort_rows_accessed": 0,
            "preflight_runner_line_07_files_written": 0,
        },
        "instrument_probe": probe,
        "competence_smoke": smoke,
        "compute": {
            "overall_wall_time_seconds": time.perf_counter() - overall_started,
            "device_type": "cuda",
            "hardware": "registered_single_RTX_5090_24GB_constraint",
            "phases": phases,
            "peak_vram_allocated_bytes": max(
                (int(phase["peak_vram_allocated_bytes"]) for phase in phases),
                default=0,
            ),
            "peak_vram_reserved_bytes": max(
                (int(phase["peak_vram_reserved_bytes"]) for phase in phases),
                default=0,
            ),
            "total_completed_optimizer_updates": (
                int(probe.get("completed_updates") or 0)
                + int(smoke.get("completed_optimizer_updates") or 0)
            ),
            "canonical_retained_updates": 0,
        },
        "state_isolation": {
            "probe_models_optimizers_dataloaders_rng_discarded": True,
            "smoke_model_optimizer_dataloader_rng_discarded": True,
            "probe_or_smoke_checkpoint_retained": False,
            "probe_or_smoke_weight_crossed_stage_boundary": False,
            "canonical_initialized": False,
            "all_throwaway_state_discarded": True,
        },
        "decision": {
            "final_token": final_token,
            "reason": reason,
            "route": route,
            "canonical_launch_authorized_now": canonical_authorized,
            "remaining_gates_if_pass": [
                "measured_compute_cap_preflight",
                "independent_pre_training_review",
                "blocking_findings_resolved",
                "explicit_clean_launch_attestation",
            ]
            if final_token == "PASS"
            else [],
            "full_0_5B_launch_blocked": True,
        },
        "final_token": final_token,
    }
    validate_preflight_result(result)
    write_immutable_preflight(result, RESULT_PATH)
    print(
        json.dumps(
            {
                "final_token": final_token,
                "reason": reason,
                "clip": clip,
                "smoke_correct": smoke.get("validation_correct"),
                "smoke_denominator": smoke.get("validation_denominator"),
                "result": str(RESULT_PATH),
            },
            indent=2,
        )
    )
    return 0 if final_token == "PASS" else 2


def _mini_config(config: Mapping[str, Any]) -> dict[str, Any]:
    fixture = json.loads(json.dumps(config))
    fixture["generator"]["examples_per_split"] = {
        name: int(config["self_test"]["examples_per_split"]) for name in E3_SPLITS
    }
    return fixture


def run_self_test_fast(config_path: Path) -> int:
    config = load_config(config_path)
    torch.set_num_threads(int(config["self_test"]["torch_threads"]))
    configure_runtime(int(config["self_test"]["seed"]))
    fixture = _mini_config(config)
    generator = E3DeductionGenerator(fixture)
    datasets = generator.generate_dataset()
    audit = generator_audit_with_isolation(generator, datasets)
    model = E3RecurrentController(config)
    control = E3NonRecurrentControl(config)
    critic = E3LatentProgressCritic(config)
    counts = validate_parameter_boundaries(model, critic, control, config)
    fake_states = torch.randn(2, 11, 1536)
    fake_mask = torch.zeros(2, 11, dtype=torch.bool)
    logits, trajectory, prompt = model(fake_states, fake_mask, 4)
    if logits.shape != (2, 4) or len(trajectory) != 4 or prompt.shape != (2, 512):
        raise AssertionError("E3 recurrent forward shapes changed")
    labels = torch.tensor([0, 3])
    optimizer = make_optimizer(model, config)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 16.0)
    if not math.isfinite(float(norm)) or float(norm) <= 0:
        raise AssertionError("miniature gradient probe failed")
    optimizer.step()
    frozen_prompt = torch.randn(2, 1536)
    critic_features = critic.features_for_trajectory(trajectory, prompt, frozen_prompt)
    critic_logits = critic(critic_features)
    if critic_logits.shape != (2, 4):
        raise AssertionError("E3 critic forward shape changed")
    synthetic_scores = torch.tensor([[0.1, 0.2, 0.15], [0.3, 0.2, 0.4]])
    selected = cumulative_first_argmax(synthetic_scores)
    hysteretic = hysteretic_incumbent_indices(synthetic_scores, 0.05)
    if selected.shape != synthetic_scores.shape or hysteretic.shape != selected.shape:
        raise AssertionError("inherited selector mechanics changed")
    tmp_root = REPO_ROOT / ".e3_self_test_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir()
    try:
        fixture_result = {
            "schema_version": "1.0.0",
            "preflight_id": "exp_e3_preflight",
            "experiment_id": "exp_e3_pretrained_latch_mechanics",
            "started_utc": utc_now(),
            "completed_utc": utc_now(),
            "preregistration": config["preregistration"],
            "hashes": {"self_test": True},
            "parameter_counts": counts,
            "pending_manifest_slot": {
                "runner_sha256": "self-test",
                "config_sha256": "self-test",
                "controller_gradient_clip_norm": "PENDING_INSTRUMENT_PROBE",
                "clip_mapping": config["instrument_probe"]["mapping"],
            },
            "feature_boundary_audit": feature_boundary_audit(config),
            "generator_audit": audit,
            "cohort_isolation": {"all_forbidden_access_counts_zero": True},
            "instrument_probe": {
                "outcome": "PASS",
                "raw_global_preclip_gradient_norms": [2.0] * 50,
                "derived_controller_gradient_clip_norm": 2,
            },
            "competence_smoke": {"outcome": "PASS", "validation_correct": 205},
            "compute": {"self_test": True},
            "state_isolation": {"all_throwaway_state_discarded": True},
            "decision": {"final_token": "PASS"},
            "final_token": "PASS",
        }
        path = tmp_root / "preflight.json"
        write_immutable_preflight(fixture_result, path)
        overwrite_rejected = False
        try:
            write_immutable_preflight(fixture_result, path)
        except RuntimeError:
            overwrite_rejected = True
        if not overwrite_rejected:
            raise AssertionError("atomic writer permitted overwrite")
        stale = tmp_root / f"{TEMP_PREFIX}stale{TEMP_SUFFIX}"
        stale.write_text("stale", encoding="utf-8")
        if scavenge_stale_preflight_temps(tmp_root) != [stale]:
            raise AssertionError("startup scavenger did not remove only stale temp")
        if not path.exists():
            raise AssertionError("scavenger touched final result")
    finally:
        shutil.rmtree(tmp_root)
    output = {
        "self_test_tier": "fast",
        "generator_namespaces": len(datasets),
        "generator_examples": sum(len(rows) for rows in datasets.values()),
        "generator_integrity": audit["all_seven_namespaces_pairwise_disjoint"],
        "parameter_counts": counts,
        "optimizer_exclusions_present": True,
        "inherited_selector_provenance_available": callable(
            selector_provenance_metrics
        ),
        "inherited_arm4_available": callable(select_hysteresis_delta),
        "inherited_diagnostics_available": callable(trajectory_diagnostics)
        and callable(trajectory_accounting_assertions)
        and callable(selector_switch_hazards),
        "inherited_pairing_available": callable(construct_within_problem_pairs)
        and callable(confidence_matched_concordance)
        and callable(binary_auroc)
        and SelectorCorpus.__name__ == "SelectorCorpus",
        "schema_validation": True,
        "atomic_no_clobber": True,
        "startup_scavenger": True,
        "canonical_launch_path_exposed": False,
        "pass": True,
    }
    print(json.dumps(output, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test-fast", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launch-attestation")
    args = parser.parse_args(argv)
    if args.self_test_fast and args.launch_attestation is not None:
        parser.error("self-test-fast does not accept a launch attestation")
    if args.preflight and args.launch_attestation != "OWNER_AUTHORIZED_E3_PREFLIGHT":
        parser.error(
            "--preflight requires --launch-attestation OWNER_AUTHORIZED_E3_PREFLIGHT"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if args.self_test_fast:
        return run_self_test_fast(config_path)
    return run_preflight(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
