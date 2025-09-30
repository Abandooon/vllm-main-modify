# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from vllm.v1.structured_output.request import get_structured_output_key
try:
    from vllm.v1.structured_output.audit_tracker import AuditEventType
except Exception:
    AuditEventType = None

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(),
                                  "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if 'additionalProperties' not in data and \
            ('properties' in data or 'patternProperties' in data):
            data['additionalProperties'] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def process_for_additional_properties(
        guide_json: Union[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):

    def __post_init__(self):
        self.disable_any_whitespace = \
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        self.disable_additional_properties = \
            self.vllm_config.structured_outputs_config.disable_additional_properties

        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, self.vocab_size)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        self.serialized_grammar = serialize_guidance_grammar(
            request_type, grammar_spec, self.disable_any_whitespace,
            self.disable_additional_properties)

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size)

    def destroy(self):
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    num_processed_tokens: int = 0  # ✅ 新增：统一统计口径
    _termination_logged: bool = False

def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)


def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
    """Accepts a list of tokens and advances the parser."""
    if self.ll_tokenizer.eos_token in tokens:
        self.terminated = True

    if self.ll_matcher.is_stopped():
        return True

    # 延迟绑定审计上下文
    if self._request_id is None and self._audit_tracker and self._audit_tracker.is_enabled():
        self.set_audit_context(request_id)

    # 记录当前状态（Guidance 没有显式状态机ID，用可追踪代替）
    def _state_id() -> str:
        return f"stopped={self.ll_matcher.is_stopped()};tokens={self.num_processed_tokens};terminated={self.terminated}"

    prev_state = _state_id()
    r = self.ll_matcher.consume_tokens(tokens)
    self.check_error()

    if r:
        # 逐 token 迁移（用步数近似状态演化）
        for t in tokens:
            self.num_processed_tokens += 1
            if self._audit_tracker and self._audit_tracker.is_enabled() and AuditEventType:
                self._audit_tracker.record_event(
                    request_id=request_id,
                    event_type=AuditEventType.STATE_TRANSITION,
                    previous_state_id=prev_state,
                    current_state_id=_state_id(),
                    selected_token=t,
                    metadata={"num_processed_tokens": self.num_processed_tokens},
                )
            prev_state = _state_id()

        if self._audit_tracker and self._audit_tracker.is_enabled():
            self._audit_tracker.record_token_acceptance(
                request_id=request_id,
                tokens=tokens,
                accepted=True,
                current_state=prev_state,
            )
    else:
        if self._audit_tracker and self._audit_tracker.is_enabled():
            self._audit_tracker.record_token_acceptance(
                request_id=request_id,
                tokens=tokens,
                accepted=False,
                current_state=prev_state,
            )

    # 若已停或收到 EOS，触发一次性终止打点
    if (self.terminated or self.ll_matcher.is_stopped()) and not self._termination_logged:
        if self._audit_tracker and self._audit_tracker.is_enabled() and AuditEventType:
            self._audit_tracker.record_event(
                request_id=request_id,
                event_type=AuditEventType.TERMINATION,
                current_state_id=_state_id(),
                metadata={"total_tokens": self.num_processed_tokens},
            )
        self._termination_logged = True

    return r


def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]


def rollback(self, num_tokens: int) -> None:
    prev_state_tokens = self.num_processed_tokens
    self.ll_matcher.rollback(num_tokens)
    self.check_error()
    self.num_processed_tokens = max(0, prev_state_tokens - num_tokens)

    if self._audit_tracker and self._audit_tracker.is_enabled() and self._request_id:
        self._audit_tracker.record_rollback(
            request_id=self._request_id,
            num_tokens=num_tokens,
            current_state=f"stopped={self.ll_matcher.is_stopped()};tokens={self.num_processed_tokens};terminated={self.terminated}",
        )


def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
    # 原逻辑
    llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
    self.check_error()

    # 审计：bitmask（允许 token 集/数量）
    if self._audit_tracker and self._audit_tracker.is_enabled() and self._request_id:
        self._audit_tracker.record_bitmask_update(
            request_id=self._request_id,
            bitmask=bitmask[idx],
            current_state=f"stopped={self.ll_matcher.is_stopped()};tokens={self.num_processed_tokens};terminated={self.terminated}",
        )

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        # 原注释：This method may be called multiple times per request
        self.printed_error = False
        self.terminated = False
        self.num_processed_tokens = 0
        self._termination_logged = False


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: Union[str, dict[str, Any]],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:

    def _process_schema(grammar_spec: Union[str, dict[str, Any]], ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            })

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            })
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(
                        f"Trigger {begin} not found in triggers {triggers}")
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    ))
            if not tags:
                raise ValueError(
                    "No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error("Validation should have already occurred. "
                         "Please file an issue.")
            raise ValueError("grammar is not of valid supported types. "
                             f"({request_type!s})")
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
        sampling_params: SamplingParams,
        tokenizer: Optional[llguidance.LLTokenizer] = None) -> None:
    tp, grm = get_structured_output_key(sampling_params)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
