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
        super().__post_init__()
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
    num_processed_tokens: int = 0

    def __post_init__(self):
        """Initialize base class and audit support."""
        super().__init__()
        self._backend_name = "guidance"
        self._termination_logged = False
        try:
            from vllm.v1.structured_output.audit_tracker import get_audit_tracker
            self._audit_tracker = get_audit_tracker()
        except ImportError:
            self._audit_tracker = None

    def _is_audit_enabled(self) -> bool:
        """统一的运行时检查方法"""
        return (self._audit_tracker is not None and
                self._audit_tracker.is_enabled())

    def _get_current_state_id(self) -> str:
        """Get current state identifier."""
        return f"stopped={self.ll_matcher.is_stopped()};tokens={self.num_processed_tokens};terminated={self.terminated}"

    def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser."""
        # ✅ 关键修改：首次绑定时，显式启动审计 trail
        if self._request_id is None:
            self._request_id = request_id

            if self._is_audit_enabled():
                try:
                    self._audit_tracker.start_trail(
                        request_id=request_id,
                        backend_type=self._backend_name,
                        grammar_spec=None,
                    )

                    # 原本是 logger.debug(...)
                    logger.warning(
                        f"[AuditGrammarInit] backend={self._backend_name} "
                        f"request_id={request_id} "
                        f"tracker_enabled={self._audit_tracker.is_enabled()} "
                        f"tracker_id={id(self._audit_tracker)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[AuditGrammarInit] FAILED to start trail for {request_id}: {e}"
                    )
            else:
                logger.warning(
                    f"[AuditGrammarInit] audit not enabled for backend={self._backend_name} "
                    f"request_id={request_id} "
                    f"tracker={self._audit_tracker} "
                    f"tracker_enabled={getattr(self._audit_tracker, 'enabled', None)}"
                )

        if self.ll_tokenizer.eos_token in tokens:
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        prev_state = self._get_current_state_id()
        r = self.ll_matcher.consume_tokens(tokens)
        self.check_error()

        if r:
            # 逐token转移
            for t in tokens:
                self.num_processed_tokens += 1
                current_state = self._get_current_state_id()

                if self._is_audit_enabled():
                    try:
                        from vllm.v1.structured_output.audit_tracker import AuditEventType
                        self._audit_tracker.record_event(
                            request_id=request_id,
                            event_type=AuditEventType.STATE_TRANSITION,
                            previous_state_id=prev_state,
                            current_state_id=current_state,
                            selected_token=t,
                            metadata={"num_processed_tokens": self.num_processed_tokens},
                        )
                    except Exception:
                        pass
                prev_state = current_state

            # 记录接受
            if self._is_audit_enabled():
                try:
                    self._audit_tracker.record_token_acceptance(
                        request_id=request_id,
                        tokens=tokens,
                        accepted=True,
                        current_state=prev_state,
                    )
                except Exception:
                    pass
        else:
            # 记录拒绝
            if self._is_audit_enabled():
                try:
                    self._audit_tracker.record_token_acceptance(
                        request_id=request_id,
                        tokens=tokens,
                        accepted=False,
                        current_state=prev_state,
                    )
                except Exception:
                    pass

        # 终止事件（一次性）
        if (self.terminated or self.ll_matcher.is_stopped()) and not self._termination_logged:
            self._termination_logged = True
            if self._is_audit_enabled():
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=request_id,
                        event_type=AuditEventType.TERMINATION,
                        current_state_id=self._get_current_state_id(),
                        metadata={"total_tokens": self.num_processed_tokens},
                    )
                except Exception:
                    pass

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

        current_state = self._get_current_state_id()
        num_tokens = self.ll_matcher.validate_tokens(tokens)
        self.check_error()

        validated_tokens = tokens[:num_tokens]

        if self._is_audit_enabled() and self._request_id:
            try:
                from vllm.v1.structured_output.audit_tracker import AuditEventType
                self._audit_tracker.record_event(
                    request_id=self._request_id,
                    event_type=AuditEventType.TOKEN_VALIDATE,
                    accepted_tokens=validated_tokens,
                    rejected_tokens=tokens[num_tokens:] if num_tokens < len(tokens) else None,
                    current_state_id=current_state,
                    metadata={"total_tokens_validated": len(tokens)}
                )
            except Exception:
                pass

        return validated_tokens

    def rollback(self, num_tokens: int) -> None:
        prev_state_tokens = self.num_processed_tokens
        self.ll_matcher.rollback(num_tokens)
        self.check_error()
        self.num_processed_tokens = max(0, prev_state_tokens - num_tokens)

        if self._is_audit_enabled() and self._request_id:
            try:
                self._audit_tracker.record_rollback(
                    request_id=self._request_id,
                    num_tokens=num_tokens,
                    current_state=self._get_current_state_id(),
                )
            except Exception:
                pass

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        current_state = self._get_current_state_id()
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

        if self._is_audit_enabled() and self._request_id:
            try:
                self._audit_tracker.record_bitmask_update(
                    request_id=self._request_id,
                    bitmask=bitmask[idx],
                    current_state=current_state,
                )
            except Exception:
                pass

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        self.printed_error = False
        self.terminated = False
        self.num_processed_tokens = 0
        self._termination_logged = False
        self.ll_matcher.reset()

        if self._audit_tracker and self._request_id:
            try:
                self._audit_tracker.cleanup_trail(self._request_id)
            except Exception:
                pass
            self._request_id = None


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