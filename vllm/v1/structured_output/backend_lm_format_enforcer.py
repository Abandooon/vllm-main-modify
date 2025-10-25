# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import SamplingParams
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)

if TYPE_CHECKING:
    import lmformatenforcer
    import lmformatenforcer.integrations.vllm as lmfe_vllm
else:
    lmformatenforcer = LazyLoader("lmformatenforcer", globals(),
                                  "lmformatenforcer")
    lmfe_vllm = LazyLoader("lmformatenforcer.integrations.vllm", globals(),
                           "lmformatenforcer.integrations.vllm")


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int) -> lmfe_vllm.TokenEnforcerTokenizerData:
    return lmfe_vllm.build_vllm_token_enforcer_tokenizer_data(
        tokenizer, use_bitmask=True, vocab_size=vocab_size)


@dataclass
class LMFormatEnforcerGrammar(StructuredOutputGrammar):
    token_enforcer: lmformatenforcer.TokenEnforcer
    current_tokens_prefix: list[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize base class and audit support."""
        super().__init__()
        self._backend_name = "lm_format_enforcer"
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
        return f"prefix_len={len(self.current_tokens_prefix)}"

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
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

        original_len = len(self.current_tokens_prefix)
        current_state = self._get_current_state_id()

        for token in tokens:
            prev_state = current_state
            if not self.token_enforcer.get_allowed_tokens(
                    self.current_tokens_prefix).is_token_allowed(token):
                # Rollback partial updates to ensure atomicity.
                del self.current_tokens_prefix[original_len:]

                # 记录拒绝
                if self._is_audit_enabled():
                    try:
                        self._audit_tracker.record_token_acceptance(
                            request_id=request_id,
                            tokens=tokens,
                            accepted=False,
                            current_state=prev_state
                        )
                    except Exception:
                        pass

                return False

            self.current_tokens_prefix.append(token)
            current_state = self._get_current_state_id()

            # 记录状态转移
            if self._is_audit_enabled():
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=request_id,
                        event_type=AuditEventType.STATE_TRANSITION,
                        previous_state_id=prev_state,
                        current_state_id=current_state,
                        selected_token=token,
                        metadata={"prefix_length": len(self.current_tokens_prefix)}
                    )
                except Exception:
                    pass

        # 记录接受
        if self._is_audit_enabled():
            try:
                self._audit_tracker.record_token_acceptance(
                    request_id=request_id,
                    tokens=tokens,
                    accepted=True,
                    current_state=current_state
                )
            except Exception:
                pass

        # 检查终止
        if self.is_terminated() and not self._termination_logged:
            self._termination_logged = True
            if self._is_audit_enabled():
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=request_id,
                        event_type=AuditEventType.TERMINATION,
                        current_state_id=current_state,
                        metadata={"total_tokens": len(self.current_tokens_prefix)}
                    )
                except Exception:
                    pass

        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        current_state = self._get_current_state_id()

        for prefix_length in range(len(tokens)):
            prefix = tokens[:prefix_length]
            next_token = tokens[prefix_length]
            if not self.token_enforcer.get_allowed_tokens(
                    self.current_tokens_prefix +
                    prefix).is_token_allowed(next_token):
                break
        else:
            validated_tokens = tokens
            if self._is_audit_enabled() and self._request_id:
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=self._request_id,
                        event_type=AuditEventType.TOKEN_VALIDATE,
                        accepted_tokens=validated_tokens,
                        current_state_id=current_state,
                        metadata={"total_tokens_validated": len(tokens)}
                    )
                except Exception:
                    pass
            return tokens

        validated_tokens = tokens[:prefix_length]

        if self._is_audit_enabled() and self._request_id:
            try:
                from vllm.v1.structured_output.audit_tracker import AuditEventType
                self._audit_tracker.record_event(
                    request_id=self._request_id,
                    event_type=AuditEventType.TOKEN_VALIDATE,
                    accepted_tokens=validated_tokens,
                    rejected_tokens=tokens[prefix_length:] if prefix_length < len(tokens) else None,
                    current_state_id=current_state,
                    metadata={"total_tokens_validated": len(tokens)}
                )
            except Exception:
                pass

        return validated_tokens

    def rollback(self, num_tokens: int) -> None:
        prev_state = self._get_current_state_id()
        self.current_tokens_prefix = self.current_tokens_prefix[:-num_tokens]
        current_state = self._get_current_state_id()

        if self._is_audit_enabled() and self._request_id:
            try:
                self._audit_tracker.record_rollback(
                    request_id=self._request_id,
                    num_tokens=num_tokens,
                    current_state=current_state
                )
            except Exception:
                pass

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        current_state = self._get_current_state_id()
        allowed_tokens = self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix)
        bitmask[batch_index] = allowed_tokens.allowed_tokens

        if self._is_audit_enabled() and self._request_id:
            try:
                self._audit_tracker.record_bitmask_update(
                    request_id=self._request_id,
                    bitmask=bitmask[batch_index],
                    current_state=current_state
                )
            except Exception:
                pass

    def is_terminated(self) -> bool:
        # We are considered terminated if the prefix ends with eos_token_id
        return_value = len(
            self.current_tokens_prefix) > 0 and self.current_tokens_prefix[
                           -1] == self.token_enforcer.eos_token_id
        return return_value

    def reset(self):
        self.current_tokens_prefix = []
        self._termination_logged = False

        if self._audit_tracker and self._request_id:
            try:
                self._audit_tracker.cleanup_trail(self._request_id)
            except Exception:
                pass
            self._request_id = None


@dataclass
class LMFormatEnforcerBackend(StructuredOutputBackend):

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
            self.tokenizer, self.vocab_size)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        character_level_parser: lmformatenforcer.CharacterLevelParser
        if request_type == StructuredOutputOptions.JSON:
            spec_dict = json.loads(grammar_spec)
            character_level_parser = lmformatenforcer.JsonSchemaParser(
                spec_dict)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            character_level_parser = lmformatenforcer.JsonSchemaParser(None)
        elif request_type == StructuredOutputOptions.REGEX:
            character_level_parser = lmformatenforcer.RegexParser(grammar_spec)
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            character_level_parser = lmformatenforcer.UnionParser(
                [lmformatenforcer.StringParser(choice) for choice in choices])
        else:
            raise ValueError(
                "Invalid request type for LM Format Enforcer backend"
                f"({request_type!s})")
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None else 0)

        if max_rollback_tokens > 0:
            raise ValueError(
                "LM Format Enforcer backend does not support speculative tokens"
            )

        token_enforcer = lmformatenforcer.TokenEnforcer(
            tokenizer_data=self.tokenizer_data,
            parser=character_level_parser,
        )
        return LMFormatEnforcerGrammar(token_enforcer)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

    def destroy(self):
        pass


def validate_structured_output_request_lm_format_enforcer(
        params: SamplingParams):
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        return
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # make sure schema is valid json
                json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        return
    elif so_params.choice:
        return
    elif so_params.grammar:
        raise ValueError("LM Format Enforcer structured outputs backend "
                         "does not support grammar specifications")