# backend_outlines.py - 移除静态AUDIT_ENABLED，添加_is_audit_enabled

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
from regex import escape as regex_escape

from vllm.sampling_params import SamplingParams
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from vllm.v1.structured_output.utils import (OutlinesVocabulary,
                                             get_outlines_cache,
                                             get_outlines_vocabulary)

# ❌ 移除静态变量
# AUDIT_ENABLED = False  # DELETE THIS LINE

if TYPE_CHECKING:
    import outlines_core as oc
    import outlines_core.json_schema as json_schema
else:
    oc = LazyLoader("oc", globals(), "outlines_core")
    json_schema = LazyLoader("json_schema", globals(),
                             "outlines_core.json_schema")


@dataclass
class OutlinesGrammarWithAudit(StructuredOutputGrammar):
    """Outlines Grammar implementation with audit support."""

    vocab_size: int
    guide: oc.Guide = field(hash=False)
    num_processed_tokens: int = field(default=0, repr=False, hash=False, init=False)
    _prev_finished: bool = field(default=False, init=False, repr=False, hash=False)

    def __post_init__(self):
        """Initialize base class and audit support."""
        super().__init__()
        self._backend_name = "outlines"
        try:
            from vllm.v1.structured_output.audit_tracker import get_audit_tracker
            self._audit_tracker = get_audit_tracker()
        except ImportError:
            self._audit_tracker = None

    def _is_audit_enabled(self) -> bool:
        """✅ 统一的运行时检查方法"""
        return (self._audit_tracker is not None and
                self._audit_tracker.is_enabled())

    def _get_current_state_id(self) -> Optional[str]:
        """Get current state ID from the guide."""
        try:
            if hasattr(self.guide, 'state'):
                state = self.guide.state
                return str(state) if state is not None else None
            elif hasattr(self.guide, 'current_state'):
                state = self.guide.current_state
                return str(state) if state is not None else None
        except:
            pass
        return f"tokens={self.num_processed_tokens}"

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM with audit tracking."""
        if self._request_id is None:
            self._request_id = request_id
            if self._is_audit_enabled():
                self.set_audit_context(request_id)

        current_state = self._get_current_state_id()

        if self.guide.accepts_tokens(tokens):
            for t in tokens:
                prev_state = current_state
                self.guide.advance(t)
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
                            metadata={"num_processed_tokens": self.num_processed_tokens}
                        )
                    except Exception:
                        pass

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

            return True
        else:
            if self._is_audit_enabled():
                try:
                    self._audit_tracker.record_token_acceptance(
                        request_id=request_id,
                        tokens=tokens,
                        accepted=False,
                        current_state=current_state
                    )
                except Exception:
                    pass

            return False

    def rollback(self, num_tokens: int) -> None:
        """Rollback with audit tracking."""
        prev_state = self._get_current_state_id()
        self.guide.rollback_state(num_tokens)
        self.num_processed_tokens -= num_tokens
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

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Validate tokens with audit tracking."""
        accepted: list[int] = []
        current_state = self._get_current_state_id()

        for tok in tokens:
            accepted.append(tok)
            if not self.guide.accepts_tokens(accepted):
                accepted.pop()
                break

        if self._is_audit_enabled() and self._request_id:
            try:
                from vllm.v1.structured_output.audit_tracker import AuditEventType
                self._audit_tracker.record_event(
                    request_id=self._request_id,
                    event_type=AuditEventType.TOKEN_VALIDATE,
                    accepted_tokens=accepted,
                    rejected_tokens=tokens[len(accepted):] if len(accepted) < len(tokens) else None,
                    current_state_id=current_state,
                    metadata={"total_tokens_validated": len(tokens)}
                )
            except Exception:
                pass

        return accepted

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """Fill bitmask with audit tracking."""
        current_state = self._get_current_state_id()
        mask = bitmask[idx]
        self.guide.write_mask_into(mask.data_ptr(), mask.numel(),
                                   mask.element_size())

        if self._is_audit_enabled() and self._request_id:
            try:
                mask_copy = mask.clone()
                self._audit_tracker.record_bitmask_update(
                    request_id=self._request_id,
                    bitmask=mask_copy,
                    current_state=current_state
                )
            except Exception:
                pass

    def is_terminated(self) -> bool:
        """Check termination with audit tracking."""
        curr = self.guide.is_finished()
        prev = self._prev_finished
        self._prev_finished = curr

        if prev and not hasattr(self, '_termination_logged'):
            self._termination_logged = True
            if self._is_audit_enabled() and self._request_id:
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=self._request_id,
                        event_type=AuditEventType.TERMINATION,
                        current_state_id=self._get_current_state_id(),
                        metadata={"total_tokens": self.num_processed_tokens}
                    )
                except Exception:
                    pass

        return prev

    def reset(self):
        """Reset with audit support."""
        self.num_processed_tokens = 0
        self._prev_finished = False
        self._termination_logged = False
        self.guide.reset()

        if self._audit_tracker and self._request_id:
            try:
                self._audit_tracker.cleanup_trail(self._request_id)
            except Exception:
                pass
            self._request_id = None


@dataclass
class OutlinesBackendWithAudit(StructuredOutputBackend):
    """Outlines backend with audit support."""

    def __post_init__(self):
        super().__post_init__()
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        self.cache = get_outlines_cache()

    def _compile_index(self, regex_string: str,
                       vocabulary: OutlinesVocabulary) -> oc.Index:
        cache_key = f"{vocabulary._hash}_{regex_string}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        index = oc.Index(regex_string, vocabulary.inner)
        self.cache[cache_key] = index

        return index

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        """Compile grammar with audit support."""
        if request_type == StructuredOutputOptions.JSON:
            regex = json_schema.build_regex_from_schema(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            regex = grammar_spec
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            choices = [regex_escape(c) for c in choices]
            regex = "(" + "|".join(choices) + ")"
        else:
            raise ValueError(
                f"Invalid request type for Outlines backend ({request_type!s})"
            )

        index = self._compile_index(regex, self.vocabulary)
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None else 0)

        grammar = OutlinesGrammarWithAudit(
            vocab_size=self.vocab_size,
            guide=oc.Guide(index, max_rollback=max_rollback_tokens)
        )

        return grammar

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

    def destroy(self):
        pass