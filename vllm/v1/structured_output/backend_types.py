# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    # Import audit tracker for type hints
    from vllm.v1.structured_output.audit_tracker import (
        StructuredOutputAuditTracker, AuditEventType
    )


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


# backend_types.py - 改进的基类
class StructuredOutputGrammar(ABC):
    def __init__(self):
        self._audit_tracker: Optional[StructuredOutputAuditTracker] = None
        self._request_id: Optional[str] = None
        self._backend_name: str = self.__class__.__name__
        self._previously_logged_termination: bool = False

    # 改用factory method而不是直接在__init__中初始化
    @classmethod
    def create_with_audit(cls, *args, **kwargs):
        """Factory method for creating grammar with audit support"""
        instance = cls(*args, **kwargs)
        instance._init_audit()  # 延迟调用审计初始化
        return instance

    def _init_audit(self):
        try:
            from vllm.v1.structured_output.audit_tracker import get_audit_tracker
            self._audit_tracker = get_audit_tracker()
        except ImportError:
            pass

    def set_audit_context(self,
                          request_id: str,
                          audit_tracker: Optional[StructuredOutputAuditTracker] = None) -> None:
        """
        Set audit context for this grammar instance.

        Args:
            request_id: The request ID for audit tracking
            audit_tracker: The audit tracker instance (if None, will try to get global)
        """
        self._request_id = request_id
        if audit_tracker is None:
            # Try to get global audit tracker
            try:
                from vllm.v1.structured_output.audit_tracker import get_audit_tracker
                self._audit_tracker = get_audit_tracker()
            except ImportError:
                self._audit_tracker = None
        else:
            self._audit_tracker = audit_tracker

        # Start audit trail if tracker is enabled
        if self._audit_tracker and self._audit_tracker.is_enabled():
            self._audit_tracker.start_trail(
                request_id=request_id,
                backend_type=self._backend_name
            )

    def _audit_record(self, event_type: str, **kwargs) -> None:
        """Internal method to record audit events."""
        if not (self._audit_tracker and self._audit_tracker.is_enabled() and self._request_id):
            return
        try:
            from vllm.v1.structured_output.audit_tracker import AuditEventType
            event_type_enum = AuditEventType(event_type)
            self._audit_tracker.record_event(
                request_id=self._request_id,
                event_type=event_type_enum,
                **kwargs
            )
        except Exception:
            # 审计失败不影响主流程
            pass

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """
        Determines whether the provided tokens are accepted for the
        given request.

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.
        """

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """
        Validates the provided tokens against the grammar.
        Will not advance the FSM.

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.
        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """
        Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the state of the structured output grammar.
        """

    def finalize_audit(self) -> None:
        """Finalize the audit trail for this grammar."""
        if self._audit_tracker and self._audit_tracker.is_enabled() and self._request_id:
            self._audit_tracker.finalize_trail(self._request_id)


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests."""

    vllm_config: VllmConfig
    tokenizer: AnyTokenizer
    vocab_size: int

    def __post_init__(self):
        """Initialize the backend with optional audit support."""
        self._audit_tracker: Optional[StructuredOutputAuditTracker] = None
        try:
            from vllm.v1.structured_output.audit_tracker import get_audit_tracker
            self._audit_tracker = get_audit_tracker()
        except ImportError:
            pass

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        """
        Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
                output request.
            grammar_spec (str): The grammar specification to compile.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.
        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.
        """

    @abstractmethod
    def destroy(self):
        """
        Backend-specific cleanup.
        """