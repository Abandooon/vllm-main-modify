# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

import vllm.envs
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from vllm.v1.structured_output.utils import (choice_as_grammar,
                                             convert_lark_to_ebnf,
                                             grammar_is_likely_lark)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class XgrammarBackend(StructuredOutputBackend):

    def __post_init__(self):
        self.disable_any_whitespace = \
            self.vllm_config.structured_outputs_config.disable_any_whitespace

        if isinstance(self.tokenizer, MistralTokenizer):
            try:
                if self.tokenizer.is_tekken:
                    encoded_vocab = self.tokenizer._vocab
                else:
                    encoded_vocab = [
                        token for token, _ in sorted(
                            self.tokenizer.get_vocab().items(),
                            key=lambda x: x[1],
                        )
                    ]
                stop_token_ids = None
                if (hasattr(self.tokenizer, "eos_token_id") and
                    self.tokenizer.eos_token_id is not None):
                    stop_token_ids = [self.tokenizer.eos_token_id]
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer "
                    f"{type(self.tokenizer)}. The tokenizer should have a "
                    "get_vocab method.") from e
            tokenizer_info = xgr.TokenizerInfo(
                encoded_vocab=encoded_vocab,
                vocab_type=xgr.VocabType.RAW if self.tokenizer.is_tekken
                           else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = \
                self.vllm_config.speculative_config.num_speculative_tokens

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            tags = [
                xgr.StructuralTagItem(
                    begin=s["begin"],
                    schema=json.dumps(s["schema"]),
                    end=s["end"],
                ) for s in s_tag["structures"]
            ]
            structural_tag = xgr.StructuralTag.from_legacy_structural_tag(
                tags, s_tag["triggers"])
            ctx = self.compiler.compile_structural_tag(structural_tag)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)
    _is_terminated: bool = field(default=False, repr=False, hash=False)
    _termination_logged: bool = field(default=False, repr=False, hash=False)

    def __post_init__(self):
        """Initialize base class and audit support."""
        super().__init__()
        self._backend_name = "xgrammar"
        # ✅ 统一：运行时检查，不再使用静态变量
        try:
            from vllm.v1.structured_output.audit_tracker import get_audit_tracker
            self._audit_tracker = get_audit_tracker()
        except ImportError:
            self._audit_tracker = None

    def _get_current_state_id(self) -> str:
        """尽力获取可读的状态ID；若后端不暴露则回退到可追踪信息。"""
        state_fn = getattr(self.matcher, "debug_state", None)
        if callable(state_fn):
            try:
                return state_fn()
            except Exception:
                pass
        return f"tokens={self.num_processed_tokens};terminated={self._is_terminated}"

    def _is_audit_enabled(self) -> bool:
        """✅ 统一的运行时检查方法"""
        return (self._audit_tracker is not None and
                self._audit_tracker.is_enabled())

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM."""
        if self._is_terminated:
            return False

        # 延迟绑定审计上下文（首包/首次进来再绑定）
        if self._request_id is None and self._is_audit_enabled():
            self.set_audit_context(request_id)

        current_state = self._get_current_state_id()

        for token in tokens:
            if not self.matcher.accept_token(token):
                # 失败路径：记录 TOKEN_REJECT
                if self._is_audit_enabled():
                    try:
                        from vllm.v1.structured_output.audit_tracker import AuditEventType
                        self._audit_tracker.record_token_acceptance(
                            request_id=request_id,
                            tokens=[token],
                            accepted=False,
                            current_state=current_state,
                        )
                    except Exception:
                        pass
                logger.error(
                    "Failed to advance FSM for request %s for tokens %s. "
                    "Please file an issue.",
                    request_id, token
                )
                return False

            # 成功路径：逐token状态迁移事件
            self.num_processed_tokens += 1
            new_state = self._get_current_state_id()
            if self._is_audit_enabled():
                try:
                    from vllm.v1.structured_output.audit_tracker import AuditEventType
                    self._audit_tracker.record_event(
                        request_id=request_id,
                        event_type=AuditEventType.STATE_TRANSITION,
                        previous_state_id=current_state,
                        current_state_id=new_state,
                        selected_token=token,
                        metadata={"num_processed_tokens": self.num_processed_tokens},
                    )
                except Exception:
                    pass
            current_state = new_state

        # 批量接受成功事件
        self._is_terminated = self.matcher.is_terminated()
        if self._is_audit_enabled():
            try:
                self._audit_tracker.record_token_acceptance(
                    request_id=request_id,
                    tokens=tokens,
                    accepted=True,
                    current_state=current_state,
                )
            except Exception:
                pass

        # 若刚刚进入终止态，则打 TERMINATION 事件（一次性）
        if self._is_terminated and not self._termination_logged:
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
            self._termination_logged = True

        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        prev_state = self._get_current_state_id()
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

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
        self.matcher.fill_next_token_bitmask(bitmask[idx])

        if self._is_audit_enabled() and self._request_id:
            try:
                bitmask_copy = bitmask[idx].clone()
                self._audit_tracker.record_bitmask_update(
                    request_id=self._request_id,
                    bitmask=bitmask_copy,
                    current_state=self._get_current_state_id(),
                )
            except Exception:
                pass

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
        self._is_terminated = False
        self._termination_logged = False


# ... (保留原有的验证函数，不变)
def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        if obj.get("type") == "array" and any(
                key in obj for key in ("uniqueItems", "contains",
                                       "minContains", "maxContains")):
            return True

        if obj.get("type") == "string" and "format" in obj:
            return True

        if obj.get("type") == "object" and any(
                key in obj for key in ("minProperties", "maxProperties",
                                       "propertyNames", "patternProperties")):
            return True

        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            xgr.Grammar.from_regex(so_params.regex)
        except Exception as err:
            raise ValueError("Failed to transform regex into a grammar: "
                             f"{err}") from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError("Failed to transform choices into a grammar: "
                             "{err}") from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError("Failed to transform json schema into a grammar: "
                             f"{err}") from err

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError("The provided JSON schema contains features not "
                             "supported by xgrammar.")
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. ") from e

        try:
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)
            tags = [
                xgr.StructuralTagItem(
                    begin=s["begin"],
                    schema=json.dumps(s["schema"]),
                    end=s["end"],
                ) for s in s_tag["structures"]
            ]
            structural_tag = xgr.StructuralTag.from_legacy_structural_tag(
                tags, s_tag["triggers"])
            xgr.Grammar.from_structural_tag(structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e