# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Structured Output Audit Tracker Module

This module provides audit tracking functionality for structured output generation
in vLLM, recording state transitions, token selections, and validation information.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict
import threading
import numpy as np
import torch
import os
import socket  # 可选：记录主机名


from vllm.logger import init_logger

logger = init_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events that can be recorded."""
    STATE_INIT = "state_init"
    TOKEN_ACCEPT = "token_accept"
    TOKEN_REJECT = "token_reject"
    TOKEN_VALIDATE = "token_validate"
    STATE_TRANSITION = "state_transition"
    BITMASK_UPDATE = "bitmask_update"
    ROLLBACK = "rollback"
    TERMINATION = "termination"
    ERROR = "error"


@dataclass
class AuditEvent:
    """A single audit event in the structured output generation process."""

    # Basic information
    timestamp: float
    step_number: int
    event_type: AuditEventType
    request_id: str
    timestamp_ns: int = 0  # 纳秒精度

    def __post_init__(self):
        import time
        self.timestamp_ns = time.time_ns()

    # State information
    current_state_id: Optional[str] = None
    previous_state_id: Optional[str] = None

    # Token information
    accepted_tokens: Optional[List[int]] = None
    rejected_tokens: Optional[List[int]] = None
    allowed_tokens: Optional[Set[int]] = None  # Token IDs allowed at this state
    allowed_tokens_count: Optional[int] = None  # Count of allowed tokens
    selected_token: Optional[int] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "timestamp_ns": self.timestamp_ns,
            "step_number": self.step_number,
            "event_type": self.event_type.value,
            "request_id": self.request_id,
        }

        if self.current_state_id is not None:
            result["current_state_id"] = self.current_state_id
        if self.previous_state_id is not None:
            result["previous_state_id"] = self.previous_state_id
        if self.accepted_tokens is not None:
            result["accepted_tokens"] = self.accepted_tokens
        if self.rejected_tokens is not None:
            result["rejected_tokens"] = self.rejected_tokens
        if self.allowed_tokens_count is not None:
            result["allowed_tokens_count"] = self.allowed_tokens_count
        if self.selected_token is not None:
            result["selected_token"] = self.selected_token
        if self.metadata:
            result["metadata"] = self.metadata
        if self.error_message:
            result["error_message"] = self.error_message

        # Don't include full allowed_tokens set in JSON to save space
        # Can be optionally included based on configuration

        return result


@dataclass
class AuditTrail:
    """Complete audit trail for a structured output request."""

    request_id: str
    backend_type: str  # "outlines", "xgrammar", "guidance", etc.
    grammar_spec: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    events: List[AuditEvent] = field(default_factory=list)

    # Summary statistics
    total_steps: int = 0
    total_tokens_generated: int = 0
    total_rollbacks: int = 0
    total_errors: int = 0

    def add_event(self, event: AuditEvent) -> None:
        """Add an audit event to the trail."""
        self.events.append(event)
        self.total_steps = len(self.events)

        # Update statistics
        if event.event_type == AuditEventType.TOKEN_ACCEPT:
            if event.accepted_tokens:
                self.total_tokens_generated += len(event.accepted_tokens)
        elif event.event_type == AuditEventType.ROLLBACK:
            self.total_rollbacks += 1
        elif event.event_type == AuditEventType.ERROR:
            self.total_errors += 1

    def finalize(self) -> None:
        """Mark the audit trail as complete."""
        self.end_time = time.time()

    def to_dict(self, include_events: bool = True) -> Dict[str, Any]:
        """Convert audit trail to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "backend_type": self.backend_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
            "total_rollbacks": self.total_rollbacks,
            "total_errors": self.total_errors,
        }

        if self.grammar_spec and len(self.grammar_spec) < 1000:
            result["grammar_spec"] = self.grammar_spec

        if include_events:
            result["events"] = [e.to_dict() for e in self.events]

        return result


class StructuredOutputAuditTracker:
    def __init__(self,
                 enabled: bool = False,
                 max_trails: int = 1000,
                 record_allowed_tokens: bool = False,
                 record_full_events: bool = True,
                 persist_to_disk: bool = False,
                 log_dir: Optional[str] = None):
        self.enabled = enabled
        self.max_trails = max_trails
        self.record_allowed_tokens = record_allowed_tokens
        self.record_full_events = record_full_events

        # 新增：持久化配置
        self.persist_to_disk = persist_to_disk
        self.log_dir = log_dir
        self.pid = os.getpid()
        self.hostname = socket.gethostname() if hasattr(socket, "gethostname") else "unknown-host"

        self._trails: Dict[str, AuditTrail] = {}
        self._lock = threading.Lock()
        self._step_counters: Dict[str, int] = defaultdict(int)

        # 预先组好日志文件路径（所有 EngineCore 进程 + APIServer 进程都会往同一卷 append）
        self._ndjson_path = None
        if self.persist_to_disk and self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self._ndjson_path = os.path.join(self.log_dir, "audit.ndjson")
            except Exception as e:
                logger.warning(f"[AuditPersist] Failed to init log dir {self.log_dir}: {e}")
                self._ndjson_path = None

        if self.enabled:
            logger.info("Structured output audit tracking enabled (persist_to_disk=%s log_dir=%s pid=%s)",
                        self.persist_to_disk, self.log_dir, self.pid)


    def _persist_line(self, record_type: str, payload: Dict[str, Any]) -> None:
        """
        以 NDJSON 形式把一条审计记录落盘，用于跨进程共享。

        record_type: "start_trail" | "event" | "finalize"
        payload: 会至少包含 request_id, 其它字段任意
        """
        if not (self.persist_to_disk and self._ndjson_path):
            return

        try:
            line = {
                "ts": time.time(),
                "pid": self.pid,
                "host": self.hostname,
                "record_type": record_type,
                **payload,
            }
            with open(self._ndjson_path, "a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[AuditPersist] Failed writing audit line: {e}")



    def is_enabled(self) -> bool:
        """Check if audit tracking is enabled."""
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable audit tracking."""
        self.enabled = enabled
        if enabled:
            logger.info("Structured output audit tracking enabled")
        else:
            logger.info("Structured output audit tracking disabled")

    def start_trail(self,
                    request_id: str,
                    backend_type: str,
                    grammar_spec: Optional[str] = None) -> None:
        if not self.enabled:
            logger.warning(
                f"[Audit] start_trail() skipped: tracker.enabled={self.enabled} "
                f"request_id={request_id}"
            )
            return

        with self._lock:
            if request_id in self._trails:
                logger.debug(f"[Audit] Trail for {request_id} already exists, skipping restart")
                return

            # 容量控制...
            if len(self._trails) >= self.max_trails:
                oldest_id = min(self._trails.keys(),
                                key=lambda k: self._trails[k].start_time)
                del self._trails[oldest_id]
                if oldest_id in self._step_counters:
                    del self._step_counters[oldest_id]

            self._trails[request_id] = AuditTrail(
                request_id=request_id,
                backend_type=backend_type,
                grammar_spec=grammar_spec
            )
            self._step_counters[request_id] = 0

            logger.warning(
                f"[Audit] start_trail(): CREATED trail for {request_id} "
                f"backend={backend_type} tracker_id={id(self)} pid={self.pid}"
            )

            # 🌟新增：把起始信息落盘
            self._persist_line(
                "start_trail",
                {
                    "request_id": request_id,
                    "backend_type": backend_type,
                    "start_time": self._trails[request_id].start_time,
                    "grammar_spec": grammar_spec if grammar_spec and len(grammar_spec) < 1000 else None,
                },
            )

    def record_event(self,
                     request_id: str,
                     event_type: AuditEventType,
                     **kwargs) -> None:
        if not self.enabled or not self.record_full_events:
            return

        with self._lock:
            if request_id not in self._trails:
                logger.warning(f"No audit trail found for request {request_id}")
                return

            self._step_counters[request_id] += 1

            event = AuditEvent(
                timestamp=time.time(),
                step_number=self._step_counters[request_id],
                event_type=event_type,
                request_id=request_id,
                **kwargs
            )

            self._trails[request_id].add_event(event)

            # 🌟新增：把事件落盘
            try:
                self._persist_line(
                    "event",
                    {
                        "request_id": request_id,
                        "event": event.to_dict(),
                    },
                )
            except Exception as e:
                logger.warning(f"[AuditPersist] Failed to persist event for {request_id}: {e}")

    def record_token_acceptance(self,
                                request_id: str,
                                tokens: List[int],
                                accepted: bool,
                                current_state: Optional[str] = None) -> None:
        """Record token acceptance or rejection."""
        if not self.enabled:
            return

        self.record_event(
            request_id=request_id,
            event_type=AuditEventType.TOKEN_ACCEPT if accepted else AuditEventType.TOKEN_REJECT,
            accepted_tokens=tokens if accepted else None,
            rejected_tokens=tokens if not accepted else None,
            current_state_id=current_state
        )

    def record_bitmask_update(self,
                              request_id: str,
                              bitmask: Union[torch.Tensor, np.ndarray],
                              current_state: Optional[str] = None) -> None:
        """Record bitmask update with allowed tokens."""
        if not self.enabled:
            return

        # 计算允许的token数量
        if isinstance(bitmask, torch.Tensor):
            allowed_count = int(torch.count_nonzero(bitmask > 0).item())
        else:
            allowed_count = int(np.count_nonzero(bitmask > 0))

        kwargs = {
            "allowed_tokens_count": allowed_count,
            "current_state_id": current_state
        }

        # 仅在明确开启且规模小的时候记录完整集合，避免内存膨胀
        if self.record_allowed_tokens and allowed_count < 1000:
            if isinstance(bitmask, torch.Tensor):
                allowed_tokens = set(torch.where(bitmask > 0)[0].tolist())
            else:
                allowed_tokens = set(np.where(bitmask > 0)[0].tolist())
            kwargs["allowed_tokens"] = allowed_tokens

        self.record_event(
            request_id=request_id,
            event_type=AuditEventType.BITMASK_UPDATE,
            **kwargs
        )

    def record_rollback(self,
                        request_id: str,
                        num_tokens: int,
                        current_state: Optional[str] = None) -> None:
        """Record a rollback event."""
        if not self.enabled:
            return

        self.record_event(
            request_id=request_id,
            event_type=AuditEventType.ROLLBACK,
            metadata={"num_tokens_rolled_back": num_tokens},
            current_state_id=current_state
        )

    def finalize_trail(self, request_id: str) -> None:
        """Finalize the audit trail for a request and persist summary (includes end_time)."""
        if not self.enabled:
            return

        with self._lock:
            trail = self._trails.get(request_id)
            if trail is None:
                return

            # 1. 内存里写上 end_time
            trail.finalize()  # 这里会把 trail.end_time = time.time()，也能计算 duration 等

            # 2. 把最终状态（不含所有 events，避免爆炸）落盘
            self._persist_line(
                "finalize",
                {
                    "request_id": request_id,
                    "summary": trail.to_dict(include_events=False),
                },
            )

    def get_trail(self, request_id: str) -> Optional[AuditTrail]:
        """Get the audit trail for a request."""
        if not self.enabled:
            return None

        with self._lock:
            return self._trails.get(request_id)

    def get_trail_dict(self,
                       request_id: str,
                       include_events: bool = True) -> Optional[Dict[str, Any]]:
        """Get the audit trail as a dictionary."""
        trail = self.get_trail(request_id)
        if trail:
            return trail.to_dict(include_events=include_events)
        return None

    def cleanup_trail(self, request_id: str) -> None:
        """Remove the audit trail for a request."""
        with self._lock:
            if request_id in self._trails:
                del self._trails[request_id]
            if request_id in self._step_counters:
                del self._step_counters[request_id]

    def get_all_trails(self) -> Dict[str, AuditTrail]:
        """Get all audit trails (for debugging/monitoring)."""
        with self._lock:
            return dict(self._trails)


# Global audit tracker instance
_global_audit_tracker = None


# audit_tracker.py
def get_audit_tracker() -> StructuredOutputAuditTracker:
    global _global_audit_tracker
    if _global_audit_tracker is None:
        import os
        enabled_raw = os.environ.get("VLLM_STRUCTURED_OUTPUT_AUDIT", "false")
        record_allowed_raw = os.environ.get("VLLM_AUDIT_RECORD_ALLOWED_TOKENS", "false")
        persist_raw = os.environ.get("VLLM_AUDIT_PERSIST", "false")
        log_dir = os.environ.get("VLLM_AUDIT_LOG_DIR")

        logger.warning(
            f"[AuditInit] get_audit_tracker(): "
            f"VLLM_STRUCTURED_OUTPUT_AUDIT={enabled_raw} "
            f"VLLM_AUDIT_RECORD_ALLOWED_TOKENS={record_allowed_raw} "
            f"VLLM_AUDIT_PERSIST={persist_raw} "
            f"VLLM_AUDIT_LOG_DIR={log_dir}"
        )

        enabled = enabled_raw.lower() == "true"
        record_allowed_tokens = record_allowed_raw.lower() == "true"
        persist_to_disk = persist_raw.lower() == "true"

        _global_audit_tracker = StructuredOutputAuditTracker(
            enabled=enabled,
            record_allowed_tokens=record_allowed_tokens,
            persist_to_disk=persist_to_disk,
            log_dir=log_dir,
        )
        logger.warning(
            f"[AuditInit] tracker created: enabled={_global_audit_tracker.enabled} "
            f"id={id(_global_audit_tracker)} persist_to_disk={_global_audit_tracker.persist_to_disk} "
            f"log_dir={_global_audit_tracker.log_dir}"
        )
    return _global_audit_tracker




def configure_audit_tracker(enabled: bool = False,
                            max_trails: int = 1000,
                            record_allowed_tokens: bool = False,
                            record_full_events: bool = True) -> StructuredOutputAuditTracker:
    """Configure and return the global audit tracker."""
    global _global_audit_tracker
    _global_audit_tracker = StructuredOutputAuditTracker(
        enabled=enabled,
        max_trails=max_trails,
        record_allowed_tokens=record_allowed_tokens,
        record_full_events=record_full_events
    )
    return _global_audit_tracker