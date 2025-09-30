# audit_integration.py
# 审计功能的集成和配置方案

"""
vLLM Structured Output Audit Integration

This module provides integration points for audit functionality
in vLLM's API responses and configuration.
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json

from vllm.config import VllmConfig
from vllm.v1.structured_output.audit_tracker import (
    get_audit_tracker,
    configure_audit_tracker,
    StructuredOutputAuditTracker
)


@dataclass
class StructuredOutputAuditConfig:
    """Configuration for structured output auditing."""

    # Main switch for audit functionality
    enabled: bool = False

    # What to record
    record_allowed_tokens: bool = False  # Record full allowed token sets (can be large)
    record_full_events: bool = True  # Record all events vs just summaries
    include_grammar_spec: bool = False  # Include grammar spec in audit trail

    # Performance settings
    max_trails_in_memory: int = 1000  # Maximum number of trails to keep in memory
    async_recording: bool = False  # Use async recording for better performance

    # Storage settings
    persist_to_disk: bool = False  # Whether to persist audit trails to disk
    audit_log_dir: Optional[str] = None  # Directory for audit logs

    # API response settings
    include_in_response: bool = True  # Include audit trail in API responses
    response_detail_level: str = "summary"  # "none", "summary", "full"

    @classmethod
    def from_env(cls) -> "StructuredOutputAuditConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.environ.get("VLLM_STRUCTURED_OUTPUT_AUDIT", "false").lower() == "true",
            record_allowed_tokens=os.environ.get("VLLM_AUDIT_RECORD_ALLOWED_TOKENS", "false").lower() == "true",
            record_full_events=os.environ.get("VLLM_AUDIT_RECORD_FULL_EVENTS", "true").lower() == "true",
            include_grammar_spec=os.environ.get("VLLM_AUDIT_INCLUDE_GRAMMAR", "false").lower() == "true",
            max_trails_in_memory=int(os.environ.get("VLLM_AUDIT_MAX_TRAILS", "1000")),
            async_recording=os.environ.get("VLLM_AUDIT_ASYNC", "false").lower() == "true",
            persist_to_disk=os.environ.get("VLLM_AUDIT_PERSIST", "false").lower() == "true",
            audit_log_dir=os.environ.get("VLLM_AUDIT_LOG_DIR"),
            include_in_response=os.environ.get("VLLM_AUDIT_IN_RESPONSE", "true").lower() == "true",
            response_detail_level=os.environ.get("VLLM_AUDIT_RESPONSE_LEVEL", "summary")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "record_allowed_tokens": self.record_allowed_tokens,
            "record_full_events": self.record_full_events,
            "include_grammar_spec": self.include_grammar_spec,
            "max_trails_in_memory": self.max_trails_in_memory,
            "async_recording": self.async_recording,
            "persist_to_disk": self.persist_to_disk,
            "audit_log_dir": self.audit_log_dir,
            "include_in_response": self.include_in_response,
            "response_detail_level": self.response_detail_level
        }


def initialize_audit_system(config: Optional[StructuredOutputAuditConfig] = None) -> StructuredOutputAuditTracker:
    """
    Initialize the global audit system with the given configuration.

    Args:
        config: Audit configuration (if None, will read from environment)

    Returns:
        The configured audit tracker instance
    """
    if config is None:
        config = StructuredOutputAuditConfig.from_env()

    tracker = configure_audit_tracker(
        enabled=config.enabled,
        max_trails=config.max_trails_in_memory,
        record_allowed_tokens=config.record_allowed_tokens,
        record_full_events=config.record_full_events
    )

    if config.enabled:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info(f"Structured output audit system initialized: {config.to_dict()}")

    return tracker


# ==============================================
# API Response Enhancement
# ==============================================

def enhance_completion_response_with_audit(
        response: Dict[str, Any],
        request_id: str,
        audit_config: Optional[StructuredOutputAuditConfig] = None
) -> Dict[str, Any]:
    """
    Enhance API completion response with audit trail data.

    Args:
        response: Original API response dictionary
        request_id: The request ID
        audit_config: Audit configuration

    Returns:
        Enhanced response with audit data
    """
    if audit_config is None:
        audit_config = StructuredOutputAuditConfig.from_env()

    if not audit_config.enabled or not audit_config.include_in_response:
        return response

    tracker = get_audit_tracker()
    if not tracker or not tracker.is_enabled():
        return response

    # Get audit trail for this request
    trail = tracker.get_trail(request_id)
    if trail is None:
        return response

    # Determine what level of detail to include
    if audit_config.response_detail_level == "none":
        return response
    elif audit_config.response_detail_level == "summary":
        # Include only summary statistics
        audit_data = {
            "audit_summary": {
                "backend_type": trail.backend_type,
                "total_steps": trail.total_steps,
                "total_tokens_generated": trail.total_tokens_generated,
                "total_rollbacks": trail.total_rollbacks,
                "total_errors": trail.total_errors,
                "duration_seconds": trail.end_time - trail.start_time if trail.end_time else None
            }
        }
    else:  # "full"
        # Include full audit trail
        audit_data = {
            "audit_trail": trail.to_dict(include_events=True)
        }

    # Add audit data to response
    if "usage" in response:
        response["usage"]["structured_output_audit"] = audit_data
    else:
        response["structured_output_audit"] = audit_data

    return response


# ==============================================
# Request Handler Integration
# ==============================================

class AuditedStructuredOutputRequest:
    """
    Wrapper for structured output requests with audit support.

    This class can be used to wrap existing request handling logic
    to add audit functionality.
    """

    def __init__(self,
                 request_id: str,
                 backend_type: str,
                 grammar_spec: Optional[str] = None,
                 audit_config: Optional[StructuredOutputAuditConfig] = None):
        """
        Initialize audited request.

        Args:
            request_id: Unique request identifier
            backend_type: Type of backend (outlines, xgrammar, etc.)
            grammar_spec: The grammar specification
            audit_config: Audit configuration
        """
        self.request_id = request_id
        self.backend_type = backend_type
        self.grammar_spec = grammar_spec
        # 新（替换以上块）
        self.audit_config = audit_config or StructuredOutputAuditConfig.from_env()
        self.tracker = get_audit_tracker()
        # 若全局 tracker 未启用，但 config 要求启用，则即时初始化全局审计系统
        if (not self.tracker) or (not self.tracker.is_enabled() and self.audit_config.enabled):
            self.tracker = initialize_audit_system(self.audit_config)

        # 启动本次请求的 trail
        if self.tracker and self.tracker.is_enabled():
            self.tracker.start_trail(
                request_id=request_id,
                backend_type=backend_type,
                grammar_spec=grammar_spec if self.audit_config.include_grammar_spec else None
            )

    def finalize(self) -> Optional[Dict[str, Any]]:
        """
        Finalize the audit trail and return audit data.

        Returns:
            Audit trail data if auditing is enabled, None otherwise
        """
        if self.tracker and self.tracker.is_enabled():
            self.tracker.finalize_trail(self.request_id)

            if self.audit_config.persist_to_disk and self.audit_config.audit_log_dir:
                self._persist_to_disk()

            return self.tracker.get_trail_dict(
                self.request_id,
                include_events=self.audit_config.record_full_events
            )
        return None

    def _persist_to_disk(self):
        """Persist audit trail to disk."""
        if not self.audit_config.audit_log_dir:
            return

        import os
        import json
        from datetime import datetime

        trail_data = self.tracker.get_trail_dict(self.request_id, include_events=True)
        if trail_data:
            os.makedirs(self.audit_config.audit_log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_{self.request_id}_{timestamp}.json"
            filepath = os.path.join(self.audit_config.audit_log_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(trail_data, f, indent=2)

    def cleanup(self):
        """Clean up audit trail from memory."""
        if self.tracker:
            self.tracker.cleanup_trail(self.request_id)


# ==============================================
# Engine Integration Points
# ==============================================

def patch_vllm_engine_for_audit():
    """
    Monkey-patch vLLM engine to add audit support.

    This is an alternative integration approach that modifies
    the engine at runtime without changing source code.
    """
    from vllm.v1.engine import LLMEngine

    # Save original methods
    original_init = LLMEngine.__init__

    def audited_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)

        # Initialize audit system
        audit_config = StructuredOutputAuditConfig.from_env()
        if audit_config.enabled:
            initialize_audit_system(audit_config)
            self._audit_config = audit_config
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info("Audit system patched into LLMEngine")

    # Replace methods
    LLMEngine.__init__ = audited_init


# ==============================================
# Environment Variable Configuration
# ==============================================

def setup_audit_from_env():
    """
    Set up audit configuration from environment variables.

    Environment variables:
    - VLLM_STRUCTURED_OUTPUT_AUDIT: Enable/disable auditing (true/false)
    - VLLM_AUDIT_RECORD_ALLOWED_TOKENS: Record allowed token sets (true/false)
    - VLLM_AUDIT_RECORD_FULL_EVENTS: Record all events (true/false)
    - VLLM_AUDIT_INCLUDE_GRAMMAR: Include grammar spec in trail (true/false)
    - VLLM_AUDIT_MAX_TRAILS: Max trails in memory (default: 1000)
    - VLLM_AUDIT_ASYNC: Use async recording (true/false)
    - VLLM_AUDIT_PERSIST: Persist to disk (true/false)
    - VLLM_AUDIT_LOG_DIR: Directory for audit logs
    - VLLM_AUDIT_IN_RESPONSE: Include in API response (true/false)
    - VLLM_AUDIT_RESPONSE_LEVEL: Response detail level (none/summary/full)
    """
    config = StructuredOutputAuditConfig.from_env()
    return initialize_audit_system(config)


# ==============================================
# Utility Functions
# ==============================================

def get_audit_statistics(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Get audit statistics for a specific request.

    Args:
        request_id: The request ID

    Returns:
        Statistics dictionary or None if not found
    """
    tracker = get_audit_tracker()
    if not tracker or not tracker.is_enabled():
        return None

    trail = tracker.get_trail(request_id)
    if trail:
        return {
            "request_id": trail.request_id,
            "backend_type": trail.backend_type,
            "total_steps": trail.total_steps,
            "total_tokens_generated": trail.total_tokens_generated,
            "total_rollbacks": trail.total_rollbacks,
            "total_errors": trail.total_errors,
            "duration": trail.end_time - trail.start_time if trail.end_time else None,
            "events_count": len(trail.events)
        }
    return None


def export_audit_trails_to_file(
        filepath: str,
        request_ids: Optional[List[str]] = None,
        include_events: bool = True
) -> int:
    """
    Export audit trails to a JSON file.

    Args:
        filepath: Path to output file
        request_ids: Specific request IDs to export (None for all)
        include_events: Whether to include full event data

    Returns:
        Number of trails exported
    """
    tracker = get_audit_tracker()
    if not tracker:
        return 0

    trails = tracker.get_all_trails()

    if request_ids:
        trails = {k: v for k, v in trails.items() if k in request_ids}

    export_data = {
        "export_timestamp": time.time(),
        "trails": [trail.to_dict(include_events=include_events)
                   for trail in trails.values()]
    }

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)

    return len(trails)


if __name__ == "__main__":
    # Example usage
    import time

    # Initialize audit system
    config = StructuredOutputAuditConfig(
        enabled=True,
        record_allowed_tokens=False,
        record_full_events=True,
        include_in_response=True,
        response_detail_level="full"
    )

    tracker = initialize_audit_system(config)
    print(f"Audit system initialized: {config.to_dict()}")