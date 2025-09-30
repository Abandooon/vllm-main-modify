# test_audit_functionality.py
# 审计功能测试套件

"""
Test suite for vLLM Structured Output Audit functionality.

This module provides comprehensive tests for the audit tracking system.
"""

import asyncio
import json
import time
import unittest
from typing import Dict, List, Any
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Test imports
import torch
import numpy as np
import os
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Import audit modules
from vllm.v1.structured_output.audit_tracker import (
    StructuredOutputAuditTracker,
    AuditEvent,
    AuditEventType,
    AuditTrail,
    get_audit_tracker,
    configure_audit_tracker
)

from vllm.v1.structured_output.audit_integration import (
    StructuredOutputAuditConfig,
    initialize_audit_system,
    enhance_completion_response_with_audit,
    AuditedStructuredOutputRequest
)


class TestAuditTracker(unittest.TestCase):
    """Test cases for the audit tracker."""

    def setUp(self):
        """Set up test environment."""
        self.tracker = StructuredOutputAuditTracker(
            enabled=True,
            max_trails=10,
            record_allowed_tokens=True,
            record_full_events=True
        )

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertTrue(self.tracker.is_enabled())
        self.assertEqual(self.tracker.max_trails, 10)
        self.assertTrue(self.tracker.record_allowed_tokens)
        self.assertTrue(self.tracker.record_full_events)

    def test_start_trail(self):
        """Test starting a new audit trail."""
        request_id = "test_request_001"
        backend_type = "outlines"
        grammar_spec = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        self.tracker.start_trail(request_id, backend_type, grammar_spec)

        trail = self.tracker.get_trail(request_id)
        self.assertIsNotNone(trail)
        self.assertEqual(trail.request_id, request_id)
        self.assertEqual(trail.backend_type, backend_type)
        self.assertEqual(trail.grammar_spec, grammar_spec)

    def test_record_token_acceptance(self):
        """Test recording token acceptance events."""
        request_id = "test_request_002"
        self.tracker.start_trail(request_id, "xgrammar")

        # Record accepted tokens
        tokens = [101, 102, 103]
        self.tracker.record_token_acceptance(
            request_id=request_id,
            tokens=tokens,
            accepted=True,
            current_state="state_1"
        )

        trail = self.tracker.get_trail(request_id)
        self.assertEqual(trail.total_tokens_generated, 3)
        self.assertEqual(len(trail.events), 1)

        event = trail.events[0]
        self.assertEqual(event.event_type, AuditEventType.TOKEN_ACCEPT)
        self.assertEqual(event.accepted_tokens, tokens)
        self.assertEqual(event.current_state_id, "state_1")

    def test_record_bitmask_update(self):
        """Test recording bitmask update events."""
        request_id = "test_request_003"
        self.tracker.start_trail(request_id, "guidance")

        # Create a sample bitmask
        bitmask = torch.zeros(1000, dtype=torch.int32)
        bitmask[:100] = 1  # First 100 tokens are allowed

        self.tracker.record_bitmask_update(
            request_id=request_id,
            bitmask=bitmask,
            current_state="state_2"
        )

        trail = self.tracker.get_trail(request_id)
        event = trail.events[0]
        self.assertEqual(event.event_type, AuditEventType.BITMASK_UPDATE)
        self.assertEqual(event.allowed_tokens_count, 100)

    def test_record_rollback(self):
        """Test recording rollback events."""
        request_id = "test_request_004"
        self.tracker.start_trail(request_id, "outlines")

        self.tracker.record_rollback(
            request_id=request_id,
            num_tokens=5,
            current_state="state_3"
        )

        trail = self.tracker.get_trail(request_id)
        self.assertEqual(trail.total_rollbacks, 1)

        event = trail.events[0]
        self.assertEqual(event.event_type, AuditEventType.ROLLBACK)
        self.assertEqual(event.metadata["num_tokens_rolled_back"], 5)

    def test_max_trails_limit(self):
        """Test that max_trails limit is enforced."""
        # Create more trails than the limit
        for i in range(15):
            request_id = f"test_request_{i:03d}"
            self.tracker.start_trail(request_id, "test_backend")

        # Should only keep the last 10 trails
        all_trails = self.tracker.get_all_trails()
        self.assertEqual(len(all_trails), 10)

    def test_trail_serialization(self):
        """Test trail serialization to dictionary."""
        request_id = "test_request_005"
        self.tracker.start_trail(request_id, "xgrammar")

        # Add some events
        self.tracker.record_token_acceptance(request_id, [1, 2, 3], True, "state_1")
        self.tracker.record_rollback(request_id, 2, "state_2")

        # Finalize and serialize
        self.tracker.finalize_trail(request_id)
        trail_dict = self.tracker.get_trail_dict(request_id, include_events=True)

        self.assertIsNotNone(trail_dict)
        self.assertEqual(trail_dict["request_id"], request_id)
        self.assertEqual(trail_dict["backend_type"], "xgrammar")
        self.assertEqual(trail_dict["total_tokens_generated"], 3)
        self.assertEqual(trail_dict["total_rollbacks"], 1)
        self.assertEqual(len(trail_dict["events"]), 2)


class TestAuditIntegration(unittest.TestCase):
    """Test cases for audit integration."""

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "VLLM_STRUCTURED_OUTPUT_AUDIT": "true",
            "VLLM_AUDIT_RECORD_ALLOWED_TOKENS": "true",
            "VLLM_AUDIT_MAX_TRAILS": "500",
            "VLLM_AUDIT_RESPONSE_LEVEL": "full"
        }

        with patch.dict(os.environ, env_vars):
            config = StructuredOutputAuditConfig.from_env()

            self.assertTrue(config.enabled)
            self.assertTrue(config.record_allowed_tokens)
            self.assertEqual(config.max_trails_in_memory, 500)
            self.assertEqual(config.response_detail_level, "full")

    def test_audited_request_wrapper(self):
        """Test AuditedStructuredOutputRequest wrapper."""
        request_id = "test_request_006"
        backend_type = "outlines"

        # Create audited request
        audited_request = AuditedStructuredOutputRequest(
            request_id=request_id,
            backend_type=backend_type,
            audit_config=StructuredOutputAuditConfig(enabled=True)
        )

        # Finalize and get audit data
        audit_data = audited_request.finalize()

        self.assertIsNotNone(audit_data)
        self.assertEqual(audit_data["request_id"], request_id)
        self.assertEqual(audit_data["backend_type"], backend_type)

    def test_response_enhancement(self):
        """Test API response enhancement with audit data."""
        # Set up tracker with some data
        tracker = configure_audit_tracker(enabled=True)
        request_id = "test_request_007"
        tracker.start_trail(request_id, "xgrammar")
        tracker.record_token_acceptance(request_id, [1, 2, 3], True, "state_1")
        tracker.finalize_trail(request_id)

        # Create original response
        original_response = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{
                "text": "Generated text",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        # Enhance with audit data
        config = StructuredOutputAuditConfig(
            enabled=True,
            include_in_response=True,
            response_detail_level="summary"
        )

        enhanced_response = enhance_completion_response_with_audit(
            original_response,
            request_id,
            config
        )

        # Check that audit data was added
        self.assertIn("structured_output_audit", enhanced_response["usage"])
        audit_summary = enhanced_response["usage"]["structured_output_audit"]["audit_summary"]
        self.assertEqual(audit_summary["backend_type"], "xgrammar")
        self.assertEqual(audit_summary["total_tokens_generated"], 3)


class TestPerformance(unittest.TestCase):
    """Performance tests for audit functionality."""

    def test_overhead_with_audit_disabled(self):
        """Test that disabled audit has minimal overhead."""
        tracker = StructuredOutputAuditTracker(enabled=False)

        start_time = time.perf_counter()

        # Simulate many operations
        for i in range(10000):
            tracker.record_token_acceptance(f"req_{i}", [1, 2, 3], True, "state")

        elapsed = time.perf_counter() - start_time

        # Should be very fast when disabled
        self.assertLess(elapsed, 0.1)  # Less than 100ms for 10k operations

    def test_memory_usage(self):
        """Test memory usage with many trails."""
        tracker = StructuredOutputAuditTracker(
            enabled=True,
            max_trails=100
        )

        # Create many trails
        for i in range(200):
            request_id = f"test_request_{i:03d}"
            tracker.start_trail(request_id, "test_backend")

            # Add some events to each trail
            for j in range(10):
                tracker.record_token_acceptance(
                    request_id,
                    list(range(j * 10, (j + 1) * 10)),
                    True,
                    f"state_{j}"
                )

        # Check that only max_trails are kept
        all_trails = tracker.get_all_trails()
        self.assertLessEqual(len(all_trails), 100)


class TestBackendIntegration(unittest.TestCase):
    """Test integration with different backends."""

    @patch('vllm.v1.structured_output.backend_outlines.oc')
    def test_outlines_backend_audit(self, mock_oc):
        """Test Outlines backend with audit."""
        from vllm.v1.structured_output.backend_outlines import OutlinesGrammarWithAudit

        # Mock the Guide
        mock_guide = MagicMock()
        mock_guide.accepts_tokens.return_value = True
        mock_guide.is_finished.return_value = False

        # Create grammar with audit support
        grammar = OutlinesGrammarWithAudit(
            vocab_size=32000,
            guide=mock_guide
        )

        # Set up audit context
        grammar.set_audit_context("test_request_008")

        # Test token acceptance
        result = grammar.accept_tokens("test_request_008", [1, 2, 3])

        self.assertTrue(result)
        mock_guide.accepts_tokens.assert_called_once()
        self.assertEqual(mock_guide.advance.call_count, 3)

    def test_audit_data_persistence(self):
        """Test saving audit data to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StructuredOutputAuditConfig(
                enabled=True,
                persist_to_disk=True,
                audit_log_dir=tmpdir
            )

            tracker = initialize_audit_system(config)

            # Create and finalize a request
            request = AuditedStructuredOutputRequest(
                request_id="test_persist_001",
                backend_type="xgrammar",
                audit_config=config
            )

            request.finalize()

            # Check that file was created
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)

            # Verify file content
            with open(os.path.join(tmpdir, files[0]), 'r') as f:
                data = json.load(f)
                self.assertEqual(data["request_id"], "test_persist_001")


def run_performance_benchmark():
    """Run performance benchmark for audit system."""
    print("\n=== Performance Benchmark ===\n")

    # Test different configurations
    configurations = [
        ("Disabled", {"enabled": False}),
        ("Basic", {"enabled": True, "record_allowed_tokens": False}),
        ("Full", {"enabled": True, "record_allowed_tokens": True})
    ]

    results = []

    for name, config_params in configurations:
        tracker = StructuredOutputAuditTracker(**config_params)

        # Warm up
        for _ in range(100):
            tracker.record_token_acceptance("warmup", [1], True, "state")

        # Benchmark
        start = time.perf_counter()
        num_operations = 10000

        for i in range(num_operations):
            request_id = f"bench_{i}"
            tracker.start_trail(request_id, "benchmark")

            # Simulate typical operations
            tracker.record_token_acceptance(request_id, [1, 2, 3], True, "state_1")
            tracker.record_bitmask_update(request_id, torch.zeros(1000), "state_2")
            tracker.record_token_acceptance(request_id, [4, 5], True, "state_3")

            if i % 10 == 0:  # 10% rollback rate
                tracker.record_rollback(request_id, 2, "state_4")

            tracker.finalize_trail(request_id)

        elapsed = time.perf_counter() - start
        ops_per_second = num_operations / elapsed

        results.append({
            "config": name,
            "total_time": elapsed,
            "ops_per_second": ops_per_second,
            "avg_time_per_op_ms": (elapsed / num_operations) * 1000
        })

    # Print results
    print("Configuration | Total Time | Ops/Sec | Avg Time/Op")
    print("-" * 60)
    for r in results:
        print(
            f"{r['config']:12} | {r['total_time']:10.3f}s | {r['ops_per_second']:8.0f} | {r['avg_time_per_op_ms']:6.3f}ms")

    # Calculate overhead
    if len(results) >= 2:
        disabled_time = results[0]["avg_time_per_op_ms"]
        for r in results[1:]:
            overhead = ((r["avg_time_per_op_ms"] - disabled_time) / disabled_time) * 100
            print(f"\nOverhead for {r['config']}: {overhead:.1f}%")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance benchmark
    run_performance_benchmark()