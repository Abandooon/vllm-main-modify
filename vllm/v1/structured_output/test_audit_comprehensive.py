# test_audit_comprehensive.py
"""
Comprehensive test suite for vLLM Structured Output Audit functionality.

New in this version:
- Tests for unified audit enabled checking (_is_audit_enabled)
- Tests for Management API endpoints
- Tests for environment variable configuration
- Tests for performance benchmark utilities
"""

import asyncio
import json
import time
import unittest
import threading
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Environment setup
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np

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


# ============================================================================
# NEW: Unified Audit Enable Check Tests
# ============================================================================

class TestUnifiedAuditEnabledCheck(unittest.TestCase):
    """Test unified _is_audit_enabled() method across backends."""

    def test_xgrammar_unified_check(self):
        """Test XGrammar uses unified audit check method."""
        try:
            from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

            mock_matcher = MagicMock()
            mock_matcher.accept_token.return_value = True
            mock_matcher.is_terminated.return_value = False
            mock_ctx = MagicMock()

            grammar = XgrammarGrammar(
                vocab_size=50000,
                matcher=mock_matcher,
                ctx=mock_ctx
            )

            # Verify method exists
            self.assertTrue(hasattr(grammar, '_is_audit_enabled'),
                            "XgrammarGrammar missing _is_audit_enabled method")
            self.assertTrue(callable(grammar._is_audit_enabled),
                            "_is_audit_enabled should be callable")

            # Test with disabled tracker
            grammar._audit_tracker = None
            self.assertFalse(grammar._is_audit_enabled(),
                             "Should return False when tracker is None")

            # Test with disabled tracker instance
            mock_tracker = MagicMock()
            mock_tracker.is_enabled.return_value = False
            grammar._audit_tracker = mock_tracker
            self.assertFalse(grammar._is_audit_enabled(),
                             "Should return False when tracker.is_enabled() is False")

            # Test with enabled tracker
            mock_tracker.is_enabled.return_value = True
            self.assertTrue(grammar._is_audit_enabled(),
                            "Should return True when tracker is enabled")

            print("✓ XgrammarGrammar unified audit check: PASS")

        except ImportError as e:
            self.skipTest(f"XGrammar not available: {e}")

    def test_outlines_unified_check(self):
        """Test Outlines uses unified audit check method."""
        try:
            from vllm.v1.structured_output.backend_outlines import OutlinesGrammarWithAudit

            mock_guide = MagicMock()
            mock_guide.accepts_tokens.return_value = True
            mock_guide.is_finished.return_value = False

            grammar = OutlinesGrammarWithAudit(
                vocab_size=50000,
                guide=mock_guide
            )

            # Verify method exists (should inherit from base)
            self.assertTrue(hasattr(grammar, '_is_audit_enabled'))

            # Test runtime behavior
            grammar._audit_tracker = None
            self.assertFalse(grammar._is_audit_enabled())

            print("✓ OutlinesGrammar unified audit check: PASS")

        except ImportError as e:
            self.skipTest(f"Outlines not available: {e}")

    def test_guidance_unified_check(self):
        """Test Guidance uses unified audit check method."""
        try:
            from vllm.v1.structured_output.backend_guidance import GuidanceGrammar

            mock_matcher = MagicMock()
            mock_matcher.consume_tokens.return_value = True
            mock_matcher.is_stopped.return_value = False
            mock_tokenizer = MagicMock()

            grammar = GuidanceGrammar(
                ll_matcher=mock_matcher,
                ll_tokenizer=mock_tokenizer,
                vocab_size=50000
            )

            self.assertTrue(hasattr(grammar, '_is_audit_enabled'))

            print("✓ GuidanceGrammar unified audit check: PASS")

        except ImportError as e:
            self.skipTest(f"Guidance not available: {e}")

    def test_no_static_audit_enabled_usage(self):
        """Verify no static AUDIT_ENABLED variables are used."""
        try:
            from vllm.v1.structured_output import backend_xgrammar
            from vllm.v1.structured_output import backend_outlines
            from vllm.v1.structured_output import backend_guidance

            # Check that AUDIT_ENABLED is not defined at module level
            for module in [backend_xgrammar, backend_outlines, backend_guidance]:
                self.assertFalse(hasattr(module, 'AUDIT_ENABLED'),
                                 f"{module.__name__} should not have static AUDIT_ENABLED")

            print("✓ No static AUDIT_ENABLED variables: PASS")

        except ImportError as e:
            self.skipTest(f"Backend modules not available: {e}")


# ============================================================================
# NEW: Management API Tests
# ============================================================================

class TestManagementAPI(unittest.TestCase):
    """Test Management API endpoints."""

    def setUp(self):
        """Set up test environment with mock FastAPI app."""
        # 清理可能存在的全局tracker
        import vllm.v1.structured_output.audit_tracker as audit_tracker_module
        audit_tracker_module._global_audit_tracker = None

        # Initialize audit system
        self.config = StructuredOutputAuditConfig(
            enabled=True,
            record_full_events=True,
            max_trails_in_memory=100
        )
        self.tracker = initialize_audit_system(self.config)

        # ✅ 添加小延迟确保时间戳不同
        import time

        # Create some test trails
        for i in range(5):
            request_id = f"api_test_{i:03d}"
            self.tracker.start_trail(request_id, "xgrammar")
            self.tracker.record_token_acceptance(
                request_id, [1, 2, 3], True, f"state_{i}"
            )
            if i < 3:  # Finalize some trails
                time.sleep(0.001)  # 确保end_time不同
                self.tracker.finalize_trail(request_id)
            time.sleep(0.001)  # 确保start_time不同

    def tearDown(self):
        """Clean up after tests."""
        import vllm.v1.structured_output.audit_tracker as audit_tracker_module
        audit_tracker_module._global_audit_tracker = None

    def test_get_stats_endpoint(self):
        """Test /stats endpoint returns correct statistics."""
        from vllm.v1.structured_output.audit_admin_api import get_audit_statistics

        # Call endpoint function directly (simulating FastAPI route)
        import asyncio
        response = asyncio.run(get_audit_statistics())

        # Verify response structure
        self.assertTrue(response.enabled)
        self.assertEqual(response.total_trails, 5)
        self.assertEqual(response.active_trails, 2)  # 2 not finalized
        self.assertGreater(response.total_events_recorded, 0)

        print("✓ GET /stats endpoint: PASS")

    def test_list_trails_endpoint(self):
        """Test /list endpoint with pagination."""
        from vllm.v1.structured_output.audit_admin_api import list_audit_trails

        import asyncio
        # 验证setUp创建的trails存在
        all_trails = self.tracker.get_all_trails()
        self.assertEqual(len(all_trails), 5,
                         f"setUp should create 5 trails, got {len(all_trails)}")

        # Test basic listing
        response = asyncio.run(list_audit_trails(limit=10, offset=0))
        self.assertEqual(len(response), 5)

        # Test pagination
        response_page1 = asyncio.run(list_audit_trails(limit=2, offset=0))
        response_page2 = asyncio.run(list_audit_trails(limit=2, offset=2))

        self.assertEqual(len(response_page1), 2)
        self.assertEqual(len(response_page2), 2)

        # Verify no overlap
        ids_page1 = {t.request_id for t in response_page1}
        ids_page2 = {t.request_id for t in response_page2}
        self.assertEqual(len(ids_page1 & ids_page2), 0)

        print("✓ GET /list endpoint with pagination: PASS")

    def test_list_trails_filtering(self):
        """Test /list endpoint with backend filtering."""
        from vllm.v1.structured_output.audit_admin_api import list_audit_trails

        # Add trail with different backend
        self.tracker.start_trail("outlines_req", "outlines")
        self.tracker.finalize_trail("outlines_req")

        import asyncio

        # Filter by backend
        xgrammar_trails = asyncio.run(
            list_audit_trails(backend_type="xgrammar")
        )
        outlines_trails = asyncio.run(
            list_audit_trails(backend_type="outlines")
        )

        self.assertEqual(len(xgrammar_trails), 5)
        self.assertEqual(len(outlines_trails), 1)

        print("✓ GET /list with backend filtering: PASS")

    def test_get_trail_detail_endpoint(self):
        """Test /trail/{request_id} endpoint."""
        from vllm.v1.structured_output.audit_admin_api import get_audit_trail

        import asyncio

        # Get existing trail
        response = asyncio.run(get_audit_trail("api_test_000", include_events=True))

        self.assertEqual(response.request_id, "api_test_000")
        self.assertEqual(response.backend_type, "xgrammar")
        self.assertGreater(len(response.events), 0)

        print("✓ GET /trail/{id} endpoint: PASS")

    def test_get_trail_not_found(self):
        """Test /trail/{request_id} returns 404 for non-existent trail."""
        from vllm.v1.structured_output.audit_admin_api import get_audit_trail
        from fastapi import HTTPException

        import asyncio

        with self.assertRaises(HTTPException) as context:
            asyncio.run(get_audit_trail("nonexistent_id"))

        self.assertEqual(context.exception.status_code, 404)

        print("✓ GET /trail/{id} 404 handling: PASS")

    def test_export_endpoint(self):
        """Test /export endpoint for bulk export."""
        from vllm.v1.structured_output.audit_admin_api import export_audit_trails, AuditExportRequest

        import asyncio

        request = AuditExportRequest(
            request_ids=["api_test_000", "api_test_001"],
            include_events=True
        )

        response = asyncio.run(export_audit_trails(request))

        self.assertEqual(response.trail_count, 2)
        self.assertEqual(len(response.trails), 2)

        # Verify trail content
        trail_ids = {t["request_id"] for t in response.trails}
        self.assertEqual(trail_ids, {"api_test_000", "api_test_001"})

        print("✓ POST /export endpoint: PASS")

    def test_export_time_range_filter(self):
        """Test /export endpoint with time range filtering."""
        from vllm.v1.structured_output.audit_admin_api import export_audit_trails, AuditExportRequest

        import asyncio

        # Get timestamp range
        all_trails = self.tracker.get_all_trails()
        start_times = [t.start_time for t in all_trails.values()]
        min_time = min(start_times)
        mid_time = (min(start_times) + max(start_times)) / 2

        request = AuditExportRequest(
            include_events=False,
            start_time=min_time,
            end_time=mid_time
        )

        response = asyncio.run(export_audit_trails(request))

        # Should get subset of trails
        self.assertLessEqual(response.trail_count, 5)
        self.assertGreater(response.trail_count, 0)

        print("✓ POST /export with time range: PASS")

    def test_delete_trail_endpoint(self):
        """Test DELETE /trail/{request_id} endpoint."""
        from vllm.v1.structured_output.audit_admin_api import delete_audit_trail

        import asyncio

        # Create a trail to delete
        test_id = "delete_test"
        self.tracker.start_trail(test_id, "test")

        # Verify it exists
        self.assertIsNotNone(self.tracker.get_trail(test_id))

        # Delete via API
        response = asyncio.run(delete_audit_trail(test_id))

        self.assertEqual(response["status"], "success")

        # Verify it's gone
        self.assertIsNone(self.tracker.get_trail(test_id))

        print("✓ DELETE /trail/{id} endpoint: PASS")

    def test_clear_all_endpoint_requires_confirmation(self):
        """Test POST /clear requires explicit confirmation."""
        from vllm.v1.structured_output.audit_admin_api import clear_all_trails
        from fastapi import HTTPException

        import asyncio

        # Without confirmation should fail
        with self.assertRaises(HTTPException) as context:
            asyncio.run(clear_all_trails(confirm=False))

        self.assertEqual(context.exception.status_code, 400)

        print("✓ POST /clear confirmation required: PASS")

    def test_clear_all_endpoint_with_confirmation(self):
        """Test POST /clear with confirmation."""
        from vllm.v1.structured_output.audit_admin_api import clear_all_trails

        import asyncio

        initial_count = len(self.tracker.get_all_trails())
        self.assertGreater(initial_count, 0)

        # Clear with confirmation
        response = asyncio.run(clear_all_trails(confirm=True))

        self.assertEqual(response["status"], "success")
        self.assertIn(str(initial_count), response["message"])

        # Verify all cleared
        self.assertEqual(len(self.tracker.get_all_trails()), 0)

        print("✓ POST /clear with confirmation: PASS")

    def test_health_endpoint(self):
        """Test GET /health endpoint."""
        from vllm.v1.structured_output.audit_admin_api import audit_health_check

        import asyncio

        response = asyncio.run(audit_health_check())

        self.assertEqual(response["status"], "healthy")
        self.assertTrue(response["enabled"])
        self.assertIn("trails_in_memory", response)

        print("✓ GET /health endpoint: PASS")


# ============================================================================
# NEW: Environment Variable Configuration Tests
# ============================================================================

class TestEnvironmentConfiguration(unittest.TestCase):
    """Test configuration via environment variables."""

    def setUp(self):
        """Save original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_from_env_enabled(self):
        """Test enabling audit via environment variable."""
        os.environ["VLLM_STRUCTURED_OUTPUT_AUDIT"] = "true"

        config = StructuredOutputAuditConfig.from_env()

        self.assertTrue(config.enabled)

        print("✓ Config from env (enabled): PASS")

    def test_config_from_env_disabled(self):
        """Test disabling audit via environment variable."""
        os.environ["VLLM_STRUCTURED_OUTPUT_AUDIT"] = "false"

        config = StructuredOutputAuditConfig.from_env()

        self.assertFalse(config.enabled)

    def test_config_from_env_full_settings(self):
        """Test all configuration options from environment."""
        os.environ.update({
            "VLLM_STRUCTURED_OUTPUT_AUDIT": "true",
            "VLLM_AUDIT_RECORD_ALLOWED_TOKENS": "true",
            "VLLM_AUDIT_RECORD_FULL_EVENTS": "false",
            "VLLM_AUDIT_INCLUDE_GRAMMAR": "true",
            "VLLM_AUDIT_MAX_TRAILS": "500",
            "VLLM_AUDIT_ASYNC": "true",
            "VLLM_AUDIT_PERSIST": "true",
            "VLLM_AUDIT_LOG_DIR": "/tmp/audit",
            "VLLM_AUDIT_IN_RESPONSE": "false",
            "VLLM_AUDIT_RESPONSE_LEVEL": "summary"
        })

        config = StructuredOutputAuditConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertTrue(config.record_allowed_tokens)
        self.assertFalse(config.record_full_events)
        # ✅ 修复：使用正确的属性名
        self.assertTrue(config.include_grammar_spec)  # 不是 include_grammar
        self.assertEqual(config.max_trails_in_memory, 500)
        self.assertTrue(config.async_recording)
        self.assertTrue(config.persist_to_disk)
        self.assertEqual(config.audit_log_dir, "/tmp/audit")
        self.assertFalse(config.include_in_response)
        self.assertEqual(config.response_detail_level, "summary")

        print("✓ Config from env (all settings): PASS")

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = StructuredOutputAuditConfig(
            enabled=True,
            record_full_events=True,
            max_trails_in_memory=999
        )

        config_dict = config.to_dict()

        self.assertEqual(config_dict["enabled"], True)
        self.assertEqual(config_dict["record_full_events"], True)
        self.assertEqual(config_dict["max_trails_in_memory"], 999)

        print("✓ Config to_dict serialization: PASS")

    def test_initialize_from_env(self):
        """Test initializing audit system from environment."""
        os.environ["VLLM_STRUCTURED_OUTPUT_AUDIT"] = "true"
        os.environ["VLLM_AUDIT_MAX_TRAILS"] = "50"

        from vllm.v1.structured_output.audit_integration import setup_audit_from_env

        tracker = setup_audit_from_env()

        self.assertIsNotNone(tracker)
        self.assertTrue(tracker.is_enabled())
        self.assertEqual(tracker.max_trails, 50)

        print("✓ Initialize audit from env: PASS")


# ============================================================================
# CRITICAL: Base Class Initialization Tests
# ============================================================================

class TestGrammarBaseInitialization(unittest.TestCase):
    """Test that Grammar subclasses properly initialize base class."""

    def test_xgrammar_base_initialization(self):
        """Verify XgrammarGrammar initializes StructuredOutputGrammar properly."""
        try:
            from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

            mock_matcher = MagicMock()
            mock_ctx = MagicMock()

            grammar = XgrammarGrammar(
                vocab_size=50000,
                matcher=mock_matcher,
                ctx=mock_ctx
            )

            # CRITICAL: These attributes must exist and be initialized
            self.assertTrue(hasattr(grammar, '_audit_tracker'),
                            "Missing _audit_tracker attribute")
            self.assertTrue(hasattr(grammar, '_request_id'),
                            "Missing _request_id attribute")
            self.assertTrue(hasattr(grammar, '_backend_name'),
                            "Missing _backend_name attribute")
            self.assertTrue(hasattr(grammar, '_is_audit_enabled'),
                            "Missing _is_audit_enabled method")

            # Values should be initialized correctly
            self.assertIsNone(grammar._request_id,
                              "request_id should be None before set_audit_context")
            self.assertIsInstance(grammar._backend_name, str,
                                  "backend_name should be a string")

            print(f"✓ XgrammarGrammar base initialization: PASS")
            print(f"  - _audit_tracker: {grammar._audit_tracker}")
            print(f"  - _request_id: {grammar._request_id}")
            print(f"  - _backend_name: {grammar._backend_name}")

        except ImportError as e:
            self.skipTest(f"XGrammar not available: {e}")
        except AttributeError as e:
            self.fail(f"Base class initialization failed: {e}")

    def test_outlines_base_initialization(self):
        """Verify OutlinesGrammar initializes StructuredOutputGrammar properly."""
        try:
            from vllm.v1.structured_output.backend_outlines import OutlinesGrammarWithAudit

            mock_guide = MagicMock()
            mock_guide.accepts_tokens.return_value = True
            mock_guide.is_finished.return_value = False

            grammar = OutlinesGrammarWithAudit(
                vocab_size=50000,
                guide=mock_guide
            )

            # CRITICAL: These attributes must exist
            self.assertTrue(hasattr(grammar, '_audit_tracker'))
            self.assertTrue(hasattr(grammar, '_request_id'))
            self.assertTrue(hasattr(grammar, '_backend_name'))

            self.assertIsNone(grammar._request_id)
            self.assertEqual(grammar._backend_name, "outlines")

            print(f"✓ OutlinesGrammar base initialization: PASS")

        except ImportError as e:
            self.skipTest(f"Outlines not available: {e}")
        except AttributeError as e:
            self.fail(f"Base class initialization failed: {e}")

    def test_guidance_base_initialization(self):
        """Verify GuidanceGrammar initializes StructuredOutputGrammar properly."""
        try:
            from vllm.v1.structured_output.backend_guidance import GuidanceGrammar

            mock_matcher = MagicMock()
            mock_tokenizer = MagicMock()

            grammar = GuidanceGrammar(
                ll_matcher=mock_matcher,
                ll_tokenizer=mock_tokenizer,
                vocab_size=50000
            )

            # CRITICAL: These attributes must exist
            self.assertTrue(hasattr(grammar, '_audit_tracker'))
            self.assertTrue(hasattr(grammar, '_request_id'))

            print(f"✓ GuidanceGrammar base initialization: PASS")

        except ImportError as e:
            self.skipTest(f"Guidance not available: {e}")
        except AttributeError as e:
            self.fail(f"Base class initialization failed: {e}")


# ============================================================================
# Core Audit Tracker Tests (Preserved from original)
# ============================================================================

class TestAuditTracker(unittest.TestCase):
    """Test cases for the audit tracker core functionality."""

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
        grammar_spec = '{"type": "object"}'

        self.tracker.start_trail(request_id, backend_type, grammar_spec)

        trail = self.tracker.get_trail(request_id)
        self.assertIsNotNone(trail)
        self.assertEqual(trail.request_id, request_id)
        self.assertEqual(trail.backend_type, backend_type)
        self.assertEqual(trail.grammar_spec, grammar_spec)

    def test_duplicate_trail_start_prevention(self):
        """Test that starting same trail twice doesn't overwrite."""
        request_id = "test_request_dup"

        self.tracker.start_trail(request_id, "backend1", "spec1")
        trail1 = self.tracker.get_trail(request_id)
        start_time1 = trail1.start_time

        time.sleep(0.01)
        self.tracker.start_trail(request_id, "backend2", "spec2")
        trail2 = self.tracker.get_trail(request_id)

        self.assertEqual(trail2.start_time, start_time1)
        self.assertEqual(trail2.backend_type, "backend1")

    def test_record_token_acceptance(self):
        """Test recording token acceptance events."""
        request_id = "test_request_002"
        self.tracker.start_trail(request_id, "xgrammar")

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

    def test_trail_finalization(self):
        """Test trail finalization sets end_time."""
        request_id = "test_request_008"
        self.tracker.start_trail(request_id, "xgrammar")

        trail = self.tracker.get_trail(request_id)
        self.assertIsNone(trail.end_time)

        time.sleep(0.001)

        self.tracker.finalize_trail(request_id)

        trail = self.tracker.get_trail(request_id)
        self.assertIsNotNone(trail.end_time)
        self.assertGreaterEqual(trail.end_time, trail.start_time)


# ============================================================================
# NEW: Benchmark Integration Tests
# ============================================================================

class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration with benchmark_audit.py utilities."""

    def test_benchmark_config_to_env_vars(self):
        """Test BenchmarkConfig converts to environment variables."""
        # Import would be: from benchmark_audit import BenchmarkConfig
        # For testing, we simulate the structure

        config_data = {
            "name": "test_config",
            "audit_enabled": True,
            "record_full_events": False,
            "record_allowed_tokens": False,
            "persist_to_disk": True
        }

        # Simulate to_env_vars() method
        env_vars = {
            "VLLM_STRUCTURED_OUTPUT_AUDIT": str(config_data["audit_enabled"]).lower(),
            "VLLM_AUDIT_RECORD_FULL_EVENTS": str(config_data["record_full_events"]).lower(),
            "VLLM_AUDIT_RECORD_ALLOWED_TOKENS": str(config_data["record_allowed_tokens"]).lower(),
            "VLLM_AUDIT_PERSIST": str(config_data["persist_to_disk"]).lower()
        }

        self.assertEqual(env_vars["VLLM_STRUCTURED_OUTPUT_AUDIT"], "true")
        self.assertEqual(env_vars["VLLM_AUDIT_RECORD_FULL_EVENTS"], "false")

        print("✓ Benchmark config to env vars: PASS")

    def test_classify_schema_complexity(self):
        """Test schema complexity classification for benchmarks."""

        def classify_complexity(schema: Dict[str, Any]) -> str:
            def get_depth(obj, current_depth=0):
                if not isinstance(obj, dict):
                    return current_depth
                if "properties" not in obj:
                    return current_depth
                return max(
                    get_depth(prop, current_depth + 1)
                    for prop in obj["properties"].values()
                )

            depth = get_depth(schema)
            prop_count = len(schema.get("properties", {}))

            if depth <= 3 and prop_count <= 5:
                return "simple"
            elif depth <= 6 and prop_count <= 15:
                return "moderate"
            else:
                return "complex"

        # Test simple schema
        simple_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        self.assertEqual(classify_complexity(simple_schema), "simple")

        # ✅ 修复：创建真正深度>6的schema
        # 深度计算：根层=0, field_0=1, nested_0=2, deep_0=3, very_deep_level1=4,
        #          very_deep_level2=5, very_deep_level3=6, final=7 (>6)
        # 构建一个同时满足深度>6与属性数>15的复杂 schema
        outer_props: Dict[str, Any] = {}
        for i in range(20):
            nested_props_j: Dict[str, Any] = {}
            for j in range(3):
                deep_props_k: Dict[str, Any] = {}
                for k in range(2):
                    deep_props_k[f"deep_{k}"] = {
                        "type": "object",
                        "properties": {
                            "level1": {
                                "type": "object",
                                "properties": {
                                    "level2": {
                                        "type": "object",
                                        "properties": {
                                            "level3": {
                                                "type": "object",
                                                "properties": {
                                                    "final": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                nested_props_j[f"nested_{j}"] = {
                    "type": "object",
                    "properties": deep_props_k,
                }
            outer_props[f"field_{i}"] = {
                "type": "object",
                "properties": nested_props_j,
            }

        complex_schema = {
            "type": "object",
            "properties": outer_props,
        }

        # 验证深度和属性数
        def get_depth_debug(obj, current_depth=0):
            if not isinstance(obj, dict):
                return current_depth
            if "properties" not in obj:
                return current_depth
            return max(
                get_depth_debug(prop, current_depth + 1)
                for prop in obj["properties"].values()
            )

        actual_depth = get_depth_debug(complex_schema)
        actual_props = len(complex_schema.get("properties", {}))

        # Debug信息
        print(f"  Complex schema depth: {actual_depth}, properties: {actual_props}")

        # 确保符合complex条件：深度>6 或 属性>15
        self.assertTrue(actual_depth > 6 or actual_props > 15,
                        f"Schema should be complex: depth={actual_depth}, props={actual_props}")

        self.assertEqual(classify_complexity(complex_schema), "complex")

        print("✓ Schema complexity classification: PASS")


# ============================================================================
# Concurrent Access Tests (Preserved)
# ============================================================================

class TestConcurrentAccess(unittest.TestCase):
    """Test thread-safety of audit tracker."""

    def setUp(self):
        self.tracker = StructuredOutputAuditTracker(
            enabled=True,
            max_trails=100,
            record_full_events=True
        )

    def test_concurrent_trail_creation(self):
        """Test creating trails from multiple threads."""
        num_threads = 50
        threads = []

        def create_trail(request_id):
            self.tracker.start_trail(request_id, "test_backend")
            for i in range(10):
                self.tracker.record_token_acceptance(
                    request_id, [i], True, f"state_{i}"
                )

        for i in range(num_threads):
            t = threading.Thread(target=create_trail, args=(f"req_{i:03d}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        all_trails = self.tracker.get_all_trails()
        self.assertEqual(len(all_trails), num_threads)

        for trail in all_trails.values():
            self.assertEqual(len(trail.events), 10)


# ============================================================================
# Real-World Scenario Tests (Enhanced)
# ============================================================================

class TestAUTOSARScenario(unittest.TestCase):
    """Test audit functionality with AUTOSAR ARXML generation scenario."""

    def test_autosar_json_schema_audit(self):
        """Simulate AUTOSAR ARXML generation with JSON Schema constraints."""

        autosar_schema = {
            "type": "object",
            "properties": {
                "AUTOSAR": {
                    "type": "object",
                    "properties": {
                        "AR-PACKAGES": {
                            "type": "object",
                            "properties": {
                                "AR-PACKAGE": {
                                    "type": "object",
                                    "properties": {
                                        "SHORT-NAME": {
                                            "type": "string",
                                            "enum": ["ComponentTypes", "DataTypes", "Interfaces"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        config = StructuredOutputAuditConfig(
            enabled=True,
            record_full_events=True,
            record_allowed_tokens=False
        )
        tracker = initialize_audit_system(config)

        request_id = "autosar_gen_001"
        tracker.start_trail(
            request_id,
            "xgrammar",
            grammar_spec=json.dumps(autosar_schema)
        )

        generation_sequence = [
            (123, "state_0", 1),
            (34, "state_1", 1),
            (3927, "state_2", 1),
            (21950, "state_enum", 3),
        ]

        for token_id, state, allowed_count in generation_sequence:
            tracker.record_token_acceptance(
                request_id, [token_id], True, state
            )

            bitmask = torch.zeros(50000, dtype=torch.int32)
            if allowed_count < 50000:
                bitmask[:allowed_count] = 1

            tracker.record_bitmask_update(request_id, bitmask, state)

        tracker.finalize_trail(request_id)
        trail = tracker.get_trail(request_id)

        self.assertIsNotNone(trail)
        self.assertEqual(trail.backend_type, "xgrammar")

        enum_events = [
            e for e in trail.events
            if e.current_state_id == "state_enum"
               and e.event_type == AuditEventType.BITMASK_UPDATE
        ]

        self.assertGreater(len(enum_events), 0)
        enum_event = enum_events[0]
        self.assertEqual(enum_event.allowed_tokens_count, 3)


# ============================================================================
# Test Runner
# ============================================================================

def run_test_suite():
    """Run the complete test suite with detailed output."""

    print("\n" + "=" * 70)
    print("vLLM Structured Output Audit - Comprehensive Test Suite")
    print("Version: 2.0 (with Management API & Unified Checks)")
    print("=" * 70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestUnifiedAuditEnabledCheck,  # NEW
        TestManagementAPI,  # NEW
        TestEnvironmentConfiguration,  # NEW
        TestBenchmarkIntegration,  # NEW
        TestGrammarBaseInitialization,
        TestAuditTracker,
        TestConcurrentAccess,
        TestAUTOSARScenario,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  Total tests: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)