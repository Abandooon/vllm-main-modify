# test_audit_helpers.py
"""
Helper utilities for testing audit functionality.

Version 2.0: Added Management API and performance testing utilities.
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
import torch
import requests


# ============================================================================
# Mock Creators (Preserved from original)
# ============================================================================

class MockGrammarFactory:
    """Factory for creating mock Grammar objects for testing."""

    @staticmethod
    def create_xgrammar_matcher():
        """Create a mock XGrammar matcher."""
        mock_matcher = MagicMock()
        mock_matcher.accept_token.return_value = True
        mock_matcher.is_terminated.return_value = False
        mock_matcher.rollback.return_value = None
        mock_matcher.fill_next_token_bitmask.return_value = None
        return mock_matcher

    @staticmethod
    def create_xgrammar_ctx():
        """Create a mock XGrammar context."""
        mock_ctx = MagicMock()
        return mock_ctx

    @staticmethod
    def create_outlines_guide():
        """Create a mock Outlines guide."""
        mock_guide = MagicMock()
        mock_guide.accepts_tokens.return_value = True
        mock_guide.is_finished.return_value = False
        mock_guide.advance.return_value = None
        mock_guide.rollback_state.return_value = None
        mock_guide.reset.return_value = None

        def mock_write_mask(data_ptr, numel, element_size):
            pass

        mock_guide.write_mask_into = mock_write_mask

        return mock_guide

    @staticmethod
    def create_guidance_matcher():
        """Create a mock Guidance matcher."""
        mock_matcher = MagicMock()
        mock_matcher.consume_tokens.return_value = True
        mock_matcher.is_stopped.return_value = False
        mock_matcher.validate_tokens.return_value = []
        mock_matcher.rollback.return_value = None
        mock_matcher.get_error.return_value = None
        return mock_matcher

    @staticmethod
    def create_guidance_tokenizer():
        """Create a mock Guidance tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 50000
        mock_tokenizer.eos_token = 0
        return mock_tokenizer


# ============================================================================
# Test Data Generators (Preserved from original)
# ============================================================================

class AuditTestDataGenerator:
    """Generate realistic test data for audit trails."""

    @staticmethod
    def generate_json_schema(complexity: str = "simple") -> Dict[str, Any]:
        """Generate JSON schemas of varying complexity."""
        schemas = {
            "simple": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            },
            "enum_constraint": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "inactive", "pending"]
                    }
                }
            },
            "autosar": {
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
                                                "enum": ["ComponentTypes", "DataTypes"]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return schemas.get(complexity, schemas["simple"])

    @staticmethod
    def generate_token_sequence(pattern: str = "json_object") -> List[int]:
        """Generate realistic token sequences."""
        sequences = {
            "json_object": [123, 34, 110, 97, 109, 101, 34, 58, 34],
            "json_array": [91, 49, 44, 50, 44, 51, 93],
            "string": [34, 72, 101, 108, 108, 111, 34],
            "number": [49, 50, 51],
            "boolean": [116, 114, 117, 101],
        }
        return sequences.get(pattern, sequences["json_object"])

    @staticmethod
    def generate_bitmask(allowed_count: int, vocab_size: int = 50000) -> torch.Tensor:
        """Generate a bitmask with specified number of allowed tokens."""
        bitmask = torch.zeros(vocab_size, dtype=torch.int32)
        bitmask[:allowed_count] = 1
        return bitmask


# ============================================================================
# NEW: Management API Test Helpers
# ============================================================================

class ManagementAPITestClient:
    """Helper for testing Management API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1/admin/audit"

    def get_stats(self) -> Dict[str, Any]:
        """Call GET /stats endpoint."""
        response = requests.get(f"{self.api_base}/stats")
        response.raise_for_status()
        return response.json()

    def list_trails(
            self,
            limit: int = 100,
            offset: int = 0,
            backend_type: Optional[str] = None,
            include_active: bool = True
    ) -> List[Dict[str, Any]]:
        """Call GET /list endpoint."""
        params = {
            "limit": limit,
            "offset": offset,
            "include_active": include_active
        }
        if backend_type:
            params["backend_type"] = backend_type

        response = requests.get(f"{self.api_base}/list", params=params)
        response.raise_for_status()
        return response.json()

    def get_trail(self, request_id: str, include_events: bool = True) -> Dict[str, Any]:
        """Call GET /trail/{request_id} endpoint."""
        params = {"include_events": include_events}
        response = requests.get(
            f"{self.api_base}/trail/{request_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def export_trails(
            self,
            request_ids: Optional[List[str]] = None,
            include_events: bool = True,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Call POST /export endpoint."""
        payload = {
            "request_ids": request_ids,
            "include_events": include_events,
            "start_time": start_time,
            "end_time": end_time
        }
        response = requests.post(f"{self.api_base}/export", json=payload)
        response.raise_for_status()
        return response.json()

    def delete_trail(self, request_id: str) -> Dict[str, Any]:
        """Call DELETE /trail/{request_id} endpoint."""
        response = requests.delete(f"{self.api_base}/trail/{request_id}")
        response.raise_for_status()
        return response.json()

    def clear_all(self, confirm: bool = False) -> Dict[str, Any]:
        """Call POST /clear endpoint."""
        params = {"confirm": confirm}
        response = requests.post(f"{self.api_base}/clear", params=params)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Call GET /health endpoint."""
        response = requests.get(f"{self.api_base}/health")
        response.raise_for_status()
        return response.json()


class ManagementAPIValidator:
    """Validate Management API responses."""

    @staticmethod
    def validate_stats_response(stats: Dict[str, Any]) -> List[str]:
        """Validate /stats response structure."""
        errors = []

        required_fields = [
            "enabled", "total_trails", "active_trails",
            "total_events_recorded", "avg_steps_per_trail",
            "avg_duration_seconds", "memory_usage_trails"
        ]

        for field in required_fields:
            if field not in stats:
                errors.append(f"Missing field: {field}")

        if "enabled" in stats and not isinstance(stats["enabled"], bool):
            errors.append("'enabled' must be boolean")

        if "total_trails" in stats and not isinstance(stats["total_trails"], int):
            errors.append("'total_trails' must be integer")

        return errors

    @staticmethod
    def validate_trail_summary(trail: Dict[str, Any]) -> List[str]:
        """Validate trail summary structure."""
        errors = []

        required_fields = [
            "request_id", "backend_type", "start_time",
            "total_steps", "total_tokens_generated"
        ]

        for field in required_fields:
            if field not in trail:
                errors.append(f"Missing field: {field}")

        return errors

    @staticmethod
    def validate_export_response(export: Dict[str, Any]) -> List[str]:
        """Validate export response structure."""
        errors = []

        if "export_timestamp" not in export:
            errors.append("Missing export_timestamp")
        if "trail_count" not in export:
            errors.append("Missing trail_count")
        if "trails" not in export:
            errors.append("Missing trails array")

        if "trail_count" in export and "trails" in export:
            if export["trail_count"] != len(export["trails"]):
                errors.append(
                    f"trail_count ({export['trail_count']}) doesn't match "
                    f"trails array length ({len(export['trails'])})"
                )

        return errors


# ============================================================================
# NEW: Performance Testing Helpers
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    operation: str
    total_operations: int
    total_time_seconds: float
    ops_per_second: float
    avg_time_per_op_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float


class PerformanceTester:
    """Helper for running performance tests on audit system."""

    @staticmethod
    def measure_operation(
            operation_func,
            num_iterations: int = 1000,
            warmup_iterations: int = 100
    ) -> PerformanceMetrics:
        """
        Measure performance of an operation.

        Args:
            operation_func: Function to measure (takes iteration number)
            num_iterations: Number of times to run the operation
            warmup_iterations: Number of warmup runs (not measured)

        Returns:
            PerformanceMetrics with detailed statistics
        """
        # Warmup
        for i in range(warmup_iterations):
            operation_func(i)

        # Measure
        times = []
        start_time = time.perf_counter()

        for i in range(num_iterations):
            op_start = time.perf_counter()
            operation_func(i)
            op_end = time.perf_counter()
            times.append((op_end - op_start) * 1000)  # Convert to ms

        end_time = time.perf_counter()
        total_time = end_time - start_time

        return PerformanceMetrics(
            operation=operation_func.__name__,
            total_operations=num_iterations,
            total_time_seconds=total_time,
            ops_per_second=num_iterations / total_time,
            avg_time_per_op_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0
        )

    @staticmethod
    def compare_configurations(
            baseline_func,
            test_func,
            num_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare performance of two configurations.

        Returns:
            Comparison metrics including overhead percentage
        """
        baseline_metrics = PerformanceTester.measure_operation(
            baseline_func, num_iterations
        )
        test_metrics = PerformanceTester.measure_operation(
            test_func, num_iterations
        )

        overhead_percent = (
                (test_metrics.avg_time_per_op_ms - baseline_metrics.avg_time_per_op_ms)
                / baseline_metrics.avg_time_per_op_ms * 100
        )

        return {
            "baseline": baseline_metrics,
            "test": test_metrics,
            "overhead_percent": overhead_percent,
            "slowdown_factor": test_metrics.avg_time_per_op_ms / baseline_metrics.avg_time_per_op_ms
        }

    @staticmethod
    def print_metrics(metrics: PerformanceMetrics, title: str = "Performance Metrics"):
        """Pretty-print performance metrics."""
        print("\n" + "=" * 70)
        print(f"{title}: {metrics.operation}")
        print("=" * 70)
        print(f"Total operations:    {metrics.total_operations:,}")
        print(f"Total time:          {metrics.total_time_seconds:.3f}s")
        print(f"Ops/second:          {metrics.ops_per_second:,.0f}")
        print(f"Avg time/op:         {metrics.avg_time_per_op_ms:.3f}ms")
        print(f"Min time:            {metrics.min_time_ms:.3f}ms")
        print(f"Max time:            {metrics.max_time_ms:.3f}ms")
        print(f"Std deviation:       {metrics.std_dev_ms:.3f}ms")
        print("=" * 70 + "\n")


class BenchmarkScenarioGenerator:
    """Generate realistic benchmark scenarios."""

    @staticmethod
    def generate_autosar_workload(num_requests: int = 100) -> List[Dict[str, Any]]:
        """
        Generate AUTOSAR-like workload for benchmarking.

        Returns:
            List of request configurations with varying complexity
        """
        workload = []

        complexity_distribution = {
            "simple": 0.5,  # 50% simple
            "moderate": 0.3,  # 30% moderate
            "complex": 0.2  # 20% complex
        }

        generator = AuditTestDataGenerator()

        for i in range(num_requests):
            # Determine complexity
            rand = (i % 10) / 10
            if rand < 0.5:
                complexity = "simple"
            elif rand < 0.8:
                complexity = "moderate"
            else:
                complexity = "complex"

            workload.append({
                "request_id": f"bench_req_{i:04d}",
                "schema": generator.generate_json_schema(complexity),
                "complexity": complexity,
                "seed": 42 + i
            })

        return workload

    @staticmethod
    def estimate_expected_overhead(config_name: str) -> Tuple[float, float]:
        """
        Get expected overhead range for a configuration.

        Returns:
            (min_overhead_percent, max_overhead_percent)
        """
        overhead_ranges = {
            "no_audit": (0.0, 0.0),
            "summary_mode": (5.0, 10.0),
            "full_logging": (80.0, 120.0),  # ~2× = 100% overhead
            "full_with_tokens": (150.0, 250.0)  # ~2.5× = 150% overhead
        }

        return overhead_ranges.get(config_name, (0.0, 0.0))


# ============================================================================
# Validation Utilities (Preserved from original, enhanced)
# ============================================================================

class AuditTrailValidator:
    """Validate audit trail integrity and completeness."""

    @staticmethod
    def validate_trail_structure(trail_dict: Dict[str, Any]) -> List[str]:
        """Validate that trail has all required fields."""
        errors = []

        required_fields = [
            "request_id", "backend_type", "start_time",
            "total_steps", "total_tokens_generated"
        ]

        for field in required_fields:
            if field not in trail_dict:
                errors.append(f"Missing required field: {field}")

        if "request_id" in trail_dict and not isinstance(trail_dict["request_id"], str):
            errors.append("request_id must be string")

        if "total_steps" in trail_dict and not isinstance(trail_dict["total_steps"], int):
            errors.append("total_steps must be integer")

        return errors

    @staticmethod
    def validate_event_sequence(events: List[Dict[str, Any]]) -> List[str]:
        """Validate that events form a valid sequence."""
        errors = []

        if not events:
            return errors

        prev_step = 0
        for i, event in enumerate(events):
            if "step_number" not in event:
                errors.append(f"Event {i} missing step_number")
                continue

            step = event["step_number"]
            if step != prev_step + 1:
                errors.append(
                    f"Step number gap: expected {prev_step + 1}, got {step}"
                )
            prev_step = step

        prev_timestamp = 0
        for i, event in enumerate(events):
            if "timestamp" not in event:
                errors.append(f"Event {i} missing timestamp")
                continue

            ts = event["timestamp"]
            if ts < prev_timestamp:
                errors.append(
                    f"Timestamp not monotonic at event {i}: "
                    f"{prev_timestamp} -> {ts}"
                )
            prev_timestamp = ts

        return errors

    @staticmethod
    def validate_constraint_effectiveness(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze constraint effectiveness from audit events."""
        stats = {
            "total_bitmask_updates": 0,
            "high_constraints": 0,
            "medium_constraints": 0,
            "low_constraints": 0,
            "avg_allowed_tokens": 0,
            "token_rejections": 0,
        }

        allowed_counts = []

        for event in events:
            event_type = event.get("event_type")

            if event_type == "bitmask_update":
                stats["total_bitmask_updates"] += 1

                count = event.get("allowed_tokens_count", 0)
                allowed_counts.append(count)

                if count < 10:
                    stats["high_constraints"] += 1
                elif count < 1000:
                    stats["medium_constraints"] += 1
                else:
                    stats["low_constraints"] += 1

            elif event_type == "token_reject":
                stats["token_rejections"] += 1

        if allowed_counts:
            stats["avg_allowed_tokens"] = sum(allowed_counts) / len(allowed_counts)

        return stats


# ============================================================================
# Debugging Tools (Preserved from original)
# ============================================================================

class AuditDebugger:
    """Tools for debugging audit functionality."""

    @staticmethod
    def print_trail_summary(trail_dict: Dict[str, Any]) -> None:
        """Pretty-print trail summary."""
        print("\n" + "=" * 70)
        print("Audit Trail Summary")
        print("=" * 70)
        print(f"Request ID:       {trail_dict.get('request_id')}")
        print(f"Backend:          {trail_dict.get('backend_type')}")
        print(f"Total Steps:      {trail_dict.get('total_steps')}")
        print(f"Tokens Generated: {trail_dict.get('total_tokens_generated')}")
        print(f"Rollbacks:        {trail_dict.get('total_rollbacks')}")
        print(f"Errors:           {trail_dict.get('total_errors')}")

        if trail_dict.get('duration'):
            print(f"Duration:         {trail_dict['duration']:.3f}s")

        print("=" * 70 + "\n")

    @staticmethod
    def print_event_timeline(events: List[Dict[str, Any]], max_events: int = 20) -> None:
        """Print event timeline."""
        print("\n" + "=" * 70)
        print(f"Event Timeline (showing first {max_events} events)")
        print("=" * 70)
        print(f"{'Step':<6} {'Type':<20} {'State':<15} {'Details'}")
        print("-" * 70)

        for event in events[:max_events]:
            step = event.get('step_number', '?')
            event_type = event.get('event_type', 'unknown')
            state = event.get('current_state_id', '-')[:15]

            details = ""
            if event_type == "token_accept":
                tokens = event.get('accepted_tokens', [])
                details = f"Accepted {len(tokens)} tokens"
            elif event_type == "bitmask_update":
                count = event.get('allowed_tokens_count', 0)
                details = f"Allowed: {count} tokens"
            elif event_type == "rollback":
                num = event.get('metadata', {}).get('num_tokens_rolled_back', 0)
                details = f"Rollback {num} tokens"

            print(f"{step:<6} {event_type:<20} {state:<15} {details}")

        if len(events) > max_events:
            print(f"... ({len(events) - max_events} more events)")

        print("=" * 70 + "\n")


# ============================================================================
# NEW: Example Usage with New Features
# ============================================================================

def example_usage_with_api():
    """Demonstrate Management API testing."""

    # Create API client
    api_client = ManagementAPITestClient("http://localhost:8000")

    try:
        # Check health
        health = api_client.health_check()
        print(f"Audit system status: {health['status']}")

        # Get statistics
        stats = api_client.get_stats()
        print(f"Total trails: {stats['total_trails']}")
        print(f"Active trails: {stats['active_trails']}")

        # List recent trails
        trails = api_client.list_trails(limit=5)
        print(f"Retrieved {len(trails)} trails")

        # Validate responses
        validator = ManagementAPIValidator()
        errors = validator.validate_stats_response(stats)

        if errors:
            print("Validation errors:", errors)
        else:
            print("✓ All API responses valid")

    except requests.exceptions.RequestException as e:
        print(f"API connection error: {e}")
        print("Note: Ensure vLLM server is running with audit enabled")


def example_performance_testing():
    """Demonstrate performance testing."""
    from vllm.v1.structured_output.audit_tracker import StructuredOutputAuditTracker

    # Create trackers
    tracker_disabled = StructuredOutputAuditTracker(enabled=False)
    tracker_enabled = StructuredOutputAuditTracker(enabled=True)

    # Define test operations
    def baseline_op(i):
        tracker_disabled.record_token_acceptance(f"req_{i}", [1, 2, 3], True, "s1")

    def test_op(i):
        tracker_enabled.start_trail(f"req_{i}", "test")
        tracker_enabled.record_token_acceptance(f"req_{i}", [1, 2, 3], True, "s1")

    # Compare performance
    tester = PerformanceTester()
    comparison = tester.compare_configurations(baseline_op, test_op, num_iterations=1000)

    print("\n" + "=" * 70)
    print("Performance Comparison: Audit Disabled vs Enabled")
    print("=" * 70)
    print(f"Overhead: {comparison['overhead_percent']:.1f}%")
    print(f"Slowdown factor: {comparison['slowdown_factor']:.2f}×")
    print("=" * 70 + "\n")

    tester.print_metrics(comparison['baseline'], "Baseline (Disabled)")
    tester.print_metrics(comparison['test'], "Test (Enabled)")


if __name__ == "__main__":
    print("=" * 70)
    print("Audit Test Helpers - Version 2.0")
    print("=" * 70 + "\n")

    print("Example 1: Management API Testing")
    print("-" * 70)
    example_usage_with_api()

    print("\n\nExample 2: Performance Testing")
    print("-" * 70)
    example_performance_testing()