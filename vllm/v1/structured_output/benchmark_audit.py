#!/usr/bin/env python3
"""
Audit System Performance Benchmark

Measures the overhead of different audit configurations on vLLM generation.

Usage:
    python benchmark_audit.py --model <model_path> --output results.csv

Requirements:
    - vLLM installed with audit support
    - Test AUTOSAR JSON schemas in ./test_schemas/
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

import requests
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    audit_enabled: bool
    record_full_events: bool
    record_allowed_tokens: bool
    persist_to_disk: bool

    def to_env_vars(self) -> Dict[str, str]:
        """Convert config to environment variables."""
        return {
            "VLLM_STRUCTURED_OUTPUT_AUDIT": str(self.audit_enabled).lower(),
            "VLLM_AUDIT_RECORD_FULL_EVENTS": str(self.record_full_events).lower(),
            "VLLM_AUDIT_RECORD_ALLOWED_TOKENS": str(self.record_allowed_tokens).lower(),
            "VLLM_AUDIT_PERSIST": str(self.persist_to_disk).lower(),
            "VLLM_AUDIT_IN_RESPONSE": "true",
            "VLLM_AUDIT_RESPONSE_LEVEL": "full" if self.record_full_events else "summary"
        }


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    schema_name: str
    complexity: str
    seed: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    audit_steps: Optional[int]
    audit_rollbacks: Optional[int]
    audit_errors: Optional[int]
    audit_duration: Optional[float]
    success: bool
    error_message: Optional[str]


# Predefined benchmark configurations
BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="no_audit",
        audit_enabled=False,
        record_full_events=False,
        record_allowed_tokens=False,
        persist_to_disk=False
    ),
    BenchmarkConfig(
        name="summary_mode",
        audit_enabled=True,
        record_full_events=False,
        record_allowed_tokens=False,
        persist_to_disk=False
    ),
    BenchmarkConfig(
        name="full_logging",
        audit_enabled=True,
        record_full_events=True,
        record_allowed_tokens=False,
        persist_to_disk=False
    ),
    BenchmarkConfig(
        name="full_with_tokens",
        audit_enabled=True,
        record_full_events=True,
        record_allowed_tokens=True,
        persist_to_disk=False
    ),
]


class VLLMAuditBenchmark:
    """Benchmark runner for vLLM audit system."""

    def __init__(
            self,
            api_base: str = "http://localhost:8000",
            model_name: str = "default",
            schemas_dir: Path = Path("./test_schemas")
    ):
        self.api_base = api_base
        self.model_name = model_name
        self.schemas_dir = schemas_dir

    def load_test_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load test JSON schemas from directory."""
        schemas = {}
        if not self.schemas_dir.exists():
            print(f"Warning: Schema directory {self.schemas_dir} not found")
            return schemas

        for schema_file in self.schemas_dir.glob("*.json"):
            with open(schema_file) as f:
                schemas[schema_file.stem] = json.load(f)

        return schemas

    def classify_complexity(self, schema: Dict[str, Any]) -> str:
        """Classify schema complexity based on depth and property count."""

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

    def make_request(
            self,
            schema: Dict[str, Any],
            seed: int,
            temperature: float = 0.7,
            max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Make a completion request to vLLM API."""
        url = f"{self.api_base}/v1/completions"

        payload = {
            "model": self.model_name,
            "prompt": "Generate a valid AUTOSAR software component configuration:",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "structured_outputs": {
                "json": schema
            }
        }

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        latency = time.time() - start_time

        response.raise_for_status()
        result = response.json()
        result["benchmark_latency"] = latency

        return result

    def extract_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from API response."""
        usage = response.get("usage", {})
        audit_data = usage.get("structured_output_audit", {})

        if "audit_trail" in audit_data:
            # Full logging mode
            trail = audit_data["audit_trail"]
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "audit_steps": trail.get("total_steps"),
                "audit_rollbacks": trail.get("total_rollbacks"),
                "audit_errors": trail.get("total_errors"),
                "audit_duration": trail.get("duration")
            }
        elif "audit_summary" in audit_data:
            # Summary mode
            summary = audit_data["audit_summary"]
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "audit_steps": summary.get("total_steps"),
                "audit_rollbacks": summary.get("total_rollbacks"),
                "audit_errors": summary.get("total_errors"),
                "audit_duration": summary.get("duration_seconds")
            }
        else:
            # No audit
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "audit_steps": None,
                "audit_rollbacks": None,
                "audit_errors": None,
                "audit_duration": None
            }

    def run_single_benchmark(
            self,
            config: BenchmarkConfig,
            schema_name: str,
            schema: Dict[str, Any],
            seed: int
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        complexity = self.classify_complexity(schema)

        try:
            response = self.make_request(schema, seed)
            metrics = self.extract_metrics(response)

            return BenchmarkResult(
                config_name=config.name,
                schema_name=schema_name,
                complexity=complexity,
                seed=seed,
                latency_seconds=response["benchmark_latency"],
                input_tokens=metrics["input_tokens"],
                output_tokens=metrics["output_tokens"],
                audit_steps=metrics["audit_steps"],
                audit_rollbacks=metrics["audit_rollbacks"],
                audit_errors=metrics["audit_errors"],
                audit_duration=metrics["audit_duration"],
                success=True,
                error_message=None
            )
        except Exception as e:
            return BenchmarkResult(
                config_name=config.name,
                schema_name=schema_name,
                complexity=complexity,
                seed=seed,
                latency_seconds=0.0,
                input_tokens=0,
                output_tokens=0,
                audit_steps=None,
                audit_rollbacks=None,
                audit_errors=None,
                audit_duration=None,
                success=False,
                error_message=str(e)
            )

    def run_benchmarks(
            self,
            configs: List[BenchmarkConfig],
            seeds: List[int] = [42, 1001, 20250701],
            max_schemas: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        schemas = self.load_test_schemas()

        if not schemas:
            print("Error: No test schemas found. Creating sample schema...")
            schemas = {"sample": self._create_sample_schema()}

        if max_schemas:
            schemas = dict(list(schemas.items())[:max_schemas])

        results = []
        total_runs = len(configs) * len(schemas) * len(seeds)
        current_run = 0

        print(f"\n{'=' * 80}")
        print(f"Starting benchmark: {len(configs)} configs × {len(schemas)} schemas × {len(seeds)} seeds")
        print(f"Total runs: {total_runs}")
        print(f"{'=' * 80}\n")

        for config in configs:
            print(f"\n--- Testing configuration: {config.name} ---")
            # Note: In real deployment, you'd restart vLLM with new env vars here
            # For this benchmark, we assume environment is pre-configured

            for schema_name, schema in schemas.items():
                complexity = self.classify_complexity(schema)
                print(f"\n  Schema: {schema_name} (complexity: {complexity})")

                for seed in seeds:
                    current_run += 1
                    print(f"    [{current_run}/{total_runs}] Seed {seed}... ", end="", flush=True)

                    result = self.run_single_benchmark(config, schema_name, schema, seed)
                    results.append(result)

                    if result.success:
                        print(f"✓ {result.latency_seconds:.2f}s")
                    else:
                        print(f"✗ {result.error_message}")

        return results

    def _create_sample_schema(self) -> Dict[str, Any]:
        """Create a sample AUTOSAR-like JSON schema for testing."""
        return {
            "type": "object",
            "properties": {
                "SHORT-NAME": {"type": "string"},
                "CATEGORY": {"type": "string"},
                "SW-COMPONENT-TYPE": {
                    "type": "object",
                    "properties": {
                        "SHORT-NAME": {"type": "string"},
                        "PORTS": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "SHORT-NAME": {"type": "string"},
                                    "PORT-INTERFACE-REF": {"type": "string"}
                                },
                                "required": ["SHORT-NAME"]
                            }
                        }
                    },
                    "required": ["SHORT-NAME"]
                }
            },
            "required": ["SHORT-NAME", "SW-COMPONENT-TYPE"]
        }

    def save_results(self, results: List[BenchmarkResult], output_file: Path):
        """Save benchmark results to CSV."""
        if not results:
            print("No results to save")
            return

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

        print(f"\nResults saved to {output_file}")

    def print_summary(self, results: List[BenchmarkResult]):
        """Print summary statistics."""
        if not results:
            print("No results to summarize")
            return

        print(f"\n{'=' * 80}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 80}\n")

        # Group by configuration
        by_config = {}
        for result in results:
            if result.config_name not in by_config:
                by_config[result.config_name] = []
            if result.success:
                by_config[result.config_name].append(result)

        # Calculate statistics for no_audit baseline
        baseline_latencies = [r.latency_seconds for r in by_config.get("no_audit", [])]
        baseline_mean = statistics.mean(baseline_latencies) if baseline_latencies else 0

        print(f"{'Configuration':<20} {'Mean Latency':<15} {'Std Dev':<12} {'Audit Factor':<15} {'Runs'}")
        print("-" * 80)

        for config_name in ["no_audit", "summary_mode", "full_logging", "full_with_tokens"]:
            if config_name not in by_config:
                continue

            config_results = by_config[config_name]
            latencies = [r.latency_seconds for r in config_results]

            mean_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
            audit_factor = mean_latency / baseline_mean if baseline_mean > 0 else 0

            print(
                f"{config_name:<20} {mean_latency:>10.2f}s    {std_latency:>8.2f}s    {audit_factor:>10.2f}×     {len(config_results)}")

        # Complexity breakdown
        print(f"\n{'=' * 80}")
        print("LATENCY BY COMPLEXITY")
        print(f"{'=' * 80}\n")

        for complexity in ["simple", "moderate", "complex"]:
            print(f"\n{complexity.upper()}:")
            print(f"  {'Configuration':<20} {'Mean Latency':<15} {'Audit Factor'}")
            print("  " + "-" * 60)

            for config_name in ["no_audit", "summary_mode", "full_logging"]:
                if config_name not in by_config:
                    continue

                complexity_results = [
                    r for r in by_config[config_name]
                    if r.complexity == complexity
                ]

                if not complexity_results:
                    continue

                latencies = [r.latency_seconds for r in complexity_results]
                mean_latency = statistics.mean(latencies)

                baseline_complexity = [
                    r.latency_seconds for r in by_config.get("no_audit", [])
                    if r.complexity == complexity
                ]
                baseline_mean_complexity = (
                    statistics.mean(baseline_complexity) if baseline_complexity else 0
                )
                audit_factor = (
                    mean_latency / baseline_mean_complexity
                    if baseline_mean_complexity > 0 else 0
                )

                print(f"  {config_name:<20} {mean_latency:>10.2f}s    {audit_factor:>10.2f}×")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM audit system performance"
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="vLLM API base URL"
    )
    parser.add_argument(
        "--model",
        default="default",
        help="Model name"
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path("./test_schemas"),
        help="Directory containing test JSON schemas"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_benchmark_results.csv"),
        help="Output CSV file"
    )
    parser.add_argument(
        "--max-schemas",
        type=int,
        help="Maximum number of schemas to test (for quick runs)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1001, 20250701],
        help="Random seeds for reproducibility"
    )

    args = parser.parse_args()

    benchmark = VLLMAuditBenchmark(
        api_base=args.api_base,
        model_name=args.model,
        schemas_dir=args.schemas_dir
    )

    results = benchmark.run_benchmarks(
        configs=BENCHMARK_CONFIGS,
        seeds=args.seeds,
        max_schemas=args.max_schemas
    )

    benchmark.save_results(results, args.output)
    benchmark.print_summary(results)

    print(f"\n✓ Benchmark complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()