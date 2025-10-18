# vLLM Structured Output Audit System - Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [API Usage](#api-usage)
5. [Management API](#management-api)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The vLLM Audit System provides comprehensive tracking of structured output generation, recording:
- **State transitions**: FSM/DFA state changes during token generation
- **Token decisions**: Which tokens were allowed/rejected at each step
- **Constraint enforcement**: Bitmask updates and validation events
- **Performance metrics**: Step counts, rollbacks, errors, latencies

### Key Features
- ✅ **Zero-overhead when disabled**: No performance impact if audit is turned off
- ✅ **Configurable granularity**: 3 modes (Summary / Key-event / Full logging)
- ✅ **Thread-safe**: Supports concurrent requests
- ✅ **Persistent storage**: Optional disk persistence with compression
- ✅ **Management API**: Query and export audit data via HTTP endpoints

---

## Quick Start

### 1. Enable Audit System (Docker)
```bash
docker run -d \
  --name vllm-audit \
  -p 8000:8000 \
  --gpus all \
  -e VLLM_STRUCTURED_OUTPUT_AUDIT=true \
  -e VLLM_AUDIT_RESPONSE_LEVEL=summary \
  vllm/vllm:latest \
  --model <your-model> \
  --dtype auto
```

### 2. Make a Request
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "Generate JSON:",
    "structured_outputs": {
      "json": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name"]
      }
    }
  }'
```

### 3. Check Audit Data in Response
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "your-model",
  "choices": [...],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35,
    "structured_output_audit": {
      "audit_summary": {
        "backend_type": "xgrammar",
        "total_steps": 25,
        "total_tokens_generated": 25,
        "total_rollbacks": 0,
        "total_errors": 0,
        "duration_seconds": 0.234
      }
    }
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_STRUCTURED_OUTPUT_AUDIT` | bool | `false` | Master switch for audit system |
| `VLLM_AUDIT_RECORD_ALLOWED_TOKENS` | bool | `false` | Record full allowed token sets (⚠️ memory intensive) |
| `VLLM_AUDIT_RECORD_FULL_EVENTS` | bool | `true` | Record all events vs summaries only |
| `VLLM_AUDIT_INCLUDE_GRAMMAR` | bool | `false` | Include grammar spec in audit trail |
| `VLLM_AUDIT_MAX_TRAILS` | int | `1000` | Maximum trails in memory (LRU eviction) |
| `VLLM_AUDIT_ASYNC` | bool | `false` | Use async recording (experimental) |
| `VLLM_AUDIT_PERSIST` | bool | `false` | Persist trails to disk |
| `VLLM_AUDIT_LOG_DIR` | string | `None` | Directory for persisted audit logs |
| `VLLM_AUDIT_IN_RESPONSE` | bool | `true` | Include audit data in API responses |
| `VLLM_AUDIT_RESPONSE_LEVEL` | string | `summary` | Response detail: `none`, `summary`, `full` |

### Configuration Presets

#### Preset 1: Production Monitoring (Minimal Overhead)
```bash
VLLM_STRUCTURED_OUTPUT_AUDIT=true
VLLM_AUDIT_RECORD_FULL_EVENTS=false
VLLM_AUDIT_RECORD_ALLOWED_TOKENS=false
VLLM_AUDIT_PERSIST=false
VLLM_AUDIT_RESPONSE_LEVEL=summary
```
**Expected overhead**: <10%

#### Preset 2: Development & Debugging (Full Detail)
```bash
VLLM_STRUCTURED_OUTPUT_AUDIT=true
VLLM_AUDIT_RECORD_FULL_EVENTS=true
VLLM_AUDIT_RECORD_ALLOWED_TOKENS=false
VLLM_AUDIT_PERSIST=true
VLLM_AUDIT_LOG_DIR=/var/log/vllm/audit
VLLM_AUDIT_RESPONSE_LEVEL=full
```
**Expected overhead**: ~2× latency

#### Preset 3: Research & Paper Experiments (Maximum Detail)
```bash
VLLM_STRUCTURED_OUTPUT_AUDIT=true
VLLM_AUDIT_RECORD_FULL_EVENTS=true
VLLM_AUDIT_RECORD_ALLOWED_TOKENS=true  # ⚠️ Memory intensive
VLLM_AUDIT_PERSIST=true
VLLM_AUDIT_LOG_DIR=/research/audit_logs
VLLM_AUDIT_RESPONSE_LEVEL=full
VLLM_AUDIT_INCLUDE_GRAMMAR=true
```
**Expected overhead**: ~2-3× latency, ~10× memory

---

## API Usage

### Standard Completion API (Automatic Audit)

No changes needed to your existing API calls. Audit data is automatically collected and returned.

**Python Example**:
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.completions.create(
    model="your-model",
    prompt="Generate:",
    extra_body={
        "structured_outputs": {
            "json": {"type": "object", "properties": {...}}
        }
    }
)

# Extract audit summary
audit = response.usage.get("structured_output_audit", {})
if "audit_summary" in audit:
    print(f"Total steps: {audit['audit_summary']['total_steps']}")
    print(f"Duration: {audit['audit_summary']['duration_seconds']}s")
```

### Accessing Full Audit Trail (Full Logging Mode)

When `VLLM_AUDIT_RESPONSE_LEVEL=full`, the response includes complete event sequences:
```python
audit = response.usage["structured_output_audit"]["audit_trail"]

for event in audit["events"]:
    print(f"Step {event['step_number']}: {event['event_type']}")
    print(f"  State: {event['current_state_id']}")
    print(f"  Allowed tokens: {event['allowed_tokens_count']}")
```

---

## Management API

### Base URL
```
http://localhost:8000/v1/admin/audit
```

### Endpoints

#### 1. Get Global Statistics
```bash
GET /v1/admin/audit/stats
```

**Response**:
```json
{
  "enabled": true,
  "total_trails": 150,
  "active_trails": 3,
  "total_events_recorded": 3750,
  "avg_steps_per_trail": 25.0,
  "avg_duration_seconds": 0.234,
  "memory_usage_trails": 150
}
```

#### 2. List Audit Trails
```bash
GET /v1/admin/audit/list?limit=10&offset=0&backend_type=xgrammar
```

**Parameters**:
- `limit` (int, default=100): Maximum trails to return
- `offset` (int, default=0): Pagination offset
- `backend_type` (string, optional): Filter by backend
- `include_active` (bool, default=true): Include non-finalized trails

**Response**:
```json
[
  {
    "request_id": "cmpl-abc123",
    "backend_type": "xgrammar",
    "start_time": 1705555555.123,
    "end_time": 1705555555.357,
    "duration": 0.234,
    "total_steps": 25,
    "total_tokens_generated": 25,
    "total_rollbacks": 0,
    "total_errors": 0
  }
]
```

#### 3. Get Specific Trail
```bash
GET /v1/admin/audit/trail/cmpl-abc123?include_events=true
```

**Response**: Full audit trail with events array

#### 4. Bulk Export
```bash
POST /v1/admin/audit/export
Content-Type: application/json

{
  "request_ids": ["cmpl-abc123", "cmpl-def456"],
  "include_events": true,
  "start_time": 1705555555.0,
  "end_time": 1705666666.0
}
```

**Response**:
```json
{
  "export_timestamp": 1705777777.0,
  "trail_count": 2,
  "trails": [...]
}
```

#### 5. Delete Trail
```bash
DELETE /v1/admin/audit/trail/cmpl-abc123
```

#### 6. Clear All Trails (Requires Confirmation)
```bash
POST /v1/admin/audit/clear?confirm=true
```

#### 7. Health Check
```bash
GET /v1/admin/audit/health
```

---

## Performance Benchmarking

### Run Benchmark Script
```bash
# Prepare test schemas
mkdir -p test_schemas
echo '{"type": "object", "properties": {"name": {"type": "string"}}}' > test_schemas/simple.json

# Run benchmark
python benchmark_audit.py \
  --api-base http://localhost:8000 \
  --model your-model \
  --schemas-dir ./test_schemas \
  --output results.csv \
  --seeds 42 1001 20250701
```

### Expected Output
```
================================================================================
Starting benchmark: 4 configs × 3 schemas × 3 seeds
Total runs: 36
================================================================================

--- Testing configuration: no_audit ---
  Schema: simple (complexity: simple)
    [1/36] Seed 42... ✓ 0.12s
    [2/36] Seed 1001... ✓ 0.11s
    ...

================================================================================
BENCHMARK SUMMARY
================================================================================

Configuration        Mean Latency    Std Dev      Audit Factor    Runs
--------------------------------------------------------------------------------
no_audit                    0.12s       0.01s          1.00×        9
summary_mode                0.13s       0.01s          1.08×        9
full_logging                0.23s       0.02s          1.92×        9
full_with_tokens            0.34s       0.03s          2.83×        9
```

### Interpreting Results

- **Audit Factor**: Latency relative to no_audit baseline
- **Summary mode**: Typically 1.05-1.10× (5-10% overhead) ✅ **Production safe**
- **Full logging**: Typically 1.8-2.1× (~2× overhead) ⚠️ **Dev/debug only**
- **Full with tokens**: Typically 2.5-3.5× (high memory) ❌ **Research only**

---

## Troubleshooting

### Issue 1: Audit Data Not Appearing in Response

**Symptoms**: `usage.structured_output_audit` is missing

**Solutions**:
1. Check `VLLM_STRUCTURED_OUTPUT_AUDIT=true` is set
2. Verify `VLLM_AUDIT_IN_RESPONSE=true`
3. Ensure audit tracker initialized (check logs for "Audit system initialized")

### Issue 2: High Memory Usage

**Symptoms**: vLLM OOM, slow performance

**Solutions**:
1. Disable `VLLM_AUDIT_RECORD_ALLOWED_TOKENS` (most common culprit)
2. Reduce `VLLM_AUDIT_MAX_TRAILS` (default 1000)
3. Enable `VLLM_AUDIT_PERSIST` + delete old trails via API
4. Switch to `summary` mode (`VLLM_AUDIT_RECORD_FULL_EVENTS=false`)

### Issue 3: Persisted Logs Not Saving

**Symptoms**: No files in `VLLM_AUDIT_LOG_DIR`

**Solutions**:
1. Check directory exists and has write permissions
2. Verify `VLLM_AUDIT_PERSIST=true`
3. Check vLLM logs for filesystem errors

### Issue 4: Management API 503 Error

**Symptoms**: `/v1/admin/audit/*` returns 503

**Solutions**:
1. Ensure audit system initialized on vLLM startup
2. Check environment variables are set before starting vLLM
3. Restart vLLM process

---

## Best Practices

### For Production Deployments

1. **Use Summary Mode**: Balance observability and performance
```bash
   VLLM_AUDIT_RESPONSE_LEVEL=summary
   VLLM_AUDIT_RECORD_FULL_EVENTS=false
```

2. **Set Reasonable Trail Limit**: Prevent unbounded memory growth
```bash
   VLLM_AUDIT_MAX_TRAILS=500
```

3. **Periodic Cleanup**: Implement log rotation for persisted files
```bash
   # Cron job to clean old logs
   0 2 * * * find /var/log/vllm/audit -type f -mtime +7 -delete
```

4. **Monitor via API**: Use `/v1/admin/audit/stats` for metrics collection

### For Research & Experiments

1. **Full Logging + Persistence**: Maximum detail for offline analysis
2. **Unique Log Directory per Experiment**: Organize by timestamp/config
3. **Document Config**: Save environment variables with results
4. **Post-Experiment Cleanup**: Clear trails via API after data collection

### For Development & Debugging

1. **Full Logging in Response**: Immediate visibility without disk I/O
2. **Selective Token Recording**: Enable only for specific failing cases
3. **Use Management API**: Query specific trails for debugging

---

## Advanced Configuration

### Programmatic Configuration (Python)
```python
from vllm.v1.structured_output.audit_integration import (
    StructuredOutputAuditConfig,
    initialize_audit_system
)

config = StructuredOutputAuditConfig(
    enabled=True,
    record_allowed_tokens=False,
    record_full_events=True,
    max_trails_in_memory=500,
    persist_to_disk=True,
    audit_log_dir="/custom/audit/path",
    response_detail_level="full"
)

tracker = initialize_audit_system(config)
```

### Custom Event Recording (Advanced)
```python
from vllm.v1.structured_output.audit_tracker import (
    get_audit_tracker,
    AuditEventType
)

tracker = get_audit_tracker()

# Record custom event
if tracker.is_enabled():
    tracker.record_event(
        request_id="custom-req-123",
        event_type=AuditEventType.BITMASK_UPDATE,
        current_state_id="state_42",
        metadata={"custom_field": "value"}
    )
```

---

## FAQ

**Q: What is the performance impact?**
A: Summary mode: ~5-10% overhead. Full logging: ~2× latency. See [Benchmarking](#performance-benchmarking).

**Q: Can I use audit with speculative decoding?**
A: Yes, audit system is fully compatible with speculative decoding.

**Q: How long are trails kept in memory?**
A: Until the LRU cache limit (`VLLM_AUDIT_MAX_TRAILS`) is reached, or manually deleted via API.

**Q: Can I use audit with streaming responses?**
A: Yes, but audit data is only available after the complete response is finalized.

**Q: Is audit data encrypted?**
A: Persisted logs are plain JSON. Apply filesystem-level encryption if needed.

**Q: Can I export to formats other than JSON?**
A: The Management API returns JSON. Use post-processing scripts for CSV/Parquet conversion.

---

## Support

- **GitHub Issues**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Documentation**: [docs.vllm.ai](https://docs.vllm.ai)
- **API Reference**: See [AUDIT_API_REFERENCE.md](./AUDIT_API_REFERENCE.md)