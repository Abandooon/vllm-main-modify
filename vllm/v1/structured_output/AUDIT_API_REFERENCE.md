# Audit System API Reference

Complete reference for the vLLM Structured Output Audit Management API.

**Base URL**: `http://localhost:8000/v1/admin/audit`

---

## Authentication

Currently, the audit API does not require authentication. In production deployments, use:
- Reverse proxy with API key validation (e.g., nginx + auth_request)
- Network-level restrictions (firewall, VPC)
- vLLM's built-in API key middleware (if available)

---

## Endpoints

### 1. GET /stats

Get global audit system statistics.

**Request**:
```http
GET /v1/admin/audit/stats HTTP/1.1
Host: localhost:8000
```

**Response** (200 OK):
```json
{
  "enabled": true,
  "total_trails": 1523,
  "active_trails": 12,
  "total_events_recorded": 38075,
  "avg_steps_per_trail": 25.0,
  "avg_duration_seconds": 0.234,
  "memory_usage_trails": 1523
}
```

**Fields**:
- `enabled` (boolean): Whether audit system is active
- `total_trails` (integer): Total trails currently in memory
- `active_trails` (integer): Trails not yet finalized (still generating)
- `total_events_recorded` (integer): Sum of events across all trails
- `avg_steps_per_trail` (float): Mean of `total_steps` across trails
- `avg_duration_seconds` (float): Mean generation duration for finalized trails
- `memory_usage_trails` (integer): Current memory footprint (trail count)

**Error Responses**:
- `503 Service Unavailable`: Audit tracker not initialized

---

### 2. GET /list

List audit trails with pagination and filtering.

**Request**:
```http
GET /v1/admin/audit/list?limit=10&offset=0&backend_type=xgrammar&include_active=true HTTP/1.1
Host: localhost:8000
```

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Max trails to return (1-1000) |
| `offset` | integer | 0 | Number of trails to skip (pagination) |
| `backend_type` | string | null | Filter by backend (`xgrammar`, `outlines`, `guidance`) |
| `include_active` | boolean | true | Include non-finalized trails |

**Response** (200 OK):
```json
[
  {
    "request_id": "cmpl-7f8a9b0c",
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

**Notes**:
- Results sorted by `start_time` (newest first)
- `duration` is null for active trails
- Empty array if no matching trails

---

### 3. GET /trail/{request_id}

Get detailed audit trail for a specific request.

**Request**:
```http
GET /v1/admin/audit/trail/cmpl-7f8a9b0c?include_events=true HTTP/1.1
Host: localhost:8000
```

**Path Parameters**:
- `request_id` (string, required): The unique request identifier

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_events` | boolean | true | Include full event sequence |

**Response** (200 OK):
```json
{
  "request_id": "cmpl-7f8a9b0c",
  "backend_type": "xgrammar",
  "grammar_spec": null,
  "start_time": 1705555555.123,
  "end_time": 1705555555.357,
  "duration": 0.234,
  "total_steps": 25,
  "total_tokens_generated": 25,
  "total_rollbacks": 0,
  "total_errors": 0,
  "events": [
    {
      "timestamp": 1705555555.125,
      "timestamp_ns": 1705555555125000000,
      "step_number": 1,
      "event_type": "state_transition",
      "request_id": "cmpl-7f8a9b0c",
      "current_state_id": "state_1",
      "previous_state_id": "state_0",
      "selected_token": 42,
      "metadata": {"num_processed_tokens": 1}
    }
  ]
}
```

**Event Types**:
- `state_init`: Initial state setup
- `token_accept`: Tokens accepted by FSM
- `token_reject`: Tokens rejected by FSM
- `token_validate`: Validation check (lookahead)
- `state_transition`: FSM state change
- `bitmask_update`: Allowed token set update
- `rollback`: Speculative tokens rolled back
- `termination`: Generation completed
- `error`: Error occurred

**Error Responses**:
- `404 Not Found`: Trail does not exist
- `503 Service Unavailable`: Audit tracker not initialized

---

### 4. POST /export

Bulk export audit trails with filtering.

**Request**:
```http
POST /v1/admin/audit/export HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "request_ids": ["cmpl-abc", "cmpl-def"],
  "include_events": true,
  "start_time": 1705555555.0,
  "end_time": 1705666666.0
}
```

**Request Body**:
```json
{
  "request_ids": ["string"],  // Optional: specific trails to export
  "include_events": true,      // Optional: include full event data
  "start_time": 1705555555.0,  // Optional: filter by start time (Unix timestamp)
  "end_time": 1705666666.0     // Optional: filter by end time (Unix timestamp)
}
```

**Response** (200 OK):
```json
{
  "export_timestamp": 1705777777.0,
  "trail_count": 2,
  "trails": [
    {
      "request_id": "cmpl-abc",
      "backend_type": "xgrammar",
      "start_time": 1705555555.0,
      ...
    }
  ]
}
```

**Use Cases**:
- Daily/weekly audit archival
- Compliance reporting
- Offline analysis dataset creation
- Backup before clearing memory

---

### 5. DELETE /trail/{request_id}

Delete a specific audit trail from memory.

**Request**:
```http
DELETE /v1/admin/audit/trail/cmpl-7f8a9b0c HTTP/1.1
Host: localhost:8000
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Deleted trail for cmpl-7f8a9b0c"
}
```

**Notes**:
- Only removes from in-memory LRU cache
- Does NOT delete persisted log files
- Useful for manual memory management

**Error Responses**:
- `404 Not Found`: Trail does not exist

---

### 6. POST /clear

Clear all audit trails from memory (destructive).

**Request**:
```http
POST /v1/admin/audit/clear?confirm=true HTTP/1.1
Host: localhost:8000
```

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | boolean | **Yes** | Must be `true` to execute |

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Cleared 1523 audit trails"
}
```

**Security Notes**:
- ⚠️ **Destructive operation**: Cannot be undone
- Requires explicit `confirm=true` parameter
- Consider exporting data before clearing
- Does NOT affect persisted log files

**Error Responses**:
- `400 Bad Request`: Missing or false `confirm` parameter

---

### 7. GET /health

Health check endpoint for audit system.

**Request**:
```http
GET /v1/admin/audit/health HTTP/1.1
Host: localhost:8000
```

**Response** (200 OK - Healthy):
```json
{
  "status": "healthy",
  "enabled": true,
  "trails_in_memory": 1523
}
```

**Response** (200 OK - Unavailable):
```json
{
  "status": "unavailable",
  "enabled": false,
  "message": "Audit tracker not initialized"
}
```

**Use Cases**:
- Kubernetes readiness/liveness probes
- Monitoring system health checks
- CI/CD pipeline validation

---

## Common Workflows

### Workflow 1: Daily Audit Export
```bash
#!/bin/bash
# Export yesterday's audit trails

TODAY=$(date +%s)
YESTERDAY=$((TODAY - 86400))

curl -X POST http://localhost:8000/v1/admin/audit/export \
  -H "Content-Type: application/json" \
  -d "{
    \"include_events\": true,
    \"start_time\": $YESTERDAY,
    \"end_time\": $TODAY
  }" \
  > "audit_export_$(date +%Y%m%d).json"
```

### Workflow 2: Monitor Memory Usage
```bash
#!/bin/bash
# Alert if audit trails exceed threshold

STATS=$(curl -s http://localhost:8000/v1/admin/audit/stats)
TRAILS=$(echo $STATS | jq '.total_trails')

if [ "$TRAILS" -gt 900 ]; then
  echo "WARNING: Audit trails approaching limit ($TRAILS/1000)"
  # Trigger cleanup or alert
fi
```

### Workflow 3: Debug Specific Request
```python
import requests

request_id = "cmpl-problematic-123"

# Get full trail with events
response = requests.get(
    f"http://localhost:8000/v1/admin/audit/trail/{request_id}",
    params={"include_events": True}
)

trail = response.json()

# Analyze events for failures
for event in trail["events"]:
    if event["event_type"] == "error":
        print(f"Error at step {event['step_number']}: {event['error_message']}")
```

---

## Rate Limits

Currently, there are no rate limits on the Audit Management API. In production:

- Implement rate limiting at the reverse proxy level
- Monitor `/stats` endpoint frequency (recommend: max 1/sec)
- Batch `/list` queries using pagination
- Use `/export` for bulk operations instead of many `/trail` calls

---

## Data Retention

**In-Memory Trails**:
- Retained until LRU cache limit (`VLLM_AUDIT_MAX_TRAILS`)
- Oldest trails evicted automatically
- Can be manually deleted via `/trail/{id}` or `/clear`

**Persisted Logs**:
- Retained indefinitely (no automatic cleanup)
- Implement log rotation (e.g., delete files older than N days)
- Compress old logs to save disk space

---

## Security Considerations

1. **Access Control**: Audit API exposes operational metrics. Restrict access via:
   - Firewall rules (allow only trusted IPs)
   - Reverse proxy authentication
   - Network segmentation (management network only)

2. **Data Sensitivity**: Audit trails may contain:
   - Grammar specifications (may reveal business logic)
   - Token sequences (potentially sensitive data)
   - Consider encryption at rest for persisted logs

3. **Denial of Service**: Protect against:
   - Excessive `/list` queries with large `limit` values
   - Rapid `/export` requests causing disk I/O spikes
   - Implement rate limiting

---

## Version History

- **v1.0** (2025-01): Initial release with core endpoints
- **v1.1** (TBD): Planned additions:
  - Streaming export for large datasets
  - Audit trail aggregation API
  - Prometheus metrics endpoint

---

## Support

For API issues or feature requests:
- GitHub Issues: [vllm-project/vllm](https://github.com/vllm-project/vllm/issues)
- Community Slack: [vllm.slack.com](https://vllm.slack.com)