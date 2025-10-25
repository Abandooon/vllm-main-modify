# audit_admin_api.py
"""
Audit Management API for vLLM Structured Output

This module provides HTTP endpoints for querying and managing audit trails.
"""

import json
import time

import os
import glob
from collections import defaultdict

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from vllm.v1.structured_output.audit_tracker import get_audit_tracker
from vllm.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# API Models
# ============================================================================

class AuditStatsResponse(BaseModel):
    """Global audit statistics."""
    enabled: bool
    total_trails: int
    active_trails: int  # Trails not yet finalized
    total_events_recorded: int
    avg_steps_per_trail: float
    avg_duration_seconds: float
    memory_usage_trails: int  # Number of trails in memory


class AuditTrailSummary(BaseModel):
    """Summary of a single audit trail."""
    request_id: str
    backend_type: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    total_steps: int
    total_tokens_generated: int
    total_rollbacks: int
    total_errors: int


class AuditTrailDetail(BaseModel):
    """Detailed audit trail with events."""
    request_id: str
    backend_type: str
    grammar_spec: Optional[str]
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    total_steps: int
    total_tokens_generated: int
    total_rollbacks: int
    total_errors: int
    events: List[Dict[str, Any]]


class AuditExportRequest(BaseModel):
    """Request model for bulk audit export."""
    request_ids: Optional[List[str]] = Field(
        None, description="Specific request IDs to export (None for all)"
    )
    include_events: bool = Field(
        True, description="Whether to include full event data"
    )
    start_time: Optional[float] = Field(
        None, description="Filter trails after this timestamp"
    )
    end_time: Optional[float] = Field(
        None, description="Filter trails before this timestamp"
    )


class AuditExportResponse(BaseModel):
    """Response model for bulk export."""
    export_timestamp: float
    trail_count: int
    trails: List[Dict[str, Any]]


# ============================================================================
# API Router
# ============================================================================

router = APIRouter(prefix="/v1/admin/audit", tags=["audit"])

def _load_persisted_trails() -> Dict[str, Dict[str, Any]]:
    """
    从 /audit-logs/*.ndjson 重建跨进程的 trail 视图。
    返回: { request_id: { "backend_type":..., "start_time":..., "end_time":..., "events":[...], "stats":{...} } }
    """
    trails: Dict[str, Dict[str, Any]] = {}
    log_dir = os.environ.get("VLLM_AUDIT_LOG_DIR")
    persist_enabled = os.environ.get("VLLM_AUDIT_PERSIST", "false").lower() == "true"

    if not (persist_enabled and log_dir and os.path.isdir(log_dir)):
        return trails  # 没挂卷/没开启持久化，就直接返回空

    # 我们默认只有一个 audit.ndjson，但也支持多文件（以后可能按时间/进程切分）
    pattern_list = [
        os.path.join(log_dir, "audit.ndjson"),
        os.path.join(log_dir, "audit_*.ndjson"),
    ]

    for pattern in pattern_list:
        for fname in glob.glob(pattern):
            try:
                with open(fname, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue

                        rid = rec.get("request_id")
                        if not rid:
                            continue

                        # 确保初始化
                        t = trails.setdefault(rid, {
                            "backend_type": None,
                            "start_time": None,
                            "end_time": None,
                            "grammar_spec": None,
                            "events": [],
                            "tot_tokens": 0,
                            "rollbacks": 0,
                            "errors": 0,
                        })

                        rtype = rec.get("record_type")

                        if rtype == "start_trail":
                            t["backend_type"] = rec.get("backend_type", t["backend_type"])
                            t["start_time"] = rec.get("start_time", t["start_time"])
                            # 短的 grammar_spec 才记录
                            if rec.get("grammar_spec"):
                                t["grammar_spec"] = rec["grammar_spec"]

                        elif rtype == "event":
                            ev = rec.get("event", {})
                            t["events"].append(ev)

                            # 统计信息
                            etype = ev.get("event_type")
                            if etype == "token_accept":
                                acc = ev.get("accepted_tokens") or []
                                t["tot_tokens"] += len(acc)
                            elif etype == "rollback":
                                t["rollbacks"] += 1
                            elif etype == "error":
                                t["errors"] += 1

                        elif rtype == "finalize":
                            summary = rec.get("summary", {})
                            # 尝试从 summary 里拉更多聚合字段
                            t["end_time"] = summary.get("end_time", t["end_time"])
                            # 如果 summary 里已经有统计，就覆盖本地的
                            if "total_tokens_generated" in summary:
                                t["tot_tokens"] = summary["total_tokens_generated"]
                            if "total_rollbacks" in summary:
                                t["rollbacks"] = summary["total_rollbacks"]
                            if "total_errors" in summary:
                                t["errors"] = summary["total_errors"]

            except Exception as e:
                logger.warning(f"[AuditPersistRead] Failed reading {fname}: {e}")

    return trails


def _merge_memory_and_disk(tracker):
    """
    返回 { request_id: merged_trail_obj }
    merged_trail_obj 的结构跟 _load_persisted_trails() 里的一样 + 补齐内存统计
    """
    disk_trails = _load_persisted_trails()

    # 合并内存里的 trails（APIServer 自己 pid=1 的 _global_audit_tracker）
    if tracker and tracker.is_enabled():
        mem_trails = tracker.get_all_trails()
        for rid, trail in mem_trails.items():
            mt = disk_trails.setdefault(rid, {
                "backend_type": trail.backend_type,
                "start_time": trail.start_time,
                "end_time": trail.end_time,
                "grammar_spec": trail.grammar_spec,
                "events": [],
                "tot_tokens": 0,
                "rollbacks": 0,
                "errors": 0,
            })

            # 覆盖基础元信息（内存一般更全）
            mt["backend_type"] = trail.backend_type or mt["backend_type"]
            mt["start_time"] = trail.start_time or mt["start_time"]
            mt["end_time"] = trail.end_time or mt["end_time"]
            if trail.grammar_spec and len(trail.grammar_spec) < 1000:
                mt["grammar_spec"] = trail.grammar_spec

            # 事件
            for ev in trail.events:
                mt["events"].append(ev.to_dict())

            # 统计
            mt["tot_tokens"] = max(mt["tot_tokens"], trail.total_tokens_generated)
            mt["rollbacks"] = max(mt["rollbacks"], trail.total_rollbacks)
            mt["errors"] = max(mt["errors"], trail.total_errors)

    return disk_trails

@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_statistics():
    tracker = get_audit_tracker()
    merged = _merge_memory_and_disk(tracker)

    if not merged:
        # 没有任何 trail
        return AuditStatsResponse(
            enabled=bool(tracker and tracker.is_enabled()),
            total_trails=0,
            active_trails=0,
            total_events_recorded=0,
            avg_steps_per_trail=0.0,
            avg_duration_seconds=0.0,
            memory_usage_trails=0
        )

    # 计算统计
    total_trails = len(merged)
    total_events_recorded = 0
    active_trails = 0
    durations = []
    steps_per_trail = []

    for rid, t in merged.items():
        evs = t.get("events", [])
        total_events_recorded += len(evs)

        start_time = t.get("start_time")
        end_time = t.get("end_time")
        if end_time is None:
            active_trails += 1
        else:
            durations.append(end_time - start_time if start_time else 0)

        steps_per_trail.append(len(evs))

    avg_steps = (sum(steps_per_trail) / len(steps_per_trail)) if steps_per_trail else 0.0
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0

    return AuditStatsResponse(
        enabled=bool(tracker and tracker.is_enabled()),
        total_trails=total_trails,
        active_trails=active_trails,
        total_events_recorded=total_events_recorded,
        avg_steps_per_trail=avg_steps,
        avg_duration_seconds=avg_duration,
        memory_usage_trails=len(merged)
    )


from typing import Annotated
@router.get("/list", response_model=List[AuditTrailSummary])
async def list_audit_trails(
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum trails to return")] = 100,
    offset: Annotated[int, Query(ge=0, description="Number of trails to skip")] = 0,
    backend_type: Annotated[Optional[str], Query(description="Filter by backend type")] = None,
    include_active: Annotated[bool, Query(description="Include active (not finalized) trails")] = True
):
    tracker = get_audit_tracker()
    merged = _merge_memory_and_disk(tracker)

    # 过滤
    filtered = []
    for rid, t in merged.items():
        if backend_type and t.get("backend_type") != backend_type:
            continue
        if not include_active and t.get("end_time") is None:
            continue
        filtered.append((rid, t))

    # 按 start_time 倒序
    filtered.sort(key=lambda pair: pair[1].get("start_time", 0.0), reverse=True)

    # 分页
    page = filtered[offset:offset + limit]

    # 组装响应
    result = []
    for rid, t in page:
        st = t.get("start_time")
        et = t.get("end_time")
        duration = (et - st) if (st is not None and et is not None) else None

        result.append(AuditTrailSummary(
            request_id=rid,
            backend_type=t.get("backend_type"),
            start_time=st,
            end_time=et,
            duration=duration,
            total_steps=len(t.get("events", [])),
            total_tokens_generated=t.get("tot_tokens", 0),
            total_rollbacks=t.get("rollbacks", 0),
            total_errors=t.get("errors", 0)
        ))

    return result



@router.get("/trail/{request_id}", response_model=AuditTrailDetail)
async def get_audit_trail(
    request_id: str,
    include_events: Annotated[bool, Query(description="Include full event data")] = True
):
    tracker = get_audit_tracker()
    merged = _merge_memory_and_disk(tracker)

    if request_id not in merged:
        raise HTTPException(
            status_code=404,
            detail=f"Audit trail not found for request_id: {request_id}"
        )

    t = merged[request_id]
    st = t.get("start_time")
    et = t.get("end_time")
    duration = (et - st) if (st is not None and et is not None) else None

    events = t.get("events", []) if include_events else []

    return AuditTrailDetail(
        request_id=request_id,
        backend_type=t.get("backend_type"),
        grammar_spec=t.get("grammar_spec"),
        start_time=st,
        end_time=et,
        duration=duration,
        total_steps=len(t.get("events", [])),
        total_tokens_generated=t.get("tot_tokens", 0),
        total_rollbacks=t.get("rollbacks", 0),
        total_errors=t.get("errors", 0),
        events=events
    )



@router.post("/export", response_model=AuditExportResponse)
async def export_audit_trails(request: AuditExportRequest):
    """Export audit trails in bulk."""
    tracker = get_audit_tracker()

    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Audit tracker not initialized"
        )

    all_trails = tracker.get_all_trails()

    # Apply filters
    filtered_trails = {}
    for req_id, trail in all_trails.items():
        # Filter by request_ids
        if request.request_ids and req_id not in request.request_ids:
            continue

        # Filter by time range
        if request.start_time and trail.start_time < request.start_time:
            continue
        if request.end_time:
            trail_end = trail.end_time if trail.end_time else time.time()
            if trail_end > request.end_time:
                continue

        filtered_trails[req_id] = trail

    # Convert to export format
    export_data = [
        trail.to_dict(include_events=request.include_events)
        for trail in filtered_trails.values()
    ]

    return AuditExportResponse(
        export_timestamp=time.time(),
        trail_count=len(export_data),
        trails=export_data
    )


@router.delete("/trail/{request_id}")
async def delete_audit_trail(request_id: str):
    """Delete a specific audit trail from memory."""
    tracker = get_audit_tracker()

    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Audit tracker not initialized"
        )

    trail = tracker.get_trail(request_id)
    if not trail:
        raise HTTPException(
            status_code=404,
            detail=f"Audit trail not found for request_id: {request_id}"
        )

    tracker.cleanup_trail(request_id)

    return {"status": "success", "message": f"Deleted trail for {request_id}"}


@router.post("/clear")
async def clear_all_trails(
        confirm: Annotated[bool, Query(description="Must be true to confirm deletion")] = False
):
    """Clear all audit trails from memory."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear all trails"
        )

    tracker = get_audit_tracker()

    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Audit tracker not initialized"
        )

    all_trails = list(tracker.get_all_trails().keys())
    for req_id in all_trails:
        tracker.cleanup_trail(req_id)

    return {
        "status": "success",
        "message": f"Cleared {len(all_trails)} audit trails"
    }


@router.get("/health")
async def audit_health_check():
    """Health check endpoint for audit system."""
    tracker = get_audit_tracker()

    if not tracker:
        return {
            "status": "unavailable",
            "enabled": False,
            "message": "Audit tracker not initialized"
        }

    return {
        "status": "healthy",
        "enabled": tracker.is_enabled(),
        "trails_in_memory": len(tracker.get_all_trails())
    }


def register_audit_routes(app):
    """Register audit admin routes with the main FastAPI app."""
    app.include_router(router)
    logger.info("Audit admin API routes registered at /v1/admin/audit")