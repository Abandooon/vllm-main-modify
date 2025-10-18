# audit_admin_api.py
"""
Audit Management API for vLLM Structured Output

This module provides HTTP endpoints for querying and managing audit trails.
"""

import json
import time
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


@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_statistics():
    """Get global audit system statistics."""
    tracker = get_audit_tracker()

    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Audit tracker not initialized"
        )

    all_trails = tracker.get_all_trails()

    if not all_trails:
        return AuditStatsResponse(
            enabled=tracker.is_enabled(),
            total_trails=0,
            active_trails=0,
            total_events_recorded=0,
            avg_steps_per_trail=0.0,
            avg_duration_seconds=0.0,
            memory_usage_trails=0
        )

    total_events = sum(len(trail.events) for trail in all_trails.values())
    active_count = sum(1 for trail in all_trails.values() if trail.end_time is None)

    finalized_trails = [t for t in all_trails.values() if t.end_time is not None]
    avg_steps = (
        sum(t.total_steps for t in all_trails.values()) / len(all_trails)
        if all_trails else 0.0
    )
    avg_duration = (
        sum(t.end_time - t.start_time for t in finalized_trails) / len(finalized_trails)
        if finalized_trails else 0.0
    )

    return AuditStatsResponse(
        enabled=tracker.is_enabled(),
        total_trails=len(all_trails),
        active_trails=active_count,
        total_events_recorded=total_events,
        avg_steps_per_trail=avg_steps,
        avg_duration_seconds=avg_duration,
        memory_usage_trails=len(all_trails)
    )

from typing import Annotated
@router.get("/list", response_model=List[AuditTrailSummary])
async def list_audit_trails(
        limit: Annotated[int, Query(ge=1, le=1000, description="Maximum trails to return")] = 100,
        offset: Annotated[int, Query(ge=0, description="Number of trails to skip")] = 0,
        backend_type: Annotated[Optional[str], Query(description="Filter by backend type")] = None,
        include_active: Annotated[bool, Query(description="Include active (not finalized) trails")] = True
):
    """List audit trails with optional filtering."""
    tracker = get_audit_tracker()

    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Audit tracker not initialized"
        )

    all_trails = tracker.get_all_trails()

    # Filter trails
    filtered_trails = []
    for trail in all_trails.values():
        if backend_type and trail.backend_type != backend_type:
            continue
        if not include_active and trail.end_time is None:
            continue
        filtered_trails.append(trail)

    # Sort by start_time (newest first)
    filtered_trails.sort(key=lambda t: t.start_time, reverse=True)

    # Apply pagination
    paginated_trails = filtered_trails[offset:offset + limit]

    # Convert to response model
    return [
        AuditTrailSummary(
            request_id=trail.request_id,
            backend_type=trail.backend_type,
            start_time=trail.start_time,
            end_time=trail.end_time,
            duration=trail.end_time - trail.start_time if trail.end_time else None,
            total_steps=trail.total_steps,
            total_tokens_generated=trail.total_tokens_generated,
            total_rollbacks=trail.total_rollbacks,
            total_errors=trail.total_errors
        )
        for trail in paginated_trails
    ]


@router.get("/trail/{request_id}", response_model=AuditTrailDetail)
async def get_audit_trail(
        request_id: str,
        include_events: Annotated[bool, Query(description="Include full event data")] = True
):
    """Get detailed information for a specific audit trail."""
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

    trail_dict = trail.to_dict(include_events=include_events)

    return AuditTrailDetail(
        request_id=trail_dict["request_id"],
        backend_type=trail_dict["backend_type"],
        grammar_spec=trail_dict.get("grammar_spec"),
        start_time=trail_dict["start_time"],
        end_time=trail_dict.get("end_time"),
        duration=trail_dict.get("duration"),
        total_steps=trail_dict["total_steps"],
        total_tokens_generated=trail_dict["total_tokens_generated"],
        total_rollbacks=trail_dict["total_rollbacks"],
        total_errors=trail_dict["total_errors"],
        events=trail_dict.get("events", [])
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