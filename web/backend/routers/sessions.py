from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..main import session_manager


class SessionResponse(BaseModel):  # Minimal session listing model
    session_id: str
    prompt: str
    asset_name: str
    current_phase: str
    completion_percentage: float
    success: bool
    created_at: str
    updated_at: str


class ProgressMessageResponse(BaseModel):  # Progress message for UI
    timestamp: str
    phase: str
    title: str
    description: str
    progress_percent: float
    estimated_remaining_seconds: float | None = None
    technical_details: dict[str, Any] | None = None


router = APIRouter()


@router.get("/api/sessions", response_model=list[SessionResponse])
async def list_sessions(limit: int = 50):  # noqa: D401
    try:
        sessions = session_manager.list_sessions(limit=limit)
        return [
            SessionResponse(
                session_id=s["session_id"],
                prompt=s["prompt"],
                asset_name=s["asset_name"],
                current_phase=s.get("current_phase", "unknown"),
                completion_percentage=s.get("completion_percentage", 0.0),
                success=s.get("success", False),
                created_at=s.get("created_at", datetime.utcnow().isoformat()),
                updated_at=s.get("updated_at", s.get("created_at", datetime.utcnow().isoformat())),
            )
            for s in sessions
        ]
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e}")


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):  # noqa: D401
    try:
        ss = session_manager.load_session(session_id)
        if not ss:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": ss.session_id,
            "prompt": ss.prompt,
            "asset_name": ss.asset_name,
            "current_phase": getattr(ss.current_phase, "value", str(ss.current_phase)),
            "completion_percentage": ss.completion_percentage,
            "success": ss.success,
            "created_at": ss.created_at.isoformat() if hasattr(ss.created_at, "isoformat") else str(ss.created_at),
            "updated_at": ss.updated_at.isoformat() if hasattr(ss.updated_at, "isoformat") else str(ss.updated_at),
            "total_duration": ss.total_duration,
            "traces": [
                {
                    "phase": getattr(t.phase, "value", str(t.phase)),
                    "start_time": t.start_time.isoformat() if hasattr(t.start_time, "isoformat") else str(t.start_time),
                    "end_time": t.end_time.isoformat() if getattr(t, "end_time", None) else None,
                    "duration_seconds": getattr(t, "duration_seconds", None),
                    "status": getattr(t, "status", None),
                    "errors": getattr(t, "errors", []),
                    "llm_calls": len(getattr(t, "llm_calls", [])),
                    "agent_actions": len(getattr(t, "agent_actions", [])),
                }
                for t in ss.traces
            ],
            "code_versions": [
                {
                    "version": v.version,
                    "timestamp": v.timestamp.isoformat() if hasattr(v.timestamp, "isoformat") else str(v.timestamp),
                    "source_phase": getattr(v.source_phase, "value", str(v.source_phase)),
                    "validation_status": getattr(v, "validation_status", None),
                    "healing_applied": getattr(v, "healing_applied", None),
                    "quality_score": getattr(v, "quality_score", None),
                    "code_length": len(getattr(v, "code_content", "")),
                }
                for v in ss.code_versions
            ],
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "artifact_type": getattr(a.artifact_type, "value", str(a.artifact_type)),
                    "file_path": a.file_path,
                    "relative_path": a.relative_path,
                    "created_at": a.created_at.isoformat() if hasattr(a.created_at, "isoformat") else str(a.created_at),
                    "phase": getattr(a.phase, "value", str(a.phase)),
                    "version": a.version,
                    "size_bytes": a.size_bytes,
                    "tags": a.tags,
                }
                for a in ss.artifacts
            ],
            "final_result": ss.final_result,
        }
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to get session: {e}")


@router.get("/api/sessions/{session_id}/progress", response_model=list[ProgressMessageResponse])
async def get_session_progress(session_id: str, limit: int = 50):  # noqa: D401
    try:
        progress_messages = session_manager.get_progress_messages(session_id, limit=limit)
        return [ProgressMessageResponse(**msg) for msg in progress_messages]
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {e}")


@router.get("/api/sessions/{session_id}/enhanced-progress")
async def get_enhanced_progress_summary(session_id: str):  # noqa: D401
    try:
        summary = session_manager.get_enhanced_progress_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced progress: {e}")


class StepControlResponse(BaseModel):  # noqa: D101
    execution_mode: str
    current_step: int
    total_steps: int
    can_pause: bool
    can_resume: bool
    can_step: bool
    awaiting_user_action: bool


class StepControlRequest(BaseModel):  # noqa: D101
    action: str  # 'pause', 'resume', 'step'


@router.get("/api/sessions/{session_id}/step-control", response_model=StepControlResponse)
async def get_step_control_status(session_id: str):  # noqa: D401
    sc = session_manager.get_step_control_status(session_id)
    if not sc:
        raise HTTPException(status_code=404, detail="Session not found")
    return StepControlResponse(**sc)


@router.post("/api/sessions/{session_id}/step-control")
async def control_session_execution(session_id: str, request: StepControlRequest):  # noqa: D401
    if request.action == "pause":
        ok = session_manager.pause_session(session_id)
    elif request.action == "resume":
        ok = session_manager.resume_session(session_id)
    elif request.action == "step":
        ok = session_manager.step_session(session_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'pause', 'resume', or 'step'")
    if not ok:
        raise HTTPException(status_code=400, detail=f"Cannot {request.action} session at this time")
    # Frontend reloads status after control; return simple payload
    return {"success": True, "action": request.action}
