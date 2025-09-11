from __future__ import annotations

"""LL3M Web Interface Backend

FastAPI server providing REST API and WebSocket support for the LL3M web interface.
Integrates with the existing LL3M stateful session system for seamless asset generation.
"""

import asyncio
import contextlib
import json
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .routers.diagnostics import router as diagnostics_router
from .routers.assets import router as assets_router

# Standalone mode: allow running backend with only web/backend requirements
STANDALONE = os.getenv("WEB_BACKEND_STANDALONE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Add project root to path for LL3M imports (only needed in full mode)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Initialize logger
logger = structlog.get_logger(__name__)

print("🚀 Starting LL3M Web Backend v1.0.0")
print(f"📁 Working directory: {Path.cwd()}")
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path includes: {list(sys.path[:3])}")
print(f"🔧 STANDALONE mode: {STANDALONE}")
print(f"🌍 USE_LOCAL_LLM: {os.getenv('USE_LOCAL_LLM', 'not set')}")
print(f"🔑 OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")

if not STANDALONE:
    print("🔄 Attempting to import LL3M services...")
    try:
        print("  📦 Importing session integration...")
        from src.services.session_integration import (  # noqa: F401
            create_session_tracker,
            resume_session_tracker,
        )

        print("  📦 Importing stateful session manager...")
        from src.services.stateful_session_manager import StatefulSessionManager

        print("  📦 Importing orchestrator...")
        from src.workflow.orchestrator import LL3MOrchestrator

        print("✅ LL3M services imported successfully - FULL MODE ENABLED")
    except Exception as e:  # pragma: no cover - defensive startup
        print(f"⚠️  Failed to import LL3M services; falling back to standalone mode: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        STANDALONE = True
        StatefulSessionManager = None  # type: ignore[assignment]
        LL3MOrchestrator = None  # type: ignore[assignment]

if STANDALONE:

    class SimpleSessionManager:  # minimal stub for standalone mode
        def list_sessions(self, limit: int = 50):
            return []

        def load_session(self, session_id: str):
            return None

        def get_progress_messages(self, session_id: str, limit: int = 50):
            return []

        def get_enhanced_progress_summary(self, session_id: str):
            return None

        def add_substep_progress(self, *args, **kwargs):  # noqa: D401
            return None

        def update_substep_progress(self, *args, **kwargs):
            return None

        def create_interaction_point(self, *args, **kwargs):
            return None

        def resolve_interaction_point(self, *args, **kwargs):
            return False

        def add_agent_reflection(self, *args, **kwargs):
            return None

        def get_step_control_status(self, *args, **kwargs):
            return None

        def pause_session(self, *args, **kwargs):
            return False

        def resume_session(self, *args, **kwargs):
            return False

        def step_session(self, *args, **kwargs):
            return False

    print("🔧 Initializing in STANDALONE mode - generation features disabled")
    session_manager = SimpleSessionManager()
    orchestrator = None
    print("✅ Standalone session manager initialized")
else:
    print("🔧 Initializing in FULL mode - all features enabled")
    session_manager = StatefulSessionManager()  # type: ignore[misc]
    print("✅ Stateful session manager initialized")
    orchestrator = LL3MOrchestrator()  # type: ignore[misc]
    print("✅ LL3M orchestrator initialized")

# Publish session manager for routers without creating circular imports
try:
    from .deps import set_session_manager

    set_session_manager(session_manager)
except Exception:
    # In unusual startup contexts, tolerate failure; routers will error if used
    pass

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan handler replacing deprecated on_event(startup)."""
    # On startup, attempt to detect Blender and persist BLENDER_PATH.
    try:
        for cand in _candidate_blender_paths():
            if cand.exists() and cand.is_file():
                rc, _ = _run_blender_cmd(cand, ["--version"], timeout_s=8)
                if rc == 0:
                    rc2, _ = _run_blender_cmd(
                        cand,
                        ["--background", "--python-expr", "import bpy;print('ok')"],
                        timeout_s=15,
                    )
                    if rc2 == 0:
                        os.environ.setdefault("BLENDER_PATH", cand.as_posix())
                        with contextlib.suppress(Exception):
                            _persist_env_var("BLENDER_PATH", cand.as_posix())
                        break
    except Exception:
        # Non-fatal: we will surface status via /api/diagnostics
        pass

    yield


app = FastAPI(
    title="LL3M Web Interface API",
    description="Backend API for the LL3M 3D asset generation web interface",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular routers
app.include_router(diagnostics_router)
app.include_router(assets_router)
# Import sessions router after session manager has been set to avoid circulars
from .routers.sessions import router as sessions_router  # noqa: E402

app.include_router(sessions_router)


# Initialize LL3M services
def _require_full_mode():
    if orchestrator is None:
        error_msg = (
            "Backend running in standalone mode. Generation and session features are disabled. "
            "Unset WEB_BACKEND_STANDALONE or use 'make web-backend' for full mode."
        )
        print(f"🚫 FULL MODE REQUIRED: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=error_msg,
        )



# WebSocket connection manager
class ConnectionManager:  # noqa: D101
    def __init__(self):  # noqa: D107
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):  # noqa: D102
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):  # noqa: D102
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):  # noqa: D102
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception:
                # Connection closed, remove it
                self.disconnect(session_id)

    async def broadcast_to_session(self, session_id: str, event_type: str, data: Any):  # noqa: D102
        await self.send_message(
            session_id,
            {
                "type": event_type,
                "data": data,
                "timestamp": asyncio.get_event_loop().time(),
            },
        )


manager = ConnectionManager()


# Pydantic models
class GenerateAssetRequest(BaseModel):  # noqa: D101
    prompt: str
    asset_name: str = "asset"
    export_formats: list[str] = ["gltf", "obj"]  # Web-compatible formats for preview
    output_directory: str | None = None


class AmendSessionRequest(BaseModel):  # noqa: D101
    feedback: str
    step_id: str | None = None


class SessionResponse(BaseModel):  # noqa: D101
    session_id: str
    prompt: str
    asset_name: str
    current_phase: str
    completion_percentage: float
    success: bool
    created_at: str
    updated_at: str
    final_result: dict[str, Any] | None = None


class StepControlRequest(BaseModel):  # noqa: D101
    action: str  # 'pause', 'resume', 'step'


class UserConfirmationRequest(BaseModel):  # noqa: D101
    action: str  # 'accept', 'modify', 'reject'
    modified_prompt: str | None = None  # Only for 'modify' action


class ProgressMessageResponse(BaseModel):  # noqa: D101
    timestamp: str
    phase: str
    title: str
    description: str
    progress_percent: float
    estimated_remaining_seconds: float | None = None
    technical_details: dict[str, Any] | None = None
    # Optional granular details for richer UX in frontend
    substeps: list[dict[str, Any]] | None = None
    interaction_points: list[dict[str, Any]] | None = None
    agent_reflections: list[dict[str, Any]] | None = None


class StepControlResponse(BaseModel):  # noqa: D101
    execution_mode: str
    current_step: int
    total_steps: int
    can_pause: bool
    can_resume: bool
    can_step: bool
    awaiting_user_action: bool


class ImportAssetRequest(BaseModel):  # noqa: D101
    asset_name: str
    asset_type: str  # 'gltf', 'obj', 'blend', 'fbx'
    description: str | None = None
    tags: list[str] | None = None


class AssetMetadataResponse(BaseModel):  # noqa: D101
    asset_id: str
    asset_name: str
    asset_type: str
    file_path: str
    preview_path: str | None = None
    size_bytes: int
    created_at: str
    description: str | None = None
    tags: list[str] = []
    import_metadata: dict[str, Any] | None = None


class EditAssetRequest(BaseModel):  # noqa: D101
    edit_instruction: str
    preserve_materials: bool = True
    export_formats: list[str] = ["gltf", "obj"]  # Web-compatible formats for preview


class ExportRequest(BaseModel):  # noqa: D101
    format: str  # 'vrm', 'vrma', 'gltf', 'glb', 'fbx', 'obj'
    options: dict[str, Any] | None = None


class ExportResponse(BaseModel):  # noqa: D101
    success: bool
    format: str
    file_path: str | None = None
    message: str | None = None


class ImportAssetFromPathRequest(BaseModel):  # noqa: D101
    file_path: str
    asset_name: str
    asset_type: str
    description: str | None = None
    tags: list[str] | None = None


class ImportAssetFromUrlRequest(BaseModel):  # noqa: D101
    url: str
    asset_name: str
    asset_type: str
    description: str | None = None
    tags: list[str] | None = None


class ServiceCheck(BaseModel):  # noqa: D101
    name: str
    ok: bool
    status: str
    message: str | None = None
    details: dict[str, Any] | None = None


class DiagnosticsResponse(BaseModel):  # noqa: D101
    ok: bool
    timestamp: str
    services: dict[str, ServiceCheck]


class VectorDbResetRequest(BaseModel):  # noqa: D101
    confirm: bool = True
    remove_file: bool = False


class VectorDbResetResponse(BaseModel):  # noqa: D101
    success: bool
    mode: str
    message: str
    db_path: str


class MCPStartRequest(BaseModel):  # noqa: D101
    port: int | None = 3001
    log_path: str | None = None


class MCPStartResponse(BaseModel):  # noqa: D101
    started: bool
    pid: int | None = None
    port: int | None = None
    log_path: str | None = None
    message: str


class MCPStatusResponse(BaseModel):  # noqa: D101
    running: bool
    pid: int | None = None
    health_http: int | None = None
    message: str | None = None


class BlenderEnsureRequest(BaseModel):  # noqa: D101
    set_env: bool = True


class BlenderEnsureResponse(BaseModel):  # noqa: D101
    ok: bool
    path: str | None
    version: str | None
    headless_ok: bool
    message: str
    suggestions: list[str] = []


class InstallVRMAddonRequest(BaseModel):  # noqa: D101
    zip_url: str | None = None
    set_env: bool = True


class InstallVRMAddonResponse(BaseModel):  # noqa: D101
    installed: bool
    enabled: bool
    message: str
    zip_path: str | None = None


# API Routes
@app.get("/")
async def root():  # noqa: D103
    return {"message": "LL3M Web Interface API", "version": "1.0.0"}


@app.get("/health")
async def health():  # Simple health check for tooling
    return {"status": "ok", "service": "web-backend", "version": "1.0.0"}

@app.get("/api/debug")
async def debug_info(debug: bool = False):
    """Debug endpoint to show system status and logs."""
    if not debug:
        return {"message": "Add ?debug=true to enable debug mode"}
    
    # Get session manager status
    session_count = len(session_manager.sessions) if hasattr(session_manager, 'sessions') else 0
    
    # Get recent sessions
    recent_sessions = []
    if hasattr(session_manager, 'sessions'):
        for session_id, session in list(session_manager.sessions.items())[-5:]:
            recent_sessions.append({
                "session_id": session_id,
                "phase": session.current_phase.value if hasattr(session.current_phase, 'value') else str(session.current_phase),
                "success": session.success,
                "created_at": session.created_at.isoformat() if hasattr(session.created_at, 'isoformat') else str(session.created_at)
            })
    
    debug_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_manager": {
            "active_sessions": session_count,
            "type": type(session_manager).__name__,
            "recent_sessions": recent_sessions
        },
        "environment": {
            "USE_LOCAL_LLM": os.getenv("USE_LOCAL_LLM", "false"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "unknown"),
            "WEB_BACKEND_STANDALONE": os.getenv("WEB_BACKEND_STANDALONE", "unset"),
            "BLENDER_PATH": os.getenv("BLENDER_PATH", "not set"),
        }
    }
    
    return debug_info


async def _http_json(url: str, headers: dict[str, str] | None = None, timeout_s: int = 5):
    """Make HTTP JSON request with error handling."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r,
        ):
            try:
                data = await r.json()
            except Exception:
                data = {}
            return r.status, data
    except Exception as e:
        return -1, {"error": str(e)}


def _env_bool(name: str, default: str = "false") -> bool:
    """Parse environment variable as boolean."""
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}




# Helper functions
def _vector_db_path() -> Path:
    return Path(__file__).parent.parent.parent / "cache" / "vector_cache.db"

def _repo_root() -> Path:
    return Path(__file__).parent.parent.parent

def _mcp_pid_file() -> Path:
    return _repo_root() / "blender_mcp.pid"


def _env_file_path() -> Path:
    return _repo_root() / ".env"

def _persist_env_var(key: str, value: str) -> None:
    """Persist or update an env var in .env at repo root."""
    env_path = _env_file_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    
    if env_path.exists():
        text = env_path.read_text()
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(f"{key}={value}", text)
        else:
            text += f"\n{key}={value}\n" if not text.endswith("\n") else f"{key}={value}\n"
        env_path.write_text(text)
    else:
        env_path.write_text(f"{key}={value}\n")

# Platform-specific Blender detection
def _candidate_blender_paths() -> list[Path]:
    """Get candidate Blender executable paths."""
    candidates = []
    
    # Environment variables
    env_path = os.getenv("BLENDER_PATH") or os.getenv("BLENDER_EXECUTABLE")
    if env_path:
        candidates.append(Path(env_path))
    
    # System PATH
    if which := shutil.which("blender"):
        candidates.append(Path(which))
    
    # Platform-specific locations
    system = platform.system().lower()
    if system == "darwin":
        candidates.extend([
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"
        ])
        # Homebrew
        hb = Path("/opt/homebrew/Cellar/blender")
        if hb.exists():
            candidates.extend(hb.glob("*/Blender.app/Contents/MacOS/Blender"))
    else:  # Linux
        candidates.extend([
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"), 
            Path("/snap/bin/blender")
        ])
        # Flatpak
        fp = Path("/var/lib/flatpak/app/org.blender.Blender")
        if fp.exists():
            candidates.extend(fp.glob("*/active/files/bin/blender"))
    
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        try:
            key = c.resolve()
        except Exception:
            key = c
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique

def _run_blender_cmd(bin_path: Path, args: list[str], timeout_s: int = 15) -> tuple[int, str]:
    """Run Blender command with timeout."""
    try:
        proc = subprocess.run(
            [bin_path.as_posix(), *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or proc.stderr
    except Exception as e:
        return -1, str(e)


@app.post("/api/vector-db/reset", response_model=VectorDbResetResponse)
async def reset_vector_db(req: VectorDbResetRequest):  # noqa: D401
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    db_path = _vector_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if req.remove_file and db_path.exists():
        try:
            db_path.unlink()
            # Recreate empty DB
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        id TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        code TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        complexity_level INTEGER DEFAULT 1,
                        quality_score REAL,
                        success_rate REAL DEFAULT 0.0,
                        usage_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_used TEXT
                    )
                    """
                )
                conn.commit()
            return VectorDbResetResponse(
                success=True,
                mode="file_recreated",
                message="Vector DB file removed and recreated",
                db_path=str(db_path),
            )
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Failed to recreate DB: {e}")

    # Truncate table
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
        return VectorDbResetResponse(
            success=True,
            mode="truncate",
            message="All entries removed",
            db_path=str(db_path),
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to reset DB: {e}")


@app.get("/api/logs/mcp")
async def get_mcp_logs(limit: int = 200):  # noqa: D401
    """Return tail of Blender MCP logs if available.

    Looks for BLENDER_MCP_LOG_PATH, else blender_mcp.log in repo root, else /tmp/blender_mcp.log
    """
    candidates = [
        os.getenv("BLENDER_MCP_LOG_PATH"),
        str(Path(__file__).parent.parent.parent / "blender_mcp.log"),
        "/tmp/blender_mcp.log",
    ]
    path = next((c for c in candidates if c and Path(c).exists()), None)
    if not path:
        return {"exists": False, "path": None, "lines": []}

    try:
        p = Path(path)
        with open(p, errors="ignore") as f:
            lines = f.readlines()
        tail = lines[-limit:]
        return {"exists": True, "path": str(p), "lines": tail}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")


@app.get("/api/mcp/status", response_model=MCPStatusResponse)
async def mcp_status():  # noqa: D401
    pid_file = _mcp_pid_file()
    pid = None
    running = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # does not kill; raises if not running
            running = True
        except Exception:
            running = False
    status, _ = await _http_json("http://localhost:3001/health")
    return MCPStatusResponse(running=running, pid=pid, health_http=status if status >= 0 else None)


@app.post("/api/mcp/start", response_model=MCPStartResponse)
async def mcp_start(req: MCPStartRequest):  # noqa: D401
    root = _repo_root()
    script = root / "setup" / "start_blender_mcp.sh"
    if not script.exists():
        raise HTTPException(status_code=404, detail="start_blender_mcp.sh not found")

    env = os.environ.copy()
    if req.log_path:
        env["BLENDER_MCP_LOG_PATH"] = req.log_path
    env.setdefault("BLENDER_MCP_SERVER_PORT", str(req.port or 3001))

    try:
        # Spawn via nohup so it backgrounds
        cmd = [
            "bash",
            "-lc",
            f"nohup '{script.as_posix()}' {int(req.port or 3001)} > /dev/null 2>&1 & echo $!",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=root.as_posix(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, _ = proc.communicate(timeout=5)
        pid_str = (out or "").strip().splitlines()[-1] if out else None
        pid = int(pid_str) if (pid_str and pid_str.isdigit()) else None
        await asyncio.sleep(1.0)
        status, _ = await _http_json(f"http://localhost:{int(req.port or 3001)}/health")
        ok = status == 200
        return MCPStartResponse(
            started=ok,
            pid=pid,
            port=req.port or 3001,
            log_path=env.get("BLENDER_MCP_LOG_PATH"),
            message=f"HTTP {status}",
        )
    except subprocess.TimeoutExpired:
        return MCPStartResponse(started=False, message="Timeout starting MCP")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to start MCP: {e}")


@app.post("/api/mcp/stop", response_model=MCPStatusResponse)
async def mcp_stop():  # noqa: D401
    pid_file = _mcp_pid_file()
    if not pid_file.exists():
        return MCPStatusResponse(running=False, pid=None, message="No PID file found")
    try:
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            pid_file.unlink()
        return MCPStatusResponse(running=False, pid=None, message="Stopped")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to stop MCP: {e}")


@app.post("/api/blender/ensure", response_model=BlenderEnsureResponse)
async def blender_ensure(req: BlenderEnsureRequest):  # noqa: D401
    suggestions: list[str] = []
    system = platform.system().lower()
    if system == "darwin":
        suggestions = [
            "brew install --cask blender",
            "Or download from https://www.blender.org/download/",
            "export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender",
        ]
    elif system == "linux":
        suggestions = [
            "sudo apt-get update && sudo apt-get install -y blender",
            "or: flatpak install flathub org.blender.Blender",
            "or: snap install blender",
            "export BLENDER_PATH=$(command -v blender)",
        ]

    for cand in _candidate_blender_paths():
        if cand.exists() and cand.is_file():
            rc, out = _run_blender_cmd(cand, ["--version"], timeout_s=10)
            if rc == 0:
                version = (out.splitlines() or [""])[0]
                rc2, _ = _run_blender_cmd(
                    cand,
                    ["--background", "--python-expr", "import bpy;print('ok')"],
                    timeout_s=20,
                )
                headless_ok = rc2 == 0
                if headless_ok and req.set_env:
                    os.environ["BLENDER_PATH"] = cand.as_posix()
                    with contextlib.suppress(Exception):
                        _persist_env_var("BLENDER_PATH", cand.as_posix())
                return BlenderEnsureResponse(
                    ok=headless_ok,
                    path=cand.as_posix(),
                    version=version,
                    headless_ok=headless_ok,
                    message=(
                        "Blender detected and headless probe passed"
                        if headless_ok
                        else "Blender detected but headless probe failed"
                    ),
                    suggestions=[] if headless_ok else suggestions,
                )

    return BlenderEnsureResponse(
        ok=False,
        path=None,
        version=None,
        headless_ok=False,
        message="Blender not found. Configure BLENDER_PATH or install Blender.",
        suggestions=suggestions,
    )


async def _download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r,
        ):
            if r.status != 200:
                return False
            data = await r.read()
            dest.write_bytes(data)
            return True
    except Exception:
        return False


@app.post("/api/blender/install-vrm-addon", response_model=InstallVRMAddonResponse)
async def install_vrm_addon(req: InstallVRMAddonRequest):  # noqa: D401
    """Download (optional) and install + enable the VRM add-on in headless Blender."""
    # Ensure Blender
    ensure = await blender_ensure(BlenderEnsureRequest(set_env=req.set_env))
    if not ensure.ok:
        raise HTTPException(status_code=500, detail=f"Blender not available: {ensure.message}")

    blender_bin = os.getenv("BLENDER_PATH")
    if not blender_bin:
        raise HTTPException(status_code=500, detail="BLENDER_PATH not set")

    tmp_dir = Path(tempfile.gettempdir())
    zip_path: Path
    if req.zip_url:
        zip_path = tmp_dir / "vrm_addon.zip"
        ok = await _download_file(req.zip_url, zip_path)
        if not ok:
            raise HTTPException(status_code=400, detail="Failed to download VRM add-on zip")
    else:
        # Try known candidates for latest release
        candidates = [
            "https://github.com/saturday06/VRM-Addon-for-Blender/releases/latest/download/VRM_Addon_for_Blender-release.zip",
            "https://github.com/saturday06/VRM-Addon-for-Blender/releases/latest/download/VRM_Addon_for_Blender-latest.zip",
        ]
        zip_path = tmp_dir / "vrm_addon.zip"
        downloaded = False
        for u in candidates:
            if await _download_file(u, zip_path):
                downloaded = True
                break
        if not downloaded:
            raise HTTPException(
                status_code=400,
                detail="Unable to download VRM add-on. Provide zip_url.",
            )

    # Install via Blender headless
    install_script = tmp_dir / "install_vrm_addon.py"
    install_script.write_text(
        """
import bpy
import addon_utils
import sys
zip_path = r"""
        + zip_path.as_posix()
        + """\n
try:
    bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
    bpy.ops.preferences.addon_enable(module='io_scene_vrm')
    bpy.ops.wm.save_userpref()
    enabled = addon_utils.check('io_scene_vrm')[0]
    print('VRM_ENABLED:', enabled)
except Exception as e:
    print('VRM_INSTALL_ERROR:', e)
    sys.exit(1)
"""
    )

    rc, out = _run_blender_cmd(
        Path(blender_bin),
        ["--background", "--python", install_script.as_posix()],
        timeout_s=120,
    )

    with contextlib.suppress(Exception):
        install_script.unlink(missing_ok=True)

    if rc != 0:
        return InstallVRMAddonResponse(
            installed=False,
            enabled=False,
            message=f"Install failed (rc={rc})\n{out[:4000]}",
        )

    enabled = "VRM_ENABLED: True" in (out or "")
    return InstallVRMAddonResponse(
        installed=True,
        enabled=enabled,
        message="Installed" if enabled else "Installed but not enabled",
        zip_path=zip_path.as_posix(),
    )


# All session routes have been moved to routers/sessions.py


@app.post("/api/generate")
async def generate_asset(request: GenerateAssetRequest, req: Request):
    """Generate a new 3D asset with comprehensive logging and error handling."""
    request_start_time = time.time()
    client_ip = req.client.host if req.client else "unknown"
    request_id = f"req_{int(time.time())}_{hash(request.prompt) % 10000:04d}"

    logger.info(
        "🎨 Asset generation request received",
        request_id=request_id,
        prompt=request.prompt[:100] + ("..." if len(request.prompt) > 100 else ""),
        prompt_length=len(request.prompt),
        asset_name=request.asset_name,
        export_formats=request.export_formats,
        output_directory=request.output_directory,
        client_ip=client_ip,
        user_agent=req.headers.get("user-agent", "unknown")[:100],
        request_timestamp=request_start_time,
    )

    # Input validation with detailed logging
    validation_errors = []
    if not request.prompt or not request.prompt.strip():
        validation_errors.append("Prompt cannot be empty")
    if len(request.prompt) > 2000:
        validation_errors.append(f"Prompt too long ({len(request.prompt)} chars, max 2000)")
    if not request.asset_name or not request.asset_name.strip():
        validation_errors.append("Asset name cannot be empty")
    if len(request.asset_name) > 100:
        validation_errors.append(f"Asset name too long ({len(request.asset_name)} chars, max 100)")

    if validation_errors:
        logger.error(
            "❌ Asset generation request validation failed",
            request_id=request_id,
            validation_errors=validation_errors,
            client_ip=client_ip,
            request_data={
                "prompt_length": len(request.prompt),
                "asset_name_length": len(request.asset_name),
                "export_formats_count": len(request.export_formats),
            },
        )
        raise HTTPException(status_code=400, detail=f"Validation failed: {', '.join(validation_errors)}")

    try:
        logger.debug("🔍 Checking if full mode is available", request_id=request_id)
        _require_full_mode()
        logger.info("✅ Full mode confirmed - proceeding with generation", request_id=request_id)

        # Start generation with comprehensive monitoring
        generation_start_time = time.time()
        logger.info(
            "🚀 Starting orchestrator asset generation",
            request_id=request_id,
            orchestrator_type=type(orchestrator).__name__,
            generation_parameters={
                "prompt_preview": request.prompt[:50] + "...",
                "asset_name": request.asset_name,
                "export_formats": request.export_formats,
                "output_directory": request.output_directory,
            },
        )

        try:
            result = await orchestrator.generate_asset(
                prompt=request.prompt,
                export_formats=request.export_formats,
                asset_name=request.asset_name,
                output_directory=request.output_directory,
            )
            generation_time = time.time() - generation_start_time

            logger.info(
                "⚡ Orchestrator generation completed",
                request_id=request_id,
                success=result.get("success", False),
                generation_time_s=f"{generation_time:.3f}",
                result_keys=(list(result.keys()) if isinstance(result, dict) else "unknown"),
                cached=result.get("cached", False),
                has_asset_path=bool(result.get("asset_path")),
                has_screenshot=bool(result.get("screenshot_path")),
            )

        except Exception as gen_error:
            generation_time = time.time() - generation_start_time
            logger.exception(
                "💥 Orchestrator generation failed",
                request_id=request_id,
                error=str(gen_error),
                error_type=type(gen_error).__name__,
                generation_time_s=f"{generation_time:.3f}",
                request_summary={
                    "prompt_length": len(request.prompt),
                    "asset_name": request.asset_name,
                    "export_formats": request.export_formats,
                },
            )
            raise

        session_id = result.get("session_id")
        total_request_time = time.time() - request_start_time

        # Enhanced session management and client notification
        if session_id:
            logger.debug(
                "📡 Broadcasting completion to WebSocket clients",
                request_id=request_id,
                session_id=session_id,
                connected_clients=len(manager.active_connections),
            )
            try:
                await manager.broadcast_to_session(session_id, "generation_complete", result)
                logger.debug(
                    "✅ WebSocket broadcast successful",
                    request_id=request_id,
                    session_id=session_id,
                )
            except Exception as broadcast_error:
                logger.warning(
                    "⚠️ Failed to broadcast completion to WebSocket clients",
                    request_id=request_id,
                    session_id=session_id,
                    error=str(broadcast_error),
                )
        else:
            logger.warning(
                "⚠️ No session ID in result, skipping WebSocket broadcast",
                request_id=request_id,
            )

        # Final response preparation with comprehensive logging
        response_data = {
            "message": "Generation completed",
            "result": result,
            "session_id": session_id,
            "request_id": request_id,
            "request_metrics": {
                "total_request_time_s": total_request_time,
                "generation_time_s": result.get("generation_time", 0),
                "success": result.get("success", False),
                "cached": result.get("cached", False),
            },
        }

        logger.info(
            "🎉 Asset generation request completed successfully",
            request_id=request_id,
            session_id=session_id,
            success=result.get("success", False),
            total_request_time_s=f"{total_request_time:.3f}",
            response_size_estimate=len(str(response_data)),
            client_ip=client_ip,
        )

        return response_data

    except HTTPException as e:
        total_request_time = time.time() - request_start_time
        logger.exception(
            "❌ HTTP Exception during generation",
            request_id=request_id,
            status_code=e.status_code,
            detail=e.detail,
            total_request_time_s=f"{total_request_time:.3f}",
            client_ip=client_ip,
        )
        raise
    except Exception as e:
        total_request_time = time.time() - request_start_time
        error_msg = f"Generation failed: {str(e)}"
        logger.exception(
            "💥 Unexpected error during generation",
            request_id=request_id,
            error=error_msg,
            error_type=type(e).__name__,
            total_request_time_s=f"{total_request_time:.3f}",
            request_summary={
                "prompt_length": (len(request.prompt) if hasattr(request, "prompt") else 0),
                "asset_name": getattr(request, "asset_name", "unknown"),
                "export_formats": getattr(request, "export_formats", []),
            },
            client_ip=client_ip,
            stack_trace=True,
        )
        raise HTTPException(status_code=500, detail=error_msg) from e


@app.post("/api/generate-async")
async def generate_asset_async(request: GenerateAssetRequest):
    """Start generation asynchronously and return a session_id immediately.

    The client should poll /api/sessions/{session_id}/progress and /api/sessions/{session_id}
    for granular updates until completion.
    """
    _require_full_mode()

    from src.services.stateful_session_manager import SessionPhase

    # Create a session immediately for the UI to attach to
    state = session_manager.create_session(
        prompt=request.prompt,
        asset_name=request.asset_name,
        output_directory=request.output_directory or "outputs",
    )
    session_id = state.session_id

    async def _runner() -> None:
        """Run the actual LangGraph workflow with interrupt capabilities."""
        try:
            from src.workflow.workflow import create_ll3m_workflow
            
            logger.info(f"Starting LangGraph workflow for session {session_id}")
            
            # Add initial progress message
            session_manager.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.INITIALIZATION,
                title="🚀 Starting Workflow",
                description="Initializing LL3M workflow system...",
                progress_percent=5.0,
            )
            await manager.broadcast_to_session(session_id, "progress_update", {
                "phase": "initialization",
                "progress": 5.0,
                "message": "Starting workflow..."
            })
            
            # Create the LangGraph workflow with interrupt capabilities
            workflow = create_ll3m_workflow()
            
            # Prepare initial state for the workflow (matching WorkflowState structure)
            input_state = {
                "session_id": session_id,
                "prompt": request.prompt,
                "asset_name": request.asset_name,
                "output_directory": request.output_directory or "outputs",
                "export_formats": request.export_formats,
                
                # Request reformulation fields
                "reformulation_data": None,
                "reformulated_prompt": "",
                "awaiting_user_confirmation": False,
                "user_confirmed_reformulation": False,
                
                # Other required fields
                "subtasks": [],
                "documentation": "",
                "generated_code": "",
                "current_phase": "initialization",
                "completion_percentage": 0.0,
                "step_control": {"can_continue": True, "current_step": "start"},
            }
            
            # Configure workflow with session thread_id for interrupt/resume capability
            config = {"configurable": {"thread_id": session_id}}
            
            logger.info(f"Invoking workflow for session {session_id} with interrupt support")
            
            # Add progress message before starting reformulation
            session_manager.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.REQUEST_REFORMULATION,
                title="🔄 Understanding Request",
                description="Analyzing and reformulating your request for clarity...",
                progress_percent=10.0,
            )
            await manager.broadcast_to_session(session_id, "progress_update", {
                "phase": "request_reformulation", 
                "progress": 10.0,
                "message": "Understanding your request..."
            })
            
            # Invoke the workflow - this will run until interrupt or completion
            result = await workflow.ainvoke(input_state, config=config)
            
            logger.info(f"Workflow execution result: {result}")
            
            # Check if we hit the user confirmation interrupt
            if result and result.get("awaiting_user_confirmation"):
                logger.info(f"Workflow paused at user confirmation for session {session_id}")
                
                # Update progress to show reformulation is ready for user review
                reformulation_data = result.get("reformulation_data", {})
                session_manager.add_progress_message(
                    session_id=session_id,
                    phase=SessionPhase.REQUEST_REFORMULATION,
                    title="✏️ Reformulated Request Ready",
                    description="Please review the reformulated request and choose an action.",
                    progress_percent=15.0,
                    technical_details={
                        "reformulated_request": reformulation_data.get("reformulated_request", ""),
                        "improvements": reformulation_data.get("improvements", ""),
                        "questions": reformulation_data.get("questions", ""),
                        "summary": reformulation_data.get("summary"),
                        "tags": reformulation_data.get("tags"),
                        "user_confirmation_required": reformulation_data.get("needs_confirmation", True),
                        "original_prompt": request.prompt,
                    },
                )
                
                # Broadcast reformulation data to frontend
                await manager.broadcast_to_session(session_id, "reformulation_ready", {
                    "phase": "request_reformulation",
                    "progress": 15.0,
                    "reformulated_request": reformulation_data.get("reformulated_request", ""),
                    "improvements": reformulation_data.get("improvements", ""),
                    "questions": reformulation_data.get("questions", ""),
                    "summary": reformulation_data.get("summary"),
                    "tags": reformulation_data.get("tags"),
                    "original_prompt": request.prompt,
                })
                
                logger.info(f"Waiting for user confirmation for session {session_id}")
            else:
                # Workflow completed fully
                logger.info(f"Workflow execution completed for session {session_id}")
                session_manager.add_progress_message(
                    session_id=session_id,
                    phase=SessionPhase.COMPLETION,
                    title="✅ Workflow Complete",
                    description="Asset generation workflow completed successfully.",
                    progress_percent=100.0,
                )
                await manager.broadcast_to_session(session_id, "generation_complete", {"result": result})
            
        except Exception as e:
            logger.error(f"Workflow execution failed for session {session_id}: {e}")
            session_manager.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.ERROR,
                title="⚠️ Workflow Error",
                description=str(e),
                progress_percent=100.0,
            )
            await manager.broadcast_to_session(session_id, "generation_failed", {"error": str(e)})

    asyncio.create_task(_runner())

    return {"accepted": True, "session_id": session_id}


# All session management endpoints moved to routers/sessions.py


## Asset file serving moved to routers/assets.py


# All progress tracking and interaction endpoints moved to routers/sessions.py


# All asset management endpoints moved to routers/assets.py


async def _generate_asset_preview(asset_path: Path, asset_id: str) -> Path | None:
    """Generate a preview image for the asset using Blender."""
    try:
        if STANDALONE:
            return None  # No preview generation in standalone mode

        # Get file extension to determine import method
        file_extension = asset_path.suffix.lower()

        # Create preview output path
        preview_dir = asset_path.parent / "previews"
        preview_dir.mkdir(exist_ok=True, parents=True)
        preview_path = preview_dir / f"{asset_id}_preview.png"

        # Generate Blender script for importing and rendering preview
        _generate_import_script(asset_path, file_extension, preview_path)

        # Execute Blender script (this would use the existing Blender service)
        # For now, return the expected preview path
        # TODO: Integrate with Blender service when available

        return preview_path if preview_path.exists() else None

    except Exception as e:
        logger.warning(f"Failed to generate preview for asset {asset_id}: {e}")
        return None


def _generate_import_script(asset_path: Path, file_extension: str, preview_path: Path) -> str:
    """Generate Blender Python script for importing assets with VRM support."""
    base_script = f"""
import bpy
import bmesh
import os

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Set up scene for preview
bpy.context.scene.render.filepath = "{preview_path}"
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.image_settings.file_format = 'PNG'

try:
"""

    # Add import logic based on file extension
    if file_extension == ".vrm":
        import_script = f"""
    # Import VRM file using VRM-Addon-for-Blender
    # Check if VRM addon is enabled
    import addon_utils
    vrm_addon = addon_utils.check("io_scene_vrm")

    if vrm_addon[0]:  # VRM addon is available
        print("VRM addon found, importing VRM file...")
        # Import VRM file
        bpy.ops.import_scene.vrm(filepath="{asset_path}")
        print("VRM file imported successfully")
    else:
        print("VRM addon not available, trying GLB import as fallback...")
        # VRM files are based on GLB, try GLB import as fallback
        bpy.ops.import_scene.gltf(filepath="{asset_path}")
        print("File imported as GLB/GLTF")
"""
    elif file_extension in [".gltf", ".glb"]:
        import_script = f"""
    # Import GLTF/GLB file
    bpy.ops.import_scene.gltf(filepath="{asset_path}")
    print("GLTF/GLB file imported successfully")
"""
    elif file_extension == ".obj":
        import_script = f"""
    # Import OBJ file
    bpy.ops.wm.obj_import(filepath="{asset_path}")
    print("OBJ file imported successfully")
"""
    elif file_extension == ".fbx":
        import_script = f"""
    # Import FBX file
    bpy.ops.import_scene.fbx(filepath="{asset_path}")
    print("FBX file imported successfully")
"""
    elif file_extension == ".dae":
        import_script = f"""
    # Import Collada file
    bpy.ops.wm.collada_import(filepath="{asset_path}")
    print("Collada file imported successfully")
"""
    elif file_extension == ".blend":
        import_script = f"""
    # Import Blender file
    with bpy.data.libraries.load("{asset_path}") as (data_from, data_to):
        data_to.objects = data_from.objects

    # Link objects to scene
    for obj in data_to.objects:
        if obj:
            bpy.context.collection.objects.link(obj)
    print("Blender file imported successfully")
"""
    else:  # Default fallback for other formats
        import_script = f"""
    # Try to import using available importers
    print("Attempting to import file with available importers...")
    try:
        bpy.ops.import_scene.gltf(filepath="{asset_path}")
    except:
        try:
            bpy.ops.wm.obj_import(filepath="{asset_path}")
        except:
            print("Could not import file with available importers")
"""

    post_import_script = """
    # Center and frame the imported objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if mesh_objects:
        # Calculate bounding box
        min_co = [float('inf')] * 3
        max_co = [float('-inf')] * 3

        for obj in mesh_objects:
            for vertex in obj.bound_box:
                world_vertex = obj.matrix_world @ obj.matrix_world @ obj.matrix_world.to_3x3() @ vertex
                for i in range(3):
                    min_co[i] = min(min_co[i], world_vertex[i])
                    max_co[i] = max(max_co[i], world_vertex[i])

        # Calculate center and size
        center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
        size = max(max_co[i] - min_co[i] for i in range(3))

        # Position camera
        camera_distance = size * 2.5
        bpy.ops.object.camera_add(location=(camera_distance, -camera_distance, camera_distance * 0.5))
        camera = bpy.context.active_object

        # Point camera at center
        constraint = camera.constraints.new(type='TRACK_TO')

        # Create empty at center for tracking
        bpy.ops.object.empty_add(location=center)
        target = bpy.context.active_object
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        # Set camera as active camera
        bpy.context.scene.camera = camera

        # Add lighting
        bpy.ops.object.light_add(type='SUN', location=(camera_distance, camera_distance, camera_distance))
        sun = bpy.context.active_object
        sun.data.energy = 5.0

        # Set up material preview shading
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break

        # Render the preview
        bpy.ops.render.render(write_still=True)
        print(f"Preview rendered successfully: {preview_path}")
    else:
        print("No mesh objects found to render")

except Exception as e:
    print(f"Error during import/render: {{e}}")
    import traceback
    traceback.print_exc()
"""

    return base_script + import_script + post_import_script


def _generate_export_script(asset_path: Path, file_extension: str, export_format: str, export_path: Path) -> str:
    """Generate Blender Python script to import an asset and export to target format.

    Supports VRM/VRMA when io_scene_vrm addon is available.
    """
    # We can reuse the import logic but without preview/camera setup
    script = f"""
import bpy
import addon_utils
import sys
try:
    # Enable VRM addon if present for VRM/VRMA flows
    try:
        addon_utils.enable("io_scene_vrm", default_set=True, persistent=True)
    except Exception:
        pass

    # Import the source asset
    ext = "{file_extension}"
    path = r"{asset_path}"
    if ext == '.vrm':
        if addon_utils.check("io_scene_vrm")[0]:
            bpy.ops.import_scene.vrm(filepath=path)
        else:
            bpy.ops.import_scene.gltf(filepath=path)
    elif ext in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == '.dae':
        bpy.ops.wm.collada_import(filepath=path)
    elif ext == '.blend':
        with bpy.data.libraries.load(path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj:
                bpy.context.collection.objects.link(obj)

    # Select all for some exporters
    bpy.ops.object.select_all(action='SELECT')

    out_path = r"{export_path}"
    fmt = "{export_format}"

    if fmt == 'vrm':
        if hasattr(bpy.ops.export_scene, 'vrm'):
            bpy.ops.export_scene.vrm(filepath=out_path)
        else:
            print('VRM export not available: io_scene_vrm exporter missing')
            sys.exit(2)
    elif fmt == 'vrma':
        # VRMA operator may be available in newer VRM addons
        if hasattr(bpy.ops.export_scene, 'vrma'):
            bpy.ops.export_scene.vrma(filepath=out_path)
        else:
            print('VRMA export not available: exporter missing')
            sys.exit(3)
    elif fmt == 'gltf':
        bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLTF_SEPARATE')
    elif fmt == 'glb':
        bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB')
    elif fmt == 'fbx':
        bpy.ops.export_scene.fbx(filepath=out_path)
    elif fmt == 'obj':
        bpy.ops.wm.obj_export(filepath=out_path)
    else:
        print('Unsupported export format:', fmt)
        sys.exit(4)
    print('EXPORT_OK:', out_path)
except SystemExit:
    raise
except Exception as e:
    print('Export error:', e)
    sys.exit(1)
"""
    return script


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates."""
    await manager.connect(websocket, session_id)

    try:
        # Send initial session state
        session_state = session_manager.load_session(session_id)
        if session_state:
            await manager.send_message(
                session_id,
                {
                    "type": "session_loaded",
                    "data": {
                        "session_id": session_state.session_id,
                        "current_phase": session_state.current_phase.value,
                        "completion_percentage": session_state.completion_percentage,
                        "success": session_state.success,
                    },
                },
            )

        # Keep connection alive and handle messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)

                # Handle different message types
                if data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong"})
                elif data.get("type") == "subscribe_updates":
                    # Client wants real-time updates for this session
                    await manager.send_message(
                        session_id,
                        {"type": "subscribed", "data": {"session_id": session_id}},
                    )

            except json.JSONDecodeError:
                await manager.send_message(session_id, {"type": "error", "data": {"message": "Invalid JSON"}})
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        await manager.send_message(session_id, {"type": "error", "data": {"message": str(e)}})
        manager.disconnect(session_id)


# Mount static files for serving frontend build
backend_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(backend_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":

    def signal_handler(sig, _frame):  # noqa: ARG001
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {sig}. Shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🌐 Starting FastAPI server...")
    print("🔗 Backend URL: http://localhost:8000")
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("🔗 Health Check: http://localhost:8000/health")
    print("🔗 Diagnostics: http://localhost:8000/api/diagnostics")
    print(f"📊 Mode: {'STANDALONE (limited functionality)' if STANDALONE else 'FULL (all features)'}")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # noqa: S104
            port=8000,
            reload=True,
            log_level="info",
            access_log=False,  # Reduce noise
        )
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        print("👋 Server stopped")
