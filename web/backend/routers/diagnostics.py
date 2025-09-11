from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..utils import http_json as _http_json
from ..utils import env_bool as _env_bool
from ..utils import candidate_blender_paths as _candidate_blender_paths
from ..utils import run_blender_cmd as _run_blender_cmd


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


router = APIRouter()


@router.get("/api/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(full: bool = False, checks: str | None = None):  # noqa: D401
    services: dict[str, ServiceCheck] = {}

    # Backend API self-check
    services["web_backend"] = ServiceCheck(name="Web Backend", ok=True, status="ok")

    # Production API (src.api) health (opt-in)
    requested = set((checks or "").split(",")) if checks else set()
    standalone = os.getenv("WEB_BACKEND_STANDALONE", "false").lower() in {"1", "true", "yes", "on"}
    if standalone:
        services["prod_api"] = ServiceCheck(name="Production API", ok=True, status="skipped", message="Standalone mode")
    elif full or ("prod" in requested or "prod_api" in requested):
        status, _ = await _http_json("http://localhost:8000/api/v1/health")
        services["prod_api"] = ServiceCheck(
            name="Production API", ok=(status == 200), status=("ok" if status == 200 else "unreachable"), message=f"HTTP {status}"
        )

    # OpenAI (opt-in)
    if _env_bool("USE_LOCAL_LLM", "false"):
        services["openai"] = ServiceCheck(name="OpenAI", ok=True, status="skipped", message="USE_LOCAL_LLM=true")
    elif full or ("openai" in requested):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            services["openai"] = ServiceCheck(name="OpenAI", ok=False, status="missing_key", message="OPENAI_API_KEY not set")
        else:
            s, _ = await _http_json("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout_s=8)
            services["openai"] = ServiceCheck(name="OpenAI", ok=(s in (200, 401)), status=("ok" if s in (200, 401) else "unreachable"), message=f"HTTP {s}")

    # LM Studio (opt-in)
    if _env_bool("USE_LOCAL_LLM", "false") and (full or ("lm" in requested or "lmstudio" in requested)):
        base = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        s, _ = await _http_json(f"{base}/models")
        services["lmstudio"] = ServiceCheck(name="LM Studio", ok=(s == 200), status=("ok" if s == 200 else "unreachable"), message=f"HTTP {s}")
    else:
        services["lmstudio"] = ServiceCheck(name="LM Studio", ok=True, status="skipped", message="USE_LOCAL_LLM=false")

    # Blender (ensure runnable)
    blender_ok = False
    blender_msg = "not found"
    headless_ok = False
    version = None
    for cand in _candidate_blender_paths():
        if cand.exists() and cand.is_file():
            rc, out = _run_blender_cmd(cand, ["--version"], timeout_s=10)
            if rc == 0:
                version = (out.splitlines() or [""])[0]
                rc2, _ = _run_blender_cmd(cand, ["--background", "--python-expr", "import bpy;print('ok')"], timeout_s=20)
                headless_ok = rc2 == 0
                blender_ok = headless_ok
                blender_msg = f"{cand} | {version} | headless: {'ok' if headless_ok else 'fail'}"
                os.environ.setdefault("BLENDER_PATH", cand.as_posix())
                break
    services["blender"] = ServiceCheck(name="Blender", ok=blender_ok, status=("ok" if blender_ok else "unavailable"), message=blender_msg)

    ok = all(s.ok or s.status == "skipped" for s in services.values())
    return DiagnosticsResponse(ok=ok, timestamp=datetime.now(UTC).isoformat(), services=services)
