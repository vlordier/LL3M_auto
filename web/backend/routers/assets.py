from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/api/assets/{file_path:path}")
async def serve_asset(file_path: str, req: Request):  # noqa: D401
    request_start_time = time.time()
    client_ip = req.client.host if req.client else "unknown"

    try:
        if not file_path or not file_path.strip():
            raise HTTPException(status_code=400, detail="File path cannot be empty")
        if len(file_path) > 500:
            raise HTTPException(status_code=400, detail="File path too long")

        base_path = Path(__file__).parent.parent.parent / "outputs"
        full_path = base_path / file_path
        if not str(full_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Asset not found")

        media_type, _ = mimetypes.guess_type(full_path.as_posix())
        return FileResponse(path=str(full_path), media_type=media_type or None, filename=full_path.name)
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to serve asset: {e}")

