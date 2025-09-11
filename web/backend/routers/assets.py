from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


def _repo_root() -> Path:
    """Find repository root by searching for a pyproject.toml upwards."""
    cur = Path(__file__).resolve()
    for candidate in [cur] + list(cur.parents):
        marker = candidate / "pyproject.toml"
        if marker.exists():
            return candidate
    return Path.cwd()


class AssetMetadataResponse(BaseModel):  # minimal shape
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


class ExportRequest(BaseModel):  # noqa: D101
    format: str
    options: dict[str, Any] | None = None


def _imported_assets_dir() -> Path:
    return Path(__file__).parent.parent.parent / "outputs" / "imported_assets"


@router.post("/api/assets/import", response_model=AssetMetadataResponse)
async def import_asset(
    file: UploadFile = File(...),
    asset_name: str = Form(...),
    asset_type: str = Form(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),
):
    try:
        allowed = {".gltf", ".glb", ".vrm", ".obj", ".blend", ".fbx", ".dae", ".3ds", ".ply", ".stl"}
        ext = Path(file.filename or "").suffix.lower()
        if ext not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        import_dir = _imported_assets_dir()
        import_dir.mkdir(parents=True, exist_ok=True)
        asset_id = str(uuid.uuid4())
        file_path = import_dir / f"{asset_id}_{file.filename}"
        content = await file.read()
        file_path.write_bytes(content)
        parsed_tags: list[str] = []
        if tags:
            try:
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                parsed_tags = [t.strip() for t in tags.split(",")]
        metadata = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "file_path": str(file_path),
            "preview_path": None,
            "size_bytes": len(content),
            "created_at": datetime.now(UTC).isoformat(),
            "description": description,
            "tags": parsed_tags,
            "import_metadata": {
                "original_filename": file.filename,
                "file_extension": ext,
                "content_type": file.content_type,
                "imported_at": datetime.now(UTC).isoformat(),
            },
        }
        (import_dir / f"{asset_id}_metadata.json").write_text(json.dumps(metadata, indent=2))
        return AssetMetadataResponse(**metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import asset: {e}")


@router.post("/api/assets/import-path", response_model=AssetMetadataResponse)
async def import_asset_from_path(req: ImportAssetFromPathRequest):
    try:
        src = Path(req.file_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail="File not found")
        import_dir = _imported_assets_dir()
        import_dir.mkdir(parents=True, exist_ok=True)
        asset_id = str(uuid.uuid4())
        dest = import_dir / f"{asset_id}_{src.name}"
        dest.write_bytes(src.read_bytes())
        metadata = {
            "asset_id": asset_id,
            "asset_name": req.asset_name,
            "asset_type": req.asset_type,
            "file_path": str(dest),
            "preview_path": None,
            "size_bytes": dest.stat().st_size,
            "created_at": datetime.now(UTC).isoformat(),
            "description": req.description,
            "tags": req.tags or [],
            "import_metadata": {
                "original_filename": src.name,
                "file_extension": src.suffix.lower(),
                "content_type": None,
                "imported_at": datetime.now(UTC).isoformat(),
            },
        }
        (import_dir / f"{asset_id}_metadata.json").write_text(json.dumps(metadata, indent=2))
        return AssetMetadataResponse(**metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import asset from path: {e}")


@router.post("/api/assets/import-url", response_model=AssetMetadataResponse)
async def import_asset_from_url(req: ImportAssetFromUrlRequest):
    try:
        import_dir = _imported_assets_dir()
        import_dir.mkdir(parents=True, exist_ok=True)
        asset_id = str(uuid.uuid4())
        filename = f"{asset_id}_{Path(req.url).name or 'download'}"
        dest = import_dir / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(req.url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    raise HTTPException(status_code=400, detail=f"Download failed: HTTP {r.status}")
                dest.write_bytes(await r.read())

        metadata = {
            "asset_id": asset_id,
            "asset_name": req.asset_name,
            "asset_type": req.asset_type,
            "file_path": str(dest),
            "preview_path": None,
            "size_bytes": dest.stat().st_size,
            "created_at": datetime.now(UTC).isoformat(),
            "description": req.description,
            "tags": req.tags or [],
            "import_metadata": {
                "original_url": req.url,
                "imported_at": datetime.now(UTC).isoformat(),
            },
        }
        (import_dir / f"{asset_id}_metadata.json").write_text(json.dumps(metadata, indent=2))
        return AssetMetadataResponse(**metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import asset from URL: {e}")


@router.get("/api/assets", response_model=list[AssetMetadataResponse])
async def list_imported_assets(asset_type: str | None = None, limit: int = 50):
    try:
        import_dir = _imported_assets_dir()
        assets: list[AssetMetadataResponse] = []
        if not import_dir.exists():
            return []
        for meta_file in import_dir.glob("*_metadata.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                if asset_type and metadata.get("asset_type") != asset_type:
                    continue
                assets.append(AssetMetadataResponse(**metadata))
            except Exception:
                continue
        # newest first
        assets.sort(key=lambda a: a.created_at, reverse=True)
        return assets[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list assets: {e}")


@router.get("/api/assets/{file_path:path}")
async def serve_asset(file_path: str, req: Request):  # noqa: D401
    """Serve files from the repository outputs directory.

    Placed before the `/api/assets/{asset_id}` metadata route to ensure static
    file paths like `test.png` are correctly handled.
    """
    request_start_time = time.time()
    client_ip = req.client.host if req.client else "unknown"

    try:
        if not file_path or not file_path.strip():
            raise HTTPException(status_code=400, detail="File path cannot be empty")
        if len(file_path) > 500:
            raise HTTPException(status_code=400, detail="File path too long")

        base_path = _repo_root() / "outputs"
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


@router.get("/api/assets/{asset_id}", response_model=AssetMetadataResponse)
async def get_imported_asset(asset_id: str):
    import_dir = _imported_assets_dir()
    meta = import_dir / f"{asset_id}_metadata.json"
    if not meta.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    try:
        return AssetMetadataResponse(**json.loads(meta.read_text()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid metadata: {e}")


@router.post("/api/assets/{asset_id}/edit")
async def edit_asset_stub(asset_id: str):
    raise HTTPException(status_code=501, detail="Edit API not available in simplified build")


@router.post("/api/assets/{asset_id}/export")
async def export_asset_stub(asset_id: str, request: ExportRequest):  # noqa: ARG001
    raise HTTPException(status_code=501, detail="Export API not available in simplified build; use CLI --force")


@router.delete("/api/assets/{asset_id}")
async def delete_imported_asset(asset_id: str):
    try:
        import_dir = _imported_assets_dir()
        files_deleted: list[str] = []
        if import_dir.exists():
            for p in import_dir.glob(f"{asset_id}_*"):
                try:
                    p.unlink()
                    files_deleted.append(p.name)
                except Exception:
                    pass
        if not files_deleted:
            raise HTTPException(status_code=404, detail="Asset not found")
        return {"success": True, "message": f"Asset {asset_id} deleted", "files_deleted": files_deleted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete asset: {e}")


# (serve_asset defined above)
