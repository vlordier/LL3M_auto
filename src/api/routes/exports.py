"""Asset export routes."""

import asyncio
from datetime import datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from ..auth import AuthUser, get_current_user
from ..database import AssetRepository, get_asset_repo
from ..models import ExportFormat, ExportRequest, ExportResponse

router = APIRouter(prefix="/api/v1/exports", tags=["Asset Export"])

# In-memory storage for export jobs (would use database in production)
export_jobs: dict[UUID, dict] = {}


@router.post(
    "/{asset_id}/export",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def export_asset(
    asset_id: UUID,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Export an asset in the specified format."""
    # Verify asset exists and user has access
    db_asset = await asset_repo.get_asset_by_id(asset_id)

    if not db_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found"
        )

    if db_asset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    if db_asset.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asset must be completed before export",
        )

    # Create export job
    export_id = uuid4()
    export_job = {
        "id": export_id,
        "asset_id": asset_id,
        "user_id": current_user.id,
        "format": request.format,
        "quality": request.quality,
        "optimize_for": request.optimize_for,
        "include_materials": request.include_materials,
        "include_textures": request.include_textures,
        "custom_settings": request.custom_settings or {},
        "status": "processing",
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow()
        + timedelta(hours=24),  # Download expires in 24 hours
    }

    export_jobs[export_id] = export_job

    # Queue export processing
    background_tasks.add_task(_process_export_background, export_id)

    # Generate download URL (would be actual storage URL in production)
    download_url = f"/api/v1/exports/{export_id}/download"

    return ExportResponse(
        export_id=export_id,
        download_url=download_url,
        format=request.format,
        file_size=0,  # Will be updated after processing
        expires_at=cast(datetime, export_job["expires_at"]),
    )


@router.get("/{export_id}/status")
async def get_export_status(
    export_id: UUID, current_user: AuthUser = Depends(get_current_user)
):
    """Get export job status."""
    if export_id not in export_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found"
        )

    export_job = export_jobs[export_id]

    # Check ownership
    if export_job["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return {
        "export_id": export_id,
        "status": export_job["status"],
        "format": export_job["format"],
        "quality": export_job["quality"],
        "file_size": export_job.get("file_size", 0),
        "progress": export_job.get("progress", 0),
        "error_message": export_job.get("error_message"),
        "download_url": export_job.get("download_url"),
        "expires_at": export_job["expires_at"],
        "created_at": export_job["created_at"],
    }


@router.get("/{export_id}/download")
async def download_export(
    export_id: UUID, current_user: AuthUser = Depends(get_current_user)
):
    """Download exported asset file."""
    if export_id not in export_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found"
        )

    export_job = export_jobs[export_id]

    # Check ownership
    if export_job["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Check if export is completed
    if export_job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export is {export_job['status']}, cannot download",
        )

    # Check if download has expired
    if datetime.utcnow() > export_job["expires_at"]:
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="Download link has expired"
        )

    # In production, this would redirect to the actual file storage URL
    # For now, return a mock redirect
    storage_url = export_job.get(
        "file_url",
        f"https://storage.ll3m.com/exports/{export_id}/asset.{export_job['format'].value}",
    )

    return RedirectResponse(url=storage_url, status_code=status.HTTP_302_FOUND)


@router.get("")
async def list_exports(
    asset_id: UUID | None = None, current_user: AuthUser = Depends(get_current_user)
):
    """List user's export jobs, optionally filtered by asset."""
    user_exports = []

    for export_id, export_job in export_jobs.items():
        if export_job["user_id"] == current_user.id:
            # Filter by asset if specified
            if asset_id and export_job["asset_id"] != asset_id:
                continue

            user_exports.append(
                {
                    "export_id": export_id,
                    "asset_id": export_job["asset_id"],
                    "format": export_job["format"],
                    "quality": export_job["quality"],
                    "status": export_job["status"],
                    "file_size": export_job.get("file_size", 0),
                    "download_url": export_job.get("download_url"),
                    "expires_at": export_job["expires_at"],
                    "created_at": export_job["created_at"],
                }
            )

    # Sort by creation time (newest first)
    user_exports.sort(key=lambda x: x["created_at"], reverse=True)

    return user_exports


@router.delete("/{export_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_export(
    export_id: UUID, current_user: AuthUser = Depends(get_current_user)
):
    """Delete an export job and its associated file."""
    if export_id not in export_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found"
        )

    export_job = export_jobs[export_id]

    # Check ownership
    if export_job["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Delete export job and file (would delete from storage in production)
    del export_jobs[export_id]


async def _process_export_background(export_id: UUID):
    """Background task for processing asset export."""
    if export_id not in export_jobs:
        return

    export_job = export_jobs[export_id]

    try:
        # Simulate export processing
        export_job["progress"] = 0

        # Step 1: Load asset
        await asyncio.sleep(1)
        export_job["progress"] = 25

        # Step 2: Convert to target format
        await asyncio.sleep(2)
        export_job["progress"] = 60

        # Step 3: Apply optimizations
        if export_job["optimize_for"]:
            await asyncio.sleep(1)
        export_job["progress"] = 80

        # Step 4: Finalize and upload to storage
        await asyncio.sleep(1)
        export_job["progress"] = 100

        # Generate file info based on format
        format_extensions = {
            ExportFormat.GLTF: "gltf",
            ExportFormat.OBJ: "obj",
            ExportFormat.FBX: "fbx",
            ExportFormat.BLEND: "blend",
            ExportFormat.USDZ: "usdz",
        }

        file_extension = format_extensions.get(export_job["format"], "unknown")

        # Mock file size based on format and quality
        base_size = 1024 * 1024  # 1MB base
        quality_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}[
            export_job["quality"]
        ]
        format_multiplier = {
            "gltf": 1.2,
            "obj": 0.8,
            "fbx": 1.5,
            "blend": 2.0,
            "usdz": 1.3,
        }[export_job["format"].value]

        estimated_size = int(base_size * quality_multiplier * format_multiplier)

        # Update job with completion info
        export_job["status"] = "completed"
        export_job["file_size"] = estimated_size
        export_job[
            "file_url"
        ] = f"https://storage.ll3m.com/exports/{export_id}/asset.{file_extension}"
        export_job["download_url"] = f"/api/v1/exports/{export_id}/download"
        export_job["completed_at"] = datetime.utcnow()

    except Exception as e:
        # Handle export failure
        export_job["status"] = "failed"
        export_job["error_message"] = str(e)
        export_job["completed_at"] = datetime.utcnow()
