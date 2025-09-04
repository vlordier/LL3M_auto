"""Batch processing routes."""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..auth import AuthUser, require_asset_batch
from ..models import BatchRequest, BatchResponse

router = APIRouter(prefix="/api/v1/batches", tags=["Batch Processing"])

# In-memory storage for demo (would use database in production)
active_batches: dict[UUID, dict] = {}


@router.post("", response_model=BatchResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthUser = Depends(require_asset_batch),
) -> BatchResponse:
    """Create a batch processing job for multiple assets."""
    batch_id = uuid4()

    # Estimate completion time based on number of assets and priority
    base_time_per_asset = 300  # 5 minutes per asset
    priority_multiplier = 1.0 / request.priority  # Higher priority = faster
    estimated_time = base_time_per_asset * len(request.requests) * priority_multiplier
    estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_time)

    # Store batch information
    batch_info = {
        "id": batch_id,
        "user_id": current_user.id,
        "name": request.name,
        "status": "queued",
        "total_assets": len(request.requests),
        "completed_assets": 0,
        "failed_assets": 0,
        "priority": request.priority,
        "requests": request.requests,
        "notify_on_completion": request.notify_on_completion,
        "created_at": datetime.utcnow(),
        "estimated_completion": estimated_completion,
    }

    active_batches[batch_id] = batch_info

    # Queue batch processing
    background_tasks.add_task(_process_batch_background, batch_id, current_user.id)

    return BatchResponse(
        batch_id=batch_id,
        total_assets=len(request.requests),
        estimated_completion=estimated_completion,
        status="queued",
    )


@router.get("/{batch_id}")
async def get_batch_status(
    batch_id: UUID, current_user: AuthUser = Depends(require_asset_batch)
) -> dict[str, Any]:
    """Get status of a batch processing job."""
    if batch_id not in active_batches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found"
        )

    batch_info = active_batches[batch_id]

    # Check ownership
    if batch_info["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return {
        "batch_id": batch_id,
        "name": batch_info["name"],
        "status": batch_info["status"],
        "total_assets": batch_info["total_assets"],
        "completed_assets": batch_info["completed_assets"],
        "failed_assets": batch_info["failed_assets"],
        "progress_percent": (
            batch_info["completed_assets"] / batch_info["total_assets"]
        )
        * 100,
        "created_at": batch_info["created_at"],
        "estimated_completion": batch_info.get("estimated_completion"),
        "completed_at": batch_info.get("completed_at"),
    }


@router.get("")
async def list_batches(
    current_user: AuthUser = Depends(require_asset_batch),
) -> list[dict[str, Any]]:
    """List user's batch processing jobs."""
    user_batches = []

    for batch_id, batch_info in active_batches.items():
        if batch_info["user_id"] == current_user.id:
            user_batches.append(
                {
                    "batch_id": batch_id,
                    "name": batch_info["name"],
                    "status": batch_info["status"],
                    "total_assets": batch_info["total_assets"],
                    "completed_assets": batch_info["completed_assets"],
                    "failed_assets": batch_info["failed_assets"],
                    "created_at": batch_info["created_at"],
                    "estimated_completion": batch_info.get("estimated_completion"),
                }
            )

    # Sort by creation time (newest first)
    user_batches.sort(key=lambda x: x["created_at"], reverse=True)

    return user_batches


@router.delete("/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_batch(
    batch_id: UUID, current_user: AuthUser = Depends(require_asset_batch)
) -> None:
    """Cancel a batch processing job."""
    if batch_id not in active_batches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found"
        )

    batch_info = active_batches[batch_id]

    # Check ownership
    if batch_info["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Only allow cancellation if not completed
    if batch_info["status"] in ["completed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel completed or already cancelled batch",
        )

    # Update status to cancelled
    batch_info["status"] = "cancelled"
    batch_info["completed_at"] = datetime.utcnow()


@router.get("/{batch_id}/assets")
async def get_batch_assets(
    batch_id: UUID, current_user: AuthUser = Depends(require_asset_batch)
) -> dict[str, Any]:
    """Get assets generated by a batch job."""
    if batch_id not in active_batches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found"
        )

    batch_info = active_batches[batch_id]

    # Check ownership
    if batch_info["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Return generated assets (would query from database in production)
    generated_assets = batch_info.get("generated_assets", [])

    return {
        "batch_id": batch_id,
        "assets": generated_assets,
        "total_assets": len(generated_assets),
    }


async def _process_batch_background(batch_id: UUID, user_id: UUID) -> None:
    """Background task for batch processing."""
    import asyncio

    if batch_id not in active_batches:
        return

    batch_info = active_batches[batch_id]

    try:
        # Update status to processing
        batch_info["status"] = "processing"
        batch_info["started_at"] = datetime.utcnow()

        generated_assets = []

        # Process each asset request
        for i, asset_request in enumerate(batch_info["requests"]):
            # Check if batch was cancelled
            if batch_info["status"] == "cancelled":
                break

            try:
                # Simulate asset generation (would use real workflow in production)
                await asyncio.sleep(2)  # Simulate processing time

                # Create mock asset
                asset_id = uuid4()
                asset = {
                    "id": str(asset_id),
                    "name": f"Batch asset {i + 1}: {asset_request.prompt[:30]}...",
                    "prompt": asset_request.prompt,
                    "status": "completed",
                    "blender_file_url": f"https://storage.ll3m.com/batches/{batch_id}/assets/{asset_id}/asset.blend",
                    "preview_image_url": f"https://storage.ll3m.com/batches/{batch_id}/assets/{asset_id}/preview.png",
                    "created_at": datetime.utcnow().isoformat(),
                }

                generated_assets.append(asset)
                batch_info["completed_assets"] += 1

            except Exception as e:
                # Handle individual asset failure
                batch_info["failed_assets"] += 1
                print(f"Failed to process asset {i + 1} in batch {batch_id}: {e}")

        # Update batch completion
        batch_info["generated_assets"] = generated_assets

        if batch_info["status"] != "cancelled":
            if batch_info["failed_assets"] == 0:
                batch_info["status"] = "completed"
            elif batch_info["completed_assets"] > 0:
                batch_info["status"] = "completed_with_errors"
            else:
                batch_info["status"] = "failed"

        batch_info["completed_at"] = datetime.utcnow()

        # Send notification if requested
        if batch_info["notify_on_completion"]:
            await _send_batch_completion_notification(batch_id, user_id)

    except Exception as e:
        # Handle batch-level failure
        batch_info["status"] = "failed"
        batch_info["error"] = str(e)
        batch_info["completed_at"] = datetime.utcnow()


async def _send_batch_completion_notification(batch_id: UUID, user_id: UUID) -> None:
    """Send notification when batch processing is completed."""
    # In production, this would send email, push notification, or webhook
    print(f"Batch {batch_id} completed for user {user_id}")

    # Mock notification
    batch_info = active_batches.get(batch_id)
    if batch_info:
        print(f"Batch '{batch_info['name']}' completed:")
        print(f"  - Total assets: {batch_info['total_assets']}")
        print(f"  - Completed: {batch_info['completed_assets']}")
        print(f"  - Failed: {batch_info['failed_assets']}")
        print(f"  - Status: {batch_info['status']}")
