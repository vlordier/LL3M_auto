"""Asset management routes."""

import asyncio
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from langgraph.graph import StateGraph

from ..auth import AuthUser, get_current_user, require_asset_create
from ..database import AssetRepository, get_asset_repo
from ..models import (
    Asset,
    AssetResponse,
    AssetStatusResponse,
    GenerateAssetRequest,
    GenerationProgress,
    RefineAssetRequest,
)

router = APIRouter(prefix="/api/v1/assets", tags=["Assets"])

# Global workflow instance (would be properly initialized in app startup)
workflow_graph: StateGraph | None = None

# WebSocket connections for real-time updates
active_connections: dict[UUID, list[WebSocket]] = {}


@router.post(
    "/generate", response_model=AssetResponse, status_code=status.HTTP_202_ACCEPTED
)
async def generate_asset(
    request: GenerateAssetRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthUser = Depends(require_asset_create),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Generate a new 3D asset from text prompt."""
    try:
        # Create asset record in database
        asset_name = request.name or f"Asset from: {request.prompt[:50]}"
        metadata = {
            "prompt": request.prompt,
            "complexity": request.complexity,
            "style": request.style,
            "materials": request.materials or [],
            "lighting": request.lighting,
            "quality": request.quality,
            "generation_settings": request.metadata or {},
        }

        db_asset = await asset_repo.create_asset(
            user_id=current_user.id,
            name=asset_name,
            prompt=request.prompt,
            metadata=metadata,
        )

        # Queue generation task
        background_tasks.add_task(
            _generate_asset_background,
            asset_id=db_asset.id,
            request=request,
            user_id=current_user.id,
        )

        # Convert to API model
        asset = Asset(
            id=db_asset.id,
            user_id=db_asset.user_id,
            name=db_asset.name,
            prompt=db_asset.prompt,
            status=db_asset.status,
            blender_file_url=db_asset.blender_file_url,
            preview_image_url=db_asset.preview_image_url,
            metadata=None,  # Would convert from JSON
            created_at=db_asset.created_at,
            updated_at=db_asset.updated_at,
        )

        return AssetResponse(
            asset=asset,
            generation_job_id=db_asset.id,  # Using asset ID as job ID for simplicity
            estimated_completion=None,  # Would calculate based on queue
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start asset generation: {str(e)}",
        ) from e


@router.post("/{asset_id}/refine", response_model=AssetResponse)
async def refine_asset(
    asset_id: UUID,
    request: RefineAssetRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Refine an existing asset with user feedback."""
    # Get existing asset
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
            detail="Asset must be completed before refinement",
        )

    # Queue refinement task
    background_tasks.add_task(
        _refine_asset_background,
        asset_id=asset_id,
        feedback=request.feedback,
        user_id=current_user.id,
    )

    # Convert to API model
    asset = Asset(
        id=db_asset.id,
        user_id=db_asset.user_id,
        name=db_asset.name,
        prompt=db_asset.prompt,
        status="in_progress",  # Status changed to in_progress
        blender_file_url=db_asset.blender_file_url,
        preview_image_url=db_asset.preview_image_url,
        created_at=db_asset.created_at,
        updated_at=db_asset.updated_at,
    )

    return AssetResponse(asset=asset)


@router.get("/{asset_id}/status", response_model=AssetStatusResponse)
async def get_asset_status(
    asset_id: UUID,
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Get real-time generation status for an asset."""
    db_asset = await asset_repo.get_asset_by_id(asset_id)

    if not db_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found"
        )

    if db_asset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return AssetStatusResponse(
        asset_id=asset_id,
        status=db_asset.status,
        progress=None,  # Would get from generation job
        error_message=None,
    )


@router.get("/{asset_id}", response_model=Asset)
async def get_asset(
    asset_id: UUID,
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Get asset details."""
    db_asset = await asset_repo.get_asset_by_id(asset_id)

    if not db_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found"
        )

    if db_asset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return Asset(
        id=db_asset.id,
        user_id=db_asset.user_id,
        name=db_asset.name,
        prompt=db_asset.prompt,
        status=db_asset.status,
        blender_file_url=db_asset.blender_file_url,
        preview_image_url=db_asset.preview_image_url,
        metadata=None,  # Would convert from JSON
        created_at=db_asset.created_at,
        updated_at=db_asset.updated_at,
    )


@router.get("", response_model=list[Asset])
async def list_assets(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """List user's assets with pagination."""
    db_assets = await asset_repo.get_user_assets(
        user_id=current_user.id, limit=limit, offset=offset
    )

    assets = []
    for db_asset in db_assets:
        assets.append(
            Asset(
                id=db_asset.id,
                user_id=db_asset.user_id,
                name=db_asset.name,
                prompt=db_asset.prompt,
                status=db_asset.status,
                blender_file_url=db_asset.blender_file_url,
                preview_image_url=db_asset.preview_image_url,
                metadata=None,
                created_at=db_asset.created_at,
                updated_at=db_asset.updated_at,
            )
        )

    return assets


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    asset_id: UUID,
    current_user: AuthUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repo),
):
    """Delete an asset."""
    db_asset = await asset_repo.get_asset_by_id(asset_id)

    if not db_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found"
        )

    if db_asset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Delete asset (would implement in repository)
    # await asset_repo.delete_asset(asset_id)


@router.websocket("/{asset_id}/stream")
async def stream_generation_progress(websocket: WebSocket, asset_id: UUID):
    """WebSocket endpoint for streaming real-time generation progress."""
    await websocket.accept()

    # Add connection to active connections
    if asset_id not in active_connections:
        active_connections[asset_id] = []
    active_connections[asset_id].append(websocket)

    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()

            # Echo back any received messages (for heartbeat)
            await websocket.send_json({"type": "heartbeat", "data": data})

    except WebSocketDisconnect:
        # Remove connection when client disconnects
        if asset_id in active_connections:
            active_connections[asset_id].remove(websocket)
            if not active_connections[asset_id]:
                del active_connections[asset_id]


async def _broadcast_progress(asset_id: UUID, progress: GenerationProgress):
    """Broadcast progress update to all connected WebSocket clients."""
    if asset_id in active_connections:
        disconnected_connections = []

        for websocket in active_connections[asset_id]:
            try:
                await websocket.send_json(
                    {"type": "progress_update", "data": progress.model_dump()}
                )
            except:
                disconnected_connections.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected_connections:
            active_connections[asset_id].remove(websocket)


async def _generate_asset_background(
    asset_id: UUID, request: GenerateAssetRequest, user_id: UUID
):
    """Background task for asset generation."""
    try:
        # Update status to in_progress
        asset_repo = get_asset_repo()
        await asset_repo.update_asset_status(asset_id, "in_progress")

        # Send initial progress update
        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="in_progress",
                progress=10,
                current_step="Planning asset generation",
                intermediate_images=[],
                log_messages=["Starting asset generation..."],
            ),
        )

        # Simulate generation process (in real implementation, would use workflow graph)
        await asyncio.sleep(2)
        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="in_progress",
                progress=30,
                current_step="Generating Blender code",
                intermediate_images=[],
                log_messages=[
                    "Planning completed",
                    "Generating Blender Python code...",
                ],
            ),
        )

        await asyncio.sleep(3)
        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="in_progress",
                progress=60,
                current_step="Executing in Blender",
                intermediate_images=[],
                log_messages=["Code generation completed", "Executing in Blender..."],
            ),
        )

        await asyncio.sleep(2)
        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="in_progress",
                progress=85,
                current_step="Rendering final asset",
                intermediate_images=[],
                log_messages=[
                    "Blender execution completed",
                    "Rendering final asset...",
                ],
            ),
        )

        await asyncio.sleep(1)

        # Complete generation
        await asset_repo.update_asset_status(
            asset_id,
            "completed",
            blender_file_url=f"https://storage.ll3m.com/assets/{asset_id}/asset.blend",
            preview_image_url=f"https://storage.ll3m.com/assets/{asset_id}/preview.png",
        )

        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="completed",
                progress=100,
                current_step="Generation completed",
                intermediate_images=[],
                log_messages=["Asset generation completed successfully!"],
            ),
        )

    except Exception as e:
        # Handle errors
        await asset_repo.update_asset_status(asset_id, "failed")
        await _broadcast_progress(
            asset_id,
            GenerationProgress(
                asset_id=asset_id,
                status="failed",
                progress=0,
                current_step="Generation failed",
                intermediate_images=[],
                log_messages=[f"Error: {str(e)}"],
            ),
        )


async def _refine_asset_background(asset_id: UUID, feedback: str, user_id: UUID):
    """Background task for asset refinement."""
    try:
        asset_repo = get_asset_repo()
        await asset_repo.update_asset_status(asset_id, "in_progress")

        # Simulate refinement process
        await asyncio.sleep(3)

        await asset_repo.update_asset_status(
            asset_id,
            "completed",
            preview_image_url=f"https://storage.ll3m.com/assets/{asset_id}/refined_preview.png",
        )

    except Exception:
        asset_repo = get_asset_repo()
        await asset_repo.update_asset_status(asset_id, "failed")
