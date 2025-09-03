"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AssetStatus(str, Enum):
    """Asset generation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Supported export formats."""

    GLTF = "gltf"
    OBJ = "obj"
    FBX = "fbx"
    BLEND = "blend"
    USDZ = "usdz"


class SubscriptionTier(str, Enum):
    """User subscription tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class GenerateAssetRequest(BaseModel):
    """Request to generate a new 3D asset."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Text description of the desired 3D asset",
    )
    name: str | None = Field(
        None, max_length=255, description="Optional name for the asset"
    )
    complexity: str | None = Field(
        "medium", description="Asset complexity level: simple, medium, complex"
    )
    style: str | None = Field(None, description="Art style preference")
    materials: list[str] | None = Field(None, description="Specific materials to use")
    lighting: str | None = Field("natural", description="Lighting setup preference")
    quality: str | None = Field(
        "high", description="Rendering quality: low, medium, high"
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class RefineAssetRequest(BaseModel):
    """Request to refine an existing asset."""

    model_config = ConfigDict(str_strip_whitespace=True)

    feedback: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Specific refinement instructions",
    )
    priority_areas: list[str] | None = Field(
        None, description="Areas to focus refinement on"
    )
    preserve_aspects: list[str] | None = Field(
        None, description="Aspects to preserve during refinement"
    )


class AssetMetadata(BaseModel):
    """Asset metadata information."""

    prompt: str
    complexity: str
    style: str | None = None
    materials: list[str] = []
    lighting: str = "natural"
    quality: str = "high"
    generation_settings: dict[str, Any] = {}
    file_size: int | None = None
    polygon_count: int | None = None
    creation_time: float | None = None


class AssetPreview(BaseModel):
    """Asset preview information."""

    thumbnail_url: str
    preview_images: list[str] = []
    preview_video_url: str | None = None
    turnaround_url: str | None = None


class Asset(BaseModel):
    """Asset information."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    name: str
    prompt: str
    status: AssetStatus
    blender_file_url: str | None = None
    preview_image_url: str | None = None
    metadata: AssetMetadata | None = None
    previews: AssetPreview | None = None
    created_at: datetime
    updated_at: datetime


class GenerationProgress(BaseModel):
    """Real-time generation progress."""

    asset_id: UUID
    status: AssetStatus
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_step: str
    estimated_completion: datetime | None = None
    intermediate_images: list[str] = []
    log_messages: list[str] = []


class AssetResponse(BaseModel):
    """Response containing asset information."""

    asset: Asset
    generation_job_id: UUID | None = None
    estimated_completion: datetime | None = None


class AssetStatusResponse(BaseModel):
    """Asset generation status response."""

    asset_id: UUID
    status: AssetStatus
    progress: GenerationProgress | None = None
    error_message: str | None = None


class ExportRequest(BaseModel):
    """Request to export an asset in specific format."""

    format: ExportFormat
    quality: str = Field("high", description="Export quality: low, medium, high")
    optimize_for: str | None = Field(
        None, description="Optimization target: web, mobile, desktop, vr"
    )
    include_materials: bool = True
    include_textures: bool = True
    custom_settings: dict[str, Any] | None = None


class ExportResponse(BaseModel):
    """Asset export response."""

    export_id: UUID
    download_url: str
    format: ExportFormat
    file_size: int
    expires_at: datetime


class User(BaseModel):
    """User information."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    name: str
    subscription_tier: SubscriptionTier
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class UserCreate(BaseModel):
    """User creation request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(..., max_length=255)
    name: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """User login request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    email: str
    password: str


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105  # OAuth token type, not a password
    expires_in: int
    refresh_token: str | None = None


class BatchRequest(BaseModel):
    """Batch processing request."""

    name: str = Field(..., max_length=255)
    requests: list[GenerateAssetRequest] = Field(..., min_items=2, max_items=50)
    priority: int = Field(
        1, ge=1, le=5, description="Batch priority (1=lowest, 5=highest)"
    )
    notify_on_completion: bool = True


class BatchResponse(BaseModel):
    """Batch processing response."""

    batch_id: UUID
    total_assets: int
    estimated_completion: datetime
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime: float
    dependencies: dict[str, str]
    metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None
    code: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None
