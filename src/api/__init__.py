"""Production API module for LL3M."""

from .app import create_app
from .auth import AuthUser, auth_manager, get_current_user
from .models import (
    AssetResponse,
    AssetStatus,
    BatchRequest,
    ExportFormat,
    ExportRequest,
    ExportResponse,
    GenerateAssetRequest,
    HealthResponse,
    SubscriptionTier,
    Token,
    User,
    UserCreate,
    UserLogin,
)
from .routes import (
    assets_router,
    auth_router,
    batches_router,
    exports_router,
    health_router,
)

__all__ = [
    "create_app",
    # Auth exports
    "auth_manager",
    "AuthUser",
    "get_current_user",
    # Model exports
    "AssetResponse",
    "AssetStatus",
    "BatchRequest",
    "ExportRequest",
    "ExportResponse",
    "ExportFormat",
    "GenerateAssetRequest",
    "HealthResponse",
    "Token",
    "User",
    "UserCreate",
    "UserLogin",
    "SubscriptionTier",
    # Router exports
    "auth_router",
    "assets_router",
    "health_router",
    "batches_router",
    "exports_router",
]
