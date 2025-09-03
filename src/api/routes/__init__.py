"""API routes module."""

from .assets import router as assets_router
from .auth import router as auth_router
from .batches import router as batches_router
from .exports import router as exports_router
from .health import router as health_router

__all__ = [
    "auth_router",
    "assets_router",
    "health_router",
    "batches_router",
    "exports_router",
]
