"""Main FastAPI application factory."""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

import src.api.database as db_module

from ..utils.config import Settings, get_settings
from .database import AssetRepository, DatabaseManager, UserRepository
from .models import ErrorResponse
from .routes import (
    assets_router,
    auth_router,
    batches_router,
    exports_router,
    health_router,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database instances
db_manager: DatabaseManager = None
user_repo: UserRepository = None
asset_repo: AssetRepository = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting LL3M API application...")

    # Initialize database
    await _initialize_database()

    # Initialize other services
    await _initialize_services()

    logger.info("LL3M API application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down LL3M API application...")

    # Clean up database connections
    if db_manager:
        await db_manager.close()

    logger.info("LL3M API application shut down")


async def _initialize_database() -> None:
    """Initialize database connection and repositories."""
    global db_manager, user_repo, asset_repo

    settings = get_settings()

    # Create database connection - require explicit database URL
    database_url = getattr(settings, "database_url", None)
    if not database_url:
        # Only allow SQLite fallback in development/testing
        env = os.getenv("ENVIRONMENT", "development")
        if env.lower() in ("production", "staging"):
            raise ValueError(
                "DATABASE_URL must be explicitly set in production/staging environments"
            )
        database_url = "sqlite+aiosqlite:///./ll3m.db"
        logger.warning(
            "Using SQLite fallback database - not recommended for production"
        )

    db_manager = DatabaseManager(database_url)

    # Create tables if they don't exist
    await db_manager.create_tables()

    # Initialize repositories
    user_repo = UserRepository(db_manager)
    asset_repo = AssetRepository(db_manager)

    logger.info("Database initialized successfully")


async def _initialize_services() -> None:
    """Initialize external services and connections."""
    # Initialize workflow graph (would be done here)
    # Initialize Context7 MCP client
    # Initialize LLM clients
    # Initialize monitoring/metrics

    logger.info("External services initialized")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    # Create FastAPI app with lifespan manager
    app = FastAPI(
        title="LL3M API",
        description=(
            "Large Language 3D Modelers - Production API "
            "for generating 3D assets from text prompts"
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    _add_middleware(app, settings)

    # Add exception handlers
    _add_exception_handlers(app)

    # Include routers
    _include_routers(app)

    # Add rate limiting to critical endpoints
    _add_rate_limiting(app)

    return app


def _add_middleware(app: FastAPI, settings: Settings) -> None:
    """Add middleware to the FastAPI application."""
    # Add rate limiter to app state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # CORS middleware with production-safe settings
    allowed_origins = settings.app.allowed_origins
    if not allowed_origins:
        # Fallback for development only
        env = os.getenv("ENVIRONMENT", "development")
        if env.lower() in ("production", "staging"):
            allowed_origins = []  # No wildcards in production
        else:
            allowed_origins = ["http://localhost:3000", "http://localhost:3001"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Trusted host middleware (for production security)
    if settings.app.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=settings.app.allowed_hosts
        )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} - {process_time:.3f}s"
        )

        return response


def _add_exception_handlers(app: FastAPI) -> None:
    """Add global exception handlers."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                code=str(exc.status_code),
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation error",
                detail=str(exc),
                code="422",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred",
                code="500",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )


def _include_routers(app: FastAPI) -> None:
    """Include all API routers."""
    # Health checks (no authentication required)
    app.include_router(health_router)

    # Authentication routes
    app.include_router(auth_router)

    # Protected routes
    app.include_router(assets_router)
    app.include_router(batches_router)
    app.include_router(exports_router)

    # Root endpoint
    @app.get("/")
    @limiter.limit("60/minute")
    async def root(request: Request) -> dict[str, str]:  # noqa: ARG001
        return {
            "name": "LL3M API",
            "version": "1.0.0",
            "description": "Large Language 3D Modelers API",
            "docs_url": "/api/docs",
            "health_url": "/api/v1/health",
        }

    # API info endpoint
    @app.get("/api")
    @limiter.limit("30/minute")
    async def api_info(request: Request) -> dict[str, Any]:  # noqa: ARG001
        return {
            "name": "LL3M API",
            "version": "1.0.0",
            "endpoints": {
                "authentication": "/api/v1/auth",
                "assets": "/api/v1/assets",
                "batches": "/api/v1/batches",
                "exports": "/api/v1/exports",
                "health": "/api/v1/health",
            },
            "documentation": {
                "swagger": "/api/docs",
                "redoc": "/api/redoc",
                "openapi": "/api/openapi.json",
            },
        }


def _add_rate_limiting(app: FastAPI) -> None:
    """Add rate limiting to existing endpoints."""
    # Apply rate limiting to critical routes
    # This would be done in the route definitions in a production app
    pass


# Dependency providers for injection
def get_database_manager() -> DatabaseManager:
    """Dependency provider for database manager."""
    if db_manager is None:
        raise RuntimeError("Database not initialized")
    return db_manager


def get_user_repository() -> UserRepository:
    """Dependency provider for user repository."""
    if user_repo is None:
        raise RuntimeError("User repository not initialized")
    return user_repo


def get_asset_repository() -> AssetRepository:
    """Dependency provider for asset repository."""
    if asset_repo is None:
        raise RuntimeError("Asset repository not initialized")
    return asset_repo


# Update database module's dependency functions
db_module.db_manager = lambda: db_manager
db_module.user_repo = lambda: user_repo
db_module.asset_repo = lambda: asset_repo
