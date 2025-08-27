# Phase 5: User Interface & API Implementation Plan

## Overview
This document outlines the implementation plan for Phase 5 of LL3M: User Interface & API development. This phase focuses on creating production-ready interfaces for interacting with the LL3M system through both REST API and CLI interfaces.

## Phase 5 Objectives
- **REST API**: FastAPI-based web service with comprehensive endpoints
- **CLI Interface**: User-friendly command-line tool for local interaction
- **WebUI**: Optional web-based interface for visual asset management
- **Documentation**: OpenAPI specs and user guides
- **Authentication**: Secure API access with rate limiting
- **Asset Management**: Web interface for browsing and managing generated assets

## Architecture Overview

### Technology Stack
- **API Framework**: FastAPI with async support
- **CLI Framework**: Click with rich formatting
- **WebUI**: Streamlit or Flask (optional)
- **Authentication**: JWT tokens with rate limiting
- **Documentation**: Swagger/OpenAPI auto-generation
- **Asset Storage**: Filesystem with metadata indexing
- **Background Tasks**: Celery or FastAPI Background Tasks

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Client    │    │   Web Client    │    │  Mobile App     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Load Balancer │
                    │    (Optional)   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   REST API      │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LL3M Core     │    │   Asset         │    │   User          │
│   Workflow      │    │   Manager       │    │   Management    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Tasks

### Task 1: FastAPI REST API Foundation
**Duration**: 2-3 days

#### 1.1 Project Structure Setup
```
src/api/
├── __init__.py
├── main.py              # FastAPI application entry point
├── dependencies.py      # Dependency injection
├── middleware.py        # Custom middleware
├── exceptions.py        # Custom exception handlers
├── models/             # Pydantic request/response models
│   ├── __init__.py
│   ├── requests.py     # Request models
│   ├── responses.py    # Response models
│   └── auth.py         # Authentication models
├── routers/            # API route modules
│   ├── __init__.py
│   ├── generate.py     # Asset generation endpoints
│   ├── assets.py       # Asset management endpoints
│   ├── auth.py         # Authentication endpoints
│   └── health.py       # Health check endpoints
└── security/           # Security components
    ├── __init__.py
    ├── auth.py         # Authentication logic
    └── rate_limit.py   # Rate limiting
```

#### 1.2 Core API Application
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import structlog

from .routers import generate, assets, auth, health
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware
from .exceptions import setup_exception_handlers
from .dependencies import get_current_user, get_asset_manager
from ..utils.config import settings

logger = structlog.get_logger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="LL3M API",
        description="Large Language 3D Modelers REST API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(generate.router, prefix="/generate", tags=["Generation"])
    app.include_router(assets.router, prefix="/assets", tags=["Assets"])
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("LL3M API starting up")
        # Initialize services
        
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("LL3M API shutting down")
        # Cleanup resources
    
    return app

app = create_app()
```

#### 1.3 Request/Response Models
```python
# src/api/models/requests.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class GenerationRequest(BaseModel):
    """Request model for 3D asset generation."""
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Text prompt describing the desired 3D asset"
    )
    quality_level: str = Field(
        default="standard",
        description="Quality level: draft, standard, high"
    )
    output_formats: List[str] = Field(
        default=["blend", "obj"],
        description="Desired output formats"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for asset organization"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt content for security."""
        forbidden_patterns = ['import os', 'subprocess', 'eval(', 'exec(']
        for pattern in forbidden_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Prompt contains forbidden pattern: {pattern}")
        return v.strip()
    
    @validator('quality_level')
    def validate_quality(cls, v):
        valid_levels = ['draft', 'standard', 'high']
        if v not in valid_levels:
            raise ValueError(f"Quality level must be one of: {valid_levels}")
        return v

class RefinementRequest(BaseModel):
    """Request model for asset refinement."""
    
    asset_id: str = Field(..., description="ID of asset to refine")
    feedback: str = Field(
        ..., 
        min_length=1, 
        max_length=300,
        description="Refinement instructions"
    )
    preserve_aspects: Optional[List[str]] = Field(
        default=None,
        description="Aspects to preserve during refinement"
    )

# src/api/models/responses.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AssetResponse(BaseModel):
    """Response model for asset information."""
    
    id: str = Field(..., description="Unique asset identifier")
    name: str = Field(..., description="Asset name")
    prompt: str = Field(..., description="Original creation prompt")
    status: str = Field(..., description="Generation status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    file_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="File paths by format"
    )
    screenshot_url: Optional[str] = Field(
        None,
        description="URL to asset screenshot"
    )
    quality_score: Optional[float] = Field(
        None,
        description="Quality assessment score"
    )
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationResponse(BaseModel):
    """Response model for generation requests."""
    
    task_id: str = Field(..., description="Generation task ID")
    status: str = Field(..., description="Task status")
    asset: Optional[AssetResponse] = Field(None, description="Generated asset")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    error_message: Optional[str] = Field(None, description="Error details")

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Dependency health status"
    )
```

### Task 2: Core API Endpoints
**Duration**: 3-4 days

#### 2.1 Asset Generation Endpoints
```python
# src/api/routers/generate.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, Any
import uuid
import asyncio

from ..models.requests import GenerationRequest, RefinementRequest
from ..models.responses import GenerationResponse, AssetResponse
from ..dependencies import get_current_user, get_workflow_manager
from ..security.auth import require_auth
from ...workflow.enhanced_graph import create_enhanced_workflow
from ...assets.manager import AssetManager
from ...utils.types import WorkflowState

router = APIRouter()
security = HTTPBearer()

# In-memory task storage (replace with Redis in production)
generation_tasks: Dict[str, Dict[str, Any]] = {}

@router.post("/", response_model=GenerationResponse)
async def generate_asset(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_auth),
    workflow_manager = Depends(get_workflow_manager)
) -> GenerationResponse:
    """Generate a new 3D asset from text prompt."""
    
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    generation_tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "user_id": current_user["user_id"],
        "request": request.dict(),
        "created_at": datetime.utcnow(),
        "asset_id": None
    }
    
    # Start background generation
    background_tasks.add_task(
        _execute_generation_workflow,
        task_id=task_id,
        request=request,
        workflow_manager=workflow_manager
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="queued",
        progress=0
    )

@router.get("/{task_id}", response_model=GenerationResponse)
async def get_generation_status(
    task_id: str,
    current_user: dict = Depends(require_auth)
) -> GenerationResponse:
    """Get generation task status."""
    
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    
    # Check user permission
    if task["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    response = GenerationResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"]
    )
    
    if task.get("asset_id"):
        # Load asset information
        asset_manager = AssetManager()
        managed_asset = asset_manager.repository.get_asset(task["asset_id"])
        if managed_asset:
            response.asset = _convert_to_asset_response(managed_asset)
    
    if task.get("error"):
        response.error_message = task["error"]
    
    return response

@router.post("/{asset_id}/refine", response_model=GenerationResponse)
async def refine_asset(
    asset_id: str,
    request: RefinementRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_auth),
    workflow_manager = Depends(get_workflow_manager)
) -> GenerationResponse:
    """Refine an existing asset with user feedback."""
    
    # Verify asset exists and user has permission
    asset_manager = AssetManager()
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    task_id = str(uuid.uuid4())
    
    generation_tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "user_id": current_user["user_id"],
        "type": "refinement",
        "asset_id": asset_id,
        "request": request.dict(),
        "created_at": datetime.utcnow()
    }
    
    background_tasks.add_task(
        _execute_refinement_workflow,
        task_id=task_id,
        asset_id=asset_id,
        request=request,
        workflow_manager=workflow_manager
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="queued",
        progress=0
    )

async def _execute_generation_workflow(
    task_id: str,
    request: GenerationRequest,
    workflow_manager
) -> None:
    """Execute generation workflow in background."""
    
    try:
        # Update status
        generation_tasks[task_id].update({
            "status": "running",
            "progress": 10
        })
        
        # Create workflow state
        initial_state = WorkflowState(
            prompt=request.prompt,
            quality_level=request.quality_level,
            output_formats=request.output_formats
        )
        
        # Execute workflow
        workflow = create_enhanced_workflow()
        final_state = await workflow.ainvoke(initial_state)
        
        if final_state.error_message:
            generation_tasks[task_id].update({
                "status": "failed",
                "error": final_state.error_message,
                "progress": 100
            })
            return
        
        # Create managed asset
        asset_manager = AssetManager()
        managed_asset = asset_manager.create_from_workflow_state(
            final_state,
            tags=request.tags
        )
        
        if managed_asset:
            generation_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "asset_id": managed_asset.id
            })
        else:
            generation_tasks[task_id].update({
                "status": "failed",
                "error": "Failed to create managed asset",
                "progress": 100
            })
            
    except Exception as e:
        generation_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 100
        })

def _convert_to_asset_response(managed_asset) -> AssetResponse:
    """Convert ManagedAsset to AssetResponse."""
    current_version = managed_asset.current_asset_version
    
    file_paths = {}
    if current_version and current_version.file_path:
        # Extract format from file extension
        file_path = Path(current_version.file_path)
        format_name = file_path.suffix.lstrip('.')
        file_paths[format_name] = str(file_path)
    
    return AssetResponse(
        id=managed_asset.id,
        name=managed_asset.name,
        prompt=managed_asset.original_prompt,
        status="completed",
        created_at=datetime.fromtimestamp(managed_asset.created_at),
        updated_at=datetime.fromtimestamp(managed_asset.updated_at),
        file_paths=file_paths,
        screenshot_url=current_version.screenshot_path if current_version else None,
        quality_score=current_version.quality_score if current_version else None,
        tags=managed_asset.tags,
        metadata=current_version.metadata if current_version else {}
    )
```

#### 2.2 Asset Management Endpoints
```python
# src/api/routers/assets.py
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from typing import List, Optional
from pathlib import Path

from ..models.responses import AssetResponse
from ..dependencies import get_current_user, get_asset_manager
from ..security.auth import require_auth

router = APIRouter()

@router.get("/", response_model=List[AssetResponse])
async def list_assets(
    tags: Optional[List[str]] = Query(None),
    min_quality: Optional[float] = Query(None, ge=0, le=10),
    limit: Optional[int] = Query(20, ge=1, le=100),
    offset: Optional[int] = Query(0, ge=0),
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> List[AssetResponse]:
    """List user's assets with optional filtering."""
    
    # Get assets (in production, filter by user_id)
    managed_assets = asset_manager.repository.list_assets(
        tags=tags,
        min_quality_score=min_quality,
        limit=limit
    )
    
    # Apply offset
    if offset:
        managed_assets = managed_assets[offset:]
    
    return [
        _convert_to_asset_response(asset)
        for asset in managed_assets
    ]

@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: str,
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> AssetResponse:
    """Get specific asset details."""
    
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return _convert_to_asset_response(managed_asset)

@router.get("/{asset_id}/download/{format}")
async def download_asset(
    asset_id: str,
    format: str,
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> FileResponse:
    """Download asset in specified format."""
    
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    current_version = managed_asset.current_asset_version
    if not current_version or not current_version.file_path:
        raise HTTPException(status_code=404, detail="Asset file not found")
    
    file_path = Path(current_version.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset file not found on disk")
    
    return FileResponse(
        path=str(file_path),
        filename=f"{managed_asset.name}.{format}",
        media_type="application/octet-stream"
    )

@router.get("/{asset_id}/screenshot")
async def get_asset_screenshot(
    asset_id: str,
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> FileResponse:
    """Get asset screenshot."""
    
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    current_version = managed_asset.current_asset_version
    if not current_version or not current_version.screenshot_path:
        raise HTTPException(status_code=404, detail="Screenshot not available")
    
    screenshot_path = Path(current_version.screenshot_path)
    if not screenshot_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot file not found")
    
    return FileResponse(
        path=str(screenshot_path),
        media_type="image/png"
    )

@router.delete("/{asset_id}")
async def delete_asset(
    asset_id: str,
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> dict:
    """Delete an asset and all its versions."""
    
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    success = asset_manager.repository.delete_asset(asset_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete asset")
    
    return {"message": "Asset deleted successfully"}

@router.get("/{asset_id}/versions")
async def list_asset_versions(
    asset_id: str,
    current_user: dict = Depends(require_auth),
    asset_manager = Depends(get_asset_manager)
) -> List[dict]:
    """List all versions of an asset."""
    
    managed_asset = asset_manager.repository.get_asset(asset_id)
    
    if not managed_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return [
        {
            "version": version.version,
            "timestamp": datetime.fromtimestamp(version.timestamp),
            "quality_score": version.quality_score,
            "refinement_request": version.refinement_request,
            "is_current": version.version == managed_asset.current_version
        }
        for version in managed_asset.versions
    ]
```

### Task 3: Authentication & Security
**Duration**: 2-3 days

#### 3.1 JWT Authentication
```python
# src/api/security/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import structlog

from ...utils.config import settings

logger = structlog.get_logger(__name__)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    return payload

async def require_auth(current_user: dict = Depends(get_current_user)) -> dict:
    """Require authentication for endpoint."""
    return current_user

# User management (simplified - use proper user DB in production)
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "email": "test@example.com"
    }
}

def authenticate_user(username: str, password: str):
    """Authenticate user credentials."""
    user = fake_users_db.get(username)
    if not user:
        return False
    if not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

# src/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from ..models.auth import TokenResponse, UserRegistration
from ..security.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["username"]}, 
        expires_delta=access_token_expires
    )
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
```

#### 3.2 Rate Limiting Middleware
```python
# src/api/middleware.py
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 3600):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if now - req_time < self.period
        ]
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        # Record this request
        self.clients[client_ip].append(now)
        
        response = await call_next(request)
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            "Request processed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
```

### Task 4: CLI Interface
**Duration**: 2-3 days

#### 4.1 Click-based CLI Framework
```python
# src/cli/main.py
import click
import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import structlog

from .client import LL3MClient
from .config import CLIConfig
from ..utils.config import settings

console = Console()
logger = structlog.get_logger(__name__)

@click.group()
@click.option('--config-file', default='~/.ll3m/config.json', help='Configuration file path')
@click.option('--api-url', help='API base URL')
@click.option('--api-key', help='API key for authentication')
@click.pass_context
def cli(ctx, config_file, api_url, api_key):
    """LL3M CLI - Large Language 3D Modelers Command Line Interface."""
    ctx.ensure_object(dict)
    
    # Load configuration
    config = CLIConfig.load(config_file)
    if api_url:
        config.api_url = api_url
    if api_key:
        config.api_key = api_key
    
    ctx.obj['config'] = config
    ctx.obj['client'] = LL3MClient(config)

@cli.command()
@click.argument('prompt')
@click.option('--quality', default='standard', type=click.Choice(['draft', 'standard', 'high']))
@click.option('--format', 'formats', multiple=True, default=['blend', 'obj'])
@click.option('--tag', 'tags', multiple=True, help='Tags for asset organization')
@click.option('--output', help='Output directory')
@click.option('--wait', is_flag=True, help='Wait for completion')
@click.pass_context
def generate(ctx, prompt, quality, formats, tags, output, wait):
    """Generate a 3D asset from text prompt.
    
    Examples:
        ll3m generate "a red sports car"
        ll3m generate "medieval castle with towers" --quality high --format gltf --wait
    """
    client = ctx.obj['client']
    
    try:
        with console.status("[bold green]Submitting generation request..."):
            response = asyncio.run(client.generate_asset(
                prompt=prompt,
                quality_level=quality,
                output_formats=list(formats),
                tags=list(tags) if tags else None
            ))
        
        console.print(f"✓ Generation started with task ID: {response.task_id}")
        
        if wait:
            _wait_for_completion(client, response.task_id, output)
        else:
            console.print("Use 'll3m status {task_id}' to check progress")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('task_id')
@click.option('--output', help='Output directory for completed assets')
@click.pass_context
def status(ctx, task_id, output):
    """Check generation task status.
    
    Examples:
        ll3m status abc123
        ll3m status abc123 --output ./downloads
    """
    client = ctx.obj['client']
    
    try:
        response = asyncio.run(client.get_generation_status(task_id))
        
        # Display status information
        table = Table(title=f"Task Status: {task_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        
        table.add_row("Status", response.status)
        table.add_row("Progress", f"{response.progress}%")
        
        if response.asset:
            table.add_row("Asset ID", response.asset.id)
            table.add_row("Asset Name", response.asset.name)
            if response.asset.quality_score:
                table.add_row("Quality Score", f"{response.asset.quality_score:.1f}/10")
        
        if response.error_message:
            table.add_row("Error", response.error_message)
        
        console.print(table)
        
        # Download asset if completed and output specified
        if response.status == "completed" and response.asset and output:
            _download_asset(client, response.asset.id, output)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--min-quality', type=float, help='Minimum quality score')
@click.option('--limit', default=20, help='Maximum number of assets to list')
@click.pass_context
def list(ctx, tags, min_quality, limit):
    """List your generated assets.
    
    Examples:
        ll3m list
        ll3m list --tags car,vehicle --min-quality 8.0
    """
    client = ctx.obj['client']
    
    try:
        tag_list = tags.split(',') if tags else None
        assets = asyncio.run(client.list_assets(
            tags=tag_list,
            min_quality=min_quality,
            limit=limit
        ))
        
        if not assets:
            console.print("No assets found.")
            return
        
        table = Table(title="Your Assets")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Created")
        table.add_column("Quality", justify="right")
        table.add_column("Tags")
        
        for asset in assets:
            quality_str = f"{asset.quality_score:.1f}" if asset.quality_score else "N/A"
            tags_str = ", ".join(asset.tags) if asset.tags else ""
            
            table.add_row(
                asset.id[:8] + "...",
                asset.name,
                asset.created_at.strftime("%Y-%m-%d %H:%M"),
                quality_str,
                tags_str
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('asset_id')
@click.argument('feedback')
@click.option('--wait', is_flag=True, help='Wait for completion')
@click.pass_context
def refine(ctx, asset_id, feedback, wait):
    """Refine an existing asset.
    
    Examples:
        ll3m refine abc123 "make it bigger and add more detail"
        ll3m refine abc123 "change color to blue" --wait
    """
    client = ctx.obj['client']
    
    try:
        with console.status("[bold green]Submitting refinement request..."):
            response = asyncio.run(client.refine_asset(asset_id, feedback))
        
        console.print(f"✓ Refinement started with task ID: {response.task_id}")
        
        if wait:
            _wait_for_completion(client, response.task_id)
        else:
            console.print("Use 'll3m status {task_id}' to check progress")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('asset_id')
@click.option('--output', default='.', help='Output directory')
@click.option('--format', help='Specific format to download')
@click.pass_context
def download(ctx, asset_id, output, format):
    """Download an asset.
    
    Examples:
        ll3m download abc123
        ll3m download abc123 --output ./models --format obj
    """
    client = ctx.obj['client']
    
    try:
        _download_asset(client, asset_id, output, format)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))

def _wait_for_completion(client, task_id, output_dir=None):
    """Wait for task completion with progress display."""
    with Progress() as progress:
        task = progress.add_task("Processing...", total=100)
        
        while True:
            try:
                response = asyncio.run(client.get_generation_status(task_id))
                progress.update(task, completed=response.progress)
                
                if response.status == "completed":
                    progress.update(task, completed=100)
                    console.print("✓ Generation completed successfully!")
                    
                    if response.asset and output_dir:
                        _download_asset(client, response.asset.id, output_dir)
                    
                    break
                elif response.status == "failed":
                    console.print(f"[red]✗ Generation failed: {response.error_message}[/red]")
                    break
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                console.print("\nStopping status check (generation continues on server)")
                break

def _download_asset(client, asset_id, output_dir, specific_format=None):
    """Download asset files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with console.status("[bold green]Downloading asset..."):
        downloaded_files = asyncio.run(client.download_asset(
            asset_id, output_path, specific_format
        ))
    
    console.print("✓ Downloaded files:")
    for file_path in downloaded_files:
        console.print(f"  • {file_path}")

if __name__ == '__main__':
    cli()
```

#### 4.2 CLI Client Implementation
```python
# src/cli/client.py
import httpx
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog

from ..api.models.requests import GenerationRequest, RefinementRequest
from ..api.models.responses import GenerationResponse, AssetResponse

logger = structlog.get_logger(__name__)

class LL3MClient:
    """LL3M API client for CLI."""
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.api_url
        self.api_key = config.api_key
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_asset(
        self,
        prompt: str,
        quality_level: str = "standard",
        output_formats: List[str] = None,
        tags: Optional[List[str]] = None
    ) -> GenerationResponse:
        """Submit asset generation request."""
        if output_formats is None:
            output_formats = ["blend", "obj"]
        
        request = GenerationRequest(
            prompt=prompt,
            quality_level=quality_level,
            output_formats=output_formats,
            tags=tags
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate/",
                json=request.dict(),
                headers=self.headers
            )
            response.raise_for_status()
            return GenerationResponse(**response.json())
    
    async def get_generation_status(self, task_id: str) -> GenerationResponse:
        """Get generation task status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/generate/{task_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return GenerationResponse(**response.json())
    
    async def refine_asset(self, asset_id: str, feedback: str) -> GenerationResponse:
        """Submit asset refinement request."""
        request = RefinementRequest(
            asset_id=asset_id,
            feedback=feedback
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate/{asset_id}/refine",
                json=request.dict(),
                headers=self.headers
            )
            response.raise_for_status()
            return GenerationResponse(**response.json())
    
    async def list_assets(
        self,
        tags: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        limit: int = 20
    ) -> List[AssetResponse]:
        """List user assets."""
        params = {"limit": limit}
        if tags:
            params["tags"] = tags
        if min_quality:
            params["min_quality"] = min_quality
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/assets/",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return [AssetResponse(**item) for item in response.json()]
    
    async def download_asset(
        self,
        asset_id: str,
        output_dir: Path,
        specific_format: Optional[str] = None
    ) -> List[Path]:
        """Download asset files."""
        # First get asset info to know available formats
        asset_info = await self.get_asset(asset_id)
        
        downloaded_files = []
        formats_to_download = [specific_format] if specific_format else asset_info.file_paths.keys()
        
        async with httpx.AsyncClient() as client:
            for format_name in formats_to_download:
                if format_name not in asset_info.file_paths:
                    continue
                
                response = await client.get(
                    f"{self.base_url}/assets/{asset_id}/download/{format_name}",
                    headers=self.headers
                )
                response.raise_for_status()
                
                filename = f"{asset_info.name}.{format_name}"
                file_path = output_dir / filename
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(file_path)
        
        return downloaded_files
    
    async def get_asset(self, asset_id: str) -> AssetResponse:
        """Get asset details."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/assets/{asset_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return AssetResponse(**response.json())

# src/cli/config.py
import json
from pathlib import Path
from pydantic import BaseModel

class CLIConfig(BaseModel):
    """CLI configuration model."""
    
    api_url: str = "http://localhost:8000"
    api_key: str = ""
    default_output_dir: str = "./ll3m_downloads"
    default_quality: str = "standard"
    default_formats: List[str] = ["blend", "obj"]
    
    @classmethod
    def load(cls, config_path: str) -> 'CLIConfig':
        """Load configuration from file."""
        path = Path(config_path).expanduser()
        
        if path.exists():
            with open(path) as f:
                config_data = json.load(f)
            return cls(**config_data)
        else:
            # Create default config
            config = cls()
            config.save(config_path)
            return config
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        path = Path(config_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
```

### Task 5: Optional Web UI (Streamlit)
**Duration**: 2-3 days

#### 5.1 Streamlit Dashboard
```python
# src/ui/main.py
import streamlit as st
import asyncio
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from ..cli.client import LL3MClient
from ..cli.config import CLIConfig
from ..api.models.responses import AssetResponse

# Page configuration
st.set_page_config(
    page_title="LL3M - 3D Asset Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'client' not in st.session_state:
    config = CLIConfig()
    st.session_state.client = LL3MClient(config)

def main():
    """Main Streamlit application."""
    st.title("🎯 LL3M - Large Language 3D Modelers")
    st.markdown("Generate 3D assets from natural language descriptions")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        api_url = st.text_input("API URL", value="http://localhost:8000")
        api_key = st.text_input("API Key", type="password")
        
        if st.button("Update Config"):
            config = CLIConfig(api_url=api_url, api_key=api_key)
            st.session_state.client = LL3MClient(config)
            st.success("Configuration updated!")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Generate", "My Assets", "Analytics", "Settings"])
    
    with tab1:
        generation_page()
    
    with tab2:
        assets_page()
    
    with tab3:
        analytics_page()
    
    with tab4:
        settings_page()

def generation_page():
    """Asset generation interface."""
    st.header("Generate 3D Asset")
    
    # Generation form
    with st.form("generation_form"):
        prompt = st.text_area(
            "Describe your 3D asset:",
            placeholder="e.g., A futuristic sports car with glowing neon lights",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            quality = st.selectbox("Quality Level", ["draft", "standard", "high"])
            formats = st.multiselect(
                "Output Formats",
                ["blend", "obj", "gltf", "stl"],
                default=["blend", "obj"]
            )
        
        with col2:
            tags = st.text_input("Tags (comma-separated)", placeholder="car, vehicle, sci-fi")
        
        submitted = st.form_submit_button("Generate Asset", type="primary")
        
        if submitted and prompt:
            generate_asset(prompt, quality, formats, tags)

def generate_asset(prompt: str, quality: str, formats: list, tags: str):
    """Handle asset generation."""
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    
    try:
        # Submit generation request
        with st.spinner("Submitting generation request..."):
            response = asyncio.run(st.session_state.client.generate_asset(
                prompt=prompt,
                quality_level=quality,
                output_formats=formats,
                tags=tag_list
            ))
        
        st.success(f"Generation started! Task ID: {response.task_id}")
        
        # Progress tracking
        progress_container = st.container()
        status_container = st.container()
        
        # Poll for completion
        progress_bar = progress_container.progress(0)
        status_text = status_container.text("Starting generation...")
        
        while True:
            status_response = asyncio.run(
                st.session_state.client.get_generation_status(response.task_id)
            )
            
            progress_bar.progress(status_response.progress / 100)
            status_text.text(f"Status: {status_response.status} ({status_response.progress}%)")
            
            if status_response.status == "completed":
                st.success("✅ Generation completed successfully!")
                if status_response.asset:
                    display_asset_result(status_response.asset)
                break
            elif status_response.status == "failed":
                st.error(f"❌ Generation failed: {status_response.error_message}")
                break
            
            time.sleep(2)
            
    except Exception as e:
        st.error(f"Error: {e}")

def display_asset_result(asset: AssetResponse):
    """Display generated asset information."""
    st.subheader("Generated Asset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Asset details
        st.write(f"**Name:** {asset.name}")
        st.write(f"**ID:** {asset.id}")
        st.write(f"**Prompt:** {asset.prompt}")
        
        if asset.quality_score:
            st.metric("Quality Score", f"{asset.quality_score:.1f}/10")
        
        if asset.tags:
            st.write(f"**Tags:** {', '.join(asset.tags)}")
    
    with col2:
        # Screenshot
        if asset.screenshot_url:
            try:
                st.image(asset.screenshot_url, caption="Asset Preview")
            except:
                st.info("Preview not available")
        
        # Download buttons
        for format_name in asset.file_paths.keys():
            if st.button(f"Download {format_name.upper()}", key=f"download_{format_name}"):
                download_asset_format(asset.id, format_name)

def assets_page():
    """Asset management interface."""
    st.header("My Assets")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        tag_filter = st.text_input("Filter by tags")
    with col2:
        min_quality = st.number_input("Min Quality", 0.0, 10.0, 0.0)
    with col3:
        limit = st.selectbox("Show", [10, 20, 50, 100], index=1)
    
    # Load and display assets
    try:
        tag_list = [tag.strip() for tag in tag_filter.split(",")] if tag_filter else None
        assets = asyncio.run(st.session_state.client.list_assets(
            tags=tag_list,
            min_quality=min_quality if min_quality > 0 else None,
            limit=limit
        ))
        
        if not assets:
            st.info("No assets found.")
            return
        
        # Asset grid
        cols_per_row = 3
        for i in range(0, len(assets), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, asset in enumerate(assets[i:i + cols_per_row]):
                with cols[j]:
                    display_asset_card(asset)
    
    except Exception as e:
        st.error(f"Error loading assets: {e}")

def display_asset_card(asset: AssetResponse):
    """Display asset in card format."""
    with st.container():
        # Screenshot
        if asset.screenshot_url:
            try:
                st.image(asset.screenshot_url, use_column_width=True)
            except:
                st.info("Preview not available")
        
        st.write(f"**{asset.name}**")
        st.write(f"ID: {asset.id[:8]}...")
        st.write(f"Created: {asset.created_at.strftime('%Y-%m-%d')}")
        
        if asset.quality_score:
            st.metric("Quality", f"{asset.quality_score:.1f}")
        
        if asset.tags:
            st.write(f"Tags: {', '.join(asset.tags)}")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Details", key=f"details_{asset.id}"):
                st.session_state.selected_asset = asset.id
        
        with col2:
            if st.button("Download", key=f"download_{asset.id}"):
                download_asset_files(asset.id)

def analytics_page():
    """Analytics and statistics."""
    st.header("Analytics")
    
    try:
        # Load assets for analysis
        assets = asyncio.run(st.session_state.client.list_assets(limit=100))
        
        if not assets:
            st.info("No assets available for analysis.")
            return
        
        # Quality score distribution
        quality_scores = [asset.quality_score for asset in assets if asset.quality_score]
        if quality_scores:
            fig_quality = px.histogram(
                x=quality_scores,
                nbins=20,
                title="Quality Score Distribution",
                labels={"x": "Quality Score", "y": "Count"}
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Assets over time
        creation_dates = [asset.created_at.date() for asset in assets]
        if creation_dates:
            date_counts = {}
            for date in creation_dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            fig_timeline = px.line(
                x=list(date_counts.keys()),
                y=list(date_counts.values()),
                title="Assets Created Over Time",
                labels={"x": "Date", "y": "Assets Created"}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Tag analysis
        all_tags = []
        for asset in assets:
            all_tags.extend(asset.tags)
        
        if all_tags:
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Top tags
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig_tags = px.bar(
                x=[tag for tag, count in top_tags],
                y=[count for tag, count in top_tags],
                title="Most Popular Tags",
                labels={"x": "Tag", "y": "Usage Count"}
            )
            st.plotly_chart(fig_tags, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", len(assets))
        
        with col2:
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            st.metric("Avg Quality", f"{avg_quality:.1f}")
        
        with col3:
            st.metric("Unique Tags", len(set(all_tags)))
        
        with col4:
            high_quality_count = len([s for s in quality_scores if s >= 8.0])
            st.metric("High Quality (8+)", high_quality_count)
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def settings_page():
    """Application settings."""
    st.header("Settings")
    
    st.subheader("API Configuration")
    current_config = st.session_state.client.config
    
    new_api_url = st.text_input("API URL", value=current_config.api_url)
    new_api_key = st.text_input("API Key", value=current_config.api_key, type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        new_quality = st.selectbox(
            "Default Quality",
            ["draft", "standard", "high"],
            index=["draft", "standard", "high"].index(current_config.default_quality)
        )
    
    with col2:
        new_formats = st.multiselect(
            "Default Formats",
            ["blend", "obj", "gltf", "stl"],
            default=current_config.default_formats
        )
    
    if st.button("Save Settings"):
        new_config = CLIConfig(
            api_url=new_api_url,
            api_key=new_api_key,
            default_quality=new_quality,
            default_formats=new_formats
        )
        st.session_state.client = LL3MClient(new_config)
        st.success("Settings saved!")
    
    st.subheader("About")
    st.write("LL3M - Large Language 3D Modelers")
    st.write("Version: 0.1.0")
    st.write("Generate 3D assets using natural language descriptions")

def download_asset_files(asset_id: str):
    """Download asset files."""
    try:
        download_dir = Path("./downloads")
        download_dir.mkdir(exist_ok=True)
        
        with st.spinner("Downloading asset files..."):
            downloaded_files = asyncio.run(
                st.session_state.client.download_asset(asset_id, download_dir)
            )
        
        st.success(f"Downloaded {len(downloaded_files)} files to ./downloads/")
        for file_path in downloaded_files:
            st.write(f"• {file_path.name}")
    
    except Exception as e:
        st.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()
```

### Task 6: Production Deployment & Configuration
**Duration**: 1-2 days

#### 6.1 Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BLENDER_VERSION=4.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender
RUN wget -O blender.tar.xz https://download.blender.org/release/Blender${BLENDER_VERSION}/blender-${BLENDER_VERSION}.0-linux-x64.tar.xz \
    && tar -xf blender.tar.xz -C /opt \
    && mv /opt/blender-* /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm blender.tar.xz

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r ll3m && useradd -r -g ll3m ll3m
RUN chown -R ll3m:ll3m /app
USER ll3m

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ll3m-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - BLENDER_PATH=/usr/local/bin/blender
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - CORS_ORIGINS=["http://localhost:3000", "http://localhost:8501"]
    volumes:
      - ./assets:/app/assets
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  ll3m-ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports:
      - "8501:8501"
    environment:
      - LL3M_API_URL=http://ll3m-api:8000
    depends_on:
      - ll3m-api
    restart: unless-stopped

volumes:
  redis_data:
```

#### 6.2 Production Configuration
```python
# src/utils/config.py (enhanced)
from pydantic import BaseSettings, Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    API_TITLE: str = Field(default="LL3M API", env="API_TITLE")
    API_VERSION: str = Field(default="0.1.0", env="API_VERSION")
    
    # Security
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )
    
    # Database (for future use)
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    
    # External Services
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    BLENDER_PATH: str = Field(default="blender", env="BLENDER_PATH")
    
    # File Storage
    ASSETS_DIRECTORY: str = Field(default="assets", env="ASSETS_DIRECTORY")
    MAX_UPLOAD_SIZE_MB: int = Field(default=100, env="MAX_UPLOAD_SIZE_MB")
    
    # Performance
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    GENERATION_TIMEOUT: int = Field(default=300, env="GENERATION_TIMEOUT")
    
    # Rate Limiting
    RATE_LIMIT_CALLS: int = Field(default=100, env="RATE_LIMIT_CALLS")
    RATE_LIMIT_PERIOD: int = Field(default=3600, env="RATE_LIMIT_PERIOD")
    
    # Monitoring
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## Testing Strategy

### Unit Tests (90%+ Coverage Required)
- **API Endpoints**: Test all REST endpoints with various inputs
- **Authentication**: JWT token validation and security
- **CLI Commands**: Command parsing and execution
- **Background Tasks**: Generation and refinement workflows

### Integration Tests
- **End-to-End Workflows**: Full asset generation pipeline
- **API-CLI Integration**: CLI commands calling API endpoints
- **Asset Management**: File storage and retrieval

### Performance Tests
- **Load Testing**: API performance under concurrent requests
- **Memory Usage**: Blender process memory management
- **Response Times**: API endpoint response benchmarks

## Security Considerations

### Input Validation
- **Prompt Sanitization**: Remove potentially dangerous code patterns
- **File Type Validation**: Ensure only valid asset formats
- **Size Limits**: Prevent resource exhaustion attacks

### Authentication & Authorization
- **JWT Tokens**: Secure token-based authentication
- **Rate Limiting**: Prevent abuse and DoS attacks
- **User Isolation**: Ensure users only access their assets

### Code Execution Safety
- **Sandboxing**: Blender execution in isolated environment
- **Code Analysis**: AST-based validation of generated code
- **Resource Limits**: CPU and memory constraints

## Success Criteria

### Functional Requirements
- [ ] REST API with all specified endpoints working
- [ ] CLI interface with complete command set
- [ ] Web UI (optional) with asset management
- [ ] Authentication and user management
- [ ] File upload/download functionality
- [ ] Background task processing

### Performance Requirements
- [ ] API response time < 200ms for non-generation endpoints
- [ ] Support for 10+ concurrent users
- [ ] Asset generation completion within 5 minutes
- [ ] 99% uptime for production deployment

### Quality Requirements
- [ ] 90%+ test coverage across all components
- [ ] Security scan passes (no high/critical vulnerabilities)
- [ ] API documentation auto-generated and complete
- [ ] Error handling for all failure scenarios

## Deployment Plan

### Phase 5.1: Development Setup (Day 1)
- Set up FastAPI application structure
- Implement core API endpoints
- Create basic authentication system

### Phase 5.2: Core Functionality (Days 2-4)
- Complete all REST API endpoints
- Implement background task processing
- Add comprehensive error handling

### Phase 5.3: CLI Interface (Days 5-6)
- Build Click-based CLI application
- Integrate with REST API client
- Add rich progress display and formatting

### Phase 5.4: Testing & Security (Day 7)
- Write comprehensive test suites
- Implement security measures
- Perform load and security testing

### Phase 5.5: Optional Web UI (Days 8-9)
- Create Streamlit-based web interface
- Add asset browsing and management
- Implement analytics dashboard

### Phase 5.6: Production Deployment (Day 10)
- Docker containerization
- Production configuration
- Deployment documentation

This implementation plan provides a comprehensive, production-ready user interface system for LL3M, following all established best practices and maintaining high code quality standards.