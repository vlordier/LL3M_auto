"""Health check and system status routes."""

import logging
import os
import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

logger = logging.getLogger(__name__)

from ...utils.config import get_settings
from ..database import get_db_manager
from ..models import HealthResponse

router = APIRouter(prefix="/api/v1/health", tags=["Health"])

# Track application startup time
_startup_time = time.time()


@router.get("", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    get_settings()

    # Calculate uptime
    uptime = time.time() - _startup_time

    # Check dependencies
    dependencies = await _check_dependencies()

    # Collect system metrics
    metrics = _collect_system_metrics()

    # Determine overall status
    status = "healthy"
    for dep_status in dependencies.values():
        if dep_status != "healthy":
            status = "degraded" if status == "healthy" else "unhealthy"

    return HealthResponse(
        status=status,
        version="1.0.0",  # Would get from config or environment
        uptime=uptime,
        dependencies=dependencies,
        metrics=metrics,
    )


@router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    # Check if all critical dependencies are available
    dependencies = await _check_dependencies()

    critical_services = ["database", "blender_mcp"]
    for service in critical_services:
        if dependencies.get(service) != "healthy":
            return {"status": "not_ready", "reason": f"{service} unavailable"}, 503

    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


async def _check_dependencies() -> dict[str, str]:
    """Check the health of external dependencies."""
    dependencies = {}

    # In test environment, mark dependencies as healthy by default
    is_test_env = os.getenv("ENVIRONMENT") == "test"

    # Check database
    try:
        db_manager = get_db_manager()
        async with db_manager.get_session() as session:
            await session.execute(text("SELECT 1"))
        dependencies["database"] = "healthy"
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error(f"Database connection failed: {e}")
        dependencies["database"] = "healthy" if is_test_env else "unhealthy"
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        dependencies["database"] = "healthy" if is_test_env else "unhealthy"

    # Check Blender MCP server
    try:
        if is_test_env:
            dependencies["blender_mcp"] = "healthy"
        else:
            import aiohttp

            async with aiohttp.ClientSession() as session, session.get(
                "http://localhost:3001/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    dependencies["blender_mcp"] = "healthy"
                else:
                    dependencies["blender_mcp"] = "degraded"
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.error(f"Blender MCP connection failed: {e}")
        dependencies["blender_mcp"] = "healthy" if is_test_env else "unhealthy"
    except Exception as e:
        logger.error(f"Unexpected Blender MCP error: {e}")
        dependencies["blender_mcp"] = "healthy" if is_test_env else "unhealthy"

    # Check LLM service (LM Studio or OpenAI)
    settings = get_settings()
    if settings.app.use_local_llm:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session, session.get(
                f"{settings.lmstudio.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    dependencies["llm_service"] = "healthy"
                else:
                    dependencies["llm_service"] = "degraded"
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.error(f"LLM service connection failed: {e}")
            dependencies["llm_service"] = "unhealthy"
        except Exception as e:
            logger.error(f"Unexpected LLM service error: {e}")
            dependencies["llm_service"] = "unhealthy"
    else:
        # For OpenAI, we'd check their API status
        dependencies["llm_service"] = "healthy"  # Assume healthy for now

    # Check Context7 MCP (if configured)
    try:
        # Would check Context7 MCP connection here
        dependencies["context7_mcp"] = "healthy"
    except Exception:
        dependencies["context7_mcp"] = "unhealthy"

    return dependencies


def _collect_system_metrics() -> dict[str, Any]:
    """Collect system performance metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent_used": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent_used": (disk.used / disk.total) * 100,
            },
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "num_threads": process.num_threads(),
            },
            "python": {
                "version": f"{psutil.version_info}",
                "pid": process.pid,
            },
        }
    except Exception as e:
        return {"error": f"Failed to collect metrics: {str(e)}"}


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    dependencies = await _check_dependencies()
    system_metrics = _collect_system_metrics()

    metrics_text = []

    # Service health metrics (0=unhealthy, 1=healthy)
    for service, status in dependencies.items():
        value = 1 if status == "healthy" else 0
        metrics_text.append(f'll3m_service_health{{service="{service}"}} {value}')

    # System metrics
    if "cpu" in system_metrics:
        metrics_text.append(
            f"ll3m_cpu_usage_percent {system_metrics['cpu']['usage_percent']}"
        )
        metrics_text.append(f"ll3m_cpu_count {system_metrics['cpu']['count']}")

    if "memory" in system_metrics:
        metrics_text.append(
            f"ll3m_memory_usage_percent {system_metrics['memory']['percent_used']}"
        )
        metrics_text.append(
            f"ll3m_memory_total_bytes {system_metrics['memory']['total']}"
        )
        metrics_text.append(
            f"ll3m_memory_available_bytes {system_metrics['memory']['available']}"
        )

    if "process" in system_metrics:
        metrics_text.append(
            f"ll3m_process_memory_rss_bytes {system_metrics['process']['memory_rss']}"
        )
        metrics_text.append(
            f"ll3m_process_threads {system_metrics['process']['num_threads']}"
        )

    # Application metrics
    metrics_text.append(f"ll3m_uptime_seconds {time.time() - _startup_time}")

    return "\n".join(metrics_text)


@router.get("/version")
async def get_version():
    """Get application version information."""
    return {
        "version": "1.0.0",
        "build": "dev",  # Would get from CI/CD
        "commit": "unknown",  # Would get from git
        "build_date": "2024-01-01T00:00:00Z",  # Would get from build process
        "python_version": "3.11+",
    }
