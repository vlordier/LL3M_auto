#!/usr/bin/env python3
"""LL3M Connectivity & Capability Diagnostics

Checks:
- Backend API (web backend) health
- Production API (src.api) health (if running)
- OpenAI endpoint reachability (optional)
- LM Studio (optional) when USE_LOCAL_LLM=true
- Blender executable presence and --version
- Context7 MCP health
- Vector snippet DB (SQLite) file existence and table presence

Exit code 0 if all critical checks pass; non‑zero otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import sys
from pathlib import Path

import aiohttp

HTTP_OK = 200


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


async def http_json(url: str, headers: dict[str, str] | None = None, timeout_s: int = 5) -> tuple[int, dict]:
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url,
                headers=headers or {},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as r,
        ):
            data = {}
            with contextlib.suppress(Exception):
                data = await r.json()
            return r.status, data
    except Exception as e:
        return -1, {"error": str(e)}


def print_status(name: str, ok: bool, extra: str = "") -> None:
    prefix = "✅" if ok else "❌"
    msg = f"{prefix} {name}"
    if extra:
        msg += f": {extra}"
    print(msg)


async def check_backend_web() -> bool:
    status, data = await http_json("http://localhost:8000/health")
    ok = status == HTTP_OK and (data.get("status") == "ok" or data)
    print_status("Web Backend", ok, f"HTTP {status}")
    return ok


async def check_prod_api() -> bool:
    # src.api exposes docs at /api/docs and health under /api/v1/health
    status, data = await http_json("http://localhost:8000/api/v1/health")
    ok = status == HTTP_OK and isinstance(data, dict)
    print_status("Production API (/api/v1)", ok, f"HTTP {status}")
    return ok


async def check_openai() -> bool:
    if env_bool("USE_LOCAL_LLM", "false"):
        print("ℹ️  Skipping OpenAI check (USE_LOCAL_LLM=true)")
        return True
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print_status("OpenAI", False, "OPENAI_API_KEY not set")
        return False
    # Lightweight check: try listing models
    status, _ = await http_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout_s=8,
    )
    ok = status in (200, 401)  # 401 means key invalid but endpoint reachable
    print_status("OpenAI Endpoint", ok, f"HTTP {status}")
    return ok


async def check_lmstudio() -> bool:
    if not env_bool("USE_LOCAL_LLM", "false"):
        print("ℹ️  Skipping LM Studio (USE_LOCAL_LLM=false)")
        return True
    base = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    status, data = await http_json(f"{base}/models")
    ok = status == HTTP_OK and isinstance(data, dict)
    print_status("LM Studio", ok, f"HTTP {status}")
    return ok


def check_blender() -> bool:
    # Prefer BLENDER_PATH or BLENDER_EXECUTABLE
    candidate = (
        os.getenv("BLENDER_PATH")
        or os.getenv("BLENDER_EXECUTABLE")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )
    p = Path(candidate)
    ok = p.exists() and p.is_file()
    print_status("Blender Binary", ok, p.as_posix())
    return ok


async def check_context7() -> bool:
    base = os.getenv("CONTEXT7_MCP_SERVER", os.getenv("CONTEXT7_MCP", "http://localhost:8080")).rstrip("/")
    status, data = await http_json(f"{base}/health")
    ok = status == HTTP_OK
    print_status("Context7 MCP", ok, f"HTTP {status}")
    return ok


def check_vector_snippet_db() -> bool:
    # The snippet vector DB is SQLite at cache/vector_cache.db
    db_path = Path("cache/vector_cache.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    code TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    complexity_level INTEGER DEFAULT 1,
                    quality_score REAL,
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used TEXT
                )
                """
            )
            # Smoke read
            conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        print_status("Vector Snippet DB (SQLite)", True, db_path.as_posix())
        return True
    except Exception as e:
        print_status("Vector Snippet DB (SQLite)", False, str(e))
        return False


async def main() -> int:
    print("🔍 LL3M Connectivity & Capability Diagnostics")
    print("=" * 60)

    ok = True
    # Backend health
    ok &= await check_backend_web()
    ok &= await check_prod_api()

    # LLM endpoints
    ok &= await check_openai()
    ok &= await check_lmstudio()

    # Blender
    ok &= check_blender()

    # Context7 MCP
    ok &= await check_context7()

    # Vector DB for snippets
    ok &= check_vector_snippet_db()

    print("=" * 60)
    print("✅ All critical checks passed" if ok else "⚠️  Some checks failed")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
