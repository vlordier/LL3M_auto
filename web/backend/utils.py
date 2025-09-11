from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import aiohttp


async def http_json(url: str, headers: dict[str, str] | None = None, timeout_s: int = 5) -> tuple[int, dict[str, Any]]:
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r,
        ):
            try:
                data = await r.json()
            except Exception:
                data = {}
            return r.status, data
    except Exception as e:  # pragma: no cover
        return -1, {"error": str(e)}


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def candidate_blender_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("BLENDER_PATH") or os.getenv("BLENDER_EXECUTABLE")
    if env_path:
        candidates.append(Path(env_path))

    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))

    system = platform.system().lower()
    if system == "darwin":
        candidates += [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        ]
        hb = Path("/opt/homebrew/Cellar/blender")
        if hb.exists():
            for p in hb.glob("*/Blender.app/Contents/MacOS/Blender"):
                candidates.append(p)
    else:
        for p in ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"]:
            candidates.append(Path(p))
        fp = Path("/var/lib/flatpak/app/org.blender.Blender")
        if fp.exists():
            for p in fp.glob("*/active/files/bin/blender"):
                candidates.append(p)

    seen = set()
    uniq: list[Path] = []
    for c in candidates:
        try:
            key = c.resolve()
        except Exception:
            key = c
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def run_blender_cmd(bin_path: Path, args: list[str], timeout_s: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [bin_path.as_posix(), *args], capture_output=True, text=True, timeout=timeout_s
        )
        out = proc.stdout or proc.stderr
        return proc.returncode, out
    except Exception as e:  # pragma: no cover
        return -1, str(e)

