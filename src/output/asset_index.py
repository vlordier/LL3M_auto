"""Asset index for fast lookups by asset_id across output structures.

Builds and reads an index at outputs/asset_index.json that maps asset_id -> asset_dir.
Supports both canonical "projects/.../assets/.../<stage>/<asset_id>" and legacy
"sessions/<session_id>/..." layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


INDEX_FILENAME = "asset_index.json"


def _find_asset_dirs(base_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    # Canonical: outputs/projects/*/assets/*/*/<asset_id>
    projects = base_dir / "projects"
    if projects.exists():
        for proj in projects.iterdir():
            assets_root = proj / "assets"
            if not assets_root.exists():
                continue
            for session_dir in assets_root.iterdir():
                if not session_dir.is_dir():
                    continue
                for stage_dir in session_dir.iterdir():
                    if not stage_dir.is_dir():
                        continue
                    for asset_dir in stage_dir.iterdir():
                        if not asset_dir.is_dir():
                            continue
                        meta = asset_dir / "asset_metadata.json"
                        if meta.exists():
                            try:
                                data = json.loads(meta.read_text())
                                asset_id = data.get("asset_id")
                                if asset_id:
                                    mapping[asset_id] = str(asset_dir.resolve())
                            except Exception:
                                pass

    # No legacy scanning: streamlined to canonical project structure only

    return mapping


def build_index(base_output_dir: Path) -> Dict[str, str]:
    mapping = _find_asset_dirs(base_output_dir)
    index_file = base_output_dir / INDEX_FILENAME
    try:
        index_file.write_text(json.dumps(mapping, indent=2))
    except Exception:
        # non-fatal
        pass
    return mapping


def load_index(base_output_dir: Path) -> Dict[str, str]:
    index_file = base_output_dir / INDEX_FILENAME
    try:
        return json.loads(index_file.read_text())
    except Exception:
        return {}


def get_asset_dir_by_id(asset_id: str, base_output_dir: Path) -> Optional[Path]:
    mapping = load_index(base_output_dir)
    path_str = mapping.get(asset_id)
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p
    # Rebuild if not found
    mapping = build_index(base_output_dir)
    path_str = mapping.get(asset_id)
    return Path(path_str) if path_str else None
