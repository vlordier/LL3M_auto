from pathlib import Path
import json

from src.output.asset_index import build_index, load_index, get_asset_dir_by_id


def test_asset_index_build_and_lookup(tmp_path: Path):
    base = tmp_path / "outputs"
    projects = base / "projects" / "Demo_Project_123"
    asset_id = "robot_L3_123456"

    asset_dir = projects / "assets" / "session_abc" / "raw" / asset_id
    asset_dir.mkdir(parents=True)
    meta = asset_dir / "asset_metadata.json"
    meta.write_text(json.dumps({"asset_id": asset_id}))

    # Build index and verify mapping
    mapping = build_index(base)
    assert asset_id in mapping
    assert Path(mapping[asset_id]).resolve() == asset_dir.resolve()

    # Load index and lookup
    loaded = load_index(base)
    assert loaded == mapping

    resolved = get_asset_dir_by_id(asset_id, base)
    assert resolved is not None
    assert resolved.resolve() == asset_dir.resolve()
