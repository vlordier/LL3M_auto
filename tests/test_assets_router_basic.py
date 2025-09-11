from pathlib import Path
from fastapi.testclient import TestClient

from web.backend.main import app


def test_list_assets_empty(monkeypatch, tmp_path: Path):
    # Ensure outputs/imported_assets is clean
    client = TestClient(app)
    r = client.get("/api/assets")
    assert r.status_code == 200
    assert r.json() == []


def test_import_path_not_found(monkeypatch, tmp_path: Path):
    client = TestClient(app)
    payload = {
        "file_path": str(tmp_path / "missing.gltf"),
        "asset_name": "x",
        "asset_type": "gltf",
    }
    r = client.post("/api/assets/import-path", json=payload)
    assert r.status_code == 404


def test_delete_asset_not_found():
    client = TestClient(app)
    r = client.delete("/api/assets/does-not-exist")
    assert r.status_code == 404
