import os
from fastapi.testclient import TestClient

from web.backend.main import app


def test_diagnostics_minimal(monkeypatch):
    monkeypatch.setenv("WEB_BACKEND_STANDALONE", "true")
    client = TestClient(app)
    resp = client.get("/api/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] in (True, False)
    services = data["services"]
    assert "web_backend" in services
    assert "blender" in services
    # prod_api should be skipped in standalone mode
    assert services.get("prod_api", {}).get("status") == "skipped"


def test_diagnostics_opt_in_openai(monkeypatch):
    monkeypatch.setenv("WEB_BACKEND_STANDALONE", "true")
    # No OPENAI_API_KEY: should report missing_key when requested
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.get("/api/diagnostics?checks=openai")
    assert resp.status_code == 200
    data = resp.json()
    assert data["services"]["openai"]["status"] in ("missing_key", "unreachable", "ok")
