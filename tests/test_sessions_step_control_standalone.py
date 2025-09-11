from fastapi.testclient import TestClient

from web.backend.main import app


def test_step_control_endpoints_standalone(monkeypatch):
    monkeypatch.setenv("WEB_BACKEND_STANDALONE", "true")
    client = TestClient(app)
    # GET should 404 for unknown session
    r = client.get("/api/sessions/unknown/step-control")
    assert r.status_code == 404
    # POST should 400 for invalid state or unknown session
    r2 = client.post("/api/sessions/unknown/step-control", json={"action": "pause"})
    assert r2.status_code in (400, 404)
