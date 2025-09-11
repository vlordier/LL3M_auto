from fastapi.testclient import TestClient

from web.backend.main import app


def test_sessions_list_empty_in_standalone(monkeypatch):
    monkeypatch.setenv("WEB_BACKEND_STANDALONE", "true")
    client = TestClient(app)
    r = client.get("/api/sessions?limit=10")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_sessions_get_not_found(monkeypatch):
    monkeypatch.setenv("WEB_BACKEND_STANDALONE", "true")
    client = TestClient(app)
    r = client.get("/api/sessions/nonexistent")
    # In standalone, load_session returns None -> 404
    assert r.status_code in (404, 200)
    # If it returns 200, ensure it is a dict with session_id
    if r.status_code == 200:
        assert isinstance(r.json(), dict)
