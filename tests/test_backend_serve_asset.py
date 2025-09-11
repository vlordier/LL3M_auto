from pathlib import Path
from fastapi.testclient import TestClient

from web.backend.main import app


def test_serve_asset_image_png(tmp_path: Path, monkeypatch):
    # Backend serves files from repo_root/outputs; ensure we create there
    backend_outputs = Path(__file__).resolve().parents[1] / "outputs"
    backend_outputs.mkdir(exist_ok=True)
    try:
        img = backend_outputs / "test.png"
        # Minimal valid PNG header + IHDR chunk
        img.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000A4944415408D763F8FFFF3F0005FE02FEA7A23D9D0000000049454E44AE426082"
            )
        )
        client = TestClient(app)
        resp = client.get("/api/assets/test.png")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("image/png")
    finally:
        # Cleanup file to not pollute working tree
        try:
            img.unlink()  # type: ignore[name-defined]
        except Exception:
            pass
