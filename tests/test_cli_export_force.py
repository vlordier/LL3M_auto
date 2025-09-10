from pathlib import Path
import re
from types import SimpleNamespace

from click.testing import CliRunner

from src.cli import cli


def test_cli_export_force_from_blend(monkeypatch, tmp_path: Path):
    # Create a fake asset directory with a .blend under outputs/
    asset_dir = tmp_path / "asset" / "raw" / "robot_L3_000000"
    outputs_dir = asset_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    blend_file = outputs_dir / "scene.blend"
    blend_file.write_bytes(b"BLENDER\x00")

    out_file = tmp_path / "exported.gltf"

    # Patch BlenderServiceImpl.execute_script used inside _force_export_from_blend
    async def fake_execute_script(self, code: str, context=None, timeout: float = 180.0):  # noqa: D401
        # Extract filepath from code
        m = re.search(r"filepath=r\"([^\"]+)\"", code)
        if m:
            export_path = Path(m.group(1))
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text("{}")
        return SimpleNamespace(success=True, errors=[])

    import src.cli as cli_mod

    monkeypatch.setattr(cli_mod.BlenderServiceImpl, "execute_script", fake_execute_script, raising=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export",
            str(asset_dir),
            "--format",
            "gltf",
            "--output",
            str(out_file),
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_file.exists(), "Expected exported file to exist"
