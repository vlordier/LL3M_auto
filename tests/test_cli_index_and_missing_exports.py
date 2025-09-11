from pathlib import Path
import json
from click.testing import CliRunner

from src.cli import cli


def test_index_rebuild_and_missing_exports(tmp_path: Path):
    base = tmp_path / "outputs"
    proj = base / "projects" / "Demo"
    # Asset A: no exports
    aid_a = "asset_A"
    adir_a = proj / "assets" / "sess" / "raw" / aid_a
    (adir_a / "exports").mkdir(parents=True)
    (adir_a / "asset_metadata.json").write_text(json.dumps({"asset_id": aid_a}))
    # Asset B: with exports
    aid_b = "asset_B"
    adir_b = proj / "assets" / "sess" / "raw" / aid_b
    (adir_b / "exports").mkdir(parents=True)
    (adir_b / "exports" / "file.gltf").write_text("{}")
    (adir_b / "asset_metadata.json").write_text(json.dumps({"asset_id": aid_b}))

    runner = CliRunner()
    with runner.isolated_filesystem(tmp_dir=str(tmp_path)):
        # Rebuild index
        res = runner.invoke(cli, ["index", "--rebuild"])
        assert res.exit_code == 0, res.output
        # List only missing exports
        res2 = runner.invoke(cli, ["list-assets", "--format", "json", "--missing-exports"])
        assert res2.exit_code == 0, res2.output
        data = json.loads(res2.output)
        ids = [a["asset_id"] for a in data["assets"]]
        assert aid_a in ids
        assert aid_b not in ids
